import hashlib
import json
import re
import threading
import urllib.request
import zipfile

import pytest

from humanoid_sim import i18n, journal, site_export
from humanoid_sim.core import write_json
from humanoid_sim.lab import Laboratory
from humanoid_sim.webapp import DashboardServer

WEB = i18n.WEB


def lab_with_journal(tmp_path):
    """A lab whose journal has one entry with a picture and a clip, an unused file, and the home page's clips."""
    lab = Laboratory(tmp_path / 'original')
    (tmp_path / 'race.webp').write_bytes(b'RIFF clip')
    (tmp_path / 'race-poster.jpg').write_bytes(b'poster')
    (tmp_path / 'picture.png').write_bytes(b'png')
    journal.add_entry(lab.root, 'first', 'Пошёл', 'Робот пошёл.', image=tmp_path / 'picture.png',
                      clip=tmp_path / 'race.webp', at=1)
    journal.add_media(lab.root, tmp_path / 'picture.png', 'unused.png')
    for page in site_export.site_pages():
        for clip in site_export.CLIP.findall((WEB / page).read_text(encoding='utf-8')):
            for lang in i18n.LANGUAGES:
                (journal.folder(lab.root) / f'site-{lang}-{clip}.webp').write_bytes(b'clip')
                (journal.folder(lab.root) / f'site-{lang}-{clip}-poster.jpg').write_bytes(b'poster')
        for name in site_export.PAGE_MEDIA.findall((WEB / page).read_text(encoding='utf-8')):
            (journal.folder(lab.root) / name).write_bytes(b'media')  # world.html's clips and posters
    return lab


def test_a_story_section_clip_goes_out_with_the_site(tmp_path):
    lab = lab_with_journal(tmp_path)
    journal.set_roadmap(lab.root, [{'title': 'Руки', 'status': 'active'}])
    clip = journal.add_media(lab.root, tmp_path / 'race.webp', 'stage3-sag.webp')
    poster = journal.add_media(lab.root, tmp_path / 'race-poster.jpg', 'stage3-sag.jpg')
    journal.set_story(lab.root, 0, {'sections': [{'title': 'Провис', 'text': 'Плечо висит.', 'clip': clip,
                                                  'poster': poster}]})
    out = tmp_path / 'site'
    summary = site_export.export(out, lab.original)
    for lang in i18n.LANGUAGES:
        data = json.loads((out / 'data' / f'journal.{lang}.json').read_text(encoding='utf-8'))
        section = data['roadmap'][0]['story']['sections'][0]
        assert (section['clip'], section['poster']) == ('stage3-sag.webp', 'stage3-sag.jpg')
    assert {'stage3-sag.webp', 'stage3-sag.jpg'} <= {path.name for path in (out / 'media').iterdir()}
    assert [name for name in summary['missing'] if name.startswith('media/')] == []


def test_export_writes_the_static_site(tmp_path):
    lab = lab_with_journal(tmp_path)
    experiment = lab.create({'name': 'Осторожный PPO'})
    write_json(lab.resolve(experiment['id']) / 'status.json', {'state': 'stopped', 'pid': 123, 'steps': 5})
    write_json(lab.root / 'queue.json', {'items': [{'id': experiment['id'], 'state': 'pending'}], 'state': 'idle'})
    out = tmp_path / 'site'
    summary = site_export.export(out, lab.original, tmp_path / 'site.zip')

    index = (out / 'index.html').read_text(encoding='utf-8')
    assert '<meta name="site-mode" content="static" />' in index and 'content="live"' not in index
    assert 'href="site.html' not in index and not (out / 'site.html').exists()
    for name in ('site.css', 'site.js', 'i18n.js', 'home.js', 'home.css', 'i18n/en.json', 'i18n/zh.json'):
        assert (out / name).is_file(), name
    assert not (out / 'vendor').exists()  # no page of the site needs three.js
    for page in ('journal.html', 'lab.html'):  # exported once the new pages carry the site-mode meta
        source = WEB / page
        assert (out / page).is_file() == (source.is_file() and 'name="site-mode"' in source.read_text(encoding='utf-8'))

    for lang in i18n.LANGUAGES:
        data = json.loads((out / 'data' / f'journal.{lang}.json').read_text(encoding='utf-8'))
        assert [entry['id'] for entry in data['entries']] == ['first']
        tree = json.loads((out / 'data' / f'lab.{lang}.json').read_text(encoding='utf-8'))
        assert not set(site_export.CONTROL) & set(tree)
        assert {entry['id'] for entry in tree['experiments']} == {'baseline', experiment['id']}
        assert all('pid' not in entry['status'] for entry in tree['experiments'])
    english = json.loads((out / 'data' / 'lab.en.json').read_text(encoding='utf-8'))
    assert english['experiments'][0]['name'] == 'Experiment 1 · original PPO'
    tasks = [task for robot in english['robots'] for task in robot['tasks']]
    assert tasks and all('race' in task and not set(site_export.TASK_CONTROL) & set(task) for task in tasks)

    media = {path.name for path in (out / 'media').iterdir()}
    assert {'first.png', 'first-clip.webp', 'first-clip.jpg'} <= media
    assert {'site-ru-walk.webp', 'site-en-walk-poster.jpg', 'site-zh-story-path.webp'} <= media
    assert 'unused.png' not in media
    assert [name for name in summary['missing'] if name.startswith('media/')] == []
    shown = {name for page in site_export.site_pages()
             for name in site_export.PAGE_MEDIA.findall((WEB / page).read_text(encoding='utf-8'))}
    assert shown <= media  # world.html's clips and posters (not animated here, so copied as they are)
    assert summary['site_media_bytes'] > 0
    assert summary['journal_media_bytes'] == len(b'RIFF clip' b'poster' b'png') + len(b'media') * len(shown)
    assert summary['video']['mp4'] == 0 and summary['video']['failed'] == {}

    with zipfile.ZipFile(tmp_path / 'site.zip') as archive:
        names = set(archive.namelist())
    assert {'index.html', 'home.js', 'data/journal.zh.json', 'media/first-clip.webp'} <= names
    assert site_export.MARKER not in names and len(names) == summary['files']


def test_export_replaces_only_its_own_folder(tmp_path):
    lab = lab_with_journal(tmp_path)
    out = tmp_path / 'site'
    site_export.export(out, lab.original)
    (out / 'stale.txt').write_text('old')
    site_export.export(out, lab.original)  # a previous export is written anew
    assert not (out / 'stale.txt').exists() and (out / 'index.html').is_file()
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'keep.txt').write_text('mine')
    with pytest.raises(ValueError, match='не пуста'):
        site_export.export(foreign, lab.original)
    assert (foreign / 'keep.txt').read_text() == 'mine'
    with pytest.raises(ValueError, match='runs'):
        site_export.export(site_export.ROOT / 'runs' / 'site', lab.original)


# sha256 of the contact address, so the address itself is nowhere in the sources, this file included.
CONTACT_SHA256 = 'c0ee09a432a9a346974a9ef01337c7277780880162e71fadae178f240caf2a87'


def test_email_is_assembled_only_on_click():
    script = (WEB / 'home.js').read_text(encoding='utf-8')
    parts = re.search(r"\['([^']+)', '([^']+)'\]\s*\.map\(\(part\) => \[\.\.\.part\]\.reverse\(\)", script)
    assert parts
    address = '@'.join(part[::-1] for part in parts.groups())
    assert hashlib.sha256(address.encode()).hexdigest() == CONTACT_SHA256
    user, domain = address.split('@')
    for name in ('site.html', 'home.js'):
        source = (WEB / name).read_text(encoding='utf-8')
        assert address not in source and user not in source and domain not in source, name
    assert 'https://t.me/mawgly' in (WEB / 'site.html').read_text(encoding='utf-8')


def test_home_page_strings_have_english(monkeypatch):
    """Every Russian string of site.html / home.js has the mockup's English; inline marks survive translation."""
    monkeypatch.setattr(i18n, 'WEB_PAGES', ('site.html',))
    monkeypatch.setattr(i18n, 'WEB_SCRIPTS', ('home.js',))
    english = i18n.catalog('en').exact
    keys = i18n.web_messages()
    assert keys and sorted(key for key in keys if key not in english) == []
    for key in keys:
        assert (key.count('**'), key.count('`')) == (english[key].count('**'), english[key].count('`')), key


def test_root_serves_the_home_page(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path / 'original')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url + '/') as response:
            assert response.read() == (WEB / 'site.html').read_bytes()
        with urllib.request.urlopen(url + '/home.js') as response:
            assert response.headers['Content-Type'].startswith('text/javascript')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_scripts_and_styles_carry_the_build(tmp_path):
    """The host keeps .js / .css a week in the browser: the export names them with ?v=<build> in the pages and in
    every module's imports, one build for the whole export."""
    from humanoid_sim import site_export
    html = '<link href="site.css" /><script type="module" src="home.js"></script><a href="journal.html">'
    assert site_export.versioned(html, '.html', 'abc') == (
        '<link href="site.css?v=abc" /><script type="module" src="home.js?v=abc"></script><a href="journal.html">')
    js = "import { t } from './i18n.js';\nconst x = import('./race.js');\nfetch('data/home.json');"
    assert site_export.versioned(js, '.js', 'abc') == (
        "import { t } from './i18n.js?v=abc';\nconst x = import('./race.js?v=abc');\nfetch('data/home.json');")
    assert site_export.versioned('body { color: red }', '.css', 'abc') == 'body { color: red }'
    assert site_export.build_id(['site.html', 'site.js']) == site_export.build_id(['site.js', 'site.html'])
