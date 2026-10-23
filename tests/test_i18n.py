import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from humanoid_sim import i18n, journal
from humanoid_sim.goal_check import items
from humanoid_sim.reports import build_report, render_report
from humanoid_sim.webapp import DashboardServer

CYRILLIC = re.compile('[А-Яа-яЁё]')


def test_catalogs_cover_every_interface_string():
    """A new Russian string in a page, a script or a module the browser reads needs its English and Chinese in
    humanoid_sim/web/i18n/en.json and zh.json (the Russian text is the key; {name} / {n} are placeholders)."""
    for lang in ('en', 'zh'):
        web, server = i18n.missing(lang)
        assert web == [], f'{lang}.json lacks page strings: {web}'
        assert server == [], f'{lang}.json lacks server strings: {server}'
    assert i18n.check_catalogs() == []


def test_page_scan_skips_pages_not_written_yet(monkeypatch):
    """The site's pages are written in parallel: a listed page or script that is not there yet is skipped."""
    monkeypatch.setattr(i18n, 'WEB_PAGES', i18n.WEB_PAGES + ('not-yet.html',))
    monkeypatch.setattr(i18n, 'WEB_SCRIPTS', i18n.WEB_SCRIPTS + ('not-yet.js',))
    found = i18n.web_messages()
    assert 'Дерево веток' in found and 'Ветка и этап дорожки {n}' in found  # lab.html, race.js


def test_messages_stay_russian_unless_a_language_is_asked():
    message = 'Выберите от 1 до 10 дорожек'
    assert i18n.translate(message, 'ru') == message
    assert i18n.language({}) == 'ru' and i18n.language({'lang': ['xx']}) == 'ru'
    assert i18n.language({'lang': ['zh']}) == 'zh'
    assert i18n.translate(message, 'en') == 'Choose 1 to 10 lanes'
    assert i18n.translate('Нечто новое', 'en') == 'Нечто новое'  # no translation: the Russian original


def test_templates_translate_values_and_numbers():
    assert i18n.translate('envs: допустимо 1e-06–0.01', 'en') == 'envs: allowed 1e-06–0.01'
    assert i18n.translate('Эта проверка недоступна для задачи «Следование скорости»', 'en') == \
        'This check is not available for the task “Velocity tracking”'
    assert i18n.translate('+0,31 / -0,29 рад/с', 'zh') == '+0.31 / -0.29 rad/s'
    assert i18n.translate('подъём 32 мм, наклон 12°', 'en') == 'lift 32 mm, tilt 12°'
    assert i18n.numbers('12%, σ 1,23; 2 000 000 шагов', 'en') == '12%, σ 1.23; 2,000,000 шагов'
    _ = i18n.gettext('zh')
    assert _('Seed: {n1}; попыток: {n2}; лимит: {n3} с.', n1=1, n2=2, n3=3) == 'Seed：1；尝试次数：2；时限：3 s。'
    assert i18n.gettext('ru')('Осталось {n} с', n=5) == 'Осталось 5 с'


def test_goal_check_rows_translate_completely():
    scenario = lambda forward, lean=3.: dict(falls='0/5', forward_m=forward, sideways_m=0., heading_deg=1.,
                                             torso_lean_deg=lean, clipped=.2)
    walk = dict(turn={(1., .5): .34, (1., -.5): -.38}, drift={(1., 'left'): [.02]}, pelvis=.77,
                feet=dict(lift_m=.06, stance_tilt_deg=4., by_foot={}))
    rows = items({'stand': scenario(-.05), 'walk': scenario(9.), 'stop_start': scenario(4.)}, .79, walk)
    check = {'at': 1, 'branch': 'exp-1', 'stage': 2, 'title': 'Ветка · 5481,4 млн', 'items': rows}
    for lang in ('en', 'zh'):
        shown = i18n.localize_journal({'check': check}, lang, i18n.Content('nowhere'))['check']
        for row in shown['items']:
            for field in ('label', 'value', 'target'):
                assert not CYRILLIC.search(row[field]), (lang, row)
                assert not re.search(r'\d,\d', row[field]), (lang, row)  # decimal point in English and Chinese
        assert shown['title'] == 'Ветка · 5481.4 M'  # the branch name is journal text: Russian until translated


def test_content_translates_paragraph_by_paragraph(tmp_path):
    content = i18n.Content(tmp_path)
    text = 'Первый абзац.\n\nВторой абзац, 0,5 м/с.'
    assert content.text(text, 'en') == text
    folder = tmp_path / 'journal' / 'i18n'
    folder.mkdir(parents=True)
    unit = {'units': {i18n.unit_id('Первый абзац.'): {'ru': 'Первый абзац.', 'text': 'First paragraph.'}}}
    (folder / 'en.json').write_text(json.dumps(unit), encoding='utf-8')
    assert content.text(text, 'en') == 'First paragraph.\n\nВторой абзац, 0,5 м/с.'
    assert content.text(text, 'zh') == text and content.text(text, 'ru') == text


def test_report_renders_in_every_language(tmp_path):
    (tmp_path / 'stage_000').mkdir()
    (tmp_path / 'stage_000' / 'metadata.json').write_text(json.dumps({'steps': 0, 'created_at': 1000}))
    (tmp_path / 'stage_000' / 'evaluation.json').write_text(json.dumps(
        {'mean_return': 1, 'mean_seconds': 2, 'mean_forward_m': 1, 'episodes': []}))
    report = build_report(tmp_path)
    assert render_report(report).startswith('<!doctype html><html lang="ru">')
    assert 'Результаты по ходу обучения' in render_report(report)
    for lang, tag in (('en', 'en'), ('zh', 'zh-Hans')):
        page = render_report({**report, 'title': 'Title'}, lang)
        assert f'<html lang="{tag}">' in page
        assert not CYRILLIC.search(page), CYRILLIC.findall(page)[:20]


def test_stream_frames_are_serialized_once_per_language():
    from humanoid_sim.webapp import _Looping
    player = _Looping()
    player._init_loop()
    player._frame = {'seq': 1, 'reward_labels': {'fall': 'Штраф за падение'}, 'lanes': [], 'error': None}
    content = i18n.Content('nowhere')
    calls = []

    def localize(frame, lang='en'):
        calls.append(lang)
        return i18n.localize_frame(frame, lang, content)
    assert json.loads(player.frame_json()[1])['reward_labels']['fall'] == 'Штраф за падение'
    assert json.loads(player.frame_json('en', localize)[1])['reward_labels']['fall'] == 'Fall penalty'
    player.frame_json('en', localize)
    assert calls == ['en']  # same frame: the English JSON is reused
    player._frame = {**player._frame, 'seq': 2}
    assert player.frame_json('en', localize)[0] == 2 and calls == ['en', 'en']


def test_http_answers_in_the_asked_language(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path / 'original')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'

    def get(path):
        with urllib.request.urlopen(url + path) as response:
            return json.load(response)

    def error(path, body=None):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Content-Type': 'application/json'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        return json.load(exc.value)['error']

    try:
        experiment = server.lab.create({'name': 'Осторожный PPO'})
        journal.add_entry(server.lab.root, 'first', 'Пошёл', 'Робот пошёл.\n\nНа 1,0 м/с.', at=1)
        folder = server.lab.root / 'journal' / 'i18n'
        folder.mkdir(parents=True)
        (folder / 'en.json').write_text(json.dumps({'units': {
            i18n.unit_id('Пошёл'): {'text': 'Walking'}, i18n.unit_id('Робот пошёл.'): {'text': 'The robot walks.'},
            i18n.unit_id('На 1,0 м/с.'): {'text': 'At 1.0 m/s.'}}}), encoding='utf-8')
        assert get('/api/lab')['experiments'][0]['name'] == 'Эксперимент 1 · исходный PPO'
        names = {e['id']: e['name'] for e in get('/api/lab?lang=en')['experiments']}
        assert names == {'baseline': 'Experiment 1 · original PPO', experiment['id']: 'Careful PPO'}
        entry = get('/api/lab/journal?lang=en')['entries'][0]
        assert (entry['title'], entry['text']) == ('Walking', 'The robot walks.\n\nAt 1.0 m/s.')
        assert get('/api/lab/journal?lang=zh')['entries'][0]['title'] == 'Пошёл'  # no Chinese yet: Russian
        robots = get('/api/lab/robots?lang=zh')
        labels = [r['label'] for r in robots] + [t['label'] for r in robots for t in r['tasks']] + \
            [p['label'] for r in robots for t in r['tasks'] for p in t['parameters']]
        assert labels and not any(CYRILLIC.search(label) for label in labels)
        assert error('/api/lab/enqueue?lang=en', {'ids': []}) == 'Choose 1 to 20 experiments'
        assert error('/api/lab/enqueue', {'ids': []}) == 'Выберите от 1 до 20 экспериментов'
        assert error('/api/missing?lang=zh') == '未找到'
        with urllib.request.urlopen(url + '/i18n/zh.json') as response:
            assert json.load(response)['Смотреть'] == '观看'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
