"""Static copy of the lab's site (Home / Journal / Lab) for public hosting.

    .venv\\Scripts\\python.exe -m humanoid_sim.site_export --out <folder> [--zip <file>] [--run-dir runs/humanoid]

The pages are the live ones (humanoid_sim/web: site.html as index.html, journal.html and lab.html once they carry
<meta name="site-mode">) switched to <meta name="site-mode" content="static">, so web/site.js reads
data/<name>.<lang>.json and media/<file> instead of the server's API. Copied beside them: the scripts and styles the
pages load (followed through their imports, vendor/ only if a page needs it) and the i18n/ catalogs.

data/ holds, for ru, en and zh, what the server answers the browser: the journal (journal.load +
i18n.localize_journal, as /api/lab/journal) and the branch tree (Laboratory.entries and notes, the benchmarks +
i18n.localize_snapshot, as /api/lab, plus `robots`: /api/lab/robots' names and race flags) without what runs the
lab: queue, worker, jobs, hardware, process ids, the settings forms; and the home page's data (humanoid_sim.home,
written by scripts/site/home_update.py; {"empty": true} before its first run), as /api/lab/home.
media/ holds the journal pictures, clips and figures that data names, those a page names (data-media, data-poster)
and the home page's site-<lang>-<name> clips (data-clip in a page, and the clips the home data names), nothing else
from the journal folder. No live race: the pages show pre-rendered clips.
Every animated WebP clip goes out as an H.264 MP4 instead (humanoid_sim.site_video; iPhone Safari does not play our
animated WebP, 29.10), and the names in data/*.json and the pages' data-media follow it; web/clip.js plays an .mp4 as
a muted inline <video> and a .webp as before. A clip whose conversion fails goes out as its WebP and is reported.
The live server keeps serving the WebP.
"""
import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from . import home, i18n, journal, site_video, stack
from .core import DEFAULT_RUN, ROOT, read_json
from .lab import Laboratory

WEB = i18n.WEB
# Source page -> exported name. A page is part of the site once it has the site-mode meta (the old lab.html has not).
# The host (LiteSpeed at IHC) served the 1.2 MB journal data uncompressed (28.10: slow to open on a phone): text is
# compressed, pictures and clips kept a day in the browser.
HTACCESS = '''<IfModule mod_deflate.c>
AddOutputFilterByType DEFLATE text/html text/css text/plain application/javascript text/javascript application/json image/svg+xml
</IfModule>
<IfModule mod_expires.c>
ExpiresActive On
ExpiresByType image/webp "access plus 1 day"
ExpiresByType image/jpeg "access plus 1 day"
ExpiresByType image/png "access plus 1 day"
ExpiresByType video/mp4 "access plus 1 day"
</IfModule>
AddType video/mp4 .mp4
'''
PAGES = {'site.html': 'index.html', 'journal.html': 'journal.html', 'lab.html': 'lab.html', 'how.html': 'how.html',
         'world.html': 'world.html'}
SITE_MODE = re.compile(r'(<meta\s+name="site-mode"\s+content=")live(")')
# Local files a page, script or style loads; page links (.html) are not followed, the pages are PAGES.
REFERENCES = {
    '.html': re.compile(r'''\b(?:src|href)\s*=\s*["']([^"'#?]+)'''),
    '.js': re.compile(r'''(?:\bfrom\s*|\bimport\s*\(?\s*)['"]([^'"]+)['"]'''),
    '.css': re.compile(r'''@import\s+['"]([^'"]+)['"]|url\(\s*['"]?([^'")#?]+)'''),
}
CLIP = re.compile(r'\bdata-clip="([A-Za-z0-9_-]+)"')
# Journal media a page shows by name (world.html: the clip and its poster).
PAGE_MEDIA = re.compile(r'\bdata-(?:media|poster)="([A-Za-z0-9_.-]+)"')
# The host keeps styles and scripts a week in the browser (Cache-Control max-age 604800, 29.10: a phone mixed the new
# site.js with last week's site.css), so every local .js / .css reference in the pages and in the modules' imports gets
# ?v=<build>, one hash of all the exported code: each module keeps one URL, and a publish with other code renames all.
VERSIONED = {
    '.html': re.compile(r'''(\b(?:src|href)\s*=\s*["'])([^"':#?]+\.(?:js|css))(["'])'''),
    '.js': re.compile(r'''((?:\bfrom\s*|\bimport\s*\(?\s*)['"])([^'":#?]+\.js)(['"])'''),
}


def build_id(sources):
    """A short hash of the exported pages, scripts and styles (WEB-relative names)."""
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode() + b'\0' + (WEB / name).read_bytes())
    return digest.hexdigest()[:10]


def versioned(text, suffix, build):
    """text with ?v=build after every local .js / .css it loads (VERSIONED)."""
    pattern = VERSIONED.get(suffix)
    return text if pattern is None else pattern.sub(lambda m: f'{m[1]}{m[2]}?v={build}{m[3]}', text)
# Parts of the /api/lab answer, and of a task in /api/lab/robots, that run the lab rather than describe it.
CONTROL = ('queue', 'worker_active', 'active_trainers', 'job', 'hardware')
TASK_CONTROL = ('parameters', 'jobs', 'backends', 'training_backends')
STORED = ('.webp', '.jpg', '.png', '.mp4')  # already compressed: zipped as they are
MARKER = 'site-export.json'


def site_pages():
    """{source name: exported name} of the pages that are part of the site."""
    pages = {}
    for source, name in PAGES.items():
        path = WEB / source
        if path.is_file() and SITE_MODE.search(path.read_text(encoding='utf-8')):
            pages[source] = name
    if 'site.html' not in pages:
        raise ValueError('Нет главной страницы сайта: web/site.html без <meta name="site-mode" content="live">')
    return pages


def static_page(text):
    """A page in static mode; links to the home page go to index.html."""
    return re.sub(r'href="site\.html', 'href="index.html', SITE_MODE.sub(r'\1static\2', text, count=1))


def dependencies(sources):
    """Files under web/ that the given pages load, directly or through other scripts and styles (relative paths)."""
    found, todo, missing = set(), [WEB / name for name in sources], set()
    while todo:
        path = todo.pop()
        pattern = REFERENCES.get(path.suffix)
        if pattern is None:
            continue
        for match in pattern.finditer(path.read_text(encoding='utf-8')):
            ref = next(group for group in match.groups() if group)
            if ':' in ref or ref.startswith('//') or '${' in ref or ref.endswith('.html'):
                continue  # another site, a URL built at run time, or a page
            file = (WEB / ref.lstrip('/') if ref.startswith('/') else path.parent / ref).resolve()
            if WEB.resolve() not in file.parents or not file.is_file():
                missing.add(ref)
                continue
            name = file.relative_to(WEB.resolve()).as_posix()
            if name not in found:
                found.add(name)
                todo.append(file)
    return sorted(found), sorted(missing)


def _scrub(value):
    """value without the lab's own folder in any string (the site is public)."""
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        for root in (str(ROOT), ROOT.as_posix()):
            value = value.replace(root, '.')
    return value


def lab_snapshot(lab):
    """/api/lab's content without what runs the lab (see LabJobs.snapshot)."""
    experiments = []
    for entry in lab.entries():
        entry = {key: value for key, value in entry.items() if key != 'request_key'}
        entry['status'] = {key: value for key, value in entry.get('status', {}).items() if key != 'pid'}
        experiments.append(entry)
    return {'experiments': experiments, 'notes': lab.notes(),
            'benchmark': read_json(lab.root / 'gpu_benchmark.json'),
            'full_gpu_benchmark': read_json(lab.root / 'full_gpu_benchmark.json')}


def robot_names(robots, lang):
    """/api/lab/robots in `lang` with the labels and which tasks race, not the settings forms and jobs."""
    return [{**robot, 'tasks': [{key: value for key, value in task.items() if key not in TASK_CONTROL}
                                for task in robot['tasks']]} for robot in i18n.localize_robots(robots, lang)]


def media_names(value):
    """Journal media names in an exported journal: every string that is a picture, clip or figure file name."""
    if isinstance(value, dict):
        return set().union(*map(media_names, value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*map(media_names, value)) if value else set()
    if isinstance(value, str) and (journal.IMAGE_NAME.match(value) or journal.FIGURE_NAME.match(value)):
        return {value}
    return set()


def renamed(value, names):
    """value with every string that is a key of `names` (a media file name) replaced by its value."""
    if isinstance(value, dict):
        return {key: renamed(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [renamed(item, names) for item in value]
    return names.get(value, value) if isinstance(value, str) else value


def page_media(text, names):
    """A page with its data-media="<clip>.webp" pointing at the MP4 the export wrote instead."""
    return re.sub(r'(\bdata-media=")([A-Za-z0-9_.-]+)(")', lambda m: m[1] + names.get(m[2], m[2]) + m[3], text)


def videos(files, cache=None):
    """({webp name: mp4 name} of the animated WebP clips among {name: file} that became MP4s, {name: error} of those
    that did not, the MP4 files by name)."""
    clips = {name: file for name, file in files.items() if name.endswith('.webp') and site_video.animated(file)}
    results = site_video.convert_all(clips.values(), cache)
    converted, failed, made = {}, {}, {}
    for name, file in clips.items():
        result = results[Path(file)]
        if isinstance(result, Path):
            converted[name] = name[:-len('.webp')] + '.mp4'
            made[converted[name]] = result
        else:
            failed[name] = result
    return converted, failed, made


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _prepare(out):
    """An empty output folder: a new one, or a previous export (it has the marker) emptied."""
    out = Path(out).resolve()
    for keep in (ROOT / 'runs', WEB):
        if out == keep.resolve() or keep.resolve() in out.parents:
            raise ValueError(f'Выгрузка сайта не пишется в {keep.relative_to(ROOT)}')
    if out.exists():
        if any(out.iterdir()):
            if not (out / MARKER).is_file():
                raise ValueError(f'Папка {out} не пуста и не похожа на прошлую выгрузку сайта')
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(json.dumps({'complete': False}), encoding='utf-8')
    return out


def export(out, run_dir=DEFAULT_RUN, zip_file=None, video_cache=None):
    """Writes the static site into `out`; returns a summary (files, bytes, media bytes, what was missing, the clips
    written as MP4). video_cache: the MP4 cache folder (default site_video.CACHE)."""
    pages = site_pages()
    out = _prepare(out)
    files, missing = dependencies(pages)
    build = build_id(list(pages) + files)
    for name in files + [f'i18n/{lang}.json' for lang in i18n.LANGUAGES[1:]]:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        if Path(name).suffix == '.js':
            (out / name).write_text(versioned((WEB / name).read_text(encoding='utf-8'), '.js', build),
                                    encoding='utf-8', newline='')
        else:
            shutil.copyfile(WEB / name, out / name)

    from .tasks.catalog import catalog
    lab = Laboratory(run_dir)
    content = i18n.Content(lab.root)
    saved, snapshot, robots = journal.load(lab.root), lab_snapshot(lab), catalog()
    front = home.load(lab.root)
    sources = {source: (WEB / source).read_text(encoding='utf-8') for source in pages}
    names = {name for text in sources.values() for name in PAGE_MEDIA.findall(text) if journal.IMAGE_NAME.match(name)}
    data = {}
    for lang in i18n.LANGUAGES:
        localized = _scrub(i18n.localize_journal(saved, lang, content))
        names |= media_names(localized)
        tree = i18n.localize_snapshot(snapshot, lang, content)
        tree = {key: value for key, value in tree.items() if key not in CONTROL}
        data |= {f'journal.{lang}': localized,
                 f'home.{lang}': {'empty': True} if front is None else _scrub(home.localize(front, lang, content)),
                 f'stack.{lang}': {**stack.versions(), **stack.counts(lab.root)},
                 f'lab.{lang}': _scrub({**tree, 'robots': robot_names(robots, lang), 'exported_at': time.time()})}

    clips = sorted({name for text in sources.values() for name in CLIP.findall(text)} | set(home.clips(front)))
    site_media = {f'site-{lang}-{clip}{suffix}' for clip in clips for lang in i18n.LANGUAGES
                  for suffix in ('.webp', '-poster.jpg')}
    found = {}
    for name in sorted(names | site_media):
        file = journal.image_file(lab.root, name) or journal.figure_file(lab.root, name)
        if file is None:
            missing.append(f'media/{name}')
        else:
            found[name] = file
    started = time.time()
    converted, failed, made = videos(found, video_cache)
    seconds = time.time() - started
    missing += [f'mp4/{name}' for name in failed]

    sizes = {'journal': 0, 'site': 0}
    (out / 'media').mkdir()
    for name, file in found.items():
        target = converted.get(name, name)
        file = made.get(target, file)
        shutil.copyfile(file, out / 'media' / target)
        sizes['site' if name in site_media else 'journal'] += file.stat().st_size
    for name, value in data.items():
        _write_json(out / 'data' / f'{name}.json', renamed(value, converted))
    for source, name in pages.items():
        page = page_media(static_page(sources[source]), converted)
        (out / name).write_text(versioned(page, '.html', build), encoding='utf-8', newline='')

    (out / '.htaccess').write_text(HTACCESS, encoding='utf-8', newline='\n')
    written = [path for path in out.rglob('*') if path.is_file() and path.name != MARKER]
    summary = {'complete': True, 'exported_at': time.time(), 'build': build, 'pages': sorted(pages.values()),
               'files': len(written),
               'bytes': sum(path.stat().st_size for path in written), 'journal_media_bytes': sizes['journal'],
               'site_media_bytes': sizes['site'], 'missing': sorted(missing),
               'video': {'mp4': len(converted), 'failed': failed, 'seconds': round(seconds, 1),
                         'webp_bytes': sum(found[name].stat().st_size for name in converted),
                         'mp4_bytes': sum(made[name].stat().st_size for name in converted.values())}}
    (out / MARKER).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if zip_file is not None:
        zip_file = Path(zip_file).resolve()
        if out in zip_file.parents:
            raise ValueError('Архив сайта не кладётся внутрь самой выгрузки')
        zip_file.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'w') as archive:
            for path in sorted(written):
                archive.write(path, path.relative_to(out).as_posix(),
                              zipfile.ZIP_STORED if path.suffix in STORED else zipfile.ZIP_DEFLATED)
        summary['zip_bytes'] = zip_file.stat().st_size
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='folder for the static site (new, or a previous export)')
    parser.add_argument('--zip', help='also pack the site into this .zip')
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN), help='the original run; its lab is <run-dir>_lab')
    args = parser.parse_args()
    summary = export(args.out, args.run_dir, args.zip)
    megabytes = lambda value: f'{value / 1e6:.1f} МБ'.replace('.', ',')
    print(f'Сайт: {Path(args.out).resolve()} — страницы {", ".join(summary["pages"])}; {summary["files"]} файлов, '
          f'{megabytes(summary["bytes"])} (медиа дневника {megabytes(summary["journal_media_bytes"])}, '
          f'ролики главной {megabytes(summary["site_media_bytes"])})')
    video = summary['video']
    print(f'Ролики MP4: {video["mp4"]} ({megabytes(video["webp_bytes"])} WebP → {megabytes(video["mp4_bytes"])}), '
          f'{video["seconds"]:.0f} с')
    for name, error in video['failed'].items():
        print(f'Не перекодирован, выгружен как WebP: {name} — {error}')
    if 'zip_bytes' in summary:
        print(f'Архив: {Path(args.zip).resolve()} — {megabytes(summary["zip_bytes"])}')
    if summary['missing']:
        print('Не найдено:', ', '.join(summary['missing']))


if __name__ == '__main__':
    main()
