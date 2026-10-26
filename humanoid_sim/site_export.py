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
media/ holds the journal pictures, clips and figures that data names and the home page's site-<lang>-<name> clips
(data-clip in a page, and the clips the home data names), nothing else from the journal folder. No live race: the
pages show pre-rendered clips.
"""
import argparse
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from . import home, i18n, journal
from .core import DEFAULT_RUN, ROOT, read_json
from .lab import Laboratory

WEB = i18n.WEB
# Source page -> exported name. A page is part of the site once it has the site-mode meta (the old lab.html has not).
PAGES = {'site.html': 'index.html', 'journal.html': 'journal.html', 'lab.html': 'lab.html'}
SITE_MODE = re.compile(r'(<meta\s+name="site-mode"\s+content=")live(")')
# Local files a page, script or style loads; page links (.html) are not followed, the pages are PAGES.
REFERENCES = {
    '.html': re.compile(r'''\b(?:src|href)\s*=\s*["']([^"'#?]+)'''),
    '.js': re.compile(r'''(?:\bfrom\s*|\bimport\s*\(?\s*)['"]([^'"]+)['"]'''),
    '.css': re.compile(r'''@import\s+['"]([^'"]+)['"]|url\(\s*['"]?([^'")#?]+)'''),
}
CLIP = re.compile(r'\bdata-clip="([A-Za-z0-9_-]+)"')
# Parts of the /api/lab answer, and of a task in /api/lab/robots, that run the lab rather than describe it.
CONTROL = ('queue', 'worker_active', 'active_trainers', 'job', 'hardware')
TASK_CONTROL = ('parameters', 'jobs', 'backends', 'training_backends')
STORED = ('.webp', '.jpg', '.png')  # already compressed: zipped as they are
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


def export(out, run_dir=DEFAULT_RUN, zip_file=None):
    """Writes the static site into `out`; returns a summary (files, bytes, media bytes, what was missing)."""
    pages = site_pages()
    out = _prepare(out)
    files, missing = dependencies(pages)
    for source, name in pages.items():
        (out / name).write_text(static_page((WEB / source).read_text(encoding='utf-8')), encoding='utf-8',
                                newline='')
    for name in files + [f'i18n/{lang}.json' for lang in i18n.LANGUAGES[1:]]:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WEB / name, out / name)

    from .tasks.catalog import catalog
    lab = Laboratory(run_dir)
    content = i18n.Content(lab.root)
    saved, snapshot, robots = journal.load(lab.root), lab_snapshot(lab), catalog()
    front = home.load(lab.root)
    names = set()
    for lang in i18n.LANGUAGES:
        localized = _scrub(i18n.localize_journal(saved, lang, content))
        names |= media_names(localized)
        _write_json(out / 'data' / f'journal.{lang}.json', localized)
        _write_json(out / 'data' / f'home.{lang}.json',
                    {'empty': True} if front is None else _scrub(home.localize(front, lang, content)))
        tree = i18n.localize_snapshot(snapshot, lang, content)
        tree = {key: value for key, value in tree.items() if key not in CONTROL}
        _write_json(out / 'data' / f'lab.{lang}.json',
                    _scrub({**tree, 'robots': robot_names(robots, lang), 'exported_at': time.time()}))

    clips = sorted({name for source in pages for name in CLIP.findall((WEB / source).read_text(encoding='utf-8'))}
                   | set(home.clips(front)))
    site_media = {f'site-{lang}-{clip}{suffix}' for clip in clips for lang in i18n.LANGUAGES
                  for suffix in ('.webp', '-poster.jpg')}
    sizes = {'journal': 0, 'site': 0}
    (out / 'media').mkdir()
    for name in sorted(names | site_media):
        file = journal.image_file(lab.root, name) or journal.figure_file(lab.root, name)
        if file is None:
            missing.append(f'media/{name}')
            continue
        shutil.copyfile(file, out / 'media' / name)
        sizes['site' if name in site_media else 'journal'] += file.stat().st_size

    written = [path for path in out.rglob('*') if path.is_file() and path.name != MARKER]
    summary = {'complete': True, 'exported_at': time.time(), 'pages': sorted(pages.values()), 'files': len(written),
               'bytes': sum(path.stat().st_size for path in written), 'journal_media_bytes': sizes['journal'],
               'site_media_bytes': sizes['site'], 'missing': sorted(missing)}
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
    if 'zip_bytes' in summary:
        print(f'Архив: {Path(args.zip).resolve()} — {megabytes(summary["zip_bytes"])}')
    if summary['missing']:
        print('Не найдено:', ', '.join(summary['missing']))


if __name__ == '__main__':
    main()
