"""Lists the Cyrillic the browser would show in English or Chinese.

    .venv\\Scripts\\python.exe -m humanoid_sim.i18n_check [--url http://127.0.0.1:8766] [--lang en zh] [--race]

Without --url: page and server strings that have no entry in humanoid_sim/web/i18n/<lang>.json. With --url, also
what that server answers in the language (read-only GETs): branch names and notes, robot and parameter labels, the
journal, report pages and, with --race, race frames (opening a race loads its stages). Every Cyrillic string found is
either the lab's own text (journal, names, notes) that humanoid_sim.journal_translate has not translated yet, or an
interface string, a bug: the exit code is 1 when there is any of the latter. For the pages as drawn, paste
scripts/i18n/dom_scan.js into the browser console.
"""
import argparse
import html
import json
import re
import sys
import urllib.request
from urllib.parse import quote

from . import i18n

CYRILLIC = i18n.CYRILLIC


def fetch(url, path, lang):
    with urllib.request.urlopen(url + path + ('&' if '?' in path else '?') + 'lang=' + lang, timeout=120) as reply:
        body = reply.read().decode('utf-8')
    return json.loads(body) if path.startswith('/api/') else body


def live(url, lang, race=False):
    """{'lab text': [...], 'interface': [...]} of Cyrillic strings in the server's answers."""
    found = {'lab text': [], 'interface': []}

    def add(kind, where, text):
        if isinstance(text, str) and CYRILLIC.search(text):
            found[kind].append(f'{where}: {text[:160]}')
    lab = fetch(url, '/api/lab', lang)
    for e in lab['experiments']:
        add('lab text', 'branch name', e['name'])
    for key, note in (lab.get('notes') or {}).items():
        add('lab text', f'note {key}', (note or {}).get('note'))
    for item in lab['queue'].get('items', []):
        add('interface', 'queue error', item.get('error'))
    add('interface', 'hardware', (lab.get('hardware') or {}).get('error'))
    for robot in fetch(url, '/api/lab/robots', lang):
        add('interface', 'robot', robot['label'])
        for task in robot['tasks']:
            add('interface', 'task', task['label'])
            for parameter in task['parameters']:
                add('interface', 'parameter', parameter['label'])
            if race and task.get('race') and any(e['config'].get('robot_id') == robot['robot_id']
                                                 and e.get('stage_count') for e in lab['experiments']):
                frame = fetch(url, f'/api/race/frame?robot={robot["robot_id"]}&task={task["task_id"]}', lang)
                for label in frame['reward_labels'].values():
                    add('interface', 'reward label', label)
                for lane in frame['lanes']:
                    add('lab text', 'race lane', lane['name'])
                add('interface', 'race error', frame.get('error'))

    def journal_text(text, kind, field):
        add('interface' if kind == 'items' else 'lab text', f'journal {kind}.{field}', text)
        return text
    i18n.map_journal(fetch(url, '/api/lab/journal', lang), journal_text)
    for e in lab['experiments']:
        if not e.get('has_report'):
            continue
        page = fetch(url, '/report.html?experiment=' + quote(e['id']), lang)
        page = re.sub(r'<(title|h1|pre)>.*?</\1>', ' ', page, flags=re.S)  # the branch name; saved data as stored
        for text in re.split(r'<[^>]+>', page):
            add('interface', f'report {e["id"]}', html.unescape(text).strip())
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', help='a running lab server, e.g. http://127.0.0.1:8766')
    parser.add_argument('--lang', nargs='+', choices=i18n.LANGUAGES[1:], default=list(i18n.LANGUAGES[1:]))
    parser.add_argument('--race', action='store_true', help='also open each robot\'s race (loads stages)')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    bad = 0
    for problem in i18n.check_catalogs():
        print(problem)
        bad += 1
    for lang in args.lang:
        web, server = i18n.missing(lang)
        for text in web + server:
            print(f'{lang}: no translation for {text!r}')
        bad += len(web) + len(server)
        if args.url:
            found = live(args.url.rstrip('/'), lang, args.race)
            print(f'{lang}: {len(found["lab text"])} lab texts still Russian (journal_translate), '
                  f'{len(found["interface"])} interface strings in Russian')
            for row in found['interface']:
                print('  ' + row)
            bad += len(found['interface'])
    print('ok' if not bad else f'{bad} problems')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
