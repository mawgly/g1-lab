"""Interface languages: Russian is the source text and the default; English and Simplified Chinese are translations.

Code writes every user-facing message in Russian, and that Russian text is the key under which
humanoid_sim/web/i18n/<lang>.json (en, zh) keeps its translation; the browser loads the same files (web/i18n.js).
An entry can be a template: "{name}" stands for any text, "{n}", "{n1}", ... for a number, so
"Выберите от 1 до {n} дорожек" also translates the message the code formatted from it, and the text that filled a
text placeholder is translated in turn. Anything without a translation stays Russian, and so does every message when
no language is asked for: the CLI, the agents and the files on disk are unchanged.

Free text the lab writes (the journal, branch names and their notes) is translated paragraph by paragraph by
humanoid_sim.journal_translate into <lab>/journal/i18n/<lang>.json; Content reads that cache. The localize_* helpers
turn the server's JSON answers into the asked language for the browser, which asks with ?lang=en or ?lang=zh.
"""
import ast
import functools
import hashlib
import json
import re
from pathlib import Path

LANGUAGES = ('ru', 'en', 'zh')
PACKAGE = Path(__file__).resolve().parent
WEB = PACKAGE / 'web'
CATALOGS = WEB / 'i18n'
CYRILLIC = re.compile('[А-Яа-яЁё]')
PLACEHOLDER = re.compile(r'\{(\w+)\}')
NUMERIC = re.compile(r'n\d*')  # placeholder names that match a number only
NUMBER = r'[-+−]?\d(?:[\d\u00a0\u202f ]*\d)?(?:[.,]\d+)?(?:e[-+]?\d+)?'
# A paragraph is the unit of journal translation: text between blank lines, as the browser splits it.
PARAGRAPHS = re.compile(r'(\n\s*\n)')


def language(query):
    """The language a request asks for (?lang=, parse_qs form), Russian unless it names a known one."""
    value = (query.get('lang') or ['ru'])[0]
    return value if value in LANGUAGES else 'ru'


def fill(text, params):
    """Puts params into {placeholders}; unknown names stay as they are (str.format would fail on other braces)."""
    return PLACEHOLDER.sub(lambda m: str(params[m[1]]) if m[1] in params else m[0], text)


def numbers(value, lang):
    """Russian-formatted numbers (decimal comma, spaces between thousands) in English and Chinese form."""
    if lang == 'ru':
        return value
    value = re.sub(r'(?<=\d),(?=\d)', '.', value)
    # Thousands groups of whole numbers only: \u00ab0,093 200\u00bb is two numbers.
    return re.sub(r'(?<![\d.,])\d{1,3}(?:[\u00a0\u202f ]\d{3})+(?!\d)',
                  lambda m: re.sub(r'[\u00a0\u202f ]', ',', m[0]), value)


class Catalog:
    """Translations of one language: exact Russian keys, and the keys with placeholders as patterns."""

    def __init__(self, entries):
        self.exact = dict(entries)
        self.patterns = []
        for key, value in entries.items():
            names = PLACEHOLDER.findall(key)
            if not names:
                continue
            pattern, position = '', 0
            for match in PLACEHOLDER.finditer(key):
                pattern += re.escape(key[position:match.start()])
                group = NUMBER if NUMERIC.fullmatch(match[1]) else '.+?'
                pattern += f'(?P<{match[1]}>{group})' if pattern.count(f'(?P<{match[1]}>') == 0 else f'(?P={match[1]})'
                position = match.end()
            pattern += re.escape(key[position:])
            # The most specific pattern first: the one with the most fixed text.
            self.patterns.append((len(PLACEHOLDER.sub('', key)), re.compile(pattern, re.S), value))
        self.patterns.sort(key=lambda item: -item[0])

    def translate(self, text, lang, inner=None):
        """text in this catalog's language, or None. inner(value) translates what filled a text placeholder."""
        if text in self.exact:
            return self.exact[text]
        for _, pattern, value in self.patterns:
            match = pattern.fullmatch(text)
            if match:
                params = {name: numbers(found, lang) if NUMERIC.fullmatch(name) else (inner or self.text)(found)
                          for name, found in match.groupdict().items()}
                return fill(value, params)
        return None

    def text(self, text):
        return self.translate(text, None) or text


@functools.lru_cache(maxsize=None)
def catalog(lang):
    if lang == 'ru':
        return Catalog({})
    return Catalog(json.loads((CATALOGS / f'{lang}.json').read_text(encoding='utf-8')))


def translate(text, lang, inner=None):
    """A server message in the asked language; Russian (or anything not in the catalog) stays as it is."""
    if lang == 'ru' or not isinstance(text, str) or not CYRILLIC.search(text):
        return text
    found = catalog(lang).translate(text, lang, inner or (lambda value: translate(value, lang)))
    return text if found is None else found


def gettext(lang):
    """_(russian, **params) for code that formats its own messages (the HTML report)."""
    table = catalog(lang).exact

    def _(text, **params):
        return fill(table.get(text, text), params)
    return _


def unit_id(text):
    """The cache key of a translated paragraph: its Russian text, stripped."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16]


class Content:
    """Translations of the lab's own text (journal, branch names, notes) from <lab>/journal/i18n/<lang>.json,
    written by humanoid_sim.journal_translate; re-read when the file changes."""

    def __init__(self, root):
        self.folder = Path(root) / 'journal' / 'i18n'
        self.cache = {}

    def units(self, lang):
        path = self.folder / f'{lang}.json'
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return {}
        if self.cache.get(lang, (None,))[0] != stamp:
            try:
                units = json.loads(path.read_text(encoding='utf-8')).get('units', {})
            except (OSError, ValueError):  # being replaced: keep the last good copy
                return self.cache.get(lang, (None, {}))[1]
            self.cache[lang] = stamp, {key: unit['text'] for key, unit in units.items() if unit.get('text')}
        return self.cache[lang][1]

    def text(self, text, lang):
        """text with each paragraph replaced by its cached translation, else by the catalog's, else kept."""
        if lang == 'ru' or not isinstance(text, str) or not CYRILLIC.search(text):
            return text
        units = self.units(lang)

        def paragraph(piece):
            if not CYRILLIC.search(piece):
                return piece
            found = units.get(unit_id(piece))
            if found is not None:
                return piece[:len(piece) - len(piece.lstrip())] + found + piece[len(piece.rstrip()):]
            return translate(piece, lang, lambda value: self.text(value, lang))
        return ''.join(piece if i % 2 else paragraph(piece) for i, piece in enumerate(PARAGRAPHS.split(text)))


# --- JSON answers in the asked language -------------------------------------------------------------------------

JOURNAL_FIELDS = {'roadmap': ('title', 'detail'), 'sections': ('title', 'text'), 'path': ('title', 'change', 'result'),
                  'races': ('title', 'text'), 'entries': ('title', 'text'), 'check': ('title',),
                  'items': ('label', 'value', 'target'), 'iterations': ('title', 'question', 'decision'),
                  'messages': ('text',)}


def map_journal(journal, change):
    """A copy of the journal (journal.py layout) with change(text, kind, field) applied to every text people read;
    kind is the JOURNAL_FIELDS key of the object holding it."""
    def items(values, kind):
        return [{**v, **{f: change(v[f], kind, f) for f in JOURNAL_FIELDS[kind] if isinstance(v.get(f), str)}}
                for v in values or []]

    result = dict(journal)
    roadmap = []
    for stage in items(journal.get('roadmap'), 'roadmap'):
        story = stage.get('story')
        if story:
            stage['story'] = {**story, **{kind: items(story.get(kind), kind) for kind in ('sections', 'path', 'races')}}
        roadmap.append(stage)
    result['roadmap'] = roadmap
    check = journal.get('check')
    if check:
        result['check'] = {**items([check], 'check')[0], 'items': items(check.get('items'), 'items')}
    result['entries'] = items(journal.get('entries'), 'entries')
    result['iterations'] = [{**it, 'messages': items(it.get('messages'), 'messages')}
                            for it in items(journal.get('iterations'), 'iterations')]
    return result


def localize_journal(journal, lang, content):
    if lang == 'ru':
        return journal

    def change(text, kind, field):
        if kind == 'items' and not CYRILLIC.search(text):
            return numbers(text, lang)  # goal_check's values carry Russian decimal commas without any word
        return content.text(text, lang)
    return map_journal(journal, change)


def localize_snapshot(snapshot, lang, content):
    """/api/lab: branch names and notes (content), queue and hardware messages (catalog)."""
    if lang == 'ru':
        return snapshot
    experiments = [{**e, 'name': content.text(e.get('name'), lang)} for e in snapshot.get('experiments', [])]
    notes = {key: {**note, 'note': content.text(note.get('note'), lang)} if isinstance(note, dict) else note
             for key, note in (snapshot.get('notes') or {}).items()}
    queue = dict(snapshot.get('queue') or {})
    queue['items'] = [{**item, 'error': translate(item.get('error'), lang)} for item in queue.get('items', [])]
    hardware = snapshot.get('hardware')
    if isinstance(hardware, dict) and 'error' in hardware:
        hardware = {**hardware, 'error': translate(hardware['error'], lang)}
    return {**snapshot, 'experiments': experiments, 'notes': notes, 'queue': queue, 'hardware': hardware}


def localize_robots(robots, lang):
    """/api/lab/robots: robot, task and parameter labels."""
    if lang == 'ru':
        return robots
    return [{**robot, 'label': translate(robot['label'], lang),
             'tasks': [{**task, 'label': translate(task['label'], lang),
                        'parameters': [{**p, 'label': translate(p.get('label'), lang)} for p in task['parameters']]}
                       for task in robot['tasks']]} for robot in robots]


def localize_frame(frame, lang, content):
    """A playback or race frame: reward labels, branch names, the error and the playback's metric label."""
    if lang == 'ru':
        return frame
    result = {**frame, 'error': translate(frame.get('error'), lang)}
    if 'reward_labels' in frame:
        result['reward_labels'] = {key: translate(label, lang) for key, label in frame['reward_labels'].items()}
    for key in ('lanes', 'experiments'):
        if key in frame:
            result[key] = [{**item, 'name': content.text(item.get('name'), lang)} for item in frame[key]]
    if isinstance(frame.get('view'), dict) and frame['view'].get('metric_label'):
        result['view'] = {**frame['view'], 'metric_label': translate(frame['view']['metric_label'], lang)}
    return result


# --- What has to be translated: used by tests/test_i18n.py and the check below ----------------------------------

# Modules whose Russian strings reach the browser (messages, labels, check rows, the report).
SERVER_MODULES = ('webapp.py', 'lab.py', 'lab_api.py', 'lab_worker.py', 'reports.py', 'tasks/catalog.py',
                  'tasks/config.py', 'goal_check.py:items')
# The site (Home, Journal, Lab), the live 3D race and the arm's stage player. A listed file that does not exist
# (yet) is skipped, so the check runs while a page is being written.
WEB_SCRIPTS = ('site.js', 'journal.js', 'labtree.js', 'home.js', 'race.js', 'app.js', 'viewer3d.js')
WEB_PAGES = ('site.html', 'journal.html', 'lab.html', 'race.html', 'index.html')


def web_messages():
    """Russian keys the pages use: t('…') / tn(n, '…') in the scripts, text and readable attributes in the HTML."""
    found = set()
    for name in WEB_SCRIPTS:
        if not (WEB / name).is_file():
            continue
        source = (WEB / name).read_text(encoding='utf-8')
        for match in re.finditer(r"\bt\(\s*'((?:[^'\\]|\\.)*)'|\btn\([^,]+,\s*'((?:[^'\\]|\\.)*)'", source):
            text = (match[1] if match[1] is not None else match[2]).replace("\\'", "'").replace('\\\\', '\\')
            if CYRILLIC.search(text):
                found.add(text)
    from html.parser import HTMLParser

    class Texts(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name in ('title', 'aria-label', 'alt', 'placeholder', 'label', 'value') and value:
                    self.add(value)

        def handle_data(self, data):
            self.add(data)

        def add(self, text):
            text = re.sub(r'\s+', ' ', text).strip()
            if CYRILLIC.search(text):
                found.add(text)
    for name in WEB_PAGES:
        if (WEB / name).is_file():
            Texts().feed((WEB / name).read_text(encoding='utf-8'))
    return found


def _template(node):
    """The catalog key of an f-string: each value becomes a placeholder named after its expression, a number one
    ({n}, {n1}, ...) when it has a format spec such as :g or .1f."""
    parts, used = [], set()
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(value.value)
            continue
        expr = value.value
        name = 'n' if value.format_spec else (expr.id if isinstance(expr, ast.Name) else
                                              expr.attr if isinstance(expr, ast.Attribute) else 'value').lower()
        base, number = name, 1
        while name in used:
            number += 1
            name = f'{base}{number}'
        used.add(name)
        parts.append('{' + name + '}')
    return ''.join(parts)


def server_messages():
    """Russian string literals of SERVER_MODULES; an f-string gives a template with a placeholder per value."""
    found = set()
    for entry in SERVER_MODULES:
        name, _, function = entry.partition(':')
        tree = ast.parse((PACKAGE / name).read_text(encoding='utf-8'))
        if function:
            tree = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function)
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                      and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
        inside = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                inside.update(id(value) for value in node.values)
                text = _template(node)
                if CYRILLIC.search(text):
                    found.add(text)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                    and id(node) not in inside and CYRILLIC.search(node.value)):
                found.add(node.value)
    return found


def missing(lang):
    """(web keys without a translation, server strings no catalog entry matches) for one language; a server
    template counts as covered by an entry of the same shape whatever its placeholders are named."""
    table = catalog(lang)
    web = sorted(key for key in web_messages() if key not in table.exact)
    shapes = {PLACEHOLDER.sub('{}', key) for key in table.exact}
    server = sorted(text for text in server_messages()
                    if text not in table.exact and PLACEHOLDER.sub('{}', text) not in shapes)
    return web, server


def check_catalogs():
    """Problems in the catalogs: Cyrillic left in a translation, placeholders that differ from the key."""
    problems = []
    for lang in LANGUAGES[1:]:
        for key, value in catalog(lang).exact.items():
            if CYRILLIC.search(value):
                problems.append(f'{lang}: Cyrillic in the translation of {key!r}: {value!r}')
            if sorted(set(PLACEHOLDER.findall(key))) != sorted(set(PLACEHOLDER.findall(value))):
                problems.append(f'{lang}: placeholders differ in {key!r}: {value!r}')
    return problems
