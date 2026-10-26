"""The home page's changing part (site.html: the «updated» line, the roadmap stage, the number cards and «what the
robot can do»), as data: <lab>/journal/home.json, written by scripts/site/home_update.py from numbers the lab already
has (the candidates' robot package and passport, skills_check, goal_check, the journal's roadmap). The browser reads it
through /api/lab/home?lang=… (webapp) or data/home.<lang>.json (site_export); the page's fixed text stays in site.html.

    {"version": 1, "at", "date": "YYYY-MM-DD",
     "roadmap": {"total", "done", "active": [{"number", "title"}]},
     "stats": [{"value", "unit"?, "label", "accent"?}],
     "candidates": [{"id", "title", "about", "summary": [row], "skills": [{"title", "clip"?, "alt"?, "rows": [row]}]}],
     "missing": [{"what", "why"}], "sources": [{"file", "what", "md5"}],
     "research": [{"title", "clip", "alt", "licence", "text", "stage"}]}
    research: research experiments shown in a block of their own after the candidates, apart from them (a policy
    trained on LAFAN1, CC BY-NC-ND, is research only: the candidates' cards come from robot packages alone, which
    package.py's licence check guards; a research block is a clip of the robot and its text, nothing else).
    row: {"label", "value": text | null, "unit"?: an i18n.UNITS symbol, "where"?: one of WHERE, "v0"?, "target"?,
          "weak"?: bool, "why"?: why there is no value}

Russian is the source, as in the journal: titles, labels and reasons are translated by humanoid_sim.journal_translate
paragraph by paragraph (texts() is part of what it exports), or by the catalog when a label is an interface string
(skills_check's and the passport's labels are); values are numbers in Russian form plus a unit symbol, so they need no
translation. A clip is the name of the home page's pre-rendered site-<lang>-<name>.webp (+ -poster.jpg) in the journal
folder.
"""
import re

from . import i18n
from .core import read_json, write_json

VERSION = 1
NAME = 'home.json'
CLIP_NAME = re.compile(r'^[A-Za-z0-9_-]{1,60}$')
# Where a row's number comes from; the page shows it as a short tag (home.js WHERE).
WHERE = ('bridge', 'skills', 'goal', 'package', 'bridge_copy')
# Fields people read, by the kind of object holding them: text (journal_translate / catalog) and values (numbers).
TEXT_FIELDS = {'home': (), 'roadmap': (), 'active': ('title',), 'stats': ('label',),
               'candidates': ('title', 'about'), 'skills': ('title', 'alt'), 'rows': ('label', 'why'),
               'missing': ('what', 'why'), 'research': ('title', 'alt', 'licence', 'text')}
VALUE_FIELDS = ('value', 'v0', 'target')


def path(root):
    from . import journal
    return journal.folder(root) / NAME


def load(root):
    """The saved home page data, or None before the first home_update."""
    data = read_json(path(root))
    return data if isinstance(data, dict) and data.get('version') == VERSION else None


def _text(value, what, limit=600, empty=False):
    if value is None and empty:
        return None
    if not isinstance(value, str) or (not value.strip() and not empty) or len(value) > limit:
        raise ValueError(f'Главная: {what} — текст до {limit} символов')
    return value.strip()


def _row(row):
    if not isinstance(row, dict):
        raise ValueError('Главная: строка числа — объект')
    clean = {'label': _text(row.get('label'), 'подпись числа', 300)}
    value = row.get('value')
    clean['value'] = None if value is None else _text(str(value), 'значение', 200)
    if row.get('unit') is not None and clean['value'] is not None:
        if row['unit'] not in i18n.UNITS:
            raise ValueError(f'Главная: единица {row["unit"]!r} не из i18n.UNITS')
        clean['unit'] = row['unit']
    if row.get('where') is not None:
        if row['where'] not in WHERE:
            raise ValueError(f'Главная: источник {row["where"]!r} не из {", ".join(WHERE)}')
        clean['where'] = row['where']
    for key in ('v0', 'target'):
        if row.get(key) not in (None, ''):
            clean[key] = _text(str(row[key]), key, 200)
    if row.get('weak'):
        clean['weak'] = True
    if clean['value'] is None:
        clean['why'] = _text(row.get('why'), 'причина пустого поля', 300)
    return clean


def validate(data):
    """data in the layout above with every field checked; ValueError otherwise."""
    if not isinstance(data, dict):
        raise ValueError('Главная: объект')
    roadmap = data.get('roadmap') or {}
    active = [{'number': int(stage['number']), 'title': _text(stage.get('title'), 'этап плана', 200)}
              for stage in roadmap.get('active') or []]
    stats = []
    for stat in data.get('stats') or []:
        row = {'value': _text(str(stat.get('value')), 'число карточки', 60),
               'label': _text(stat.get('label'), 'подпись карточки', 400)}
        if stat.get('unit') is not None:
            if stat['unit'] not in i18n.UNITS:
                raise ValueError(f'Главная: единица {stat["unit"]!r} не из i18n.UNITS')
            row['unit'] = stat['unit']
        if stat.get('accent'):
            row['accent'] = True
        stats.append(row)
    candidates = []
    for candidate in data.get('candidates') or []:
        skills = []
        for card in candidate.get('skills') or []:
            clip = card.get('clip')
            if clip is not None and not CLIP_NAME.match(str(clip)):
                raise ValueError('Главная: ролик — имя из латинских букв, цифр, «_» и «-»')
            skills.append({'title': _text(card.get('title'), 'навык', 200), **({'clip': clip} if clip else {}),
                           **({'alt': _text(card['alt'], 'описание ролика', 300)} if card.get('alt') else {}),
                           'rows': [_row(row) for row in card.get('rows') or []]})
        candidates.append({'id': _text(candidate.get('id'), 'кандидат', 40),
                           'title': _text(candidate.get('title'), 'заголовок кандидата', 200),
                           'about': _text(candidate.get('about'), 'о кандидате', 600, empty=True) or '',
                           'summary': [_row(row) for row in candidate.get('summary') or []], 'skills': skills})
    research = []
    for block in data.get('research') or []:
        clip = block.get('clip')
        if not CLIP_NAME.match(str(clip or '')):
            raise ValueError('Главная: ролик исследования — имя из латинских букв, цифр, «_» и «-»')
        research.append({'title': _text(block.get('title'), 'заголовок исследования', 200), 'clip': clip,
                         'alt': _text(block.get('alt'), 'описание ролика', 300),
                         'licence': _text(block.get('licence'), 'лицензия исследования', 600),
                         'text': _text(block.get('text'), 'текст исследования', 1200),
                         'stage': _text(block.get('stage'), 'этап исследования', 80)})
    return {'version': VERSION, 'at': float(data['at']), 'date': _text(data.get('date'), 'дата', 10),
            'research': research,
            'roadmap': {'total': int(roadmap.get('total') or 0), 'done': int(roadmap.get('done') or 0),
                        'active': active},
            'stats': stats, 'candidates': candidates,
            'missing': [{'what': _text(m.get('what'), 'пустое поле', 300), 'why': _text(m.get('why'), 'причина', 300)}
                        for m in data.get('missing') or []],
            'sources': [{key: str(s.get(key) or '') for key in ('file', 'what', 'md5')} for s in data.get('sources') or []]}


def save(root, data):
    clean = validate(data)
    write_json(path(root), clean)
    return clean


def map_home(home, text=None, value=None):
    """A copy of the home data with text(string, kind, field) applied to every text people read and value(row,
    field) to every value (kind: a TEXT_FIELDS key). Either may be None (kept)."""
    def fields(item, kind):
        item = dict(item)
        for field in TEXT_FIELDS[kind]:
            if text and isinstance(item.get(field), str) and item[field]:
                item[field] = text(item[field], kind, field)
        if kind in ('rows', 'stats') and value:
            for field in VALUE_FIELDS:
                if isinstance(item.get(field), str):
                    item[field] = value(item, field)
        return item

    result = dict(home)
    result['roadmap'] = {**home.get('roadmap', {}),
                         'active': [fields(s, 'active') for s in (home.get('roadmap') or {}).get('active', [])]}
    result['stats'] = [fields(s, 'stats') for s in home.get('stats', [])]
    result['candidates'] = [{**fields(c, 'candidates'),
                             'summary': [fields(r, 'rows') for r in c.get('summary', [])],
                             'skills': [{**fields(card, 'skills'), 'rows': [fields(r, 'rows') for r in card.get('rows', [])]}
                                        for card in c.get('skills', [])]}
                            for c in home.get('candidates', [])]
    result['missing'] = [fields(m, 'missing') for m in home.get('missing', [])]
    result['research'] = [fields(r, 'research') for r in home.get('research', [])]
    return result


def texts(home):
    """Every Russian text of the home data people read (for journal_translate)."""
    found = []
    if home:
        map_home(home, lambda text, kind, field: found.append(text) or text)
    return found


def localize(home, lang, content):
    """The home data for the browser in `lang`: texts translated (content: i18n.Content), values in the language's
    number form with their unit symbol joined (Russian included). Sources stay out (they name local files)."""
    def text(value, kind, field):
        return content.text(value, lang)

    def number(item, field):
        shown = item[field]
        shown = i18n.numbers(shown, lang) if not i18n.CYRILLIC.search(shown) else content.text(shown, lang)
        return i18n.unit(shown, item['unit'], lang) if field == 'value' and item.get('unit') else shown

    result = map_home(home, text if lang != 'ru' else None, number)
    for item in [*result['stats'], *(r for c in result['candidates'] for r in c['summary']),
                 *(r for c in result['candidates'] for card in c['skills'] for r in card['rows'])]:
        item.pop('unit', None)
    result.pop('sources', None)
    return result


def clips(home):
    """Names of the clips the home data shows."""
    return sorted({card['clip'] for c in (home or {}).get('candidates', []) for card in c.get('skills', [])
                   if card.get('clip')} | {block['clip'] for block in (home or {}).get('research', [])})
