"""English and Chinese for the lab's own Russian text (the «Дневник» journal, branch names, branch notes) and for
interface strings the catalogs humanoid_sim/web/i18n/{en,zh}.json do not have yet.

Only what is new or changed gets translated. The unit of lab text is a paragraph (text between blank lines, as the
browser shows it), cached under the hash of its Russian text in <lab>/journal/i18n/<lang>.json; humanoid_sim/i18n.py
reads that cache and the browser shows a paragraph's translation once it exists, the Russian until then. An
interface string goes into the catalogs (code: commit them).

Language models translate, Python only checks: the lab-translator agent (Haiku) takes one batch, writes its answer
and hands it in with `submit`, which stores every section that keeps the Russian's numbers, units, branch ids and
Latin-letter words (and {placeholders}) and rejects the rest with the reasons; the agent fixes those and submits
again. A section rejected three times is flagged (flags.json) and left out of later batches. Terms come from
docs/i18n-glossary.md, which every batch carries.

    python -m humanoid_sim.journal_translate status
    python -m humanoid_sim.journal_translate export [--chars 8000]       # <work>/batch-001.md, ...
    python -m humanoid_sim.journal_translate submit <answer.md> ...       # check and store; rejections listed
    python -m humanoid_sim.journal_translate import                       # submit every answer in <work>
    python -m humanoid_sim.journal_translate review --lang zh --sample 30 # for a stronger model's spot check

A batch section is "@@ <id> en zh" and the Russian; an answer has one section per language, "@@ <id> en" and the
translation. `export --redo zh` also exports stored, unreviewed translations to translate again (after the glossary
changed); `--match связк` limits a batch to the texts with that term. A review batch shows the Russian with the stored translation; the reviewer answers <batch>.fix.md with
the paragraphs it corrects, which replace the stored text, and the batch counts as reviewed.

The work folder is .tools/journal-i18n of this checkout (--work). In runs/ only the cache files are written.
"""
import argparse
import collections
import random
import re
import sys
import time
from pathlib import Path

from . import i18n
from .core import DEFAULT_RUN, ROOT, read_json, write_json

LANGS = ('en', 'zh')
NAMES = {'en': 'English', 'zh': 'Simplified Chinese'}
MAX_ATTEMPTS = 3
WORK = ROOT / '.tools' / 'journal-i18n'
GLOSSARY = ROOT / 'docs' / 'i18n-glossary.md'
HEADER = re.compile(r'^@@ ([0-9a-f]{16})((?:[ \t]+(?:en|zh))*)[ \t]*$', re.M)
# A Latin-letter word: identifiers, ids (exp-6986f1d9…), file names, paths. After '/', '.' or ':' only a letter
# continues it, so "mm/2.1°" is the unit and a number.
LATIN = re.compile(r'(?<![A-Za-zА-Яа-яЁё0-9_])[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+|[./:][A-Za-z_][A-Za-z0-9_]*)*')
NUMBER = re.compile(r'\d+(?:\.\d+)?')
# Russian unit after a number -> the symbol a translation must write after the same number.
UNITS = {'рад/с': 'rad/s', 'м/с': 'm/s', 'Н·м': 'N·m', 'млрд': 'B', 'млн': 'M', 'тыс.': 'k', 'мин': 'min', 'мм': 'mm',
         'см': 'cm', 'мс': 'ms', 'рад': 'rad', 'Гц': 'Hz', 'Дж': 'J', 'кг': 'kg', 'ГБ': 'GB', 'м': 'm', 'с': 's',
         'ч': 'h', 'Н': 'N'}
ALIASES = {'zh': {'s': ('秒',), 'ms': ('毫秒',), 'min': ('分钟',), 'h': ('小时',), 'm': ('米',), 'mm': ('毫米',),
                  'cm': ('厘米',), 'kg': ('千克',), 'M': ('百万',), 'k': ('千',)}}
# «с» after a number is seconds or the preposition «with» («корень 4 с исправленным обучением», «от 5451,4 с
# lock_wrists»): count it as seconds only before punctuation, the end or a word that follows a time («10 с при 1 м/с»,
# «за 10 с без падений»). A number inside a word (WSL2, G1) is no quantity.
AFTER_SECONDS = (r'(?=\s*(?:$|[^\sА-Яа-яЁёA-Za-z0-9]|(?:при|без|вместо|и|на|до|после|против|а|но|или|за|из|в|у|от|по|'
                 r'когда|пока|уже|подряд)(?![А-Яа-яЁё])))')
RU_UNIT = re.compile(r'(?<![A-Za-z0-9_.])(\d+(?:[.,]\d+)?)[   ]?(?:('
                     + '|'.join(map(re.escape, [u for u in UNITS if u != 'с']))
                     + r')(?![А-Яа-яЁёA-Za-z])|(с)' + AFTER_SECONDS + ')')
# «20 октября» is «10月20日» in Chinese: the month's number may appear in a translation.
MONTHS = {name: number for number, name in enumerate(
    ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября',
     'декабря'), start=1)}

RULES = '''Translate every section below from Russian. A section starts with "@@ <id> <languages>" (en = English,
zh = Simplified Chinese). Answer with one section per language listed: the line "@@ <id> <language>", then the
translation. The texts are a robotics lab's journal, branch names and notes (reinforcement learning of a Unitree G1
humanoid in MuJoCo) and interface strings: plain, exact, technical. Translate every sentence; do not shorten,
explain or add anything; no comments and no Russian in the answer.

A program checks each answer section and rejects it with the reason if it breaks a rule:
- Every Latin-letter word stays exactly as written: identifiers (feet_flat, arm_press, lock_wrists), branch ids
  (exp-facb20de4bfe472c), file names, commands, PPO, GPU, CPU, Unitree, G1, MuJoCo, seed.
- Every number digit for digit; a decimal point instead of the Russian comma (0,5 -> 0.5); no digits turned into
  words or words into digits. Dates such as 22.10 and times such as 14:05 as written.
- Units after numbers as symbols: м m, мм mm, см cm, с s, мс ms, мин min, ч h, м/с m/s, рад/с rad/s, рад rad, Н N,
  Н·м N·m, Гц Hz, Дж J, кг kg, ГБ GB; млн (steps) M, млрд B, тыс. k: "4975,4 млн шагов" -> "4975.4 M steps" (en),
  "4975.4 M 步" (zh).
- An interface string keeps each {{placeholder}} exactly.
- The same number of lines as the Russian.
Quotes «…» become “…”; Chinese uses Chinese punctuation.

Terms ({glossary}): Russian = English = Chinese (note)
{terms}
'''
REVIEW = '''Review these {name} translations of a robotics lab journal (reinforcement learning of a Unitree G1
humanoid in MuJoCo, sim2sim with Unitree's unitree_mujoco and g1_ctrl). Each section is "@@ <id> {lang}", the Russian
("[ru]") and the stored translation ("[{lang}]"). Fix wrong or inconsistent terms, meaning errors, omissions and
clumsy phrasing; leave correct translations alone (no style rewrites).

Write ONLY the sections you correct: the "@@ <id> {lang}" line unchanged, then the whole corrected paragraph (no
[ru]/[{lang}] prefix, no comments). The same program checks a correction as a translation: every number digit for
digit (decimal point), units as symbols after numbers (m, s, m/s, rad/s, mm, N, N·m; M for million steps), every
Latin-letter word unchanged, the same number of lines.

Terms ({glossary}): Russian = English = Chinese (note)
{terms}
'''


def lab_root(run_dir=DEFAULT_RUN):
    from .lab import Laboratory
    return Laboratory(run_dir).root


def folder(root):
    return Path(root) / 'journal' / 'i18n'


def glossary():
    """The terms of docs/i18n-glossary.md, one line each: Russian = English = Chinese (note)."""
    rows = []
    for line in GLOSSARY.read_text(encoding='utf-8').splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')] if line.startswith('|') else []
        if len(cells) == 4 and i18n.CYRILLIC.search(cells[0]) and cells[0] != 'Русский':
            rows.append(' = '.join(cells[:3]) + (f' ({cells[3]})' if cells[3] else ''))
    return '\n'.join('- ' + row for row in rows)


def texts(root):
    """Every Russian text the browser shows from the lab's own files: journal, branch names, current notes."""
    from . import journal
    found = []
    i18n.map_journal(journal.load(root), lambda text, kind, field: found.append(text) or text)
    for path in sorted(Path(root).glob('exp-*/experiment.json')):
        found.append((read_json(path, {}) or {}).get('name'))
    for note in (read_json(Path(root) / 'notes.json', {}) or {}).values():
        if isinstance(note, dict):
            found.append(note.get('note'))
    return [text for text in found if isinstance(text, str)]


def units(root):
    """{id: Russian paragraph} of lab text that needs a translation of its own (the catalog does not cover it)."""
    result = {}
    for text in texts(root):
        for i, piece in enumerate(i18n.PARAGRAPHS.split(text)):
            piece = piece.strip()
            if i % 2 or not i18n.CYRILLIC.search(piece):
                continue
            if all(not i18n.CYRILLIC.search(i18n.translate(piece, lang)) for lang in LANGS):
                continue
            result[i18n.unit_id(piece)] = piece
    return result


def interface():
    """{id: (Russian key, languages missing it)} of interface strings without a catalog entry (i18n.missing)."""
    result = {}
    for lang in LANGS:
        for key in sum(i18n.missing(lang), []):
            entry = result.setdefault(i18n.unit_id('catalog\n' + key), (key, []))
            entry[1].append(lang)
    return result


def load_cache(root, lang):
    return read_json(folder(root) / f'{lang}.json', {}) or {}


def load_flags(root):
    return read_json(folder(root) / 'flags.json', {}) or {}


def needed(root, langs=LANGS, flagged=False, redo=(), match=None):
    """[(id, Russian, languages, interface?)] to translate: interface strings missing in a catalog, lab text without a
    translation (flagged ones only when asked) and, for languages in redo, stored translations nobody reviewed;
    match (a regular expression) keeps the texts whose Russian it finds, e.g. a term the glossary changed."""
    flags = load_flags(root)
    caches = {lang: load_cache(root, lang).get('units', {}) for lang in langs}
    rows = [(key, russian, [lang for lang in missing if lang in langs], True)
            for key, (russian, missing) in interface().items() if not match or re.search(match, russian)]
    for key, russian in units(root).items():
        if match and not re.search(match, russian):
            continue
        wanted = [lang for lang in langs
                  if (key not in caches[lang] or (lang in redo and not caches[lang][key].get('reviewed')))
                  and (flagged or flags.get(lang, {}).get(key, {}).get('attempts', 0) < MAX_ATTEMPTS)]
        rows.append((key, russian, wanted, False))
    return [row for row in rows if row[2]]


# --- Checking an answer ---------------------------------------------------------------------------------------

def _ungroup(text, separators):
    """Joins thousands groups (2 000 000, 2,000,000) of whole numbers only: «0,093 200» stays two numbers."""
    return re.sub(r'(?<![\d.,])\d{1,3}(?:[' + separators + r']\d{3})+(?!\d)',
                  lambda m: re.sub('[' + separators + ']', '', m[0]), text)


def _canonical_ru(text):
    return re.sub(r'(?<=\d),(?=\d)', '.', _ungroup(text, '   '))


def _numbers(text):
    """Numbers outside Latin-letter words, as floats without sign."""
    return collections.Counter(float(n) for n in NUMBER.findall(LATIN.sub(' ', text)))


def _target_pairs(text, lang):
    pairs = collections.Counter()
    symbols = sorted({*UNITS.values(), *(a for v in ALIASES.get(lang, {}).values() for a in v)}, key=len, reverse=True)
    pattern = r'(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)[   ]?(' + '|'.join(map(re.escape, symbols)) + r')(?![A-Za-z])'
    aliases = {alias: unit for unit, values in ALIASES.get(lang, {}).items() for alias in values}
    for number, unit in re.findall(pattern, text):
        pairs[(float(number), aliases.get(unit, unit))] += 1
    return pairs


def problems(russian, text, lang):
    """Why `text` is not an acceptable translation of `russian` into `lang`; [] when it is."""
    text = text.strip()
    if not text:
        return ['empty']
    found = []
    if i18n.CYRILLIC.search(text):
        found.append('Cyrillic left: ' + ', '.join(sorted(set(re.findall(r'[А-Яа-яЁё]+', text)))[:5]))
    # A Latin word stays as it is; English may add a plural ending to a loanword (seed -> seeds).
    missing = sorted(word for word in set(LATIN.findall(russian)) if re.search('[A-Za-z]', word)
                     and not re.search(r'(?<![A-Za-z0-9_])' + re.escape(word) + r'(?:e?s)?(?![A-Za-z0-9_])', text))
    if missing:
        found.append('Latin words changed or missing: ' + ', '.join(missing[:8]))
    if sorted(set(i18n.PLACEHOLDER.findall(russian))) != sorted(set(i18n.PLACEHOLDER.findall(text))):
        found.append('placeholders differ: keep ' + ' '.join('{' + p + '}' for p in i18n.PLACEHOLDER.findall(russian)))
    if re.search(r'\d,(?!\d{3}(?!\d))\d', text):
        found.append('decimal comma: write a decimal point')
    source = _canonical_ru(russian)
    target = _ungroup(text, ',   ')
    lost = _numbers(source) - _numbers(target)
    extra = _numbers(target) - _numbers(source)
    extra -= collections.Counter(float(MONTHS[m]) for m in re.findall(r'\d{1,2}\s+(' + '|'.join(MONTHS) + ')', russian))
    if lost or extra:
        found.append('numbers differ' + (': missing ' + ', '.join(f'{n:g}' for n in lost) if lost else '')
                     + (': extra ' + ', '.join(f'{n:g}' for n in extra) if extra else ''))
    wanted = collections.Counter((float(n.replace(',', '.')), UNITS[u or s]) for n, u, s in RU_UNIT.findall(source))
    absent = wanted - _target_pairs(target, lang)
    if absent:
        found.append('units: write ' + ', '.join(f'{n:g} {u}' for (n, u) in absent))
    if russian.strip().count('\n') != text.count('\n'):
        found.append(f'line breaks: the Russian has {russian.strip().count(chr(10)) + 1} lines')
    low, high = (.45, 2.4) if lang == 'en' else (.12, 1.4)
    if len(russian) >= 60 and not low <= len(text) / len(russian) <= high:
        found.append('length looks wrong: translate the whole paragraph, nothing more')
    return found


# --- Batches and answers ----------------------------------------------------------------------------------------

def sections(text, default_lang=None):
    """[(id, [languages], body)] of a batch or an answer file; comments are dropped. A section without a language
    takes default_lang (a <lang>-… file)."""
    text = re.sub(r'<!--.*?-->\n?', '', text, flags=re.S)
    parts = HEADER.split(text)
    return [(parts[i], parts[i + 1].split() or ([default_lang] if default_lang else []), parts[i + 2].strip())
            for i in range(1, len(parts) - 2, 3)]


def answers(path):
    """{(id, language): text} of an answer or fix file."""
    lang = Path(path).name[:2] if Path(path).name[:3] in ('en-', 'zh-') else None
    return {(key, langs[0]): body for key, langs, body in sections(Path(path).read_text(encoding='utf-8'), lang)
            if len(langs) == 1}


def export(root, work=WORK, chars=8000, langs=LANGS, flagged=False, redo=(), match=None):
    """Writes <work>/batch-NNN.md of about `chars` Russian characters times languages; returns their paths."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob('batch-[0-9][0-9][0-9].md'):
        old.unlink()  # earlier batches in the work folder (outside runs/)
    flags = load_flags(root)
    head = '<!--\n' + RULES.format(glossary=GLOSSARY.relative_to(ROOT).as_posix(), terms=glossary()) + '-->\n\n'
    paths, batch, size = [], [], 0
    rows = needed(root, langs, flagged, redo, match)
    for row in [*rows, None]:
        if row is None or (batch and size + len(row[1]) * len(row[2]) > chars):
            if batch:
                path = work / f'batch-{len(paths) + 1:03d}.md'
                path.write_text(head + '\n\n'.join(batch) + '\n', encoding='utf-8')
                paths.append(path)
            batch, size = [], 0
        if row is not None:
            key, russian, wanted, is_interface = row
            notes = [f'<!-- interface string: keep each {{placeholder}} -->'] if is_interface else []
            notes += [f'<!-- previous {lang} answer rejected: {flags[lang][key]["last"]} -->' for lang in wanted
                      if flags.get(lang, {}).get(key, {}).get('last')]
            batch.append(f'@@ {key} {" ".join(wanted)}\n' + ''.join(n + '\n' for n in notes) + russian)
            size += len(russian) * len(wanted)
    return paths


def review(root, work=WORK, chars=8000, langs=LANGS, sample=None, seed=0):
    """Writes <work>/<lang>-review-NNN.md: stored translations nobody has reviewed yet (or a random `sample` of them),
    each with its Russian, for a stronger model, which answers <lang>-review-NNN.fix.md with the paragraphs it
    corrects. Returns the batch paths."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sources, paths = units(root), []
    for lang in langs:
        for old in work.glob(f'{lang}-review-[0-9][0-9][0-9].md'):
            old.unlink()
        cached = load_cache(root, lang).get('units', {})
        head = '<!--\n' + REVIEW.format(name=NAMES[lang], lang=lang, glossary=GLOSSARY.relative_to(ROOT).as_posix(),
                                        terms=glossary()) + '-->\n\n'
        items = [(k, sources[k], cached[k]['text']) for k in sources if k in cached and not cached[k].get('reviewed')]
        if sample is not None:
            items = random.Random(seed).sample(items, min(sample, len(items)))
        batch, size, number = [], 0, 1
        for item in [*items, None]:
            if item is None or (batch and size + len(item[1]) > chars):
                if batch:
                    path = work / f'{lang}-review-{number:03d}.md'
                    path.write_text(head + '\n\n'.join(batch) + '\n', encoding='utf-8')
                    paths.append(path)
                    number += 1
                batch, size = [], 0
            if item is not None:
                batch.append(f'@@ {item[0]} {lang}\n[ru] {item[1]}\n[{lang}] {item[2]}')
                size += len(item[1])
    return paths


def add_to_catalog(lang, entries):
    """Appends {Russian key: translation} to humanoid_sim/web/i18n/<lang>.json, keeping the file's layout."""
    import json
    path = i18n.CATALOGS / f'{lang}.json'
    text = path.read_text(encoding='utf-8').rstrip()
    assert text.endswith('}')
    lines = ''.join(f',\n  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}'
                    for k, v in entries.items())
    text = text[:-1].rstrip() + lines + '\n}\n'
    json.loads(text)
    path.write_text(text, encoding='utf-8', newline='\n')  # the catalogs are LF; text mode on Windows would write CRLF
    i18n.catalog.cache_clear()


def store(root, given, model, replace=False, reviewed=()):
    """Checks {(id, language): translation} against the current Russian and stores what passes: lab text into the
    cache, interface strings into the catalogs. Returns ({(id, language): problems} rejected, number stored,
    {(id, language): why} skipped). A stored translation that was reviewed is kept unless replace (a reviewed fix,
    which also marks it reviewed); reviewed: {language: ids} of review batches, marked reviewed unless their
    correction was rejected."""
    from .lab import edit_lock
    sources, strings = units(root), interface()
    rejected, stored, skipped = {}, 0, {}
    folder(root).mkdir(parents=True, exist_ok=True)
    with edit_lock(folder(root), 'lock'):
        flags = load_flags(root)
        for lang in LANGS:
            cache = load_cache(root, lang)
            cache.setdefault('units', {})
            mine, catalog = flags.setdefault(lang, {}), {}
            for (key, answer_lang), text in given.items():
                if answer_lang != lang:
                    continue
                if key in strings and lang in strings[key][1]:
                    russian = strings[key][0]
                elif key in sources:
                    russian = sources[key]
                    old = cache['units'].get(key)
                    if old and old.get('reviewed') and not replace:
                        skipped[(key, lang)] = 'a reviewed translation is kept'
                        continue
                    if old and old.get('text') == text.strip():
                        skipped[(key, lang)] = 'the same as the stored translation'
                        continue
                else:
                    skipped[(key, lang)] = 'the Russian changed or was translated since the batch was written'
                    continue
                found = problems(russian, text, lang)
                if found:
                    entry = mine.setdefault(key, {'ru': russian, 'attempts': 0})
                    if entry.get('answer') != text:  # the same answer read again is not another attempt
                        entry.update(attempts=entry['attempts'] + 1, last='; '.join(found), answer=text, at=time.time())
                    rejected[(key, lang)] = found
                    continue
                mine.pop(key, None)
                stored += 1
                if key in strings:
                    catalog[russian] = text.strip()
                else:
                    cache['units'][key] = {'ru': russian, 'text': text.strip(), 'at': time.time(), 'by': model,
                                           **({'reviewed': True} if replace else {})}
            for key in (reviewed or {}).get(lang, ()):
                unit = cache['units'].get(key)
                if unit and (key, lang) not in rejected and unit.get('ru') == sources.get(key):
                    unit['reviewed'] = True
            write_json(folder(root) / f'{lang}.json', cache)
            if catalog:
                add_to_catalog(lang, catalog)
        write_json(folder(root) / 'flags.json', flags)
    return rejected, stored, skipped


def submit(root, paths, model='claude-haiku-5-5', reviewer='review'):
    """Stores the answers of the given files (*.fix.md: reviewed corrections, which replace stored text and mark
    their review batch reviewed). Returns ({(id, language): problems}, number stored, {(id, language): why skipped})."""
    rejected, stored, skipped = {}, 0, {}
    for path in map(Path, paths):
        fix = path.name.endswith('.fix.md')
        reviewed = {}
        batch = path.with_name(path.name[:-len('.fix.md')] + '.md') if fix else None
        if batch and batch.is_file():
            default = batch.name[:2] if batch.name[:3] in ('en-', 'zh-') else None
            for key, langs, _ in sections(batch.read_text(encoding='utf-8'), default):
                for lang in langs:
                    reviewed.setdefault(lang, set()).add(key)
        bad, count, same = store(root, answers(path), reviewer if fix else model, fix, reviewed)
        rejected.update(bad)
        skipped.update(same)
        stored += count
    return rejected, stored, skipped


def import_answers(root, work=WORK, model='claude-haiku-5-5', reviewer='review'):
    """submit() for every *.answer.md, then every *.fix.md in the work folder."""
    work = Path(work)
    return submit(root, sorted(work.glob('*.answer.md')) + sorted(work.glob('*.fix.md')), model, reviewer)


def status(root):
    all_units, flags, strings = units(root), load_flags(root), interface()
    rows = {}
    for lang in LANGS:
        cached = load_cache(root, lang).get('units', {})
        done = [key for key in all_units if key in cached]
        flagged = [key for key in all_units if key not in cached
                   and flags.get(lang, {}).get(key, {}).get('attempts', 0) >= MAX_ATTEMPTS]
        rows[lang] = dict(units=len(all_units), translated=len(done), missing=len(all_units) - len(done),
                          missing_chars=sum(len(all_units[k]) for k in all_units if k not in cached),
                          flagged=flagged, reviewed=sum(1 for k in done if cached[k].get('reviewed')),
                          interface=sum(1 for _, langs in strings.values() if lang in langs))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('status', 'export', 'submit', 'import', 'review'))
    parser.add_argument('files', nargs='*', type=Path, help='submit: answer files')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--chars', type=int, default=8000, help='Russian characters per batch (times languages)')
    parser.add_argument('--lang', choices=LANGS, nargs='+', default=list(LANGS))
    parser.add_argument('--flagged', action='store_true', help='export: flagged paragraphs too')
    parser.add_argument('--redo', choices=LANGS, nargs='+', default=[],
                        help='export: also stored, unreviewed translations of these languages, to translate again')
    parser.add_argument('--match', help='export: only texts whose Russian matches this regular expression')
    parser.add_argument('--sample', type=int, help='review: a random sample of this many paragraphs')
    parser.add_argument('--model', default='claude-haiku-5-5', help='recorded with each stored translation')
    parser.add_argument('--reviewer', default='review', help='recorded with each stored review correction')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = lab_root(args.run_dir)
    rejected = {}
    if args.action == 'export':
        for path in export(root, args.work, args.chars, args.lang, args.flagged, args.redo, args.match):
            print(path)
    elif args.action == 'review':
        for path in review(root, args.work, args.chars, args.lang, args.sample):
            print(path)
    elif args.action in ('submit', 'import'):
        if args.action == 'submit':
            rejected, stored, skipped = submit(root, args.files, args.model, args.reviewer)
        else:
            rejected, stored, skipped = import_answers(root, args.work, args.model, args.reviewer)
        print(f'stored {stored}, rejected {len(rejected)}, skipped {len(skipped)} (nothing to fix there)')
        for (key, lang), found in sorted(rejected.items()):
            print(f'  @@ {key} {lang}: {"; ".join(found)}')
        for (key, lang), why in sorted(skipped.items()):
            print(f'  skipped @@ {key} {lang}: {why}')
    for lang, row in status(root).items():
        print(f"{lang}: {row['translated']}/{row['units']} paragraphs translated ({row['reviewed']} reviewed), "
              f"{row['missing']} missing ({row['missing_chars']} characters), {len(row['flagged'])} flagged; "
              f"{row['interface']} interface strings missing")
    sys.exit(1 if rejected else 0)


if __name__ == '__main__':
    main()
