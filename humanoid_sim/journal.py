"""Lab journal («Дневник»): the roadmap, the latest goal check and blog-style entries about the research.

The research supervisor writes it at decisions; the browser tab only reads it. Everything lives in
<lab>/journal/: journal.json plus the images entries show (copied in, so the journal is self-contained).

    {"roadmap": [{"title", "status": done|active|next, "detail",
                  "story": {"sections": [{"title", "text", "figure"?, "clip"?, "poster"?}],
                            "path": [{"title", "change", "result", "verdict", "race", "clip"?, "poster"?}],
                            "races": [{"title", "text", ...race}]} | null}],
     "check": {"at", "branch", "stage", "title", "items": [{"label", "value", "target", "ok"}]} | null,
     "skills": {"at", "branch", "stage", "title", "against": {"branch", "stage", "title"}, "seconds", "runs",
                "items": [{"skill", "label", "target", "value", "reference", "ok", "note"}]} (absent until the first),
     "entries": [{"id", "at", "title", "text", "image" | null,
                  "race": {"robot", "task", "scenario", "lanes": [{"experiment", "stage"}]} | null}],
     "iterations": [{"id", "at", "title", "question", "messages": [{"role", "text"}], "decision" | null}]}
"""
import re
import shutil
import time
from pathlib import Path

from .core import read_json, write_json

ROADMAP_STATUS = ('done', 'active', 'next')
IMAGE_NAME = re.compile(r'^[A-Za-z0-9_-]{1,80}\.(png|jpg|webp)$')  # webp: animated clips (humanoid_sim.clip)
CONTENT_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp'}
# Interactive explanatory figures (humanoid_sim/figures, anatomy kit): standalone pages the tab shows sandboxed.
FIGURE_NAME = re.compile(r'^[A-Za-z0-9_-]{1,80}\.html$')
ENTRY_ID = re.compile(r'^[a-z0-9-]{1,80}$')
EMPTY = {'roadmap': [], 'check': None, 'entries': [], 'iterations': []}
# Who speaks in «Совет лаборатории»: the lead (Claude) and the lab's agents (.claude/agents/lab-*.md).
ROLES = {'lead': 'Руководитель', 'critic': 'Критик', 'scout': 'Разведчик', 'analyst': 'Аналитик',
         'watch': 'Дежурный', 'chronicler': 'Летописец', 'engineer': 'Инженер', 'reviewer': 'Ревизор'}


def folder(root):
    return Path(root) / 'journal'


def load(root):
    return {**EMPTY, **read_json(folder(root) / 'journal.json', {})}


def _edit(root, change):
    from .lab import edit_lock
    folder(root).mkdir(parents=True, exist_ok=True)
    with edit_lock(root, 'journal-edit'):
        data = load(root)
        change(data)
        write_json(folder(root) / 'journal.json', data)
        return data


def _text(value, what, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{what}: непустой текст до {limit} символов')
    return value.strip()


def _race(race):
    """{robot, task, scenario, lanes: [{experiment, stage: steps | 'latest'}]} for the race control API."""
    lanes = race.get('lanes') if isinstance(race, dict) else None
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 10 or not all(
            isinstance(l, dict) and isinstance(l.get('experiment'), str)
            and (type(l.get('stage')) is int or l.get('stage') == 'latest') for l in lanes):
        raise ValueError('Забег: от 1 до 10 дорожек {experiment, stage}')
    return {'robot': race['robot'], 'task': race['task'], 'scenario': race.get('scenario'),
            'lanes': [{'experiment': l['experiment'], 'stage': l['stage']} for l in lanes]}


def add_media(root, source, name):
    """Copies an image or clip into the journal folder under `name`; returns the name for entries to use."""
    if not isinstance(name, str) or not IMAGE_NAME.match(name) or not Path(source).is_file():
        raise ValueError('Файл дневника: PNG, JPG или WebP')
    folder(root).mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() != (folder(root) / name).resolve():
        shutil.copyfile(source, folder(root) / name)
    return name


def _clip(item):
    """{clip, poster} file names of a pre-rendered clip (see humanoid_sim.clip), or nothing."""
    names = {key: item.get(key) for key in ('clip', 'poster') if item.get(key)}
    if any(not IMAGE_NAME.match(str(name)) for name in names.values()):
        raise ValueError('Ролик: имя файла WebP/JPG')
    return names


def _figure(item):
    """{figure} — the file name of an interactive figure page in the journal folder (add_figure), or nothing."""
    name = item.get('figure')
    if not name:
        return {}
    if not FIGURE_NAME.match(str(name)):
        raise ValueError('Схема: имя файла .html')
    return {'figure': name}


def _story(story):
    """A roadmap stage told in detail: text sections (each with an optional figure and clip), the path through
    branches, before/after races. Clips and posters are file names already in the journal folder (add_media)."""
    if story is None:
        return None
    if not isinstance(story, dict):
        raise ValueError('Подробности этапа: объект')
    sections = [{'title': _text(s.get('title'), 'Раздел', 200), 'text': _text(s.get('text'), 'Текст раздела', 6000),
                 **_figure(s), **_clip(s)} for s in story.get('sections', [])]
    path = [{'title': _text(p.get('title'), 'Шаг пути', 200), 'change': _text(p.get('change'), 'Что меняли', 1000),
             'result': _text(p.get('result'), 'Что вышло', 1000), 'verdict': p.get('verdict'),
             'race': _race(p['race']) if p.get('race') else None, **_clip(p)} for p in story.get('path', [])]
    races = [{'title': _text(r.get('title'), 'Забег', 200), 'text': str(r.get('text') or ''), **_race(r), **_clip(r)}
             for r in story.get('races', [])]
    return {'sections': sections, 'path': path, 'races': races}


def set_roadmap(root, stages):
    if not isinstance(stages, list) or not 1 <= len(stages) <= 20:
        raise ValueError('План: от 1 до 20 этапов')
    clean = []
    for stage in stages:
        if not isinstance(stage, dict) or stage.get('status') not in ROADMAP_STATUS:
            raise ValueError('Этап плана: статус done, active или next')
        clean.append({'title': _text(stage.get('title'), 'Этап плана', 200), 'status': stage['status'],
                      'detail': _text(stage.get('detail') or '—', 'Описание этапа', 1000),
                      'story': _story(stage.get('story'))})
    return _edit(root, lambda data: data.update(roadmap=clean))


def set_story(root, index, story):
    """Replaces the detailed story of one roadmap stage (0-based), keeping the rest of the plan."""
    clean = _story(story)

    def change(data):
        if not 0 <= index < len(data['roadmap']):
            raise ValueError('Нет такого этапа плана')
        data['roadmap'][index]['story'] = clean
    return _edit(root, change)


def set_check(root, branch, stage, title, items, at=None):
    """items: [{label, value, target, ok: bool | None}] — one row per goal condition."""
    if not isinstance(items, list) or not items:
        raise ValueError('Проверка: нужен хотя бы один пункт')
    rows = []
    for item in items:
        if not isinstance(item, dict) or item.get('ok') not in (True, False, None):
            raise ValueError('Пункт проверки: ok — true, false или null')
        rows.append({'label': _text(item.get('label'), 'Пункт проверки', 200),
                     'value': str(item.get('value', '')), 'target': str(item.get('target', '')), 'ok': item['ok']})
    check = {'at': time.time() if at is None else float(at), 'branch': branch, 'stage': int(stage),
             'title': _text(title, 'Проверка', 300), 'items': rows}
    return _edit(root, lambda data: data.update(check=check))


def set_skills(root, branch, stage, title, against, items, seconds=None, runs=None, at=None):
    """The latest skills check (scripts/lab/skills_check.py), shown beside the goal check: against {branch, stage,
    title}, the reference; items: [{skill, label, target, value, reference, ok: bool | None (н/п), note}] — one row
    per measure; seconds, runs: how long it took and how many runs. Stored under 'skills' (absent until the first)."""
    if not isinstance(items, list) or not items:
        raise ValueError('Проверка навыков: нужен хотя бы один пункт')
    if not isinstance(against, dict):
        raise ValueError('Проверка навыков: эталон {branch, stage, title}')
    rows = []
    for item in items:
        if not isinstance(item, dict) or item.get('ok') not in (True, False, None):
            raise ValueError('Пункт проверки: ok — true, false или null')
        rows.append({'skill': _text(item.get('skill'), 'Навык', 200), 'label': _text(item.get('label'), 'Мера', 300),
                     **{key: str(item.get(key, '')) for key in ('target', 'value', 'reference', 'note')},
                     'ok': item['ok']})
    skills = {'at': time.time() if at is None else float(at), 'branch': branch, 'stage': int(stage),
              'title': _text(title, 'Проверка', 300),
              'against': {'branch': str(against.get('branch', '')), 'stage': int(against.get('stage') or 0),
                          'title': _text(against.get('title'), 'Эталон', 300)},
              'seconds': None if seconds is None else float(seconds), 'runs': None if runs is None else int(runs),
              'items': rows}
    return _edit(root, lambda data: data.update(skills=skills))


def add_entry(root, entry_id, title, text, image=None, race=None, at=None, clip=None):
    """Adds or replaces (same id) an entry. image: a PNG/JPG path; clip: a .webp from humanoid_sim.clip with its
    «-poster.jpg» beside it. Both are copied into the journal folder."""
    if not isinstance(entry_id, str) or not ENTRY_ID.match(entry_id):
        raise ValueError('Запись: id из латинских букв, цифр и дефисов')
    name = None if image is None else add_media(root, image, f'{entry_id}{Path(image).suffix.lower()}')
    media = {}
    if clip is not None:
        clip = Path(clip)
        media = {'clip': add_media(root, clip, f'{entry_id}-clip.webp'),
                 'poster': add_media(root, clip.with_name(clip.stem + '-poster.jpg'), f'{entry_id}-clip.jpg')}
    if race is not None:
        race = _race(race)
    entry = {'id': entry_id, 'at': time.time() if at is None else float(at), 'title': _text(title, 'Заголовок', 200),
             'text': _text(text, 'Текст записи', 6000), 'image': name, 'race': race, **media}

    def change(data):
        data['entries'] = sorted([e for e in data['entries'] if e['id'] != entry_id] + [entry],
                                 key=lambda e: e['at'], reverse=True)
    return _edit(root, change)


def add_iteration(root, iteration_id, title, question, messages, decision=None, at=None):
    """Adds or replaces (same id) one iteration of the lab council: the question, who said what (short), and the
    decision taken, or None while it is still being discussed. messages: [{"role": one of ROLES, "text"}]."""
    if not isinstance(iteration_id, str) or not ENTRY_ID.match(iteration_id):
        raise ValueError('Итерация: id из латинских букв, цифр и дефисов')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 12:
        raise ValueError('Итерация: от 1 до 12 реплик')
    said = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ROLES:
            raise ValueError(f'Реплика: роль из {", ".join(ROLES)}')
        said.append({'role': message['role'], 'text': _text(message.get('text'), 'Реплика', 800)})
    item = {'id': iteration_id, 'at': time.time() if at is None else float(at), 'title': _text(title, 'Итерация', 200),
            'question': _text(question, 'Вопрос итерации', 800), 'messages': said,
            'decision': None if decision is None else _text(decision, 'Решение', 1000)}

    def change(data):
        data['iterations'] = sorted([i for i in data.get('iterations', []) if i['id'] != iteration_id] + [item],
                                    key=lambda i: i['at'], reverse=True)
    return _edit(root, change)


def image_file(root, name):
    """Path of an entry image, or None for anything that is not one."""
    if not isinstance(name, str) or not IMAGE_NAME.match(name):
        return None
    path = folder(root) / name
    return path if path.is_file() else None


def add_figure(root, source, name):
    """Copies a built figure page (humanoid_sim/figures/out/*.html) into the journal folder under `name`."""
    if not isinstance(name, str) or not FIGURE_NAME.match(name) or not Path(source).is_file():
        raise ValueError('Схема дневника: страница .html')
    folder(root).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder(root) / name)
    return name


def figure_file(root, name):
    """Path of a journal figure page, or None for anything that is not one."""
    if not isinstance(name, str) or not FIGURE_NAME.match(name):
        return None
    path = folder(root) / name
    return path if path.is_file() else None
