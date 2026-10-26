"""The home page's data (CPU only; reads only, except the clips it renders when one is missing): writes
<lab>/journal/home.json (humanoid_sim.home) for site.html — the «updated» line, the roadmap stage, the number cards
and «what the robot can do» for the walking and the getting-up candidate — from numbers the lab already has.

Usage: .venv\\Scripts\\python.exe scripts\\site\\home_update.py [--package DIR] [--walk-skills JSON]
       [--getup-skills JSON] [--no-clips] [--redo-clips] [--adopt-clips] [--workers 4] [--dry-run] [--lab DIR]

Sources, nothing measured again:
  the robot package    (--package, default the newest runs/humanoid_lab/robot/* whose main candidate is «основной»):
                       scripts/robot/passport.py's collect() — the bridge (unitree_mujoco + g1_ctrl) numbers of the
                       walking candidate from REFERENCE.md, the packaged getting-up stage's, goal_check_<steps>.json and
                       chatter_<steps>.json beside it; package.py's check_licence() refuses a policy trained on LAFAN1
  skills_check --json  of the walking stage (--walk-skills, default <lab>/journal/home/skills_<steps>.json) and of the
                       getting-up stage (--getup-skills, default the same name, else the journal's «skills» card when
                       it is that stage's)
  the journal          its roadmap (the stages of docs/g1-roadmap.md as the journal shows them)
A number its source lacks stays empty with the reason. A row is a weak spot when skills_check says its threshold is not
met («порог не взят») or Unitree v0 does better, and for a bridge number when it misses the threshold of the same
measure in skills_check / goal_check (named in WEAK_LIMITS).

Clips: site-<lang>-<name>.webp (+ -poster.jpg) in the journal folder, rendered by humanoid_sim.clip from the packaged
stages when missing or showing another stage than the current candidate (<lab>/journal/home/clips.json records which);
--adopt-clips records existing files without an entry as the current candidate's (once, for clips rendered before this
script by .tools/site/render_skills.py from the same stage). The candidates' clips show the packaged stages only, so
no LAFAN1 policy gets there; the one exception is RESEARCH, a separate block after the candidates («research» in
home.json: the robot's clip and its text only, numbers checked against their journal section).

Daily (9:00): home_update, then the translation of new paragraphs (journal_translate export / lab-translator /
submit), then scripts/publish.py site.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / 'scripts' / 'robot'))
import passport as pp  # noqa: E402
from humanoid_sim import home, journal  # noqa: E402
from humanoid_sim.core import read_json, write_json  # noqa: E402

LAB = SIM / 'runs' / 'humanoid_lab'
SIZE = (640, 480)
# Walking clips: name -> (humanoid_sim.clip scenario, seconds, view), as .tools/site/render_skills.py made them.
WALK_CLIPS = {'walk': ('walk', 8., 'side'), 'slow_front': ('slow', 8., 'front'), 'back': ('back', 8., 'side'),
              'side_left': ('side_left', 8., 'front'), 'turn_left': ('turn_left', 8., 'front'),
              'walk_turn': ('walk_turn', 10., 'side'), 'stop_start': ('stop_start', 15., 'side'),
              'side_stop': ('side_stop', 8., 'front'), 'push_back': ('push_back', 6., 'side'),
              'push_left': ('push_left', 6., 'front'), 'push_walk': ('push_walk', 8., 'side'),
              'sprint': ('sprint', 8., 'side'), 'arm_pose_stand': ('arm_pose_stand', 10., 'front')}
GETUP_CLIPS = {'getup_back': ('getup_back', 8., None), 'getup_front': ('getup_front', 8., None),
               'getup_side': ('getup_side', 8., None)}
# Tile titles baked into a clip per language (render_skills.py's); the getting-up ones need none (GetUp <stage>).
TITLES = {
    'en': {'walk': 'Walk 1 m/s', 'slow_front': 'Walk 0.5 m/s, front', 'back': 'Backward 0.5 m/s',
           'side_left': 'Side step left 0.3 m/s', 'turn_left': 'Turn in place 0.5 rad/s',
           'walk_turn': 'Walk and turn', 'stop_start': 'Stand, walk, stop', 'side_stop': 'Side step, then stop',
           'push_back': 'Standing, pushed', 'push_left': 'Standing, pushed', 'push_walk': 'Walking, pushed',
           'sprint': 'Run, command 2 m/s'},
    'zh': {'walk': '行走 1 m/s', 'slow_front': '行走 0.5 m/s，正面', 'back': '后退 0.5 m/s',
           'side_left': '向左侧移 0.3 m/s', 'turn_left': '原地转向 0.5 rad/s', 'walk_turn': '边走边转',
           'stop_start': '站立、行走、停下', 'side_stop': '侧移后停下', 'push_back': '站立受推', 'push_left': '站立受推',
           'push_walk': '行走中受推', 'sprint': '跑步，指令 2 m/s'},
    'ru': {'walk': 'Ходьба 1 м/с', 'slow_front': 'Ходьба 0,5 м/с, спереди', 'back': 'Назад 0,5 м/с',
           'side_left': 'Шаг вбок влево 0,3 м/с', 'turn_left': 'Поворот на месте 0,5 рад/с',
           'walk_turn': 'Идёт и поворачивает', 'stop_start': 'Стоит, идёт, стоп', 'side_stop': 'Вбок, затем стоп',
           'push_back': 'Стоит, толчок', 'push_left': 'Стоит, толчок', 'push_walk': 'Идёт, толчок',
           'sprint': 'Бег, команда 2 м/с'},
}
# A bridge number's weak-spot threshold: the same measure's in skills_check (row key) or goal_check (item label).
WEAK_LIMITS = dict(speed='speed:1', drift='Увод без команды (0,5 и 1 м/с, старт с любой ноги)',
                   heading='Курс при ходьбе', side_drift='side:03', brake05='brake:0.5', brake1='brake:1',
                   side_stop='side:stop', turn='turn:place', turn_walk='turn:walk')
# The lab's own words in skills_check's labels, as the public page says them (the page is for people outside the lab).
SPEED_RUNS = len(ast.literal_eval(re.search(r'^SPEED_SEEDS = (.+)$', (SIM / 'scripts' / 'lab' / 'skills_check.py')
                                            .read_text(encoding='utf-8'), re.M)[1]))  # «худшее зерно» of how many
PUBLIC = (('копия связки с капсулами стоп', 'физика Unitree, со столкновением стоп'),
          ('копия связки', 'физика Unitree, со столкновением стоп'), ('худшее зерно', f'худший из {SPEED_RUNS} прогонов'),
          ('Наша физика', 'Наша симуляция'), ('наша физика', 'наша симуляция'), ('все физики', 'все варианты физики'),
          (' (getup_probe)', ''))
NO_BRIDGE = 'нет числа в REFERENCE.md пакета'  # passport's reason for a bridge field without a number
PHYSICS = dict(ours='наша симуляция', delay='задержка 20 мс', foot='трение стоп 0,3', floor='скользкий пол',
               noise='шум датчиков')


def fail(message):
    raise SystemExit(f'home_update.py: {message}')


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def rel(path):
    return pp.rel(path)


def ru(x, digits, sign=False):
    return pp.number(x, digits, 'ru', sign)


def span(pair):
    return pp.span(pair, 'ru')


def show(value):
    return pp.show(value, 'ru')


def row(label, value=None, unit=None, where=None, weak=False, why=None, v0=None, target=None):
    out = {'label': label, 'value': value}
    if value is None:
        out['why'] = why or NO_BRIDGE
    else:
        out.update({key: item for key, item in (('unit', unit), ('v0', v0), ('target', target)) if item})
        if weak:
            out['weak'] = True
    if where:
        out['where'] = where
    return out


# --- sources ------------------------------------------------------------------------------------------------------


def newest_package(lab):
    """The newest robot package whose main candidate is the main one («основной»)."""
    found = []
    for manifest in (lab / 'robot').glob('*/package.json'):
        info = read_json(manifest, {}) or {}
        if info.get('kind') == pp.KIND and (info.get('candidates') or [{}])[0].get('role') == 'основной':
            found.append((str(info.get('created') or ''), manifest.parent))
    if not found:
        fail(f'нет пакета робота с основным кандидатом в {lab / "robot"}')
    return max(found)[1]


def licence(package, slot):
    """package.py's LAFAN1 check of a packaged policy (stops the script on a LAFAN1 one)."""
    import package as pk
    pk.check_licence(package / 'policy' / slot)


def limit(target):
    """('≤', 0.25) of a threshold text such as «≤ 0,25» / «< 10°» / «≥ 0,9», or None."""
    m = re.search(r'([<≤≥>])\s*([+−-]?\d+(?:,\d+)?)', target or '')
    return (m[1], float(m[2].replace(',', '.').replace('−', '-'))) if m else None


def misses(value, threshold):
    """Whether value misses a (sign, number) threshold; None without one."""
    if threshold is None or value is None:
        return False
    sign, bound = threshold
    return {'<': value >= bound, '≤': value > bound, '>': value <= bound, '≥': value < bound}[sign]


def skills_file(path, steps):
    """skills_check --json of the stage `steps` (its rows), or None if the file is missing or of another stage."""
    if path is None or not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if str(steps) not in str((payload.get('stage') or {}).get('label') or ''):
        fail(f'{path}: проверка навыков этапа {(payload.get("stage") or {}).get("label")}, а не {steps}')
    return payload


def skill_rows(rows):
    """{key or label: row} of skills_check rows that ran (н/п rows left out)."""
    return {r.get('key') or r['label']: r for r in rows if r.get('ok') is not None}


def public(label):
    """A skills_check label in the public page's words (PUBLIC)."""
    for old, new in PUBLIC:
        label = label.replace(old, new)
    return label


def skill_weak(r):
    return 'порог не взят' in (r.get('note') or '') or r.get('v0_vs') == 'worse'


def from_skills(r, label=None):
    """A home row of a skills_check row: its value, threshold and Unitree v0's, weak by skill_weak."""
    v0 = r.get('v0') if r.get('v0') and not str(r.get('v0')).startswith('н/п') else None
    where = 'bridge_copy' if 'копия связки' in r['label'] else 'skills'
    return row(label or public(r['label']), r['value'], where=where, weak=skill_weak(r), v0=v0,
               target=None if r.get('target') == 'не судится' else r.get('target'))


def counts(rows):
    judged = [r for r in rows if r.get('ok') is not None]
    v0 = [r.get('v0_vs') for r in rows if r.get('v0_vs')]
    return dict(ok=sum(r['ok'] is True for r in judged), judged=len(judged),
                na=sum(r.get('ok') is None for r in rows), better=v0.count('better'), same=v0.count('same'),
                worse=v0.count('worse'), v0=bool(v0))


# --- the walking candidate ----------------------------------------------------------------------------------------


def walking(data, skills, missing):
    """The walking candidate's summary and skill cards."""
    c, b = data['candidate'], data['bridge']
    get = (lambda key: b.get(key) if b.get('found') else None)
    sk = skill_rows(skills['rows']) if skills else {}
    goal = (data['goal_file'] or {}).get('items') if data['goal_file'] and not data['goal_file']['mismatch'] else (
        (data['goal'] or {}).get('items'))
    goal_items = {item['label']: item for item in goal or []}
    no_skills = ('нет проверки навыков этого этапа (scripts/lab/skills_check.py --json → '
                 f'runs/humanoid_lab/journal/home/skills_{c["steps"]}.json)')
    if not skills:
        missing.append(dict(what=f'Проверка навыков кандидата {c["title"]}', why=no_skills))

    def threshold(name):
        key = WEAK_LIMITS[name]
        source = sk.get(key) or goal_items.get(key)
        return limit(source['target']) if source else None

    def skill(key, label=None):
        """The skills_check row `key`, or None (left out: without the file the summary says why; a measure the check
        did not run, н/п, is no number of this candidate)."""
        return from_skills(sk[key], label) if key in sk else None

    def walk_rows(w, speed):
        if not w:
            return [row(f'Скорость при команде {ru(speed, 1 if speed < 1 else 0)} м/с', where='bridge')]
        ratio = threshold('speed')
        slow = ratio and misses(w['speed'][0].value / speed, ratio)
        drift_limit, heading_limit = threshold('drift'), threshold('heading')
        worst = max(abs(x.value) for x in w['drift'])
        turned = max(abs(x.value) for x in w['heading'])
        label = ru(speed, 1 if speed < 1 else 0)
        return [row(f'Скорость при команде {label} м/с', span(w['speed']), 'm/s', 'bridge', slow),
                row(f'Увод курса без команды поворота, {label} м/с', span(w['drift']), 'rad/s', 'bridge',
                    misses(worst, drift_limit)),
                row(f'Курс за 10 с ходьбы {label} м/с', span(w['heading']) + '°', None, 'bridge',
                    misses(turned, heading_limit))]

    w05, w1 = get('walk05'), get('walk1')
    turn, brake, stop = get('turn'), get('brake'), get('side_stop')
    side = {round(x['speed'].value, 2): x for x in get('side') or []}
    pushes = get('pushes') or []
    tilt = get('push_tilt')
    ranges = c['ranges'] or {}
    top = max(ranges.get('lin_vel_x') or [0])

    def turns(where_key, label, pair):
        share = threshold(where_key)
        rate = turn['rate'].value
        worst = min(abs(x.value) for x in pair) / rate
        return row(label, f'{show(pair[0])} / {show(pair[1])}', 'rad/s', 'bridge', misses(worst, share))

    def brakes():
        if not brake:
            return [row('Торможение после стопа', where='bridge')]
        out = []
        for speed, distance in brake['rows']:
            key = 'brake05' if abs(speed.value - .5) < 1e-9 else 'brake1' if abs(speed.value - 1) < 1e-9 else None
            out.append(row(f'Торможение: путь за {brake["seconds"]} с после стопа с {show(speed)} м/с', span(distance),
                           'm', 'bridge', key is not None and misses(distance[1].value, threshold(key))))
        return out

    def sides():
        out = []
        for speed in (0.3, 0.5):
            x = side.get(speed)
            label = ru(speed, 1)
            if not x:
                out.append(row(f'Шаг вбок ±{label} м/с: скорость', where='bridge'))
                continue
            worst = max(abs(d.value) for d in x['drift'])
            out += [row(f'Шаг вбок ±{label} м/с: скорость', span(x['speeds']), 'm/s', 'bridge'),
                    row(f'Шаг вбок ±{label} м/с: увод', span(x['drift']), 'rad/s', 'bridge',
                        misses(worst, threshold('side_drift')))]
        return out

    def push_rows(state):
        mine = [p for p in pushes if p['state'] == state and not p.get('missing')]
        if not mine:
            return [row('Толчки стоя, физика Unitree: падений' if state == 'stand' else 'Толчки на ходу, физика Unitree: падений',
                        where='bridge')]
        where = 'стоя' if state == 'stand' else f'на ходу {show(mine[0]["speed"])} м/с'
        return [row(f'Толчок {where}, Δv {show(p["dv"])} м/с ({ru(p["force"], 0)} Н за {pp.short(p["seconds"], "ru")} с)'
                    ': падений', pp.group(p['falls'][0], 'ru') + ' из ' + pp.group(p['falls'][1], 'ru'), None, 'bridge',
                    p['falls'][0] > 0) for p in mine]

    cards = [
        dict(title=f'Ходьба {ru(1, 0)} м/с', clip='walk', alt='Идёт со скоростью 1 м/с, вид сбоку',
             rows=walk_rows(w1, 1.) + [skill('speed:1'), skill('goal:1')]),
        dict(title=f'Ходьба {ru(.5, 1)} м/с', clip='slow_front', alt='Идёт со скоростью 0,5 м/с, вид спереди',
             rows=walk_rows(w05, .5) + [
                 row('Снос вбок за 10 с ходьбы 0,5 м/с', span(get('side10')), 'm', 'bridge') if get('side10') else
                 row('Снос вбок за 10 с ходьбы 0,5 м/с', where='bridge'), skill('speed:0.5')]),
        dict(title=f'Назад {ru(.5, 1)} м/с', clip='back', alt='Идёт назад', rows=[
            skill('speed:-0.5'), row('Назад 0,5 м/с в физике Unitree', where='bridge')]),
        dict(title='Шаг вбок', clip='side_left', alt='Шаг вбок влево',
             rows=sides() + [skill('side:03'), skill('side:05'), skill('side:1'), skill('side:1drift'),
                             skill('feet:1')]),
        dict(title='Поворот на месте', clip='turn_left', alt='Поворачивается на месте', rows=[
            turns('turn', f'Поворот по команде ±{show(turn["rate"])} рад/с: на месте', turn['place']) if turn else
            row('Поворот по команде: на месте', where='bridge'), skill('turn:place'), skill('goal:6')]),
        dict(title='Ходьба с поворотом', clip='walk_turn', alt='Идёт и поворачивает', rows=[
            turns('turn_walk', f'Поворот по команде ±{show(turn["rate"])} рад/с: на ходу {show(turn["speed"])} м/с',
                  turn['walking']) if turn else row('Поворот по команде: на ходу', where='bridge'),
            skill('turn:walk')]),
        dict(title='Стоять, идти, остановиться', clip='stop_start', alt='Стоит, идёт, останавливается',
             rows=brakes() + [skill('brake:0.5'), skill('brake:1'), skill('goal:0'), skill('goal:3')]),
        dict(title='Шаг вбок, затем стоп', clip='side_stop', alt='Шаг вбок, затем остановка', rows=[
            row(f'Стоп после шага вбок ±{show(stop["rows"][0][0])} / ±{show(stop["rows"][1][0])} м/с: сдвиг за '
                f'{stop["seconds"]} с', f'{span(stop["rows"][0][1])} / {span(stop["rows"][1][1])}', 'm', 'bridge',
                misses(max(r[1][1].value for r in stop['rows']), threshold('side_stop'))) if stop else
            row('Стоп после шага вбок: сдвиг', where='bridge'), skill('side:stop')]),
        dict(title='Толчки стоя', clip='push_back', alt='Толчок, стоя', rows=push_rows('stand') + [
            row('Наибольший наклон корпуса при толчках стоя',
                f'{show(tilt["deg"])}° при Δv {"–".join(show(x) for x in tilt["dv"])}', None, 'bridge') if tilt else
            row('Наибольший наклон корпуса при толчках стоя', where='bridge')]),
        dict(title='Толчки с восьми сторон', clip='push_left', alt='Толчок вбок, стоя', rows=[skill('push:stand')]),
        dict(title='Толчок на ходу', clip='push_walk', alt='Толчок на ходу',
             rows=push_rows('walk') + [skill('push:walk')]),
        dict(title=f'Бег, команда {ru(2, 0)} м/с', clip='sprint', alt='Бег по команде 2 м/с', rows=[
            skill('speed:2'),
            row('Бег в физике Unitree', why=f'команды deploy.yaml вперёд до {pp.short(top, "ru")} м/с: в физике Unitree '
                'не проверить',
                where='package') if top < 2 else row('Бег в физике Unitree', where='bridge')]),
        dict(title='Руки в случайной позе', clip='arm_pose_stand',
             alt='Стоит, руки в случайной позе до ±1 рад',
             rows=[skill('armp:falls'), skill('armp:turn'), skill('armp:creep'), skill('armp:drift')]),
        dict(title='Моторы: дрожание моментов', rows=[
            skill('chatter:torque_hf_rms_nm'), skill('chatter:action_hf_rms_rad'), skill('chatter:stopstart'),
            skill('hoist')]),
    ]
    for card in cards:
        card['rows'] = [r for r in card['rows'] if r]
    cards = [card for card in cards if card['rows']]  # a skill without any number of its own is not shown

    summary = []
    if skills:
        n = counts(skills['rows'])
        summary.append(row('Проверка навыков: выполнено', f'{n["ok"]} из {n["judged"]}', where='skills'))
        if n['v0']:
            summary.append(row('Против Unitree v0: лучше / так же / хуже', f'{n["better"]} / {n["same"]} / {n["worse"]}',
                               where='skills'))
    else:
        summary.append(row('Проверка навыков: выполнено', why=no_skills, where='skills'))
    if goal:
        ok = sum(item.get('ok') is True for item in goal)
        summary.append(row('Условия цели ходьбы: выполнено', f'{ok} из {len(goal)}', where='goal',
                           weak=ok < len(goal)))
    else:
        summary.append(row('Условия цели ходьбы: выполнено', where='goal',
                           why='нет goal_check этого этапа в пакете и в «Дневнике»'))
    runs = get('runs')
    summary.append(row('Прогонов в физике Unitree; падений', f'{runs[0]}; {runs[1]}', where='bridge', weak=runs[1] > 0)
                   if runs else row('Прогонов в физике Unitree; падений', where='bridge'))
    about = (f'Ветка {c["branch"]}, этап {c["steps"]} шагов ({ru(c["steps"] / 1e6, 1)} млн), выгрузка {c["export"]}; '
             f'пакет для робота {data["package"]}.')
    return dict(id='walk', title=f'Ходьба: кандидат {c["title"]}', about=about, summary=summary, skills=cards)


# --- the getting-up candidate -------------------------------------------------------------------------------------


GETUP_GROUPS = (('Вставание: готовность к роботу', 'Готовность к роботу'),
                ('Вставание: стойка после подъёма', 'Стойка после подъёма'),
                ('Вставание: передача в ходьбу №8', 'Передача в ходьбу'), ('Вставание: нагрузки', 'Нагрузки'))
GETUP_STARTS = (('getup_back', 'Встаёт со спины', 'Встаёт, лёжа на спине', 'Старт на спине, все физики'),
                ('getup_front', 'Встаёт с живота', 'Встаёт, лёжа на животе',
                 'Старт на животе, все физики', 'Старт сидя, все физики', 'Старт из падения №8, все физики'),
                ('getup_side', 'Встаёт с боку', 'Встаёт, лёжа на боку', 'Старт на боку, все физики'))


def getting_up(data, skills, missing, against):
    """The getting-up candidate's summary and cards: the start classes (with clips), then skills_check's groups."""
    g = data['getup']
    rows = skills or []
    if not rows:
        missing.append(dict(what=f'Проверка навыков вставания этапа {g["stage"]}',
                            why='нет skills_check этого этапа: ни файла, ни карточки «Дневника»'))
    used = set()
    cards = []
    for clip, title, alt, *labels in GETUP_STARTS:
        mine = [r for r in rows if r['label'] in labels and r.get('ok') is not None]
        used.update(r['label'] for r in mine)
        cards.append(dict(title=title, clip=clip, alt=alt, rows=[from_skills(r) for r in mine] or [
            row(labels[0], why='нет этой меры в проверке навыков', where='skills')]))
    for skill, title in GETUP_GROUPS:
        mine = [from_skills(r) for r in rows if r.get('skill') == skill and r['label'] not in used
                and r.get('ok') is not None]
        if skill.endswith('передача в ходьбу №8'):
            b = g['bridge'] or {}
            mine.append(row('Ходьба после подъёма, физика Unitree: падений', pp.group(b['velocity'][0], 'ru') + ' из ' +
                            pp.group(b['velocity'][1], 'ru'), where='bridge', weak=b['velocity'][0] > 0)
                        if b.get('velocity') else row('Ходьба после подъёма, физика Unitree: падений', where='bridge'))
        if mine:
            cards.append(dict(title=title, rows=mine))
    b = g['bridge'] or {}
    bridge = []
    if b.get('rose'):
        bridge.append(row('Встал в физике Unitree', f'{b["rose"][0]} из {b["rose"][1]}', where='bridge',
                          weak=b['rose'][0] < b['rose'][1]))
        if b.get('rise'):
            bridge.append(row('Подъём в физике Unitree, с: медиана / p90 / наибольший',
                              ' / '.join(show(x) for x in b['rise']), where='bridge'))
        if b.get('lay'):
            lay = b['lay']
            bridge.append(row('Как лёг перед подъёмом, физика Unitree',
                              f'на спине {lay[0]}, на животе {lay[1]}' + (f', на боку {lay[2]}' if lay[2] else ''),
                              where='bridge'))
        if b.get('cutoff'):
            bridge.append(row('Отсечки (тайм-аут, скорость сустава), физика Unitree', f'{b["cutoff"][0]} из {b["cutoff"][1]}',
                              where='bridge', weak=b['cutoff'][0] > 0))
    else:
        bridge.append(row('Встал в физике Unitree', where='bridge',
                          why='этап в физике Unitree не гоняли' if g['bridged'] else NO_BRIDGE))
    cards.insert(len(GETUP_STARTS), dict(title='В физике Unitree (unitree_mujoco + g1_ctrl)', rows=bridge))

    summary = []
    if rows:
        n = counts(rows)
        # against the previous getting-up stage, not a threshold: its misses are not weak spots of the robot
        summary.append(row(f'Проверка навыков против {against}: выполнено', f'{n["ok"]} из {n["judged"]}',
                           where='skills'))
    physics = g['physics'] or []
    if physics:
        k, total = sum(p['succeeded'] for p in physics), sum(p['total'] for p in physics)
        summary.append(row(f'Встал за {g["within"]} с и передан в ходьбу {g["title"]}: '
                           + ', '.join(PHYSICS.get(p['id'], p['id']) for p in physics), f'{k} из {total}',
                           where='skills'))
    stage = int(g['stage'])
    branch = g['export'].rpartition('-')[0]
    about = (f'Ветка {branch}, этап {stage} шагов ({ru(stage / 1e6, 1)} млн), выгрузка {g["export"]}; в том же '
             f'пакете, что и ходьба {data["candidate"]["title"]}. Политика вставания обучена без записей LAFAN1.')
    return dict(id='getup', title=f'Вставание после падения: этап {stage // 1_000_000}', about=about,
                summary=summary, skills=cards)


# --- cards with numbers -------------------------------------------------------------------------------------------


def stats(data, walk_skills, getup):
    out = []
    b = data['bridge']
    runs = b.get('runs') if b.get('found') else None
    if runs:
        out.append(dict(value=f'{runs[1]} / {runs[0]}',
                        label=f'падений кандидата {data["candidate"]["title"]} в прогонах в физике Unitree '
                              '(unitree_mujoco + g1_ctrl)'))
    pushes = [p for p in (b.get('pushes') or []) if not p.get('missing')]
    if pushes:
        held = sum(p['falls'][1] - p['falls'][0] for p in pushes)
        total = sum(p['falls'][1] for p in pushes)
        stand = [p for p in pushes if p['state'] == 'stand']
        walk = [p for p in pushes if p['state'] == 'walk']
        parts = []
        if stand:
            top = max(stand, key=lambda p: p['dv'].value)
            parts.append(f'стоя до Δv {show(top["dv"])} м/с ({ru(top["force"], 0)} Н, '
                         f'{pp.short(top["seconds"], "ru")} с)')
        if walk:
            parts.append(f'на ходу до Δv {show(max(walk, key=lambda p: p["dv"].value)["dv"])} м/с')
        out.append(dict(value=f'{held} / {total}', label='толчков выдержано в физике Unitree — ' + ', '.join(parts)))
    physics = (getup or {}).get('physics') or []
    if physics:
        k, total = sum(p['succeeded'] for p in physics), sum(p['total'] for p in physics)
        rose = ((getup.get('bridge') or {}).get('rose'))
        label = (f'подъёмов после падения с передачей в ходьбу {getup["title"]} — этап {int(getup["stage"]) // 1_000_000}, '
                 + ', '.join(PHYSICS.get(p['id'], p['id']) for p in physics))
        if rose:
            label += f'; в физике Unitree {rose[0]} из {rose[1]}'
        out.append(dict(value=f'{k} / {total}', label=label))
    sk = skill_rows(walk_skills['rows']) if walk_skills else {}
    if 'speed:2' in sk:
        r = sk['speed:2']
        top = max((data['candidate']['ranges'] or {}).get('lin_vel_x') or [0])
        label = f'такую долю команды 2 м/с держит бег — худший из {SPEED_RUNS} прогонов, наша симуляция'
        if r.get('v0') and not str(r['v0']).startswith('н/п'):
            label += f' (Unitree v0: {r["v0"]})'
        if top < 2:
            label += f'; в физике Unitree стик даёт не больше {pp.short(top, "ru")} м/с'
        out.append(dict(value=r['value'], label=label, accent=True))
    return out


def roadmap(root):
    stages = journal.load(root).get('roadmap') or []
    return dict(total=len(stages), done=sum(s['status'] == 'done' for s in stages),
                active=[dict(number=i + 1, title=s['title']) for i, s in enumerate(stages) if s['status'] == 'active'])


# --- clips --------------------------------------------------------------------------------------------------------


def clip_manifest(root):
    return journal.folder(root) / 'home' / 'clips.json'


def clip_job(args):
    lang, name, scenario, seconds, view, title, stage, out = args
    from humanoid_sim.clip import render
    clip, poster = render([(title, Path(stage))], scenario, Path(out), seconds, size=SIZE, view=view, lang=lang)
    return lang, name, str(clip), str(poster)


def clips(root, lab, wanted, workers, redo=False, adopt=False):
    """Renders the clips of `wanted` ({name: (scenario, seconds, view, stage id, titles {lang: text})}) that are missing
    or recorded for another stage; returns [(name, why)] of clips left missing."""
    from humanoid_sim.i18n import LANGUAGES
    manifest = read_json(clip_manifest(root), {}) or {}
    jobs = []
    with tempfile.TemporaryDirectory() as work:
        for name, (scenario, seconds, view, stage_id, titles) in wanted.items():
            files = [journal.folder(root) / f'site-{lang}-{name}{suffix}' for lang in LANGUAGES
                     for suffix in ('.webp', '-poster.jpg')]
            entry = manifest.get(name) or {}
            if adopt and not entry and all(f.is_file() for f in files):
                manifest[name] = entry = dict(stage=stage_id, scenario=scenario, adopted=time.strftime('%Y-%m-%d'))
            if not redo and entry.get('stage') == stage_id and all(f.is_file() for f in files):
                continue
            branch, _, steps = stage_id.partition(':')
            stage = lab / branch / f'stage_{int(steps):012d}'
            if not stage.is_dir():
                return [(name, f'нет этапа {stage_id} в лаборатории: ролик не снять')]
            for lang in LANGUAGES:
                jobs.append((lang, name, scenario, seconds, view, titles[lang], str(stage),
                             str(Path(work) / lang / f'{name}.webp')))
        if jobs:
            print(f'ролики: снимаю {len(jobs)} ({", ".join(sorted({j[1] for j in jobs}))})', flush=True)
            with Pool(workers) as pool:
                for lang, name, clip, poster in pool.imap_unordered(clip_job, jobs):
                    journal.add_media(root, clip, f'site-{lang}-{name}.webp')
                    journal.add_media(root, poster, f'site-{lang}-{name}-poster.jpg')
                    print(f'  site-{lang}-{name}.webp', flush=True)
            for name in {j[1] for j in jobs}:
                scenario, _, _, stage_id, _ = wanted[name]
                manifest[name] = dict(stage=stage_id, scenario=scenario, rendered=time.strftime('%Y-%m-%d %H:%M'))
    write_json(clip_manifest(root), manifest)
    return []


# --- research ------------------------------------------------------------------------------------------------------

# Research experiments the home page shows in a block of their own after the candidates (the user's permission of
# 26.10: «вставание как человек», a LAFAN1 policy). Not from a robot package and not licence-checked: such a stage
# never reaches the candidates' cards, which come from packages alone (licence()). The block is the robot's clip and
# its text; its numbers are the journal section's (`journal`, the section title's start), checked to be there still,
# else the block is left out with the reason.
RESEARCH = (dict(
    stage='exp-4131bbebbba849a4:150405120', clip='research_getup64', scenario='getup_track', seconds=8.,
    title='Исследование: вставание как человек', alt='Робот встаёт по записи человека: перекат, колени, выпад с '
                                                      'колена, стоя',
    tile='LAFAN1 · exp-4131bbeb',
    licence='Исследовательский эксперимент. Движение по записи LAFAN1 © Ubisoft La Forge, лицензия CC BY-NC-ND 4.0, '
            'некоммерческое использование. Для робота клиента обучаем на записях CMU.',
    text='Робот следует записи человека, а не придумывает движение сам: перекат, колени, выпад с колена, стоя — за '
         '2,8 с. С первого кадра записи доходит до конца в 9–10 из 10 попыток в четырёх вариантах физики; стопы не '
         'шире 0,57 м против 1,2 м у кандидата вставания 1164.',
    journal='Итерации 64–67', numbers=('2,8 с', '95–100%', 'четырёх физиках', '0,57 м', '1,2 м')),)


def research(root, missing):
    """The research blocks whose numbers their journal section still has."""
    sections = [sec for stage in journal.load(root).get('roadmap') or []
                for sec in (stage.get('story') or {}).get('sections') or []]
    out = []
    for block in RESEARCH:
        text = next((sec['text'] for sec in sections if sec['title'].startswith(block['journal'])), '')
        lost = [n for n in block['numbers'] if n not in text]
        if lost:
            missing.append(dict(what=block['title'], why=f'в разделе «Дневника» «{block["journal"]}…» нет чисел '
                                                         + ', '.join(lost)))
            continue
        out.append({key: block[key] for key in ('title', 'clip', 'alt', 'licence', 'text', 'stage')})
    return out


# --- main ---------------------------------------------------------------------------------------------------------


def build(lab, package, walk_skills=None, getup_skills=None, render=True, redo=False, adopt=False, workers=4):
    """home.json's content (not saved) from the package and the skills files; renders missing clips when render.
    lab: the lab root (runs/humanoid_lab), where the journal and the stages are."""
    root = lab
    chatter = next(iter(sorted(package.glob('chatter_*.json'))), None)
    goal = next(iter(sorted(package.glob('goal_check_*.json'))), None)
    data = pp.collect(package, lab, chatter=chatter, goal_check=goal)
    info = json.loads((package / 'package.json').read_text(encoding='utf-8'))
    main = info['candidates'][0]
    licence(package, main['slot'])
    sources = [dict(file=rel(package / 'package.json'), what='пакет робота', md5=md5(package / 'package.json'))]
    sources += [dict(file=f, what=what, md5=m or '') for f, what, m in data['sources']]
    missing = [dict(what='поле паспорта', why=w) for w in data['warnings']]

    c = data['candidate']
    walk_file = Path(walk_skills) if walk_skills else journal.folder(root) / 'home' / f'skills_{c["steps"]}.json'
    walk = skills_file(walk_file, c['steps'])
    if walk:
        sources.append(dict(file=rel(walk_file), what='проверка навыков ходьбы', md5=md5(walk_file)))
    candidates = [walking(data, walk, missing)]

    getup = data['getup']
    if getup:
        licence(package, info['getup']['slot'])
        steps = int(getup['stage'])
        getup_file = Path(getup_skills) if getup_skills else journal.folder(root) / 'home' / f'skills_{steps}.json'
        payload = skills_file(getup_file, steps)
        rows, against = None, '—'
        if payload:
            rows = payload['rows']
            against = payload['reference']['label']
            sources.append(dict(file=rel(getup_file), what='проверка навыков вставания', md5=md5(getup_file)))
        else:
            card = journal.load(root).get('skills') or {}
            if int(card.get('stage') or -1) == steps and card.get('branch') == getup['export'].rpartition('-')[0]:
                rows = card['items']
                against = f'этапа {int(card["against"]["stage"]) // 1_000_000}'
                sources.append(dict(file=rel(journal.folder(root) / 'journal.json'),
                                    what='карточка «Проверка навыков» «Дневника»', md5=''))
        candidates.append(getting_up(data, rows, missing, against))
    else:
        missing.append(dict(what='Вставание', why='в пакете нет политики вставания'))

    if render:
        stage_walk = f'{c["branch"]}:{c["steps"]}'
        wanted = {name: (*spec, stage_walk, {lang: TITLES[lang].get(name, c['title']) for lang in TITLES})
                  for name, spec in WALK_CLIPS.items()}
        if getup:
            stage_getup = f'{getup["export"].rpartition("-")[0]}:{int(getup["stage"])}'
            wanted.update({name: (*spec, stage_getup, {lang: f'GetUp {int(getup["stage"]) // 1_000_000}'
                                                        for lang in TITLES}) for name, spec in GETUP_CLIPS.items()})
        wanted.update({block['clip']: (block['scenario'], block['seconds'], None, block['stage'],
                                       {lang: block['tile'] for lang in TITLES}) for block in RESEARCH})
        for name, why in clips(root, lab, wanted, workers, redo, adopt):
            missing.append(dict(what=f'ролик {name}', why=why))
    present = {name for name in [*WALK_CLIPS, *GETUP_CLIPS]
               if (journal.folder(root) / f'site-ru-{name}.webp').is_file()}
    for candidate in candidates:
        for card in candidate['skills']:
            if card.get('clip') and card['clip'] not in present:
                missing.append(dict(what=f'ролик {card["clip"]}', why='файла нет в папке «Дневника»'))
                card.pop('clip')
    blocks = []
    for block in research(root, missing):
        if (journal.folder(root) / f'site-ru-{block["clip"]}.webp').is_file():
            blocks.append(block)
        else:
            missing.append(dict(what=f'ролик {block["clip"]}', why='файла нет в папке «Дневника»'))
    return dict(version=home.VERSION, at=time.time(), date=time.strftime('%Y-%m-%d'), roadmap=roadmap(root),
                stats=stats(data, walk, getup), candidates=candidates, missing=missing, sources=sources,
                research=blocks)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', type=Path, help='пакет робота (по умолчанию новейший с основным кандидатом)')
    parser.add_argument('--walk-skills', type=Path, help='skills_check --json этапа ходьбы')
    parser.add_argument('--getup-skills', type=Path, help='skills_check --json этапа вставания')
    parser.add_argument('--no-clips', action='store_true', help='не снимать недостающие ролики')
    parser.add_argument('--redo-clips', action='store_true', help='снять все ролики заново')
    parser.add_argument('--adopt-clips', action='store_true',
                        help='записать имеющиеся ролики без отметки как ролики текущего кандидата')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='напечатать, не записывая home.json')
    parser.add_argument('--lab', type=Path, default=LAB)
    args = parser.parse_args(argv)
    lab = args.lab.resolve()
    package = (args.package or newest_package(lab)).resolve()
    data = build(lab, package, args.walk_skills, args.getup_skills, not args.no_clips and not args.dry_run,
                 args.redo_clips, args.adopt_clips, args.workers)
    if args.dry_run:
        print(json.dumps(home.validate(data), ensure_ascii=False, indent=1))
    else:
        home.save(lab, data)
        print(f'записано: {home.path(lab)}')
    stage = data['roadmap']['active']
    print(f'пакет {package.name}; этапы {", ".join(str(s["number"]) for s in stage)} из {data["roadmap"]["total"]}; '
          f'карточек чисел {len(data["stats"])}; навыков '
          + ', '.join(f'{c["id"]} {len(c["skills"])}' for c in data['candidates']))
    for item in data['missing']:
        print(f'  пусто: {item["what"]} — {item["why"]}')


if __name__ == '__main__':
    main()
