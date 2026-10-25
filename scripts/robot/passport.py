"""Candidate passport (CPU only, reads only): one page per language from the numbers a robot package already has.

Usage: .venv\\Scripts\\python.exe scripts\\robot\\passport.py <package dir> [--out DIR] [--chatter JSON]
       [--goal-check JSON] [--lang ru,en,zh] [--lab DIR]

Writes PASSPORT.<lang>.md (by default into the package folder) for a package of scripts/robot/package.py
(runs/humanoid_lab/robot/<name>). Nothing is measured or recomputed here; the numbers come from:
  package.json        the candidate, its export, the contract (ONNX input / output), command ranges, the knee clip,
                      the build date and commit, the getting-up policy if the package has one
  REFERENCE.md        the package's tables as package.py wrote them: the bridge (unitree_mujoco + g1_ctrl; the main
                      candidate's column of SIM2SIM_TABLE*, from docs/sim2sim.md) and, with getting up, the critic's
                      probe and the bridge of the packaged stage (GETUP_PROBE / GETUP_BRIDGE)
  reference/enc_bias_<stage>.json   the encoder bias probe of the main candidate (the package's copy)
  <lab>/<branch>/stage_<steps>/     the stage's own evaluation (core.evaluate_stage: CPU, the training disturbances
                      of its task_config; report.json's copy if the stage was pruned)
  --goal-check JSON   the line `python -m humanoid_sim.goal_check <branch>:<steps>` (or --export) prints last (its
                      items, stage, export), if it is this stage or its export; without the option the journal's
                      current check (<lab>/journal/journal.json), if that is this stage
  --chatter JSON      the summary of scripts/lab/probes/chatter.py --json (torque chatter), if it is this stage's
Derived numbers, each said so on the page: a push of Δv lasting T is the force F = M·Δv/T and the impulse M·Δv (M:
mj_getTotalmass of the G1 model; T: the push's own duration, in the bridge the push patch's kSeconds); the heading after
100 m of walking without correction is the measured drift extrapolated (ω·100 m / v, arc radius v/ω). A field without a
number stays empty, the reason beside it; nothing is made up (motor heating is measured on the robot only).
Every Russian text of the page is a catalog key of humanoid_sim/web/i18n/<lang>.json (render(); tests/test_i18n.py and
journal_translate export list the missing ones for the lab's translator); without a translation en / zh show the
Russian, with the numbers in their own form (decimal point), as the rest of the interface.
"""

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1]
sys.path.insert(0, str(SIM))
from humanoid_sim.i18n import LANGUAGES, UNITS, Content, gettext, localize_journal, unit  # noqa: E402

LAB = SIM / 'runs' / 'humanoid_lab'
GIT = SIM / '.tools' / 'git' / 'cmd' / 'git.exe'
PUSH_PATCH = SIM / 'scripts' / 'sim2sim' / 'patches' / 'unitree_mujoco-push.patch'
MODEL = 'assets/robots/g1/scene_29dof.xml'  # the file of humanoid_sim.tasks.g1_velocity.load_model
KIND = 'g1-robot-package'
NAME = 'PASSPORT.{lang}.md'
HUNDRED = 100.  # m: the walk the measured drift is extrapolated over
# REFERENCE.md's sections (scripts/robot/package.py, reference_md and getup_reference_md)
SIM2SIM_HEADING = '## Связка Unitree в симуляции (sim2sim)'
GETUP_HEADING = '## Вставание (GetUp)'
PROBE_HEADING = '### Проба критика'
BRIDGE_HEADING = '### Связка (sim2sim)'
PACKAGED = '(в пакете)'  # the getting-up tables' column of the packaged stage
# package.py's words for the roles and the critic's getting-up physics (GETUP_PROBE's first row), as ids render() labels
ROLES = {'основной': 'main', 'запасной': 'spare', 'экспериментальный': 'experimental'}
GETUP_PHYSICS = {'наша': 'ours', 'задержка 20 мс': 'delay', 'пол 0,4–0,2': 'floor', 'шум': 'noise'}
# The bridge's rows (REFERENCE.md, the first cell) by what they give; 'side' may match several rows
ROWS = dict(
    runs=r'Прогонов; падений$',
    delay=r'Задержка: состояние → первый шаг физики с новой целью, медиана / 95% / макс$',
    lowcmd=r'lowstate пришёл → lowcmd с ответом \(часы ВМ\): медиана; 95%$',
    walk05=r'Прямо 0,5 м/с: скорость; увод после 2 с; курс за 10 с$',
    walk1=r'Прямо 1 м/с: скорость; увод; курс за 10 с$',
    side10=r'За 10 с ходьбы 0,5 м/с: вперёд; вбок$',
    turn=r'Поворот ±(?P<rate>[\d,]+) рад/с: на месте; на ходу (?P<speed>[\d,]+) м/с$',
    brake=r'Торможение за (?P<seconds>\d+) с от стопа с (?P<speeds>[\d, /]+) м/с$',
    side=r'Шаг вбок ±(?P<speed>[\d,]+) м/с\b',
    side_stop=r'Стоп после шага вбок ±(?P<a>[\d,]+) / ±(?P<b>[\d,]+): сдвиг за (?P<seconds>\d+) с',
    stand30=r'Стояние 30 с: поворот; смещение$',
    push_stand=r'Толчки стоя \(сила на таз (?P<seconds>[\d,]+) с\), падений: Δv (?P<dv>.+) м/с$',
    push_walk=r'Толчки на ходу (?P<speed>[\d,]+) м/с, падений: Δv (?P<dv>.+) м/с$',
    push_tilt=r'Наклон при толчках стоя Δv (?P<dv>[\d,–]+), наибольший$',
    clip=r'Цели вне clip по всем 29 суставам',
)
PROBE_ROWS = dict(
    success=r'Успех: встал за (?P<within>\d+) с, без отсечки скорости и повторного падения до передачи, Velocity '
            r'(?P<title>№\d+) через (?P<after>\d+) с без падения \((?P<physics>[^()]+)\)$',
    rise=r'Подъём, с: медиана / p90 / наибольший$',
    cutoff=r'Отсечка скорости по порогам GetUp',
    refall=r'Повторное падение в GetUp после подъёма',
    velocity=r'Velocity №\d+ из стойки: падений$',
)
BRIDGE_ROWS = dict(
    rose=r'Встал; время подъёма$',
    lay=r'Как лёг перед GetUp$',
    cutoff=r'Отсечки \(тайм-аут, скорость сустава\)$',
    velocity=r'Velocity (?P<title>№\d+): падений',
)
STARTS = re.compile(r'(\d+) новых стартов — спина, живот, бок, сидя по (\d+) и падения (№\d+) (\d+)')
RISE = re.compile(r'([\d,]+)–([\d,]+) с, медиана ([\d,]+), p90 ([\d,]+)')
NUM = re.compile(r'[+−-]?\d+(?:,\d+)?')
COUNT = re.compile(r'(\d{1,3}(?: \d{3})+|\d+) из (\d{1,3}(?: \d{3})+|\d+)')
# scripts/lab/probes/chatter.py's summary: the keys shown, with the decimals its FORMATS print them with
CHATTER = dict(torque_hf_rms_nm=2, action_hf_rms_rad=4, action_hf_share=3, torque_hf_share=3, torque_rms_peak_share=3,
               mean_abs_delta_action=3)
CUTOFFS = dict(action_cutoff=5., torque_cutoff=10.)  # chatter.py's ACTION_CUTOFF / TORQUE_CUTOFF, if settings lack them
EXPORT_LABEL = ' (выгрузка)'  # chatter.py's label of an export folder

# A number as the source wrote it: its value, decimals and whether it carried a sign.
Num = collections.namedtuple('Num', 'value digits signed')


def fail(message):
    raise SystemExit(f'passport.py: {message}')


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(SIM)).replace('\\', '/')
    except ValueError:
        return str(path)


# --- the package's tables ----------------------------------------------------------------------------------------


def section(text, heading):
    """The lines under the heading that starts with `heading`, up to the next heading of its level or above."""
    level = len(heading) - len(heading.lstrip('#'))
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith('#'):
            depth = len(line) - len(line.lstrip('#'))
            if inside and depth <= level:
                break
            if line.startswith(heading):
                inside = True
                continue
        if inside:
            lines.append(line)
    return '\n'.join(lines)


def cells(line):
    return [c.strip().replace('\\|', '|') for c in re.split(r'(?<!\\)\|', line.strip())[1:-1]]


def table(text):
    """The first markdown table in text: (its header cells, its rows of cells); (None, []) if there is none."""
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.startswith('|') and re.fullmatch(r'\|(\s*:?-+:?\s*\|)+', lines[i + 1].strip()):
            rows = []
            for other in lines[i + 2:]:
                if not other.startswith('|'):
                    break
                rows.append(cells(other))
            return cells(line), rows
    return None, []


def column(text, pick):
    """{first cell: the cell of the first column whose header pick() accepts} of the first table in text, or {}."""
    head, rows = table(text)
    found = [k for k, name in enumerate(head or []) if k and pick(name)]
    return {row[0]: row[found[0]] for row in rows if len(row) > found[0]} if found else {}


def find(values, patterns):
    """{key: (match, cell)} of the first row whose first cell matches each pattern; for 'side' every such row."""
    out = {}
    for key, pattern in patterns.items():
        hits = [(m, cell) for what, cell in values.items() for m in [re.match(pattern, what)] if m]
        if hits:
            out[key] = hits if key == 'side' else hits[0]
    return out


def num(token):
    return Num(float(token.replace('−', '-').replace(',', '.')), len(token.partition(',')[2]), token[0] in '+−-')


def clean(text):
    """A cell without its remarks and the numbers that are not this measurement: parentheses, titles such as №8,
    ordinals such as 2-й, names such as p90 or q20."""
    text = re.sub(r'\s*\([^()]*\)', '', text)
    return re.sub(r'№\s*\d+|\d+-(?:й|я|е|го)\b|\b[a-z]\d+\b', '', text)


def nums(text):
    return [num(t) for t in NUM.findall(clean(text))]


def parts(text, sep='; '):
    return [p.strip() for p in clean(text).split(sep)]


def count(text):
    m = COUNT.search(clean(text))
    return (int(m[1].replace(' ', '')), int(m[2].replace(' ', ''))) if m else None


def spread(values):
    """(lowest, highest) of the numbers, or None without any."""
    return (min(values, key=lambda x: x.value), max(values, key=lambda x: x.value)) if values else None


def extrapolate(speed, drift, metres=HUNDRED):
    """The measured drift held for `metres` of walking: the heading then (ω·metres/v, degrees, lowest and highest over
    the measured speeds and drifts) and the arc's radius v/ω (m; None when the drift changes sign or is 0)."""
    rates, speeds = [d.value for d in drift], [s.value for s in speed]
    if not rates or min(speeds) <= 0:
        return None
    turns = [math.degrees(w * metres / v) for w in (min(rates), max(rates)) for v in speeds]
    radius = None
    if min(rates) > 0 or max(rates) < 0:
        radius = (min(speeds) / max(abs(w) for w in rates), max(speeds) / min(abs(w) for w in rates))
    return dict(metres=metres, heading=(min(turns), max(turns)), radius=radius)


def walk(cell):
    """'0,46–0,47 м/с; +0,032 / +0,034 / +0,037 рад/с; +17,2…+20,2°': speed, drift after 2 s, heading over 10 s."""
    p = [nums(x) for x in parts(cell)]
    if len(p) < 3 or not all(p[:3]):
        return None
    speed = spread(p[0])
    return dict(speed=speed, drift=spread(p[1]), heading=spread(p[2]), extrapolated=extrapolate(speed, p[1]))


def bridge_data(values, mass, patch_seconds, warnings):
    """The main candidate's bridge column (REFERENCE.md) as numbers; a row that is missing or says no number (such as
    «в связке не мерили») gives None."""
    rows = find(values, ROWS)
    out = dict(found=bool(values), side=[], pushes=[])
    cell = lambda key: rows[key][1]
    if 'runs' in rows:
        p = [nums(x) for x in parts(cell('runs'))]
        out['runs'] = (int(p[0][0].value), int(p[1][0].value)) if len(p) > 1 and p[0] and p[1] else None
    if 'delay' in rows:
        p = [nums(x) for x in parts(cell('delay'), ' / ')]
        out['delay'] = [spread(x) for x in p] if len(p) == 3 and all(p) else None
    if 'lowcmd' in rows:
        p = [nums(x) for x in parts(cell('lowcmd'))]
        out['lowcmd'] = [spread(x) for x in p] if len(p) == 2 and all(p) else None
    for key in ('walk05', 'walk1'):
        if key in rows:
            out[key] = walk(cell(key))
    if 'side10' in rows:
        p = parts(cell('side10'))
        out['side10'] = spread(nums(p[1])) if len(p) > 1 else None
    if 'turn' in rows:
        m = rows['turn'][0]
        p = [nums(x) for x in parts(cell('turn'))]
        out['turn'] = dict(rate=num(m['rate']), speed=num(m['speed']), place=p[0], walking=p[1]) if (
            len(p) == 2 and len(p[0]) == 2 and len(p[1]) == 2) else None
    if 'brake' in rows:
        m = rows['brake'][0]
        speeds = [num(x.strip()) for x in m['speeds'].split('/')]
        p = [nums(x) for x in parts(cell('brake'), ' / ')]
        out['brake'] = dict(seconds=int(m['seconds']), rows=[(v, spread(x)) for v, x in zip(speeds, p)]) if (
            len(p) == len(speeds) and all(p)) else None
    for m, text in rows.get('side', []):
        p = [nums(x) for x in parts(text)]
        if len(p) >= 3 and all(p[:3]):
            out['side'].append(dict(speed=num(m['speed']), speeds=spread(p[0]), drift=spread(p[1] + p[2])))
    if 'side_stop' in rows:
        m = rows['side_stop'][0]
        p = [nums(x) for x in parts(parts(cell('side_stop'))[0], ' / ')]
        out['side_stop'] = dict(seconds=int(m['seconds']), rows=list(zip((num(m['a']), num(m['b'])), map(spread, p)))
                                ) if len(p) == 2 and all(p) else None
    if 'stand30' in rows:  # '−8,0° / −8,1° / −7,3° (3 прогона); 2 см; без шага', or '…; 3 см, после 2-й с 1 см'
        p = [nums(x.split(', ')[0]) for x in parts(cell('stand30'))]
        out['stand30'] = dict(heading=spread(p[0]), offset=spread(p[1])) if len(p) > 1 and p[0] and p[1] else None
    for key, state in (('push_stand', 'stand'), ('push_walk', 'walk')):
        if key not in rows:
            continue
        m = rows[key][0]
        found = m.groupdict()
        seconds = patch_seconds
        if found.get('seconds'):
            seconds = num(found['seconds']).value
            if patch_seconds is not None and abs(seconds - patch_seconds) > 1e-9:
                warnings.append(f'толчок в REFERENCE.md {seconds:g} с, а в {PUSH_PATCH.name} {patch_seconds:g} с: '
                                'беру из REFERENCE.md')
        speed = num(found['speed']) if found.get('speed') else None
        dvs = [num(x.strip()) for x in m['dv'].split('/')]
        falls = [count(x) for x in cell(key).split(' / ')]
        if len(falls) != len(dvs) or None in falls or seconds is None:
            out['pushes'].append(dict(state=state, speed=speed, missing=True))
            continue
        for dv, fell in zip(dvs, falls):
            out['pushes'].append(dict(state=state, speed=speed, dv=dv, seconds=seconds, falls=fell,
                                      force=mass * dv.value / seconds, impulse=mass * dv.value))
    if 'push_tilt' in rows:
        m = rows['push_tilt'][0]
        deg = nums(cell('push_tilt'))
        out['push_tilt'] = dict(dv=[num(x) for x in m['dv'].split('–')], deg=deg[0]) if deg else None
    if 'clip' in rows:
        out['clip'] = count(cell('clip'))
    return out


def getup_data(reference, getup):
    """The packaged getting-up stage's probe and bridge numbers (REFERENCE.md, «Вставание (GetUp)»): only its own
    columns; a bridge shown there for another stage of its branch is not this stage's."""
    text = section(reference, GETUP_HEADING)
    probe = column(section(text, PROBE_HEADING), lambda head: head.endswith(PACKAGED))
    bridge = column(section(text, BRIDGE_HEADING), lambda head: head.endswith(PACKAGED))
    # bridged: REFERENCE.md has its bridge section; without the stage's own column there it did not run in the bridge
    out = dict(stage=getup['stage'], export=getup['export'], physics=None, bridge=None, starts=None,
               bridged=bool(section(text, BRIDGE_HEADING).strip()))
    rows = find(probe, PROBE_ROWS)
    if 'success' in rows:
        m, cell = rows['success']
        ids = [GETUP_PHYSICS.get(x.strip()) for x in m['physics'].split('/')]
        found = re.fullmatch(r'([\d / ]+) из (\d+)', clean(cell))
        if found and None not in ids and len(found[1].split('/')) == len(ids):
            rise = [nums(x) for x in parts(rows['rise'][1])] if 'rise' in rows else []
            refall = [int(x.value) for x in nums(rows['refall'][1].split(',')[0])] if 'refall' in rows else []
            out['physics'] = [dict(id=pid, total=int(found[2]), succeeded=int(k),
                                   rise=rise[i] if len(rise) == len(ids) and len(rise[i]) == 3 else None,
                                   refall=refall[i] if len(refall) == len(ids) else None)
                              for i, (pid, k) in enumerate(zip(ids, found[1].split('/')))]
            out.update(within=int(m['within']), after=int(m['after']), title=m['title'])
    out['cutoff'] = count(rows['cutoff'][1]) if 'cutoff' in rows else None
    velocity = nums(rows['velocity'][1]) if 'velocity' in rows else []
    out['velocity_falls'] = int(velocity[0].value) if velocity else None
    starts = STARTS.search(text)
    if starts:
        out['starts'] = dict(total=int(starts[1]), each=int(starts[2]), title=starts[3], falls=int(starts[4]))
    if bridge:
        rows = find(bridge, BRIDGE_ROWS)
        b = dict(rose=None, rise=None, lay=None, cutoff=None, velocity=None)
        if 'rose' in rows:
            b['rose'] = count(rows['rose'][1].split('; ')[0])
            t = RISE.search(rows['rose'][1])
            b['rise'] = [num(t[3]), num(t[4]), num(t[2])] if t else None  # median, p90, the longest
        if 'lay' in rows:
            lay = re.search(r'спина (\d+), живот (\d+)', rows['lay'][1])
            b['lay'] = (int(lay[1]), int(lay[2])) if lay else None
        if 'cutoff' in rows:
            b['cutoff'] = count(rows['cutoff'][1])
        if 'velocity' in rows:
            b['velocity'] = count(rows['velocity'][1].split('; ')[0])
        out['bridge'] = b
    return out


# --- the other sources -------------------------------------------------------------------------------------------


def total_mass():
    """mj_getTotalmass of the G1 model (the foot and hand capsules have no mass), kg."""
    import mujoco
    from humanoid_sim.tasks.g1_velocity import load_model
    return float(mujoco.mj_getTotalmass(load_model()[0]))


def patch_seconds():
    """The bridge's push length, Push::kSeconds of unitree_mujoco-push.patch (s), or None."""
    if not PUSH_PATCH.is_file():
        return None
    m = re.search(r'kSeconds = ([\d.]+);', PUSH_PATCH.read_text(encoding='utf-8'))
    return float(m[1]) if m else None


def stage_evaluation(lab, branch, steps):
    """(evaluation, task_config, path) of the stage: its evaluation.json and metadata.json, else report.json's entry;
    (None, None, None) without either."""
    stage = lab / branch / f'stage_{steps:012d}'
    if (stage / 'evaluation.json').is_file() and (stage / 'metadata.json').is_file():
        evaluation = json.loads((stage / 'evaluation.json').read_text(encoding='utf-8'))
        metadata = json.loads((stage / 'metadata.json').read_text(encoding='utf-8'))
        return evaluation, metadata.get('task_config') or {}, stage / 'evaluation.json'
    report = lab / branch / 'report.json'
    if report.is_file():
        for entry in json.loads(report.read_text(encoding='utf-8')).get('stages', []):
            if entry.get('steps') == steps and entry.get('episodes'):
                return entry, entry.get('task_config') or {}, report
    return None, None, None


def training_data(evaluation, config):
    """The stage evaluation's falls, and the training disturbances of its task_config that the CPU task draws at a
    reset without a command, as evaluate_stage resets: force pushes, arm poses, motor friction, the action delay."""
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.config import ARM_PERTURB_RADIANS
    episodes = evaluation.get('episodes') or []
    out = dict(episodes=len(episodes), falls=sum(bool(e.get('fell')) for e in episodes),
               seconds=max((float(e.get('seconds') or 0) for e in episodes), default=None),
               delay=float(config.get('action_delay') or 0.), push=None, arms=None, friction=None)
    if config.get('force_push'):
        out['push'] = dict(dv=float(config['force_push']), seconds=task.FORCE_PUSH_SECONDS,
                           interval=task.FORCE_PUSH_INTERVAL)
    if config.get('arm_perturb'):
        out['arms'] = dict(share=float(config['arm_perturb']),
                           radians=float(config.get('arm_perturb_radians') or ARM_PERTURB_RADIANS))
    if config.get('motor_friction'):
        out['friction'] = float(config['motor_friction'])
    return out


def encoder_data(path):
    """The encoder bias probe (reference/enc_bias.py) under the robot's model, the bias in the observation and in the
    PD: draws, runs, falls, the largest bias and the drift at 0.5 and 1 m/s and the heading after 30 s standing."""
    if not path.is_file():
        return None
    rows = [r for r in json.loads(path.read_text(encoding='utf-8')) if r['model'] == 'calib' and r['draw'] >= 0]
    values = {key: [r[field] for r in rows if r['plan'] == key and r['fell'] is None]
              for key, field in (('w05', 'yaw_rate'), ('w1', 'yaw_rate'), ('stand', 'heading_deg'))}
    if not rows or not all(values.values()):
        return None
    return dict(draws=len({r['draw'] for r in rows}), runs=len(rows), falls=sum(r['fell'] is not None for r in rows),
                bias=max(abs(b) for r in rows for b in r['bias']), **{k: (min(v), max(v)) for k, v in values.items()})


def goal_data(lab, branch, steps):
    """(the journal's current check if it is this stage's, else None; else the branch and stage it is of)."""
    path = lab / 'journal' / 'journal.json'
    check = json.loads(path.read_text(encoding='utf-8')).get('check') if path.is_file() else None
    if not check:
        return None, None
    if check.get('branch') == branch and int(check.get('stage') or -1) == steps:
        return check, None
    return None, dict(branch=check.get('branch'), stage=check.get('stage'))


def goal_file_data(path, steps, export):
    """goal_check's JSON (its last printed line; a whole saved output works too, the last non-empty line is read): its
    items, the stage and export it checked and whether that is not this candidate (neither its stage nor its export)."""
    lines = [line for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else None
    except ValueError:
        payload = None
    items = payload.get('items') if isinstance(payload, dict) else None
    if not isinstance(items, list) or not all(isinstance(r, dict) and 'label' in r for r in items):
        fail(f'{path}: нет items с label/value/target/ok — это не строка JSON из python -m humanoid_sim.goal_check')
    stage = payload.get('stage')
    checked = Path(str(payload['export'])).name if payload.get('export') else None
    mine = (stage is not None and int(stage) == steps) or checked == export
    return dict(items=items, stage=None if stage is None else int(stage), export=checked, file=Path(path).name,
                mismatch=not mine)


def chatter_data(path, steps, export):
    """chatter.py --json: its summary and settings, the label of the policy it ran and whether that is not this
    candidate (its stage or its export)."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    label = str(payload.get('stage') or '')
    where = f"{label} {payload.get('path') or ''}"
    return dict(summary=payload.get('summary') or {}, settings=payload.get('settings') or {},
                label=label.removesuffix(EXPORT_LABEL), export=label.endswith(EXPORT_LABEL),
                mismatch=str(steps) not in where and export not in where)


def git_head():
    if not GIT.exists():
        return None
    out = subprocess.run([str(GIT), '-C', str(SIM), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return out.stdout.strip() or None


def collect(package, lab=LAB, chatter=None, goal_check=None):
    """Everything the passport shows, language-free: numbers (Num as the source wrote them, floats where derived),
    None where a source has none; warnings for the console. goal_check: a saved goal_check JSON, used instead of the
    journal's check."""
    package, lab = Path(package), Path(lab)
    manifest = package / 'package.json'
    if not manifest.is_file():
        fail(f'нет {manifest}: это не пакет scripts/robot/package.py')
    info = json.loads(manifest.read_text(encoding='utf-8'))
    if info.get('kind') != KIND:
        fail(f'{manifest}: kind {info.get("kind")!r}, а не {KIND!r}')
    main, spare = info['candidates'][0], info['candidates'][1:]
    branch, _, steps = main['export'].rpartition('-')
    steps = int(steps)
    warnings, sources = [], [(rel(manifest), 'package', md5(manifest))]
    reference_path = package / 'REFERENCE.md'
    reference = ''
    if reference_path.is_file():
        reference = reference_path.read_text(encoding='utf-8')
        sources.append((rel(reference_path), 'reference', md5(reference_path)))
    else:
        warnings.append(f'нет {reference_path}: чисел связки и вставания нет')
    deploy = package / 'policy' / main['slot'] / 'deploy.yaml'
    step_dt = float(yaml.safe_load(deploy.read_text(encoding='utf-8')).get('step_dt') or 0) if deploy.is_file() else 0
    mass, seconds = total_mass(), patch_seconds()
    sources.append((MODEL, 'mass', None))
    if seconds is not None:
        sources.append((rel(PUSH_PATCH), 'push', md5(PUSH_PATCH)))
    # the column of REFERENCE.md: «№8 (7221)», millions of steps (packages of 23.10 before n7 lack 'reference')
    title = f"{main['title']} ({main.get('reference') or steps // 1_000_000})"
    values = column(section(reference, SIM2SIM_HEADING), lambda head: head == title)
    if reference and not values:
        warnings.append(f'в REFERENCE.md нет столбца «{title}» в разделе «{SIM2SIM_HEADING[3:]}»')
    bridge = bridge_data(values, mass, seconds, warnings)
    about = re.match(r'итерация (\d+)', main.get('about') or '')
    data = dict(
        package=package.name, created=info.get('created'), build_commit=(info.get('git') or {}).get('commit'),
        build_dirty=bool((info.get('git') or {}).get('uncommitted')), lab_commit=git_head(), lab=lab, mass=mass,
        push_seconds=next((p['seconds'] for p in bridge['pushes'] if 'seconds' in p), seconds),
        date=time.strftime('%Y-%m-%d'),
        candidate=dict(title=main['title'], role=ROLES.get(main.get('role'), main.get('role')), export=main['export'],
                       branch=branch, steps=steps, iteration=int(about[1]) if about else None,
                       onnx=main.get('onnx') or {}, step_dt=step_dt or None, ranges=main.get('command_ranges') or {},
                       knee=(main.get('knee_clip') or [None])[0], sim2sim=(main.get('sim2sim') or '')[3:] or None),
        spare=[c['title'] for c in spare], bridge=bridge, warnings=warnings, sources=sources)
    encoders = package / 'reference' / f"enc_bias_{main.get('reference') or steps // 1_000_000}.json"
    data['encoders'] = encoder_data(encoders)
    if data['encoders']:
        sources.append((rel(encoders), 'encoders', md5(encoders)))
    evaluation, config, path = stage_evaluation(lab, branch, steps)
    data['training'] = training_data(evaluation, config) if evaluation else None
    if path is not None:
        sources.append((rel(path), 'training', md5(path)))
    data['goal'], data['goal_other'], data['goal_file'] = None, None, None
    if goal_check is not None:
        data['goal_file'] = goal_file_data(goal_check, steps, main['export'])
        sources.append((rel(goal_check), 'goal', md5(goal_check)))
        if data['goal_file']['mismatch']:
            g = data['goal_file']
            warnings.append(f"goal_check в {g['file']} — этап {g['stage']}" + (f", выгрузка {g['export']}" if
                            g['export'] else '') + f', а не {steps}: поле проверки цели пустое')
    else:
        data['goal'], data['goal_other'] = goal_data(lab, branch, steps)
        if data['goal']:
            sources.append((rel(lab / 'journal' / 'journal.json'), 'goal', None))
    data['getup'] = getup_data(reference, info['getup']) if info.get('getup') else None
    data['chatter'] = None
    if chatter is not None:
        data['chatter'] = chatter_data(chatter, steps, main['export'])
        sources.append((rel(chatter), 'chatter', md5(chatter)))
        if data['chatter']['mismatch']:
            warnings.append(f"проба chatter — {data['chatter']['label']}, а не этап {steps}: поля дрожания пустые")
    return data


# --- the page ----------------------------------------------------------------------------------------------------


def number(x, digits, lang, sign=False):
    """x with `digits` decimals, a typographic minus, a decimal comma in Russian and a point otherwise."""
    text = f'{x:+.{digits}f}' if sign and round(x, digits) != 0 else f'{x:.{digits}f}'
    text = text.replace('-', '−')
    return text.replace('.', ',') if lang == 'ru' else text


def short(x, lang, digits=3):
    """x with at most `digits` decimals, trailing zeros dropped (1.25, 1, 0.5)."""
    text = f'{x:.{digits}f}'.rstrip('0').rstrip('.') if digits else f'{x:.0f}'
    return number(float(text), len(text.partition('.')[2]), lang)


def show(value, lang):
    """A source number as it was written (its decimals, its sign), in the language's form."""
    return number(value.value, value.digits, lang, value.signed)


def span(pair, lang):
    """'a–b' (unsigned) or 'a…b' (signed) of a (lowest, highest) pair; one number when both print the same."""
    low, high = (show(x, lang) for x in pair)
    return low if low == high else low + ('…' if pair[0].signed or pair[1].signed else '–') + high


def group(n, lang):
    """A whole number with its thousands grouped: 719 902 in Russian (from five digits on, 1104 as it is), 719,902 and
    1,104 otherwise."""
    if lang == 'ru':
        return f'{n:,}'.replace(',', ' ') if abs(n) >= 10_000 else str(n)
    return f'{n:,}'


def esc(text):
    return '' if text is None else str(text).replace('|', '\\|').replace('\n', ' ')


def render(data, lang='ru'):
    """The passport in `lang` (ru, en, zh) and the labels of its empty fields. Every Russian text here is a catalog key
    (humanoid_sim/web/i18n; tests/test_i18n.py checks this function and journal_translate export hands its missing keys
    to the translator); the numbers go in through placeholders."""
    _ = gettext(lang)
    empty = []
    n = lambda x, digits=2, sign=False: number(x, digits, lang, sign)
    s = lambda x: show(x, lang)
    sp = lambda pair: span(pair, lang)
    of = lambda pair: _('{k} из {n}', k=group(pair[0], lang), n=group(pair[1], lang))
    u = lambda value, symbol: unit(value, symbol, lang)  # numbers and a unit (i18n.UNITS), no catalog key needed
    sym = lambda symbol: UNITS[symbol] if lang == 'ru' else symbol
    c, b = data['candidate'], data['bridge']
    bridge = _('связка unitree_mujoco + g1_ctrl')
    ours = _('наша физика MuJoCo (CPU)')
    package = _('пакет')
    nothing = _('нет числа в REFERENCE.md пакета')
    critic = _('проба критика')

    def kv(rows):
        """(label, value, where, reason) rows: a row without a value is an empty field, the reason in its last cell."""
        lines = [f"| {_('Показатель')} | {_('Значение')} | {_('Откуда')} |", '|---|---|---|']
        for label, value, where, reason in rows:
            if not value:
                empty.append(label)
                value, where = '', reason or where
            lines.append(f'| {esc(label)} | {esc(value)} | {esc(where)} |')
        return '\n'.join(lines)

    def get(key):
        return b.get(key) if b.get('found') else None

    roles = dict(main=_('основной'), spare=_('запасной'), experimental=_('экспериментальный'))
    role = roles.get(c['role'], c['role'])
    candidate = f"{c['title']} ({role})" if role else c['title']
    out = ['# ' + _('Паспорт кандидата: Unitree G1, {title}', title=candidate), '',
           _('Собран скриптом {script} {date} из уже посчитанных чисел пакета {package}; ничего не мерилось заново. '
             'Всё ниже — симуляция: наша физика MuJoCo и связка unitree_mujoco + g1_ctrl (sim2sim). Данных '
             'настоящего робота пока нет. Пустое поле — числа нет, его не придумывали.',
             script='`scripts/robot/passport.py`', date=data['date'], package=f"`{data['package']}`"), '']

    # who
    onnx, ranges, getup = c['onnx'], c['ranges'], data['getup']
    contract = (f"ONNX `{onnx['input'][0]}` {onnx['input'][1]} → `{onnx['output'][0]}` {onnx['output'][1]}"
                if onnx.get('input') and onnx.get('output') else None)

    def window(key):
        lo, hi = (float(v) for v in ranges[key])
        return '±' + short(hi, lang) if lo == -hi else short(lo, lang) + '…' + short(hi, lang)
    commands = (_('вперёд {x} м/с, вбок {y} м/с, поворот {z} рад/с', x=window('lin_vel_x'), y=window('lin_vel_y'),
                  z=window('ang_vel_z')) if all(k in ranges for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')) else None)
    inside = _('ходьба (Velocity): {main}', main=c['title'])
    if data['spare']:
        inside += '; ' + _('запасной: {spare}', spare=', '.join(data['spare']))
    inside += '; ' + (_('вставание (GetUp): этап {n}', n=int(getup['stage']) // 1_000_000) if getup else
                      _('вставания нет'))
    out += ['## ' + _('Кто'), '', kv([
        (_('Кандидат'), candidate + (', ' + _('итерация {n}', n=c['iteration']) if c['iteration'] else ''), package,
         None),
        (_('Ветка и этап'), _('{branch}, этап {steps} шагов ({n} млн)', branch=f"`{c['branch']}`", steps=c['steps'],
                              n=n(c['steps'] / 1e6, 1)), package, None),
        (_('Выгрузка'), f"`{c['export']}`", package, None),
        (_('Пакет; дата сборки'), f"`{data['package']}`; {data['created']}" if data['created'] else None, package,
         _('нет даты в package.json')),
        (_('Контракт'), contract, package, _('нет в package.json')),
        (_('Частота политики'), u(n(1 / c['step_dt'], 0), 'Hz') if c['step_dt'] else None, 'deploy.yaml',
         _('нет step_dt в deploy.yaml пакета')),
        (_('Команды (диапазоны deploy.yaml)'), commands, package, _('нет в package.json')),
        (_('В пакете'), inside, package, None),
        (_('Масса модели (mj_getTotalmass)'), u(n(data['mass'], 3), 'kg'), f'`{MODEL}`', None),
    ]), '']

    # pushes
    seconds = data['push_seconds']
    out += ['## ' + _('Устойчивость к толчкам'), '',
            _('Толчок — горизонтальная сила на таз в течение T: F = M·Δv/T, импульс M·Δv, M = {mass} кг (полная масса '
              'модели). В связке толчок даёт патч {patch}, T = {t} с.', mass=n(data['mass'], 3),
              patch='`unitree_mujoco-push.patch`', t=short(seconds, lang) if seconds else '—'), '',
            f"| {_('Где')} | {_('Положение')} | Δv, {sym('m/s')} | T, {sym('s')} | F, {sym('N')} | "
            f"{_('Импульс, Н·с')} | {_('Падений')} |", '|---|---|---|---|---|---|---|']
    pushes = get('pushes') or []
    for state in ('stand', 'walk'):
        mine = [p for p in pushes if p['state'] == state]
        where = (_('стоя') if state == 'stand' else
                 _('на ходу {n} м/с', n=s(mine[0]['speed'])) if mine and mine[0]['speed'] else _('на ходу'))
        if not mine or mine[0].get('missing'):
            empty.append(_('Толчки, связка: {where}', where=where))
            out.append(f'| {bridge} | {where} |  |  |  |  |  |')
            continue
        for p in mine:
            out.append(f"| {bridge} | {where} | {s(p['dv'])} | {short(p['seconds'], lang)} | {n(p['force'], 0)} | "
                       f"{n(p['impulse'], 1)} | {of(p['falls'])} |")
    tilt = get('push_tilt')
    out += ['', kv([(_('Наибольший наклон корпуса при толчках стоя'),
                     _('{deg}° при Δv {dv} м/с', deg=s(tilt['deg']), dv='–'.join(s(x) for x in tilt['dv'])) if tilt
                     else None, bridge, nothing)]), '']

    # walking
    def drift(w):
        return _('{rate} рад/с; курс за 10 с {deg}°', rate=sp(w['drift']), deg=sp(w['heading'])) if w else None

    def hundred(w):
        e = w and w['extrapolated']
        if not e:
            return None
        low, high = (n(x, 0, True) for x in e['heading'])
        heading = low if low == high else f'{low}…{high}'
        if not e['radius']:
            return f'{heading}°'
        low, high = (n(r, 0 if r >= 10 else 1) for r in e['radius'])
        return _('{deg}°, дуга радиусом {r} м', deg=heading, r=low if low == high else f'{low}–{high}')
    w05, w1 = get('walk05'), get('walk1')
    extrapolation = _('экстраполяция измеренного: курс = увод·100 м / скорость, радиус = скорость / увод; увод '
                      'считается постоянным')
    turn, brake, stop, stand30 = get('turn'), get('brake'), get('side_stop'), get('stand30')
    side = {round(x['speed'].value, 2): x for x in get('side') or []}
    rows = [
        (_('Скорость при команде 0,5 м/с'), u(sp(w05['speed']), 'm/s') if w05 else None, bridge, nothing),
        (_('Скорость при команде 1 м/с'), u(sp(w1['speed']), 'm/s') if w1 else None, bridge, nothing),
        (_('Увод курса без команды поворота, 0,5 м/с'), drift(w05), bridge, nothing),
        (_('Увод курса без команды поворота, 1 м/с'), drift(w1), bridge, nothing),
        (_('Курс через 100 м без поправки, 0,5 м/с (экстраполяция)'), hundred(w05), extrapolation, nothing),
        (_('Курс через 100 м без поправки, 1 м/с (экстраполяция)'), hundred(w1), extrapolation, nothing),
        (_('Снос вбок за 10 с ходьбы 0,5 м/с'), u(sp(get('side10')), 'm') if get('side10') else None, bridge,
         nothing),
        (_('Поворот по команде ±{n} рад/с: на месте', n=s(turn['rate'])) if turn else _('Поворот по команде: на месте'),
         u(f"{s(turn['place'][0])} / {s(turn['place'][1])}", 'rad/s') if turn else None, bridge, nothing),
        (_('Поворот по команде ±{n1} рад/с: на ходу {n2} м/с', n1=s(turn['rate']), n2=s(turn['speed'])) if turn else
         _('Поворот по команде: на ходу'),
         u(f"{s(turn['walking'][0])} / {s(turn['walking'][1])}", 'rad/s') if turn else None, bridge, nothing),
    ]
    for speed, distance in (brake['rows'] if brake else [(None, None)]):
        rows.append((_('Торможение: путь за {n1} с после стопа с {n2} м/с', n1=brake['seconds'], n2=s(speed)) if brake
                     else _('Торможение после стопа'), u(sp(distance), 'm') if distance else None, bridge,
                     nothing))
    for speed in (0.3, 0.5):
        x = side.get(speed)
        rows.append((_('Шаг вбок ±{n} м/с: скорость; увод', n=short(speed, lang)),
                     u(sp(x['speeds']), 'm/s') + '; ' + u(sp(x['drift']), 'rad/s') if x else None, bridge, nothing))
    rows += [
        (_('Стоп после шага вбок ±{a} / ±{b} м/с: сдвиг за {n} с', a=s(stop['rows'][0][0]), b=s(stop['rows'][1][0]),
           n=stop['seconds']) if stop else _('Стоп после шага вбок: сдвиг'),
         u(f"{sp(stop['rows'][0][1])} / {sp(stop['rows'][1][1])}", 'm') if stop else None, bridge, nothing),
        (_('Стояние 30 с: поворот; смещение'),
         f"{sp(stand30['heading'])}°; {u(sp(stand30['offset']), 'cm')}" if stand30 else None, bridge,
         nothing),
        (_('Прогонов в связке; падений'), '{}; {}'.format(*get('runs')) if get('runs') else None, bridge, nothing),
    ]
    out += ['## ' + _('Ходьба'), '',
            _('Увод и курс: «+» — влево, «−» — вправо; увод — после первых 2 с отрезка, курс — за весь отрезок 10 с.'),
            '', kv(rows), '']

    # goal_check
    out += ['## ' + _('Проверка цели ходьбы (goal_check, наша физика)'), '']
    goal, saved = data['goal'], data['goal_file']
    usable = saved if saved and not saved['mismatch'] else None
    if goal or usable:
        check = goal or dict(title='', items=usable['items'])
        shown = localize_journal({'check': check}, lang, Content(data['lab']))['check']
        if goal:
            intro = _('Проверка «{title}» (условий: {n}), детерминированная политика без помех.', title=shown['title'],
                      n=len(shown['items']))
        elif usable['export']:
            intro = _('Проверка из файла {file}: выгрузка {export} (этап {steps}), условий: {n}; детерминированная '
                      'политика без помех.', file=f"`{usable['file']}`", export=f"`{usable['export']}`",
                      steps=usable['stage'], n=len(shown['items']))
        else:
            intro = _('Проверка из файла {file}: этап {steps}, условий: {n}; детерминированная политика без помех.',
                      file=f"`{usable['file']}`", steps=usable['stage'], n=len(shown['items']))
        out += [intro, '', f"| {_('Условие')} | {_('Значение')} | {_('Цель')} | {_('Выполнено')} |",
                '|---|---|---|---|']
        out += [f"| {esc(row['label'])} | {esc(row['value'])} | {esc(row['target'])} | "
                f"{'✓' if row['ok'] else '✗' if row['ok'] is False else ''} |" for row in shown['items']]
    else:
        other = data['goal_other']
        if saved:
            reason = _('файл {file} — проверка другого этапа: {stage}', file=f"`{saved['file']}`",
                       stage=saved['export'] or saved['stage'])
        elif other:
            reason = _('нет сохранённого результата для этого этапа: в «Дневнике» — проверка {branch} @ {n} млн',
                       branch=f"`{other['branch']}`", n=n(int(other['stage']) / 1e6, 1))
        else:
            reason = _('нет сохранённого результата для этого этапа')
        out.append(kv([(_('goal_check: условия цели этапа 1'), None, ours, reason)]))
    out.append('')

    # getting up
    if getup:
        names = dict(ours=_('наша физика'), delay=_('задержка 20 мс'), floor=_('скользкий пол: подошвы 0,4, пол 0,2'),
                     noise=_('шум датчиков и смещение энкодеров'))
        intro = _('Политика вставания — этап {n} ({steps} шагов), выгрузка {export}.',
                  n=int(getup['stage']) // 1_000_000, steps=getup['stage'], export=f"`{getup['export']}`")
        if getup['physics']:
            intro += ' ' + _('Успех — встал за {n1} с, без отсечки скорости суставов и без повторного падения до '
                             'передачи, затем ходьба {title} через {n2} с без падения.', n1=getup['within'],
                             n2=getup['after'], title=getup['title'])
        if getup['starts']:
            st = getup['starts']
            intro += ' ' + _('Старты пробы: спина, живот, бок, сидя — по {n1}, падения {title} — {n2}; всего {n}.',
                             n1=st['each'], n2=st['falls'], title=st['title'], n=st['total'])
        out += ['## ' + _('Вставание'), '', intro, '',
                f"| {_('Где')} | {_('Успех')} | {_('Подъём, с: медиана / p90 / наибольшее')} | "
                f"{_('Повторных падений после подъёма')} |", '|---|---|---|---|']
        for p in getup['physics'] or []:
            rise = ' / '.join(s(x) for x in p['rise']) if p['rise'] else ''
            share = short(100 * p['succeeded'] / p['total'], lang, 1)
            out.append(f"| {_('проба критика без подтяжки: {name}', name=names[p['id']])} | "
                       f"{of((p['succeeded'], p['total']))} ({share}%) | {rise} | "
                       f"{'' if p['refall'] is None else p['refall']} |")
        if not getup['physics']:
            empty.append(_('Вставание: проба критика'))
            out.append(f"| {critic} |  |  |  |")
        g = getup['bridge']
        if g and g['rose']:
            rise = ' / '.join(s(x) for x in g['rise']) if g['rise'] else ''
            out.append(f"| {bridge} | {of(g['rose'])} | {rise} |  |")
            empty.append(_('Вставание, связка: повторных падений после подъёма'))
        else:
            empty.append(_('Вставание: связка'))
            out.append(f'| {bridge} |  |  |  |')
        unbridged = _('этап в связке не гоняли') if getup['bridged'] else nothing
        out += ['', kv([
            (_('Отсечка по скорости суставов (пороги GetUp), проба'), of(getup['cutoff']) if getup['cutoff'] else None,
             critic, nothing),
            (_('Ходьба после подъёма, проба: падений'),
             str(getup['velocity_falls']) if getup['velocity_falls'] is not None else None, critic, nothing),
            (_('Как лёг перед подъёмом, связка'),
             _('на спине {n1}, на животе {n2}', n1=g['lay'][0], n2=g['lay'][1]) if g and g['lay'] else None, bridge,
             nothing if g else unbridged),
            (_('Отсечки (тайм-аут, скорость сустава), связка'), of(g['cutoff']) if g and g['cutoff'] else None, bridge,
             nothing if g else unbridged),
            (_('Ходьба после подъёма, связка: падений'), of(g['velocity']) if g and g['velocity'] else None, bridge,
             nothing if g else unbridged),
        ]), '']

    # reliability
    t, e = data['training'], data['encoders']
    disturbances = []
    if t and t['push']:
        p = t['push']
        disturbances.append(_('толчки силой на таз до Δv {dv} м/с за {t1}–{t2} с: F до {f} Н, импульс до {i} Н·с, раз '
                              'в {a}–{b} с', dv=short(p['dv'], lang), t1=short(p['seconds'][0], lang),
                              t2=short(p['seconds'][1], lang), f=n(data['mass'] * p['dv'] / p['seconds'][0], 0),
                              i=n(data['mass'] * p['dv'], 1), a=short(p['interval'][0], lang),
                              b=short(p['interval'][1], lang)))
    if t and t['arms']:
        disturbances.append(_('руки в случайной позе до ±{r} рад {p}% времени', r=short(t['arms']['radians'], lang),
                              p=short(100 * t['arms']['share'], lang, 1)))
    if t and t['friction']:
        disturbances.append(_('трение суставов моторов ×U(0; {n})', n=short(t['friction'], lang)))
    signed = lambda pair, digits: n(pair[0], digits, True) + '…' + n(pair[1], digits, True)
    delay = get('delay')
    lowcmd = get('lowcmd')
    rows = [
        (_('Оценка этапа со случайными командами и помехами обучения'),
         _('падений {k} из {n}, эпизоды по {s} с', k=t['falls'], n=t['episodes'], s=short(t['seconds'], lang))
         if t and t['episodes'] else None, '; '.join([ours] + disturbances), _('нет оценки этапа в лаборатории')),
        (_('Задержка политика → мотор в обучении'),
         (u('0', 'ms') if t['delay'] == 0 else _('0–{n} мс, случайно на эпизод', n=short(1000 * t['delay'], lang)))
         if t else None, 'task_config', _('нет этапа в лаборатории')),
        (_('Смещение энкодеров до ±{n} рад, как на роботе', n=n(e['bias'], 3)) if e else _('Смещение энкодеров'),
         _('падений {k} из {n}; увод 0,5 м/с {a} рад/с, 1 м/с {b} рад/с; поворот стоя за 30 с {c}°', k=e['falls'],
           n=e['runs'], a=signed(e['w05'], 3), b=signed(e['w1'], 3), c=signed(e['stand'], 1)) if e else None,
         _('{where}, розыгрышей: {n} (reference/enc_bias.py)', where=ours, n=e['draws']) if e else ours,
         _('нет данных пробы в пакете')),
        (_('Задержка в связке: состояние → шаг физики с новой целью'),
         _('медиана {a} мс, 95% — {b} мс, наибольшая {c} мс', a=sp(delay[0]), b=sp(delay[1]), c=sp(delay[2]))
         if delay else None, bridge, nothing),
        (_('Задержка lowstate → lowcmd (часы ВМ)'),
         _('медиана {a} мс, 95% — {b} мс', a=sp(lowcmd[0]), b=sp(lowcmd[1])) if lowcmd else None, bridge, nothing),
        (_('Нижняя граница цели колен (clip deploy.yaml)'), u(n(c['knee'], 3), 'rad') if c['knee'] else None,
         package, _('нет в package.json')),
        (_('Цели вне clip, все 29 суставов (сообщения lowcmd в Velocity)'), of(get('clip')) if get('clip') else None,
         bridge, nothing),
    ]
    out += ['## ' + _('Надёжность, задержка, пределы'), '', kv(rows), '']

    # torque chatter
    ch = data['chatter']
    usable = bool(ch) and not ch['mismatch']
    summary = ch['summary'] if usable else {}
    reference = summary.get('reference') or {}
    settings = ch['settings'] if usable else {}
    cutoffs = {key: settings.get(key, value) for key, value in CUTOFFS.items()}
    label = ''
    if ch:
        label = ch['label'] + (' (' + _('выгрузка') + ')' if ch['export'] else '')
    probe = (_('проба chatter.py ({label}), наша физика (CPU) с шумом датчиков; прогонов {n1}, падений {n2}',
               label=label, n1=summary.get('runs', ''), n2=len(summary.get('falls') or [])) if usable else '')
    reason = _('проба другого этапа: {label}', label=label) if ch else _('проба не приложена (--chatter)')

    gate = settings.get('gate') or {}
    judged = set(gate.get('keys') or ()) if gate.get('factor') is not None else set()
    units = dict(torque_hf_rms_nm=lambda x: u(x, 'N·m'), action_hf_rms_rad=lambda x: u(x, 'rad'))

    def value(key):
        """The worst joint's value and name, Unitree v0's in the same runs and, for a judged key, the threshold."""
        mine, theirs = summary.get(key), reference.get(key)
        if mine is None:
            return None
        unit = units.get(key, lambda x: x)
        text = unit(n(mine, CHATTER[key]))
        joint = (summary.get('worst_joints') or {}).get(key)
        if joint:
            text += f" (`{joint.removesuffix('_joint')}`)"
        if theirs is not None:
            text += '; ' + _('Unitree v0: {n}', n=unit(n(theirs, CHATTER[key])))
            if key in judged:
                text += '; ' + _('порог {n} × v0 = {limit}', n=short(float(gate['factor']), lang),
                                 limit=unit(n(float(gate['factor']) * theirs, CHATTER[key])))
        return text
    verdict = None
    if judged and summary.get('ok') is not None:
        verdict = _('худший сустав: {keys} не больше {n} × Unitree v0, без падений; {verdict}',
                    keys=', '.join(f'`{k}`' for k in gate['keys']), n=short(float(gate['factor']), lang),
                    verdict=_('проходит') if summary['ok'] else _('не проходит'))
    mark = lambda key, label: label + (' — ' + _('судится') if key in judged else '')
    action, torque = (short(cutoffs[key], lang) for key in ('action_cutoff', 'torque_cutoff'))
    out += ['## ' + _('Дрожание моментов'), '', kv([
        (mark('torque_hf_rms_nm', _('СКЗ момента мотора выше {n} Гц, худший сустав', n=torque)),
         value('torque_hf_rms_nm'), probe, reason),
        (mark('action_hf_rms_rad', _('СКЗ цели ПД выше {n} Гц, худший сустав', n=action)),
         value('action_hf_rms_rad'), probe, reason),
        (mark('action_hf_share', _('Доля дисперсии цели ПД выше {n} Гц, худший сустав', n=action)),
         value('action_hf_share'), probe, reason),
        (mark('torque_hf_share', _('Доля дисперсии момента мотора выше {n} Гц, худший сустав', n=torque)),
         value('torque_hf_share'), probe, reason),
        (_('СКЗ момента / пик мотора (без модели нагрева)'), value('torque_rms_peak_share'), probe, reason),
        (_('Среднее |Δa| за шаг управления'), value('mean_abs_delta_action'), probe, reason),
        (_('Порог и итог'), verdict, probe, reason),
    ]), '']

    # motor heating
    out += ['## ' + _('Нагрев моторов'), '', kv([
        (_('Нагрев моторов при удержании груза'), None, '',
         _('измеряется на роботе ({tool} читает температуру моторов)', tool='`record_check.py`'))]), '']

    # conditions and sources
    what = dict(package=_('кандидат, контракт, диапазоны команд, колени, дата и коммит сборки'),
                reference=_('таблицы связки и вставания, как их записал package.py'),
                mass=_('масса M (mj_getTotalmass)'), push=_('длительность толчка T в связке (kSeconds)'),
                encoders=_('проба смещения энкодеров'), training=_('оценка этапа и помехи обучения (task_config)'),
                goal=_('результат goal_check'), chatter=_('проба дрожания моментов'))
    # the section's name stays as the Russian document has it, marked so on the English and Chinese pages
    note = '' if lang == 'ru' else ' ' + _('(раздел русского документа docs/sim2sim.md)')
    origin = (_('Числа связки — из docs/sim2sim.md, раздел «{section}»{note}, через REFERENCE.md пакета.',
                section=c['sim2sim'], note=note) if c['sim2sim'] else
              _('Числа связки — из docs/sim2sim.md через REFERENCE.md пакета.'))
    commits = _('Коммит сборки пакета {build}{dirty}; коммит лаборатории при выпуске паспорта {lab}.',
                build=f"`{data['build_commit']}`" if data['build_commit'] else '—',
                dirty=' ' + _('(с незакоммиченными правками источников)') if data['build_dirty'] else '',
                lab=f"`{data['lab_commit']}`" if data['lab_commit'] else '—')
    out += ['## ' + _('Условия и источники'), '',
            '- ' + _('Всё — симуляция: наша физика MuJoCo (CPU, профиль real_robot: моторы Unitree, наблюдения как в '
                     'deploy.yaml) и связка unitree_mujoco + g1_ctrl в WSL2 (sim2sim). Данных настоящего робота пока '
                     'нет.'), '- ' + origin, '- ' + commits, '',
            f"| {_('Файл')} | {_('Что взято')} | md5 |", '|---|---|---|']
    out += [f"| `{path}` | {what[kind]} | {f'`{digest}`' if digest else ''} |"
            for path, kind, digest in data['sources']]
    return '\n'.join(out) + '\n', empty


def write(package, out=None, langs=LANGUAGES, chatter=None, lab=LAB, goal_check=None):
    """PASSPORT.<lang>.md for each language: ([(path, labels of its empty fields)], warnings)."""
    data = collect(package, lab, chatter, goal_check)
    folder = Path(out or package)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for lang in langs:
        text, empty = render(data, lang)
        path = folder / NAME.format(lang=lang)
        path.write_text(text, encoding='utf-8', newline='\n')
        written.append((path, empty))
    return written, data['warnings']


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Паспорт кандидата: одна страница на русском, английском и китайском '
                                                 'из чисел пакета для робота')
    parser.add_argument('package', type=Path, help='папка пакета scripts/robot/package.py (runs/humanoid_lab/robot/…)')
    parser.add_argument('--out', type=Path, help='куда писать PASSPORT.<язык>.md (по умолчанию — в папку пакета)')
    parser.add_argument('--chatter', type=Path, help='JSON пробы scripts/lab/probes/chatter.py --json этого этапа')
    parser.add_argument('--goal-check', type=Path,
                        help='JSON, который python -m humanoid_sim.goal_check <ветка>:<этап> печатает последней '
                             'строкой (вместо проверки из «Дневника»)')
    parser.add_argument('--lang', default=','.join(LANGUAGES), help='языки через запятую (ru,en,zh)')
    parser.add_argument('--lab', type=Path, default=LAB, help='папка лаборатории (оценка этапа, «Дневник»)')
    args = parser.parse_args()
    langs = [x.strip() for x in args.lang.split(',') if x.strip()]
    if not langs or any(x not in LANGUAGES for x in langs):
        fail(f'языки: {", ".join(LANGUAGES)}, а не {args.lang!r}')
    for name, path in (('пробы', args.chatter), ('goal_check', args.goal_check)):
        if path is not None and not path.is_file():
            fail(f'нет файла {name} {path}')
    written, warnings = write(args.package, args.out, langs, args.chatter, args.lab, args.goal_check)
    for line in warnings:
        print(f'ВНИМАНИЕ: {line}')
    for path, empty in written:
        print(f'{path}: пустых полей {len(empty)}' + (': ' + '; '.join(empty) if empty and path.name.endswith('.ru.md')
                                                       else ''))


if __name__ == '__main__':
    main()
