"""Robot package for the first G1 run on the hoist (CPU only): runs/humanoid_lab/robot/<name>/.

Usage: .venv\\Scripts\\python.exe scripts\\robot\\package.py [--name NAME]

Puts together, for the person at the robot:
  policy/      candidate 7 (main) and 6 (spare) as exported (policy.onnx + deploy.yaml), after checking they share one
               contract (deploy.yaml equal but for comments; ONNX obs [1, 490] -> actions [1, 29]; locked wrists
               exactly 0; actions within +-10; command ranges within the floor's -0.5..1 / +-0.5 / +-1)
  patches/     every scripts/sim2sim/patches/g1_ctrl-*.patch, applied by apply_patches.sh in the order
               scripts/sim2sim/setup.sh applies them (others after, by name), on unitree_rl_mjlab at setup.sh's commit
  tools/       the DDS recorder (scripts/sim2sim/probe), analyze.py and record_check.py (checks a robot recording)
  reference/   the hoist, FixStand and encoder probes (scripts/robot/reference, <probe>_<stage>.json per candidate),
               REFERENCE.md: those tables and the sim2sim numbers of docs/sim2sim.md (each candidate's section)
  PROTOCOL.md  docs/robot-first-run.md;  README.md;  install_policy.sh;  package.json;  MD5SUMS (md5sum -c)
Every copied file is checked against its source by md5. A rerun rebuilds the folder: built aside, then swapped in
(only a folder this script made, by package.json, is replaced).
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1]
EXPORTS = SIM / 'runs' / 'humanoid_lab' / 'export'
ROBOT = SIM / 'runs' / 'humanoid_lab' / 'robot'
SIM2SIM = SIM / 'scripts' / 'sim2sim'
PROTOCOL = SIM / 'docs' / 'robot-first-run.md'
SIM2SIM_DOC = SIM / 'docs' / 'sim2sim.md'
GIT = SIM / '.tools' / 'git' / 'cmd' / 'git.exe'
KIND = 'g1-robot-package'
LOCKED_WRISTS = (20, 21, 27, 28)
KNEES = (3, 9)
KNEE_FLOOR = 0.061

# Lead's decision, 23.10 (docs/lab-knowledge.md, iteration 24, «Решение»): candidate 7 for the hoist and the floor,
# candidate 6 the spare. reference: the stage's probe data in scripts/robot/reference (<probe>_<reference>.json);
# sim2sim: its section of docs/sim2sim.md (SIM2SIM_TABLE's numbers).
CANDIDATES = [
    dict(label='n7', title='№7', role='основной', export='exp-c362df8e578148d0-6368231424', reference='6368',
         sim2sim='## Этап 6368 (итерация 24)',
         about='итерация 24: стопы сталкиваются капсулами Unitree с их трением; ветка от №6 через итерации 22–23'),
    dict(label='n6', title='№6', role='запасной', export='exp-8ce0cbd70d0a4f1d-6039109632', reference='6039',
         sim2sim='## Этап 6039 (итерация 20)', about='итерация 20'),
]
SIM2SIM_SECTIONS = tuple(c['sim2sim'] for c in CANDIDATES)
REFERENCE_PROBES = ('hoist', 'fixstand_air', 'enc_bias')
# docs/robot-first-run.md, item 9: on the floor only exports with `export_g1 --command-ranges -0.5 1 0.5 1`
COMMAND_RANGES = dict(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
# docs/sim2sim.md, the two sections above (unitree_mujoco + g1_ctrl in WSL2; 22 and 17 runs): (what, №7, №6)
SIM2SIM_TABLE = [
    ('Прогонов; падений', '22; 0', '17; 0'),
    ('Задержка: состояние → первый шаг физики с новой целью, медиана / 95% / макс',
     '6 (в трёх прогонах 4) / 10 (в двух 8) / 12–22 мс', '6 / 10 (в одном прогоне 8) / 12–16 мс'),
    ('lowstate пришёл → lowcmd с ответом (часы ВМ): медиана; 95%', '2,21–3,05 мс; 5,11–7,47 мс',
     '2,38–3,20 мс; 5,68–7,60 мс'),
    ('Прямо 0,5 м/с: скорость; увод после 2 с; курс за 10 с', '0,51 м/с; −0,001 / −0,004 / +0,001 рад/с; '
     '−1,7…−3,1°', '0,41–0,42 м/с; +0,001 / +0,006 / +0,010 рад/с; +0,4…+4,5°'),
    ('За 10 с ходьбы 0,5 м/с: вперёд; вбок', '4,84–4,93 м; −0,02…+0,05 м', '4,01–4,08 м; −0,11…0,00 м'),
    ('Прямо 1 м/с: скорость; увод; курс за 10 с', '0,97 м/с; +0,028 / +0,031 рад/с; +11,9° / +12,6°',
     '0,90 м/с; +0,026 / +0,020 рад/с; +10,0° / +9,1°'),
    ('Шаг вбок ±0,3 м/с: скорость; увод при шаге влево; вправо', '0,26–0,27 м/с; −0,001 / +0,004; −0,004 / −0,005',
     '0,27–0,28 м/с; −0,036 / −0,038; +0,009 / +0,008'),
    ('Поворот ±0,5 рад/с: на месте; на ходу 0,5 м/с', '+0,505 / −0,453; +0,500 / −0,461 рад/с',
     '+0,476 / −0,512; +0,498 / −0,516 рад/с'),
    ('Толчки стоя (сила на таз 0,1 с), падений: Δv 0,5 / 1 / 1,25 / 1,5 м/с', '0 из 8 / 0 из 8 / 0 из 20 / 0 из 8',
     '0 из 8 / 0 из 8 / 0 из 8 / 0 из 8'),
    ('Наклон при толчках стоя Δv 1,25–1,5, наибольший', '11°', '13°'),
    ('Толчки на ходу 0,5 м/с, падений: Δv 0,5 / 1 м/с', '0 из 4 / 0 из 4', '0 из 4 / 0 из 4'),
    ('Стопы при боковых толчках стоя Δv 1–1,5 (поза из логов в нашей модели с капсулами Unitree): капсулы '
     'перекрылись бы; ноги накрест', '12 из 18, на 1–2 такта; 4 из 18, до 1,7 см', '12 из 12; 4 из 12, до 4,1 см'),
    ('Колено стоя, левое / правое; в упоре; цель на границе clip',
     '+2,3…+2,7° / +4,7…+5,6°; 0%; левое 100%, правое 0% (цель +0,11)', '+1,7…+2,7° / +2,9…+3,5°; 0%; 100%'),
    ('Колено на ходу 0,5 м/с, левое / правое; наименьший угол',
     '+28,1…+28,5° / +26,7…+26,8°; +0,15 рад (в повороте +0,033)',
     '+23,2…+23,4° / +25,6…+25,8°; +0,028 рад (в повороте +0,016)'),
    ('Колено на ходу 1 м/с, левое / правое', '+36,6° / +37,0°', '+29,0° / +31,3…+31,4°'),
    ('Цель колена ниже clip (гонка g1_ctrl)', '0 сообщений в 22 прогонах (с патчем clip-race)',
     '0 сообщений в 17 прогонах (до патча clip-race)'),
    ('Стоя после ленты, рысканье по секундам 1–5, рад/с (медиана)', '+0,003 +0,001 −0,005 −0,005 −0,005',
     '−0,010 +0,003 +0,007 +0,019 +0,022'),
    ('Стояние 30 с: поворот; смещение', '−5,8° / −5,9° / −6,3° (3 прогона); 2–3 см; без шага',
     '+12,7°; 6 см; один подшаг без команды на 19-й с'),
    ('Торможение за 3 с от стопа с 0,5 / 1 м/с', '0,23–0,26 / 0,37–0,50 м', '0,17–0,19 / 0,42–0,45 м'),
    ('После стопа с 0,5 м/с: смещение с 1-й по 5-ю с', '8,1–9,0 см (таз уходит вперёд над стопами, без шага)',
     '3,6–4,2 см'),
    ('После стопа с 1 м/с: смещение с 1-й по 5-ю с; шаг после 1-й с',
     '12,5 / 3,2 см (во втором — шаг назад через 1,5 с); 1 из 17 стопов', '5,7–6,1 см; 0 из 17'),
    ('Таз на ходу не ниже: 0,5 / 1 м/с', '0,768 / 0,747 м', '0,772 / 0,743 м'),
    ('Запястья; входы в Velocity с нулевой целью', 'q20 = q27 = 0 («ни одного»); 0', 'q20 = q27 = 0 («ни одного»); 0'),
    ('Патч «падение → Passive» в этих прогонах', '0 срабатываний в 22', 'прогоны до патча (проверен отдельно: 3 из 3 '
     'падений → Passive, 0 ложных в 7)'),
]
HOIST_PLANS = dict(air5='в воздухе, стопы 5 см над полом', air15='в воздухе, стопы 15 см над полом',
                   half='стопы на полу, трос натянут', lower='опускание из Velocity',
                   slack='FixStand и Velocity на полу, провис троса 5 см', floor='без троса')
HOIST_SEGMENTS = dict(stand='стоять (команда 0)', walk='идти 0,5 м/с', hang='висеть (команда 0)',
                      pay='трос отпускают 2 см/с', stop='стоп')
PHYSICS = dict(ours='наша', um='unitree_mujoco')


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fail(message):
    raise SystemExit(f'package.py: {message}')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def copy(source, target, copied):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied[target] = source


def setup_value(name):
    """A pinned value from scripts/sim2sim/setup.sh (NAME=value)."""
    m = re.search(rf'^{name}=(\S+)', (SIM2SIM / 'setup.sh').read_text(encoding='utf-8'), re.M)
    return m.group(1) if m else None


def patch_order(names):
    """setup.sh's order of `apply_patch unitree_rl_mjlab <patch>`, then the patches it does not name, by name."""
    text = (SIM2SIM / 'setup.sh').read_text(encoding='utf-8')
    order = [n for n in re.findall(r'apply_patch unitree_rl_mjlab (g1_ctrl-\S+\.patch)', text) if n in names]
    return order + sorted(n for n in names if n not in order), [n for n in names if n not in order]


def patch_lines(path):
    """spdlog info / warn lines a patch adds (first line of each call)."""
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^\+.*spdlog::(info|warn)\("([^"]*)"', line)
        if m:
            out.append(dict(kind=m.group(1), text=m.group(2)))
    return out


def deploy_body(path):
    return [line for line in path.read_text(encoding='utf-8').splitlines() if not line.lstrip().startswith('#')]


def check_export(candidate):
    folder = EXPORTS / candidate['export']
    for name in ('policy.onnx', 'deploy.yaml'):
        if not (folder / name).exists():
            fail(f'нет {folder / name}')
    deploy = yaml.safe_load((folder / 'deploy.yaml').read_text(encoding='utf-8'))
    first = (folder / 'deploy.yaml').read_text(encoding='utf-8').splitlines()[0]
    m = re.search(r'stage_(\d+)', first)
    action = deploy['actions']['JointPositionAction']
    clip = np.array(action['clip'], float)
    terms = deploy['observations']
    obs_size = sum(len(t['scale']) * int(t.get('history_length', 1)) for t in terms.values())
    problems = []
    if obs_size != 490:
        problems.append(f'вход наблюдений {obs_size}, а не 490')
    if float(deploy['step_dt']) != 0.02:
        problems.append(f"step_dt {deploy['step_dt']}")
    if not np.allclose(clip[list(KNEES), 0], KNEE_FLOOR):
        problems.append(f'нижняя граница clip колен {clip[list(KNEES), 0].tolist()}, а не {KNEE_FLOOR}')
    ranges = deploy.get('commands', {}).get('base_velocity', {}).get('ranges', {})
    for key, (lo, hi) in COMMAND_RANGES.items():
        r = ranges.get(key)
        if r is None or float(r[0]) < lo or float(r[1]) > hi:
            problems.append(f'диапазон команды {key} {r}, а для пола нужен в [{lo:g}, {hi:g}] '
                            '(export_g1 --command-ranges -0.5 1 0.5 1)')
    onnx = onnx_contract(folder / 'policy.onnx')
    if onnx['input'] != ['obs', [1, 490]] or onnx['output'] != ['actions', [1, 29]]:
        problems.append(f"ONNX {onnx['input']} -> {onnx['output']}")
    if not onnx['wrists_zero']:
        problems.append('ONNX: выходы запястий 20, 21, 27, 28 не равны 0')
    if not onnx['finite'] or onnx['max_abs'] > 10:
        problems.append(f"ONNX: выход вне ±10 (макс {onnx['max_abs']})")
    if problems:
        fail(f"{candidate['export']}: " + '; '.join(problems))
    comments = [line for line in (folder / 'deploy.yaml').read_text(encoding='utf-8').splitlines()
                if line.lstrip().startswith('#')]
    return dict(folder=folder, stage=m.group(1).lstrip('0') if m else None, onnx=onnx,
                knee_clip=clip[list(KNEES), 0].tolist(), comments=comments,
                command_ranges={k: ranges[k] for k in COMMAND_RANGES})


def onnx_contract(path, n=64):
    """Input / output names and shapes; on n random observations (scales 0.1-10): locked wrists exactly 0,
    finite actions within the action limit."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    i, o = session.get_inputs()[0], session.get_outputs()[0]
    rng = np.random.default_rng(0)
    obs = (rng.normal(size=(n, 490)) * rng.choice([0.1, 1., 10.], size=(n, 1))).astype(np.float32)
    out = np.concatenate([session.run(None, {i.name: x[None]})[0] for x in obs])
    return dict(input=[i.name, list(i.shape)], output=[o.name, list(o.shape)], samples=n,
                wrists_zero=bool(np.all(out[:, list(LOCKED_WRISTS)] == 0)), finite=bool(np.all(np.isfinite(out))),
                max_abs=float(np.abs(out).max()))


def git_state():
    if not GIT.exists():
        return None
    run = lambda *a: subprocess.run([str(GIT), '-C', str(SIM), *a], capture_output=True, text=True).stdout
    paths = ['scripts/sim2sim/patches', 'scripts/sim2sim/probe', 'scripts/sim2sim/analyze.py', 'scripts/robot',
             'docs/robot-first-run.md']
    # porcelain lines are 'XY path': no strip(), its leading blank is the X of the first line
    return dict(commit=run('rev-parse', 'HEAD').strip(),
                uncommitted=[line for line in run('status', '--porcelain', '--', *paths).splitlines() if line.strip()])


def ru(text):
    """Decimal commas and minus signs for the Russian tables (numbers only: 0.05 -> 0,05, -0.1 -> −0,1)."""
    text = re.sub(r'(?<=\d)\.(?=\d)', ',', text)
    return re.sub(r'(?<![\w-])-(?=\d)', '−', text)


def hoist_table(data):
    lines = ['| Физика | Опыт | Отрезок | Шагов | Скорость ног p95 / макс, рад/с | Σ\\|τ\\| ног среднее / макс, Н·м | '
             '\\|Δa\\| среднее / макс | Суставов у предела | Касание себя макс, Н | Трос среднее / макс, доля веса | '
             'Наклон макс, ° | Таз мин, м | Доля с опорой о пол | Падение |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in data:
        fix = f"FixStand: таз {r['fix']['z']:.3f} м, наклон {r['fix']['tilt']:.1f}°"
        for k, s in enumerate(r['segs']):
            fell = '—' if r['fell_at'] is None or k < len(r['segs']) - 1 else f"через {r['fell_at']:.2f} с"
            plan = f"{HOIST_PLANS[r['plan']]} ({fix})" if k == 0 else ''
            lines.append(
                f"| {PHYSICS[r['physics']] if k == 0 else ''} | {plan} | {HOIST_SEGMENTS[s['name']]} | {s['steps']} | "
                f"{s['qv_leg_p95']:.2f} / {s['qv_leg_max']:.2f} | {s['tau_leg_mean']:.1f} / {s['tau_leg_max']:.1f} | "
                f"{s['da']:.4f} / {s['da_max']:.3f} | {s['at_lim'] * 29:.1f} из 29 | {s['self_max']:.1f} | "
                f"{s['rope_mean']:.2f} / {s['rope_max']:.2f} | {s['tilt_max']:.1f} | {s['pelz_min']:.3f} | "
                f"{s['floor_contact_share']:.2f} | {fell} |")
    return '\n'.join(lines)


def fixstand_table(data):
    lines = ['| Физика | Секунда FixStand | \\|q − q_fix\\| макс, рад | Хуже всех | Гравитация (среднее) | '
             'Гироскоп среднее, рад/с | \\|гироскоп\\| макс, рад/с |', '|---|---|---|---|---|---|---|']
    for r in data:
        for s in r['seconds']:
            g, w = s['gravity'], s['gyro_mean']
            lines.append(f"| {PHYSICS[r['physics']]} | {s['second']}–{s['second'] + 1} | {s['max_err']:.4f} | "
                         + ', '.join(f'{name} {v:.3f}' for _, name, v in s['worst'][:2])
                         + f" | ({g[0]:+.3f}; {g[1]:+.3f}; {g[2]:+.3f}) | ({w[0]:+.3f}; {w[1]:+.3f}; {w[2]:+.3f}) | "
                         f"{s['gyro_max']:.3f} |")
    return '\n'.join(lines)


def encoder_table(data):
    base = {r['plan']: r for r in data if r['draw'] < 0}
    lines = ['| Модель смещения | Опыт | Розыгрышей | Падений | Среднее | σ | Мин | Макс |', '|---|---|---|---|---|---|---|---|']
    plans = dict(w05=('прямо 0,5 м/с, увод, рад/с', 'yaw_rate', '{:+.4f}', 'увод при 0,5 м/с {} рад/с'),
                 w1=('прямо 1 м/с, увод, рад/с', 'yaw_rate', '{:+.4f}', 'при 1 м/с {} рад/с'),
                 stand=('стоять 30 с, поворот, °', 'heading_deg', '{:+.1f}', 'поворот стоя за 30 с {}°'))
    models = dict(calib='энкодер смещён и в наблюдении, и в ПД (как на роботе)', obs='только в наблюдении (как в обучении)')
    for model, title in models.items():
        for plan, (what, key, fmt, _) in plans.items():
            rs = [r for r in data if r['model'] == model and r['plan'] == plan and r['draw'] >= 0]
            v = np.array([r[key] for r in rs if r['fell'] is None])
            lines.append(f"| {title} | {what} | {len(rs)} | {sum(r['fell'] is not None for r in rs)} | "
                         f"{fmt.format(v.mean())} | {fmt.format(v.std(ddof=1)).lstrip('+')} | {fmt.format(v.min())} | "
                         f"{fmt.format(v.max())} |")
    no_bias = ', '.join(plans[p][3].format(plans[p][2].format(base[p][plans[p][1]])) for p in plans if p in base)
    return '\n'.join(lines), no_bias


def span(values, fmt='{:.1f}'):
    """'a–b' over the values as formatted (one number when both ends print the same)."""
    values = list(values)
    a, b = fmt.format(min(values)), fmt.format(max(values))
    return a if a == b else f'{a}–{b}'


def segments(data, plans, names):
    return [s for r in data if r['plan'] in plans for s in r['segs'] if s['name'] in names]


def hoist_summary(data):
    """The critic's conclusions from the hoist probe (iteration 21), with the numbers of this stage's data."""
    air = segments(data, ('air5', 'air15'), ('stand',)) + segments(data, ('lower',), ('hang',))
    walk_air = segments(data, ('air5', 'air15'), ('walk',))
    lower = [r for r in data if r['plan'] == 'lower']
    landed = segments(data, ('lower',), ('pay', 'stand'))
    stand = segments(data, ('lower',), ('stand',))
    slack = [r for r in data if r['plan'] == 'slack']
    slack_stand = segments(data, ('slack',), ('stand',))
    floor = [r for r in data if r['plan'] == 'floor']
    walks = segments(data, ('lower', 'slack'), ('walk',))  # from the anchor over the start: xy is the distance
    self_air = max(s['self_max'] for s in air)
    falls = sum(r['fell_at'] is not None for r in lower)
    return [
        f"в воздухе при команде 0 робот спокоен: скорость ног p95 {span(s['qv_leg_p95'] for s in air)} рад/с (на "
        f"входе в Velocity до {max(s['qv_leg_max'] for s in air):.1f}), |Δa| {span((s['da'] for s in air), '{:.4f}')}, "
        + ('касаний себя нет' if self_air == 0 else f'касание себя до {self_air:.0f} Н')
        + f", целей на границе `clip` {span(s['at_lim'] * 29 for s in air)} из 29;",
        f"ходьба в воздухе: ноги до {max(s['qv_leg_max'] for s in walk_air):.0f} рад/с, нога о ногу "
        f"{span((s['self_max'] for s in walk_air), '{:.0f}')} Н — в воздухе не ходить;",
        "опускание тросом 2 см/с из Velocity — " + ('встал, 0 падений' if not falls else f'падений {falls} из {len(lower)}')
        + f"; наклон до {max(s['tilt_max'] for s in landed):.1f}°, стоя Σ|τ| ног "
        f"{span((s['tau_leg_mean'] for s in stand), '{:.0f}')} Н·м в среднем (пик {max(s['tau_leg_max'] for s in stand):.0f}"
        f"), целей на границе `clip` {span(s['at_lim'] * 29 for s in stand)} из 29;",
        f"FixStand на полу с провисом 5 см: таз {span((r['fix']['z'] for r in slack), '{:.3f}')} м, наклон "
        f"{span(r['fix']['tilt'] for r in slack)}°; за 10 с стояния в Velocity таз уходит на "
        f"{span((math.hypot(*s['xy']) for s in slack_stand), '{:.2f}')} м (шаг-спасение), Σ|τ| ног до "
        f"{span((s['tau_leg_max'] for s in slack_stand), '{:.0f}')} Н·м — FixStand и вход в Velocity только на "
        f"натянутом тросе; без троса FixStand роняет робота (наклон к концу FixStand "
        f"{span((r['fix']['tilt'] for r in floor), '{:.0f}')}°);",
        f"ходьба от неподвижной точки: в {span((math.hypot(*s['xy']) for s in walks), '{:.2f}')} м от точки под "
        f"подвесом трос натягивается до {span(s['rope_max'] for s in walks)} веса, наклон на ходу "
        f"{span((s['tilt_max'] for s in walks), '{:.0f}')}° — тележка или провис ≥ x²/(2L) + 5 см.",
    ]


def encoder_note(data):
    """The spread under the robot's model (the encoder biased in the observation and in the PD alike)."""
    def values(plan, key):
        return np.array([r[key] for r in data if r['model'] == 'calib' and r['plan'] == plan and r['draw'] >= 0
                         and r['fell'] is None])
    stand = values('stand', 'heading_deg')
    walk = [values(p, 'yaw_rate').std(ddof=1) for p in ('w05', 'w1')]
    return (f"поворот стоя за 30 с разбросан с σ ≈ {stand.std(ddof=1):.1f}° ({stand.min():+.1f}…{stand.max():+.1f}°), "
            f"увод при ходьбе — с σ {span(walk, '{:.3f}')} рад/с")


def load_reference(candidate, probe):
    return json.loads((HERE / 'reference' / f"{probe}_{candidate['reference']}.json").read_text(encoding='utf-8'))


def reference_md(info, sources):
    cands = info['candidates']
    name = lambda c: f"{c['title']} ({c['reference']}, {c['role']})"
    sections = ' и '.join(f"«{c['sim2sim'][3:]}» ({c['title']})" for c in cands)
    sim_rows = '\n'.join('| ' + ' | '.join(row) + ' |' for row in SIM2SIM_TABLE)
    hoist = []
    for c in cands:
        data = load_reference(c, 'hoist')
        hoist.append(f"### {name(c)}\n\nДанные `reference/hoist_{c['reference']}.json`, этап {c['stage']}.\n\n"
                     f"{ru(hoist_table(data))}\n\nЧто из этого следует (выводы критика, итерация 21; числа — из "
                     "таблицы выше):\n" + '\n'.join(f'- {ru(line)}' for line in hoist_summary(data)))
    fix = load_reference(cands[0], 'fixstand_air')
    strip = lambda data: [{k: v for k, v in r.items() if k != 'stage'} for r in data]
    fix_rest = []
    for c in cands[1:]:
        other = load_reference(c, 'fixstand_air')
        if strip(other) == strip(fix):
            fix_rest.append(f"У {c['title']} (`reference/fixstand_air_{c['reference']}.json`) таблица та же до "
                            'последнего знака: FixStand от политики не зависит.')
        else:
            fix_rest.append(f"### {name(c)}\n\n{ru(fixstand_table(other))}")
    window = [s['max_err'] for r in fix for s in r['seconds'] if s['second'] >= 3]
    encoders = []
    for c in cands:
        data = load_reference(c, 'enc_bias')
        table, no_bias = (ru(x) for x in encoder_table(data))
        encoders.append(f"### {name(c)}\n\nДанные `reference/enc_bias_{c['reference']}.json`, этап {c['stage']}. Без "
                        f"смещения: {no_bias}.\n\n{table}\n\nНа роботе работает первая модель: "
                        f"{ru(encoder_note(data))}.")
    hoist, fix_rest, encoders = ('\n\n'.join(x) for x in (hoist, fix_rest, encoders))
    fix_rest = fix_rest + '\n\n' if fix_rest else ''
    columns = ' | '.join(f"{c['title']} ({c['reference']})" for c in cands)
    about = ', '.join(f"{c['title']} — {c['role']} ({c['about']})" for c in cands)
    return f"""# Эталоны для первого запуска

Собрано `scripts/robot/package.py` {info['created']}. Числа ниже — то, с чем сравнивать робота; пороги остановки — в
`PROTOCOL.md`.

Кандидаты: {about}.

## Связка Unitree в симуляции (sim2sim)

`unitree_mujoco` + `g1_ctrl` (unitree_rl_mjlab deploy) в WSL2, тот же `policy.onnx` и `deploy.yaml`, что в этом пакете.
Лента симулятора, а не трос: FixStand и вход в Velocity на ленте 0,6 м (держит ~95% веса), потом лента снята. В
`unitree_mujoco` у стоп нет капсул: стопы проходят друг сквозь друга, и столкновение стоп связка не видит (строка «Стопы
при боковых толчках» — по позе из логов в нашей модели с капсулами Unitree).

Источник — `docs/sim2sim.md`, разделы {sections}.

| Что | {columns} |
|---|{'---|' * len(cands)}
{sim_rows}

Задержка «состояние → шаг физики» на роботе не мерится (нет времени физики); на роботе мерится «lowstate пришёл →
lowcmd с ответом» (`tools/record_check.py`). Тайминги WSL неровные: период политики там 13–27 мс, на роботе ждём
ровнее (порог протокола — 20 ± 2 мс в 95% тактов).

## Подвес в симуляции

`reference/hoist.py` (проба критика, итерация 21; для каждого кандидата свой файл данных): CPU, детерминированная
политика, удержания курса нет. Трос — только натяжение, 20 кН/м и 1 кН·с/м, к торсу, точка крепления 3 м над стартом.
FixStand 2 с (усиления `g1_ctrl`), затем Velocity. Физика «наша» — профиль `real_robot` (кривые и трение моторов
Unitree) с моделью стоп из обучения этапа (у №7 стопы сталкиваются капсулами Unitree), «unitree_mujoco» — копия физики
их симулятора (`scripts/lab/probes/um_probe.py`, капсул нет). «Суставов у предела» — сколько из 29 целей стоят на
границе `clip` (в среднем по шагам; 2 — оба колена на 0,061). Σ|τ| ног — сумма модулей моментов 12 моторов ног.
«Трос» — натяжение в долях веса (35,1 кг).

{hoist}

## FixStand на подвесе (из Passive)

`reference/fixstand_air.py`, данные `reference/fixstand_air_{cands[0]['reference']}.json`: тот же трос, стопы 5 см над
полом; Passive (kp 0, kd 3) 2 с, затем FixStand: 2 с переход из позы Passive в позу FixStand и 4 с удержания. Робот на
тросе качается маятником (горизонтально трос не гасит), отсюда гироскоп до 0,1 рад/с и среднее за секунду до 0,02.

{ru(fixstand_table(fix))}

{fix_rest}Отсюда окно «покоя» в `record_check.py`: с 4-й по 6-ю секунду FixStand (|q − q_fix| там {ru(span(window, '{:.3f}'))}
рад, порог 0,05), гироскоп — среднее за окно; если |гироскоп| p95 > 0,05 рад/с, робот качался и замер надо повторить.

## Калибровка энкодеров

`reference/enc_bias.py` (проба критика, итерация 21): наша физика, постоянное смещение каждого энкодера U(±0,015 рад)
(как `ENCODER_BIAS` в обучении), 12 розыгрышей; увод — после 2 с, поворот стоя — за 30 с. Две модели: энкодер смещён и
в наблюдении, и в ПД (как на роботе: энкодер один), или только в наблюдении (как в обучении).

{encoders}

Поворот стоя поэтому мерить на самом роботе и поправлять стиком; увод при ходьбе от калибровки меняется мало.

## Источники

| Файл | md5 |
|---|---|
""" + '\n'.join(f'| `{name}` | `{digest}` |' for name, digest in sources) + '\n'


def readme_md(info):
    cands = info['candidates']
    main, spare = cands[0], cands[1:]
    rows = '\n'.join(
        f"| `policy/{c['slot']}` | {c['title']} ({c['role']}; {c['about']}) | `{c['export']}` | {c['stage']} | "
        f"`{c['md5']['policy.onnx']}` | `{c['md5']['deploy.yaml']}` |" for c in cands)
    headers = '\n\n'.join(f"`policy/{c['slot']}/deploy.yaml`:\n\n```\n" + '\n'.join(c['comments']) + '\n```'
                          for c in cands)
    ranges = ru(', '.join(f"`{key}` {r[0]:g}…{r[1]:g}" for key, r in main['command_ranges'].items()))
    title = f"кандидат {main['title']} ({main['role']})" + ''.join(f" и {c['title']} ({c['role']})" for c in spare)
    spare_labels = ' или '.join(c['label'] for c in spare)
    patches = '\n'.join(
        f"{k}. `patches/{p['name']}`"
        + ''.join(f"\n   - в выводе `g1_ctrl` {'при старте или входе в Velocity' if x['kind'] == 'info' else 'при срабатывании'}: "
                  f"`{x['text']}`" for x in p['log_lines'])
        for k, p in enumerate(info['patches'], 1))
    extra = ''
    if info['patches_not_in_setup']:
        extra = ('\nНе в `scripts/sim2sim/setup.sh` (в связке, возможно, не проверены; порядок — по имени, после '
                 'остальных): ' + ', '.join(f'`{n}`' for n in info['patches_not_in_setup']) + '.\n')
    dirty = ''
    if info['git'] and info['git']['uncommitted']:
        dirty = ('\nИсточники пакета с незакоммиченными правками на момент сборки: '
                 + ', '.join(f"`{line[3:]}`" for line in info['git']['uncommitted']) + '.\n')
    commit = info['git']['commit'] if info['git'] else 'нет git'
    return f"""# Пакет для робота: G1, {title}

Собран {info['created']} скриптом `scripts/robot/package.py` (лаборатория, коммит `{commit}`). Перед сеансом прочитать
`PROTOCOL.md` (копия `docs/robot-first-run.md`): там шаги, что смотреть и когда останавливаться. Эталоны —
`REFERENCE.md`.{dirty}

## Состав

| Путь | Что |
|---|---|
| `policy/<слот>/policy.onnx`, `deploy.yaml` | политика и её разъём для `g1_ctrl` (выгрузка `humanoid_sim.export_g1`) |
| `patches/g1_ctrl-*.patch`, `apply_patches.sh` | правки лаборатории к `unitree_rl_mjlab/deploy` и их наложение |
| `install_policy.sh` | кладёт выбранную политику туда, откуда её берёт `g1_ctrl` |
| `tools/sim2sim_probe/` | регистратор DDS (`rt/lowstate`, `rt/lowcmd`) — исходник, собирать на машине у робота |
| `tools/record_check.py`, `tools/analyze.py` | разбор записи с робота по порогам протокола (Python: numpy, onnxruntime, pyyaml) |
| `reference/` | данные проб подвеса, FixStand и калибровки (из них `REFERENCE.md`) |
| `PROTOCOL.md`, `REFERENCE.md`, `package.json` | протокол, эталоны, опись сборки |
| `MD5SUMS` | md5 всех файлов пакета |

Целостность: `md5sum -c MD5SUMS` в папке пакета (Linux) — все строки `OK`.

## Политики

Один разъём у всех: `deploy.yaml` совпадают построчно, кроме комментариев (шапки ниже); ONNX `obs` [1, 490] →
`actions` [1, 29]; на {main['onnx']['samples']} случайных наблюдениях выходы запястий 20, 21, 27, 28 ровно 0, действия
в ±10; нижняя граница `clip` колен (SDK 3, 9) — 0,061. md5 сверены с файлами выгрузки в `runs/humanoid_lab/export/`.

Диапазоны команд: {ranges}. Для пола — не шире `export_g1 --command-ranges -0.5 1 0.5 1` (протокол, п. 9
«Безопасности»): сборка это проверяет.

| Слот | Кандидат | Выгрузка | Этап, шагов | md5 `policy.onnx` | md5 `deploy.yaml` |
|---|---|---|---|---|---|
{rows}

{headers}

## Патчи g1_ctrl

Сделаны на `unitree_rl_mjlab` `{info['unitree_rl_mjlab']}` (как в `scripts/sim2sim/setup.sh`; `unitree_sdk2`
`{info['unitree_sdk2']}`). Порядок наложения — как в `setup.sh`:

{patches}
{extra}
На машине, где собирают `g1_ctrl` для робота:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git && git -C unitree_rl_mjlab checkout {info['unitree_rl_mjlab']}
bash apply_patches.sh ./unitree_rl_mjlab --build    # наложить (уже наложенные пропускает) и собрать g1_ctrl
```

Вручную — то же самое: в корне `unitree_rl_mjlab` для каждого патча по порядку `git apply --check <патч>`, затем
`git apply <патч>`; потом `cd deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make -j`. Если патч «не
накладывается», в дереве уже есть его прежняя версия или другой коммит: взять чистый клон на коммите выше.

## Политика на место

```bash
bash install_policy.sh ./unitree_rl_mjlab {main['label']}    # или {spare_labels} — запасной
```

Скрипт кладёт `policy.onnx` в `deploy/robots/g1/config/policy/velocity/v9_lab/exported/`, `deploy.yaml` — в
`…/v9_lab/params/`, сверяет md5 и проверяет, что `g1_ctrl` возьмёт именно `v9_lab` (он берёт последнюю по имени
папку с `exported` в `config/policy/velocity`). В выводе `g1_ctrl` при старте: `Policy directory: …/velocity/v9_lab`.

## Запись и разбор

Регистратор: `cd tools/sim2sim_probe && mkdir -p build && cd build && cmake .. && make` (нужен `unitree_sdk2` в
`/usr/local`, как для `g1_ctrl`). Запуск: `./sim2sim_probe <папка> <секунд> <сетевой интерфейс> 0` (Ctrl+C — записать
и выйти раньше). Разбор: `python3 tools/record_check.py <папка> --policy policy/{main['slot']} --log <папка>/g1_ctrl.log`.
Подробно — `PROTOCOL.md`, раздел «Запись и разбор».
"""


APPLY_SH = """#!/bin/bash
# Патчи лаборатории к g1_ctrl (unitree_rl_mjlab deploy) в порядке scripts/sim2sim/setup.sh.
# Запуск: bash apply_patches.sh <папка unitree_rl_mjlab> [--build]
# Уже наложенный патч пропускается; не накладывающийся останавливает скрипт (git apply --check до наложения,
# так что наполовину ничего не накладывается). --build: cmake и make в deploy/robots/g1 (g1_ctrl).
set -eu
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO="${{1:?запуск: bash apply_patches.sh <папка unitree_rl_mjlab> [--build]}}"
BUILD="${{2:-}}"
EXPECTED={commit}
PATCHES=({patches})
g() {{ git -c 'safe.directory=*' -C "$REPO" "$@"; }}
head=$(g rev-parse HEAD)
[ "$head" = "$EXPECTED" ] || echo "ВНИМАНИЕ: unitree_rl_mjlab на $head, патчи сделаны на $EXPECTED"
for p in "${{PATCHES[@]}}"; do
  f="$HERE/patches/$p"
  if g apply --reverse --check "$f" 2>/dev/null; then
    echo "$p: уже наложен"
  elif g apply --check "$f"; then
    g apply "$f"
    echo "$p: наложен"
  else
    echo "$p: не накладывается (в дереве прежняя версия патча или другой коммит, см. README.md)"
    exit 1
  fi
done
if [ "$BUILD" = --build ]; then
  cd "$REPO/deploy/robots/g1" && mkdir -p build && cd build && cmake .. && make -j"$(nproc)" && ls -la g1_ctrl
fi
"""

INSTALL_SH = """#!/bin/bash
# Кладёт политику пакета туда, откуда её берёт g1_ctrl: <unitree_rl_mjlab>/deploy/robots/g1/config/policy/velocity/v9_lab
# Запуск: bash install_policy.sh <папка unitree_rl_mjlab> {labels}
set -eu
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO="${{1:?запуск: bash install_policy.sh <папка unitree_rl_mjlab> {labels}}}"
case "${{2:-}}" in
{cases}
  *) echo "какую политику: {labels}"; exit 1 ;;
esac
SRC="$HERE/policy/$SLOT"
VEL="$REPO/deploy/robots/g1/config/policy/velocity"
DST="$VEL/v9_lab"
[ -d "$VEL" ] || {{ echo "нет $VEL: это не unitree_rl_mjlab?"; exit 1; }}
mkdir -p "$DST/exported" "$DST/params"
cp "$SRC/policy.onnx" "$DST/exported/policy.onnx"
cp "$SRC/deploy.yaml" "$DST/params/deploy.yaml"
echo "$SLOT (пакет $(basename "$HERE"))" > "$DST/PACKAGE.txt"
for pair in "policy.onnx:exported/policy.onnx" "deploy.yaml:params/deploy.yaml"; do
  a=$(md5sum < "$SRC/${{pair%%:*}}" | cut -d' ' -f1)
  b=$(md5sum < "$DST/${{pair#*:}}" | cut -d' ' -f1)
  [ "$a" = "$b" ] || {{ echo "md5 не совпал: $DST/${{pair#*:}}"; exit 1; }}
  echo "$DST/${{pair#*:}}: md5 $b"
done
[ -d "$VEL/exported" ] && echo "ВНИМАНИЕ: в $VEL есть exported/ — g1_ctrl возьмёт его, а не v9_lab"
last=$(for d in "$VEL"/*/; do [ -d "$d/exported" ] && basename "$d"; done | LC_ALL=C sort | tail -1)
if [ "$last" = v9_lab ]; then
  echo "g1_ctrl возьмёт v9_lab ($SLOT): в его выводе при старте должно быть «Policy directory: $DST»"
else
  echo "ВНИМАНИЕ: g1_ctrl возьмёт $last (последняя по имени папка с exported), а не v9_lab"
  exit 1
fi
"""


def build(name):
    info = dict(kind=KIND, name=name, created=time.strftime('%Y-%m-%d %H:%M:%S'), git=git_state(),
                unitree_rl_mjlab=setup_value('RL_MJLAB_SHA'), unitree_sdk2=setup_value('SDK2_SHA'))
    if not PROTOCOL.exists():
        fail(f'нет {PROTOCOL}')
    text = SIM2SIM_DOC.read_text(encoding='utf-8')
    missing = [s for s in SIM2SIM_SECTIONS if s not in text]
    if missing:
        print(f'ВНИМАНИЕ: в docs/sim2sim.md нет разделов {missing}: числа sim2sim в REFERENCE.md сверить вручную')
    # the probes, then each candidate's data (scripts/robot/reference/<probe>_<stage>.json)
    reference = [f'{p}.py' for p in REFERENCE_PROBES]
    reference += [f"{p}_{c['reference']}.json" for c in CANDIDATES for p in REFERENCE_PROBES]
    missing = [f for f in reference if not (HERE / 'reference' / f).exists()]
    if missing:
        fail(f"нет данных проб в {HERE / 'reference'}: {', '.join(missing)} (scripts/robot/reference/<проба>.py "
             '<этап> <проба>_<этап>.json)')
    # policies: one contract
    checked = [check_export(c) for c in CANDIDATES]
    bodies = [deploy_body(x['folder'] / 'deploy.yaml') for x in checked]
    if any(b != bodies[0] for b in bodies[1:]):
        fail('deploy.yaml кандидатов различаются не только комментариями: разъём разный')
    if any(x['onnx']['input'] != checked[0]['onnx']['input'] for x in checked):
        fail('вход ONNX кандидатов различается')
    # patches
    found = {p.name: p for p in sorted(SIM2SIM.joinpath('patches').glob('g1_ctrl-*.patch'))}
    if not found:
        fail('в scripts/sim2sim/patches нет g1_ctrl-*.patch')
    order, not_in_setup = patch_order(list(found))
    info['patches_not_in_setup'] = not_in_setup

    ROBOT.mkdir(parents=True, exist_ok=True)
    final, pending, old = ROBOT / name, ROBOT / f'.pending-{name}', ROBOT / f'.old-{name}'
    if final.exists():
        manifest = final / 'package.json'
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')).get('kind') != KIND:
            fail(f'{final} есть, но это не пакет package.py (нет package.json): не трогаю')
    for leftover in (pending, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    copied = {}
    info['candidates'] = []
    for c, x in zip(CANDIDATES, checked):
        slot = f"{c['label']}-{x['stage'] or c['export']}"
        for f in ('policy.onnx', 'deploy.yaml'):
            copy(x['folder'] / f, pending / 'policy' / slot / f, copied)
        info['candidates'].append(dict(c, slot=slot, stage=x['stage'], onnx=x['onnx'], knee_clip=x['knee_clip'],
                                       comments=x['comments'], command_ranges=x['command_ranges'],
                                       md5={f: md5(x['folder'] / f) for f in ('policy.onnx', 'deploy.yaml')}))
    info['patches'] = []
    for n in order:
        copy(found[n], pending / 'patches' / n, copied)
        info['patches'].append(dict(name=n, md5=md5(found[n]), log_lines=patch_lines(found[n])))
    for f in ('sim2sim_probe.cpp', 'CMakeLists.txt'):
        copy(SIM2SIM / 'probe' / f, pending / 'tools' / 'sim2sim_probe' / f, copied)
    copy(SIM2SIM / 'analyze.py', pending / 'tools' / 'analyze.py', copied)
    copy(HERE / 'record_check.py', pending / 'tools' / 'record_check.py', copied)
    for f in reference:
        copy(HERE / 'reference' / f, pending / 'reference' / f, copied)
    copy(PROTOCOL, pending / 'PROTOCOL.md', copied)
    sources = [(str(p.relative_to(SIM)).replace('\\', '/'), md5(p)) for p in
               [SIM2SIM_DOC] + [HERE / 'reference' / f for f in reference]]
    write(pending / 'REFERENCE.md', reference_md(info, sources))
    write(pending / 'README.md', readme_md(info))
    write(pending / 'apply_patches.sh', APPLY_SH.format(commit=info['unitree_rl_mjlab'], patches=' '.join(order)))
    labels = '|'.join(c['label'] for c in info['candidates'])
    cases = '\n'.join(f"  {c['label']}) SLOT={c['slot']} ;;" for c in info['candidates'])
    write(pending / 'install_policy.sh', INSTALL_SH.format(labels=labels, cases=cases))
    info['sources'] = {str(t.relative_to(pending)).replace('\\', '/'): dict(source=str(s.relative_to(SIM)).replace(
        '\\', '/'), md5=md5(s)) for t, s in copied.items()}
    write(pending / 'package.json', json.dumps(info, ensure_ascii=False, indent=1, default=str) + '\n')

    # md5: every copy against its source, then MD5SUMS over the whole package and read back
    mismatched = [str(t) for t, s in copied.items() if md5(t) != md5(s)]
    if mismatched:
        fail('копия не совпала с источником по md5: ' + ', '.join(mismatched))
    files = sorted(p for p in pending.rglob('*') if p.is_file())
    sums = [(md5(p), str(p.relative_to(pending)).replace('\\', '/')) for p in files]
    write(pending / 'MD5SUMS', ''.join(f'{d}  {rel}\n' for d, rel in sums))
    listed = {}
    for line in (pending / 'MD5SUMS').read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        listed[rel] = digest
    on_disk = {str(p.relative_to(pending)).replace('\\', '/') for p in pending.rglob('*') if p.is_file()}
    if set(listed) != on_disk - {'MD5SUMS'} or any(md5(pending / rel) != d for rel, d in listed.items()):
        fail('MD5SUMS не сошёлся с файлами пакета')

    if final.exists():
        final.rename(old)
    pending.rename(final)
    if old.exists():
        shutil.rmtree(old)
    return final, info, copied, sums


def main():
    parser = argparse.ArgumentParser(description='Пакет для первого запуска G1 на подвесе')
    parser.add_argument('--name', default=f"g1-{CANDIDATES[0]['label']}-{CANDIDATES[0]['export']}",
                        help='имя папки в runs/humanoid_lab/robot')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9._-]+', args.name) or args.name.startswith('.'):
        fail(f'имя папки: латиница, цифры, . _ -, не с точки ({args.name!r})')
    final, info, copied, sums = build(args.name)
    print(f'пакет: {final}')
    print('политики (md5 выгрузки = md5 в пакете):')
    for c in info['candidates']:
        for f in ('policy.onnx', 'deploy.yaml'):
            target = final / 'policy' / c['slot'] / f
            print(f"  {c['title']} {c['slot']}/{f}: {c['md5'][f]} = {md5(target)}"
                  f" {'совпал' if c['md5'][f] == md5(target) else 'НЕ СОВПАЛ'}")
    print('патчи по порядку: ' + ', '.join(p['name'] for p in info['patches'])
          + (f"; не из setup.sh: {', '.join(info['patches_not_in_setup'])}" if info['patches_not_in_setup'] else ''))
    print(f'файлов {len(sums) + 1}, MD5SUMS сверен; копий с источником сверено {len(copied)}')
    if info['git'] and info['git']['uncommitted']:
        print('незакоммиченные источники: ' + ', '.join(line[3:] for line in info['git']['uncommitted']))


if __name__ == '__main__':
    main()
