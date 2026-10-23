"""CPU only. stage_select.py <stage dir | export dir> [<stage>…] [--fresh] [--workers 8] [--out x.json]

The stage selection rule of iteration 18 (the critic, 23.10; his data in lab-critic/it18), its pushes replaced in
iteration 20 (the critic, 23.10; lab-critic/it20: sweep.py, analyze.py, stand8*.json, walk8*.json). Stages of one
branch are independent draws (neighbours correlate 0.03-0.5 in drift, 0.05 in pushes), so a stage is judged on its
own:
  1. the conditions written down in advance (goal_check; not here);
  2. step 2: the stage in 7 physics variants through the Unitree bridge copy (stand_turn.run): ours and ours +6 ms,
     motor friction x0.25 and x0.25 +6 ms, no motor model, Unitree and Unitree +6 ms (+6 ms is the bridge: the PD on
     a 2 ms older state, the target 3 substeps late). Runs: stand_turn's 'bridge' group (40: ours and Unitree +6 ms,
     regular and real timing windows, our reset starts), walk 0.5 / walk 1.0 / side step for the other five (15).
     Passes with 0 falls, the worst |yaw rate| < 0.05 rad/s walking 0.5 and 1.0 m/s and stepping left and right, the
     stop right after a side step moving < 0.3 m. Then pushes in the bridge copy (bridge_kick.run: Unitree physics,
     PD on a 2 ms older state, 6 ms delay, a force M*dv/0.1 s on the pelvis), 8 directions 45 degrees apart from the
     pelvis heading: standing, 10 moments (ticks 0-457 past 3 s after the band, 0-0.91 s) x dv 0.8 / 0.9 / 1.0 /
     1.1 / 1.25 / 1.5 m/s (480 runs), and walking 0.5 m/s, 3 s into the walk, 6 phases of the 0.6 s gait cycle x dv
     0.75 / 1.0 / 1.25 (144). They pass if standing up to 1.25 m/s at most 12 of 400 fall (3 %), no cell direction x
     dv up to 1.25 has 4 falls of 10, under half of each direction falls at 1.25 (1.5 is printed, not judged), and
     walking at most 1 of 144 falls. Why (the critic): the rule before, 16 pushes of one dv (4 directions x 4
     moments 0-0.23 s), was ~8 distinct states (3 of the 4 moments left the same pelvis speed within 0.01), and the
     outcome is not monotonic in dv (a middling push gets a hesitant half step and a fall, a strong one a decisive
     step): 5968920576 "failed" 1.1 (2 of 16) holding 1.25 (0 of 16), and candidate 4's hole forward (1.0: 6 of
     10) comes from moments 0.3-0.9 s the old ones did not reach. Also printed, with no limit: the turn over the 30 s
     stand of the bridge copy (the 'bridge' group's stand30 run, Unitree +6 ms: the critic's stand30.py um6), and
     since iteration 23 whether the feet pass through each other (foot_overlap.py: Unitree's foot capsules, the share
     of control steps with an overlap while both soles are below 2 cm, side steps ±0.5 / ±1 m/s in our physics and
     sideways pushes 1.0 / 1.25 m/s in the bridge copy; 20 more runs).
     Since iteration 24 (the critic, lab-critic/it24/shock.py, now foot_shock.py) the pushes are judged in the bridge
     copy with the feet colliding as in Unitree's training model ('Unitree +6 мс + капсулы стоп': um_probe physics
     'um_caps', task_config foot_collision's capsules with condim 3, friction 0.6, priority 1): on the robot a graze
     is a trip, and the stages of iterations 22-23 fall in 40-80 % of sideways pushes with the capsules, none without.
     Same rule, same pushes; the same 624 without the capsules ('Unitree +6 мс без капсул', the pushes of the rule
     before, comparable with earlier outputs: summary 'pushes_sim2sim') are printed as the sim2sim forecast
     (unitree_mujoco's feet pass through each other), not judged. Also judged there: side steps at ±1 m/s (stand 5
     s, left 10 s, stop 3 s, right 10 s, stop; the bridge procedure and three reset starts), 0 falls; their sideways
     speed, drift and stop move are printed. 1327 runs a stage (was 699), about 8 min on 8 workers;
  3. --fresh: one check of the winner on cases step 2 did not see, in the bridge copy (Unitree +6 ms): other timing
     windows (side step at offsets 5-61, walks at 13; step 2 has 0 and 0-53), delay 4 and 8 ms, stop-start, a right
     step alone (19 runs, no pushes since iteration 20: step 2 covers them). Same limits; the 4 / 8 ms runs and
     stop-start are judged on falls and the stop only, their drift is printed (the delay itself moves it: 5631836160
     stepping left -0.037 at 6 ms, -0.051 at 8 ms).
An export_g1 folder is judged as exported (export_g1.ExportedPolicy: its policy.onnx with the stand trim, in the
environment of its stage), as in stand_turn. Prints per stage a table by physics variant, the 30 s turn, the standing
pushes by direction x dv, the walking pushes and the verdict, then the share of stages that pass (the measure of a
recipe). Every run is deterministic (seed 73000): the same stage gives the same numbers, run for run equal to the
critic's files (the 624 pushes of each stage below: falls, peak speed and lowest pelvis bit for bit as in his
stand8*.json / walk8*.json; the 30 s turn as his stand30*.json um6). 8 workers (23.10, the trainer on the GPU, 24
logical cores): ~4 min a stage for step 2 (679 runs, 624 of them pushes, ~1800 s of processes), ~12 s for step 3
(4 stages with --fresh: 990 s).
The stages of the critic's check (iteration 20, exp-8ce0cbd70d0a4f1d, and candidate 4, exp-902b8958ee504780 @
5631836160), step 2:
  5968920576 standing 8 of 400 (worst backwards 1.0: 3 of 10), walking 0 of 144, turn 30 s +4.5 deg: passes;
  5631836160 standing 45 of 400 (forward 1.0: 6 of 10; 1.25 forward and backwards 10 of 10, right 8 of 10), walking
  8 of 144, turn +21.9 deg: fails on the pushes only;
  5988974592, 6039109632 0 of 400 and 0 of 144, turn +18.2 / +14.6 deg: pass (step 3: 5988974592 fails, the stop
  after a side step 0.35 m in another timing window).
Iteration 18 (the critic's runs of 25 stages, the pushes before): step 2 passed 5631836160 only.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge_kick as kick  # noqa: E402
import foot_overlap as feet  # noqa: E402
import stand_turn as st  # noqa: E402
import um_probe as um  # noqa: E402

STEPPER = kick.PushStepper.__bases__[0]
um.Stepper = STEPPER  # bridge_kick puts its push stepper in on import; only push runs use it (work)
DIRS8 = {'a%03d' % (45 * i): (math.cos(math.radians(45 * i)), math.sin(math.radians(45 * i))) for i in range(8)}
kick.DIRS = DIRS8  # the critic's 8 (lab-critic/it20/sweep.py), in the pelvis heading: 0 = forward, 90 = left
DIR_LABELS = dict(a000='вперёд', a045='вперёд-влево', a090='влево', a135='назад-влево', a180='назад',
                  a225='назад-вправо', a270='вправо', a315='вперёд-вправо')

DRIFT_LIMIT = .05  # rad/s, worst |yaw rate| walking or stepping sideways with no turn command
STOP_LIMIT = .3  # m, the move in the stop right after a side step (the falls of iteration 18 came 1.5-2.6 s into it)
STAND_MAGS = (.8, .9, 1., 1.1, 1.25, 1.5)  # m/s, dv of the standing pushes
STAND_OFFSETS = (0, 37, 71, 113, 151, 197, 251, 307, 379, 457)  # ticks of 2 ms past 3 s after the band
WALK_MAGS = (.75, 1., 1.25)  # m/s, dv of the pushes walking 0.5 m/s
WALK_OFFSETS = (0, 50, 100, 150, 200, 250)  # ticks past 3 s into the walk: 6 phases of the 0.6 s gait cycle
PUSHES = (('stand', STAND_MAGS, STAND_OFFSETS), ('walk', WALK_MAGS, WALK_OFFSETS))
JUDGED_MAG = 1.25  # standing pushes up to this dv are judged; stronger ones are printed
STAND_FALLS = 12  # most falls of the 400 judged standing pushes (3 %)
CELL_FALLS = 4  # falls of the 10 moments in one direction x dv cell (dv < JUDGED_MAG: a systematic hole in the
# moderate pushes; at JUDGED_MAG itself, the edge of the margin, HALF judges a direction) that fail a stage
HALF = .5  # share of falls at JUDGED_MAG in one direction that fails a stage
WALK_FALLS = 1  # most falls of the 144 walking pushes
SIX_MS = dict(stale=True, delay=3)
VARIANTS = (  # name, label, stand_turn case fields
    ('ours', 'наша', dict(physics='ours', stale=False, delay=0)),
    ('ours_6ms', 'наша +6 мс', dict(physics='ours', **SIX_MS)),
    ('fric', 'трение моторов x0.25', dict(physics='ours', extra='fric0.25', stale=False, delay=0)),
    ('fric_6ms', 'трение x0.25 +6 мс', dict(physics='ours', extra='fric0.25', **SIX_MS)),
    ('nomotor', 'без моторной модели', dict(physics='nomotor', stale=False, delay=0)),
    ('um', 'Unitree', dict(physics='um', stale=False, delay=0)),
    ('um_6ms', 'Unitree +6 мс (связка)', dict(physics='um', **SIX_MS)),
)
BRIDGE = ('ours_6ms', 'um_6ms')  # stand_turn's 'bridge' group runs these two
PLANS = ('side', 'walk', 'walk1')  # the other five: side step left then right, walk 0.5, walk 1.0
FRESH = (  # group, label, judged on drift
    ('windows', 'другие окна времени', True),
    ('right', 'шаг вправо', True),
    ('stopstart', 'стоп-старт', False),
    ('delay4', 'задержка 4 мс', False),
    ('delay8', 'задержка 8 мс', False),
)
DRIFT = (('w05', '0.5 м/с'), ('w1', '1 м/с'), ('L', 'влево'), ('R', 'вправо'))
# Since iteration 24 the pushes and the ±1 m/s side check are judged in the Unitree bridge copy with the feet colliding
# as in Unitree's training model (um_probe 'um_caps': task_config foot_collision's capsules, condim 3, friction 0.6):
# on the robot a foot grazing the other is a trip, and without the capsules the feet pass through each other (the
# critic, lab-critic/it24: the same stages fall in 40-80 % of sideways pushes with them, none without). The same
# pushes without the capsules ('um', as unitree_mujoco, whose feet pass through each other: the sim2sim forecast, and
# the numbers of the rule before) are printed, not judged.
PUSH_PHYSICS = 'um_caps'
FORECAST_PHYSICS = 'um'
PHYSICS_LABELS = {PUSH_PHYSICS: 'Unitree +6 мс + капсулы стоп', FORECAST_PHYSICS: 'Unitree +6 мс без капсул'}
SIDE_CHECK = 'side1'  # stand_turn plan: stand, 1 m/s left 10 s, stop, 1 m/s right 10 s, stop; judged on falls
SIDE_CHECK_SEGMENTS = (('left1', 'влево'), ('right1', 'вправо'))


def push_jobs(stage):
    """The pushes of step 2 (lab-critic/it20/sweep.py): bridge_kick.run jobs, standing and walking 0.5 m/s, in the
    Unitree bridge copy with the foot capsules (judged) and without them (printed)."""
    return [('push', dict(name=label(stage), path=str(stage), left=False, phys=phys, dir=d, mag=m, offset=o,
                          plan=plan)) for phys in (PUSH_PHYSICS, FORECAST_PHYSICS)
            for plan, mags, offsets in PUSHES for d in DIRS8 for m in mags for o in offsets]


def side_check_jobs(stage):
    """The ±1 m/s side check of step 2 in the bridge copy with the foot capsules: the bridge procedure (FixStand on the
    band) and three of our reset starts (gait phase 0, 15, 30), as the 'bridge' group's side step."""
    fields = dict(physics=PUSH_PHYSICS, plan=SIDE_CHECK, stage=str(stage), **SIX_MS)
    return [st.case(**fields)] + [st.case(start='reset', phase=15 * k, seed=73000 + k, **fields) for k in range(3)]


def select_jobs(stage):
    """Step 2 of one stage: [(kind, job)], kind 'run' (stand_turn.run case: the 7 physics variants, then the side
    check), 'push' (bridge_kick.run job) or 'feet' (foot_overlap.run job, printed only)."""
    jobs = [('run', case) for case in st.grid(str(stage), 'bridge')]
    for name, _, fields in VARIANTS:
        if name not in BRIDGE:
            jobs += [('run', st.case(plan=plan, stage=str(stage), **fields)) for plan in PLANS]
    jobs += [('run', case) for case in side_check_jobs(stage)]
    return jobs + push_jobs(stage) + [('feet', job) for job in feet.jobs(stage)]


def fresh_jobs(stage):
    """Step 3 of one stage, the critic's fresh check (lab-critic/it18/hold_make.py)."""
    bridge = dict(physics='um', **SIX_MS)
    cases = [st.case(plan='side', timing='real', which=0, offset=o, **bridge) for o in (5, 17, 29, 43, 61)]
    cases += [st.case(plan='walk', timing='real', which=w, offset=13, **bridge) for w in range(6)]
    cases += [st.case(plan='walk1', timing='real', which=w, offset=13, **bridge) for w in range(2)]
    cases += [st.case(plan=plan, physics='um', stale=True, delay=d) for d in (2, 4) for plan in ('side', 'walk1')]
    cases += [st.case(plan=plan, **bridge) for plan in ('stopstart', 'sideR')]
    return [('run', dict(case, stage=str(stage))) for case in cases]


def variant_of(case):
    key = (case['physics'], case.get('extra'), bool(case.get('stale')), int(case.get('delay', 0)))
    for name, _, fields in VARIANTS:
        if key == (fields['physics'], fields.get('extra'), fields['stale'], fields['delay']):
            return name
    raise ValueError(f'not one of the 7 physics variants: {case}')


def fresh_group(case):
    if case.get('timing') == 'real':
        return 'windows'
    if case['plan'] in ('sideR', 'stopstart'):
        return 'right' if case['plan'] == 'sideR' else 'stopstart'
    if case['physics'] != 'um' or not case.get('stale') or case['delay'] not in (2, 4):
        raise ValueError(f'not a case of the fresh check: {case}')
    return 'delay%d' % (2 * case['delay'])


def measures(result):
    """One stand_turn.run result: yaw rates walking ('w05', 'w1') and stepping ('L', 'R'), speeds ('v05', 'v1'),
    the move in a stop right after a side step ('stop') and in the stops of stop-start ('brake'). A run that fell
    gives none (its fall is counted)."""
    out = {key: [] for key in ('w05', 'w1', 'L', 'R', 'v05', 'v1', 'stop', 'brake')}
    if result['fell_at'] is not None:
        return out
    plan, prev = result['case']['plan'], None
    for seg in result['segs']:
        name = seg['name']
        if not seg.get('missing'):
            if plan == 'stopstart':  # 3 s walks: no drift
                if name == 'stop':
                    out['brake'].append(seg['disp'])
            elif name == 'walk':
                key = 'w1' if seg['cmd'][0] > .75 else 'w05'
                out[key].append(seg['yaw_rate'])
                out['v' + key[1:]].append(seg['fwd'])
            elif name in ('left', 'right'):
                out[name[0].upper()].append(seg['yaw_rate'])
            elif name == 'stop' and prev in ('left', 'right'):
                out['stop'].append(seg['disp'])
        prev = name
    return out


def fall_of(result):
    c = result['case']
    real = c.get('timing') == 'real'
    where = c['plan'] + (' окно %s/%s' % (c.get('which', 0), c.get('offset', 0)) if real else '')
    where += ' старт reset' if c.get('start') == 'reset' else ''
    return where + (' %.1f с' % result['fell_at'] if result['fell_at'] >= 0 else '')


def row(results):
    """Table row of a group of stand_turn.run results: worst (largest |.|) yaw rates, largest stop moves, speeds."""
    got = {}
    for r in results:
        for key, values in measures(r).items():
            got.setdefault(key, []).extend(values)
    worst = {key: (max(got.get(key, []), key=abs) if got.get(key) else None) for key, _ in DRIFT}
    return dict(runs=len(results), falls=[fall_of(r) for r in results if r['fell_at'] is not None], drift=worst,
                stop=max(got.get('stop', []), default=None), brake=max(got.get('brake', []), default=None),
                speed={key: (float(np.mean(got[key])) if got.get(key) else None) for key in ('v05', 'v1')})


def judge(rows, labels, drift_rows):
    """Reasons a stage does not pass ([] = passes). rows: name -> row; drift_rows: the names judged on drift."""
    reasons = []
    falls = ['%s: %s' % (labels[name], f) for name, r in rows.items() for f in r['falls']]
    if falls:
        reasons.append('падений %d (%s)' % (len(falls), '; '.join(falls)))
    for key, what in DRIFT:
        values = [(abs(rows[name]['drift'][key]), rows[name]['drift'][key], name) for name in drift_rows
                  if rows[name]['drift'][key] is not None]
        if not values:
            reasons.append('увод %s не измерен' % what)
            continue
        size, value, name = max(values)
        if size >= DRIFT_LIMIT:
            reasons.append('увод %s %+.3f, |.| >= %.2f (%s)' % (what, value, DRIFT_LIMIT, labels[name]))
    stops = [(r['stop'], name) for name, r in rows.items() if r['stop'] is not None]
    if not stops:
        reasons.append('сдвиг после стопа за шагом вбок не измерен')
    elif max(stops)[0] >= STOP_LIMIT:
        reasons.append('сдвиг после стопа за шагом вбок %.2f м >= %.1f (%s)' % (
            max(stops)[0], STOP_LIMIT, labels[max(stops)[1]]))
    return reasons


def dv(mag):
    """1.0 -> '1.0', 1.25 -> '1.25'."""
    text = '%.2f' % mag
    return text[:-1] if text.endswith('0') else text


def push_table(results, plan, mags, physics=PUSH_PHYSICS):
    """bridge_kick.run results of one plan in one physics: falls and runs by direction, one entry per dv of `mags`,
    and the falls that came before the push ('early', counted among the falls)."""
    falls, runs, early = {d: [0] * len(mags) for d in DIRS8}, {d: [0] * len(mags) for d in DIRS8}, 0
    for r in results:
        job = r['job']
        if job['plan'] != plan or job['phys'] != physics:
            continue
        k = [k for k, m in enumerate(mags) if abs(m - job['mag']) < 1e-6]
        if not k:
            raise ValueError(f'a push of dv {job["mag"]} is not one of {mags}')
        runs[job['dir']][k[0]] += 1
        falls[job['dir']][k[0]] += r['fell'] is not None
        early += bool(r.get('fell_before'))
    return dict(mags=list(mags), falls=falls, runs=runs, early=early)


def judged(table):
    """Indices of the judged dv of a standing push table and the falls and runs over them."""
    ks = [k for k, m in enumerate(table['mags']) if m <= JUDGED_MAG + 1e-9]
    return (ks, sum(table['falls'][d][k] for d in DIRS8 for k in ks),
            sum(table['runs'][d][k] for d in DIRS8 for k in ks))


def push_reasons(stand, walk):
    """Reasons the pushes of a stage do not pass ([] = pass): stand and walk are push_table()s."""
    reasons = []
    for table, what, offsets in ((stand, 'стоя', STAND_OFFSETS), (walk, 'на ходу', WALK_OFFSETS)):
        got = sum(sum(n) for n in table['runs'].values())
        if any(n != len(offsets) for runs in table['runs'].values() for n in runs):
            reasons.append('толчков %s %d, нужно %d (по %d в клетке)' % (
                what, got, len(DIRS8) * len(table['mags']) * len(offsets), len(offsets)))
    if reasons:
        return reasons
    ks, falls, runs = judged(stand)
    if falls > STAND_FALLS:
        reasons.append('толчки стоя до %s м/с: %d из %d, можно до %d' % (dv(JUDGED_MAG), falls, runs, STAND_FALLS))
    cells = ['%s %s — %d' % (DIR_LABELS[d], dv(stand['mags'][k]), stand['falls'][d][k])
             for d in DIRS8 for k in ks
             if stand['mags'][k] < JUDGED_MAG - 1e-9 and stand['falls'][d][k] >= CELL_FALLS]
    if cells:
        reasons.append('клетки стоя с %d из %d и больше: %s' % (CELL_FALLS, len(STAND_OFFSETS), ', '.join(cells)))
    top = stand['mags'].index(JUDGED_MAG)
    half = ['%s %d из %d' % (DIR_LABELS[d], stand['falls'][d][top], stand['runs'][d][top]) for d in DIRS8
            if stand['falls'][d][top] >= HALF * stand['runs'][d][top]]
    if half:
        reasons.append('толчки стоя %s м/с: падает половина и больше (%s)' % (dv(JUDGED_MAG), ', '.join(half)))
    falls = sum(sum(n) for n in walk['falls'].values())
    if falls > WALK_FALLS:
        reasons.append('толчки на ходу 0.5 м/с: %d из %d, можно до %d (%s)' % (
            falls, sum(sum(n) for n in walk['runs'].values()), WALK_FALLS, walk_where(walk)))
    return reasons


def walk_where(walk):
    """Where the walking pushes fell: 'вперёд-влево 3, влево 2'."""
    return ', '.join('%s %d' % (DIR_LABELS[d], sum(walk['falls'][d])) for d in DIRS8 if sum(walk['falls'][d]))


def stand30(runs):
    """The turn over the 30 s stand of the bridge copy (the 'bridge' group's stand30 run, Unitree +6 ms; the critic's
    stand30.py um6), degrees, printed with no limit; None if it fell or was not run."""
    for r in runs:
        if r['case']['plan'] == 'stand30' and variant_of(r['case']) == 'um_6ms':
            seg = r['segs'][0] if r['segs'] else dict(missing=True)
            ok = r['fell_at'] is None and not seg.get('missing')
            return dict(heading=seg['heading_deg'] if ok else None, fell=r['fell_at'])
    return dict(heading=None, fell=None)


def side_check(results):
    """The ±1 m/s side check from its stand_turn.run results: runs, falls, the mean sideways speed and the worst
    |yaw rate| stepping left and right, the largest move in the stops after them (printed; falls are judged)."""
    speeds, drift, stops = {key: [] for key, _ in SIDE_CHECK_SEGMENTS}, {key: [] for key, _ in SIDE_CHECK_SEGMENTS}, []
    for r in results:
        if r['fell_at'] is not None:
            continue
        prev = None
        for seg in r['segs']:
            if not seg.get('missing'):
                if seg['name'] in speeds:
                    speeds[seg['name']].append(seg['side'])
                    drift[seg['name']].append(seg['yaw_rate'])
                elif seg['name'] == 'stop' and prev in speeds:
                    stops.append(seg['disp'])
            prev = seg['name']
    return dict(runs=len(results), falls=[fall_of(r) for r in results if r['fell_at'] is not None],
                speed={key: (float(np.mean(v)) if v else None) for key, v in speeds.items()},
                drift={key: (max(v, key=abs) if v else None) for key, v in drift.items()},
                stop=max(stops, default=None))


def side_reasons(side):
    """Reasons the ±1 m/s side check does not pass ([] = passes): a fall, or no run."""
    if not side['runs']:
        return ['шаг вбок ±1 м/с не измерен']
    if side['falls']:
        return ['шаг вбок ±1 м/с (%s): падений %d из %d (%s)' % (
            PHYSICS_LABELS[PUSH_PHYSICS], len(side['falls']), side['runs'], '; '.join(side['falls']))]
    return []


def select_summary(runs, push_results):
    """Step 2 from the stage's stand_turn.run results (7 variants and the ±1 m/s side check) and bridge_kick.run
    results (standing and walking, with the foot capsules and without). 'pushes' are the judged ones (since iteration
    24: Unitree +6 ms with the capsules), 'pushes_sim2sim' the ones without them (the rule's numbers before)."""
    checked = [r for r in runs if r['case']['plan'] == SIDE_CHECK]
    runs = [r for r in runs if r['case']['plan'] != SIDE_CHECK]
    rows = {name: row([r for r in runs if variant_of(r['case']) == name]) for name, _, _ in VARIANTS}
    labels = {name: text for name, text, _ in VARIANTS}
    pushes = {plan: push_table(push_results, plan, mags, PUSH_PHYSICS) for plan, mags, _ in PUSHES}
    forecast = {plan: push_table(push_results, plan, mags, FORECAST_PHYSICS) for plan, mags, _ in PUSHES}
    side = side_check(checked)
    reasons = (judge(rows, labels, [name for name, _, _ in VARIANTS]) + push_reasons(pushes['stand'], pushes['walk'])
               + side_reasons(side))
    return dict(passes=not reasons, reasons=reasons, rows=rows, pushes=pushes, pushes_sim2sim=forecast, side1=side,
                stand30=stand30(runs))


def fresh_summary(runs):
    """Step 3 from the fresh check's stand_turn.run results."""
    rows = {name: row([r for r in runs if fresh_group(r['case']) == name]) for name, _, _ in FRESH}
    labels = {name: text for name, text, _ in FRESH}
    reasons = judge(rows, labels, [name for name, _, judged in FRESH if judged])
    return dict(passes=not reasons, reasons=reasons, rows=rows)


def label(path):
    path = Path(path)
    if (path / 'policy.onnx').is_file():
        return path.name + ' (выгрузка)'
    return '%s @ %s' % (path.parent.name, path.name[6:].lstrip('0'))


def number(value, fmt):
    return '-' if value is None else fmt % value


def print_rows(title, summary, labels):
    print(title)
    print('  %-24s %8s %8s %8s %8s %8s %8s %7s %12s' % (
        '', 'прогонов', 'падений', '0.5 м/с', '1 м/с', 'влево', 'вправо', 'стоп, м', 'v 0.5 / 1'))
    for name, text in labels:
        r = summary['rows'][name]
        speed = '%s / %s' % (number(r['speed']['v05'], '%.2f'), number(r['speed']['v1'], '%.2f'))
        print('  %-24s %8d %8d %8s %8s %8s %8s %7s %12s' % (
            text, r['runs'], len(r['falls']), *(number(r['drift'][key], '%+.3f') for key, _ in DRIFT),
            number(r['stop'], '%.2f'), speed) + ('' if r['brake'] is None else '  стоп-старт %.2f м' % r['brake']))


def print_push_tables(pushes, physics, note):
    """The standing pushes by direction x dv and the walking pushes of one physics (its label and a note)."""
    stand, walk = pushes['stand'], pushes['walk']
    what = '%s, %s' % (PHYSICS_LABELS[physics], note)
    print('  толчки стоя (%s): падений из %d моментов 0-%.2f с, по Δv, м/с' % (
        what, len(STAND_OFFSETS), STAND_OFFSETS[-1] * st.TICK))
    print('    %-16s' % '' + ''.join('%6s' % dv(m) for m in stand['mags']))
    for d in DIRS8:
        print('    %-16s' % DIR_LABELS[d] + ''.join('%6d' % n for n in stand['falls'][d]))
    _, falls, runs = judged(stand)
    print('    до %s м/с: %d из %d (можно до %d)%s' % (dv(JUDGED_MAG), falls, runs, STAND_FALLS,
                                                    '; до толчка %d' % stand['early'] if stand['early'] else ''))
    by_mag = [sum(walk['falls'][d][k] for d in DIRS8) for k in range(len(walk['mags']))]
    where = walk_where(walk)
    print('  толчки на ходу 0.5 м/с (%s, %d фаз шага): %d из %d (Δv %s: %s)%s%s' % (
        what, len(WALK_OFFSETS), sum(by_mag), sum(sum(n) for n in walk['runs'].values()),
        ' / '.join(dv(m) for m in walk['mags']), ' / '.join('%d' % n for n in by_mag), '; ' + where if where else '',
        '; до толчка %d' % walk['early'] if walk['early'] else ''))


def print_pushes(summary):
    """The 30 s turn, the judged pushes of step 2 (with the foot capsules), the ±1 m/s side check and the pushes
    without the capsules (the sim2sim forecast)."""
    turn = summary['stand30']
    print('  стояние 30 с (Unitree +6 мс): ' + (
        'поворот %+.1f°' % turn['heading'] if turn['heading'] is not None
        else 'не измерено' if turn['fell'] is None else 'упал на %.1f с' % turn['fell']))
    print_push_tables(summary['pushes'], PUSH_PHYSICS, 'судится')
    side = summary['side1']
    print('  шаг вбок ±1 м/с (%s, судится по падениям): прогонов %d, падений %d%s; скорость вбок %s / %s м/с, увод '
          '%s / %s рад/с, стоп после шага %s м' % (
              PHYSICS_LABELS[PUSH_PHYSICS], side['runs'], len(side['falls']),
              ' (%s)' % '; '.join(side['falls']) if side['falls'] else '',
              *(number(side['speed'][key], '%+.2f') for key, _ in SIDE_CHECK_SEGMENTS),
              *(number(side['drift'][key], '%+.3f') for key, _ in SIDE_CHECK_SEGMENTS), number(side['stop'], '%.2f')))
    if summary.get('pushes_sim2sim'):
        print_push_tables(summary['pushes_sim2sim'], FORECAST_PHYSICS,
                          'прогноз sim2sim: стопы проходят друг сквозь друга, как в unitree_mujoco; не судится')


def verdict(summary):
    print('  итог: ' + ('проходит' if summary['passes'] else 'не проходит: ' + '; '.join(summary['reasons'])))


def report(stage):
    print('\n== %s' % stage['label'])
    print_rows('шаг 2 — 7 вариантов физики и толчки', stage['select'], [(n, t) for n, t, _ in VARIANTS])
    print_pushes(stage['select'])
    if stage.get('feet'):
        print('  ' + feet.line(stage['feet']))
    verdict(stage['select'])
    if stage.get('fresh'):
        print_rows('шаг 3 — свежая проверка (Unitree +6 мс)', stage['fresh'], [(n, t) for n, t, _ in FRESH])
        verdict(stage['fresh'])
    print('  время: шаг 2 — %d прогонов, %.0f с процессов%s' % (
        stage['runs']['select'], stage['seconds']['select'],
        '' if not stage.get('fresh') else '; шаг 3 — %d прогонов, %.0f с процессов' % (
            stage['runs']['fresh'], stage['seconds']['fresh'])))


def cost(kind, job):
    """Rough run length in physics ticks, to start the long runs first."""
    if kind == 'feet':
        return feet.cost(job)
    if kind == 'push':
        plan, physics = st.PLANS['kstand' if job['plan'] == 'stand' else 'kwalk'], job['phys']
    else:
        plan, physics = st.PLANS[job['plan']], job['physics']
    fix = 3500 if physics.startswith('um') else 14000  # FixStand at 2 ms / 0.5 ms
    return sum(secs for *_, secs in plan) / st.TICK + fix + 700


def init():
    import torch
    torch.set_num_threads(1)


def work(item):
    index, kind, job = item
    t0 = time.perf_counter()
    if kind == 'push':
        um.Stepper, kick.PLAN = kick.PushStepper, job['plan']
        try:
            result = kick.run(job)
        finally:
            um.Stepper = STEPPER
    elif kind == 'feet':  # sets and restores its own stepper and push plan
        result = feet.run(job)
    else:
        result = st.run(job)
    return index, result, time.perf_counter() - t0


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Отбор этапа: 7 вариантов физики и толчки в копии связки (шаг 2), '
                                                 'свежая проверка (шаг 3)')
    parser.add_argument('stages', nargs='+', type=Path, help='папка этапа или выгрузки export_g1')
    parser.add_argument('--fresh', action='store_true', help='ещё шаг 3: свежая проверка')
    parser.add_argument('--workers', type=int, default=8, help='рабочих процессов (пока идёт обучение, не больше 8)')
    parser.add_argument('--out', type=Path, help='JSON: итоги и все прогоны')
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.stages]
    for p in paths:
        if not ((p / 'policy.onnx').is_file() or ((p / 'metadata.json').is_file() and (p / 'policy.zip').is_file())):
            raise SystemExit(f'Не этап и не выгрузка export_g1: {p}')
    if not st.LOGS.is_dir():
        raise SystemExit(f'Нет логов связки (окна времени): {st.LOGS} — запущена ли WSL Ubuntu?')
    items = []  # (stage index, step, kind, job)
    for i, p in enumerate(paths):
        items += [(i, 'select', kind, job) for kind, job in select_jobs(p)]
        if args.fresh:
            items += [(i, 'fresh', kind, job) for kind, job in fresh_jobs(p)]
    order = sorted(range(len(items)), key=lambda k: -cost(items[k][2], items[k][3]))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # one core per worker
        os.environ.setdefault(key, '1')
    print('этапов: %d, прогонов: %d, процессов: %d' % (len(paths), len(items), args.workers), flush=True)
    results, seconds = [None] * len(items), [0.] * len(items)
    t0 = time.time()
    with Pool(args.workers, initializer=init, maxtasksperchild=16) as pool:
        done = 0
        for index, result, secs in pool.imap_unordered(work, [(k, items[k][2], items[k][3]) for k in order]):
            results[index], seconds[index] = result, secs
            done += 1
            if done % 100 == 0 or done == len(items):
                print('  %d из %d, %.0f с' % (done, len(items), time.time() - t0), flush=True)
    wall = time.time() - t0
    stages = []
    for i, p in enumerate(paths):
        def of(step, kind):
            return [results[k] for k, item in enumerate(items) if item[:3] == (i, step, kind)]

        def spent(step):
            return sum(seconds[k] for k, item in enumerate(items) if item[:2] == (i, step))

        stage = dict(label=label(p), path=str(p), select=select_summary(of('select', 'run'), of('select', 'push')),
                     feet=feet.summary(of('select', 'feet')),
                     runs={'select': sum(item[:2] == (i, 'select') for item in items)},
                     seconds={'select': spent('select')})
        if args.fresh:
            stage['fresh'] = fresh_summary(of('fresh', 'run'))
            stage['runs']['fresh'] = sum(item[:2] == (i, 'fresh') for item in items)
            stage['seconds']['fresh'] = spent('fresh')
        report(stage)
        stage['results'] = {step: {kind: of(step, kind) for kind in ('run', 'push', 'feet')}
                            for step in (('select', 'fresh') if args.fresh else ('select',))}
        stages.append(stage)
    passed = [s for s in stages if s['select']['passes']]
    print('\nшаг 2 проходят %d из %d этапов: %s' % (len(passed), len(stages), ', '.join(s['label'] for s in passed)
                                                    or '-'))
    if args.fresh:
        both = [s for s in passed if s['fresh']['passes']]
        print('шаг 2 и шаг 3 проходят %d из %d: %s' % (len(both), len(stages), ', '.join(s['label'] for s in both)
                                                       or '-'))
    print('%.0f с на %d процессах (%.0f с на этап)' % (wall, args.workers, wall / len(stages)))
    if args.out:
        rule = dict(drift=DRIFT_LIMIT, stop=STOP_LIMIT, variants=[name for name, _, _ in VARIANTS],
                    push_physics=PUSH_PHYSICS, forecast_physics=FORECAST_PHYSICS, side_check=SIDE_CHECK,
                    pushes=dict(dirs=DIRS8, stand_mags=STAND_MAGS, stand_offsets=STAND_OFFSETS, walk_mags=WALK_MAGS,
                                walk_offsets=WALK_OFFSETS, judged_mag=JUDGED_MAG, stand_falls=STAND_FALLS,
                                cell_falls=CELL_FALLS, half=HALF, walk_falls=WALK_FALLS))
        args.out.write_text(json.dumps(dict(rule=rule, workers=args.workers, seconds=wall, stages=stages), indent=1),
                            encoding='utf-8')


if __name__ == '__main__':
    main()
