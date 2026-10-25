"""CPU only. skills_check.py <stage | experiment[:steps] | export dir> [--against parent|n8|<stage | experiment[:steps]>]
[--workers 8] [--json <file>] [--journal]

Has a stage forgotten a skill its reference had? (the user's request of 25.10: at a branch switch only goal_check ran;
pushes, speed, side steps, crouch and the arms were judged at the candidate's selection or by hand.) One table, a row a
measure: skill, measure, threshold, the stage's value, the reference's, the verdict. Every row comes from quick parts of
the lab's probes, called, not copied (scripts/lab/probes):
  1. walking: goal_check's conditions (supervise.check_stage and goal_check.walks, GOAL_SEEDS seeds, its items());
  2. speed: speed_accept.episode at 0.5 / 1 / 2 m/s forward and -0.5 back, the worst seed's speed over the command;
  3. turns: goal_check.walks at ±0.5 rad/s in place and walking 0.5 m/s, the smaller share of the command;
  4. side steps: stand_turn.run in our physics ('side' ±0.3, 'side05' ±0.5: worst |yaw rate|, the stop right after),
     stage_select's ±1 m/s side check in the bridge copy with the foot capsules (falls; drift printed);
  5. braking: speed_accept.episode, 6 s at 0.5 / 1 m/s, the path in the 3 s after the stop;
  6. pushes: bridge_kick.run in the bridge copy with the foot capsules (stage_select's 'um_caps'): standing, 8
     directions x dv 1.0 / 1.25 x 2 moments, and walking 0.5 m/s, 8 directions x dv 1.0 x 2 gait phases;
  7. physics: stand_turn.run's walk 0.5 m/s with motor friction x0.25, without the motor model, ours +6 ms;
  8. feet: foot_overlap's side steps ±1 m/s, the share of steps with the two feet's capsules overlapping;
  9. arms in any pose: arm_probe.run at 1.0 rad, 3 draws standing, walking 0.5 and 1 m/s;
 10. crouch (task_config height_command): crouch.run standing 0.70 / 0.64 and walking 0.5 m/s at 0.70, and
     stage_select's crouched walk in the bridge copy;
 11. arms on targets (task_config arm_targets): arm_joints.run on the carry poses at 0 / 1 kg, train_falls.run
     'free' and 'target:1' (FALLS_EPISODES each, the exact 95 % Poisson interval);
 12. torque chattering: chatter.run's six scenarios x 3 seeds, against Unitree v0 at the stage's delay;
 13. the hoist: scripts/robot/reference/hoist.py 'air5' in our physics, the legs at command 0 in the air.
A row passes when it meets its threshold (the rules already in use: goal_check, stage_select, prefilter, speed_accept,
arm_probe, the robot package's hoist, the stage 3 milestones of iterations 41-44) or the reference misses it too, and
is not worse than the reference beyond a margin (falls: + 1 + runs // 20; falls per episode: overlapping 95 %
intervals; drift, distances, speeds: a share of the reference's value or a floor, whichever is larger); the note says
which held. A row a policy cannot run is 'н/п' with the reason (no height command, no arm targets, a probe that does
not take an export folder). The reference: the stage its branch started from (the manifest's source, or a warm_start's
for a root), 'n8' candidate 8 (exp-28691193718e4d0e @ 7221510144), or any stage. The same stage as its own reference
runs once. --journal records the table as the «Дневник» tab's skills check, next to the goal check (journal.set_skills).
About 10 minutes for a stage and its reference on 8 processes, ~800 runs.
A g1_getup stage (iteration 46) gets the getting-up rows instead (getup_table; the reference a getting-up stage too,
--against getup823: exp-f19f12fb1eff4ef3 @ 823984128), on GETUP_PER_CLASS fresh starts per class (getup_stance's,
ready38's seeds): getup_probe.run in ours, a 20 ms delay and sole friction 0.3 — ready38's robot success (up within
6 s, standing 1 s later with no g1_ctrl speed cutoff, candidate 8's Velocity from then on without a fall), per
physics and per start class, the cutoffs, the median rise; the turn over the 5 s standing after the hand-over
(turn39), its falls; the loads (locked wrists and the other motors above 90 % of peak, head or torso hits above
300 N, the first 0.2 s above 60 %); getup_stance.run: the feet's width at +1 / +2 s, the hips' roll pull, arms on the
body, the shoulders' roll, flights, the touchdown speed and the peak sole force. Thresholds: the critic's acceptance
of «Вставание 823» (docs/lab-knowledge.md). The journal card is titled «… · вставание»."""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / 'scripts' / 'lab' / 'probes', ROOT / 'scripts' / 'robot',
               ROOT / 'scripts' / 'robot' / 'reference'):
    sys.path.insert(0, str(folder))
import stage_select as ss  # noqa: E402  (first: it puts um_probe's own stepper back after bridge_kick's import)
import arm_joints as aj  # noqa: E402
import arm_probe as ap  # noqa: E402
import crouch as cr  # noqa: E402
import getup_probe as gp  # noqa: E402
import getup_stance as gs  # noqa: E402
import hoist as ho  # noqa: E402
import package as pk  # noqa: E402
import speed_accept as sa  # noqa: E402
import train_falls as tf  # noqa: E402

ch, feet, st = ss.ch, ss.feet, ss.st
N8 = ROOT / 'runs' / 'humanoid_lab' / 'exp-28691193718e4d0e' / 'stage_007221510144'
GOAL_SEEDS = 3
TURNS = ((0., .5), (0., -.5), (.5, .5), (.5, -.5))  # (forward m/s, yaw rad/s)
TURN_SEEDS = 2
SPEEDS = (.5, 1., 2., -.5)  # m/s forward, then back
SPEED_SEEDS = (73000, 73001)
BRAKES = (.5, 1.)  # m/s walked 6 s before the stop
BRAKE_AT, BRAKE_STEPS = 300, 150  # control steps: the stop at 6 s, measured over the 3 s after it
STAND_PUSHES, STAND_MOMENTS = (1., 1.25), (0, 197)  # dv m/s; ticks of 2 ms past 3 s after the band
WALK_PUSHES, WALK_PHASES = (1.,), (0, 150)  # dv m/s; ticks past 3 s into the walk: half a 0.6 s gait cycle apart
PHYSICS = ('fric', 'nomotor', 'ours_6ms')  # stage_select.VARIANTS walking 0.5 m/s
FEET_SIDES = ('влево 1', 'вправо 1')  # foot_overlap.SIDE_COMMANDS at ±1 m/s
ARM_AMPLITUDE, ARM_DRAWS = 1., 3
CROUCH_STAND_SEEDS, CROUCH_WALK_SEEDS = 1, 2
JOINT_LOADS = (0., 1.)  # kg per hand on the carry poses
FALLS_EPISODES, FALLS_SEED = 120, 90000
HOIST = ('ours', 'air5')
# g1_getup stages (iteration 46): their own rows, the critic's acceptance of «Вставание 823» (docs/lab-knowledge.md).
GETUP_823 = ROOT / 'runs' / 'humanoid_lab' / 'exp-f19f12fb1eff4ef3' / 'stage_000823984128'
GETUP_PHYSICS = ('ours', 'delay20', 'foot0.3')  # getup_probe.PHYSICS
GETUP_PER_CLASS = 24  # fresh starts per class (getup_stance.fresh_starts: ready38's seeds), 120 in all
HAND_OVER_AT = 1.  # s after up: the hand-over to candidate 8's Velocity (ready38's h1, turn39)
LOCKED_WRISTS = (20, 21, 27, 28)


# --- What to run --------------------------------------------------------------------------------------------------

def policy_path(spec):
    """A stage folder, an export_g1 folder or <experiment>[:steps] of the lab (its latest stage)."""
    return Path(tf.resolve(spec)).resolve()


def stage_of(path):
    """The stage a policy folder is: itself, or an export folder's stage."""
    from humanoid_sim.export_g1 import ExportedPolicy, is_export
    return Path(ExportedPolicy(path).stage).resolve() if is_export(path) else Path(path)


def parent_of(path):
    """The stage the branch of `path` (an export: of its stage) started from: the manifest's source (source_id,
    source_stage) or, for a root a stage was moved into (warm_start), the warm_start's source."""
    from humanoid_sim.core import read_json
    stage = stage_of(path)
    manifest = read_json(stage.parent / 'experiment.json', {}) or {}
    config = manifest.get('config') or {}
    source = config.get('source_id'), config.get('source_stage')
    if not all(source):
        warm = manifest.get('warm_start') or {}
        source = warm.get('source_id'), warm.get('source_stage')
    if not all(source):
        raise SystemExit(f'{stage.parent.name}: у ветки нет родителя (ни source, ни warm_start) — задайте --against')
    parent = stage.parent.parent / source[0] / f'stage_{int(source[1]):012d}'
    if not parent.is_dir():
        raise SystemExit(f'Нет этапа родителя {parent}')
    return parent


def reference_path(spec, path):
    if spec in (None, 'parent'):
        return parent_of(path)
    if spec == 'n8':
        return N8
    if spec == 'getup823':
        return GETUP_823
    return policy_path(spec)


def task_of(path):
    """The task of a policy folder (an export: its stage's)."""
    from humanoid_sim.core import read_json
    return (read_json(stage_of(path) / 'metadata.json', {}) or {}).get('task_id')


def traits(path):
    """What a policy can run: export folder or stage, the height command, the arm targets, its training delay."""
    from humanoid_sim.export_g1 import is_export
    text = str(path)
    return dict(path=text, export=bool(is_export(path)), height=bool(st.has_height(text)),
                arms=bool(ch.has_arm_targets(text)), delay=ch.stage_delay(text), label=ss.label(path))


def jobs_of(owner, t, arms_group):
    """[(owner, tag, kind, job)] of one policy; arms_group: the judged stage has arm targets (row group 11)."""
    p, out = t['path'], []

    def add(tag, kind, job):
        out.append((owner, tag, kind, job))
    add('goal:check', 'goal', dict(path=p, part='check', seeds=GOAL_SEEDS))
    add('goal:walks', 'goal', dict(path=p, part='walks', seeds=GOAL_SEEDS))
    for speed, yaw in TURNS:
        add(f'turn:{speed:g}:{yaw:g}', 'goal', dict(path=p, part='turns', seeds=TURN_SEEDS, turns=[[speed, yaw]]))
    for command in SPEEDS:
        for seed in SPEED_SEEDS:
            add(f'speed:{command:g}:{seed}', 'speed', dict(path=p, command=command, seed=seed))
    for command in BRAKES:
        for seed in SPEED_SEEDS:
            add(f'brake:{command:g}:{seed}', 'speed', dict(path=p, command=command, seed=seed, stop_at=BRAKE_AT,
                                                         steps=BRAKE_AT + BRAKE_STEPS))
    ours = variant('ours')
    add('side:03', 'run', st.case(plan='side', stage=p, **ours))
    add('side:05', 'run', st.case(plan=ss.SIDE05, stage=p, **ours))
    for k, case in enumerate(ss.side_check_jobs(p)):
        add(f'side:1:{k}', 'run', case)
    for plan, mags, moments in (('stand', STAND_PUSHES, STAND_MOMENTS), ('walk', WALK_PUSHES, WALK_PHASES)):
        for d in ss.DIRS8:
            for mag in mags:
                for offset in moments:
                    add(f'push:{plan}:{d}:{mag:g}:{offset}', 'push',
                        dict(name=ss.label(p), path=p, left=False, phys=ss.PUSH_PHYSICS, dir=d, mag=mag, offset=offset,
                             plan=plan))
    for name in PHYSICS:
        add(f'phys:{name}', 'run', st.case(plan='walk', stage=p, **variant(name)))
    for job in feet.jobs(p):
        if job['kind'] == 'side' and job['name'] in FEET_SIDES:
            add(f'feet:{job["name"]}', 'feet', job)
    for plan in ap.PLANS:
        for draw in range(ARM_DRAWS):
            add(f'armp:{plan}:{draw}', 'arm_probe', dict(stage=p, plan=plan, A=ARM_AMPLITUDE, draw=draw, mask=False))
    if t['height']:
        stand, walk = crouch_plans()
        for k in range(CROUCH_STAND_SEEDS):
            add(f'crouch:stand:{k}', 'crouch', dict(stage=p, name='stand', plan=stand, seed=cr.SEED + k))
        for k in range(CROUCH_WALK_SEEDS):
            add(f'crouch:walk:{k}', 'crouch', dict(stage=p, name='walk', plan=walk, seed=cr.SEED + k))
        add('crouch:bridge', 'run', ss.height_jobs(p)[0])
    if arms_group and not t['export']:
        conditions = ('free', 'target:1') if t['arms'] else ('free',)
        for condition in conditions:
            for k in range(FALLS_EPISODES):
                add(f'falls:{condition}:{k}', 'falls', dict(stage=p, condition=condition, seed=FALLS_SEED + k, beta=1.))
        if t['arms']:
            for name, names, pose in aj.poses():
                if name.startswith('carry'):
                    for load in JOINT_LOADS:
                        add(f'joints:{name}:{load:g}', 'arm_joints',
                            dict(stage=p, pose_name=name, names=names, pose=pose, load=load))
    for job in ch.jobs(p, t['arms']):
        add(f'chatter:{job["scenario"]}:{job["seed"]}', 'chatter', job)
    if not t['export']:
        add('hoist', 'hoist', dict(path=p, physics=HOIST[0], plan=HOIST[1]))
    return out


def variant(name):
    """stand_turn case fields of a stage_select physics variant."""
    return next(fields for key, _, fields in ss.VARIANTS if key == name)


def crouch_plans():
    """crouch.py's standing plan (0.78, 0.70, 0.64, 0.78) and its walk at 0.5 m/s at 0.70."""
    stand = next(plan for name, plan in cr.schedules('A').items())
    walk = next(plan for plan in cr.schedules('B').values() if plan[0][1][0] == .5)
    return stand, walk


def cost(kind, job):
    """Rough run length in physics ticks of 2 ms, to start the long runs first."""
    if kind in ('run', 'push', 'feet', 'chatter'):
        return ss.cost(kind, job)
    if kind == 'goal':
        return {'check': 9, 'walks': 18, 'turns': 2}[job['part']] * job['seeds'] / 3 * 5000
    if kind == 'falls':
        return 10000
    if kind == 'crouch':
        return sum(segment[0] for segment in job['plan']) / st.TICK
    if kind == 'hoist':
        return 15000
    if kind == 'getup':
        return 8000
    return 5000


# --- Running ------------------------------------------------------------------------------------------------------

_exports = {}


def acting(path):
    """(stage, acting) for goal_check: an export folder's stage and its policy.onnx (as policy and normalizer)."""
    from humanoid_sim.export_g1 import ExportedPolicy, is_export
    if not is_export(path):
        return Path(path), None
    if path not in _exports:
        _exports[path] = ExportedPolicy(path)
    exported = _exports[path]
    return Path(exported.stage), (exported, exported)


def goal(job):
    """goal_check's two parts or extra turns, as plain lists (picklable, JSON)."""
    from humanoid_sim.goal_check import walks
    from humanoid_sim.supervise import check_stage
    stage, actor = acting(job['path'])
    if job['part'] == 'check':
        return check_stage(stage, seeds=job['seeds'], acting=actor)
    extra = {} if job['part'] == 'walks' else dict(turns=[tuple(t) for t in job['turns']], straight=())
    walk = walks(stage, seeds=job['seeds'], acting=actor, **extra)
    return dict(turn=[[s, y, v] for (s, y), v in walk['turn'].items()],
                drift=[[s, start, rates] for (s, start), rates in walk['drift'].items()],
                pelvis=walk['pelvis'], feet=walk['feet'])


def init():
    ss.init()


def work(item):
    index, kind, job = item
    started = time.perf_counter()
    if kind in ('run', 'push', 'feet', 'chatter'):
        result = ss.work((index, kind, job))[1]
    elif kind == 'goal':
        result = goal(job)
    elif kind == 'speed':
        result = sa.episode(job['path'], job['command'], job['seed'], stop_at=job.get('stop_at'),
                            steps=job.get('steps', 500))
    elif kind == 'arm_probe':
        result = ap.run(job)
    elif kind == 'crouch':
        result = cr.run(job)
    elif kind == 'arm_joints':
        result = aj.run(job)
    elif kind == 'falls':
        try:
            result = tf.run(job)
        finally:
            tf._cache['mode'] = None  # the forced arm mode stays with train_falls' jobs
    elif kind == 'hoist':
        result = ho.run((job['path'], job['physics'], job['plan']))
    elif kind == 'getup':
        result = gp.run(job)
    elif kind == 'stance':
        result = gs.run(job)
    else:
        raise ValueError(kind)
    return index, result, time.perf_counter() - started


# --- Measures -----------------------------------------------------------------------------------------------------

def walk_of(result):
    """goal()'s walks back in goal_check.walks' layout."""
    return dict(turn={(s, y): v for s, y, v in result['turn']},
                drift={(s, start): rates for s, start, rates in result['drift']},
                pelvis=result['pelvis'], feet=result['feet'])


def measures(got, t, v0=None):
    """{row key: measure} of one policy from its results by tag; a measure is {'v': number (math.inf: fell where less
    is better), ...}; a key is absent when the policy did not run it. v0: Unitree v0's chatter runs at its delay."""
    from humanoid_sim.goal_check import items
    inf, out = math.inf, {}
    report, walked = got.get('goal:check'), got.get('goal:walks')
    if report and walked:
        walk = walk_of(walked)
        scenarios = report['scenarios']
        falls = [tuple(int(x) for x in v['falls'].split('/')) for v in scenarios.values()]
        stand, walking = scenarios['stand'], scenarios['walk']
        numbers = [sum(f for f, _ in falls), walking['forward_m'], walking['heading_deg'],
                   math.hypot(stand['forward_m'], stand['sideways_m']), None, walk['pelvis'],
                   min(walk['turn'][(1., .5)], -walk['turn'][(1., -.5)]),
                   max(abs(rate) for rates in walk['drift'].values() for rate in rates),
                   max(v['clipped'] for v in scenarios.values()), walk['feet']['lift_m']]
        for k, (item, number) in enumerate(zip(items(scenarios, report['policy_std'], walk), numbers)):
            out[f'goal:{k}'] = dict(v=number, ok=item['ok'], text=item['value'], label=item['label'],
                                    target=item['target'], n=sum(n for _, n in falls))
    for command in SPEEDS:
        runs = [got.get(f'speed:{command:g}:{seed}') for seed in SPEED_SEEDS]
        if all(runs):
            shares = [0. if r['fell'] is not None or r['speed'] is None else r['speed'] / command for r in runs]
            out[f'speed:{command:g}'] = dict(v=min(shares), falls=sum(r['fell'] is not None for r in runs))
    for name, speed in (('place', 0.), ('walk', .5)):
        found = [got.get(f'turn:{speed:g}:{yaw:g}') for yaw in (.5, -.5)]
        if all(found):
            out[f'turn:{name}'] = dict(v=min(rate / yaw for r in found for _, yaw, rate in r['turn']))
    side03, side05 = got.get('side:03'), got.get('side:05')
    if side03 and side05:
        for key, result, names in (('side:03', side03, ('L', 'R')), ('side:05', side05, ('L05', 'R05'))):
            got_side = ss.measures(result)
            values = [abs(x) for name in names for x in got_side[name]]
            out[key] = dict(v=inf if result['fell_at'] is not None or len(values) < 2 else max(values))
        stops = ss.measures(side03)['stop'] + ss.measures(side05)['stop05']
        fell = side03['fell_at'] is not None or side05['fell_at'] is not None
        out['side:stop'] = dict(v=inf if fell or not stops else max(stops))
    checked = [got.get(f'side:1:{k}') for k in range(len(ss.side_check_jobs(t['path'])))]
    if all(checked):
        side = ss.side_check(checked)
        drift = [abs(v) for v in side['drift'].values() if v is not None]
        out['side:1'] = dict(v=len(side['falls']), n=side['runs'])
        out['side:1drift'] = dict(v=inf if side['falls'] or len(drift) < 2 else max(drift))
    for command in BRAKES:
        runs = [got.get(f'brake:{command:g}:{seed}') for seed in SPEED_SEEDS]
        if all(runs):
            moved = [r['moved'] for r in runs]
            out[f'brake:{command:g}'] = dict(v=inf if any(m is None for m in moved) else max(moved))
    for plan in ('stand', 'walk'):
        pushes = [r for tag, r in got.items() if tag.startswith(f'push:{plan}:')]
        if pushes:
            out[f'push:{plan}'] = dict(v=sum(r['fell'] is not None for r in pushes), n=len(pushes))
    for name in PHYSICS:
        result = got.get(f'phys:{name}')
        if result:
            drift = ss.measures(result)['w05']
            out[f'phys:{name}'] = dict(v=inf if result['fell_at'] is not None or not drift else max(map(abs, drift)))
    sides = [got.get(f'feet:{name}') for name in FEET_SIDES]
    if all(sides):
        shares = [feet.summary(sides)['side'][name]['share'] for name in FEET_SIDES]
        out['feet:1'] = dict(v=None if None in shares else max(shares))
    probes = [r for tag, r in got.items() if tag.startswith('armp:')]
    if len(probes) == len(ap.PLANS) * ARM_DRAWS:
        v = ap.verdict(probes, t['path'], ARM_AMPLITUDE)
        drift = [d for d in (v['drift_05'], v['drift_1']) if d is not None]
        out['armp:falls'] = dict(v=v['falls'], n=v['runs'])
        out['armp:turn'] = dict(v=inf if v['turn_worst'] is None else abs(v['turn_worst']))
        out['armp:creep'] = dict(v=inf if v['creep_worst'] is None else v['creep_worst'])
        out['armp:drift'] = dict(v=inf if len(drift) < 2 else max(drift))
    stands = [got.get(f'crouch:stand:{k}') for k in range(CROUCH_STAND_SEEDS)]
    walks = [got.get(f'crouch:walk:{k}') for k in range(CROUCH_WALK_SEEDS)]
    if t['height'] and all(stands) and all(walks):
        for key, segment in (('crouch:stand70', 1), ('crouch:stand64', 2)):
            fell = any(len(r['segments']) <= segment or r['fell'] is not None for r in stands)
            out[key] = dict(v=inf if fell else max(r['segments'][segment]['pelvis'] for r in stands))
        fell = any(r['fell'] is not None or not r['segments'] for r in walks)
        out['crouch:walk'] = dict(v=inf if fell else max(r['segments'][0]['pelvis'] for r in walks))
        out['crouch:speed'] = dict(v=0. if fell else min(r['segments'][0]['speed'] for r in walks))
    bridge = got.get('crouch:bridge')
    if t['height'] and bridge:
        seg = ss.crouch_run([bridge], ss.CROUCH_WALK, 'walk')['seg']
        out['crouch:bridge'] = dict(v=inf if seg is None else abs(seg['yaw_rate']))
    joints = [r for tag, r in got.items() if tag.startswith('joints:')]
    if joints:
        with contextlib.redirect_stdout(io.StringIO()):
            rows = {row['load']: row for row in aj.report(joints, [t['path']], list(JOINT_LOADS))}
        fell = any(r['fell'] is not None for r in joints) or len(rows) < len(JOINT_LOADS)
        for load in JOINT_LOADS:
            row = rows.get(load)
            out[f'joints:roll{load:g}'] = dict(v=inf if fell else max(row['shoulder_roll']))
            out[f'joints:error{load:g}'] = dict(v=inf if fell else row['error'])
        out['joints:pitch'] = dict(v=inf if fell else max(max(row['pitch']) for row in rows.values()))
        out['joints:elbow'] = dict(v=inf if fell else max(max(row['elbow']) for row in rows.values()))
        out['joints:palms'] = dict(v=inf if fell else max(abs(row['palms']) for row in rows.values()))
    for condition in ('free', 'target:1'):
        runs = [r for tag, r in got.items() if tag.startswith(f'falls:{condition}:')]
        if runs:
            falls, exposure = sum(r['fell'] for r in runs), sum(r['steps'] for r in runs) / tf.EPISODE_STEPS
            lower, upper = tf.poisson_interval(falls, exposure)
            out[f'falls:{condition}'] = dict(v=falls / exposure, lower=lower, upper=upper, falls=falls,
                                             episodes=len(runs))
    chatter = [r for tag, r in got.items() if tag.startswith('chatter:')]
    if chatter:
        summary = ch.summary_of(chatter)
        reference = ch.summary_of(v0) if v0 else None
        for key in ch.GATED:
            out[f'chatter:{key}'] = dict(v=summary.get(key), falls=len(summary['falls']),
                                         v0=None if reference is None else reference.get(key))
    hoisted = got.get('hoist')
    if hoisted:
        segments = pk.segments([hoisted], (HOIST[1],), ('stand',))
        out['hoist'] = dict(v=max(s['qv_leg_p95'] for s in segments), da=max(s['da'] for s in segments))
    return out


# --- Getting up (g1_getup stages) -----------------------------------------------------------------------------------

def getup_jobs(owner, t, starts, limits):
    """[(owner, tag, kind, job)] of a getting-up policy on the starts: getup_probe.run in each GETUP_PHYSICS (the
    hand-over to candidate 8 HAND_OVER_AT s after up, speed cutoffs at g1_ctrl's limits) and getup_stance.run (ours)."""
    from humanoid_sim.tasks.g1_getup_bank import FALL_STAGE
    out = []
    for physics in GETUP_PHYSICS:
        for k, start in enumerate(starts):
            out.append((owner, f'ready:{physics}:{k}', 'getup', dict(
                policy=t['path'], velocity=str(FALL_STAGE), physics=physics, row=None, seed=73000 + k,
                state=(start['qpos'], start['qvel']), fixstand=0., pd_bound='min', snapshot=HAND_OVER_AT,
                max_joint_speed=limits, **{'class': start['cls']})))
    out += [(owner, f'stance:{k}', 'stance', dict(stage=t['path'], **start)) for k, start in enumerate(starts)]
    return out


def ready(record):
    """The critic's ready38 robot success (h1): up within 6 s, not fallen HAND_OVER_AT s later (the walking task's
    rule) with no speed cutoff before then, and candidate 8's Velocity from that moment without a fall (5 s standing,
    5 s walking 0.5 m/s)."""
    cut = record.get('cut')
    return bool(record['up_s'] is not None and (cut is None or cut[0] > record['up_s'] + HAND_OVER_AT)
                and record['hand_over'] is not None and record['hand_over']['fell'] is None)


def finite(value):
    """math.inf for a missing or NaN measure where less is better."""
    return math.inf if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def getup_measures(got):
    """{row key: measure} of a getting-up policy from its results by tag (getup_table's rows)."""
    out = {}
    runs = {p: [r for tag, r in got.items() if tag.startswith(f'ready:{p}:')] for p in GETUP_PHYSICS}
    every = [r for p in GETUP_PHYSICS for r in runs[p]]
    for physics, records in runs.items():
        if records:
            ok = sum(map(ready, records))
            out[f'ready:{physics}'] = dict(v=len(records) - ok, n=len(records), ok=ok)
    for kind in gs.KINDS:
        records = [r for r in every if r['job']['class'] == kind]
        if records:
            ok = sum(map(ready, records))
            out[f'ready:{kind}'] = dict(v=len(records) - ok, n=len(records), ok=ok)
    ours = runs['ours']
    if every:
        cut = [r for r in every if r.get('cut') is not None
               and (r['up_s'] is None or r['cut'][0] <= r['up_s'] + HAND_OVER_AT)]
        out['ready:cut'] = dict(v=len(cut), n=len(every))
        handed = [r['hand_over'] for r in every if r['hand_over'] is not None]
        out['handover:falls'] = dict(v=sum(h['fell'] in ('fixstand', 'stand') for h in handed), n=len(handed))
        out['loads:hits'] = dict(v=sum(r['impact_n'] > gp.IMPACT for r in every), n=len(every))
        out['loads:early'] = dict(v=sum(r['early_share'] > gp.EARLY_SHARE for r in every), n=len(every))
    if ours:
        held = sum(r['success'] for r in ours)
        out['ready:hold'] = dict(v=len(ours) - held, n=len(ours), ok=held)
        ups = [r['up_s'] for r in ours if r['up_s'] is not None]
        out['ready:time'] = dict(v=float(np.median(ups)) if ups else math.inf)
        turns = [abs(r['hand_over']['turn_deg']) for r in ours
                 if r['hand_over'] is not None and r['hand_over'].get('turn_deg') is not None]
        out['handover:turn'] = dict(v=float(np.median(turns)) if turns else math.inf)
        out['handover:turn15'] = dict(v=sum(turn > 15 for turn in turns), n=len(turns))
        seconds = sum(r['seconds'] for r in ours) or 1.
        out['loads:wrists'] = dict(v=max(sum(r['saturated_joint'][j] * r['seconds'] for r in ours) / seconds
                                         for j in LOCKED_WRISTS))
        out['loads:free'] = dict(v=sum(r['saturated_free'] * r['seconds'] for r in ours) / seconds)
    stance = [r for tag, r in got.items() if tag.startswith('stance:')]
    if stance:
        s = gs.summary(stance)
        for mark in (1., 2.):
            median, p90 = s['widths'][mark]
            out[f'stance:width{mark:g}'] = dict(v=finite(median), p90=finite(p90))
        out['stance:pull'] = dict(v=finite(s['pull']))
        out['stance:arms'] = dict(v=finite(s['arm_share']))
        out['stance:shoulder'] = dict(v=finite(max(map(abs, s['shoulder_roll']))))
        out['stance:flights'] = dict(v=s['flights'], n=s['up'])
        out['stance:touchdown'] = dict(v=finite(s['touchdown']), max=finite(s['touchdown_max']))
        out['stance:force'] = dict(v=finite(s['force_max']), median=finite(s['force']))
    return out


def getup_table():
    """getting up's rows, as table(): the critic's acceptance of «Вставание 823» (docs/lab-knowledge.md), ready38's
    shares (≥ 99 % ours, ≥ 96 % the other physics), getup_probe's limits of the loads."""
    def num(m, digits=2):
        v = m.get('v')
        return '—' if v is None else 'упал' if v == math.inf else f'{v:.{digits}f}'.replace('.', ',')

    def done(m):
        ok, n = m['ok'], m['n']
        return f'{ok} из {n}'

    def falls(m):
        n, runs = m['v'], m['n']
        return f'{n} из {runs}'

    def share(m):
        v = m.get('v')
        return '—' if v is None or v == math.inf else f'{100 * v:.1f}%'.replace('.', ',')

    def spread(key, digits):
        def show(m):
            first, second = num(m, digits), num(dict(v=m.get(key)), digits)
            return f'{first} / {second}'
        return show

    def ready_share(limit):
        return lambda m: m['n'] > 0 and m['ok'] >= limit * m['n'] - 1e-9

    def at_most(limit):
        return lambda m: m.get('v') is not None and m['v'] <= limit

    def at_most_share(limit):
        return lambda m: m['v'] <= limit * m['n'] + 1e-9

    lower = 'lower'
    ready_skill, stance_skill = 'Вставание: готовность к роботу', 'Вставание: стойка после подъёма'
    over, loads = 'Вставание: передача в ходьбу №8', 'Вставание: нагрузки'
    return [
        ('ready:ours', ready_skill, 'Наша физика: встал, без отсечки скорости, передан в ходьбу №8', '≥ 99%',
         ready_share(.99), ('falls',), None, done),
        ('ready:delay20', ready_skill, 'Задержка 20 мс: встал, без отсечки скорости, передан в ходьбу №8', '≥ 96%',
         ready_share(.96), ('falls',), None, done),
        ('ready:foot0.3', ready_skill, 'Трение стоп 0,3: встал, без отсечки скорости, передан в ходьбу №8', '≥ 96%',
         ready_share(.96), ('falls',), None, done),
        ('ready:back', ready_skill, 'Старт на спине, все физики', 'не судится', None, ('falls',), None, done),
        ('ready:front', ready_skill, 'Старт на животе, все физики', 'не судится', None, ('falls',), None, done),
        ('ready:side', ready_skill, 'Старт на боку, все физики', 'не судится', None, ('falls',), None, done),
        ('ready:sit', ready_skill, 'Старт сидя, все физики', 'не судится', None, ('falls',), None, done),
        ('ready:fall', ready_skill, 'Старт из падения №8, все физики', 'не судится', None, ('falls',), None, done),
        ('ready:hold', ready_skill, 'Наша физика: встал за 6 с и простоял 3 с без уползания (getup_probe)',
         'не судится', None, ('falls',), None, done),
        ('ready:cut', ready_skill, 'Отсечки скорости g1_ctrl до +1 с после подъёма, все физики', '0', at_most(0),
         ('falls',), None, falls),
        ('ready:time', ready_skill, 'Подъём, медиана, наша физика, с', 'не судится', None, (lower, .25, .3), None,
         num),
        ('stance:width1', stance_skill, 'Ширина стоп через 1 с, м: медиана / p90', '≤ 0,34 / ≤ 0,40',
         lambda m: m['v'] <= .34 and m['p90'] <= .40, (lower, .1, .07), None, spread('p90', 3)),
        ('stance:width2', stance_skill, 'Ширина стоп через 2 с, м: медиана / p90', '≤ 0,34 / ≤ 0,40',
         lambda m: m['v'] <= .34 and m['p90'] <= .40, (lower, .1, .07), None, spread('p90', 3)),
        ('stance:pull', stance_skill, 'Тяга крена бёдер внутрь, Н·м', '≤ 6', at_most(6.), (lower, .25, 1.2), None,
         lambda m: num(m, 1)),
        ('stance:arms', stance_skill, 'Касание рука–корпус > 10 Н, доля шагов', '≤ 10%', at_most(.1),
         (lower, 0., .02), None, share),
        ('stance:shoulder', stance_skill, 'Крен плеч от home, худшее плечо, рад', '≤ 0,08', at_most(.08),
         (lower, .25, .016), None, lambda m: num(m, 3)),
        ('stance:flights', stance_skill, 'Полёт ≥ 40 мс при подъёме: подъёмов', '≤ 10%', at_most_share(.1),
         ('falls',), None, falls),
        ('stance:touchdown', stance_skill, 'Касание стопы, м/с вниз: медиана / max', '≤ 1,0', at_most(1.),
         (lower, .25, .2), None, spread('max', 2)),
        ('stance:force', stance_skill, 'Пик силы на стопу, Н: max / медиана', '≤ 700', at_most(700.),
         (lower, .25, 140.), None, spread('median', 0)),
        ('handover:turn', over, 'Поворот за 5 с стоя после передачи, медиана |°|, наша физика', '≤ 8', at_most(8.),
         (lower, .25, 1.6), None, lambda m: num(m, 1)),
        ('handover:turn15', over, 'Поворот больше 15°: передач, наша физика', '≤ 12,5%', at_most_share(.125),
         ('falls',), None, falls),
        ('handover:falls', over, 'Падения в первые 5 с после передачи, все физики', '0', at_most(0), ('falls',),
         None, falls),
        ('loads:wrists', loads, 'Закреплённые запястья > 90% пика момента: доля времени, наша физика', '≤ 3%',
         at_most(.03), (lower, .25, .006), None, share),
        ('loads:free', loads, 'Остальные моторы > 90% пика: доля времени, наша физика', 'не судится', None,
         (lower, .25, .01), None, share),
        ('loads:hits', loads, 'Удары головой или корпусом о пол > 300 Н: стартов, все физики', '0', at_most(0),
         ('falls',), None, falls),
        ('loads:early', loads, 'Моменты в первые 0,2 с > 60% пика: стартов, все физики', '0', at_most(0),
         ('falls',), None, falls),
    ]


# --- Rows ---------------------------------------------------------------------------------------------------------

def table():
    """The rows: (key, skill, measure, threshold text, threshold (measure -> bool) or None, comparison with the
    reference, needs, show (measure -> text)). Comparisons: ('lower' | 'higher', share, floor): within the larger of
    share x the reference's value and floor; ('falls',): at most 1 + runs // 20 falls more; ('rate',): overlapping 95 %
    intervals; None: none. needs: 'height' (task_config height_command), 'arms' (arm_targets), 'stage' (the probe
    needs a stage folder, not an export). The Russian texts are catalog keys (humanoid_sim/i18n.py)."""
    def num(m, digits=2):
        v = m.get('v')
        return '—' if v is None else 'упал' if v == math.inf else f'{v:.{digits}f}'.replace('.', ',')

    def deg(m):
        return num(m, 1)

    def drift(m):
        return num(m, 3)

    def falls(m):
        n, runs = m['v'], m['n']
        return f'{n} из {runs}'

    def share(m):
        v = m.get('v')
        return '—' if v is None else f'{100 * v:.1f}%'.replace('.', ',')

    def rate(m):
        return f'{100 * m["v"]:.1f}% [{100 * m["lower"]:.1f}–{100 * m["upper"]:.1f}]'.replace('.', ',')

    def chatter(m):
        text = num(m, 3)
        return f'{text} (упал)' if m.get('falls') else text

    def hoist(m):
        return f'p95 {m["v"]:.2f}, |Δa| {m["da"]:.3f}'.replace('.', ',')

    def below(limit):
        return lambda m: m.get('v') is not None and m['v'] < limit

    def at_most(limit):
        return lambda m: m.get('v') is not None and m['v'] <= limit

    def at_least(limit):
        return lambda m: m.get('v') is not None and m['v'] >= limit

    def v0_gate(m):  # None: no Unitree v0 runs (its policy is read from WSL), not judged
        if m.get('v0') is None:
            return None
        return m.get('v') is not None and not m['falls'] and m['v'] <= ch.GATE_FACTOR * m['v0']

    lower, higher = 'lower', 'higher'
    walk, speed, turns, side = 'Ходьба (условия goal_check)', 'Скорость', 'Повороты', 'Шаги вбок'
    rows = [(f'goal:{k}', walk, None, None, None, compare, None, None) for k, compare in enumerate((
        ('falls',), ('near', 10., 1.), (lower, .25, 2.), (lower, .25, .06), None, (higher, 0., .02),
        (higher, .1, .05), (lower, .25, .01), (lower, 0., .06), (higher, 0., .006)))]
    rows += [
        ('speed:0.5', speed, 'Вперёд 0,5 м/с: скорость к команде, худшее зерно', '≥ 0,9', at_least(.9),
         (higher, 0., .05), None, num),
        ('speed:1', speed, 'Вперёд 1 м/с: скорость к команде, худшее зерно', '≥ 0,9', at_least(.9),
         (higher, 0., .05), None, num),
        ('speed:2', speed, 'Вперёд 2 м/с: скорость к команде, худшее зерно', '≥ 0,9', at_least(.9),
         (higher, 0., .05), None, num),
        ('speed:-0.5', speed, 'Назад 0,5 м/с: скорость к команде, худшее зерно', '≥ 0,9', at_least(.9),
         (higher, 0., .05), None, num),
        ('turn:place', turns, 'На месте ±0,5 рад/с: доля команды, худшая сторона', '≥ 0,5', at_least(.5),
         (higher, 0., .1), None, num),
        ('turn:walk', turns, 'На ходу 0,5 м/с ±0,5 рад/с: доля команды, худшая сторона', '≥ 0,5', at_least(.5),
         (higher, 0., .1), None, num),
        ('side:03', side, '±0,3 м/с, наша физика: увод, рад/с', '< 0,05', below(ss.DRIFT_LIMIT), (lower, .25, .01),
         None, drift),
        ('side:05', side, '±0,5 м/с, наша физика: увод, рад/с', '< 0,05', below(ss.DRIFT_LIMIT), (lower, .25, .01),
         None, drift),
        ('side:stop', side, 'Стоп сразу после шага вбок: сдвиг, м', '< 0,3', below(ss.STOP_LIMIT), (lower, .25, .06),
         None, num),
        ('side:1', side, '±1 м/с, копия связки с капсулами стоп: падения', '0', at_most(0), ('falls',), None, falls),
        ('side:1drift', side, '±1 м/с, копия связки: увод, рад/с', 'не судится', None, (lower, .25, .01), None, drift),
        ('brake:0.5', 'Торможение', 'После 0,5 м/с: путь за 3 с, м', '≤ 0,25', at_most(.25), (lower, .25, .05), None,
         num),
        ('brake:1', 'Торможение', 'После 1 м/с: путь за 3 с, м', '≤ 0,5', at_most(.5), (lower, .25, .1), None, num),
        ('push:stand', 'Толчки (копия связки, капсулы стоп)',
         'Стоя: 8 направлений × Δv 1,0 / 1,25 × 2 момента, падения', '≤ 1', at_most(1), ('falls',), None, falls),
        ('push:walk', 'Толчки (копия связки, капсулы стоп)', 'На ходу 0,5 м/с: 8 направлений × Δv 1,0 × 2 фазы, падения',
         '0', at_most(0), ('falls',), None, falls),
        ('phys:fric', 'Устойчивость к физике', 'Ходьба 0,5 м/с, трение моторов ×0,25: увод, рад/с', '< 0,05',
         below(ss.DRIFT_LIMIT), (lower, .25, .01), None, drift),
        ('phys:nomotor', 'Устойчивость к физике', 'Ходьба 0,5 м/с без модели мотора: увод, рад/с', '< 0,05',
         below(ss.DRIFT_LIMIT), (lower, .25, .01), None, drift),
        ('phys:ours_6ms', 'Устойчивость к физике', 'Ходьба 0,5 м/с, наша +6 мс: увод, рад/с', '< 0,05',
         below(ss.DRIFT_LIMIT), (lower, .25, .01), None, drift),
        ('feet:1', 'Стопы', 'Шаги вбок ±1 м/с: доля шагов с перекрытием капсул стоп', 'не судится', None,
         (lower, 0., .02), None, share),
        ('armp:falls', 'Руки в любой позе (±1 рад)', 'Падения: стоя, 0,5 и 1 м/с, 3 позы', '0', at_most(0),
         ('falls',), None, falls),
        ('armp:turn', 'Руки в любой позе (±1 рад)', 'Стоя: худший поворот, °', '≤ 10', at_most(ap.TURN),
         (lower, .25, 2.), None, deg),
        ('armp:creep', 'Руки в любой позе (±1 рад)', 'Стоя: худшее уползание, м', '≤ 0,2', at_most(ap.CREEP),
         (lower, .25, .04), None, num),
        ('armp:drift', 'Руки в любой позе (±1 рад)', 'На ходу 0,5 и 1 м/с: худший увод, рад/с', '≤ 0,05',
         at_most(ap.RATE), (lower, .25, .01), None, drift),
        ('crouch:stand70', 'Присед', 'Стоя на 0,70: таз, м', '≤ 0,72', at_most(.72), (lower, 0., .01), ('height',),
         num),
        ('crouch:stand64', 'Присед', 'Стоя на 0,64: таз, м', '≤ 0,66', at_most(.66), (lower, 0., .01), ('height',),
         num),
        ('crouch:walk', 'Присед', 'Ходьба 0,5 м/с на 0,70: таз, м', '≤ 0,72', at_most(.72), (lower, 0., .01),
         ('height',), num),
        ('crouch:speed', 'Присед', 'Ходьба 0,5 м/с на 0,70: скорость, м/с', '≥ 0,4', at_least(.4), (higher, 0., .08),
         ('height',), num),
        ('crouch:bridge', 'Присед', 'Ходьба 0,5 м/с на 0,70, копия связки: увод, рад/с', '< 0,03',
         below(ss.CROUCH_DRIFT_LIMIT), (lower, .25, .006), ('height',), drift),
        ('joints:roll0', 'Руки по целям', 'Поза «нести», 0 кг: крен плеча |e|, худшая рука, рад', '≤ 0,15',
         at_most(.15), (lower, .2, .03), ('arms', 'stage'), num),
        ('joints:roll1', 'Руки по целям', 'Поза «нести», 1 кг: крен плеча |e|, худшая рука, рад', '≤ 0,15',
         at_most(.15), (lower, .2, .03), ('arms', 'stage'), num),
        ('joints:pitch', 'Руки по целям', 'Поза «нести»: провис тангажа плеча, рад', '≤ 0,15', at_most(.15),
         (lower, .2, .03), ('arms', 'stage'), num),
        ('joints:elbow', 'Руки по целям', 'Поза «нести»: локоть |e|, рад', '≤ 0,10', at_most(.1), (lower, .2, .02),
         ('arms', 'stage'), num),
        ('joints:error0', 'Руки по целям', 'Поза «нести», 0 кг: средняя |e|, рад', '≤ 0,07', at_most(.07),
         (lower, .2, .014), ('arms', 'stage'), num),
        ('joints:error1', 'Руки по целям', 'Поза «нести», 1 кг: средняя |e|, рад', '≤ 0,09', at_most(.09),
         (lower, .2, .018), ('arms', 'stage'), num),
        ('joints:palms', 'Руки по целям', 'Поза «нести»: ладони против цели, см', '≤ 5', at_most(5.), (lower, .2, 1.),
         ('arms', 'stage'), lambda m: num(m, 1)),
        ('falls:free', 'Руки по целям', 'Падения в обучающих условиях, руки свободны, на эпизод', '≤ 1,5%',
         at_most(.015), ('rate',), ('stage',), rate),
        ('falls:target:1', 'Руки по целям', 'Падения в обучающих условиях, цели рук, 1 кг, на эпизод', '≤ 4%',
         at_most(.04), ('rate',), ('arms', 'stage'), rate),
        ('chatter:torque_hf_rms_nm', 'Дрожание моментов', 'Момент выше 10 Гц, худший сустав, Н·м', '≤ 1,5 × v0',
         v0_gate, (lower, .2, 0.), None, chatter),
        ('chatter:action_hf_rms_rad', 'Дрожание моментов', 'Цель выше 5 Гц, худший сустав, рад', '≤ 1,5 × v0',
         v0_gate, (lower, .2, 0.), None, chatter),
        ('hoist', 'Подвес', 'В воздухе при команде 0: скорость ног p95, рад/с, и |Δa|', '≤ 2,0 и ≤ 0,03',
         lambda m: pk.air_calm([dict(qv_leg_p95=m['v'], da=m['da'])]), (lower, .25, .4), ('stage',), hoist),
    ]
    return rows


def not_applicable(needs, t):
    """Why a row that needs `needs` (a tuple, or None) is not for this policy (text), or None."""
    for need in needs or ():
        if need == 'height' and not t['height']:
            return 'н/п: нет команды высоты'
        if need == 'arms' and not t['arms']:
            return 'н/п: нет arm_targets'
        if need == 'stage' and t['export']:
            return 'н/п: проба не берёт выгрузку'
    return None


def group_needs(key):
    """Rows of group 11 are run only when the judged stage has arm targets."""
    return key.startswith(('joints:', 'falls:'))


def compared(compare, mine, theirs):
    """Whether `mine` is not worse than `theirs` within the comparison's margin (None: nothing to compare)."""
    if compare is None or theirs is None or mine.get('v') is None or theirs.get('v') is None:
        return None
    kind = compare[0]
    if kind == 'falls':
        return mine['v'] <= theirs['v'] + 1 + theirs['n'] // 20
    if kind == 'rate':
        return mine['lower'] <= theirs['upper']
    a, b = mine['v'], theirs['v']
    if kind == 'near':
        return abs(a - compare[1]) <= abs(b - compare[1]) + compare[2]
    if math.isinf(b):  # the reference fell: anything is not worse
        return True
    margin = max(compare[1] * abs(b), compare[2])
    return a <= b + margin if kind == 'lower' else a >= b - margin


def verdict(judged, reference_ok, relative, mine_na, theirs_na):
    """(ok, note) of a row: judged, the threshold on the stage (None: no threshold); reference_ok, on the reference;
    relative, the comparison with it (None: none); mine_na / theirs_na: why a policy cannot run it."""
    if mine_na:
        return None, mine_na
    if judged is None and relative is None:
        return None, 'не судится'
    threshold = judged is not False or reference_ok is False
    ok = threshold and relative is not False
    if relative is False:
        return False, 'порог не взят, хуже эталона' if judged is False else 'хуже эталона'
    if judged is False:
        return ok, 'порог не взят, как и у эталона' if reference_ok is False else 'порог не взят'
    if judged is None:
        return ok, 'не хуже эталона'
    if relative is None:
        return ok, 'порог; эталон н/п' if theirs_na else 'порог'
    return ok, 'порог и эталон'


def rows(mine, theirs, t_mine, t_theirs, arms_group, getup=False):
    """Table rows: {key, skill, label, target, value, reference, ok, note} from both policies' measures; getup: the
    rows of a getting-up stage (getup_table)."""
    out = []
    for key, skill, label, target, good, compare, needs, show in (getup_table() if getup else table()):
        mine_na = not_applicable(needs, t_mine)
        theirs_na = None if theirs is None else not_applicable(needs, t_theirs)
        if group_needs(key) and not arms_group:
            mine_na = theirs_na = 'н/п: нет arm_targets'
        m, r = mine.get(key), None if theirs is None or theirs_na else theirs.get(key)
        if key.startswith('goal:'):  # goal_check's own row: its label, target, value and tick
            known = m or r or {}
            label, target = known.get('label'), known.get('target')
            show, good = (lambda x: x['text']), (lambda x: x['ok'])
        reference = '—' if theirs is None else 'н/п' if theirs_na else '—' if r is None else show(r)
        if m is None and not mine_na:
            out.append(dict(key=key, skill=skill, label=label or key, target=target or '', value='—',
                            reference=reference, ok=False, note='не измерено'))
            continue
        judged = None if mine_na or good is None else good(m)
        reference_ok = None if r is None or good is None else good(r)
        judged, reference_ok = (None if x is None else bool(x) for x in (judged, reference_ok))
        relative = None if mine_na else compared(compare, m, r)
        ok, note = verdict(judged, reference_ok, relative, mine_na, theirs_na if theirs is not None else None)
        if key.startswith('chatter:') and m.get('v0') is not None:
            target = f'≤ 1,5 × v0 ({m["v0"]:.3f})'.replace('.', ',')
        out.append(dict(key=key, skill=skill, label=label, target=target, ok=ok, note=note,
                        value='—' if mine_na else show(m), reference=reference))
    return out


# --- Main ---------------------------------------------------------------------------------------------------------

def clean(value):
    """value for strict JSON: ±infinity (a fall in a measure) as the string 'inf' / '-inf', NaN as None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else 'inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def unclean(value):
    """clean() undone for the measures of a --json file: 'inf' / '-inf' back to floats."""
    if isinstance(value, dict):
        return {k: unclean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unclean(v) for v in value]
    return float(value) if value in ('inf', '-inf') else value


def print_rows(table_rows, stage_label, reference_label):
    marks = {True: '✓', False: '✗', None: '·'}
    print(f'{"":2}{"навык · мера":<70} {"порог":<22} {"этап":>16} {"эталон":>16}  итог')
    skill = None
    for row in table_rows:
        if row['skill'] != skill:
            skill = row['skill']
            print(f'— {skill}')
        print(f'{marks[row["ok"]]} {row["label"][:70]:<70} {row["target"][:22]:<22} {row["value"]:>16} '
              f'{row["reference"]:>16}  {row["note"]}')
    judged = [r for r in table_rows if r['ok'] is not None]
    print(f'{stage_label} против {reference_label}: выполнено {sum(r["ok"] for r in judged)} из {len(judged)}, '
          f'н/п {len(table_rows) - len(judged)}')


def title_of(path):
    """'<branch name> · <steps> млн' (an export: with its folder) and (branch, steps) of a policy."""
    from humanoid_sim.core import read_json
    from humanoid_sim.export_g1 import is_export
    stage = stage_of(path)
    branch, steps = stage.parent.name, int(stage.name[6:])
    name = (read_json(stage.parent / 'experiment.json', {}) or {}).get('name', branch)
    millions = f'{steps / 1e6:.1f}'.replace('.', ',')
    title = f'{name} · {millions} млн' + (f' · выгрузка {Path(path).name}' if is_export(path) else '')
    return title, branch, steps


def run_check(path, reference, workers, getup=False):
    """Every run of the stage, the reference (none when it is the same) and Unitree v0's chatter; the measures.
    getup: a getting-up stage, getup_jobs on fresh starts (getup_stance.fresh_starts, made first) instead."""
    same = reference == path
    t_mine = traits(path)
    t_theirs = t_mine if same else traits(reference)
    arms_group = t_mine['arms']
    v0 = not getup and Path(st.UNITREE_V0).is_file()
    delays = sorted({t_mine['delay'], t_theirs['delay']})
    if getup:
        started = time.time()
        starts = gs.fresh_starts(GETUP_PER_CLASS, workers)
        limits = pk.getup_patch_info(pk.SIM2SIM / 'patches' / pk.GETUP_PATCH)['max_joint_speed']
        print(f'стартов {len(starts)}, {time.time() - started:.0f} с', flush=True)
        items = getup_jobs('stage', t_mine, starts, limits) + ([] if same else getup_jobs('reference', t_theirs,
                                                                                            starts, limits))
    else:
        items = jobs_of('stage', t_mine, arms_group) + ([] if same else jobs_of('reference', t_theirs, arms_group))
    if v0:
        items += [(f'v0:{delay:g}', f'chatter:{job["scenario"]}:{job["seed"]}', 'chatter', job)
                  for delay in delays for job in ch.reference_jobs(delay)]
    order = sorted(range(len(items)), key=lambda k: -cost(items[k][2], items[k][3]))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    print(f'этап {t_mine["label"]}, эталон {t_theirs["label"]}{" (он же)" if same else ""}: прогонов {len(items)}, '
          f'процессов {workers}', flush=True)
    results, seconds = [None] * len(items), [0.] * len(items)
    started = time.time()
    with Pool(workers, initializer=init, maxtasksperchild=32) as pool:
        done = 0
        for index, result, spent in pool.imap_unordered(work, [(k, items[k][2], items[k][3]) for k in order]):
            results[index], seconds[index] = result, spent
            done += 1
            if done % 100 == 0 or done == len(items):
                print(f'  {done} из {len(items)}, {time.time() - started:.0f} с', flush=True)

    def of(owner):
        return {items[k][1]: results[k] for k in range(len(items)) if items[k][0] == owner}
    v0_runs = {delay: list(of(f'v0:{delay:g}').values()) for delay in delays} if v0 else {}
    if getup:
        mine = getup_measures(of('stage'))
        theirs = mine if same else getup_measures(of('reference'))
    else:
        mine = measures(of('stage'), t_mine, v0_runs.get(t_mine['delay']))
        theirs = mine if same else measures(of('reference'), t_theirs, v0_runs.get(t_theirs['delay']))
    return dict(mode='getup' if getup else 'walking', stage=t_mine, reference=t_theirs, same=same,
                measures=dict(stage=mine, reference=theirs), runs=len(items), seconds=time.time() - started,
                process_seconds=sum(seconds), workers=workers,
                counts={owner: sum(item[0] == owner for item in items) for owner in ('stage', 'reference')}
                | {'v0': sum(item[0].startswith('v0') for item in items)})


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Проверка навыков: не забыл ли этап то, что умел эталон')
    parser.add_argument('stage', nargs='?', help='папка этапа или выгрузки export_g1, или <ветка>[:шаги]')
    parser.add_argument('--against', default='parent',
                        help='parent (по умолчанию), n8, getup823 (вставание) или этап / ветка[:шаги]')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--journal', action='store_true', help='записать в «Дневник» рядом с проверкой цели')
    parser.add_argument('--rejudge', type=Path, help='--json прошлого прогона: судить его меры заново, без прогонов')
    args = parser.parse_args(argv)
    if (args.stage is None) == (args.rejudge is None):
        parser.error('нужен этап или --rejudge <json>, что-то одно')
    if args.rejudge:
        check = unclean(json.loads(args.rejudge.read_text(encoding='utf-8')))
        path, reference = Path(check['stage']['path']), Path(check['reference']['path'])
    else:
        path = policy_path(args.stage)
        reference = reference_path(args.against, path).resolve()
        getup = task_of(path) == 'g1_getup'
        if getup != (task_of(reference) == 'g1_getup'):
            raise SystemExit(f'Этап и эталон — разные задачи ({task_of(path)} и {task_of(reference)})')
        check = run_check(path, reference, args.workers, getup)
    getup = check.get('mode') == 'getup'
    t_mine, t_theirs = check['stage'], check['reference']
    table_rows = rows(check['measures']['stage'], check['measures']['reference'], t_mine, t_theirs, t_mine['arms'],
                      getup)
    check['rows'] = table_rows
    print_rows(table_rows, t_mine['label'], t_theirs['label'])
    counts, wall = check['counts'], check['seconds']
    print(f'{wall:.0f} с на {check["workers"]} процессах ({check["process_seconds"]:.0f} с процессов), прогонов '
          f'{check["runs"]}: этап {counts["stage"]}, эталон {counts["reference"]}, v0 {counts["v0"]}')
    if args.json:
        args.json.write_text(json.dumps(clean(check), ensure_ascii=False, indent=1), encoding='utf-8')
    if args.journal:
        from humanoid_sim import journal
        from humanoid_sim.core import DEFAULT_RUN
        from humanoid_sim.lab import Laboratory
        lab = Laboratory(DEFAULT_RUN)
        title, branch, steps = title_of(path)
        if getup:
            title = f'{title} · вставание'
        if stage_of(path).parent.parent.resolve() != lab.root.resolve():
            raise SystemExit(f'Этап не из лаборатории ({path}): в «Дневник» не записать')
        against, against_branch, against_steps = title_of(reference)
        journal.set_skills(lab.root, branch, steps, title, dict(branch=against_branch, stage=against_steps,
                                                                title=against), table_rows, seconds=wall,
                           runs=check['runs'])
        print('записано в «Дневник»')


if __name__ == '__main__':
    main()
