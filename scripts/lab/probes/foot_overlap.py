"""CPU only. foot_overlap.py <stage dir | export dir | unitree_v0> [...] [--workers 8] [--out x.json]

Do the feet pass through each other? Our G1 feet touch the world only with four 5 mm sole spheres (the foot mesh has
no collision), so one foot passes through the other; Unitree's training model (unitree_rl_mjlab g1.xml) gives each
foot 7 capsules of radius 0.01 that collide, foot against foot included (task_config foot_collision adds exactly
those, colliding only with each other). Per stage, with Unitree's capsules (tasks/g1_velocity.FOOT_CAPSULES, the
exact capsule-capsule distance): the share of control steps where the two feet's capsules overlap (distance < 0)
while both soles are below 2 cm, and the smallest distance, in
  * side steps (0, ±0.5, 0) and (0, ±1, 0) m/s in the stage's own CPU task (heading hold as trained, a reset with the
    command at gait phase 0, seed 73000, deterministic, 8 s, measured after 2 s: the critic's foot_gap.py);
  * sideways pushes standing in the bridge copy (bridge_kick.run: Unitree physics, PD on a 2 ms older state, 6 ms
    delay, a force M·dv/0.1 s on the pelvis), left and right, dv 1.0 and 1.25, 4 moments (bridge_kick.OFFSETS), the
    3 s after the push, one sample per control step, and how many of the 16 pushes overlap at all (push_feet.py).
In its own task a foot_collision stage's feet collide, so there its overlap is their contact; the bridge copy has no
foot collision (unitree_mujoco's model, um_probe.apply_physics), so there it is the policy's own foot placement.
The critic's numbers (iteration 23, lab-critic/it23, with 13 points sampled along each capsule, which can only miss an
overlap): side ±1 candidate 6 (exp-8ce0cbd70d0a4f1d @ 6039109632) 9-10 %, 6159 7-9 % (2 % at +0.5), Unitree v0 0 %;
sideways pushes 16 of 16 for both, v0 0 of 8. Printed, not judged; stage_select.py prints the same per stage.
About 20 runs a stage, ~1 min on 8 workers.
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
import stand_turn as st  # noqa: E402
import um_probe as um  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.g1_velocity import FOOT_CAPSULE_RADIUS, FOOT_CAPSULES  # noqa: E402

um.Stepper = kick.PushStepper.__bases__[0]  # bridge_kick puts its push stepper in on import; only push_run uses one

SOLES_DOWN = .02  # m, both soles below this: the feet overlap on the ground, not one passing over the other
SIDE_COMMANDS = (('влево 0.5', (0., .5, 0.)), ('вправо 0.5', (0., -.5, 0.)), ('влево 1', (0., 1., 0.)),
                 ('вправо 1', (0., -1., 0.)))
SIDE_SEED, SIDE_STEPS, SIDE_SETTLE = 73000, 400, 100  # 8 s, measured after 2 s
PUSH_SIDES = {'l': (0., 1.), 'r': (0., -1.)}  # in the pelvis heading, as bridge_kick.DIRS
PUSH_LABELS = dict(l='влево', r='вправо')
PUSH_MAGS = (1., 1.25)
PUSH_WINDOW = 1500  # physics ticks (2 ms) after the push starts: 3 s
SAMPLE_TICKS = 10  # one sample per control step
LOCAL = {name: np.array(capsules, dtype=float).reshape(len(capsules), 2, 3) for name, capsules in FOOT_CAPSULES.items()}


def segment_distance(p1, q1, p2, q2):
    """Closest distance between the segments p1-q1 and p2-q2 (arrays broadcasting to (..., 3), no zero-length
    segment), Ericson's ClosestPtSegmentSegment (Real-Time Collision Detection, 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    denom = a * e - b * b
    general = denom > 1e-12 * a * e  # not parallel: the closest point of the infinite lines, clamped
    s = np.where(general, np.clip((b * f - c * e) / np.where(general, denom, 1.), 0., 1.), 0.)
    t = (b * s + f) / e
    s = np.where(t < 0., np.clip(-c / a, 0., 1.), np.where(t > 1., np.clip((b - c) / a, 0., 1.), s))
    t = np.clip(t, 0., 1.)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def capsule_ends(raw):
    """Per foot (task.FEET order), the world ends of Unitree's capsules, (7, 2, 3), from the ankle_roll frames."""
    ends = []
    for foot, name in zip(raw.feet, task.FEET):
        rotation = raw.data.xmat[foot].reshape(3, 3)
        ends.append(raw.data.xpos[foot] + LOCAL[name] @ rotation.T)
    return ends


def feet_distance(raw):
    """Smallest distance (m) between the left and the right foot's capsules; < 0: they overlap."""
    left, right = capsule_ends(raw)
    distance = segment_distance(left[:, None, 0], left[:, None, 1], right[None, :, 0], right[None, :, 1])
    return float(distance.min() - 2 * FOOT_CAPSULE_RADIUS)


def soles_down(raw):
    return bool((raw._sole_height()[0].numpy() < SOLES_DOWN).all())


class Tally:
    """Samples of one run: how many, how many overlap, how many overlap with both soles down, the smallest distance."""

    def __init__(self):
        self.samples, self.overlap, self.low, self.dmin = 0, 0, 0, math.inf

    def add(self, raw):
        distance = feet_distance(raw)
        self.samples += 1
        self.dmin = min(self.dmin, distance)
        if distance < 0:
            self.overlap += 1
            self.low += soles_down(raw)

    def result(self, job, fell):
        return dict(job={k: v for k, v in job.items() if k != 'path'}, samples=self.samples, overlap=self.overlap,
                    low=self.low, dmin=None if math.isinf(self.dmin) else self.dmin, fell=fell)


class FeetStepper(kick.PushStepper):
    """bridge_kick's push stepper that also samples the feet once per control step in the PUSH_WINDOW after the push."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tally = Tally()

    def substep(self, target):
        super().substep(target)
        since = self.after - kick.PUSH['tick']
        if self.d is not None and 0 < since <= PUSH_WINDOW and since % SAMPLE_TICKS == 0:
            self.tally.add(self.raw)


def jobs(stage):
    """The runs of one stage (a stage dir, an export_g1 folder or 'unitree_v0'): side steps, then sideways pushes."""
    stage = str(stage)
    side = [dict(kind='side', path=stage, name=name, cmd=list(cmd)) for name, cmd in SIDE_COMMANDS]
    pushes = [dict(kind='push', path=stage, left=False, phys='um', dir=d, mag=m, offset=o)
              for d in PUSH_SIDES for m in PUSH_MAGS for o in kick.OFFSETS]
    return side + pushes


def cost(job):
    """Rough run length in physics ticks (stage_select starts the long runs first)."""
    if job['kind'] == 'side':
        return SIDE_STEPS * 10
    return sum(secs for *_, secs in st.PLANS['kstand']) / st.TICK + 3500 + 700


def side_run(job):
    import torch
    from humanoid_sim.tasks.catalog import make_task
    torch.set_num_threads(1)
    model, norm, config = st.stage_data(job['path'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 100000})
    raw, tally, fell = env.unwrapped, Tally(), None
    try:
        obs, _ = env.reset(seed=SIDE_SEED, options={'command': job['cmd'], 'phase_steps': 0})
        for step in range(SIDE_STEPS):
            obs, _, terminated, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if terminated:
                fell = step
                break
            if step >= SIDE_SETTLE:
                tally.add(raw)
    finally:
        env.close()
    return tally.result(job, fell)


def push_run(job):
    """bridge_kick.run with FeetStepper as the stepper, its standing plan and the sideways directions."""
    saved = um.Stepper, kick.PLAN, kick.DIRS
    um.Stepper, kick.PLAN, kick.DIRS = FeetStepper, 'stand', {**kick.DIRS, **PUSH_SIDES}
    try:
        result = kick.run(job)
    finally:
        um.Stepper, kick.PLAN, kick.DIRS = saved
    return kick.PUSH['stepper'].tally.result(job, result['fell'])


def run(job):
    return side_run(job) if job['kind'] == 'side' else push_run(job)


def share(results):
    samples = sum(r['samples'] for r in results)
    return sum(r['low'] for r in results) / samples if samples else None


def lowest(results):
    values = [r['dmin'] for r in results if r['dmin'] is not None]
    return min(values) if values else None


def summary(results):
    """Per side command: overlap share with both soles down, smallest distance, fall step; the sideways pushes per
    direction: pushes with any overlap, overlap share with both soles down, smallest distance, falls."""
    side = {}
    for name, _ in SIDE_COMMANDS:
        rs = [r for r in results if r['job']['kind'] == 'side' and r['job']['name'] == name]
        side[name] = dict(share=share(rs), dmin=lowest(rs), fell=next((r['fell'] for r in rs if r['fell'] is not None),
                                                                     None), runs=len(rs))
    pushes = {}
    for d in PUSH_SIDES:
        rs = [r for r in results if r['job']['kind'] == 'push' and r['job']['dir'] == d]
        pushes[d] = dict(runs=len(rs), overlapping=sum(r['overlap'] > 0 for r in rs), share=share(rs), dmin=lowest(rs),
                         falls=sum(r['fell'] is not None for r in rs))
    every = [r for r in results if r['job']['kind'] == 'push']
    return dict(side=side, pushes=pushes, push_runs=len(every), push_overlapping=sum(r['overlap'] > 0 for r in every),
                push_share=share(every))


def percent(value):
    return '-' if value is None else '%.1f%%' % (100 * value)


def cm(value):
    return '-' if value is None else '%+.1f см' % (100 * value)


def line(feet):
    """One line for stage_select: the side-step shares and the sideways pushes."""
    side = ' / '.join(percent(feet['side'][name]['share']) for name, _ in SIDE_COMMANDS)
    return ('стопы накрест (капсулы Unitree, доля шагов с перекрытием при обеих подошвах ниже 2 см; не судится): '
            'вбок %s — %s; толчки вбок (Unitree +6 мс, Δv %s) — перекрытие в %d из %d, %s шагов за 3 с' % (
                ' / '.join(name for name, _ in SIDE_COMMANDS), side, ' / '.join(kick_dv(m) for m in PUSH_MAGS),
                feet['push_overlapping'], feet['push_runs'], percent(feet['push_share'])))


def kick_dv(mag):
    text = '%.2f' % mag
    return text[:-1] if text.endswith('0') else text


def report(label, feet):
    print('\n== %s' % label)
    print('  вбок (наша физика, 8 с, после 2 с): доля шагов с перекрытием капсул при обеих подошвах ниже 2 см')
    for name, _ in SIDE_COMMANDS:
        s = feet['side'][name]
        print('    %-12s %7s   мин. расстояние %9s%s' % (name, percent(s['share']), cm(s['dmin']),
                                                         '' if s['fell'] is None else '   УПАЛ на шаге %d' % s['fell']))
    print('  толчки вбок стоя (копия связки, Unitree +6 мс, Δv %s, %d момента, 3 с после толчка)' % (
        ' / '.join(kick_dv(m) for m in PUSH_MAGS), len(kick.OFFSETS)))
    for d in PUSH_SIDES:
        p = feet['pushes'][d]
        print('    %-12s перекрытие в %d из %d, %7s шагов, мин. расстояние %9s%s' % (
            PUSH_LABELS[d], p['overlapping'], p['runs'], percent(p['share']), cm(p['dmin']),
            '' if not p['falls'] else ', падений %d' % p['falls']))


def label(path):
    path = Path(path)
    if str(path) == 'unitree_v0':
        return 'Unitree v0'
    if (path / 'policy.onnx').is_file():
        return path.name + ' (выгрузка)'
    return '%s @ %s' % (path.parent.name, path.name[6:].lstrip('0'))


def init():
    import torch
    torch.set_num_threads(1)


def work(item):
    index, job = item
    return index, run(job)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Проходят ли стопы друг сквозь друга (капсулы стоп Unitree)')
    parser.add_argument('stages', nargs='+', help='папка этапа, выгрузка export_g1 или unitree_v0')
    parser.add_argument('--workers', type=int, default=8, help='рабочих процессов (пока идёт обучение, не больше 8)')
    parser.add_argument('--out', type=Path, help='JSON: итоги и все прогоны')
    args = parser.parse_args(argv)
    paths = [p if p == 'unitree_v0' else str(Path(p).resolve()) for p in args.stages]
    for p in paths:
        if p != 'unitree_v0' and not ((Path(p) / 'policy.onnx').is_file() or (Path(p) / 'policy.zip').is_file()):
            raise SystemExit(f'Не этап и не выгрузка export_g1: {p}')
    items = [(i, job) for i, p in enumerate(paths) for job in jobs(p)]
    order = sorted(range(len(items)), key=lambda k: -cost(items[k][1]))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # one core per worker
        os.environ.setdefault(key, '1')
    print('этапов: %d, прогонов: %d, процессов: %d' % (len(paths), len(items), args.workers), flush=True)
    results, t0 = [None] * len(items), time.time()
    with Pool(args.workers, initializer=init, maxtasksperchild=16) as pool:
        for k, result in pool.imap_unordered(work, [(k, items[k][1]) for k in order]):
            results[k] = result
    stages = []
    for i, p in enumerate(paths):
        feet = summary([results[k] for k, item in enumerate(items) if item[0] == i])
        report(label(p), feet)
        stages.append(dict(label=label(p), path=p, feet=feet,
                           results=[results[k] for k, item in enumerate(items) if item[0] == i]))
    print('\n%.0f с на %d процессах' % (time.time() - t0, args.workers))
    if args.out:
        args.out.write_text(json.dumps(dict(stages=stages), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
