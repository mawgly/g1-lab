"""CPU only. foot_shock.py <A|B|C> <stage dir | export dir | unitree_v0> [...] [--workers 8] [--seeds 24] [--out x.json]

How big is the shock when the feet start to collide (task_config foot_collision: Unitree's foot capsules, colliding
foot against foot with Unitree's contact, condim 3, friction 0.6, priority 1), measured on stages trained without it,
before any training with it (the critic's lab-critic/it24/shock.py, iteration 24). Each run twice, the capsules off
('off', the stage's model without them) and on ('on', as task_config foot_collision builds them; the critic's 'c3'),
whatever the stage's own setting:
  A  training-like episodes in the stage's own CPU task (a training reset: random commands, force pushes, arm
     perturbation, random phase, as its task_config says), 1000 steps, the policy with its own σ noise ('noise') and
     deterministic ('det'), paired seeds 91000..: falls (and those only with or only without the capsules), episode
     length, reward per step, self_contact share, steps with a capsule-capsule contact, overlap of Unitree's capsules
     with both soles down when off (foot_overlap), the largest capsule force. 'noise' needs an SB3 stage (its log σ).
  B  sideways pushes standing in the bridge procedure (bridge_kick.run: FixStand on the band, PD on a 2 ms older state,
     6 ms delay, force M·dv/0.1 s on the pelvis), our physics ('ours') and Unitree's ('um'; with the capsules on that
     is um_probe's 'um_caps', the feet as Unitree's training model has them), left / right, dv 1.0 / 1.25 / 1.5, 4
     moments: falls, pushes with an overlap, the largest capsule force and the time above 10 N in the 3 s after.
  C  side commands (0, ±0.5, 0) and (0, ±1, 0) in the stage's own task, deterministic, 8 s from a reset with the
     command (seed 73000 at phase 0, 73001 at phase 15), measured after 2 s: falls, sideways speed, self_contact share,
     capsule contact share, overlap share when off, the largest capsule force.
The critic's numbers (6159 = exp-94b331444b3b4788 @ 6159433728, candidate 6 = exp-8ce0cbd70d0a4f1d @ 6039109632, the
iteration-23 branch exp-64a96a60e591468b @ 6199-6209 M): frictionless (condim 1, the first version) the stages almost
never fell (A: 0-1 of 24, the feet sliding past each other at 400-1000 N, contact in 0.2-0.4 % of steps); with
Unitree's contact A falls 1-7 of 24 per stage and mode, B 2-4 of 4 in most sideways cells at dv 1.25-1.5 (6159 to the
right from dv 1.0, in both physics), C at ±1 m/s 2 of 2 for every stage (6159 also once at +0.5).
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
import foot_overlap as fo  # noqa: E402  (imports bridge_kick, stand_turn, um_probe)

kick, st, um = fo.kick, fo.st, fo.um
CAPS = {'on': False}  # this job's variant, read by the stage_data wrapper (one job at a time per process)
VARIANTS = (('off', False), ('on', True))
A_SEED, A_STEPS = 91000, 1000
B_MAGS = (1., 1.25, 1.5)
C_COMMANDS = ((0., .5, 0.), (0., -.5, 0.), (0., 1., 0.), (0., -1., 0.))
C_STARTS = ((73000, 0), (73001, 15))
C_STEPS, C_SETTLE = 400, 100
TOUCH_FORCE = 10.  # N, as the self_contact term's threshold
_original_stage_data = st.stage_data


def stage_data(stage):
    """The stage's policy, normalizer and task config, with foot_collision as this job's variant."""
    model, norm, config = _original_stage_data(stage)
    return model, norm, {**config, 'foot_collision': CAPS['on']}


def capsule_force(raw):
    """Largest capsule-capsule contact force (N) at the current physics step, 0 without one."""
    import mujoco
    if not raw.foot_capsules.size or not raw.data.ncon:
        return 0.
    capsules, force, best = set(raw.foot_capsules.ravel().tolist()), np.zeros(6), 0.
    for k in range(raw.data.ncon):
        geom = raw.data.contact.geom[k]
        if int(geom[0]) in capsules and int(geom[1]) in capsules:
            mujoco.mj_contactForce(raw.model, raw.data, k, force)
            best = max(best, float(np.linalg.norm(force[:3])))
    return best


def run_a(job):
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = stage_data(job['path'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': A_STEPS})
    raw = env.unwrapped
    rng = np.random.default_rng(job['seed'] + 7)
    sigma = np.exp(model.policy.log_std.detach().numpy()) if job['mode'] == 'noise' else None
    weight = max(config['reward_weights']['self_contact'], 1e-9)
    steps, reward, share, touch, overlap, fmax, fell = 0, 0., 0., 0, 0, 0., False
    try:
        obs, _ = env.reset(seed=job['seed'])  # a training reset: no command given
        while True:
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            if sigma is not None:
                action = action + sigma * rng.standard_normal(action.shape)
            obs, r, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            steps += 1
            reward += r
            share += -info['reward_components']['self_contact'] / weight
            if job['caps']:
                force = capsule_force(raw)
                touch += force > 0
                fmax = max(fmax, force)
            else:
                overlap += fo.feet_distance(raw) < 0 and fo.soles_down(raw)
            if terminated:
                fell = True
                break
            if truncated:
                break
    finally:
        env.close()
    return dict(job={k: v for k, v in job.items() if k != 'path'}, steps=steps, reward=reward, sc=share, touch=touch,
                overlap=overlap, fmax=fmax, fell=fell)


class ShockStepper(fo.FeetStepper):
    """foot_overlap's push stepper that also keeps the largest capsule force and the substeps above TOUCH_FORCE in the
    window after the push."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fmax, self.ftouch = 0., 0

    def substep(self, target):
        super().substep(target)
        since = self.after - kick.PUSH['tick']
        if self.d is not None and 0 < since <= fo.PUSH_WINDOW:
            force = capsule_force(self.raw)
            self.fmax = max(self.fmax, force)
            self.ftouch += force > TOUCH_FORCE


def run_b(job):
    saved = um.Stepper, kick.PLAN, kick.DIRS, st.stage_data
    um.Stepper, kick.PLAN, kick.DIRS, st.stage_data = ShockStepper, 'stand', {**kick.DIRS, **fo.PUSH_SIDES}, stage_data
    physics = job['phys'] + ('_caps' if job['caps'] and job['phys'] == 'um' else '')  # um_caps: capsules kept
    try:
        result = kick.run(dict(path=job['path'], left=False, phys=physics, dir=job['dir'], mag=job['mag'],
                               offset=job['offset']))
    finally:
        um.Stepper, kick.PLAN, kick.DIRS, st.stage_data = saved
    stepper = kick.PUSH['stepper']
    tally = stepper.tally
    return dict(job={k: v for k, v in job.items() if k != 'path'}, fell=result['fell'],
                fell_before=result['fell_before'], overlap=tally.overlap, low=tally.low, samples=tally.samples,
                dmin=None if math.isinf(tally.dmin) else tally.dmin, fmax=stepper.fmax,
                ftouch_s=stepper.ftouch * st.TICK, zmin=result['zmin'], vmax=result['vmax'])


def run_c(job):
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = stage_data(job['path'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 100000})
    raw = env.unwrapped
    weight = max(config['reward_weights']['self_contact'], 1e-9)
    fell, vy, share, touch, fmax, overlap, n = None, [], 0., 0, 0., 0, 0
    try:
        obs, _ = env.reset(seed=job['seed'], options={'command': job['cmd'], 'phase_steps': job['phase']})
        for step in range(C_STEPS):
            obs, _, terminated, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if terminated:
                fell = step
                break
            if step >= C_SETTLE:
                n += 1
                vy.append(float(info['velocity'][1]))
                share += -info['reward_components']['self_contact'] / weight
                if job['caps']:
                    force = capsule_force(raw)
                    touch += force > 0
                    fmax = max(fmax, force)
                else:
                    overlap += fo.feet_distance(raw) < 0 and fo.soles_down(raw)
    finally:
        env.close()
    return dict(job={k: v for k, v in job.items() if k != 'path'}, fell=fell, n=n,
                vy=float(np.mean(vy)) if vy else None, sc=share / max(n, 1), touch=touch / max(n, 1),
                overlap=overlap / max(n, 1), fmax=fmax)


def run(job):
    CAPS['on'] = job['caps']
    return {'A': run_a, 'B': run_b, 'C': run_c}[job['part']](job)


def sb3(path):
    return path != 'unitree_v0' and (Path(path) / 'policy.zip').is_file()


def jobs(part, stages, seeds=24):
    """stages: {label: path}. The runs of one part, each with the capsules off and on."""
    out = []
    for name, path in stages.items():
        if part == 'A':
            for mode in (('noise', 'det') if sb3(path) else ('det',)):
                for seed in range(A_SEED, A_SEED + seeds):
                    out += [dict(part='A', name=name, path=path, mode=mode, seed=seed, caps=caps) for _, caps in VARIANTS]
        elif part == 'B':
            for phys in ('ours', 'um'):
                for _, caps in VARIANTS:
                    out += [dict(part='B', name=name, path=path, phys=phys, caps=caps, dir=d, mag=m, offset=o)
                            for d in fo.PUSH_SIDES for m in B_MAGS for o in kick.OFFSETS]
        elif part == 'C':
            for cmd in C_COMMANDS:
                for seed, phase in C_STARTS:
                    out += [dict(part='C', name=name, path=path, cmd=list(cmd), seed=seed, phase=phase, caps=caps)
                            for _, caps in VARIANTS]
        else:
            raise ValueError(f'part A, B or C, not {part}')
    return out


def tag(caps):
    return 'on' if caps else 'off'


def report(part, stages, results):
    of = lambda **match: [r for r in results if all(r['job'].get(k) == v for k, v in match.items())]
    if part == 'A':
        for name in stages:
            for mode in ('noise', 'det'):
                base = {r['job']['seed']: r for r in of(name=name, mode=mode, caps=False)}
                for _, caps in VARIANTS:
                    rs = of(name=name, mode=mode, caps=caps)
                    if not rs:
                        continue
                    steps = sum(r['steps'] for r in rs)
                    only_on = sum(r['fell'] and not base[r['job']['seed']]['fell'] for r in rs)
                    only_off = sum(base[r['job']['seed']]['fell'] and not r['fell'] for r in rs)
                    print(f'{name:24s} {mode:5s} {tag(caps):3s}: falls {sum(r["fell"] for r in rs):2d}/{len(rs)}'
                          f' (vs off: +{only_on} -{only_off}), ep_len {steps / len(rs):6.0f}, '
                          f'rew/step {sum(r["reward"] for r in rs) / steps:+.3f}, '
                          f'self_contact {sum(r["sc"] for r in rs) / steps:.4f}, '
                          f'capsule touch {sum(r["touch"] for r in rs) / steps:.4f}, '
                          f'overlap(geom) {sum(r["overlap"] for r in rs) / steps:.4f}, '
                          f'fmax {max(r["fmax"] for r in rs):.0f} N', flush=True)
    elif part == 'B':
        for name in stages:
            for phys in ('ours', 'um'):
                for _, caps in VARIANTS:
                    for d in fo.PUSH_SIDES:
                        cells = []
                        for mag in B_MAGS:
                            rs = of(name=name, phys=phys, caps=caps, dir=d, mag=mag)
                            cells.append('dv %.2f falls %d/%d ovl %d/%d fmax %4.0f N >10N %.2f s' % (
                                mag, sum(r['fell'] is not None for r in rs), len(rs), sum(r['overlap'] > 0 for r in rs),
                                len(rs), max(r['fmax'] for r in rs), max(r['ftouch_s'] for r in rs)))
                        print(f'{name:24s} {phys:4s} {tag(caps):3s} {d}: ' + ' | '.join(cells))
    else:
        for name in stages:
            for _, caps in VARIANTS:
                cells = []
                for cmd in C_COMMANDS:
                    rs = of(name=name, caps=caps, cmd=list(cmd))
                    vy = [r['vy'] for r in rs if r['vy'] is not None]
                    cells.append('%+.1f: vy %s F%d sc %.3f t %.3f o %.3f %.0fN' % (
                        cmd[1], '/'.join('%+.2f' % v for v in vy), sum(r['fell'] is not None for r in rs),
                        np.mean([r['sc'] for r in rs]), np.mean([r['touch'] for r in rs]),
                        np.mean([r['overlap'] for r in rs]), max(r['fmax'] for r in rs)))
                print(f'{name:24s} {tag(caps):3s} ' + ' | '.join(cells))


def init():
    import torch
    torch.set_num_threads(1)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Удар от столкновения стоп (foot_collision) на этапах, обученных без него')
    parser.add_argument('part', choices=('A', 'B', 'C'))
    parser.add_argument('stages', nargs='+', help='папка этапа, выгрузка export_g1 или unitree_v0')
    parser.add_argument('--workers', type=int, default=8, help='рабочих процессов (пока идёт обучение, не больше 8)')
    parser.add_argument('--seeds', type=int, default=24, help='эпизодов части A на этап и режим')
    parser.add_argument('--out', type=Path, help='JSON: все прогоны')
    args = parser.parse_args(argv)
    paths = [p if p == 'unitree_v0' else str(Path(p).resolve()) for p in args.stages]
    for p in paths:
        if p != 'unitree_v0' and not ((Path(p) / 'policy.onnx').is_file() or (Path(p) / 'policy.zip').is_file()):
            raise SystemExit(f'Не этап и не выгрузка export_g1: {p}')
    stages = {fo.label(p): p for p in paths}
    work = jobs(args.part, stages, args.seeds)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # one core per worker
        os.environ.setdefault(key, '1')
    print('part %s, %d runs, %d workers' % (args.part, len(work), args.workers), flush=True)
    t0 = time.time()
    with Pool(args.workers, initializer=init, maxtasksperchild=12) as pool:
        results = pool.map(run, work, chunksize=1)
    print('%.0f s' % (time.time() - t0))
    report(args.part, stages, results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
