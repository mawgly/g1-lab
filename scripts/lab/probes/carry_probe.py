"""Arms on targets with loads (task_config arm_targets, iteration 40; the critic's lab-critic/it40/step0.py), CPU:
carry_probe.py <stage | experiment[:steps]> [--workers 4] [--json <file>]

A stage of an arm_targets branch, deterministic policy, heading hold off, reset seed 73000 at phase 0, the window from
t = 3 s:
  extended  both arms forward (shoulder pitch -1.57, elbow +1.57, roll, yaw and wrist roll 0), a quintic from home over
            1 s from t = 1 s (set_arm_targets), then held; 0 / 1 / 2 kg per hand (set_hand_load); standing 12 s, and
            standing pushed at t = 6 s (Δv 1.0 m/s for 0.1 s forward, back, left, right; set_force_push).
  random    walking 0.5 and 1 m/s for 12 s, targets drawn as in training (task.arm_segment_draw: the carry or free pose
            libraries, a quintic between them, held 1-8 s), from home at t = 1 s; 0 and 1 kg per hand, 3 draws each.
Prints per run: the fall (s), standing creep (m) and turn (deg), walking speed (m/s) and drift (rad/s), the arms' mean
|error| over the ten free joints and its 95th percentile (rad), the self_contact share of substeps (when weighted)."""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SECONDS, WINDOW, PUSH_AT, MOVE_AT, RESET_SEED = 12., 3., 6., 1., 73000
PUSHES = {'stand': None, 'push_f': (1., 0.), 'push_b': (-1., 0.), 'push_l': (0., 1.), 'push_r': (0., -1.)}
EXTENDED = {'shoulder_pitch': -1.57, 'shoulder_roll': 0., 'shoulder_yaw': 0., 'elbow': 1.57, 'wrist_roll': 0.}
_cache = {}


def resolve(spec):
    """A stage folder, or <experiment>[:steps] of the lab under runs/ (its latest stage)."""
    path = Path(spec)
    if path.is_dir():
        return path
    from humanoid_sim.core import DEFAULT_RUN, list_stages
    from humanoid_sim.lab import Laboratory
    name, _, steps = spec.partition(':')
    run = Laboratory(DEFAULT_RUN).resolve(name)
    return run / f'stage_{int(steps):012d}' if steps else list_stages(run)[-1]


def stage(path):
    if 'stage' not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(path))
        _cache['stage'] = (model, norm, stage_task_config(Path(path)))
    return _cache['stage']


def heading(qpos):
    w, x, y, z = qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def extended_pose(raw):
    names = [raw.model.joint(j).name for j in raw.joints]
    pose = raw.home.copy()
    for side in ('left', 'right'):
        for part, value in EXTENDED.items():
            pose[names.index(f'{side}_{part}_joint')] = value
    return pose


def run(job):
    import torch
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = stage(job['stage'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    obs, _ = env.reset(seed=RESET_SEED, options={'command': list(job['command']), 'phase_steps': 0})
    raw.set_hand_load(job['load'])
    rng = np.random.default_rng(1000 + job['draw'])
    uniform = lambda n: torch.from_numpy(rng.random(n))
    steps, start, move_at = round(SECONDS / raw.dt), round(WINDOW / raw.dt), round(MOVE_AT / raw.dt)
    push = PUSHES.get(job['plan'])
    next_target, offsets = move_at, torch.zeros((1, raw.model.nu), dtype=torch.float64)
    yaw, xy, errors, contact, fell = [], [], [], [], None
    weight = raw.reward_weights['self_contact']
    for k in range(steps):
        if k == move_at and job['target'] == 'extended':
            raw.set_arm_targets(extended_pose(raw), 1, seconds=1.)
            obs = raw._observation()
        if job['target'] == 'random' and k == next_target:
            end, move, hold = task.arm_segment_draw(uniform, offsets, raw.arm_poses, raw.arm_carry, raw.track_joints,
                                                    raw.dt)
            raw.set_arm_targets(raw.home + end[0].numpy(), 1, seconds=int(move[0]) * raw.dt)
            obs = raw._observation()
            offsets, next_target = end, k + int(move[0]) + int(hold[0])
        if push is not None and k == round(PUSH_AT / raw.dt):
            raw.set_force_push(push, .1)
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        obs, _, terminated, _, info = env.step(action)
        if k >= start and info.get('arm_error') is not None:
            errors.append(info['arm_error'])
        if k >= start and weight:
            contact.append(-info['reward_components']['self_contact'] / weight)
        yaw.append(heading(raw.data.qpos))
        xy.append(raw.data.qpos[:2].copy())
        if terminated:
            fell = round((k + 1) * raw.dt, 2)
            break
    env.close()
    out = {key: value for key, value in job.items() if key != 'stage'}
    out['fell'] = fell
    if errors:
        out.update(error=float(np.mean(errors)), error_p95=float(np.percentile(errors, 95)))
    if contact:
        out['self_contact'] = float(np.mean(contact))
    if fell is None:
        yaw_u = np.unwrap(yaw)
        span = (len(yaw) - 1 - start) * raw.dt
        out.update(turn_deg=math.degrees(yaw_u[-1] - yaw_u[start]), rate=float((yaw_u[-1] - yaw_u[start]) / span),
                   speed=float((xy[-1][0] - xy[start][0]) / span), creep=float(np.linalg.norm(xy[-1] - xy[start])))
    return out


def jobs(path):
    out = []
    for load in (0., 1., 2.):
        for plan in PUSHES:
            out.append(dict(stage=str(path), target='extended', plan=plan, command=(0., 0., 0.), load=load, draw=0))
    for speed in (.5, 1.):
        for load in (0., 1.):
            for draw in range(3):
                out.append(dict(stage=str(path), target='random', plan=f'walk{speed:g}', command=(speed, 0., 0.),
                                load=load, draw=draw))
    return out


def number(value, form):
    return 'n/a' if value is None else form % value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    path = resolve(args.stage)
    from humanoid_sim.core import stage_task_config
    if not stage_task_config(path).get('arm_targets'):
        raise SystemExit(f'{path}: not an arm_targets stage (set_arm_targets needs the slot\'s arm command)')
    todo = jobs(path)
    with Pool(args.workers) as pool:
        runs = pool.map(run, todo, chunksize=1)
    print(path)
    print('target   plan     load | fell  | stand: creep turn | walk: v rate | arm error p95 | self_contact')
    for r in runs:
        print(f"{r['target']:<8} {r['plan']:<8} {r['load']:.1f}  | {number(r['fell'], '%5.2f'):>5} | "
              f"{number(r.get('creep'), '%5.2f')} {number(r.get('turn_deg'), '%+6.1f')} | "
              f"{number(r.get('speed'), '%5.2f')} {number(r.get('rate'), '%+.3f')} | "
              f"{number(r.get('error'), '%.3f')} {number(r.get('error_p95'), '%.3f')} | "
              f"{number(r.get('self_contact'), '%.2f')}")
    falls = sum(r['fell'] is not None for r in runs)
    print(f'falls {falls} of {len(runs)}')
    if args.json:
        args.json.write_text(json.dumps(runs, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
