"""Counterfactual arm tracking, CPU (the critic's lab-critic/it42/cf.py, cf_quick.py and cf_report.py, iteration 42):
arm_counterfactual.py <stage | experiment[:steps]> [--workers 4] [--weights 2,4,6] [--json <file>]

What would exact arm tracking cost the rest of an arm_targets stage's reward, and from which arm_track weight on does it
pay? Each run: the stage's policy, deterministic, no noise, heading hold off, a reset with the command (standing, or
0.5 m/s forward; seed 73000, phase 0), a load of 0 / 1 / 2 kg per hand from the start (set_hand_load), at 1 s the arms
go to a library pose in a 1 s quintic (flag 1, set_arm_targets), 700 control steps (14 s). Nine poses: the carry poses
nearest a shoulder pitch of -1.5 / -1.1 / -0.7 / -0.4 with the elbow near 1.1, four free poses (rng 42) and both arms
extended forward (shoulder pitch -1.57, elbow 1.57). Each runs twice: base (the policy) and oracle, an integrator from
2 s on that adds c to the ten tracked joints' actions, c <- clip(c - 0.02 e / scale, +-9) with e = q - target, so the
arms track almost exactly. Over the window from 5 s on: dT, the oracle's mean unweighted arm_track (info['arm_track'])
minus the policy's, and dO, the same for the sum of every other reward term; w* = -dO/dT is the arm_track weight above
which exact tracking earns more than the policy's own (dT <= 1e-4, nothing to gain: 0, as the critic's report; a run
whose base falls is left out). Prints per run the arm errors (mean |e|; the left shoulder pitch and elbow q - target),
the oracle's fall, dT, dO, w* and the three terms the oracle loses most, then per command and load the mean errors,
w*'s median and largest, the share of poses with w* at or below each of --weights (by default the stage's arm_track
weight, twice and three times it) and the mean change of the main terms. About 2 minutes on 4 processes."""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_falls import resolve  # noqa: E402

# Control steps: 14 s; the pose from 1 s, the oracle from 2 s, the window from 5 s.
STEPS, MOVE_AT, CORRECT_AT, WINDOW_AT = 700, 50, 100, 250
GAIN, LIMIT = .02, 9.  # the oracle's integrator gain (per step, of e / scale) and its bound on c
SEED = 73000
COMMANDS = ((0., 0., 0.), (.5, 0., 0.))
LOADS = (0., 1., 2.)
TERMS = ('stand_pose', 'torso', 'posture', 'hips', 'velocity', 'still')
_cache = {}


def setup(stage):
    """This worker's policy, normalization and the stage's task_config, once per stage."""
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(stage))
        _cache[stage] = (model, norm, stage_task_config(Path(stage)))
    return _cache[stage]


def poses():
    """(name, joint names, angles) of the nine poses: four carry poses over shoulder pitch, four free ones, extended."""
    from humanoid_sim.tasks.g1_velocity import arm_pose_library
    names, free, carry, _ = arm_pose_library()
    out = []
    for pitch in (-1.5, -1.1, -.7, -.4):  # the carry rows nearest this shoulder pitch with the elbow near 1.1
        row = int(np.argmin(np.abs(carry[:, 0] - pitch) + .2 * np.abs(carry[:, 3] - 1.1)))
        out.append((f'carry{pitch:+.1f}', names, carry[row].tolist()))
    for k, row in enumerate(np.random.default_rng(42).choice(len(free), 4, replace=False)):
        out.append((f'free{k}', names, free[row].tolist()))
    extended = {'shoulder_pitch': -1.57, 'shoulder_roll': 0., 'shoulder_yaw': 0., 'elbow': 1.57, 'wrist_roll': 0.}
    out.append(('extended', names, [extended[n.split('_', 1)[1].rsplit('_joint', 1)[0]] for n in names]))
    return out


def episode(job, oracle):
    """One run: over the window the mean reward components, unweighted arm_track and arm |error|, the mean signed
    error per tracked joint, the oracle's mean c and the creep; the step of a fall (None)."""
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = setup(job['stage'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    try:
        obs, _ = env.reset(seed=SEED, options={'command': list(job['command']), 'phase_steps': 0})
        raw.set_hand_load(job['load'])
        names = [raw.model.joint(j).name for j in raw.joints]
        pose = raw.home.copy()
        for name, value in zip(job['names'], job['pose']):
            pose[names.index(name)] = value
        tracked = list(raw.track_joints)
        scale = raw.scale[tracked]
        c = np.zeros(len(tracked))
        sums, track, errors, signed, cs, xy, fell = {}, [], [], [], [], [], None
        for k in range(STEPS):
            if k == MOVE_AT:
                raw.set_arm_targets(pose, 1, seconds=1.)
                obs = raw._observation()
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0].copy()
            if oracle and k >= CORRECT_AT:
                error = raw.data.qpos[raw.qadr][tracked] - (raw.home[tracked] + raw.arm_offset[tracked])
                c = np.clip(c - GAIN * error / scale, -LIMIT, LIMIT)
                action[tracked] += c
            obs, _, terminated, _, info = env.step(action)
            if k >= WINDOW_AT:
                for key, value in info['reward_components'].items():
                    sums[key] = sums.get(key, 0.) + value
                track.append(info['arm_track'])
                errors.append(info['arm_error'])
                signed.append(raw.data.qpos[raw.qadr][tracked] - (raw.home[tracked] + raw.arm_offset[tracked]))
                cs.append(c.copy())
                xy.append(raw.data.qpos[:2].copy())
            if terminated:
                fell = k
                break
    finally:
        env.close()
    n = max(1, len(errors))
    return dict(variant='oracle' if oracle else 'base', fell=fell, comps={key: v / n for key, v in sums.items()},
                track=float(np.mean(track)) if track else None, error=float(np.mean(errors)) if errors else None,
                signed=np.mean(signed, 0).tolist() if signed else None, c=np.mean(cs, 0).tolist() if cs else None,
                creep=float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else None)


def run(job):
    """Base and oracle of one pose, command and load."""
    rows = [episode(job, False), episode(job, True)]
    return dict({key: value for key, value in job.items() if key not in ('stage', 'names')}, rows=rows)


def break_even(base, oracle):
    """(dT, dO, w*) of a pose: the oracle's mean unweighted arm_track and sum of the other terms minus the policy's;
    w* = -dO/dT, NaN when tracking gains nothing (dT <= 1e-4). None when the base fell or either has no window."""
    if base['fell'] is not None or not base['comps'] or not oracle['comps']:
        return None
    other = {variant: sum(v for key, v in row['comps'].items() if key != 'arm_track')
             for variant, row in (('base', base), ('oracle', oracle))}
    gain, cost = oracle['track'] - base['track'], other['oracle'] - other['base']
    return gain, cost, (-cost / gain if gain > 1e-4 else float('nan'))


def report(results, weights):
    """The per-run table and the summary per command and load (printed); the summary rows."""
    print('pose        cmd  kg | base |e| sp_e  el_e | oracle |e| fell | dT     dO      w*    | the oracle loses most')
    summary = []
    for job in results:
        base, oracle = job['rows'][0], job['rows'][1]
        pair = break_even(base, oracle)
        head = f"{job['pose_name']:<11} {job['command'][0]:.1f} {job['load']:.0f}"
        if pair is None:
            print(f"{head} | base fell at {base['fell']}" if base['fell'] is not None
                  else f"{head} | oracle fell at {oracle['fell']}, before the window")
            continue
        gain, cost, wstar = pair
        costs = {key: oracle['comps'].get(key, 0.) - base['comps'].get(key, 0.)
                 for key in base['comps'] if key != 'arm_track'}
        top = sorted(costs.items(), key=lambda kv: kv[1])[:3]
        print(f"{head} | {base['error']:.3f} {base['signed'][0]:+.2f} {base['signed'][3]:+.2f} | "
              f"{oracle['error']:.3f} {str(oracle['fell']):>4} | {gain:+.3f} {cost:+.3f} {wstar:6.2f} | "
              + ' '.join(f'{key} {v:+.2f}' for key, v in top))
        summary.append(dict(pose=job['pose_name'], cmd=job['command'][0], load=job['load'], gain=gain, cost=cost,
                            wstar=wstar, base_error=base['error'], oracle_error=oracle['error'],
                            oracle_fell=oracle['fell'], costs=costs))
    print()
    for cmd in sorted({row['cmd'] for row in summary}):
        for load in sorted({row['load'] for row in summary}):
            rows = [row for row in summary if row['cmd'] == cmd and row['load'] == load]
            if not rows:
                continue
            ws = np.nan_to_num(np.array([row['wstar'] for row in rows]), nan=0.)
            falls = sum(row['oracle_fell'] is not None for row in rows)
            print(f'cmd {cmd} load {load}: n {len(rows)} base |e| {np.mean([r["base_error"] for r in rows]):.3f} '
                  f'oracle |e| {np.mean([r["oracle_error"] for r in rows]):.3f}'
                  + (f' (oracle falls {falls})' if falls else '')
                  + f' | w* median {np.median(ws):.2f} max {ws.max():.2f} share '
                  + ' '.join(f'<={w:g} {np.mean(ws <= w):.2f}' for w in weights)
                  + f' | mean dT {np.mean([r["gain"] for r in rows]):.3f} dO {np.mean([r["cost"] for r in rows]):+.3f} | '
                  + ' '.join(f'{key} {np.mean([r["costs"].get(key, 0.) for r in rows]):+.3f}' for key in TERMS))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--weights', help='comma-separated arm_track weights (default: the stage\'s, x2, x3)')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    stage = resolve(args.stage)
    from humanoid_sim.core import stage_task_config
    config = stage_task_config(stage)
    if not config.get('arm_targets'):
        raise SystemExit(f'{stage}: not an arm_targets stage')
    weight = float(config['reward_weights']['arm_track'])
    weights = ([float(w) for w in args.weights.split(',')] if args.weights else [weight, 2 * weight, 3 * weight])
    jobs = [dict(stage=str(stage), pose_name=name, names=names, pose=pose, command=command, load=load)
            for name, names, pose in poses() for command in COMMANDS for load in LOADS]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    print(f'{stage}: arm_track weight {weight:g}, {len(jobs)} poses x commands x loads, {time.time() - started:.0f} s')
    report(results, weights)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
