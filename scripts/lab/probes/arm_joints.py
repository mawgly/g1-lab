"""The arms' error per joint on library poses, several stages in one table, CPU (the critic's lab-critic/it42b/trend2.py
and the palm distance of lab-critic/it43/c43.py, iteration 43):
arm_joints.py <stage | experiment[:steps]>... [--workers 4] [--loads 0,1] [--set carry|all] [--json <file>]

Each run: an arm_targets stage's policy, deterministic, no noise, heading hold off, standing (a reset with the stand
command, seed 73000, phase 0), a load per hand from the start (set_hand_load), at 1 s the arms go to the pose in a 1 s
quintic (flag 1, set_arm_targets), 400 control steps (8 s), the window from 5 s on. Poses: by default the four carry
poses of arm_counterfactual.poses() (nearest a shoulder pitch of -1.5 / -1.1 / -0.7 / -0.4, the elbow near 1.1),
--set all its nine. Per stage and load, over the poses that did not fall: the arms' mean |error|; the shoulder pitch's
mean q - target with its sign (+ is a sag), left / right; the shoulder roll and yaw, elbow and wrist roll |mean q -
target| per pose, averaged, left / right; the distance between the palms (HAND_LOAD_POSITION on the wrist_yaw links,
where the loads sit) minus that of the target pose (cm; legs and waist at home, the same forward kinematics), + is
wider apart. Two stages, the carry poses and two loads (16 runs) take about 10 s on 4 processes."""
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
from arm_counterfactual import poses, setup  # noqa: E402
from train_falls import resolve  # noqa: E402

# Control steps: 8 s; the pose from 1 s, the window from 5 s.
STEPS, MOVE_AT, WINDOW_AT = 400, 50, 250
SEED = 73000
PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')


def palm_distance(raw, data):
    """Distance (m) between the two palms, HAND_LOAD_POSITION in each wrist_yaw link's frame, of `data`."""
    from humanoid_sim.tasks import g1_tensor_math as task
    palms = [data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.asarray(task.HAND_LOAD_POSITION) for b in raw.hand_bodies]
    return float(np.linalg.norm(palms[0] - palms[1]))


def target_palms(raw, q):
    """The palm distance (m) of the joint angles q (nu,), the base upright: forward kinematics on a data of its own."""
    import mujoco
    data = mujoco.MjData(raw.model)
    data.qpos[:] = raw.model.qpos0
    data.qpos[raw.qadr] = q
    mujoco.mj_kinematics(raw.model, data)
    return palm_distance(raw, data)


def run(job):
    """One stage, pose and load: the mean signed error per tracked joint over the window (named), the mean palm
    distance and the target's, the step of a fall (None)."""
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = setup(job['stage'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    try:
        obs, _ = env.reset(seed=SEED, options={'command': [0., 0., 0.], 'phase_steps': 0})
        raw.set_hand_load(job['load'])
        names = [raw.model.joint(j).name for j in raw.joints]
        pose = raw.home.copy()
        for name, value in zip(job['names'], job['pose']):
            pose[names.index(name)] = value
        tracked = list(raw.track_joints)
        errors, palms, fell = [], [], None
        for k in range(STEPS):
            if k == MOVE_AT:
                raw.set_arm_targets(pose, 1, seconds=1.)
                obs = raw._observation()
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            obs, _, terminated, _, _ = env.step(action)
            if k >= WINDOW_AT:
                errors.append(raw.data.qpos[raw.qadr][tracked] - (raw.home[tracked] + raw.arm_offset[tracked]))
                palms.append(palm_distance(raw, raw.data))
            if terminated:
                fell = k
                break
        target = target_palms(raw, raw.home + raw.arm_offset)
    finally:
        env.close()
    joints = [names[j] for j in tracked]
    return dict(stage=job['stage'], pose_name=job['pose_name'], load=job['load'], fell=fell,
                error=dict(zip(joints, np.mean(errors, 0).tolist())) if errors else None,
                palm=float(np.mean(palms)) if palms else None, palm_target=target)


def label(stage):
    """exp-xxxxxxxx:steps of a stage folder."""
    stage = Path(stage)
    return f'{stage.parent.name[:12]}:{int(stage.name[6:])}'


def report(results, stages, loads):
    """The table, a row per stage and load (printed); the rows."""
    print(f'{"stage":<24} kg poses falls  |e|   pitch L/R (signed)  roll L/R   yaw L/R    elbow L/R  wrist L/R  '
          'palms - target cm')
    summary = []
    for stage in stages:
        for load in loads:
            runs = [r for r in results if r['stage'] == stage and r['load'] == load]
            rows = [r for r in runs if r['fell'] is None]
            falls = len(runs) - len(rows)
            if not rows:
                print(f'{label(stage):<24} {load:2g} {len(runs):5d} {falls:5d}  (every pose fell)')
                continue

            def side(part, sign=False):
                values = [[r['error'][f'{s}_{part}_joint'] for r in rows] for s in ('left', 'right')]
                return [float(np.mean(v)) if sign else float(np.mean(np.abs(v))) for v in values]
            row = dict(stage=stage, load=load, poses=len(runs), falls=falls,
                       error=float(np.mean([np.abs(list(r['error'].values())).mean() for r in rows])),
                       pitch=side('shoulder_pitch', True), **{part: side(part) for part in PARTS[1:]},
                       palms=float(np.mean([r['palm'] - r['palm_target'] for r in rows])) * 100)
            summary.append(row)
            print(f'{label(stage):<24} {load:2g} {len(runs):5d} {falls:5d}  {row["error"]:.3f} '
                  f'{row["pitch"][0]:+.2f}/{row["pitch"][1]:+.2f}         '
                  + '  '.join(f'{row[part][0]:.2f}/{row[part][1]:.2f}' for part in PARTS[1:])
                  + f'  {row["palms"]:+5.1f}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--loads', default='0,1', help='comma-separated kg per hand')
    parser.add_argument('--set', choices=('carry', 'all'), default='carry')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    from humanoid_sim.core import stage_task_config
    stages = [str(resolve(spec)) for spec in args.stages]
    for stage in stages:
        if not stage_task_config(Path(stage)).get('arm_targets'):
            raise SystemExit(f'{stage}: not an arm_targets stage')
    loads = [float(load) for load in args.loads.split(',')]
    chosen = [p for p in poses() if args.set == 'all' or p[0].startswith('carry')]
    jobs = [dict(stage=stage, pose_name=name, names=names, pose=pose, load=load)
            for stage in stages for name, names, pose in chosen for load in loads]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    print(f'{len(stages)} stages, {len(chosen)} poses ({args.set}), loads {args.loads} kg, {time.time() - started:.0f} s')
    report(results, stages, loads)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
