"""States of a crowd of G1s getting up for a clip (29.10, Pavel: film ~4000 of our robots at once): each robot is its
own run of the getting-up chain (scripts/lab/probes/getup_chain.py: 1164 rolls, the exact g1_ctrl transition, the
tracker), from its own row of the start bank (back, front, side, falls), its own seed and the probe's sensor noise and
encoder bias; nothing is copied between robots. The pose (qpos) is kept every SAMPLE physics substeps (25 Hz).
With --march-at T (Pavel 29.10: after getting up they should all move) a robot that is up when the chain ends goes
on in candidate 8's Velocity from the chain's end state, as g1_ctrl's R2+A (last action 0, history filled), standing
until T s and then walking forward at --walk-speed with heading hold on the heading it has at T.

    python scripts/lab/crowd/crowd_record.py [--robots 4096] [--workers 16] [--out runs/humanoid_lab/crowd/getup.npz]
        [--second <tracker stage>] [--seconds 12] [--march-at 15 --walk-speed 1.0]

CPU only, so the GPU's training goes on. The .npz holds qpos (robots, frames, 36) float32 padded with each run's last
pose, frames (robots,) the recorded length, start class, row and whether it was up at the end."""
import argparse
import os
import sys
import time

# CPU only: SB3 loads a policy onto the GPU by default, and 16 such processes took the training's memory (29.10 the
# trainer died of CUDA out of memory). Hidden before torch is imported, inherited by the pool's processes.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))

SAMPLE = 20  # physics substeps of 2 ms between kept poses: 25 Hz
FIRST = ROOT / 'runs/humanoid_lab/exp-207eb498dc824027/stage_001164902400'
SECOND = ROOT / 'runs/humanoid_lab/exp-c264b0b92b604f61/stage_001051656192'  # iteration 90, frame 21
WALK = ROOT / 'runs/humanoid_lab/exp-28691193718e4d0e/stage_007221510144'  # walking candidate 8
CLASSES = {'back': .4, 'front': .4, 'side': .1, 'fall': .1}
_poses = []
_hooked = []
_count = [0]
_last = [None]  # the data of the latest physics substep: the chain's end state for the march


def _hook():
    """Wraps mujoco.mj_step once per process: every SAMPLE-th substep of any data keeps its qpos. Once by a module
    flag, not a function attribute: getup_probe's monitor wraps mj_step on top of this one, and a second wrapper per
    run nested the hooks and kept every pose several times (only a run's first seconds fitted then)."""
    import mujoco
    if _hooked:
        return
    _hooked.append(True)
    original = mujoco.mj_step

    def step(model, data, *args, **kwargs):
        original(model, data, *args, **kwargs)
        _count[0] += 1
        _last[0] = data
        if _count[0] % SAMPLE == 0:
            _poses.append(data.qpos.astype(np.float32).copy())
    step.crowd = True
    mujoco.mj_step = step


def one(job):
    import torch
    torch.set_num_threads(1)
    import getup_chain
    _hook()
    _poses.clear()
    _count[0] = 0
    record = getup_chain.run(job)
    up = bool(record.get('standing'))
    march = None
    if job.get('march_at') is not None and up and len(_poses) < round(job['march_at'] * 25):
        march = walk(job, _last[0].qpos.copy(), _last[0].qvel.copy())
    poses = np.array(_poses[:job['frames']], dtype=np.float32)
    return poses, up, march


def walk(job, qpos, qvel):
    """Candidate 8's Velocity from the chain's end state: standing until march_at, then walk_speed forward, heading
    hold on the heading at march_at. (heading at march_at, pelvis xy there, fell at s or None)."""
    import mujoco
    import getup_probe
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = getup_probe._policy(str(WALK))
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': True, 'hand_collision': True,
                                                 'max_episode_steps': 100000})
    raw = env.unwrapped
    try:
        kept = len(_poses)
        env.reset(seed=job['seed'], options={'command': [0., 0., 0.], 'phase_steps': 0})
        del _poses[kept:]
        _count[0] = 0
        raw.data.qpos[:], raw.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(raw.model, raw.data)
        raw.steps, raw.phase_steps = 0, 0
        raw.previous_action[:] = 0.
        raw.set_command(np.zeros(3))
        obs = raw._observation('fill')
        at, frames = round(job['march_at'] * 25), job['frames']
        heading = start = fell = None
        while len(_poses) < frames:
            if heading is None and len(_poses) >= at:
                rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
                heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                start = raw.data.qpos[:2].copy()
                raw.set_command(np.array([job['walk_speed'], 0., 0.]))
                raw.heading_target = heading
                obs = raw._observation('replace')
            obs, _, terminated, _, _ = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if terminated:
                fell = round(len(_poses) / 25, 2)
                break
        if heading is None:  # fell standing
            heading, start = 0., raw.data.qpos[:2].copy()
        return heading, start, fell
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robots', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--second', default=str(SECOND))
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--march-at', type=float, help='s: from then every robot up walks forward')
    parser.add_argument('--walk-speed', type=float, default=1.)
    parser.add_argument('--out', default=str(ROOT / 'runs/humanoid_lab/crowd/getup.npz'))
    args = parser.parse_args()
    from humanoid_sim.tasks import g1_getup_bank
    bank = np.load(g1_getup_bank.BANK if hasattr(g1_getup_bank, 'BANK')
                   else ROOT / 'runs/humanoid_lab/getup_bank/bank.npz', allow_pickle=False)
    classes = list(bank['classes'])
    rng = np.random.default_rng(29_10)
    kinds = rng.choice(list(CLASSES), size=args.robots, p=list(CLASSES.values()))
    jobs, starts, rows = [], [], []
    for k, kind in enumerate(kinds):
        pool = np.flatnonzero(~bank['holdout'].astype(bool) & (bank['start_class'] == classes.index(kind)))
        row = int(rng.choice(pool))
        jobs.append(dict(first=str(FIRST), second=args.second, handover='pd', pd_seconds=3.5,
                         pd=dict(arms=.3, kp=.3, kd=3., kp_to=1., kd_to=1.), physics='ours', start=str(kind), row=row,
                         seed=290_000 + k, first_seconds=3., rule='pelvis', second_seconds=None, left='reset',
                         retry=dict(retry_angle=float(np.radians(90)), abort_angle=float(np.radians(150)),
                                    max_retries=2),
                         legs_home=dict(hip=2., seconds=1.), frames=int(args.seconds * 25),
                         march_at=args.march_at, walk_speed=args.walk_speed))
        starts.append(str(kind))
        rows.append(row)
    began = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(one, jobs, chunksize=1)
    lengths = np.array([len(poses) for poses, _, _ in results])
    frames = int(args.seconds * 25)
    qpos = np.zeros((len(results), frames, results[0][0].shape[1]), dtype=np.float32)
    for k, (poses, _, _) in enumerate(results):
        qpos[k, :len(poses)] = poses
        qpos[k, len(poses):] = poses[-1]
    marched = np.array([m is not None for _, _, m in results])
    heading = np.array([m[0] if m else 0. for _, _, m in results])
    march_xy = np.array([m[1] if m else [0., 0.] for _, _, m in results])
    walk_fell = np.array([m[2] if m and m[2] is not None else -1. for _, _, m in results])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, frames=lengths, start=np.array(starts), row=np.array(rows),
                        up=np.array([up for _, up, _ in results]), marched=marched, heading=heading,
                        march_xy=march_xy, walk_fell=walk_fell, march_at=args.march_at or -1.)
    print(f'{out}: {len(results)} robots, {frames} frames, up {sum(up for _, up, _ in results)} of {len(results)}, '
          f'marched {int(marched.sum())}, fell walking {int((walk_fell >= 0).sum())}, '
          f'{time.time() - began:.0f} s on {args.workers} processes')


if __name__ == '__main__':
    main()
