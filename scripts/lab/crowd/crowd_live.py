"""States of a "live field" of G1s for a clip (29.10, Pavel: they should be moving all the time, each doing something of
its own, not all at once): every robot is its own run of our policies with its own random programme over the whole
clip, so at any moment some walk, some turn, some get pushed, fall and get up.

- walkers (candidate 8's Velocity) and arm robots (47c @ 9886, arm targets: the named poses, switched every 2-4 s) walk
  segments of 1-4 s drawn from MENU (stand, turn in place, forward, back, sideways, arcs) around their own spot: once
  the pelvis is more than HOME m from where it started, it walks back (they do not see each other, so each stays near
  its cell); every 3-9 s a push on the pelvis (the sim2sim push patch's force), now and then a hard one;
- a robot that falls gets up by the getting-up chain (scripts/lab/probes/getup_chain.py: 1164, the exact g1_ctrl
  transition, the tracker of iteration 90, retry, legs home) from the state it fell in, with the probe's sensor noise
  and encoder bias, then walks on with its own policy;
- some start lying (bank rows: back, front, side, falls) and get up first.

    python scripts/lab/crowd/crowd_live.py [--robots 4096] [--workers 12] [--seconds 30]
        [--out runs/humanoid_lab/crowd/live.npz]

CPU only, so the GPU's training goes on. The .npz holds qpos (robots, frames, 36) float32 at 25 Hz (a robot that ends
lying keeps its last pose), kind, falls (how many times each fell) and lying_end."""
import argparse
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # see crowd_record.py: SB3 would load every policy onto the training's GPU
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowd_record as rec  # noqa: E402  (the pose hook and the chain's settings)

ARMS = ROOT / 'runs/humanoid_lab/exp-c21fcdbcaeeb453e/stage_009886531584'  # stage 3 frozen on 47c (arm targets)
KINDS = {'walk': .6, 'arms': .2, 'lying': .2}
LYING = {'back': .4, 'front': .4, 'side': .1, 'fall': .1}
HOME = .6  # m from its spot before a robot walks back
# (name, weight, command sampler): commands in the robot frame (vx, vy, yaw rate)
MENU = (
    ('stand', .08, lambda r: [0., 0., 0.]),
    ('turn', .2, lambda r: [0., 0., r.choice([-1, 1]) * r.uniform(.5, 1.)]),
    ('forward', .25, lambda r: [r.uniform(.3, .8), 0., 0.]),
    ('back', .15, lambda r: [-r.uniform(.3, .5), 0., 0.]),
    ('side', .15, lambda r: [0., r.choice([-1, 1]) * r.uniform(.2, .4), 0.]),
    ('arc', .17, lambda r: [r.uniform(.3, .6), 0., r.choice([-1, 1]) * r.uniform(.3, .6)]),
)
PUSH_EVERY = (3., 8.)  # s between pushes
HARD = .35  # share of hard pushes (Δv 2.5-4 m/s: candidate 8 held 2.2 in 24 of 24); the others 0.4-1.2 m/s
HARD_SPEED, SOFT_SPEED = (2.5, 4.), (.4, 1.2)


def segment(rng, raw, home):
    """(command, seconds) of the next segment; back towards home when too far from it."""
    offset = home - raw.data.qpos[:2]
    if np.linalg.norm(offset) > HOME:
        rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
        local = rotation[:2, :2].T @ offset / np.linalg.norm(offset)
        return [float(np.clip(.6 * local[0], -.4, .6)), float(np.clip(.4 * local[1], -.3, .3)), 0.], 1.
    weights = np.array([w for _, w, _ in MENU])
    name, _, draw = MENU[rng.choice(len(MENU), p=weights / weights.sum())]
    return draw(rng), rng.uniform(1., 2.) if name == 'stand' else rng.uniform(1.5, 4.)


def walking(job, rng, state, home):
    """The walking (or arm) policy from state ((qpos, qvel) or None: standing at the origin) until the clip's end or a
    fall: (qpos, qvel) at the fall, or None."""
    import mujoco
    import getup_probe
    from humanoid_sim.export_g1 import named_arm_poses
    from humanoid_sim.tasks.catalog import make_task
    arms = job['kind'] == 'arms'
    policy, norm, config = getup_probe._policy(str(ARMS if arms else rec.WALK))
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': True, 'hand_collision': True,
                                                 'max_episode_steps': 1_000_000})
    raw = env.unwrapped
    try:
        kept, count = len(rec._poses), rec._count[0]
        obs, _ = env.reset(seed=int(rng.integers(2 ** 31)), options={'command': [0., 0., 0.], 'phase_steps': 0})
        del rec._poses[kept:]
        rec._count[0] = count
        if state is not None:
            raw.data.qpos[:], raw.data.qvel[:] = state
            mujoco.mj_forward(raw.model, raw.data)
            raw.steps, raw.phase_steps = 0, 0
            raw.previous_action[:] = 0.
        poses = named_arm_poses(raw)[1:] if arms else None  # not home: the arms are busy
        dt, frames = raw.dt, job['frames']
        segment_end = arm_next = 0
        push_at = int(rng.uniform(*PUSH_EVERY) / dt)
        step = 0
        obs = raw._observation('fill')
        while len(rec._poses) < frames:
            changed = False
            if step >= segment_end:
                command, seconds = segment(rng, raw, home)
                raw.set_command(np.array(command))
                rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
                raw.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                segment_end = step + int(seconds / dt)
                push_at = max(push_at, step + 1)  # set_command ends a running push
                changed = True
            if arms and step >= arm_next:
                _, _, q = poses[rng.integers(len(poses))]
                raw.set_arm_targets(q, True, seconds=1.)
                arm_next = step + int(rng.uniform(2., 4.) / dt)
                changed = True
            if step == push_at:
                speed = rng.uniform(*HARD_SPEED) if rng.random() < HARD else rng.uniform(*SOFT_SPEED)
                angle = rng.uniform(-np.pi, np.pi)
                raw.set_force_push((speed * np.cos(angle), speed * np.sin(angle)), .1)
                push_at = step + int(rng.uniform(*PUSH_EVERY) / dt)
            if changed:
                obs = raw._observation('replace')
            obs, _, terminated, _, _ = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            step += 1
            if terminated:
                return raw.data.qpos.copy(), raw.data.qvel.copy()
        return None
    finally:
        env.close()


def chain(job, rng, row, state=None):
    """The getting-up chain from a bank row (or from state, with that row for its bookkeeping): whether it ended up."""
    import getup_chain
    base = dict(first=str(rec.FIRST), second=str(rec.SECOND), handover='pd', pd_seconds=3.5,
                pd=dict(arms=.3, kp=.3, kd=3., kp_to=1., kd_to=1.), physics='ours', start='fall', row=row,
                seed=int(rng.integers(2 ** 31)), first_seconds=3., rule='pelvis', second_seconds=None, left='reset',
                retry=dict(retry_angle=float(np.radians(90)), abort_angle=float(np.radians(150)), max_retries=2),
                legs_home=dict(hip=2., seconds=1.), state=state)
    return bool(getup_chain.run(base).get('standing'))


def live(job):
    import torch
    torch.set_num_threads(1)
    rec._hook()
    rec._poses.clear()
    rec._count[0] = 0
    rng = np.random.default_rng((29_10, job['index']))
    falls, state, home = 0, None, np.zeros(2)
    if job['kind'] == 'lying':
        up = chain(job, rng, job['row'])
        home = rec._poses[0][:2].astype(float)
        if not up:
            return finish(job, falls, True)
        state = (rec._last[0].qpos.copy(), rec._last[0].qvel.copy())
        job = dict(job, kind='walk')
    while len(rec._poses) < job['frames']:
        fell = walking(job, rng, state, home)
        if fell is None:
            break
        falls += 1
        if not chain(job, rng, job['fall_row'], state=fell):
            return finish(job, falls, True)
        state = (rec._last[0].qpos.copy(), rec._last[0].qvel.copy())
    return finish(job, falls, False)


def finish(job, falls, lying):
    poses = np.array(rec._poses[:job['frames']], dtype=np.float32)
    return poses, falls, lying


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--robots', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--out', default=str(ROOT / 'runs/humanoid_lab/crowd/live.npz'))
    args = parser.parse_args()
    bank = np.load(ROOT / 'runs/humanoid_lab/getup_bank/bank.npz', allow_pickle=False)
    classes = list(bank['classes'])
    training = ~bank['holdout'].astype(bool)
    rows = {c: np.flatnonzero(training & (bank['start_class'] == classes.index(c))) for c in LYING}
    rng = np.random.default_rng(29_10)
    kinds = rng.choice(list(KINDS), size=args.robots, p=list(KINDS.values()))
    jobs = []
    for k, kind in enumerate(kinds):
        lying = rng.choice(list(LYING), p=list(LYING.values()))
        jobs.append(dict(index=k, kind=str(kind), row=int(rng.choice(rows[lying])), fall_row=int(rng.choice(rows['fall'])),
                         frames=int(args.seconds * 25)))
    began = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(live, jobs, chunksize=1)
    frames = int(args.seconds * 25)
    qpos = np.zeros((len(results), frames, results[0][0].shape[1]), dtype=np.float32)
    for k, (poses, _, _) in enumerate(results):
        qpos[k, :len(poses)] = poses
        qpos[k, len(poses):] = poses[-1]
    falls = np.array([f for _, f, _ in results])
    lying = np.array([end for _, _, end in results])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, kind=kinds, falls=falls, lying_end=lying)
    print(f'{out}: {len(results)} robots, {frames} frames, falls {int(falls.sum())} '
          f'(robots that fell {int((falls > 0).sum())}), lying at the end {int(lying.sum())}, '
          f'{time.time() - began:.0f} s on {args.workers} processes')


if __name__ == '__main__':
    main()
