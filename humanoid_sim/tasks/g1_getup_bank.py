"""Start-pose bank of the g1_getup task (iteration 33): ~40 thousand G1 states to get up from, made once on the CPU.

    python -m humanoid_sim.tasks.g1_getup_bank [--total 40000] [--workers 3] [--fall-stage <stage>] [--out <file>]
        [--redo <class> ...]

--redo makes only the named classes again in an existing bank and keeps the others (every class has its own seeds, so
this gives what a whole new run would). A whole run takes ~35 min on 3 workers, mostly the falls.

Classes and their shares of the training part (CLASSES, SHARES; the critic's specification, lab-critic/it33):
  back 30 %, front 30 %, side 10 %  dropped from a pelvis height of 0.5 m lying on the back, the front or either side
        (random heading), joints as HoST's reset (home·U(0.9, 1.1) + U(±0.1)) for half of them, wide random (the range's
        middle ± 0.6 of its half, as the critic's lying_contacts.py) for the other half, the wrist pitch and yaw at 0
        (lock_wrists), then SETTLE_SECONDS with damping only (ctrl −SETTLE_KD·q̇);
  fall 15 %  candidate 8 (FALL_STAGE) standing or walking pushed over, as the critic's fall_poses.py: a force push of
        Δv U(2.5, 3.5) m/s for 0.1 s in a random direction; its policy acts until the pelvis tilt passes 1 rad, then
        Passive (damping SETTLE_KD) for FALL_PASSIVE_SECONDS; runs that do not fall are dropped;
  sit 5 %  dropped upright 2 cm above the floor with folded legs, sitting (hips flexed ~90°) or kneeling (knees 2.4-2.85),
        then SETTLE_SECONDS of damping;
  stand 10 %  the walking task's reset pose (home ± 0.01 rad, pelvis at 0.8 m, at rest), as the walking task starts.
The model is the g1_getup task's (config.G1_GETUP_PROFILE: real robot, foot and hand capsules), MuJoCo C. A state that
blows up (non-finite, a MuJoCo instability reset, a joint faster than 50 rad/s) is drawn again. Each class also gets
HOLDOUT states the training never draws (probes). The file is not part of the model revision (runs/, untracked): the
header records the model revision it was made for, and tests/test_g1_getup_gate.py checks that it matches.
"""
import argparse
import json
import math
import os
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / 'runs' / 'humanoid_lab' / 'getup_bank' / 'bank.npz'
FALL_STAGE = ROOT / 'runs' / 'humanoid_lab' / 'exp-28691193718e4d0e' / 'stage_007221510144'  # candidate 8
GENERATOR = 'g1-getup-bank-v1'
CLASSES = ('back', 'front', 'side', 'fall', 'sit', 'stand')
SHARES = dict(back=.30, front=.30, side=.10, fall=.15, sit=.05, stand=.10)
HOLDOUT = 10  # per class, never drawn in training
SETTLE_SECONDS, SETTLE_KD, DROP_HEIGHT = 1., 3., .5
FALL_PASSIVE_SECONDS, FALL_TILT, FALL_DV, FALL_SPEEDS = 2., 1., (2.5, 3.5), (0., .5, 1.)
MAX_JOINT_SPEED = 50.  # rad/s, g1_getup's termination
SEED_BASE = 33_000_000  # class k draws seeds SEED_BASE + k·10^6 + i


def quat_axis(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# Lying orientations (the critic's POSES): the pelvis x axis (forward) up on the back, down on the front; its y axis
# (left) up or down on a side.
LYING = dict(back=((0, 1, 0), -math.pi / 2), front=((0, 1, 0), math.pi / 2), side=((1, 0, 0), math.pi / 2))


@lru_cache(maxsize=1)
def getup_model_env():
    """A G1VelocityEnv with g1_getup's model (one per process)."""
    from .config import G1_GETUP_PROFILE
    from .g1_velocity import G1VelocityEnv
    return G1VelocityEnv(**G1_GETUP_PROFILE)


def joint_names(env):
    return [env.model.joint(j).name for j in env.joints]


def instabilities(data):
    """How many times MuJoCo has reset this data for a bad position, velocity or acceleration (it then warns and goes on
    from the model's reference pose, which would read as a robot standing)."""
    return sum(int(data.warning[kind].number) for kind in (mujoco.mjtWarning.mjWARN_BADQPOS,
                                                           mujoco.mjtWarning.mjWARN_BADQVEL,
                                                           mujoco.mjtWarning.mjWARN_BADQACC))


def settle(env, seconds, kd=SETTLE_KD):
    """Damping only (ctrl −kd·q̇ within the motor ranges) for `seconds`, MuJoCo C. False if the state blew up."""
    m, d = env.model, env.data
    before = instabilities(d)
    for _ in range(round(seconds / m.opt.timestep)):
        d.ctrl[:] = np.clip(-kd * d.qvel[env.vadr], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
    return healthy(env, before)


def healthy(env, warnings_before):
    d = env.data
    return bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and instabilities(d) == warnings_before
                and np.abs(d.qvel[env.vadr]).max() <= MAX_JOINT_SPEED)


def lowest_gap(env):
    """Smallest distance (m) between the floor and a robot geom that collides with it (negative: penetration)."""
    m, d = env.model, env.data
    floor = m.geom('floor').id
    root = m.body_rootid[env.pelvis]
    return min(mujoco.mj_geomDistance(m, d, floor, g, 2., None) for g in range(m.ngeom)
               if m.body_rootid[m.geom_bodyid[g]] == root and (
                   m.geom_contype[g] & m.geom_conaffinity[floor] or m.geom_contype[floor] & m.geom_conaffinity[g]))


def _start(env, joints, orientation, height):
    m, d = env.model, env.data
    low, high = m.jnt_range[env.joints].T
    joints = np.clip(joints, low, high)
    joints[list(env.wrist_joints)] = 0.  # lock_wrists: the wrist pitch and yaw stay at home
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = [0., 0., height]
    d.qpos[3:7] = orientation
    d.qpos[env.qadr] = joints
    mujoco.mj_forward(m, d)


def lying(env, rng, kind):
    """back / front / side: (variant 0 HoST, 1 wide) and whether the settled state is healthy."""
    low, high = env.model.jnt_range[env.joints].T
    variant = int(rng.integers(2))
    if variant == 0:
        joints = env.home * rng.uniform(.9, 1.1, env.model.nu) + rng.uniform(-.1, .1, env.model.nu)
    else:
        joints = (low + high) / 2 + rng.uniform(-.6, .6, env.model.nu) * (high - low) / 2
    axis, angle = LYING[kind]
    if kind == 'side' and rng.random() < .5:
        angle = -angle
    orientation = quat_mul(quat_axis((0, 0, 1), rng.uniform(-math.pi, math.pi)), quat_axis(axis, angle))
    _start(env, joints, orientation, DROP_HEIGHT)
    return variant, settle(env, SETTLE_SECONDS)


def sitting(env, rng):
    """sit: (variant 0 sitting, 1 kneeling) dropped upright with folded legs."""
    names = joint_names(env)
    variant = int(rng.integers(2))
    joints = env.home * rng.uniform(.9, 1.1, env.model.nu) + rng.uniform(-.1, .1, env.model.nu)
    for side in ('left', 'right'):
        part = lambda name: names.index(f'{side}_{name}_joint')
        if variant == 0:  # on the buttocks, thighs forward, shins down
            joints[part('hip_pitch')] = rng.uniform(-1.9, -1.3)
            joints[part('knee')] = rng.uniform(1., 2.2)
            joints[part('ankle_pitch')] = rng.uniform(-.6, .3)
        else:  # on the knees, shins back along the floor
            joints[part('hip_pitch')] = rng.uniform(-.5, .2)
            joints[part('knee')] = rng.uniform(2.4, 2.85)
            joints[part('ankle_pitch')] = rng.uniform(-.3, .5)
    tilt = quat_mul(quat_axis((1, 0, 0), rng.uniform(-.1, .1)), quat_axis((0, 1, 0), rng.uniform(-.1, .1)))
    orientation = quat_mul(quat_axis((0, 0, 1), rng.uniform(-math.pi, math.pi)), tilt)
    _start(env, joints, orientation, 1.)
    env.data.qpos[2] += .02 - lowest_gap(env)
    mujoco.mj_forward(env.model, env.data)
    return variant, settle(env, SETTLE_SECONDS)


def standing(env, rng):
    """stand: the walking task's reset pose (home ± 0.01 rad, the pelvis at 0.8 m, at rest, random heading), not
    simulated: held by the PD at home (action 0) the torso sags back onto the waist pitch stop within 0.5 s (kp 28.5
    N·m/rad, barely above the 21.7 N·m/rad of the upper body's weight, 16.6 kg 0.13 m above the axis); a policy keeps
    it up."""
    orientation = quat_axis((0, 0, 1), rng.uniform(-math.pi, math.pi))
    _start(env, env.home + rng.uniform(-.01, .01, env.model.nu), orientation, .8)
    return 0, True


@lru_cache(maxsize=1)
def fall_policy(stage):
    """Candidate 8's policy, normalizer and walking environment with g1_getup's model (its hands collide)."""
    import torch
    from ..core import load_stage, stage_task_config
    from .catalog import make_task
    torch.set_num_threads(1)
    policy, norm = load_stage(Path(stage))
    config = {**stage_task_config(Path(stage)), 'heading_hold': False, 'max_episode_steps': 100000,
              'hand_collision': True}
    return policy, norm, make_task('g1_velocity', 'g1_29dof', config)


def falling(stage, rng, seed):
    """fall: (variant: the walking speed's index, whether it fell and settled) with the walking environment's state."""
    policy, norm, env = fall_policy(str(stage))
    raw = env.unwrapped
    variant = int(rng.integers(len(FALL_SPEEDS)))
    dv, angle, push_at = rng.uniform(*FALL_DV), rng.uniform(0, 2 * math.pi), int(rng.integers(25, 101))
    obs, _ = env.reset(seed=seed, options={'command': [FALL_SPEEDS[variant], 0., 0.], 'phase_steps': 0})
    m, d = raw.model, raw.data
    before = instabilities(d)
    try:
        for t in range(push_at + 400):
            if t == push_at:
                raw.set_force_push((dv * math.cos(angle), dv * math.sin(angle)), .1)
            obs = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])[0]
            if math.acos(max(-1., min(1., d.xmat[raw.pelvis].reshape(3, 3)[2, 2]))) > FALL_TILT:
                break
        else:
            return raw, variant, False
    except FloatingPointError:
        return raw, variant, False
    if not healthy(raw, before):
        return raw, variant, False
    return raw, variant, settle(raw, FALL_PASSIVE_SECONDS)


def make_state(job):
    """One bank entry: job (class, seed, fall stage) -> (qpos, qvel, variant) or None (drawn again)."""
    kind, seed, stage = job
    rng = np.random.default_rng(seed)
    if kind == 'fall':
        env, variant, ok = falling(stage, rng, seed)
    else:
        env = getup_model_env()
        if kind in LYING:
            variant, ok = lying(env, rng, kind)
        elif kind == 'sit':
            variant, ok = sitting(env, rng)
        else:
            variant, ok = standing(env, rng)
    if not ok:
        return None
    return env.data.qpos.copy(), env.data.qvel.copy(), variant


def _init():
    import torch
    torch.set_num_threads(1)


def counts(total):
    """Training states per class for `total` (rounded shares; HOLDOUT more each)."""
    return {kind: int(round(SHARES[kind] * total)) for kind in CLASSES}


def generate(total=40000, workers=3, fall_stage=FALL_STAGE, out=BANK, progress=print, redo=None):
    """Makes the bank (counts(total) training states per class plus HOLDOUT each) and writes it to `out` (npz, atomic)
    with a summary next to it (<out>.json). redo: only these classes, the others kept from the bank at `out` (its
    total). Returns the summary."""
    from ..core import write_json
    from .identity import task_identity
    from .config import G1_GETUP_PROFILE
    started = time.time()
    entries = {kind: [] for kind in CLASSES}
    tried = {kind: 0 for kind in CLASSES}
    previous = {}
    if redo:
        if set(redo) - set(CLASSES):
            raise ValueError(f'Unknown start classes: {sorted(set(redo) - set(CLASSES))}')
        old = load(out)
        if old['classes'] != CLASSES:
            raise ValueError('The bank has other classes: make it anew')
        previous, total = old['header'], old['header']['total']
        for number, kind in enumerate(CLASSES):
            if kind not in redo:
                rows = np.flatnonzero(old['start_class'] == number)
                entries[kind] = [(int(old['seed'][row]), old['qpos'][row].copy(), old['qvel'][row].copy(),
                                  int(old['variant'][row])) for row in rows]
                tried[kind] = previous['tried'][kind]
    wanted = counts(total)
    pool = Pool(workers, initializer=_init) if workers > 1 else None
    try:
        for number, kind in enumerate(CLASSES):
            if redo and kind not in redo:
                continue
            needed = wanted[kind] + HOLDOUT
            while len(entries[kind]) < needed:
                left = needed - len(entries[kind])
                size = left + left // 8 + 16 if kind == 'fall' else left  # about 9 of 10 pushes fell (critic: 20 of 32)
                seeds = [SEED_BASE + number * 10 ** 6 + tried[kind] + i for i in range(size)]
                jobs = [(kind, seed, str(fall_stage)) for seed in seeds]
                results = pool.map(make_state, jobs, chunksize=8) if pool else list(map(make_state, jobs))
                tried[kind] += size
                for seed, result in zip(seeds, results):
                    if result is not None and len(entries[kind]) < needed:
                        entries[kind].append((seed, *result))
                progress(f'{kind}: {len(entries[kind])}/{needed} (tried {tried[kind]}), '
                         f'{time.time() - started:.0f} s')
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    qpos, qvel, start_class, holdout, seeds, variants = [], [], [], [], [], []
    for number, kind in enumerate(CLASSES):
        for k, (seed, position, velocity, variant) in enumerate(entries[kind]):
            qpos.append(position)
            qvel.append(velocity)
            start_class.append(number)
            holdout.append(k >= wanted[kind])  # the last HOLDOUT of each class
            seeds.append(seed)
            variants.append(variant)
    identity = task_identity('g1_29dof', 'g1_velocity', {key: value for key, value in G1_GETUP_PROFILE.items()
                                                          if key != 'gait'})
    if previous and previous['model_revision'] != identity['model_revision']:
        raise ValueError('The bank was made for another model: make it anew, not by classes')
    header = dict(generator=GENERATOR, model_revision=identity['model_revision'], total=total,
                  counts=wanted, holdout=HOLDOUT, tried=tried, shares=SHARES,
                  settle=dict(seconds=SETTLE_SECONDS, kd=SETTLE_KD, drop_height=DROP_HEIGHT),
                  fall=dict(stage=str(Path(fall_stage).resolve()), passive_seconds=FALL_PASSIVE_SECONDS,
                            tilt=FALL_TILT, dv=FALL_DV, speeds=FALL_SPEEDS),
                  seconds=round(time.time() - started) + previous.get('seconds', 0), created_at=time.time(),
                  redone=sorted(set(previous.get('redone', [])) | set(redo or ())))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp.npz')
    np.savez_compressed(temporary, qpos=np.array(qpos, dtype=np.float32), qvel=np.array(qvel, dtype=np.float32),
                        start_class=np.array(start_class, dtype=np.uint8), holdout=np.array(holdout),
                        seed=np.array(seeds, dtype=np.int64), variant=np.array(variants, dtype=np.uint8),
                        classes=np.array(CLASSES), header=np.array(json.dumps(header)))
    os.replace(temporary, out)
    summary = {**header, 'stats': statistics(load(out))}
    write_json(out.with_suffix('.json'), summary)
    return summary


def load(path=BANK):
    """The bank as a dict of arrays (qpos, qvel float64; start_class, holdout, seed, variant) plus 'classes' (names) and
    'header' (dict), read once per file version in a process (read-only: environments share it). Raises
    FileNotFoundError with the command that makes it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup start bank {path} is missing: python -m humanoid_sim.tasks.g1_getup_bank')
    status = path.stat()
    return _load(str(path.resolve()), status.st_mtime_ns, status.st_size)


@lru_cache(maxsize=2)
def _load(path, mtime, size):
    with np.load(path, allow_pickle=False) as data:
        bank = {key: data[key] for key in data.files}
    bank['qpos'], bank['qvel'] = bank['qpos'].astype(np.float64), bank['qvel'].astype(np.float64)
    bank['classes'] = tuple(str(name) for name in bank['classes'])
    bank['header'] = json.loads(str(bank['header']))
    for value in bank.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return bank


def statistics(bank):
    """Per class: training and holdout counts, pelvis height, pelvis tilt (deg) and head height (m) ranges."""
    from . import g1_tensor_math as task
    env = getup_model_env()
    result = {}
    for number, kind in enumerate(bank['classes']):
        rows = np.flatnonzero(bank['start_class'] == number)
        heads, tilts = [], []
        for row in rows:
            mujoco.mj_resetData(env.model, env.data)
            env.data.qpos[:] = bank['qpos'][row]
            mujoco.mj_kinematics(env.model, env.data)
            heads.append(float(task.head_height(env.data.xpos[env.torso][None],
                                                env.data.xmat[env.torso].reshape(1, 3, 3))[0]))
            tilts.append(math.degrees(math.acos(max(-1., min(1., env.data.xmat[env.pelvis][8])))))
        span = lambda values: [round(float(np.min(values)), 3), round(float(np.median(values)), 3),
                               round(float(np.max(values)), 3)] if len(values) else None
        result[kind] = dict(training=int((~bank['holdout'][rows]).sum()), holdout=int(bank['holdout'][rows].sum()),
                            pelvis_z=span(bank['qpos'][rows, 2]), tilt_deg=span(tilts), head_z=span(heads),
                            joint_speed_max=round(float(np.abs(bank['qvel'][rows][:, 6:]).max()), 3) if len(rows)
                            else None)
    return result


def main():
    parser = argparse.ArgumentParser(description='Банк стартовых поз задачи g1_getup (вставание после падения)')
    parser.add_argument('--total', type=int, default=40000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--fall-stage', type=Path, default=FALL_STAGE)
    parser.add_argument('--out', type=Path, default=BANK)
    parser.add_argument('--redo', nargs='+', choices=CLASSES, help='только эти классы, остальные — из банка --out')
    args = parser.parse_args()
    summary = generate(args.total, args.workers, args.fall_stage, args.out, progress=lambda line: print(line, flush=True),
                       redo=args.redo)
    print(json.dumps(summary['stats'], indent=1))


if __name__ == '__main__':
    main()
