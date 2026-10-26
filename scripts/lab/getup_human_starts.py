"""CPU only. getup_human_starts.py [--data <folder>] [--holdout 5] [--workers 8] [--out <file>]
The human starts of g1_getup's task_config human_starts (iteration 56, reference state init; the critic's
lab-critic/it53/rsi53.py): runs/humanoid_lab/getup_bank/human_starts.npz (tasks/g1_getup_human describes the arrays).

Frames: LAFAN1's fallAndGetUp clips retargeted to the G1 by Unitree (as getup_style_library.py: 30 fps, the root
quaternion x y z w turned w x y z, resampled to the 50 Hz control step), subjects other than --holdout, the head top
0.3-1.1 m up (STYLE_HEAD_BAND), not a clip's first or last frame. Velocities by central differences at 50 Hz: the root
position and the joints linearly, the root's angular velocity in the body frame by mju_subQuat. Each pose is lifted by
the depth of its deepest floor contact in the getting-up model (Unitree's foot and hand capsules included).
Kept when the PD holds it: from the frame, the action (q − home)/scale for 0.5 s (no assist, delay or bound, motor
friction ×1, body friction 0.6) with no flight of 40 ms or more (no floor contact at two control steps running), every
state finite, no joint faster than 25 rad/s, and the style distance (g1_tensor_math.style_distance to the style library
lafan1) at 0.2 s at most 2.0. LAFAN1 is CC BY-NC-ND 4.0: the file stays under runs/ (untracked), never published.
Prints the kept rows and the file's md5, which the setting names.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from getup_style_library import DATA, SOURCE_RATE, resample  # noqa: E402

HOLD_SECONDS, STYLE_SECONDS = .5, .2
FLIGHT_SECONDS, MAX_JOINT_SPEED, MAX_STYLE = .04, 25., 2.
FRICTION = .6
_ENV = _STYLE = None


def _worker():
    global _ENV, _STYLE
    if _ENV is None:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.tasks import g1_getup_style
        from humanoid_sim.tasks.g1_getup import G1GetupEnv
        _ENV = G1GetupEnv(max_episode_steps=100_000)
        _STYLE = g1_getup_style.load('lafan1')
    return _ENV, _STYLE


def floor_depth(env):
    """The deepest robot-floor contact distance (m, ≤ 0) of the current state, 0 without a penetrating one."""
    d = env.data
    return min([float(c.dist) for c in d.contact[:d.ncon] if env.floor in (c.geom1, c.geom2)] + [0.])


def hold(job):
    """(lifted qpos, lift (m), kept, reason) of one candidate frame (qpos, qvel)."""
    import mujoco
    from humanoid_sim.tasks import g1_tensor_math as task
    env, style = _worker()
    qpos, qvel = np.array(job[0]), np.array(job[1])
    env.reset(seed=0, options=dict(start=0, assist_force=0., action_delay_steps=0, motor_friction_scale=1.,
                                   body_friction=FRICTION, pd_bound=None))
    m, d = env.model, env.data
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)
    lift = -floor_depth(env)
    d.qpos[2] += lift
    mujoco.mj_forward(m, d)
    lifted = d.qpos.copy()
    env._observation('fill')
    env.head = float(env._head()[0])
    action = (qpos[7:] - env.home) / env.scale
    airborne = 0
    for k in range(round(HOLD_SECONDS / env.dt)):
        _, _, terminated, _, _ = env.step(np.clip(action, -env.action_limit, env.action_limit))
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or terminated:
            return lifted, lift, False, 'state'
        if np.abs(d.qvel[6:]).max() > MAX_JOINT_SPEED:
            return lifted, lift, False, 'speed'
        touching = any(env.floor in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
        airborne = 0 if touching else airborne + 1
        if airborne * env.dt >= FLIGHT_SECONDS - 1e-9:
            return lifted, lift, False, 'flight'
        if k + 1 == round(STYLE_SECONDS / env.dt):
            distance, _ = task.style_distance(env.style_features(), style['features'], style['norms'])
            if float(distance[0]) > MAX_STYLE:
                return lifted, lift, False, 'style'
    return lifted, lift, True, None


def velocities(qpos, dt):
    """(k, 35) qvel of 50 Hz frames by central differences (the ends by one-sided ones): root linear in the world,
    angular in the body frame (mju_subQuat), joints."""
    import mujoco
    k = len(qpos)
    before, after = np.maximum(np.arange(k) - 1, 0), np.minimum(np.arange(k) + 1, k - 1)
    span = (after - before)[:, None] * dt
    qvel = np.zeros((k, 35))
    qvel[:, :3] = (qpos[after, :3] - qpos[before, :3]) / span
    qvel[:, 6:] = (qpos[after, 7:] - qpos[before, 7:]) / span
    for i in range(k):
        w = np.zeros(3)
        mujoco.mju_subQuat(w, qpos[after[i], 3:7], qpos[before[i], 3:7])
        qvel[i, 3:6] = w / span[i, 0]
    return qvel


def main():
    import mujoco
    from humanoid_sim.tasks import g1_getup_bank, g1_getup_human
    from humanoid_sim.tasks import g1_tensor_math as task
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--data', type=Path, default=DATA, help='папка с fallAndGetUp*_subject*.csv')
    parser.add_argument('--holdout', type=int, default=5, help='отложенный исполнитель (не берётся)')
    parser.add_argument('--workers', type=int, default=8, help='процессов')
    parser.add_argument('--out', type=Path, default=g1_getup_human.HUMAN_STARTS, help='файл стартов')
    args = parser.parse_args()
    files = sorted(args.data.glob('fallAndGetUp*_subject*.csv'))
    if not files:
        raise SystemExit(f'нет клипов fallAndGetUp в {args.data}')
    env = g1_getup_bank.getup_model_env()
    m, d = env.model, env.data
    dt = env.dt
    clips, jobs, meta = [], [], []
    for file in files:
        subject = int(re.search(r'subject(\d+)', file.stem).group(1))
        if subject == args.holdout:
            continue
        qpos, _, _ = resample(np.loadtxt(file, delimiter=','), dt)
        qvel = velocities(qpos, dt)
        heads = []
        for row in qpos:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            heads.append(float(task.head_height(d.xpos[env.torso][None], d.xmat[env.torso].reshape(1, 3, 3))[0]))
        heads = np.array(heads)
        frames = np.flatnonzero(task.style_band(heads) & (np.arange(len(qpos)) > 0)
                                & (np.arange(len(qpos)) < len(qpos) - 1))
        clips.append(dict(file=file.name, subject=subject, frames=int(len(qpos)), candidates=int(len(frames))))
        for i in frames:
            jobs.append((qpos[i].tolist(), qvel[i].tolist()))
            meta.append((subject, len(clips) - 1, int(i), heads[i]))
    with Pool(args.workers) as pool:
        results = pool.map(hold, jobs, chunksize=64)
    kept = np.array([r[2] for r in results])
    reasons = {}
    for r in results:
        if not r[2]:
            reasons[r[3]] = reasons.get(r[3], 0) + 1
    for number, clip in enumerate(clips):
        rows = [k for k, info in enumerate(meta) if info[1] == number]
        clip['kept'] = int(kept[rows].sum())
        print(f"{clip['file']}: кандидатов {clip['candidates']}, прошло {clip['kept']}")
    index = np.flatnonzero(kept)
    qvel_rows = np.array([jobs[k][1] for k in index])
    header = dict(version=g1_getup_human.HUMAN_STARTS_VERSION, rate=1 / dt, source_rate=SOURCE_RATE,
                  band=list(task.STYLE_HEAD_BAND), holdout_subject=args.holdout, clips=clips,
                  filter=dict(hold_seconds=HOLD_SECONDS, flight_seconds=FLIGHT_SECONDS,
                              max_joint_speed=MAX_JOINT_SPEED, style_seconds=STYLE_SECONDS, max_style=MAX_STYLE,
                              body_friction=FRICTION, candidates=len(jobs), kept=int(len(index)), rejected=reasons),
                  data='LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0), retargeted to the G1 by Unitree; not for publication')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f'.pending-{args.out.name}')
    with open(temporary, 'wb') as out:
        np.savez(out, qpos=np.array([results[k][0] for k in index]), qvel=qvel_rows,
                 subject=np.array([meta[k][0] for k in index], dtype=np.int64),
                 clip=np.array([meta[k][1] for k in index], dtype=np.int64),
                 frame=np.array([meta[k][2] for k in index], dtype=np.int64),
                 head=np.array([meta[k][3] for k in index]), lift=np.array([results[k][1] for k in index]),
                 header=np.array(json.dumps(header)))
    os.replace(temporary, args.out)
    digest = hashlib.md5(args.out.read_bytes()).hexdigest()
    g1_getup_human.load(digest, args.out)
    lifts = np.array([r[1] for r in results])
    print(f'{args.out}: прошло {len(index)} из {len(jobs)} ({len(index) / len(jobs):.0%}); отсев {reasons}; '
          f'подъём медиана {np.median(lifts) * 100:.1f} см, макс {lifts.max() * 100:.1f} см')
    print(f'md5 {digest}')


if __name__ == '__main__':
    main()
