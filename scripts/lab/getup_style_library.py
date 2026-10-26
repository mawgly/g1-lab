"""CPU only. getup_style_library.py [--name lafan1] [--data <folder>] [--holdout 5]
The style library of g1_getup's style reward weight (iteration 53; the critic's lab-critic/it53, human53.py, knn53.py):
human frames getting up, as g1_tensor_math.style_features, written to runs/humanoid_lab/getup_style/<name>.npz
(tasks/g1_getup_style describes the arrays).

Source: LAFAN1's fallAndGetUp clips retargeted to the G1 29 DoF by Unitree (.tools/datasets/lafan1_g1_unitree/g1,
fallAndGetUp*_subject*.csv: 30 frames/s; per frame the root position, its quaternion x y z w and the 29 joints in our
joint order). LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0: the library stays under runs/ (untracked), never published,
and a policy trained with it does not go into a client's package.
Each clip is resampled to the 50 Hz control step (positions linearly, the root quaternion by slerp; joint and pelvis
vertical speeds as np.gradient at 30 Hz, then linearly); the head top (g1_tensor_math.head_height) comes from the
getting-up model's kinematics. Kept: the frames with the head in STYLE_HEAD_BAND (0.3-1.1 m, the rise itself).
Subjects other than --holdout make the library (features), the held-out one the positive control (holdout).
Printed: frames per clip and the held-out frames' distance to the library (median / p90; the critic's 1.26 median at
30 Hz).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / '.tools' / 'datasets' / 'lafan1_g1_unitree' / 'g1'
SOURCE_RATE = 30.


def slerp(q0, q1, s):
    """Unit quaternions between q0 and q1 (n, 4) at s (n,) in [0, 1], the shorter arc."""
    dot = (q0 * q1).sum(-1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1., 1.))
    small = angle < 1e-6
    sin = np.where(small, 1., np.sin(angle))
    a = np.where(small, 1. - s, np.sin((1. - s) * angle) / sin)
    b = np.where(small, s, np.sin(s * angle) / sin)
    q = a[:, None] * q0 + b[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def resample(rows, dt):
    """A clip's (n, 36) rows at SOURCE_RATE as (qpos (k, 36) with the quaternion w x y z, joint speeds (k, 29), pelvis
    vertical speed (k,)) every dt seconds."""
    n = len(rows)
    t = np.arange(n) / SOURCE_RATE
    out = np.arange(0., t[-1] + 1e-9, dt)
    def linear(values):
        return np.stack([np.interp(out, t, values[:, k]) for k in range(values.shape[1])], 1)
    quat = rows[:, [6, 3, 4, 5]]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    index = np.minimum((out * SOURCE_RATE).astype(np.int64), n - 2)
    s = np.clip(out * SOURCE_RATE - index, 0., 1.)
    qpos = np.concatenate([linear(rows[:, :3]), slerp(quat[index], quat[index + 1], s), linear(rows[:, 7:36])], 1)
    speed = linear(np.gradient(rows[:, 7:36], 1 / SOURCE_RATE, axis=0))
    vertical = np.interp(out, t, np.gradient(rows[:, 2], 1 / SOURCE_RATE))
    return qpos, speed, vertical


def main():
    import mujoco
    from humanoid_sim.tasks import g1_getup_bank, g1_getup_style
    from humanoid_sim.tasks import g1_tensor_math as task
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--name', default='lafan1', help='имя библиотеки (файл runs/humanoid_lab/getup_style/<имя>.npz)')
    parser.add_argument('--data', type=Path, default=DATA, help='папка с fallAndGetUp*_subject*.csv')
    parser.add_argument('--holdout', type=int, default=5, help='отложенный исполнитель')
    args = parser.parse_args()
    path = g1_getup_style.library_path(args.name)
    files = sorted(args.data.glob('fallAndGetUp*_subject*.csv'))
    if not files:
        raise SystemExit(f'нет клипов fallAndGetUp в {args.data}')
    env = g1_getup_bank.getup_model_env()
    m, d = env.model, env.data
    if m.nq != 36:
        raise SystemExit(f'модель с {m.nq} координатами, нужен G1 29 DoF')
    joints = tuple(int(j) for j in np.flatnonzero(~env.wrist_mask.numpy()))
    dt = env.dt
    features, holdout, source, times, clips = [], [], [], [], []
    for number, file in enumerate(files):
        subject = int(re.search(r'subject(\d+)', file.stem).group(1))
        qpos, speed, vertical = resample(np.loadtxt(file, delimiter=','), dt)
        heads, rotations = [], []
        for row in qpos:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            heads.append(float(task.head_height(d.xpos[env.torso][None], d.xmat[env.torso].reshape(1, 3, 3))[0]))
            rotations.append(d.xmat[env.pelvis].reshape(3, 3).copy())
        heads = np.array(heads)
        keep = task.style_band(heads)
        frame = task.style_features(qpos[:, 7:], speed, np.array(rotations), qpos[:, 2], vertical, joints)[keep]
        clips.append(dict(file=file.name, subject=subject, frames=int(len(qpos)), kept=int(keep.sum()),
                          seconds=round(len(qpos) * dt, 2)))
        print(f'{file.name}: {len(qpos)} кадров 50 Гц, в полосе головы {int(keep.sum())}'
              + (' (отложенный)' if subject == args.holdout else ''))
        if subject == args.holdout:
            holdout.append(frame)
        else:
            features.append(frame)
            source.append(np.full(len(frame), number, dtype=np.int64))
            times.append(np.flatnonzero(keep) * dt)
    env.close()
    if not features or not holdout:
        raise SystemExit('нужны и база, и отложенный исполнитель')
    features, holdout = np.concatenate(features), np.concatenate(holdout)
    header = dict(version=task.STYLE_VERSION, rate=1 / dt, source_rate=SOURCE_RATE, band=list(task.STYLE_HEAD_BAND),
                  height_scale=task.STYLE_HEIGHT_SCALE, speed_scale=task.STYLE_SPEED_SCALE, holdout_subject=args.holdout,
                  clips=clips, data='LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0), retargeted to the G1 by Unitree; '
                                    'not for publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.pending-{path.name}')
    with open(temporary, 'wb') as out:
        np.savez(out, features=features, holdout=holdout, joints=np.array(joints, dtype=np.int64),
                 source=np.concatenate(source), time=np.concatenate(times), header=np.array(json.dumps(header)))
    os.replace(temporary, path)
    library = g1_getup_style.load(args.name)
    distance, _ = task.style_distance(library['holdout'], library['features'], library['norms'])
    print(f'{path}: база {len(features)} кадров × {features.shape[1]} признаков, отложенных {len(holdout)}; '
          f'd отложенного до базы: медиана {np.median(distance):.2f}, p90 {np.percentile(distance, 90):.2f}')


if __name__ == '__main__':
    main()
