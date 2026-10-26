"""CPU only. getup_amp_library.py [--name lafan1] [--data <folder>] [--holdout 5]
The human state pairs of g1_getup's AMP reward weight (iteration 55; humanoid_sim/getup_amp describes them), written to
runs/humanoid_lab/getup_style/<name>-amp.npz next to the style library of the same name.

Source as getup_style_library.py: LAFAN1's fallAndGetUp clips retargeted to the G1 29 DoF by Unitree
(.tools/datasets/lafan1_g1_unitree/g1, 30 frames/s), each resampled to the 50 Hz control step (joints linearly, the root
quaternion by slerp). Every frame goes through mj_kinematics of the getting-up model, giving getup_amp.amp_state (the
free joints, the gravity in the pelvis frame, pelvis and head heights, the palms, elbows, knees and feet relative to the
pelvis in its heading frame); a pair is two neighbouring 50 Hz frames of one clip, kept when the second one has the head
in STYLE_HEAD_BAND (0.3-1.1 m), as the policy's pairs are paid. Subjects other than --holdout make the library
(human), the held-out one the positive control (heldout, train/disc_heldout). LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND
4.0: the file stays under runs/ (untracked), never published.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from getup_style_library import DATA, SOURCE_RATE, resample  # noqa: E402


def main():
    import mujoco
    from humanoid_sim import getup_amp
    from humanoid_sim.tasks import g1_getup_bank
    from humanoid_sim.tasks import g1_tensor_math as task
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--name', default='lafan1',
                        help='имя библиотеки стиля (файл runs/humanoid_lab/getup_style/<имя>-amp.npz)')
    parser.add_argument('--data', type=Path, default=DATA, help='папка с fallAndGetUp*_subject*.csv')
    parser.add_argument('--holdout', type=int, default=5, help='отложенный исполнитель')
    args = parser.parse_args()
    path = getup_amp.library_path(args.name)
    files = sorted(args.data.glob('fallAndGetUp*_subject*.csv'))
    if not files:
        raise SystemExit(f'нет клипов fallAndGetUp в {args.data}')
    env = g1_getup_bank.getup_model_env()
    m, d = env.model, env.data
    if m.nq != 36 or list(env.qadr) != list(range(7, 36)):
        raise SystemExit(f'модель с {m.nq} координатами, нужен G1 29 DoF')
    joints = tuple(int(j) for j in np.flatnonzero(~env.wrist_mask.numpy()))
    bodies, offsets = getup_amp.point_bodies(m)
    human, heldout, clips = [], [], []
    for file in files:
        subject = int(re.search(r'subject(\d+)', file.stem).group(1))
        qpos, _, _ = resample(np.loadtxt(file, delimiter=','), env.dt)
        xpos, xmat = [], []
        for row in qpos:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            xpos.append(d.xpos.copy())
            xmat.append(d.xmat.copy())
        xpos, xmat = np.array(xpos), np.array(xmat)
        head = task.head_height(xpos[:, env.torso], xmat[:, env.torso].reshape(-1, 3, 3))
        states = getup_amp.amp_state(qpos[:, 7:], qpos[:, :3], xmat[:, env.pelvis].reshape(-1, 3, 3), head,
                                     getup_amp.amp_points(xpos, xmat, bodies, offsets), joints)
        keep = task.style_band(head[1:])
        pairs = np.concatenate([states[:-1], states[1:]], axis=1)[keep]
        clips.append(dict(file=file.name, subject=subject, frames=int(len(qpos)), pairs=int(len(pairs))))
        print(f'{file.name}: {len(qpos)} кадров 50 Гц, пар в полосе головы {len(pairs)}'
              + (' (отложенный)' if subject == args.holdout else ''))
        (heldout if subject == args.holdout else human).append(pairs)
    env.close()
    if not human or not heldout:
        raise SystemExit('нужны и база, и отложенный исполнитель')
    human, heldout = np.concatenate(human), np.concatenate(heldout)
    header = dict(version=getup_amp.AMP_VERSION, rate=1 / env.dt, source_rate=SOURCE_RATE,
                  band=list(task.STYLE_HEAD_BAND), points=[name for name, _ in getup_amp.AMP_POINTS],
                  holdout_subject=args.holdout, clips=clips,
                  data='LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0), retargeted to the G1 by Unitree; not for publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.pending-{path.name}')
    with open(temporary, 'wb') as out:
        np.savez(out, human=human, heldout=heldout, joints=np.array(joints, dtype=np.int64),
                 header=np.array(json.dumps(header)))
    os.replace(temporary, path)
    library = getup_amp.load_library(args.name)
    print(f'{path}: база {len(library["human"])} пар × {library["human"].shape[1]} чисел, '
          f'отложенных {len(library["heldout"])}; md5 {library["md5"]}')


if __name__ == '__main__':
    main()
