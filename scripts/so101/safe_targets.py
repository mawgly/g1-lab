"""A chain of random SO-101 targets that our MuJoCo model finds safe, for record_random.py --targets: each target in the
box BOX (degrees, LeRobot's = so101_new_calib's angles), the gripper at the model's home, and every pose on the
straight line from the previous target to it (PATH_POINTS of them) free of any contact (the arm with itself and with
the floor, the plane of the base) and with the gripper tip at least TIP_HEIGHT above it.

    .venv\\Scripts\\python.exe scripts\\so101\\safe_targets.py --start <5 angles of the arm now> [--count 2000]
        [--out runs\\so101_real\\targets-wide.json]"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
BOX = dict(shoulder_pan=(-60., 60.), shoulder_lift=(-60., 60.), elbow_flex=(-60., 60.), wrist_flex=(-60., 88.),
           wrist_roll=(-90., 90.))
PATH_POINTS, TIP_HEIGHT = 8, .05  # m above the base's plane; --tip lowers it (floor_from_teach.py: the
# surface lies 0.3-2.6 cm below that plane in front of the arm, touches within 1.1 cm)


def main():
    from humanoid_sim.tasks.catalog import make_task
    parser = argparse.ArgumentParser(description='безопасная цепочка целей для записи SO-101')
    parser.add_argument('--start', required=True, help='5 углов руки сейчас, градусы, через запятую')
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tip', type=float, default=TIP_HEIGHT, help='кончик хвата не ниже, м над основанием')
    parser.add_argument('--out', default=str(ROOT / 'runs/so101_real/targets-wide.json'))
    args = parser.parse_args()
    env = make_task('arm_reach', 'so101', {}).unwrapped
    model, data = env.model, env.probe
    names = [model.joint(i).name for i in range(model.njnt)]
    qadr = model.jnt_qposadr[[names.index(n) for n in JOINTS]]

    def safe(angles):
        data.qpos[:] = env.home
        data.qpos[qadr] = np.radians(angles)
        mujoco.mj_forward(model, data)
        return data.ncon == 0 and data.site_xpos[env.tip_id][2] > args.tip

    rng = np.random.default_rng(args.seed)
    current = np.array([float(v) for v in args.start.split(',')])
    low = np.array([BOX[n][0] for n in JOINTS])
    high = np.array([BOX[n][1] for n in JOINTS])
    chain, tries = [], 0
    first = True
    while len(chain) < args.count and tries < 200 * args.count:
        tries += 1
        target = rng.uniform(low, high)
        points = [current + (target - current) * k / PATH_POINTS for k in range(1, PATH_POINTS + 1)]
        if first:  # from the rest pose the first leg only needs its end and the second half of the way clear
            points = points[PATH_POINTS // 2:]
        if all(safe(p) for p in points):
            chain.append([round(float(v), 2) for v in target])
            current, first = target, False
    print(f'целей {len(chain)} из {tries} попыток; первая {chain[0] if chain else None}')
    Path(args.out).write_text(json.dumps(dict(joints=JOINTS, box=BOX, targets=chain, start=args.start),
                                         ensure_ascii=False), encoding='utf-8')
    print(f'записано {args.out}')


if __name__ == '__main__':
    main()
