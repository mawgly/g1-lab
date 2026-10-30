"""Where the surface is, from teach.py's touches: the gripper tip (our model's gripperframe site, the arm's angles from
the log, the gripper at home) in the model's base frame for every pose; the still stretches (all joints slower than
STILL °/s for at least HOLD s) are the touches, each its median tip; a plane z = a·x + b·y + c through them by least
squares, dropping the stretches more than OUTLIER above it (the arm held still in the air) until none is.

    .venv\\Scripts\\python.exe scripts\\so101\\floor_from_teach.py [runs\\so101_real\\teach-….json]"""
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
STILL, HOLD, OUTLIER = 5., .5, .015


def main():
    from humanoid_sim.tasks.catalog import make_task
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(ROOT / 'runs/so101_real/teach-*.json')))[-1])
    log = json.loads(path.read_text(encoding='utf-8'))['log']
    t = np.array([row['t'] for row in log])
    q = np.array([[row['present'][n] for n in JOINTS] for row in log])
    env = make_task('arm_reach', 'so101', {}).unwrapped
    model, data = env.model, env.probe
    names = [model.joint(i).name for i in range(model.njnt)]
    qadr = model.jnt_qposadr[[names.index(n) for n in JOINTS]]
    tips = np.empty((len(q), 3))
    for i, angles in enumerate(q):
        data.qpos[:] = env.home
        data.qpos[qadr] = np.radians(angles)
        mujoco.mj_kinematics(model, data)
        tips[i] = data.site_xpos[env.tip_id]
    speed = np.abs(np.gradient(q, t, axis=0)).max(1)
    still = speed < STILL
    touches, start = [], None
    for i, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i - 1] - t[start] >= HOLD:
                touches.append(np.median(tips[start:i], 0))
            start = None
    touches = np.array(touches)
    print(f'{path.name}: поз {len(q)}, неподвижных отрезков ≥ {HOLD} с: {len(touches)}; кончик ниже всего '
          f'{tips[:, 2].min() * 100:.1f} см над плоскостью основания модели')
    keep = np.ones(len(touches), bool)
    for _ in range(10):
        if keep.sum() < 3:
            break
        a = np.c_[touches[keep, :2], np.ones(keep.sum())]
        plane, *_ = np.linalg.lstsq(a, touches[keep, 2], rcond=None)
        above = touches[:, 2] - np.c_[touches[:, :2], np.ones(len(touches))] @ plane
        new = above < OUTLIER
        if (new == keep).all():
            break
        keep = new
    for k, (x, y, z) in enumerate(touches):
        print(f'  касание {k + 1}: x {x * 100:+6.1f} y {y * 100:+6.1f} z {z * 100:+6.1f} см' + ('' if keep[k] else
                                                                                           '  (в воздухе)'))
    if keep.sum() >= 3:
        normal = np.array([-plane[0], -plane[1], 1.])
        tilt = np.degrees(np.arccos(1 / np.linalg.norm(normal)))
        residual = above[keep]
        print(f'плоскость по {keep.sum()} касаниям: z = {plane[0]:+.3f}·x {plane[1]:+.3f}·y {plane[2] * 100:+.1f} см, '
              f'наклон {tilt:.1f}°, разброс касаний {np.abs(residual).max() * 100:.1f} см')
        (ROOT / 'runs/so101_real/floor.json').write_text(json.dumps(dict(
            log=path.name, plane=plane.tolist(), tilt_degrees=float(tilt), touches=touches[keep].tolist()),
            ensure_ascii=False, indent=1), encoding='utf-8')
        print('записано runs/so101_real/floor.json')


if __name__ == '__main__':
    main()
