"""Fits our MuJoCo SO-101's motors and joints to a record_random.py log of the real arm: per joint the position
actuator's kp, kv and force limit and the joint's damping, friction loss and armature, by cyclic coordinate search on
a log scale, to the open-loop error of WINDOW control steps (from the real state, on the real goals, against the real
angles). The first 80 % of the log fits, the last 20 % judges. Nothing under models/so101 changes (its files are the
model revision): the fit goes to runs/so101_real/sysid.json, and apply(model, fit) puts it on a loaded model.

    .venv\\Scripts\\python.exe scripts\\so101\\sysid.py [runs\\so101_real\\random-….json] [--sweeps 4] [--windows 600]"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
WINDOW, SUBSTEPS = 5, 20
FIELDS = ('kp', 'kv', 'force', 'damping', 'friction', 'armature')
BOUNDS = dict(kp=(1., 3000.), kv=(0., 20.), force=(.3, 10.), damping=(0., 5.), friction=(0., 1.), armature=(1e-4, .2))


class Arm:
    def __init__(self):
        from humanoid_sim.tasks.catalog import make_task
        self.env = make_task('arm_reach', 'so101', {}).unwrapped
        self.model, self.data = self.env.model, self.env.data
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joints = [names.index(n) for n in JOINTS]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS]

    def get(self):
        m = self.model
        return {name: dict(kp=float(m.actuator_gainprm[a, 0]), kv=float(-m.actuator_biasprm[a, 2]),
                           force=float(m.actuator_forcerange[a, 1]), damping=float(m.dof_damping[v]),
                           friction=float(m.dof_frictionloss[v]), armature=float(m.dof_armature[v]))
                for name, a, v in zip(JOINTS, self.actuators, self.vadr)}

    def apply(self, fit):
        apply(self.model, fit)

    def rollouts(self, q, speed, goal, starts):
        """(windows, WINDOW, joints) simulated angles from the real states at `starts` on the real goals."""
        m, d, env = self.model, self.data, self.env
        out = np.empty((len(starts), WINDOW, len(JOINTS)))
        for i, k in enumerate(starts):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = env.home
            d.qpos[self.qadr] = q[k]
            d.qvel[self.vadr] = speed[k]
            d.ctrl[:] = env.home
            mujoco.mj_forward(m, d)
            for j in range(WINDOW):
                d.ctrl[:5] = goal[k + j]
                mujoco.mj_step(m, d, nstep=SUBSTEPS)
                out[i, j] = d.qpos[self.qadr]
        return out


def apply(model, fit):
    """Puts a sysid.json fit on a loaded SO-101 MjModel (actuators and joints by name)."""
    for name, values in fit.items():
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        v = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        model.actuator_gainprm[a, 0] = values['kp']
        model.actuator_biasprm[a, 1] = -values['kp']
        model.actuator_biasprm[a, 2] = -values['kv']
        model.actuator_forcerange[a] = (-values['force'], values['force'])
        model.dof_damping[v] = values['damping']
        model.dof_frictionloss[v] = values['friction']
        model.dof_armature[v] = values['armature']


def load(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    log = record['log']
    q = np.radians([[row['present'][n] for n in JOINTS] for row in log])
    goal = np.radians([[row['goal'][n] for n in JOINTS] for row in log])
    t = np.array([row['t'] for row in log])
    return q, goal, np.gradient(q, t, axis=0)


def error(arm, q, speed, goal, starts):
    """Mean squared open-loop error over the windows' steps, degrees², per joint (joints,) and in all."""
    simulated = arm.rollouts(q, speed, goal, starts)
    truth = np.stack([q[starts + j + 1] for j in range(WINDOW)], 1)
    per_joint = (np.degrees(simulated - truth) ** 2).mean((0, 1))
    return float(per_joint.mean()), per_joint


def still(q, starts):
    truth = np.stack([q[starts + j + 1] for j in range(WINDOW)], 1)
    return (np.degrees(truth - q[starts][:, None]) ** 2).mean((0, 1))


def main():
    parser = argparse.ArgumentParser(description='подгонка моторов и суставов модели SO-101 под запись настоящей руки')
    parser.add_argument('log', nargs='?')
    parser.add_argument('--sweeps', type=int, default=4)
    parser.add_argument('--windows', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    path = Path(args.log) if args.log else Path(sorted(glob.glob(str(ROOT / 'runs/so101_real/random-*.json')))[-1])
    q, goal, speed = load(path)
    n = len(q)
    split = int(.8 * n)
    rng = np.random.default_rng(args.seed)
    train = rng.choice(np.arange(1, split - WINDOW - 1), args.windows, replace=False)
    test = rng.choice(np.arange(split, n - WINDOW - 1), min(args.windows, n - split - WINDOW - 2), replace=False)
    arm = Arm()
    start = arm.get()
    before, before_joint = error(arm, q, speed, goal, test)
    print(f'{path.name}: до подгонки на проверочной части {np.sqrt(before):.2f}° (RMS за {WINDOW} шагов), «стоять» '
          f'{np.sqrt(still(q, test).mean()):.2f}°', flush=True)
    fit = {name: dict(values) for name, values in start.items()}
    best, _ = error(arm, q, speed, goal, train)
    began = time.time()
    for sweep in range(args.sweeps):
        factor = 2. ** (1. / (sweep + 1))
        for name in JOINTS:
            for field in FIELDS:
                for direction in (factor, 1 / factor):
                    while True:
                        value = fit[name][field]
                        trial = value * direction if value > 0 else BOUNDS[field][0] + 1e-3 * (direction > 1)
                        trial = min(max(trial, BOUNDS[field][0]), BOUNDS[field][1])
                        if trial == value:
                            break
                        fit[name][field] = trial
                        arm.apply(fit)
                        score, _ = error(arm, q, speed, goal, train)
                        if score < best:
                            best = score
                        else:
                            fit[name][field] = value
                            arm.apply(fit)
                            break
        print(f'проход {sweep + 1}: обучающая часть {np.sqrt(best):.3f}°, {time.time() - began:.0f} с', flush=True)
    arm.apply(fit)
    after, after_joint = error(arm, q, speed, goal, test)
    print(f'после подгонки на проверочной части {np.sqrt(after):.2f}° (до {np.sqrt(before):.2f}°)')
    for i, name in enumerate(JOINTS):
        print(f'  {name:14s} {np.sqrt(before_joint[i]):.2f} → {np.sqrt(after_joint[i]):.2f}°; '
              + ', '.join(f'{f} {start[name][f]:.3g}→{fit[name][f]:.3g}' for f in FIELDS))
    out = ROOT / 'runs' / 'so101_real' / 'sysid.json'
    out.write_text(json.dumps(dict(log=path.name, window=WINDOW, before=float(np.sqrt(before)),
                                   after=float(np.sqrt(after)), still=float(np.sqrt(still(q, test).mean())),
                                   fit=fit, start=start), ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'записано {out}')


if __name__ == '__main__':
    main()
