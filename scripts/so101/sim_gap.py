"""How far our MuJoCo SO-101 is from the real arm, one control step at a time: for each tick of a record_random.py
log the simulator starts from the real state (angles; speeds by central differences of the logged angles), takes the
same goal as ArmReachEnv takes a ctrl (position actuators, 20 substeps = 40 ms) and is compared with the real angles
of the next tick. Beside it «standing still» (the next angle = this one). The gripper is held and left out.

    .venv\\Scripts\\python.exe scripts\\so101\\sim_gap.py [runs\\so101_real\\random-….json]"""
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def main():
    from humanoid_sim.tasks.catalog import make_task
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(ROOT / 'runs/so101_real/random-*.json')))[-1])
    record = json.loads(path.read_text(encoding='utf-8'))
    log = record['log']
    q = np.radians([[row['present'][n] for n in JOINTS] for row in log])
    goal = np.radians([[row['goal'][n] for n in JOINTS] for row in log])
    t = np.array([row['t'] for row in log])
    temperatures = [max(row['temperature'].values()) for row in log if 'temperature' in row]
    print(f'{path.name}: тактов {len(log)}, {t[-1]:.0f} с, температура моторов {min(temperatures)}–{max(temperatures)} °C')
    env = make_task('arm_reach', 'so101', {}).unwrapped
    model, data = env.model, env.data
    names = [model.joint(i).name for i in range(model.njnt)]
    joints = [names.index(n) for n in JOINTS]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    speed = np.gradient(q, t, axis=0)
    simulated, still = [], []
    for k in range(1, len(log) - 1):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = env.home
        data.qpos[qadr] = q[k]
        data.qvel[vadr] = speed[k]
        data.ctrl[:] = env.home
        data.ctrl[:5] = goal[k]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data, nstep=20)
        simulated.append(data.qpos[qadr] - q[k + 1])
        still.append(q[k] - q[k + 1])
    sim, hold = np.degrees(np.array(simulated)), np.degrees(np.array(still))
    print('ошибка предсказания следующего такта (40 мс), градусы, RMS:')
    for i, name in enumerate(JOINTS):
        print(f'  {name:14s} симулятор {np.sqrt((sim[:, i] ** 2).mean()):5.2f}   «стоять» {np.sqrt((hold[:, i] ** 2).mean()):5.2f}'
              f'   среднее смещение симулятора {sim[:, i].mean():+.2f}')
    print(f'  все суставы    симулятор {np.sqrt((sim ** 2).mean()):5.2f}   «стоять» {np.sqrt((hold ** 2).mean()):5.2f}')
    lag = np.degrees(goal - q)
    print('отставание от цели (настоящая рука), градусы, RMS:', ', '.join(
        f'{n} {np.sqrt((lag[:, i] ** 2).mean()):.2f}' for i, n in enumerate(JOINTS)))


if __name__ == '__main__':
    main()
