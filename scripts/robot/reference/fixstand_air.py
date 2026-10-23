"""CPU only. fixstand_air.py <stage> <out.json>. FixStand in the critic's stiff hoist (hoist.py: rope on the torso,
20 kN/m, anchor 3 m up, feet 5 cm above the floor), as on the robot: Passive (kp 0, kd 3) 2 s from the reset pose, then
FixStand (g1_ctrl gains, 2 s interpolation from where Passive left the joints) held 4 s. Per second of FixStand: the
largest |q - q_fix| and its joints, gravity in the pelvis frame (mean), the gyro (mean per axis and the largest norm).
Physics `ours` and `um` (unitree_mujoco, um_probe.py). Reference for docs/robot-first-run.md, step 1."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SIM / 'scripts' / 'lab' / 'probes'))
sys.path.insert(0, str(SIM))
import um_probe as um  # noqa: E402
import hoist  # noqa: E402

PASSIVE_S, FIX_S, FEET_UP = 2.0, 6.0, 0.05


def run(stage, physics):
    import mujoco
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = um._stage(stage)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    um.apply_physics(raw, physics)
    env.reset(seed=73000, options={'command': [0., 0., 0.], 'phase_steps': 0})
    st = hoist.Hoist(raw, False)
    weight = float(raw.model.body_mass.sum()) * 9.81
    st.anchor = np.array([raw.data.xpos[st.torso][0], raw.data.xpos[st.torso][1], 3.])
    st.rope = 3. - float(raw.data.xpos[st.torso][2]) - FEET_UP - weight / hoist.K
    h = .002 if physics.startswith('um') else .0005  # as hoist.py: kd 10 of the wrists diverges at 2 ms in ours
    h_policy = raw.model.opt.timestep
    raw.model.opt.timestep = h
    st.kp, st.kd = np.zeros(raw.model.nu), np.full(raw.model.nu, 3.)  # Passive (config.yaml)
    for _ in range(int(round(PASSIVE_S / h))):
        st.substep(raw.data.qpos[raw.qadr].copy())
    q0 = raw.data.qpos[raw.qadr].copy()
    st.kp, st.kd = um.FIX_KP.copy(), um.FIX_KD.copy()
    every = int(round(.01 / h))
    err, grav, gyro = [], [], []
    for i in range(int(round(FIX_S / h))):
        st.substep(q0 + (um.FIX_QS - q0) * min(i * h / um.FIX_T, 1.))
        if i % every == 0:
            mujoco.mj_forward(raw.model, raw.data)
            rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
            err.append(np.abs(raw.data.qpos[raw.qadr] - um.FIX_QS))
            grav.append(rotation.T @ np.array([0., 0., -1.]))
            gyro.append(raw.data.qvel[3:6].copy())  # free joint: angular velocity in the pelvis frame
    raw.model.opt.timestep = h_policy
    err, grav, gyro = np.array(err), np.array(grav), np.array(gyro)
    names = [raw.model.joint(j).name for j in raw.joints]
    seconds = []
    per = int(round(1. / .01))
    for s in range(int(FIX_S)):
        w = slice(s * per, (s + 1) * per)
        e = err[w].max(axis=0)
        seconds.append(dict(second=s, max_err=float(e.max()),
                            worst=[[int(j), names[j], float(e[j])] for j in np.argsort(-e)[:4]],
                            gravity=grav[w].mean(axis=0).tolist(), gyro_mean=gyro[w].mean(axis=0).tolist(),
                            gyro_max=float(np.linalg.norm(gyro[w], axis=1).max())))
    env.close()
    return dict(stage=stage, physics=physics, passive_max_err=float(np.abs(q0 - um.FIX_QS).max()), seconds=seconds)


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    res = [run(stage, physics) for physics in ('ours', 'um')]
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    for r in res:
        print('%s: after Passive max|q - q_fix| %.3f rad' % (r['physics'], r['passive_max_err']))
        for s in r['seconds']:
            print('  FixStand %d-%d s: max|q - q_fix| %.4f %s  g %s  gyro mean %s max %.3f' % (
                s['second'], s['second'] + 1, s['max_err'], [(j, round(v, 4)) for j, _, v in s['worst'][:2]],
                np.round(s['gravity'], 3).tolist(), np.round(s['gyro_mean'], 3).tolist(), s['gyro_max']))
