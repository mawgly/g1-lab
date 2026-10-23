"""CPU only. enc_bias.py <stage> <out.json> [workers]. Drift of a stage under a constant joint encoder offset b
(uniform +-0.015 rad per joint, as ENCODER_BIAS in training), two models: 'obs' (only the observed angle is q + b,
as in training) and 'calib' (the real robot: the observed angle and the motor PD both use q + b). Our physics,
deterministic, heading hold off; walk 0.5 / 1.0 m/s 10 s from reset (yaw rate after 2 s), stand 30 s (heading change)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, math, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, r'C:\Users\Admin\Documents\Codex\Sim\scripts\lab\probes')
sys.path.insert(0, r'C:\Users\Admin\Documents\Codex\Sim')
import um_probe as um


class Biased(um.Stepper):
    bias = None
    calib = False

    def substep(self, target):
        if not self.calib:
            return super().substep(target)
        raw = self.raw
        q = raw.data.qpos[raw.qadr]
        q += self.bias  # PD reads the biased encoder
        try:
            return super().substep(target)
        finally:
            pass

    # the super().substep reads q at its start and then steps; restore must happen before mj_step: override fully
def substep_calib(self, target):
    import mujoco, torch
    from humanoid_sim.tasks import g1_tensor_math as task
    raw = self.raw
    q, v = raw.data.qpos[raw.qadr].copy() + self.bias, raw.data.qvel[raw.vadr].copy()
    torque = self.kp * (target - q) - self.kd * v
    if raw.motors is not None:
        torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(v.copy()), *raw._motor_tensors).numpy()
    raw.data.ctrl[:] = np.clip(torque, raw.model.actuator_ctrlrange[:, 0], raw.model.actuator_ctrlrange[:, 1])
    self.apply_band()
    mujoco.mj_step(raw.model, raw.data)


def run(job):
    stage, model_kind, draw, plan = job
    import torch
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = um._stage(stage)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    rng = np.random.default_rng(1000 + draw)
    bias = np.zeros(raw.model.nu) if draw < 0 else rng.uniform(-.015, .015, raw.model.nu)
    v = dict(w05=.5, w1=1., stand=0.)[plan]
    n = 1500 if plan == 'stand' else 500
    obs, _ = env.reset(seed=73000, options={'command': [v, 0., 0.], 'phase_steps': 0})
    st = um.Stepper(raw, False)
    if model_kind == 'calib':
        st.bias = bias
        st.substep = substep_calib.__get__(st)
    orig = raw._observation

    def observe(history='replace'):
        raw.data.qpos[raw.qadr] += bias
        try:
            return orig(history)
        finally:
            raw.data.qpos[raw.qadr] -= bias
    raw._observation = observe
    obs = raw._observation('fill')
    head = lambda: math.atan2(raw.data.xmat[raw.pelvis][3], raw.data.xmat[raw.pelvis][0])
    track, fell, start = [], None, None
    for i in range(n):
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        f = st.policy_step(action)
        obs = raw._observation('push')
        if i >= 100:
            track.append(head())
            if start is None:
                start = raw.data.qpos[:2].copy()
        if f:
            fell = i * raw.dt
            break
    tr = np.unwrap(track) if track else np.zeros(2)
    secs = max((len(tr) - 1) * raw.dt, 1e-9)
    moved = raw.data.qpos[:2] - (start if start is not None else raw.data.qpos[:2])
    env.close()
    return dict(model=model_kind, draw=draw, plan=plan, fell=fell, yaw_rate=float((tr[-1] - tr[0]) / secs),
                heading_deg=float(math.degrees(tr[-1] - tr[0])), speed=float(np.linalg.norm(moved) / secs),
                bias=bias.tolist())


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    todo = [(stage, 'obs', -1, p) for p in ('w05', 'w1', 'stand')]
    todo += [(stage, m, d, p) for m in ('obs', 'calib') for d in range(12) for p in ('w05', 'w1', 'stand')]
    with Pool(workers) as pool:
        res = pool.map(run, todo, chunksize=1)
    open(out, 'w', encoding='utf-8').write(json.dumps(res))
    base = {r['plan']: r for r in res if r['draw'] < 0}
    print('no bias:', {p: ('%+.3f' % r['yaw_rate'] if p != 'stand' else '%+.1f deg' % r['heading_deg']) for p, r in base.items()},
          'speed', {p: round(r['speed'], 3) for p, r in base.items()})
    for m in ('obs', 'calib'):
        for p in ('w05', 'w1', 'stand'):
            rs = [r for r in res if r['model'] == m and r['plan'] == p and r['draw'] >= 0]
            falls = sum(r['fell'] is not None for r in rs)
            key = 'heading_deg' if p == 'stand' else 'yaw_rate'
            vals = np.array([r[key] for r in rs if r['fell'] is None])
            print('%-5s %-5s falls %d/%d  %s: mean %+.3f sd %.3f min %+.3f max %+.3f  speed %.3f' % (
                m, p, falls, len(rs), key, vals.mean(), vals.std(ddof=1), vals.min(), vals.max(),
                np.mean([r['speed'] for r in rs if r['fell'] is None])))
