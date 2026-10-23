"""CPU only. speed_terms.py <workers> <out.json>: what holds the G1 below ~1.3 m/s? Policies n6 (exp-8ce0 @ 6039.1 M),
i17 (exp-9435 @ 5549.5 M, max_speed 2), v0 (Unitree's shipped velocity policy) in our env (real_robot physics with the
Unitree motor curves), fixed command vx 1 / 1.5 / 2, 10 s, metrics after 3 s: speed, reward terms per step (n6's weights
for all), gait (duty, flight, step rate, foot tilt, swing ankle height), motor saturation by the torque-speed curve."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, 'scripts', 'lab', 'probes'))
LAB = os.path.join(SIM, 'runs', 'humanoid_lab')
POL = {'n6': os.path.join(LAB, 'exp-8ce0cbd70d0a4f1d', 'stage_006039109632'),
       'i17': os.path.join(LAB, 'exp-9435f12bfb664428', 'stage_005549457408'),
       'v0': 'unitree_v0'}
GROUPS = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll', 'waist', 'shoulder', 'elbow', 'wrist')
WARM = 150
STEPS = 500


def run(job):
    import torch
    torch.set_num_threads(1)
    import stand_turn as st
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.core import stage_task_config
    model, norm, config = st.stage_data(POL[job['pol']])
    weights = stage_task_config(POL['n6'])['reward_weights']
    rec = {'on': False, 'sat': None, 'der': None, 'n': 0, 'tau': None}
    orig = task.motor_torque

    def wrapped(torque, v, x1, x2, y1, y2, sf, vf, activation=.01):
        if rec['on']:
            t, w = torque.numpy(), v.numpy()
            peak = np.where(w * t > 0, y1.numpy(), y2.numpy())
            sp = np.abs(w)
            peak = np.where(sp < x1.numpy(), peak, np.clip(peak - peak / (x2.numpy() - x1.numpy()) * (sp - x1.numpy()), 0, None))
            rec['sat'] += np.abs(t) > peak
            rec['der'] += sp > x1.numpy()
            rec['vmax'] = np.maximum(rec['vmax'], sp / x1.numpy())
            rec['n'] += 1
        return orig(torque, v, x1, x2, y1, y2, sf, vf, activation)
    task.motor_torque = wrapped
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'reward_weights': weights, 'max_episode_steps': 100000})
    raw = env.unwrapped
    nu = raw.model.nu
    rec['sat'], rec['der'], rec['vmax'] = np.zeros(nu), np.zeros(nu), np.zeros(nu)
    names = [raw.model.joint(j).name for j in raw.joints]
    cmd = [job['cmd'], 0., 0.]
    obs, _ = env.reset(seed=job['seed'], options={'command': cmd, 'phase_steps': 0})
    comps, vs, hs, vz, tilt_st, tilt_sw, contact, ank_sw, qd = {}, [], [], [], [], [], [], [], []
    fell = None
    det = job.get('det', True)
    for t in range(STEPS):
        rec['on'] = t >= WARM
        act = model.predict(norm.normalize_obs(obs), deterministic=det)[0]
        obs, r, term, _, info = env.step(act)
        if t >= WARM:
            for k, v in info['reward_components'].items():
                comps[k] = comps.get(k, 0.) + v
            h = st.heading_of(raw.data.qpos)
            v = raw.data.qvel[:2]
            vs.append((v[0] * math.cos(h) + v[1] * math.sin(h), -v[0] * math.sin(h) + v[1] * math.cos(h)))
            hs.append(raw.data.qpos[2])
            vz.append(raw.data.qvel[2])
            sole = raw._sole_height()[0].numpy()
            c = sole < task.SOLE_CONTACT_HEIGHT
            contact.append(c)
            R = raw.data.xmat[raw.feet].reshape(2, 3, 3)
            ang = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
            for f in range(2):
                (tilt_st if c[f] else tilt_sw).append(ang[f])
                if not c[f]:
                    ank_sw.append(raw.data.xpos[raw.feet[f], 2])
            qd.append(np.abs(raw.data.qvel[raw.vadr]))
        if term:
            fell = (t + 1) * raw.dt
            break
    env.close()
    n = max(len(vs), 1)
    contact = np.array(contact) if contact else np.zeros((1, 2), bool)
    td = ((~contact[:-1]) & contact[1:]).sum(axis=0) / (len(contact) * raw.dt) if len(contact) > 1 else [0, 0]
    qd = np.array(qd) if qd else np.zeros((1, nu))
    grp = {}
    for g in GROUPS:
        idx = [i for i, nm in enumerate(names) if g in nm]
        grp[g] = dict(sat=float(rec['sat'][idx].sum() / max(rec['n'] * len(idx), 1)),
                      der=float(rec['der'][idx].sum() / max(rec['n'] * len(idx), 1)),
                      vmax_x1=float(rec['vmax'][idx].max()),
                      qd95=float(np.percentile(qd[:, idx], 95)))
    std = None
    try:
        import torch as _t
        std = np.exp(model.policy.log_std.detach().numpy()).tolist()
    except Exception:
        pass
    v = np.mean(vs, axis=0) if vs else (np.nan, np.nan)
    return dict(pol=job['pol'], cmd=job['cmd'], seed=job['seed'], det=det, fell=fell, vx=float(v[0]), vy=float(v[1]),
                comps={k: c / n for k, c in comps.items()}, total=sum(comps.values()) / n,
                h_mean=float(np.mean(hs)) if hs else None, h_min=float(np.min(hs)) if hs else None,
                vz_rms=float(np.sqrt(np.mean(np.square(vz)))) if vz else None,
                duty=contact.mean(axis=0).tolist(), flight=float((~contact.any(axis=1)).mean()),
                double=float(contact.all(axis=1).mean()), touchdowns_per_s=[float(x) for x in td],
                tilt_stance=float(np.mean(tilt_st)) if tilt_st else None,
                tilt_swing=float(np.mean(tilt_sw)) if tilt_sw else None,
                tilt_swing95=float(np.percentile(tilt_sw, 95)) if tilt_sw else None,
                ankle_swing_mean=float(np.mean(ank_sw)) if ank_sw else None,
                ankle_swing95=float(np.percentile(ank_sw, 95)) if ank_sw else None,
                groups=grp, std=std, names=names)


def main():
    workers, out = int(sys.argv[1]), sys.argv[2]
    pols = sys.argv[3].split(',') if len(sys.argv) > 3 else ['n6', 'i17', 'v0']
    jobs = [dict(pol=p, cmd=c, seed=s) for p in pols for c in (1., 1.5, 2.) for s in (73000, 73001)]
    jobs += [dict(pol=p, cmd=2., seed=73002, det=False) for p in pols if p != 'v0']
    with Pool(workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(out, 'w'), indent=1)
    for r in res:
        print(f"{r['pol']:4s} cmd {r['cmd']:.1f} s{r['seed']} det {int(r['det'])}: vx {r['vx']:+.2f} vy {r['vy']:+.2f}"
              f" fell {r['fell']} total {r['total']:+.2f} h {r['h_mean']:.3f} duty {r['duty'][0]:.2f}/{r['duty'][1]:.2f}"
              f" flight {r['flight']:.2f} td/s {r['touchdowns_per_s'][0]:.2f}")


if __name__ == '__main__':
    main()
