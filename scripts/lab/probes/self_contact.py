"""CPU only. self_contact.py <stage> <out.json> [workers]
Self-contact of the candidate (robot geoms touching each other, as unitree_rl_mjlab's self_collisions term counts with
a 10 N threshold) and whether walking needs it: arms' actions replaced by 0 (target = home) or the arms' torque cut
to what holds them at their own position (target = q). The action then goes as env.step applies it (um_probe.Stepper:
G1VelocityEnv.applied_action, so with lock_wrists the wrist pitch and yaw stay at home in every mode). Prints the mean
wrist pitch (° from home, left / right) over the measured part.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import um_probe as U

ARMS = list(range(15, 29))


def run(job):
    stage, physics, mode, v, phase, seed = job
    import mujoco
    import torch
    from humanoid_sim.goal_check import foot_state, sole_geoms
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = U._stage(stage)
    torch.manual_seed(seed)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    U.apply_physics(raw, physics)
    floor = raw.model.geom('floor').id
    soles = sole_geoms(raw)
    obs, _ = env.reset(seed=seed, options={'command': [v, 0., 0.], 'phase_steps': phase})
    st = U.Stepper(raw)
    heading = lambda: math.atan2(raw.data.xmat[raw.pelvis][3], raw.data.xmat[raw.pelvis][0])
    names = [raw.model.joint(j).name for j in raw.joints]
    wrists = [names.index('left_wrist_pitch_joint'), names.index('right_wrist_pitch_joint')]
    force = np.zeros(6)
    forces, over10, track, feet, start, fell_at, pairs = [], [], [], [], None, None, {}
    arm_tau, wrist = [], []
    for i in range(500):
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0].astype(float)
        policy_out = raw.applied_action(action)  # g1_ctrl's last_action: the exported output, locked wrists at 0
        if mode in ('arms_home', 'arms_home_obs'):
            action[ARMS] = 0.
        elif mode == 'arms_hold':  # target = current q for the arms (only damping acts on them)
            q = raw.data.qpos[raw.qadr]
            action[ARMS] = ((q - raw.home) / raw.scale)[ARMS]
        fell = st.policy_step(action)
        if mode == 'arms_home_obs':  # g1_ctrl-like: last_action observed = what the policy output
            raw.previous_action[:] = policy_out
        obs = raw._observation('push')
        if i >= 100:
            total, n10 = 0., 0
            for k in range(raw.data.ncon):
                c = raw.data.contact[k]
                if c.geom1 == floor or c.geom2 == floor:
                    continue
                mujoco.mj_contactForce(raw.model, raw.data, k, force)
                total += abs(force[0])
                key = tuple(sorted((raw.model.body(raw.model.geom_bodyid[c.geom1]).name,
                                    raw.model.body(raw.model.geom_bodyid[c.geom2]).name)))
                pairs[key] = pairs.get(key, 0.) + abs(force[0])
            forces.append(total)
            over10.append(total > 10.)
            arm_tau.append(float(np.abs(raw.data.ctrl[ARMS]).sum()))
            wrist.append(np.degrees(raw.data.qpos[raw.qadr][wrists] - raw.home[wrists]))
            if start is None:
                start = raw.data.qpos[:2].copy()
            track.append(heading())
            feet.append(foot_state(raw, soles, floor))
        if fell:
            fell_at = (i + 1) * raw.dt
            break
    out = dict(physics=physics, mode=mode, v=v, phase=phase, seed=seed, fell_at=fell_at)
    if len(track) > 10:
        tr = np.unwrap(track)
        secs = (len(tr) - 1) * raw.dt
        moved = raw.data.qpos[:2] - start
        yaw0 = tr[0]
        rows = np.array([[step[f][0], step[f][1]] for step in feet for f in range(2)], dtype=float).reshape(
            len(feet), 2, 2)
        lift = []
        for f in range(2):
            contact = rows[:, f, 0] > 0
            lift.append(float(np.percentile(rows[~contact, f, 1], 95)) if (~contact).any() else 0.)
        n = len(forces)
        out.update(speed=float((moved[0] * math.cos(yaw0) + moved[1] * math.sin(yaw0)) / secs),
                   yaw_rate=float((tr[-1] - tr[0]) / secs), crawl=float(np.linalg.norm(moved)), lift=min(lift),
                   self_force_mean=float(np.mean(forces)), self_over10=float(np.mean(over10)),
                   arm_tau_sum=float(np.mean(arm_tau)), wrist_pitch=np.mean(wrist, axis=0).round(2).tolist(),
                   pairs={f'{a}|{b}': round(f / n, 1) for (a, b), f in sorted(pairs.items(), key=lambda x: -x[1])[:6]})
    env.close()
    return out


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    todo = []
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['policy', 'arms_home', 'arms_hold']
    phys = sys.argv[5].split(',') if len(sys.argv) > 5 else ['ours', 'um']
    for physics in phys:
        for mode in modes:
            todo += [(stage, physics, mode, 0., 0, s) for s in (73000, 73001)]
            for v in (.5, 1.):
                todo += [(stage, physics, mode, v, p, s) for p in (0, 15) for s in (73000, 73001)]
    with Pool(workers) as pool:
        results = pool.map(run, todo, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1))
    for r in results:
        if r.get('speed') is None:
            print(r['physics'], r['mode'], r['v'], r['phase'], r['seed'], 'fell', r['fell_at'])
            continue
        print(f"{r['physics']:4s} {r['mode']:9s} v{r['v']:.1f} ph{r['phase']:2d} s{r['seed']}: fell {r['fell_at']} "
              f"speed {r['speed']:.3f} yaw {r['yaw_rate']:+.3f} crawl {r['crawl']:.2f} lift {1000 * r['lift']:.0f} mm "
              f"| self-contact {r['self_force_mean']:.0f} N, >10 N {r['self_over10']:.2f}, arm |tau| sum "
              f"{r['arm_tau_sum']:.0f} N m | wrist pitch L/R {r['wrist_pitch'][0]:+.1f}/{r['wrist_pitch'][1]:+.1f}° "
              f"| {r['pairs']}")
