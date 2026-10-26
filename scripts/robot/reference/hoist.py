"""CPU only. hoist.py <stage> <out.json> [workers]. The policy in a gantry (hoist) before the floor: a stiff rope on the
torso (tension only, 20 kN/m, 1 kN s/m, anchor 3 m above the start), FixStand 2 s (g1_ctrl gains), then Velocity.
Plans: air5 / air15 (feet 5 / 15 cm above the floor; cmd 0 10 s, then 0.5 m/s 5 s), half (feet on the floor, rope
holds ~50% of the weight; cmd 0 10 s), lower (air5, cmd 0 3 s, then the rope paid out 2 cm/s for 6 s, stand 8 s,
walk 0.5 m/s 5 s, stand 4 s), slack (rope 5 cm slack; stand 10 s, walk 0.5 5 s, stand 4 s), floor (no rope, ref).
A height-command stage (task_config height_command) runs them at the nominal height c = 0.78 (the right stick
centred, height term 0; `height` in each segment) and also air5_crouch (air5, cmd 0: c 0.78 5 s, 0.70 8 s, 0.64 8 s,
0.78 5 s), the first crouch check on the robot (docs/robot-first-run.md, «Экспериментальный пакет №9 (присед)»)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, math, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, r'C:\Users\Admin\Documents\Codex\Sim\scripts\lab\probes')
sys.path.insert(0, r'C:\Users\Admin\Documents\Codex\Sim')
import um_probe as um

K, C = 20000., 1000.


class Hoist(um.Stepper):
    rope = None  # rope length (m) from the anchor, or None
    anchor = None

    def apply_band(self):
        raw = self.raw
        raw.data.xfrc_applied[self.torso, :] = 0.
        self.band_force = 0.
        if self.rope is None:
            return
        x = raw.data.xpos[self.torso]
        delta = self.anchor - x
        d = float(np.linalg.norm(delta))
        if d <= self.rope:
            return
        u = delta / d
        vel = np.zeros(6)
        import mujoco
        mujoco.mj_objectVelocity(raw.model, raw.data, mujoco.mjtObj.mjOBJ_BODY, self.torso, vel, 0)
        t = max(0., K * (d - self.rope) - C * float(vel[3:] @ u))
        raw.data.xfrc_applied[self.torso, :3] = t * u
        self.band_force = t


PLANS = {
    'air5': [('stand', 500, 0.), ('walk', 250, .5)],
    'air15': [('stand', 500, 0.), ('walk', 250, .5)],
    'half': [('stand', 500, 0.)],
    'lower': [('hang', 150, 0.), ('pay', 300, 0.), ('stand', 400, 0.), ('walk', 250, .5), ('stop', 200, 0.)],
    'slack': [('stand', 500, 0.), ('walk', 250, .5), ('stop', 200, 0.)],
    'floor': [('stand', 500, 0.), ('walk', 250, .5), ('stop', 200, 0.)],
}
# Height-command stages only: segments (name, steps, v, c), c the height command set at the segment's start.
CROUCH_PLANS = {
    'air5_crouch': [('stand', 250, 0., .78), ('crouch70', 400, 0., .70), ('crouch64', 400, 0., .64),
                    ('rise', 250, 0., .78)],
}


def plans(stage):
    """The plans a stage runs: CROUCH_PLANS too with task_config height_command."""
    from pathlib import Path
    from humanoid_sim.core import stage_task_config
    return list(PLANS) + (list(CROUCH_PLANS) if stage_task_config(Path(stage)).get('height_command') else [])


def run(job):
    stage, physics, plan_name = job
    import mujoco, torch
    from humanoid_sim.tasks.catalog import make_task
    import stand_turn  # its stage_data also takes an export_g1 folder (policy.onnx in its stage's task)
    model, norm, config = stand_turn.stage_data(stage)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    um.apply_physics(raw, physics)
    obs, _ = env.reset(seed=73000, options={'command': [0., 0., 0.], 'phase_steps': 0})
    st = Hoist(raw, False)
    mass = float(raw.model.body_mass.sum())
    g = mass * 9.81
    floor = raw.model.geom('floor').id
    torso_z = float(raw.data.xpos[st.torso][2])
    st.anchor = np.array([raw.data.xpos[st.torso][0], raw.data.xpos[st.torso][1], 3.])
    base = 3. - torso_z
    stretch = g / K
    if plan_name in ('air5', 'lower', 'air5_crouch'):
        st.rope = base - .05 - stretch
    elif plan_name == 'air15':
        st.rope = base - .15 - stretch
    elif plan_name == 'half':
        st.rope = base - .5 * stretch
    elif plan_name == 'slack':
        st.rope = base + .05
    h_fix = .002 if physics.startswith('um') else .0005
    fix = um.fixstand(raw, st, 2., raw.data.qpos[raw.qadr].copy(), h_fix)
    st.kp, st.kd = raw.kp.copy(), raw.kd.copy()
    raw.steps, raw.phase_steps = 0, 2
    raw.previous_action[:] = 0.
    raw.set_command(np.zeros(3))
    obs = raw._observation('fill')
    names = [raw.model.joint(j).name for j in raw.joints]
    legs = [i for i, n in enumerate(names) if any(p in n for p in ('hip', 'knee', 'ankle'))]
    segs, prev_a, fell_at, t = [], None, None, 0
    sole_min = None
    for name, n, v, *height in {**PLANS, **CROUCH_PLANS}[plan_name]:
        raw.set_command(np.array([v, 0., 0.]))
        if height:
            raw.set_height(height[0])
        obs = raw._observation('replace')
        rec = dict(qv=[], tau=[], at_lim=[], da=[], selff=[], rope=[], tilt=[], footz=[], kneeq=[], pelz=[],
                   contacts=[])
        for i in range(n):
            if name == 'pay':
                st.rope += .02 * raw.dt
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            a = raw.applied_action(action)
            target = np.clip(raw.home + raw.scale * a, st.lo, st.hi)
            rec['at_lim'].append(float(np.mean((np.abs(target - st.lo) < 1e-6) | (np.abs(target - st.hi) < 1e-6))))
            rec['da'].append(float(np.abs(a - prev_a).mean()) if prev_a is not None else 0.)
            prev_a = a.copy()
            fell = st.policy_step(action)
            t += 1
            obs = raw._observation('push')
            d = raw.data
            rec['qv'].append(np.abs(d.qvel[raw.vadr][legs]).max())
            rec['tau'].append(np.abs(d.ctrl[legs]).sum())
            sf = 0.
            fc = 0
            for k in range(d.ncon):
                con = d.contact[k]
                f6 = np.zeros(6)
                mujoco.mj_contactForce(raw.model, d, k, f6)
                if con.geom1 == floor or con.geom2 == floor:
                    fc += 1
                else:
                    sf = max(sf, float(np.linalg.norm(f6[:3])))
            rec['selff'].append(sf)
            rec['contacts'].append(fc)
            rec['rope'].append(st.band_force / g)
            rec['tilt'].append(um.tilt_deg(raw, raw.pelvis))
            rec['footz'].append(min(float(d.xpos[f][2]) for f in raw.feet))
            rec['kneeq'].append(float(min(d.qpos[raw.qadr][names.index('left_knee_joint')],
                                          d.qpos[raw.qadr][names.index('right_knee_joint')])))
            rec['pelz'].append(float(d.qpos[2]))
            if fell and st.rope is None or (fell and plan_name in ('floor', 'slack') and d.qpos[2] < .5):
                fell_at = t * raw.dt
                break
        R = {k: np.array(v) for k, v in rec.items()}
        segs.append(dict(name=name, v=v, steps=len(R['qv']),
                         qv_leg_max=float(R['qv'].max()), qv_leg_p95=float(np.percentile(R['qv'], 95)),
                         tau_leg_mean=float(R['tau'].mean()), tau_leg_max=float(R['tau'].max()),
                         at_lim=float(R['at_lim'].mean()), da=float(R['da'].mean()), da_max=float(R['da'].max()),
                         self_max=float(R['selff'].max()), self_share=float((R['selff'] > 10).mean()),
                         rope_mean=float(R['rope'].mean()), rope_max=float(R['rope'].max()),
                         tilt_max=float(R['tilt'].max()), footz_min=float(R['footz'].min()),
                         footz_mean=float(R['footz'].mean()), knee_min=float(R['kneeq'].min()),
                         pelz_min=float(R['pelz'].min()), floor_contact_share=float((R['contacts'] > 0).mean()),
                         xy=d.qpos[:2].tolist(), **({'height': raw.height} if raw.height_command else {})))
        if fell_at is not None:
            break
    over = (st.over / max(st.n, 1))
    env.close()
    return dict(stage=stage, physics=physics, plan=plan_name, fell_at=fell_at, fix=dict(z=fix['z'], tilt=fix['tilt']),
                segs=segs, over_peak_max=float(over.max()), over_peak_joint=names[int(over.argmax())],
                max_ctrl=st.maxabs.tolist(), names=names)


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    todo = [(stage, ph, p) for ph in ('ours', 'um') for p in plans(stage)]
    with Pool(workers) as pool:
        res = pool.map(run, todo, chunksize=1)
    open(out, 'w', encoding='utf-8').write(json.dumps(res, indent=1))
    for r in res:
        print('%-5s %-6s fell %s  fix z %.3f  over_peak %.3f (%s)' % (r['physics'], r['plan'], r['fell_at'], r['fix']['z'],
              r['over_peak_max'], r['over_peak_joint']))
        for s in r['segs']:
            print('   %-8s v%.1f%s n%4d qv_leg max %5.1f p95 %5.1f  tau_leg %5.0f/%5.0f  lim %.2f  da %.4f/%.3f  self %5.0fN %.2f  rope %.2f/%.2f  tilt %4.1f  footz %.3f  knee_min %.3f  pelz %.3f  floor %.2f  xy %+.2f %+.2f' % (
                s['name'], s['v'], ' c%.2f' % s['height'] if 'height' in s else '', s['steps'], s['qv_leg_max'], s['qv_leg_p95'], s['tau_leg_mean'], s['tau_leg_max'],
                s['at_lim'], s['da'], s['da_max'], s['self_max'], s['self_share'], s['rope_mean'], s['rope_max'],
                s['tilt_max'], s['footz_min'], s['knee_min'], s['pelz_min'], s['floor_contact_share'], *s['xy']))
