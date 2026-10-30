"""CPU only. um_probe.py <stage> <out.json> [workers] [which]; workers may be left out (um_probe.py <stage> x selftest)
Our CPU env vs unitree_mujoco physics, and the g1_ctrl entry Passive -> FixStand -> Velocity.
The policy's action is applied as env.step applies it (G1VelocityEnv.applied_action: the box clip, then lock_wrists
puts the wrist pitch and yaw at 0), also in the observed last action; selftest checks the whole step against env.step.
physics:
  ours      real_robot as trained/evaluated (Unitree motor curves + friction, armature kp/(2pi*10)^2, implicitfast,
            foot friction 0.6 with priority)
  nomotor   ours without the motor model (plain PD clipped to ctrlrange), joint params unchanged
  um_joint  nomotor + unitree_mujoco g1_29dof.xml joint params (armature 0.01, damping 0.05, frictionloss 0.2/0.1)
  um        um_joint + MuJoCo defaults of unitree_mujoco (Euler integrator, foot/floor friction 1.0, no priority); its
            feet pass through each other, as unitree_mujoco's (a foot_collision stage's capsules switched off)
  um_stale  um + PD torque from the state one physics step old (bridge thread 1 kHz vs physics 500 Hz, worst case)
  ours_caps, um_caps  ours / um with the feet colliding as Unitree's training model (task_config foot_collision's
            capsules, whatever the stage's setting: physics_config builds the environment with them)
kinds (deterministic policy, heading hold off):
  reset      reset walking v, clock at `phase`; 10 s, measured after 2 s
  standwalk  reset standing, 60 steps, walk v 10 s (after 2 s)
  stand      10 s standing (after 2 s)
  stopstart  stand 3 s, walk 1.0 5 s, stand 3 s, walk 0.5 5 s, stand 3 s
  fix_home   FixStand gains holding home 2 s from our reset pose, then Velocity (history = copies of first frame,
             last action 0, clock restarted), command 0 3 s, then v 10 s
  fix_straight  joints at 0 with feet on the floor, FixStand interpolates to home over 2 s, holds 1 s, then as above
  fix_glitch    fix_home, but the first `extra` physics steps of Velocity use target q = 0 (g1_ctrl processed_actions
             is zero-initialised until the policy thread's first step)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
_cache = {}

# unitree_rl_mjlab deploy/robots/g1/config/config.yaml, FSM.FixStand (SDK joint order)
FIX_KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [200] * 3 + [40] * 14, dtype=float)
FIX_KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [5] * 3 + [10] * 14, dtype=float)
FIX_QS = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0, 0, 0] + [0.35, 0.18, 0, 0.87, 0, 0, 0]
                  + [0.35, -0.18, 0, 0.87, 0, 0, 0], dtype=float)
FIX_T = 2.0


def _stage(stage):
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(stage))
        _cache[stage] = (model, norm, stage_task_config(Path(stage)))
    return _cache[stage]


CAPSULES = '_caps'  # physics suffix: the feet collide (foot_collision's capsules), see physics_config


def physics_config(physics):
    """task_config entries a physics variant needs in the environment it runs: the foot capsules for *_caps."""
    return {'foot_collision': True} if physics.endswith(CAPSULES) else {}


def apply_physics(raw, physics):
    import mujoco
    from humanoid_sim.tasks.g1_velocity import load_model
    caps = physics.endswith(CAPSULES)
    if caps:
        if not raw.foot_capsules.size:
            raise ValueError(f'{physics}: build the environment with physics_config (foot_collision)')
        physics = physics[:-len(CAPSULES)]
    if physics == 'ours':
        return
    raw.motors = None
    if physics == 'nomotor':
        return
    # the stage's geoms (task_config foot_collision, hand_collision, terrain's tiles)
    from humanoid_sim.tasks.config import terrain_generator, terrain_seed
    terrain = getattr(raw, 'terrain', None)
    fresh, _ = load_model(foot_collision=raw.foot_collision, hand_collision=raw.hand_collision,
                          terrain=terrain is not None, terrain_seed=terrain_seed(terrain),
                          terrain_generator=terrain_generator(terrain))
    raw.model.dof_armature[:] = fresh.dof_armature
    raw.model.dof_damping[:] = fresh.dof_damping
    raw.model.dof_frictionloss[:] = fresh.dof_frictionloss
    if physics == 'um_joint':
        return
    raw.model.geom_friction[:] = fresh.geom_friction
    raw.model.geom_priority[:] = fresh.geom_priority
    raw.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    # unitree_mujoco's g1_29dof.xml is like ours without foot_collision: its feet touch the world only with the sole
    # spheres and pass through each other, so the bridge copy switches the capsules off (um_caps keeps them: the
    # feet as Unitree's training model has them, with their contact, friction 0.6 and priority 1 kept too).
    if not caps:
        raw.model.geom_contype[raw.foot_capsules.ravel()] = 0
        raw.model.geom_conaffinity[raw.foot_capsules.ravel()] = 0


class Stepper:
    def __init__(self, raw, stale=False):
        from humanoid_sim.tasks.g1_velocity import motor_parameters
        self.raw, self.stale = raw, stale
        # The environment's PD target bounds: joint ranges, knee_floor (as g1_ctrl's deploy.yaml clip).
        self.lo, self.hi = raw.target_low.copy(), raw.target_high.copy()
        self.kp, self.kd = raw.kp.copy(), raw.kd.copy()
        self.prev = None
        names = [raw.model.joint(j).name for j in raw.joints]
        self.peak = motor_parameters(names)[2]  # y1: the real motor's low-speed peak torque
        self.over = np.zeros(raw.model.nu)
        self.maxabs = np.zeros(raw.model.nu)
        self.n = 0
        self.record = False
        self.band = None  # band length m: unitree_mujoco ElasticBand (anchor 0,0,3; 200 N/m, 100 N s/m), torso
        self.torso = raw.model.body('torso_link').id
        self.band_force = 0.

    def apply_band(self):
        raw = self.raw
        raw.data.xfrc_applied[self.torso, :] = 0.
        if self.band is None:
            self.band_force = 0.
            return
        x, dx = raw.data.qpos[:3], raw.data.qvel[:3]
        delta = np.array([0., 0., 3.]) - x
        distance = float(np.linalg.norm(delta))
        direction = delta / distance
        f = (200. * (distance - self.band) - 100. * float(dx @ direction)) * direction
        raw.data.xfrc_applied[self.torso, :3] = f
        self.band_force = float(f[2])

    def substep(self, target):
        import mujoco
        import torch
        from humanoid_sim.tasks import g1_tensor_math as task
        raw = self.raw
        q, v = raw.data.qpos[raw.qadr].copy(), raw.data.qvel[raw.vadr].copy()
        qs, vs = self.prev if (self.stale and self.prev is not None) else (q, v)
        torque = self.kp * (target - qs) - self.kd * vs
        if raw.motors is not None:
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(v.copy()),
                                       *raw._motor_tensors).numpy()
        ctrl = np.clip(torque, raw.model.actuator_ctrlrange[:, 0], raw.model.actuator_ctrlrange[:, 1])
        raw.data.ctrl[:] = ctrl
        if self.record:
            self.over += np.abs(ctrl) > self.peak
            self.maxabs = np.maximum(self.maxabs, np.abs(ctrl))
            self.n += 1
        self.prev = (q, v)
        self.apply_band()
        mujoco.mj_step(raw.model, raw.data)

    def fell(self):
        from humanoid_sim.tasks import g1_tensor_math as task
        raw = self.raw
        rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
        return bool(raw.data.qpos[2] < task.FALL_HEIGHT or rotation[2, 2] < task.FALL_UPRIGHT)

    def policy_step(self, action):
        import mujoco
        raw = self.raw
        action = raw.applied_action(action)  # as env.step: box clip, lock_wrists (wrist pitch and yaw at 0)
        target = np.clip(raw.home + raw.scale * action, self.lo, self.hi)
        for _ in range(raw.frame_skip):
            self.substep(target)
        mujoco.mj_forward(raw.model, raw.data)
        raw.steps += 1
        raw.previous_action[:] = action
        return self.fell()


def plan_of(kind, v):
    if kind == 'reset':
        return [(500, [v, 0., 0.], 100)]
    if kind == 'standwalk':
        return [(60, [0., 0., 0.], None), (500, [v, 0., 0.], 100)]
    if kind == 'stand':
        return [(500, [0., 0., 0.], 100)]
    if kind == 'stopstart':
        return [(150, [0., 0., 0.], 50), (250, [1., 0., 0.], 100), (150, [0., 0., 0.], 50),
                (250, [.5, 0., 0.], 100), (150, [0., 0., 0.], 50)]
    if kind in ('fix_long', 'fix_bandonly', 'fix_short', 'fix_stand12'):
        return [(600, [0., 0., 0.], 100)]
    if kind.startswith('fix'):
        return [(200, [0., 0., 0.], 100), (500, [v, 0., 0.], 100)]
    raise ValueError(kind)


def tilt_deg(raw, body):
    return math.degrees(math.acos(min(1., max(-1., raw.data.xmat[body][8]))))


def place_straight(raw, soles):
    import mujoco
    raw.data.qpos[raw.qadr] = 0.
    raw.data.qvel[:] = 0.
    raw.data.qpos[2] = .793
    mujoco.mj_forward(raw.model, raw.data)
    low = min(raw.data.geom_xpos[g][2] - raw.model.geom_size[g][0] for foot in soles for g in foot)
    raw.data.qpos[2] -= low - .0005
    mujoco.mj_forward(raw.model, raw.data)


def fixstand(raw, st, secs, q_start, h, band=None):
    import mujoco
    st.kp, st.kd = FIX_KP.copy(), FIX_KD.copy()
    h_policy = raw.model.opt.timestep
    raw.model.opt.timestep = h
    n = int(round(secs / h))
    fell_at = None
    for i in range(n):
        a = min(i * h / FIX_T, 1.)
        if band is not None:
            st.band = band(i * h)
        st.substep(q_start + (FIX_QS - q_start) * a)
        if i % 40 == 39:
            mujoco.mj_forward(raw.model, raw.data)
            if st.fell():
                fell_at = (i + 1) * h
                break
    raw.model.opt.timestep = h_policy
    st.prev = None
    mujoco.mj_forward(raw.model, raw.data)
    err = raw.data.qpos[raw.qadr] - FIX_QS
    worst = int(np.argmax(np.abs(err)))
    up = raw.data.xmat[raw.pelvis].reshape(3, 3)[:, 2]
    return dict(fell_at=fell_at, z=float(raw.data.qpos[2]), tilt=tilt_deg(raw, raw.pelvis),
                pitch=math.degrees(math.atan2(up[0], up[2])), band_force=st.band_force,
                max_err=float(np.abs(err[worst])), worst=raw.model.joint(raw.joints[worst]).name,
                err=err.tolist(), max_qvel=float(np.abs(raw.data.qvel[raw.vadr]).max()))


def run(job):
    stage, physics, kind, v, phase, seed, extra = job
    import mujoco
    import torch
    from humanoid_sim.goal_check import foot_state, sole_geoms
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = _stage(stage)
    torch.manual_seed(seed)
    stale = physics.endswith('_stale')
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000,
                                                **physics_config(physics.replace('_stale', ''))})
    raw = env.unwrapped
    apply_physics(raw, physics.replace('_stale', ''))
    torso = raw.model.body('torso_link').id
    soles, floor = sole_geoms(raw), raw.model.geom('floor').id
    heading = lambda: math.atan2(raw.data.xmat[raw.pelvis][3], raw.data.xmat[raw.pelvis][0])
    plan = plan_of(kind, v)
    fix, entry = None, None
    if kind.startswith('fix'):
        obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': 0})
        st = Stepper(raw, stale)
        # FixStand physics step: unitree_mujoco steps 2 ms; for our model 0.5 ms (the robot's PD runs in the motor
        # drivers; with our wrist armature kd 10 at 2 ms is an explicit-damping divergence, kd*h/M = 5)
        h_fix = .002 if (physics.startswith('um') or kind == 'fix_coarse') else .0005
        release = None
        if kind == 'fix_straight':  # no band, straight legs on the floor, FixStand bends to home
            place_straight(raw, soles)
            fix = fixstand(raw, st, 3., raw.data.qpos[raw.qadr].copy(), h_fix)
        elif kind == 'fix_air':  # unitree_mujoco sim2sim: band L 0 (hanging), FixStand in the air, then key 8 x6
            place_straight(raw, soles)
            raw.data.qpos[2] += .45
            mujoco.mj_forward(raw.model, raw.data)
            schedule = lambda s: 0. if s < 3. else min(.1 * (1 + int((s - 3.) / .5)), .6)
            fix = fixstand(raw, st, 7., raw.data.qpos[raw.qadr].copy(), h_fix, band=schedule)
            release = 50
        elif kind == 'fix_bandonly':  # band, no FixStand: the policy starts from our reset pose on the band
            st.band = float(extra)
            fix = fixstand(raw, st, 0., raw.data.qpos[raw.qadr].copy(), h_fix)
            release = 50
        elif kind == 'fix_short':  # FixStand 0.5 s on the floor without band (before it tips), then Velocity
            fix = fixstand(raw, st, .5, raw.data.qpos[raw.qadr].copy(), h_fix)
        elif kind == 'fix_stand12':  # no FixStand, no band: our reset, g1_ctrl-like policy restart
            fix = fixstand(raw, st, 0., raw.data.qpos[raw.qadr].copy(), h_fix)
        elif kind in ('fix_band', 'fix_glitch', 'fix_long'):  # feet on the floor, band length extra (0.6: 6 presses of key 8)
            st.band = .6 if kind == 'fix_glitch' else float(extra)
            fix = fixstand(raw, st, 2., raw.data.qpos[raw.qadr].copy(), h_fix)
            release = 50
        else:  # fix_ground, fix_coarse: no band, from our reset pose
            fix = fixstand(raw, st, 2., raw.data.qpos[raw.qadr].copy(), h_fix)
        # g1_ctrl Velocity enter: deploy.yaml gains, env->reset(): clock 0, last action 0, history = first frame
        st.kp, st.kd = raw.kp.copy(), raw.kd.copy()
        if kind == 'fix_glitch':
            zero = np.clip(np.zeros(raw.model.nu), st.lo, st.hi)
            for _ in range(int(extra)):
                st.substep(zero)
            mujoco.mj_forward(raw.model, raw.data)
        raw.steps, raw.phase_steps = 0, 2
        raw.previous_action[:] = 0.
        raw.set_command(np.zeros(3))
        obs = raw._observation('fill')
        entry = dict(z0=float(raw.data.qpos[2]), feet0=raw.data.xpos[raw.feet, :2].copy().tolist(),
                     band_force=st.band_force, weight=float(raw.model.body_mass.sum() * 9.81))
    else:
        release = None
        obs, _ = env.reset(seed=seed, options={'command': plan[0][1], 'phase_steps': int(phase) % 30})
        st = Stepper(raw, stale)
    early0 = release or 0
    st.record = True
    fell_at, segs, t = None, [], 0
    first_target, early_tilt, early_z, early_liftoffs = None, 0., 9., 0
    prev_contact = None
    for si, (n, command, measure_from) in enumerate(plan):
        if si:
            raw.set_command(command)
            obs = raw._observation('replace')
        track, feet, start, leans, qs = [], [], None, [], []
        for i in range(n):
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            if first_target is None:
                a = raw.applied_action(action)
                jump = raw.scale * a
                k = int(np.argmax(np.abs(jump)))
                first_target = dict(max=float(abs(jump[k])), joint=raw.model.joint(raw.joints[k]).name,
                                    mean=float(np.abs(jump).mean()))
            fell = st.policy_step(action)
            t += 1
            if release is not None and t == release:
                st.band = None  # key 9
            obs = raw._observation('push')
            if early0 < t <= early0 + 50:  # first second after the start / the band release
                early_tilt = max(early_tilt, tilt_deg(raw, raw.pelvis))
                early_z = min(early_z, float(raw.data.qpos[2]))
                rows = foot_state(raw, soles, floor)
                contact = [r[0] for r in rows]
                if prev_contact is not None:
                    early_liftoffs += sum(1 for a0, a1 in zip(prev_contact, contact) if a0 and not a1)
                prev_contact = contact
            if measure_from is not None and i >= measure_from:
                if start is None:
                    start = raw.data.qpos[:2].copy()
                    yaw_start = heading()
                track.append(heading())
                feet.append(foot_state(raw, soles, floor))
                up = raw.data.xmat[torso].reshape(3, 3)[:, 2]
                leans.append(math.degrees(math.atan2(up[0], up[2])))
                qs.append(raw.data.qpos[raw.qadr].copy())
            if fell:
                fell_at = t * raw.dt
                break
        if measure_from is not None and len(track) > 10:
            tr = np.unwrap(track)
            moved = raw.data.qpos[:2] - start
            rows = np.array([[step[f][0], step[f][1], step[f][2]] for step in feet for f in range(2)],
                            dtype=float).reshape(len(feet), 2, 3)
            lift, tilt = [], []
            for f in range(2):
                contact = rows[:, f, 0] > 0
                lift.append(float(np.percentile(rows[~contact, f, 1], 95)) if (~contact).any() else 0.)
                tilt.append(float(rows[contact, f, 2].mean()) if contact.any() else float('nan'))
            secs = (len(tr) - 1) * raw.dt
            yaw0 = tr[0]
            fwd = float(moved[0] * math.cos(yaw0) + moved[1] * math.sin(yaw0))
            side = float(-moved[0] * math.sin(yaw0) + moved[1] * math.cos(yaw0))
            liftoffs = int(((rows[1:, :, 0] == 0) & (rows[:-1, :, 0] > 0)).sum())
            segs.append(dict(cmd=command, yaw_rate=float((tr[-1] - tr[0]) / secs), speed=fwd / secs, side=side,
                             crawl=float(np.linalg.norm(moved)), yaw_change=float(math.degrees(tr[-1] - tr[0])),
                             lift=min(lift), tilt=float(np.nanmax(tilt)), lean=float(np.mean(leans)),
                             liftoffs=liftoffs, secs=secs, q_mean=np.mean(qs, axis=0).tolist()))
        if fell_at is not None:
            break
    env.close()
    over = (st.over / max(st.n, 1)).tolist()
    return dict(physics=physics, kind=kind, v=v, phase=phase, seed=seed, extra=extra, fell_at=fell_at, segs=segs,
                fix=fix, entry=entry, first=first_target, early=dict(tilt=early_tilt, z=early_z,
                                                                       liftoffs=early_liftoffs),
                over_peak=over, max_ctrl=st.maxabs.tolist())


def selftest(stage):
    """physics ours, not stale, must reproduce env.step exactly."""
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = _stage(stage)
    finals = []
    for mine in (False, True):
        env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 5000})
        raw = env.unwrapped
        obs, _ = env.reset(seed=73000, options={'command': [1., 0., 0.], 'phase_steps': 0})
        st = Stepper(raw, False)
        for i in range(200):
            a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            if mine:
                st.policy_step(a)
                obs = raw._observation('push')
            else:
                obs = env.step(a)[0]
        finals.append((raw.data.qpos.copy(), obs.copy()))
        env.close()
    print('selftest qpos diff', float(np.abs(finals[0][0] - finals[1][0]).max()),
          'obs diff', float(np.abs(finals[0][1] - finals[1][1]).max()))


def stability(stage):
    """Explicit PD damping ratio kd*h/M_ii per joint (> 2 diverges with an explicit velocity term)."""
    import mujoco
    from humanoid_sim.tasks.catalog import make_task
    _, _, config = _stage(stage)
    for physics in ('ours', 'um'):
        env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False})
        raw = env.unwrapped
        apply_physics(raw, physics)
        env.reset(seed=1, options={'command': [0., 0., 0.]})
        mii = []
        for d in raw.vadr:
            e, res = np.zeros(raw.model.nv), np.zeros(raw.model.nv)
            e[d] = 1.
            mujoco.mj_mulM(raw.model, raw.data, res, e)
            mii.append(res[d])
        mii = np.array(mii)
        h = raw.model.opt.timestep
        names = [raw.model.joint(j).name for j in raw.joints]
        rp, rf = raw.kd * h / mii, FIX_KD * h / mii
        mass = float(raw.model.body_mass.sum())
        com = float(raw.data.subtree_com[0][2])
        ankle = float(raw.data.xanchor[raw.model.joint('left_ankle_pitch_joint').id][2])
        print(physics, f'mass {mass:.2f} kg, COM z {com:.3f}, ankle z {ankle:.3f}, m*g*h {mass * 9.81 * (com - ankle):.0f} '
              f'N m/rad vs ankle-pitch kp x2: FixStand {2 * FIX_KP[4]:.0f}, policy {2 * raw.kp[4]:.0f}')
        print(physics, 'armature', np.round(raw.model.dof_armature[raw.vadr], 4).tolist())
        print(physics, 'M_ii', np.round(mii, 4).tolist())
        for label, r in (('policy kd', rp), ('FixStand kd', rf)):
            order = np.argsort(-r)[:5]
            print(physics, label, 'max kd*h/M', [(names[i], round(float(r[i]), 3)) for i in order])
        env.close()


def jobs(stage, which):
    j = []
    S = (73000, 73001)
    if which in ('all', 'physics'):
        for physics in ('ours', 'nomotor', 'um_joint', 'um', 'um_stale'):
            for v in (.5, 1.):
                for b in (0, 15):
                    j += [(stage, physics, 'reset', v, b, s, 0) for s in S]
                j += [(stage, physics, 'standwalk', v, 0, s, 0) for s in S]
            j += [(stage, physics, 'stand', 0., 0, s, 0) for s in S]
            j += [(stage, physics, 'stopstart', 0., 0, s, 0) for s in S]
    if which == 'long':
        for physics in ('ours', 'um'):
            for L in (.6, .8):
                j += [(stage, physics, 'fix_long', 0., 0, s, L) for s in S]
    if which == 'spin':
        for physics in ('ours', 'um'):
            j += [(stage, physics, 'fix_bandonly', 0., 0, s, .6) for s in S]
            j += [(stage, physics, 'fix_short', 0., 0, s, 0) for s in S]
            j += [(stage, physics, 'fix_stand12', 0., 0, s, 0) for s in S]
    if which in ('all', 'fix'):
        for physics in ('ours', 'um'):
            for kind in ('fix_ground', 'fix_straight'):
                j += [(stage, physics, kind, .5, 0, s, 0) for s in S]
            for v in (.5, 1.):
                j += [(stage, physics, 'fix_air', v, 0, s, 0) for s in S]
                for L in (.6, .8):
                    j += [(stage, physics, 'fix_band', v, 0, s, L) for s in S]
            for k in (1, 5):
                j += [(stage, physics, 'fix_glitch', .5, 0, s, k) for s in S]
        j += [(stage, 'ours', 'fix_coarse', .5, 0, s, 0) for s in S]
    return j


if __name__ == '__main__':
    stage, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    workers = int(rest.pop(0)) if rest and rest[0].isdigit() else 10
    which = rest[0] if rest else 'all'
    if which == 'selftest':
        selftest(stage)
        raise SystemExit
    if which == 'stability':
        stability(stage)
        raise SystemExit
    todo = jobs(stage, which)
    with Pool(workers) as pool:
        results = pool.map(run, todo, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1))
    print(len(results), 'runs')
