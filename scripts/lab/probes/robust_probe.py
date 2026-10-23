"""CPU only. robust_probe.py <stage> <out.json> [workers] [which]; workers may be left out (<stage> x selftest)
Robustness of a stage to the robot's timing: gait-clock offsets and jumps, clock jitter, action delay.
Deterministic policy, heading hold off (as goal_check and as g1_ctrl), the env's own physics re-done step by step
so the applied PD target can lag the policy output by `delay` physics substeps (10 per control step, 2 ms each);
the action is applied as env.step applies it (G1VelocityEnv.applied_action: the box clip, then lock_wrists puts the
wrist pitch and yaw at 0, as the exported policy's output), and the observed last action is that applied action
without delay (as g1_ctrl's last_action).
kinds:
  reset      goal_check start: reset walking v, clock at `phase` steps; 10 s, measured after 2 s
  standwalk  robot-like: reset standing with the clock at `phase`, stand 60 steps, walk v 10 s (after 2 s)
  jump       reset walking v at clock `phase`, 5 s, then the clock jumps by `k` steps, 10 s more (after 2 s)
  jitter     reset walking v at clock `phase`, every frame the clock reads +-U{0..k} steps off
  stand      reset standing 10 s (crawl over 2..10 s)
  stopstart  stand 3 s, walk 1.0 5 s, stand 3 s, walk 0.5 5 s, stand 3 s
"""
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
_cache = {}


def _stage(stage):
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(stage))
        _cache[stage] = (model, norm, stage_task_config(Path(stage)))
    return _cache[stage]


class Stepper:
    def __init__(self, raw, delay):
        self.raw, self.delay = raw, int(delay)
        lo, hi = raw.target_low, raw.target_high  # the environment's PD target bounds: joint ranges, knee_floor
        self.lo, self.hi = lo, hi
        home_target = np.clip(raw.home, lo, hi)
        self.sub = [home_target] * (self.delay + raw.frame_skip)  # applied targets, one per physics substep

    def step(self, action):
        import mujoco
        import torch
        from humanoid_sim.tasks import g1_tensor_math as task
        raw = self.raw
        action = raw.applied_action(action)  # as env.step: box clip, lock_wrists (wrist pitch and yaw at 0)
        target = np.clip(raw.home + raw.scale * action, self.lo, self.hi)
        self.sub.extend([target] * raw.frame_skip)
        n = len(self.sub)
        for j in range(raw.frame_skip):
            applied = self.sub[n - raw.frame_skip + j - self.delay]
            joint_velocity = raw.data.qvel[raw.vadr]
            torque = raw.kp * (applied - raw.data.qpos[raw.qadr]) - raw.kd * joint_velocity
            if raw.motors is not None:
                torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                           *raw._motor_tensors).numpy()
            raw.data.ctrl[:] = np.clip(torque, raw.model.actuator_ctrlrange[:, 0], raw.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(raw.model, raw.data)
        mujoco.mj_forward(raw.model, raw.data)
        self.sub = self.sub[-(self.delay + raw.frame_skip):]
        raw.steps += 1
        raw.previous_action[:] = action
        rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
        return bool(raw.data.qpos[2] < task.FALL_HEIGHT or rotation[2, 2] < task.FALL_UPRIGHT)


def plan_of(kind, v):
    """[(steps, command, measure_from)]; measure_from: steps into the segment after which it is measured."""
    if kind in ('reset', 'jitter'):
        return [(500, [v, 0., 0.], 100)]
    if kind == 'standwalk':
        return [(60, [0., 0., 0.], None), (500, [v, 0., 0.], 100)]
    if kind == 'jump':
        return [(250, [v, 0., 0.], None), (500, [v, 0., 0.], 100)]
    if kind == 'stand':
        return [(500, [0., 0., 0.], 100)]
    if kind == 'stopstart':
        return [(150, [0., 0., 0.], 50), (250, [1., 0., 0.], 100), (150, [0., 0., 0.], 50),
                (250, [.5, 0., 0.], 100), (150, [0., 0., 0.], 50)]
    raise ValueError(kind)


def run(job):
    stage, kind, v, phase, k, delay, seed = job
    import torch
    from humanoid_sim.goal_check import foot_state, sole_geoms
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = _stage(stage)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 991)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 5000})
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    torso = raw.model.body('torso_link').id
    soles, floor = sole_geoms(raw), raw.model.geom('floor').id
    heading = lambda: math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0])
    plan = plan_of(kind, v)
    obs, _ = env.reset(seed=seed, options={'command': plan[0][1], 'phase_steps': int(phase) % 30})
    base_phase = raw.phase_steps
    stepper = Stepper(raw, delay)
    fell_at, segs, t = None, [], 0
    for si, (n, command, measure_from) in enumerate(plan):
        if si:
            raw.set_command(command)
            obs = raw._observation('replace')
        if kind == 'jump' and si == 1:
            raw.phase_steps = base_phase + k  # the clock skips k control steps (negative: repeats)
            base_phase = raw.phase_steps
            obs = raw._observation('replace')
        track, feet, start, leans = [], [], None, []
        for i in range(n):
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            fell = stepper.step(action)
            t += 1
            if kind == 'jitter':
                raw.phase_steps = base_phase + int(rng.integers(-k, k + 1))
            obs = raw._observation('push')
            if measure_from is not None and i >= measure_from:
                if start is None:
                    start = raw.data.qpos[:2].copy()
                track.append(heading())
                feet.append(foot_state(raw, soles, floor))
                up = raw.data.xmat[torso].reshape(3, 3)[:, 2]
                leans.append(math.degrees(math.atan2(up[0], up[2])))
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
                             crawl=float(np.linalg.norm(moved)), lift=min(lift), tilt=float(np.nanmax(tilt)),
                             lean=float(np.mean(leans)), liftoffs=liftoffs, secs=secs))
        if fell_at is not None:
            break
    env.close()
    return dict(kind=kind, v=v, phase=phase, k=k, delay=delay, seed=seed, fell_at=fell_at, segs=segs)


def selftest(stage):
    """delay 0 must reproduce env.step exactly."""
    import torch
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = _stage(stage)
    finals = []
    for mine in (False, True):
        env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 5000})
        raw = env.unwrapped
        obs, _ = env.reset(seed=73000, options={'command': [1., 0., 0.], 'phase_steps': 0})
        st = Stepper(raw, 0)
        for i in range(200):
            a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            if mine:
                st.step(a)
                obs = raw._observation('push')
            else:
                obs = env.step(a)[0]
        finals.append((raw.data.qpos.copy(), obs.copy()))
        env.close()
    print('selftest qpos diff', float(np.abs(finals[0][0] - finals[1][0]).max()),
          'obs diff', float(np.abs(finals[0][1] - finals[1][1]).max()))


def jobs(stage, which):
    j = []
    S = (73000, 73001)
    if which in ('all', 'phase'):
        for v in (.5, 1.):
            for b in (0, 15):
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    j += [(stage, 'reset', v, b + k, 0, 0, s) for s in S]
                    j += [(stage, 'jump', v, b, k, 0, s) for s in S]
            for p in range(0, 30, 2):
                j.append((stage, 'standwalk', v, p, 0, 0, 73000))
            for J in (1, 2):
                j += [(stage, 'jitter', v, 0, J, 0, s) for s in S]
    if which in ('all', 'delay'):
        for d in (0, 5, 10, 20):
            for v in (.5, 1.):
                for b in (0, 15):
                    j += [(stage, 'reset', v, b, 0, d, s) for s in S]
            j += [(stage, 'stand', 0., 0, 0, d, s) for s in S]
            j += [(stage, 'stopstart', 0., 0, 0, d, s) for s in S]
        for d in (5, 10):
            for v in (.5, 1.):
                for p in range(0, 30, 5):
                    j.append((stage, 'standwalk', v, p + 2, 0, d, 73000))
    return j


if __name__ == '__main__':
    stage, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    workers = int(rest.pop(0)) if rest and rest[0].isdigit() else 10
    which = rest[0] if rest else 'all'
    if which == 'selftest':
        selftest(stage)
        raise SystemExit
    todo = jobs(stage, which)
    with Pool(workers) as pool:
        results = pool.map(run, todo, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1))
    print(len(results), 'runs')
