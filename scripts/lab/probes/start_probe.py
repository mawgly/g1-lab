"""CPU only. start_probe.py <stage> <out.json> [workers]
Does the drift depend on how a straight walk starts, and on the policy noise?
Each job walks [v, 0, 0] for 10 s after its start and reports the yaw rate after the first 2 s (as goal_check).
kinds:
  reset      goal_check start: reset with the walk command, gait clock at `phase` steps (0 = 'left', 15 = 'right')
  stand      reset standing [0,0,0] for 60 + phase steps (clock at `phase` when told to walk), then walk
  switch     reset walking at v0 with clock `phase`, 5 s, then switch to v (walk -> walk, cycle kept?)
  wsw        reset walking v with clock 0 for 5 s, stand 60 + 15 - (250 % 30) steps, walk v again at clock 15
sigma: 'det' deterministic, 'cur' policy noise as trained, or a cap (0.3): log_std clamped.
hold: heading hold on (training-like): then the yaw_drift EMA e (tau 1.2 s, of omega - c_eff) is the measure.
push: 6-DoF pelvis kicks as the CUDA env (mean every 2 s, REAL_PUSH)."""
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
REAL_PUSH = np.array((.5, .5, .4, .52, .52, .78))
_cache = {}


def _stage(stage):
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, SIM)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(stage))
        _cache[stage] = (model, norm, stage_task_config(Path(stage)), model.policy.log_std.data.clone())
    return _cache[stage]


def run(job):
    stage, kind, speed, phase, seed, sigma, hold, push, v0 = job
    import torch
    import mujoco
    model, norm, config, base = _stage(stage)
    from humanoid_sim.tasks.catalog import make_task
    if sigma in ('det', 'cur'):
        model.policy.log_std.data.copy_(base)
    else:
        model.policy.log_std.data.copy_(base.clamp(max=math.log(float(sigma))))
    det = sigma == 'det'
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': bool(hold), 'max_episode_steps': 2000})
    raw = env.unwrapped
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pelvis = raw.model.body('pelvis').id
    heading = lambda: math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0])
    plan = []  # (steps, command)
    if kind == 'reset':
        obs, _ = env.reset(seed=seed, options={'command': [speed, 0., 0.], 'phase_steps': phase})
        plan = [(500, None)]
    elif kind == 'stand':
        obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.]})
        plan = [(60 + phase, 'pre'), (500, [speed, 0., 0.])]
    elif kind == 'switch':
        obs, _ = env.reset(seed=seed, options={'command': [v0, 0., 0.], 'phase_steps': phase})
        plan = [(250, 'pre'), (500, [speed, 0., 0.])]
    elif kind == 'short':  # phase = k standing steps; the clock starts at 15 - k so the walk begins at clock 15
        obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': (15 - phase) % 30})
        plan = [(phase, 'pre'), (500, [speed, 0., 0.])]
    elif kind == 'wsw':
        obs, _ = env.reset(seed=seed, options={'command': [speed, 0., 0.]})
        plan = [(250, 'pre'), (60 + 15 - 250 % 30, [0., 0., 0.]), (500, [speed, 0., 0.])]
    fell = False
    alpha = raw.dt / 1.2
    out = {}
    for n, command in plan:
        last = command is None or command not in ('pre',) and n == 500
        if command not in (None, 'pre'):
            raw.set_command(command)
            if hold:  # as the CUDA env on a straight command: hold the heading from here
                raw.heading_target = heading()
            obs = raw._observation('replace')
        track, emas, ema = [heading()], [], 0.
        for t in range(n):
            action = model.predict(norm.normalize_obs(obs), deterministic=det)[0]
            c = float(raw.observed_command[2])
            if push and rng.random() < raw.dt / 2.:
                raw.data.qvel[:6] += (2 * rng.random(6) - 1) * REAL_PUSH
            obs, _, term, trunc, _ = env.step(action)
            track.append(heading())
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(raw.model, raw.data, mujoco.mjtObj.mjOBJ_BODY, pelvis, vel, 1)
            ema += alpha * (vel[2] - c - ema)
            if t >= 100:
                emas.append(ema)
            if term:
                fell = True
                break
        if last:
            track = np.unwrap(track)
            first = min(100, len(track) - 2)
            e = np.array(emas) if emas else np.zeros(1)
            out = dict(drift=float((track[-1] - track[first]) / ((len(track) - 1 - first) * raw.dt)),
                       ema_mean=float(e.mean()), ema_abs=float(np.abs(e).mean()))
        if fell:
            break
    env.close()
    return dict(kind=kind, speed=speed, phase=phase, seed=seed, sigma=sigma, hold=hold, push=push, v0=v0,
                fell=fell, **out)


def confirm(stage):
    """reset vs stand start at clock 0 / 15, both speeds, deterministic; plus short stands before walking."""
    S = lambda n: range(73000, 73000 + n)
    j = []
    for v in (.5, 1.):
        for ph in (0, 15):
            j += [(stage, 'reset', v, ph, s, 'det', 0, 0, None) for s in S(2)]
            j += [(stage, 'stand', v, ph, s, 'det', 0, 0, None) for s in S(2)]
    return j


def short(stage):
    """reset standing for k steps, then walk 0.5 with the clock at 15 + k (k < 60: the clock is not aligned)."""
    S = lambda n: range(73000, 73000 + n)
    return [(stage, 'short', .5, k, s, 'det', 0, 0, None) for k in (1, 5, 10, 25) for s in S(2)]


def jobs(stage):
    j = []
    S = lambda n: range(73000, 73000 + n)
    for ph in (0, 4, 8, 11, 15, 19, 23, 26):
        j += [(stage, 'reset', .5, ph, s, 'det', 0, 0, None) for s in S(2)]
    for ph in (0, 15):
        j += [(stage, 'reset', 1., ph, s, 'det', 0, 0, None) for s in S(2)]
    for ph in (0, 8, 15, 23):
        j += [(stage, 'stand', .5, ph, s, 'det', 0, 0, None) for s in S(2)]
    for ph in (0, 15):
        j += [(stage, 'switch', .5, ph, s, 'det', 0, 0, 1.) for s in S(2)]
    j += [(stage, 'wsw', .5, 15, s, 'det', 0, 0, None) for s in S(2)]
    for v in (.5, 1.):
        for ph in (0, 15):
            for sg in ('cur', '0.3'):
                j += [(stage, 'reset', v, ph, s, sg, 0, 0, None) for s in S(4)]
    for ph in (0, 15):
        for sg in ('det', 'cur', '0.3'):
            for push in (0, 1):
                j += [(stage, 'reset', .5, ph, s, sg, 1, push, None) for s in S(3)]
    return j


if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    which = sys.argv[4] if len(sys.argv) > 4 else 'jobs'
    todo = {'jobs': jobs, 'confirm': confirm, 'short': short,
            'both': lambda s: confirm(s) + short(s)}[which](stage)
    with Pool(workers) as pool:
        results = pool.map(run, todo, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1))
    groups = {}
    for r in results:
        groups.setdefault((r['kind'], r['speed'], r['phase'], r['sigma'], r['hold'], r['push'], r['v0']), []).append(r)
    for k, rs in groups.items():
        d = [r.get('drift', float('nan')) for r in rs]
        line = (f"{k[0]:6s} v={k[1]} ph={k[2]:2d} sig={k[3]:4s} hold={k[4]} push={k[5]} v0={k[6]} fell={sum(r['fell'] for r in rs)} "
                f"drift {np.mean(d):+.3f} [{min(d):+.3f},{max(d):+.3f}]")
        if k[4]:
            line += (f" | e mean {np.mean([r['ema_mean'] for r in rs]):+.4f} |e| {np.mean([r['ema_abs'] for r in rs]):.4f}")
        print(line, flush=True)
