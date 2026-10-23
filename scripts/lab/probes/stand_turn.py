"""CPU only. stand_turn.py <stage dir | export dir | unitree_v0> <out.json> [workers] [group]

Why does a stage turn standing / drift in the unitree_mujoco + g1_ctrl bridge ("the bridge") while our env shows less?
Copy of the bridge from um_probe.py (physics variants, PD on a 2 ms older state, FixStand on the band) plus what that
probe lacks:
  * lock_wrists applied (the locked wrist entries 0), as env.step does (um_probe.Stepper and the probes built on it
    did not until 22.10: they drove the wrists with the policy's raw output, um_probe.py <stage> selftest: qpos diff
    0.6). This file reproduces env.step exactly (checked);
  * a physics-tick loop: policy step i observes at tick O_i (2 ms) and its target reaches the motors at tick P_i.
    Regular: O = 10 i, P = O + delay. timing 'real': gaps and delays tiled from a bridge run's targets_sim.csv
    (obs = arrived - 1 tick, applied as recorded);
  * the bridge procedure: band 0 -> 0.6 m over 7 s, FixStand, Velocity with the band for 1.4 s, band off, then the
    scenario plan (stand / walk 0.5 / walk 1.0 / side +-0.3 / ...), per-second yaw rate per segment, gait phase at the
    first walk 0.73 as in the logs; optional first-order lag on the command (lag, s);
  * motor-model splits (extra), forced joint targets (force), left-right symmetrised policy (sym), Unitree's shipped
    v0 policy (stage 'unitree_v0': obs 98, same layout, deploy/robots/g1/config/policy/velocity/v0);
  * an export_g1 folder instead of a stage: its policy.onnx acts (humanoid_sim.export_g1.ExportedPolicy: what
    g1_ctrl runs, stand trim included) in the environment of the stage it came from.
Case = dict(physics, stale, delay, timing, which, offset, start, plan, seed, phase, lag, extra, force, sym).
physics: ours | nomotor | um_joint | um | ours_caps | um_caps (um_probe.apply_physics; *_caps: the feet collide as in
Unitree's training model).  start: fixair | reset (our reset, no FixStand/band).
Groups: smoke, stand (160 runs), bridge, motor, friction, delay, start, wrist, sym, unitree.  Prints one line per case.
Candidate 3 (exp-6986f1d9544b4d00 @ 5481431040), 22.10: see the Russian summary in the lab's answer.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import um_probe as um  # noqa: E402

LOGS = Path(r'\\wsl.localhost\Ubuntu\home\mawgly\unitree\sim2sim_logs')
RUNS = {  # bridge runs of candidate 3 (targets_sim.csv): walk 0.5 / walk 1.0 / side / stopstart
    'walk': ['205942', '210022', '210101', '210141', '210221', '210301'],
    'walk1': ['210341', '210421'],
    'side': ['210612-side'],
    'stopstart': ['210706-stopstart'],
}
CAND3 = 'exp-6986f1d9544b4d00-5481431040-20261022-'
TICK = 0.002
STAGE3 = os.path.join(SIM, 'runs', 'humanoid_lab', 'exp-6986f1d9544b4d00', 'stage_005481431040')
BAND_OFF_TICKS = 700  # 1.4 s after the Velocity entry (bridge runs: 1.400-1.402 s)
PLANS = {  # (name, vx, vy, wz, seconds), after band_off
    'stand': [('stand', 0., 0., 0., 5.)],
    'stand30': [('stand', 0., 0., 0., 30.)],
    'walk_now': [('walk', .5, 0., 0., 10.), ('stop', 0., 0., 0., 5.)],
    'walk1_now': [('walk', 1., 0., 0., 10.), ('stop', 0., 0., 0., 5.)],
    'walk': [('stand', 0., 0., 0., 5.), ('walk', .5, 0., 0., 10.), ('stop', 0., 0., 0., 5.)],
    'walk1': [('stand', 0., 0., 0., 5.), ('walk', 1., 0., 0., 10.), ('stop', 0., 0., 0., 5.)],
    'side': [('stand', 0., 0., 0., 5.), ('left', 0., .3, 0., 10.), ('stop', 0., 0., 0., 3.),
             ('right', 0., -.3, 0., 10.), ('stop', 0., 0., 0., 5.)],
    'sideR': [('stand', 0., 0., 0., 5.), ('right', 0., -.3, 0., 10.), ('stop', 0., 0., 0., 5.)],
    # Side steps at the stage-2 goal ±1 m/s (stage_select's side check; segments named apart from 'side''s ±0.3).
    'side1': [('stand', 0., 0., 0., 5.), ('left1', 0., 1., 0., 10.), ('stop', 0., 0., 0., 3.),
              ('right1', 0., -1., 0., 10.), ('stop', 0., 0., 0., 5.)],
    'stopstart': [('stand', 0., 0., 0., 5.)] + [('walk', .5, 0., 0., 3.), ('stop', 0., 0., 0., 3.)] * 3,
}
_cache = {}


def real_timing(plan_key, which):
    """(intervals, delays) in ticks from one bridge run's targets_sim.csv: obs = arrived - 1 tick, applied as recorded."""
    names = RUNS[plan_key]
    run = LOGS / (CAND3 + names[which % len(names)])
    rows = list(csv.reader((run / 'targets_sim.csv').open(encoding='utf-8')))[1:]
    arrived = np.array([int(r[0]) for r in rows], float) / 2.
    applied = np.array([int(r[1]) if r[1] else np.nan for r in rows], float) / 2.
    obs = arrived - 1.
    delay = np.where(np.isnan(applied), 3., applied - obs)
    delay = np.clip(delay, 1., 12.)
    return np.diff(obs), delay


UNITREE_V0 = r'\\wsl.localhost\Ubuntu\home\mawgly\unitree\unitree_rl_mjlab\deploy\robots\g1\config\policy\velocity\v0\exported\policy.onnx'


class _Onnx:
    """Unitree's shipped G1 velocity policy (deploy/robots/g1/.../velocity/v0): obs [1, 98] -> actions [1, 29]."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.name = self.sess.get_inputs()[0].name

    def predict(self, obs, deterministic=True):
        return self.sess.run(None, {self.name: np.asarray(obs, np.float32)[None]})[0][0], None


class _Identity:
    def normalize_obs(self, obs):
        return obs


class _Sym:
    """Left-right symmetrised policy, pi_sym(s) = (pi(s) + M_a pi(M_s s)) / 2 (humanoid_sim.symmetry.Mirror), as both
    `norm` and `model` of run(): normalize_obs returns the observation and its mirror, predict averages the two."""

    def __init__(self, model, norm, mirror):
        self.model, self.norm, self.mirror = model, norm, mirror
        self.oi, self.os = mirror.obs_index.numpy(), mirror.obs_sign.numpy()
        self.ai, self.as_ = mirror.action_index.numpy(), mirror.action_sign.numpy()

    def normalize_obs(self, obs):
        obs = np.asarray(obs, np.float32)
        return np.stack([self.norm.normalize_obs(obs), self.norm.normalize_obs(obs[self.oi] * self.os)])

    def predict(self, x, deterministic=True):
        a = np.asarray(self.model.predict(x, deterministic=True)[0], float)
        return 0.5 * (a[0] + a[1][self.ai] * self.as_), None


def stage_data(stage):
    if str(stage) == 'unitree_v0':
        _, _, config = um._stage(STAGE3)
        config = {**config, 'history': 1, 'critic_extras': False, 'action_limit': 100., 'lock_wrists': False,
                  'arm_perturb': 0., 'random_phase': False}
        return _Onnx(UNITREE_V0), _Identity(), config
    from humanoid_sim.export_g1 import ExportedPolicy, is_export
    if is_export(stage):  # policy and normalizer at once; the config is its stage's
        if stage not in _cache:
            from humanoid_sim.core import stage_task_config
            exported = ExportedPolicy(stage)
            _cache[stage] = (exported, exported, stage_task_config(exported.stage))
        return _cache[stage]
    return um._stage(stage)


def lock(raw, action):
    a = np.clip(np.asarray(action, dtype=float), -raw.action_limit, raw.action_limit)
    if raw.lock_wrists:  # = task.lock_joints (masked_fill 0), as env.step
        a[raw.wrist_mask.numpy()] = 0.
    return a


def heading_of(q):
    w, x, y, z = q[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(case):
    import mujoco
    import torch
    from humanoid_sim.goal_check import sole_geoms
    from humanoid_sim.tasks.catalog import make_task
    stage = case['stage']
    model, norm, config = stage_data(stage)
    seed = case['seed']
    torch.manual_seed(seed)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 1000000,
                                                **um.physics_config(case['physics'])})
    raw = env.unwrapped
    if case.get('sym'):
        from humanoid_sim.symmetry import Mirror
        model = norm = _Sym(model, norm, Mirror(env))
    um.apply_physics(raw, case['physics'])
    extra = case.get('extra')  # split of our motor model: nofric = curve only, nocurve = friction only
    if extra and raw.motors is not None:
        x1, x2, y1, y2, sf, vf = raw._motor_tensors
        if extra == 'nofric':
            raw._motor_tensors = (x1, x2, y1, y2, torch.zeros_like(sf), torch.zeros_like(vf))
        elif extra == 'nocurve':
            raw._motor_tensors = (x1 * 0 + 1e3, x2 * 0 + 2e3, y1 * 0 + 1e4, y2 * 0 + 1e4, sf, vf)
        elif extra == 'staticonly':
            raw._motor_tensors = (x1, x2, y1, y2, sf, torch.zeros_like(vf))
        elif extra.startswith('fric'):  # fric0.5: static and viscous motor friction x 0.5 (curve kept)
            k = float(extra[4:])
            raw._motor_tensors = (x1, x2, y1, y2, sf * k, vf * k)
        else:
            raise ValueError(extra)
    st = um.Stepper(raw, bool(case.get('stale')))
    soles = sole_geoms(raw)
    pelvis, torso = raw.pelvis, raw.model.body('torso_link').id
    plan = PLANS[case['plan']]
    fix = None
    if case['start'] == 'fixair':
        env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': 0})
        h_fix = .002 if case['physics'].startswith('um') else .0005
        um.place_straight(raw, soles)
        raw.data.qpos[2] += .45
        mujoco.mj_forward(raw.model, raw.data)
        schedule = lambda s: 0. if s < 3. else min(.1 * (1 + int((s - 3.) / .5)), .6)
        fix = um.fixstand(raw, st, 7., raw.data.qpos[raw.qadr].copy(), h_fix, band=schedule)
        st.kp, st.kd = raw.kp.copy(), raw.kd.copy()
        raw.steps, raw.phase_steps = 0, 2
        release = BAND_OFF_TICKS
        fix_target = np.clip(um.FIX_QS, st.lo, st.hi)
    else:
        env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': int(case.get('phase', 0))})
        release = 0
        fix_target = None
    raw.previous_action[:] = 0.
    raw.set_command(np.zeros(3))
    obs = raw._observation('fill')
    # the schedule of policy steps in ticks
    seg_ticks = []
    t0 = release
    for name, vx, vy, wz, secs in plan:
        seg_ticks.append((name, (vx, vy, wz), t0, t0 + int(round(secs / TICK))))
        t0 += int(round(secs / TICK))
    total = t0 + 12
    n_steps = total // 10 + 20
    if case.get('timing') == 'real':
        gaps, delays = real_timing(case['plan'] if case['plan'] in RUNS else 'walk', case.get('which', seed))
        k0 = case.get('offset', 0)
        gaps = np.tile(gaps, 4)
        delays = np.tile(delays, 4)
        O = np.concatenate([[0.], np.cumsum(gaps[k0:k0 + n_steps - 1])])
        D = delays[k0:k0 + n_steps]
        O = np.round(O).astype(int)
        P = O + np.round(D).astype(int)
    else:
        O = np.arange(n_steps) * 10
        P = O + int(case.get('delay', 0))
    lag = float(case.get('lag', 0.))
    alpha = 1. - math.exp(-raw.dt / lag) if lag > 0 else 1.
    cmd_f = np.zeros(3)
    yaw = np.zeros(total)
    xy = np.zeros((total, 2))
    zmin = np.zeros(total)
    obs_t, lean, qrec = [], [], []
    pending, target_now = [], fix_target
    i, t, fell_at = 0, 0, None
    first = True
    while t < total and fell_at is None:
        while i < n_steps and O[i] <= t:
            mujoco.mj_forward(raw.model, raw.data)
            if i > 0:
                obs = raw._observation('push')
            seg = next((s for s in seg_ticks if s[2] <= t < s[3]), None)
            want = np.array(seg[1]) if seg else np.zeros(3)
            cmd_f += alpha * (want - cmd_f)
            if np.abs(cmd_f - raw.command).max() > 1e-9 or i == 0:
                raw.set_command(cmd_f)
                obs = raw._observation('replace')
            g = -raw.data.xmat[pelvis].reshape(3, 3)[2, :]
            obs_t.append(t)
            lean.append((math.degrees(math.atan2(g[0], -g[2])), math.degrees(math.atan2(-g[1], -g[2]))))
            qrec.append(raw.data.qpos[raw.qadr].copy())
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            a = lock(raw, action)
            target = raw._target(a)
            for j, val in (case.get('force') or {}).items():  # test: hold a joint's target (e.g. a wrist at +40 deg)
                target[int(j)] = val
            pending.append((max(int(P[i]), t), target))
            raw.steps += 1
            raw.previous_action[:] = a
            i += 1
        while pending and pending[0][0] <= t:
            target_now = pending.pop(0)[1]
        if target_now is None:
            target_now = pending[0][1]
        if release and t == release:
            st.band = None
        st.substep(target_now)
        yaw[t] = heading_of(raw.data.qpos)
        xy[t] = raw.data.qpos[:2]
        zmin[t] = raw.data.qpos[2]
        if t % 25 == 0:
            mujoco.mj_forward(raw.model, raw.data)
            if st.fell():
                fell_at = t * TICK
        t += 1
    env.close()
    n = t
    yaw = np.unwrap(yaw[:n])
    out_segs = []
    for name, cmd, a, b in seg_ticks:
        b = min(b, n - 1)
        settle = 1000 if (b - a) >= 4000 else 500
        if b - a - settle < 50:  # not reached, or the run fell before the measured part (yaw[s] past the end)
            out_segs.append(dict(name=name, cmd=cmd, missing=True))
            continue
        s = a + settle
        secs = (b - s) * TICK
        h = yaw[s:b]
        d = np.diff(xy[s:b + 1], axis=0)
        fwd = float(np.sum(d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h))) / secs
        side = float(np.sum(-d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h))) / secs
        per_s = [round(float((yaw[min(a + 500 * (k + 1), n - 1)] - yaw[a + 500 * k]) / 1.0), 4)
                 for k in range(int((b - a) // 500))]
        # 3-5 s lean and joints (stand segment)
        sel = [j for j, tt in enumerate(obs_t) if a + 1500 <= tt < min(a + 2500, b)]
        out_segs.append(dict(
            name=name, cmd=cmd, yaw_rate=float((yaw[b] - yaw[s]) / secs), fwd=fwd, side=side,
            heading_deg=math.degrees(float(yaw[b] - yaw[a])), disp=float(np.linalg.norm(xy[b] - xy[a])),
            zmin=float(zmin[a:b].min()), per_s=per_s,
            pitch=float(np.mean([lean[j][0] for j in sel])) if sel else None,
            roll=float(np.mean([lean[j][1] for j in sel])) if sel else None,
            q=np.mean([qrec[j] for j in sel], axis=0).round(4).tolist() if sel and name == 'stand' else None,
            dq_deg=[np.degrees(qrec[min(range(len(obs_t)), key=lambda j: abs(obs_t[j] - (a + 500 * k)))][[2, 8, 12]]
                    - qrec[min(range(len(obs_t)), key=lambda j: abs(obs_t[j] - a))][[2, 8, 12]]).round(2).tolist()
                    for k in range(1, 6)] if name == 'stand' else None))
    return dict(case={k: v for k, v in case.items() if k != 'stage'}, fell_at=fell_at, segs=out_segs,
                fix=None if fix is None else {k: fix[k] for k in ('fell_at', 'z', 'tilt', 'pitch', 'max_err', 'worst')},
                pre=[round(float(yaw[min(500 * (k + 1), n - 1)] - yaw[500 * k]), 4) for k in range(int(release // 500))])


def case(**kw):
    base = dict(physics='um', stale=True, delay=3, start='fixair', plan='stand', seed=73000)
    base.update(kw)
    return base


def grid(stage, which):
    S = [73000 + k for k in range(5)]
    j = []
    if which == 'smoke':
        j += [case(physics='um')]
    if which == 'stand':  # start x physics x stale PD x delay (6 ms = 3 substeps); fixair is deterministic
        for start in ('fixair', 'reset'):
            for physics in ('ours', 'nomotor', 'um_joint', 'um'):
                for stale in (False, True):
                    for delay in (0, 3):
                        j += [case(physics=physics, stale=stale, delay=delay, start=start, seed=s) for s in S]
    if which == 'bridge':  # the validated copy: um + stale PD + 6 ms, plus ours, regular timing and 6 real windows
        for physics in ('ours', 'um'):
            for plan in ('stand30', 'walk', 'walk1', 'side'):
                j.append(case(physics=physics, plan=plan))
            for w in range(6):
                j.append(case(physics=physics, plan='walk', timing='real', which=w))
            for w in range(2):
                j.append(case(physics=physics, plan='walk1', timing='real', which=w))
            for off in (0, 11, 23, 37, 53):
                j.append(case(physics=physics, plan='side', timing='real', which=0, offset=off))
            for k in range(3):  # scripts/sim2sim/ours.py starts
                j.append(case(physics=physics, plan='side', start='reset', phase=15 * k, seed=73000 + k))
    if which == 'motor':  # which part of our motor model removes the bridge's right-side drift
        for plan in ('side', 'walk'):
            for physics, extra in (('ours', None), ('ours', 'nofric'), ('ours', 'nocurve'), ('ours', 'staticonly'),
                                   ('nomotor', None), ('um_joint', None), ('um', None)):
                for stale, delay in ((True, 3), (False, 0)):
                    j.append(case(physics=physics, extra=extra, plan=plan, stale=stale, delay=delay))
    if which == 'friction':  # motor friction scale x delay, side step
        for k in (0., .1, .25, .5, 1., 2.):
            for delay in (0, 3, 5):
                j.append(case(physics='ours', extra='fric%g' % k, plan='side', delay=delay))
    if which == 'delay':  # delay sweep, substeps of 2 ms
        for physics in ('um', 'ours'):
            for delay in range(8):
                for plan in ('side', 'walk'):
                    j.append(case(physics=physics, plan=plan, delay=delay))
    if which == 'start':  # walk drift versus start: our reset (phase 0, 15), after 5 s standing, the bridge procedure
        for physics in ('ours', 'um'):
            for plan in ('walk_now', 'walk1_now', 'walk', 'walk1'):
                for phase in (0, 15):
                    j.append(case(physics=physics, plan=plan, start='reset', phase=phase))
                j.append(case(physics=physics, plan=plan))
            for plan in ('walk_now', 'walk1_now'):
                for phase in (0, 15):
                    j.append(case(physics=physics, plan=plan, start='reset', phase=phase, stale=False, delay=0))
    if which == 'wrist':  # hold a wrist pitch target (rad) to see what hand mass does
        for force in (None, {20: .7}, {27: .7}, {20: .7, 27: .7}, {20: -.7}):
            for physics in ('ours', 'um'):
                j.append(case(physics=physics, force=force))
    if which == 'sym':
        for sym in (False, True):
            for physics in ('ours', 'um'):
                for plan in ('stand30', 'walk', 'walk1', 'side'):
                    j.append(case(physics=physics, plan=plan, sym=sym))
                for w in range(3):
                    j.append(case(physics=physics, plan='walk', timing='real', which=w, sym=sym))
                for k in range(3):
                    for plan in ('stand', 'walk_now', 'walk1_now'):
                        j.append(case(physics=physics, plan=plan, start='reset', phase=15 * k, seed=73000 + k,
                                      stale=False, delay=0, sym=sym))
    if which == 'unitree':  # Unitree's shipped policy (stage argument 'unitree_v0')
        for physics in ('ours', 'um'):
            for stale, delay in ((True, 3), (False, 0)):
                for plan in ('stand30', 'walk', 'walk1', 'side'):
                    j.append(case(physics=physics, stale=stale, delay=delay, plan=plan))
            for w in range(3):
                j.append(case(physics=physics, plan='walk', timing='real', which=w))
    for job in j:
        job['stage'] = stage
    return j


def describe(r):
    c = r['case']
    head = (f"{'SYM ' if c.get('sym') else ''}{c['physics']}{'+' + c['extra'] if c.get('extra') else ''} "
            f"stale={int(c.get('stale', 0))} d={c.get('delay', 0)} {c['start']} {c['plan']} "
            f"{c.get('timing', 'reg')}{'' if c.get('which') is None else ' w=%s' % c['which']}"
            f"{'' if c.get('offset') is None else ' off=%s' % c['offset']}"
            f"{'' if not c.get('phase') else ' ph=%s' % c['phase']}{'' if not c.get('force') else ' force=%s' % c['force']}")
    parts = []
    for s in r['segs']:
        if s.get('missing'):
            continue
        if s['name'] == 'stand':
            parts.append('stand/s ' + ' '.join('%+.3f' % x for x in s['per_s'][:5]) + ' lean %.1f' % (s['pitch'] or 0))
        elif s['name'] != 'stop':
            parts.append("%s w%+.3f v(%+.2f,%+.2f) h%+.0f" % (s['name'], s['yaw_rate'], s['fwd'], s['side'], s['heading_deg']))
    return head + ' | ' + ' | '.join(parts) + ('' if r['fell_at'] is None else ' FELL %.1f s' % r['fell_at'])


def main():
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    which = sys.argv[4] if len(sys.argv) > 4 else 'smoke'
    if which.startswith('file:'):
        jobs = json.loads(Path(which[5:]).read_text(encoding='utf-8'))
        for job in jobs:
            job['stage'] = stage
    else:
        jobs = grid(stage, which)
    if workers <= 1:
        results = [run(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(run, jobs, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    for r in results:
        print(describe(r))
    print(len(results), 'runs')


if __name__ == '__main__':
    main()
