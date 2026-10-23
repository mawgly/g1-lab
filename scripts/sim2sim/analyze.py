"""Report for one sim2sim run (unitree_mujoco + g1_ctrl): falls, standing and walking, and latency.

Usage: python analyze.py <run dir>    (run.sh calls it in $ROOT/venv: numpy, onnxruntime, pyyaml)
       python analyze.py --wrist <run dir>...    (only the wrist mode of earlier runs; writes nothing)

Inputs written by run.sh: lowstate.csv / lowcmd.csv / sport.csv (sim2sim_probe), events.csv
(gamepad.py), deploy.yaml + policy.onnx (the exported candidate). Writes report.json and
latency_steps.json (per policy step) and prints the report in Russian. Times are CLOCK_MONOTONIC (probe
and gamepad), poses come from the simulator (sportmodestate position of the pelvis IMU site, lowstate
IMU quaternion).

Scenarios other than `walk` (gamepad.py --scenario) are reported per commanded segment (`seg` events):
after the first SETTLE s (1 s for segments under 8 s) the yaw rate, and the forward / sideways speed of
the path integrated in the pelvis heading frame (so a turning walk counts along the body); over the whole
segment the displacement, heading change and falls. `passive` also gets the fall after LT + B and the way
back (band, FixStand, Velocity again). Latency is measured on the first Velocity entry only.

Wrist mode (`wrist`): candidate 2 holds exactly one wrist pitch at +40 deg (its action limit) for the whole run,
picked on the Velocity entry, and which one goes with the drift. Mean lowcmd target of the left (SDK 20) and
right (27) wrist pitch standing and moving, mode left / right / both / none by WRIST_BENT_DEG, and FixStand's
target of both (g1_ctrl_config.yaml, the config.yaml the run used; run.sh --fixstand-wrist edits it).

Pushes (`pushes`, scenario push; the unitree_mujoco push patch logs each push to sim.log with its simulator
time and force): each push is matched to its gamepad `push` event and measured until the next scenario event:
fall, peak pelvis velocity along the push, when the motion settled (the pelvis velocity in its heading frame and
the yaw rate, averaged over PUSH_AVERAGE s, back within PUSH_SETTLE_SPEED / PUSH_SETTLE_YAW of the second before
the push for PUSH_SETTLE_HOLD s), offset from the pre-push motion and heading.

Velocity entries (`entries`): every switch of lowcmd to the Velocity gains, and what g1_ctrl sent until the first
policy target: the previous target kept (the g1_ctrl patch, its g1_ctrl.log line), or all-zero targets (the race
of unpatched g1_ctrl: State_RLBase::run copies processed_actions before the policy thread's first step).

Timing (`timing`, over the drift window of a walk, else the first Velocity entry after the band): the
real-time factor, policy step intervals in simulator time and how old the state behind the bridge's PD was
(pd_ages). targets_sim.csv lists every new policy target with its simulator time, to replay it elsewhere.

Latency is measured without touching g1_ctrl. For every new lowcmd target (a policy step) the
observation is rebuilt as g1_ctrl builds it (deploy.yaml terms, 5-frame history, gait phase counted
from the Velocity reset, command from the remote; a height export's height_command from the right stick as
patches/g1_ctrl-height.patch builds it) from each lowstate received in the 15 ms before,
policy.onnx is replayed, and the lowstate that reproduces the target is the one the policy saw.
  obs -> cmd: lowcmd arrival minus the first arrival of that lowstate (wall clock);
  sim delay: simulator ticks from that state to the first physics step whose motor torque
             (lowstate tau_est) follows the new target, i.e. the action delay in simulation time.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

J = 29
FALL_Z = 0.5  # m, pelvis height below which the robot counts as fallen
FALL_TILT = 1.0  # rad, g1_ctrl's bad_orientation limit (the check itself is disabled in g1_ctrl)
SETTLE = 2.0  # s, goal_check measures drift after the first 2 s of a walk
WINDOW_NS = 15_000_000  # lowstates considered before a new target / after it for the torque
PHYSICS_MS = 2  # unitree_mujoco G1 timestep: simulator ticks (ms) per physics step
PD_FIT = 0.05  # N m, median torque misfit under which a PD (target, state) explains tau_est
WRIST_PITCH = (20, 27)  # SDK indices of the left / right wrist pitch
WRIST_BENT_DEG = 20.0  # |mean target| past which a wrist counts as bent (candidate 2's latch: +40.1 deg)
WRIST_MODES = {(True, False): 'left', (False, True): 'right', (True, True): 'both', (False, False): 'none'}
PUSH_LINE = re.compile(r'sim2sim push (start|end) (.*)$')  # unitree_mujoco push patch, sim.log
PUSH_MATCH = (-0.2e9, 0.5e9)  # ns, a logged push belongs to the gamepad `push` event this close before / after it
PUSH_BEFORE = 1.0  # s before a push whose motion is the reference (velocity, yaw rate)
PUSH_AVERAGE = 0.6  # s, the settle check averages motion over a gait period starting at each moment
PUSH_SETTLE_SPEED = 0.1  # m/s, |averaged pelvis velocity - reference| (pelvis heading frame) to count as settled
PUSH_SETTLE_YAW = 0.1  # rad/s, |averaged yaw rate - reference| to count as settled
PUSH_SETTLE_HOLD = 1.0  # s both have to stay within from the settle moment on
PUSH_GRID = 20e6  # ns between the moments the settle check looks at
PUSH_DIRECTIONS = dict(forward='вперёд', back='назад', left='влево', right='вправо')
HOLD_LINE = re.compile(r'kept the previous target for (\d+) FSM ticks')  # g1_ctrl patch, g1_ctrl.log

# lowstate.csv columns (sim2sim_probe.cpp)
LS_T, LS_TICK = 0, 1
LS_Q, LS_DQ, LS_TAU = slice(2, 31), slice(31, 60), slice(60, 89)
LS_QUAT, LS_GYRO = slice(89, 93), slice(93, 96)
LS_LX, LS_RX, LS_RY, LS_LY = 97, 98, 99, 101
# lowcmd.csv columns
LC_T, LC_Q, LC_KP0 = 0, slice(1, 30), 30


def table(run, name):
    return np.loadtxt(run / name, delimiter=',', skiprows=1, ndmin=2)


def scenario_events(run):
    """events.csv (gamepad.py): the first (time, value) of each event, and the whole timeline."""
    events, timeline = {}, []
    for line in (run / 'events.csv').read_text(encoding='utf-8').splitlines()[1:]:
        t, name, value = line.split(',')
        events.setdefault(name, (int(t), value))
        timeline.append((int(t), name, value))
    return events, timeline


def load(run):
    events, timeline = scenario_events(run)
    deploy = yaml.safe_load((run / 'deploy.yaml').read_text(encoding='utf-8'))
    return (table(run, 'lowstate.csv'), table(run, 'lowcmd.csv'), table(run, 'sport.csv'), events, timeline,
            deploy)


def yaw_of(quat):
    w, x, y, z = quat.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def gravity_of(quat):
    """Gravity in the pelvis frame: R(q)^T (0, 0, -1), as g1_ctrl's projected_gravity_b."""
    w, x, y, z = quat.T
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)


def at(times, values, t):
    """Last sample at or before t."""
    return values[max(np.searchsorted(times, t, side='right') - 1, 0)]


def fsm_timeline(lc, deploy, t0):
    """FSM state per lowcmd from motor 0's kp: Passive 0, FixStand 100 (config.yaml), Velocity deploy.yaml."""
    names = {0.0: 'Passive', 100.0: 'FixStand', round(float(deploy['stiffness'][0]), 3): 'Velocity'}
    states, last = [], None
    for t, kp in zip(lc[:, LC_T], lc[:, LC_KP0]):
        name = names.get(round(float(kp), 3), f'kp {kp:g}')
        if name != last:
            states.append(((t - t0) / 1e9, name))
            last = name
    return states


def pose_metrics(sport, ls, t_from, t_to):
    """Pelvis displacement and heading between two times (wall clock ns)."""
    ts, tl = sport[:, 0], ls[:, LS_T]
    yaw = yaw_of(ls[:, LS_QUAT])
    p0, p1 = at(ts, sport[:, 1:3], t_from), at(ts, sport[:, 1:3], t_to)
    y0, y1 = at(tl, yaw, t_from), at(tl, yaw, t_to)
    d = p1 - p0
    forward = float(d[0] * math.cos(y0) + d[1] * math.sin(y0))
    lateral = float(-d[0] * math.sin(y0) + d[1] * math.cos(y0))
    inside = (ts >= t_from) & (ts <= t_to)
    path = 0.0
    if inside.sum() > 1:
        path = float(np.linalg.norm(np.diff(sport[inside, 1:3], axis=0), axis=1).sum())
    seconds = (t_to - t_from) / 1e9
    return dict(seconds=seconds, forward_m=forward, lateral_m=lateral, distance_m=float(np.linalg.norm(d)),
                path_m=path, heading_deg=float(math.degrees(y1 - y0)), yaw_rate=float((y1 - y0) / seconds))


def falls(sport, ls, t_from, t_to):
    """First time after t_from the pelvis drops below FALL_Z or tilts past FALL_TILT; None if never."""
    ts, tl = sport[:, 0], ls[:, LS_T]
    low = np.where((ts >= t_from) & (ts <= t_to) & (sport[:, 3] < FALL_Z))[0]
    tipped = np.where((tl >= t_from) & (tl <= t_to) & (tilt_of(ls) > FALL_TILT))[0]
    times = ([ts[low[0]]] if len(low) else []) + ([tl[tipped[0]]] if len(tipped) else [])
    return min(times) if times else None


def tilt_of(ls):
    """Pelvis tilt from vertical, rad, per lowstate."""
    return np.arccos(np.clip(-gravity_of(ls[:, LS_QUAT])[:, 2], -1, 1))


def motion(sport, ls, t_from, t_to):
    """Pelvis motion between two times: displacement, heading change, mean yaw rate, and the path integrated
    in the pelvis heading frame (forward / sideways speed; a turning walk counts along the body)."""
    ts = sport[:, 0]
    inside = (ts >= t_from) & (ts <= t_to)
    p = sport[inside, 1:3]
    yaw = np.interp(ts[inside], ls[:, LS_T], yaw_of(ls[:, LS_QUAT]))
    d = np.diff(p, axis=0)
    h = yaw[:-1]
    seconds = (t_to - t_from) / 1e9
    forward = float(np.sum(d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)))
    sideways = float(np.sum(-d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)))
    turn = float(yaw[-1] - yaw[0])
    return dict(seconds=seconds, forward_speed=forward / seconds, sideways_speed=sideways / seconds,
                yaw_rate=turn / seconds, heading_deg=math.degrees(turn),
                distance_m=float(np.linalg.norm(p[-1] - p[0])), path_m=float(np.linalg.norm(d, axis=1).sum()),
                pelvis_z_min=float(sport[inside, 3].min()), pelvis_z_mean=float(sport[inside, 3].mean()))


def scenario_segments(timeline):
    """gamepad.py `seg` events: name, command (vx, vy, wz), start and end ns (the next event)."""
    out = []
    for k, (t, name, value) in enumerate(timeline):
        if name != 'seg' or k + 1 >= len(timeline):
            continue
        label, *command = value.split(':')
        out.append(dict(name=label, command=[float(c) for c in command], start=t, end=timeline[k + 1][0]))
    return out


def segment_report(sport, ls, seg, t0):
    seconds = (seg['end'] - seg['start']) / 1e9
    settle = SETTLE if seconds >= 8 else 1.0
    fell = falls(sport, ls, seg['start'], seg['end'])
    return dict(name=seg['name'], command=seg['command'], start_s=round((seg['start'] - t0) / 1e9, 2),
                seconds=seconds, settle_s=settle, fell_s=None if fell is None else round((fell - t0) / 1e9, 2),
                whole=motion(sport, ls, seg['start'], seg['end']),
                settled=motion(sport, ls, seg['start'] + settle * 1e9, seg['end']))


def passive_report(sport, ls, fsm, timeline, t0):
    """LT + B from Velocity: when g1_ctrl went Passive, how the robot fell, and the way back to Velocity."""
    marks = {name: t for t, name, _ in reversed(timeline)}  # first occurrence of each event
    t_cmd = marks['passive']
    entry = next((t for t, name in fsm if name == 'Passive' and t >= (t_cmd - t0) / 1e9 - 0.5), None)
    out = dict(command_s=round((t_cmd - t0) / 1e9, 2), entered_s=None if entry is None else float(entry),
               states_after=[(round(float(t), 2), name) for t, name in fsm if entry is not None and t >= entry])
    if entry is None:
        return out
    t_entry = t0 + entry * 1e9
    t_band = marks.get('band_on', timeline[-1][0])
    ts, tl = sport[:, 0], ls[:, LS_T]
    window = (ts >= t_entry) & (ts <= t_band)
    fell = falls(sport, ls, t_entry, t_band)
    tilt = tilt_of(ls)[(tl >= t_entry) & (tl <= t_band)]
    out.update(z_at_entry=float(at(ts, sport[:, 3], t_entry)), z_min=float(sport[window, 3].min()),
               z_before_band=float(at(ts, sport[:, 3], t_band)), tilt_max_deg=float(np.degrees(tilt.max())),
               fall_after_s=None if fell is None else round((fell - t_entry) / 1e9, 2),
               moved_m=float(np.linalg.norm(at(ts, sport[:, 1:3], t_band) - at(ts, sport[:, 1:3], t_entry))))
    if 'fixstand_again' in marks:
        out['z_hanging'] = float(at(ts, sport[:, 3], marks['fixstand_again']))
    return out


def stand_by_second(ls, t_from, t_to):
    """Yaw rate (rad/s) over each whole second from t_from (the band off) to t_to, from the pelvis IMU heading."""
    edges = t_from + np.arange(int((t_to - t_from) // 1e9) + 1) * 1e9
    return [float(v) for v in np.diff(np.interp(edges, ls[:, LS_T], yaw_of(ls[:, LS_QUAT])))]


def sim_pushes(run):
    """Pushes the patched unitree_mujoco logged to sim.log, in order: start and end fields (numbers as float)."""
    path = run / 'sim.log'
    pushes = []
    if not path.exists():
        return pushes
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = PUSH_LINE.search(line)
        if not m:
            continue
        fields = {}
        for item in m.group(2).split():
            key, _, value = item.partition('=')
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
        if m.group(1) == 'start':
            pushes.append(dict(start=fields, end=None))
        elif pushes:
            pushes[-1]['end'] = fields
    return pushes


def wall_of_sim(ls, t_sim):
    """Arrival (ns) of the first lowstate at or after simulator time t_sim (s); None if there is none."""
    i = np.searchsorted(ls[:, LS_TICK], round(t_sim * 1e3), side='left')
    return None if i >= len(ls) else float(ls[i, LS_T])


def push_report(sport, ls, t_push, seconds, direction, t_end):
    """Pelvis motion after a push that started at t_push (wall ns), lasted `seconds` and pushed along `direction`
    (world unit vector), until t_end (the next scenario event): fall, peak velocity change along the push (the
    simulator's pelvis IMU velocity in the first second, minus its mean over the second before), settle time,
    offset from the motion before and heading.

    Reference: the second before the push (velocity from the path, rotated into the pelvis heading; yaw rate).
    Settled: the first moment after the push from which, for PUSH_SETTLE_HOLD s, the motion averaged over the
    next PUSH_AVERAGE s stays within PUSH_SETTLE_SPEED / PUSH_SETTLE_YAW of the reference (None: not before
    t_end leaves room for the check, or fell). Offset: path minus the reference velocity carried on, in the
    heading before the push (forward, left)."""
    ts, tl = sport[:, 0], ls[:, LS_T]
    yaw_all = yaw_of(ls[:, LS_QUAT])

    def pos(t):
        return np.stack([np.interp(t, ts, sport[:, 1]), np.interp(t, ts, sport[:, 2])], axis=-1)

    def yaw(t):
        return np.interp(t, tl, yaw_all)

    span = PUSH_AVERAGE * 1e9

    def averaged(t):
        """(forward, left) velocity in the heading at t and yaw rate, averaged over [t, t + PUSH_AVERAGE]."""
        v, h = (pos(t + span) - pos(t)) / PUSH_AVERAGE, yaw(t)
        forward = v[:, 0] * np.cos(h) + v[:, 1] * np.sin(h)
        left = -v[:, 0] * np.sin(h) + v[:, 1] * np.cos(h)
        return np.stack([forward, left], axis=-1), (yaw(t + span) - yaw(t)) / PUSH_AVERAGE

    before = np.arange(t_push - PUSH_BEFORE * 1e9, t_push - span + 1, PUSH_GRID)
    v_ref, w_ref = averaged(before)
    v_ref, w_ref = v_ref.mean(axis=0), float(w_ref.mean())
    p0, h0 = pos(np.array([t_push]))[0], float(yaw(t_push))
    v_world = (p0 - pos(np.array([t_push - PUSH_BEFORE * 1e9]))[0]) / PUSH_BEFORE

    def offset(t):
        x, y = pos(np.array([t]))[0] - p0 - v_world * (t - t_push) / 1e9
        return [float(x * math.cos(h0) + y * math.sin(h0)), float(-x * math.sin(h0) + y * math.cos(h0))]

    fell = falls(sport, ls, t_push, t_end)
    settle = None
    grid = np.arange(t_push + seconds * 1e9, t_end - span + 1, PUSH_GRID)
    hold = int(round(PUSH_SETTLE_HOLD * 1e9 / PUSH_GRID))
    if fell is None and len(grid) > hold:
        v, w = averaged(grid)
        calm = (np.linalg.norm(v - v_ref, axis=1) < PUSH_SETTLE_SPEED) & (np.abs(w - w_ref) < PUSH_SETTLE_YAW)
        first = next((k for k in range(len(grid) - hold) if calm[k:k + hold + 1].all()), None)
        settle = None if first is None else float((grid[first] - t_push) / 1e9)
    window = (ts >= t_push) & (ts <= t_end)
    tilt = tilt_of(ls)[(tl >= t_push) & (tl <= t_end)]
    pre = (ts >= t_push - PUSH_BEFORE * 1e9) & (ts < t_push)
    after = (ts >= t_push) & (ts <= min(t_push + 1e9, t_end))
    along = (sport[after, 4:6] - sport[pre, 4:6].mean(axis=0)) @ direction
    out = dict(window_s=(t_end - t_push) / 1e9,
               fell_after_s=None if fell is None else round((fell - t_push) / 1e9, 2),
               peak_along_ms=float(along.max()), peak_after_s=float((ts[after][along.argmax()] - t_push) / 1e9),
               settle_s=settle, reference=dict(forward=float(v_ref[0]), left=float(v_ref[1]), yaw_rate=w_ref),
               offset_end_m=offset(t_end), heading_end_deg=math.degrees(float(yaw(t_end)) - h0),
               pelvis_z_min=float(sport[window, 3].min()), tilt_max_deg=float(np.degrees(tilt.max())))
    out['heading_trend_end_deg'] = out['heading_end_deg'] - math.degrees(w_ref * (t_end - t_push) / 1e9)
    if settle is not None:
        t_settle = t_push + settle * 1e9
        out.update(offset_settle_m=offset(t_settle), heading_settle_deg=math.degrees(float(yaw(t_settle)) - h0))
    return out


def pushes_report(run, sport, ls, timeline, t0, t_end):
    """Every gamepad `push` event with the sim.log push that belongs to it (PUSH_MATCH), measured until the next
    scenario event. Returns the list and the number of logged pushes no event claimed."""
    logged = sim_pushes(run)
    starts = [wall_of_sim(ls, p['start']['t']) for p in logged]
    out, used = [], set()
    for k, (t, name, value) in enumerate(timeline):
        if name != 'push':
            continue
        direction, dv = value.split(':')
        t_next = timeline[k + 1][0] if k + 1 < len(timeline) else t_end
        item = dict(direction=direction, dv_command=float(dv), key_s=round((t - t0) / 1e9, 2))
        match = next((j for j, s in enumerate(starts)
                      if j not in used and s is not None and PUSH_MATCH[0] <= s - t <= PUSH_MATCH[1]), None)
        if match is None:
            item['missing'] = True  # the key did not reach the simulator (or it is not patched)
            out.append(item)
            continue
        used.add(match)
        start, end = logged[match]['start'], logged[match]['end']
        seconds = start['steps'] * start['dt']
        item.update(start_s=round((starts[match] - t0) / 1e9, 2), sim_t=start['t'], dir=start['dir'],
                    dv=start['dv'], force_n=start['force'], heading_at_push_deg=start['heading_deg'],
                    seconds=seconds, impulse_ns=None if end is None else end['impulse'])
        z = at(sport[:, 0], sport[:, 3], starts[match])
        tilt = at(ls[:, LS_T], tilt_of(ls), starts[match])
        if z < FALL_Z or tilt > FALL_TILT:  # pushed after a fall: nothing to measure
            item['lying'] = True
        else:
            direction = np.array([start['fx'], start['fy']]) / max(start['force'], 1e-9)
            item.update(push_report(sport, ls, starts[match], seconds, direction, t_next))
        out.append(item)
    return out, len(logged) - len(used)


def velocity_entries(lc, deploy, t0):
    """Every switch of lowcmd to the Velocity gains, and what g1_ctrl sent until the first policy target:
    messages that kept the target of the message before the entry (FixStand's pose: the g1_ctrl patch) and
    all-zero targets (the race of unpatched g1_ctrl), the largest target jump of the first Velocity message,
    and joints at exactly 0 in the first policy message that are not 0 5 ms later (a target copied while the
    policy thread was writing it)."""
    velocity_kp = round(float(deploy['stiffness'][0]), 3)
    on = np.round(lc[:, LC_KP0], 3) == velocity_kp
    out = []
    for s in np.where(on & ~np.r_[False, on[:-1]])[0]:
        e = s
        while e < len(lc) and on[e]:
            e += 1
        q, t = lc[s:e, LC_Q], lc[s:e, LC_T]
        prev = lc[s - 1, LC_Q] if s > 0 else np.full(J, np.nan)
        held, zero = np.all(q == prev, axis=1), np.all(q == 0, axis=1)
        new = np.where(~held & ~zero)[0]
        f = int(new[0]) if len(new) else len(q)
        item = dict(t_s=round((t[0] - t0) / 1e9, 2), messages=int(len(q)), before_policy=f,
                    held=int(held[:f].sum()), zero=int(zero[:f].sum()), zero_all=int(zero.sum()),
                    first_jump_rad=float(np.max(np.abs(q[0] - prev))) if s > 0 else None,
                    policy_after_ms=None, torn_zero_joints=None)
        if f < len(q):
            ref = q[min(np.searchsorted(t, t[f] + 5e6), len(q) - 1)]
            item.update(policy_after_ms=float((t[f] - t[0]) / 1e6),
                        torn_zero_joints=int(np.sum((q[f] == 0) & (ref != 0))))
        out.append(item)
    return out


def entry_holds(run):
    """FSM ticks the patched g1_ctrl kept the previous target for at each Velocity entry (g1_ctrl.log); [] if none
    logged (g1_ctrl without the patch)."""
    path = run / 'g1_ctrl.log'
    if not path.exists():
        return []
    return [int(m.group(1)) for m in HOLD_LINE.finditer(path.read_text(encoding='utf-8', errors='replace'))]


def first_velocity(lc, deploy, fsm, t0):
    """The first Velocity entry (lowcmd ns) and what g1_ctrl left it for: (t_enter, left, t_leave); None if never.
    It lasts until g1_ctrl leaves it (passive scenario) or the recording ends (t_leave inf)."""
    velocity_kp = round(float(deploy['stiffness'][0]), 3)
    entered = np.where(np.round(lc[:, LC_KP0], 3) == velocity_kp)[0]
    if len(entered) == 0:
        return None
    t_enter = lc[entered[0], LC_T]
    left = [s for s in fsm if s[0] > (t_enter - t0) / 1e9 and s[1] != 'Velocity']
    return t_enter, left, (t0 + left[0][0] * 1e9 if left else math.inf)


def wrist_windows(events, timeline, t_leave):
    """Where the wrist mode is measured, lists of (from, to) ns within the first Velocity entry. The walk
    scenario: stand from 1 s after the band is off, walk after the first SETTLE s. Other scenarios: the settled
    part of each segment (as segment_report), standing (zero command) or moving."""
    if 'walk' in events:
        walk_t, stop_t = events['walk'][0], events['stop'][0]
        windows = dict(stand=[(events['band_off'][0] + 1e9, walk_t)], moving=[(walk_t + SETTLE * 1e9, stop_t)])
    else:
        windows = dict(stand=[], moving=[])
        for s in scenario_segments(timeline):
            settle = SETTLE if (s['end'] - s['start']) / 1e9 >= 8 else 1.0
            windows['moving' if any(s['command']) else 'stand'].append((s['start'] + settle * 1e9, s['end']))
    return {name: [(a, min(b, t_leave)) for a, b in spans if min(b, t_leave) > a] for name, spans in windows.items()}


def fixstand_wrists(run):
    """FixStand's target of the left / right wrist pitch, deg, from the config.yaml the run used (run.sh saves it
    as g1_ctrl_config.yaml; None for runs before that)."""
    path = run / 'g1_ctrl_config.yaml'
    if not path.exists():
        return None
    pose = yaml.safe_load(path.read_text(encoding='utf-8'))['FSM']['FixStand']['qs'][-1]
    return [math.degrees(float(pose[j])) for j in WRIST_PITCH]


def wrist_mode(lc, windows, fixstand):
    """Which wrist pitch the policy holds bent: mean lowcmd target of the left / right one (deg) per window, mode
    left / right / both / none by |mean| > WRIST_BENT_DEG; overall the windows' mode, `mixed` if they differ."""
    out = dict(threshold_deg=WRIST_BENT_DEG, fixstand_deg=fixstand, windows={})
    t = lc[:, LC_T]
    for name, spans in windows.items():
        inside = np.zeros(len(lc), bool)
        for a, b in spans:
            inside |= (t >= a) & (t < b)
        if not inside.any():
            continue
        left, right = (float(np.degrees(lc[inside, LC_Q.start + j].mean())) for j in WRIST_PITCH)
        out['windows'][name] = dict(left_deg=left, right_deg=right, seconds=sum((b - a) / 1e9 for a, b in spans),
                                    mode=WRIST_MODES[abs(left) > WRIST_BENT_DEG, abs(right) > WRIST_BENT_DEG])
    modes = {w['mode'] for w in out['windows'].values()}
    out['mode'] = (modes.pop() if len(modes) == 1 else 'mixed') if modes else None
    return out


class Replay:
    """g1_ctrl's observation (deploy.yaml) and action processing around policy.onnx, in float32."""

    def __init__(self, run, deploy):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(run / 'policy.onnx'),
                                                    providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        terms = deploy['observations']
        self.order = list(terms)
        self.history = {name: int(terms[name].get('history_length', 1)) for name in self.order}
        action = deploy['actions']['JointPositionAction']
        self.scale = np.float32(action['scale'])
        self.offset = np.float32(action['offset'])
        clip = np.float32(action['clip']) if action.get('clip') else None
        self.lo, self.hi = (clip[:, 0], clip[:, 1]) if clip is not None else (None, None)
        self.default = np.float32(deploy['default_joint_pos'])
        ranges = deploy['commands']['base_velocity']['ranges']
        self.ranges = [ranges['lin_vel_x'], ranges['lin_vel_y'], ranges['ang_vel_z']]
        self.delta = np.float32(np.float32(deploy['step_dt']) * (np.float32(1.0) / np.float32(
            terms['gait_phase']['params']['period'])))
        for name in self.order:
            if terms[name].get('clip') is not None or any(s != 1 for s in terms[name]['scale']):
                raise SystemExit(f'analyze.py: observation {name} has clip/scale, not replayed')
        # Terms of patches/g1_ctrl-height.patch (a height export): the height command from the right stick, the
        # upper-body slot's zeros.
        self.height = terms['height_command'].get('params') if 'height_command' in terms else None
        self.slot = len(terms['upper_body_command']['scale']) if 'upper_body_command' in terms else None

    def command(self, lx, ly, rx):
        """velocity_commands: (ly, -lx, -rx) clamped to the deploy ranges."""
        raw = [ly, -lx, -rx]
        return np.float32([min(max(v, lo), hi) for v, (lo, hi) in zip(raw, self.ranges)])

    def height_term(self, command, ry):
        """height_command as the patch builds it: c = nominal + (nominal - lowest) min(0, ry), at least
        walking_lowest while the command's norm is at least walking_norm, within [lowest, nominal]; c - nominal."""
        p = {key: np.float32(value) for key, value in self.height.items()}
        height = np.float32(p['nominal'] + (p['nominal'] - p['lowest']) * np.float32(min(0.0, float(ry))))
        if float(np.sqrt(np.sum(command.astype(np.float32) ** 2))) >= float(p['walking_norm']):
            height = max(height, p['walking_lowest'])
        height = min(max(height, p['lowest']), p['nominal'])
        return np.float32([height - p['nominal']])

    def frame(self, row, command, phase, last_action, ry=0.):
        quat = np.float32(row[LS_QUAT])
        gait = np.float32([math.sin(float(np.float32(phase * 2)) * math.pi),
                           math.cos(float(np.float32(phase * 2)) * math.pi)])
        if float(np.sqrt(np.sum(command.astype(np.float32) ** 2))) < 0.1:
            gait = np.zeros(2, np.float32)
        terms = dict(base_ang_vel=np.float32(row[LS_GYRO]),
                     projected_gravity=np.float32(gravity_of(quat[None])[0]),
                     velocity_commands=command, gait_phase=gait,
                     joint_pos_rel=np.float32(row[LS_Q]) - self.default, joint_vel_rel=np.float32(row[LS_DQ]),
                     last_action=np.float32(last_action))
        if self.height is not None:
            terms['height_command'] = self.height_term(command, ry)
        if self.slot is not None:
            terms['upper_body_command'] = np.zeros(self.slot, np.float32)
        return terms

    def act(self, frames):
        """frames: oldest to newest; each term's own history buffer, terms in deploy.yaml order."""
        parts = []
        for name in self.order:
            n = self.history[name]
            parts += [f[name] for f in frames[-n:]]
        obs = np.concatenate(parts).astype(np.float32)[None]
        raw = self.session.run(None, {self.input: obs})[0][0]
        target = raw * self.scale + self.offset
        if self.lo is not None:
            target = np.clip(target, self.lo, self.hi)
        return raw, target


def smooth_remote(ls, columns=(LS_LX, LS_LY, LS_RX)):
    """g1_ctrl's joystick axes (lx, ly, rx; columns=(LS_RY,): the right stick's vertical, the height command of
    g1_ctrl-height.patch): unitree_joystick.hpp Axis smoothing applied again to the sim's values."""
    out = np.zeros((len(ls), len(columns)), np.float32)
    state = [np.float32(0)] * len(columns)
    smooth = np.float32(0.03)
    for i, row in enumerate(ls):
        for k, col in enumerate(columns):
            v = np.float32(row[col])
            dz = 0.0 if abs(v) < np.float32(0.01) else float(v)
            state[k] = np.float32(float(state[k]) * (1.0 - float(smooth)) + dz * float(smooth))
            out[i, k] = state[k]
    return out


def policy_steps(lc, t_enter, t_leave):
    """New targets between the Velocity entry and exit: (first arrival ns, final target, zero-target messages).
    The first Velocity message is a new target too when it differs from the message before the entry: the
    policy's first target can come in it (no message with FixStand's pose in between); not counting it shifted
    every step number by one and half the steps went unmatched (22.10: 210221, 175918-side)."""
    rows = np.where((lc[:, LC_T] >= t_enter) & (lc[:, LC_T] < t_leave))[0]
    q = lc[rows][:, LC_Q]
    zeros = int(np.sum(np.all(q == 0, axis=1)))
    changed = np.where(np.any(np.diff(q, axis=0) != 0, axis=1))[0] + 1
    if len(rows) and rows[0] > 0 and np.any(q[0] != lc[rows[0] - 1, LC_Q]):
        changed = np.r_[0, changed]
    steps = []
    for i in changed:
        if np.all(q[i] == 0):
            continue
        t = lc[rows[i], LC_T]
        if steps and t - steps[-1][0] < 5e6:  # a target written while the FSM thread copied it
            steps[-1][1] = i
            continue
        steps.append([t, i])
    out = []
    for k, (t, i) in enumerate(steps):
        end = steps[k + 1][1] - 1 if k + 1 < len(steps) else len(q) - 1
        out.append((t, q[end]))
    return out, zeros


def latency(run, ls, lc, deploy, t_enter, t_leave, segments):
    """segments: (label, start ns, end ns) to tag each policy step with."""
    replay = Replay(run, deploy)
    remote = smooth_remote(ls)
    ry = smooth_remote(ls, (LS_RY,)) if replay.height is not None else None
    t_ls, ticks = ls[:, LS_T], ls[:, LS_TICK]
    # first arrival of each simulator state (the bridge publishes at 1 kHz, physics steps at 500 Hz)
    first = np.ones(len(ls), bool)
    first[1:] = ticks[1:] != ticks[:-1]
    first_rows = np.where(first)[0]
    steps, zero_msgs = policy_steps(lc, t_enter, t_leave)
    kp = np.float32(deploy['stiffness'])
    kd = np.float32(deploy['damping'])

    def candidates(t_c):
        lo, hi = np.searchsorted(t_ls[first_rows], [t_c - WINDOW_NS, t_c], side='right')
        return first_rows[lo:hi]

    def cmd(r):
        lx, ly, rx = remote[r]
        return replay.command(float(lx), float(ly), float(rx))

    def frame_at(r, phase, last):  # g1_ctrl's frame from lowstate r (the height command from its right stick)
        return replay.frame(ls[r], cmd(r), phase, last, 0. if ry is None else float(ry[r, 0]))

    # Step number on the policy thread's fixed grid (sleep_until += step_dt; a late step is followed by
    # quick catch-up steps, a step that ends within one FSM tick of the next is never published). The
    # grid is anchored on recent steps (low percentile of arrival minus grid time: the least delayed
    # ones); where the gait phase is visible (command norm >= 0.1) the step number is also searched
    # +-2 and the exact match wins. Gait phase: one gait_phase call at reset, one per step.
    period_ns = float(deploy['step_dt']) * 1e9
    span = int((steps[-1][0] - steps[0][0]) / period_ns) + 50 if steps else 0
    phases = [replay.delta]
    for _ in range(span):
        phases.append(np.float32(np.fmod(np.float32(phases[-1] + replay.delta), np.float32(1))))
    offsets, last_index, shift = [], 0, 0

    def nearest(table, i):
        """Known entry for step i, else the closest earlier one (an approximation: that step won't match)."""
        while i > 0 and i not in table:
            i -= 1
        return table.get(i)

    frames, raws, targets = {}, {0: np.zeros(J, np.float32)}, {}  # by step number; frames[0]: reset frame
    rows_out = []
    for k, (t_c, target) in enumerate(steps):
        if k == 0:
            i = 1
        else:
            ref = np.percentile(offsets[-25:], 10)
            i = max(int(np.floor((t_c - ref + 2.5e6) / period_ns)) + 1 + shift, last_index + 1)
        cands = candidates(t_c)
        if len(cands) == 0:
            continue
        best = None
        if i == 1:  # env->reset(): frame F0 (history = copies, last action 0); then step 1 with F1
            for a in cands[-4:]:
                for b in cands[-4:]:
                    if b < a:
                        continue
                    f0 = frame_at(a, phases[0], raws[0])
                    hist = [f0] * 4 + [frame_at(b, phases[1], raws[0])]
                    raw, pred = replay.act(hist)
                    err = float(np.max(np.abs(pred - target)))
                    if best is None or err < best[0]:
                        best = (err, b, raw, hist[-1], f0, 0)
        else:
            moving = float(np.linalg.norm(cmd(cands[-1]))) >= 0.1
            for d in ((0, -1, 1, -2, 2) if moving else (0,)):
                j = i + d
                if j <= last_index or j >= len(phases):
                    continue
                history = [nearest(frames, max(n, 0)) for n in range(j - 4, j)]
                last = nearest(raws, j - 1)
                for r in cands:
                    f = frame_at(r, phases[j], last)
                    raw, pred = replay.act(history + [f])
                    err = float(np.max(np.abs(pred - target)))
                    if best is None or err < best[0] - 1e-9:
                        best = (err, r, raw, f, None, d)
                if best is not None and best[0] < 1e-5:
                    break
        err, r, raw, frame, reset_frame, d = best
        if err < 1e-4:
            shift += d
            i += d
        last_index = i
        offsets.append(t_c - (i - 1 - shift) * period_ns)
        frames[i] = frame
        if reset_frame is not None:
            frames[0] = reset_frame
        # last_action of the next step: the replayed output if it reproduced the target, else the target
        # mapped back through scale/offset (exact up to rounding where the joint-range clip did not act)
        recovered = (target.astype(np.float32) - replay.offset) / replay.scale
        clipped = np.zeros(J, bool)
        if replay.lo is not None:
            clipped = (target <= replay.lo + 1e-6) | (target >= replay.hi - 1e-6)
        raws[i] = raw if err < 1e-5 else np.where(clipped, raw, recovered).astype(np.float32)
        targets[i] = target
        # the first physics step whose motor torque follows the new target
        applied = None
        if i - 1 in targets:
            prev_target = targets[i - 1]
            after = first_rows[(t_ls[first_rows] >= t_c) & (t_ls[first_rows] <= t_c + WINDOW_NS)]
            for a in after:
                p = first_rows[np.searchsorted(first_rows, a) - 1]  # state the bridge's PD used
                q, dq, tau = np.float32(ls[p, LS_Q]), np.float32(ls[p, LS_DQ]), np.float32(ls[a, LS_TAU])
                new = np.median(np.abs(tau - (kp * (target - q) - kd * dq)))
                old = np.median(np.abs(tau - (kp * (prev_target - q) - kd * dq)))
                if new < old:
                    applied = a
                    break
        segment = next((name for name, a, b in segments if a <= t_c < b), None)
        rows_out.append(dict(step=i, segment=segment, error=err, obs_to_cmd_ms=(t_c - t_ls[r]) / 1e6,
                             sim_delay_ms=None if applied is None else float(ticks[applied] - ticks[r])))
    periods = np.diff([s[0] for s in steps]) / 1e6
    missed = last_index - len(rows_out)  # step numbers with no published target seen by the probe
    return rows_out, zero_msgs, periods, missed


def state_rows(ls):
    """Rows with the first arrival of each simulator state (the bridge publishes at 1 kHz, physics 500 Hz)."""
    ticks = ls[:, LS_TICK]
    first = np.ones(len(ls), bool)
    first[1:] = ticks[1:] != ticks[:-1]
    return np.where(first)[0]


def torque_misfit(ls, state, row, target, kp, kd):
    """Median over the joints of |tau_est(row) - (kp (target - q) - kd dq)| with q, dq of `state`, N m."""
    q, dq = np.float32(ls[state, LS_Q]), np.float32(ls[state, LS_DQ])
    return float(np.median(np.abs(np.float32(ls[row, LS_TAU]) - (kp * (target - q) - kd * dq))))


def target_ticks(ls, steps, kp, kd):
    """Per new policy target (policy_steps): simulator tick (ms) of the last state received before it arrived,
    and of the first physics step whose motor torque follows it rather than the previous target (None if not
    seen within WINDOW_NS after the arrival)."""
    t_ls, ticks = ls[:, LS_T], ls[:, LS_TICK]
    rows = state_rows(ls)
    out, previous = [], None
    for t_c, target in steps:
        arrived = int(ticks[max(np.searchsorted(t_ls, t_c, side='right') - 1, 0)])
        applied = None
        if previous is not None:
            for a in rows[(t_ls[rows] >= t_c) & (t_ls[rows] <= t_c + WINDOW_NS)]:
                p = rows[np.searchsorted(rows, a) - 1]  # the state the bridge's PD used at best
                if torque_misfit(ls, p, a, target, kp, kd) < torque_misfit(ls, p, a, previous, kp, kd):
                    applied = int(ticks[a])
                    break
        out.append((arrived, applied, target))
        previous = target
    return out


def pd_ages(ls, lc, kp, kd, t_from, t_to):
    """How old the state behind each physics step's motor torque was, in physics steps.

    The bridge (1 kHz, no lock with the physics thread) sets ctrl = PD(lowcmd target, sensordata). MuJoCo
    computes sensordata inside mj_step before integrating, so the state a lowstate row carries is the one at
    the start of the step that produced it, and the best the bridge can do is age 1: a PD on the state one
    physics step (PHYSICS_MS) before the one the torque acts on (our training: age 0). For each recorded state
    whose three predecessors were recorded too, the age 1-3 and the target (the last two received) whose PD
    fits tau_est best, if within PD_FIT N m; otherwise `other`; states with a gap before them are `unknown`."""
    rows = state_rows(ls)
    t_ls, ticks = ls[:, LS_T], ls[:, LS_TICK]
    by_tick = {int(ticks[r]): r for r in rows}
    t_lc, q_lc = lc[:, LC_T], np.float32(lc[:, LC_Q])
    changes = np.where(np.any(np.diff(q_lc, axis=0) != 0, axis=1))[0] + 1  # rows starting a new target
    counts = dict.fromkeys((1, 2, 3, 'other', 'unknown'), 0)
    misfits = []
    for row in rows[(t_ls[rows] >= t_from) & (t_ls[rows] <= t_to)]:
        k = int(ticks[row])
        before = [by_tick.get(k - PHYSICS_MS * n) for n in (1, 2, 3)]
        i = np.searchsorted(t_lc, t_ls[row], side='right') - 1
        if any(b is None for b in before) or i < 0:
            counts['unknown'] += 1
            continue
        start = np.searchsorted(changes, i, side='right') - 1  # the current target began at changes[start]
        targets = [q_lc[i]] + ([q_lc[changes[start] - 1]] if start >= 0 else [])
        fit, age = min((torque_misfit(ls, b, row, target, kp, kd), n)
                       for n, b in enumerate(before, 1) for target in targets)
        misfits.append(fit)
        counts[age if fit < PD_FIT else 'other'] += 1
    known = sum(v for key, v in counts.items() if key != 'unknown')
    return dict(counts=counts, shares={key: (v / known if known else None) for key, v in counts.items()
                                       if key != 'unknown'},
                misfit_nm=stats(misfits))


def timing(ls, lc, deploy, steps, t_from, t_to):
    """Simulator timing between two wall times (the drift window of a walk): real-time factor (simulator
    seconds per wall second, overall and the lowest over 0.5 s), policy step intervals in simulator time (the
    first physics step with each new target), and how fresh the bridge's PD was (pd_ages)."""
    t_ls, ticks = ls[:, LS_T], ls[:, LS_TICK]
    sim = lambda t: np.interp(t, t_ls, ticks) / 1e3  # simulator s at a wall time
    edges = np.arange(t_from, t_to + 1, 0.5e9)
    rtf_windows = np.diff(sim(edges)) / 0.5
    kp, kd = np.float32(deploy['stiffness']), np.float32(deploy['damping'])
    marks = target_ticks(ls, steps, kp, kd)
    lo, hi = sim(t_from) * 1e3, sim(t_to) * 1e3
    applied = np.array([a for _, a, _ in marks if a is not None and lo <= a <= hi])
    intervals = np.diff(applied).astype(float)
    period_ms = float(deploy['step_dt']) * 1e3
    some = len(intervals) > 0
    return dict(rtf=float((sim(t_to) - sim(t_from)) / ((t_to - t_from) / 1e9)),
                rtf_min_05s=float(rtf_windows.min()), rtf_max_05s=float(rtf_windows.max()),
                policy_sim_ms=stats(intervals), policy_sim_sd_ms=float(intervals.std()) if some else None,
                policy_sim_off=float(np.mean(intervals != period_ms)) if some else None,
                policy_sim_off2=float(np.mean(np.abs(intervals - period_ms) > PHYSICS_MS)) if some else None,
                pd=pd_ages(ls, lc, kp, kd, t_from, t_to)), marks


def write_targets(run, marks):
    """targets_sim.csv: every new policy target of the first Velocity entry with its simulator time, ms (the
    state received before it arrived, and the first physics step whose torque follows it; empty if unseen)."""
    lines = ['sim_ms_arrived,sim_ms_applied,' + ','.join(f'q{i}' for i in range(J))]
    for arrived, applied, target in marks:
        lines.append(f"{arrived},{'' if applied is None else applied}," + ','.join(f'{v:.9g}' for v in target))
    (run / 'targets_sim.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return None
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()))


def wrist_only(runs):
    """`--wrist`: the wrist mode of earlier runs next to their report.json drift, without rewriting anything."""
    for run in map(Path, runs):
        events, timeline = scenario_events(run)
        deploy = yaml.safe_load((run / 'deploy.yaml').read_text(encoding='utf-8'))
        lc = table(run, 'lowcmd.csv')
        t0 = events['start'][0]
        velocity = first_velocity(lc, deploy, fsm_timeline(lc, deploy, t0), t0)
        if velocity is None:
            print(f'{run.name}: g1_ctrl did not enter Velocity')
            continue
        w = wrist_mode(lc, wrist_windows(events, timeline, velocity[2]), fixstand_wrists(run))
        report = run / 'report.json'
        r = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
        ws = r.get('walk_settled')
        walk = '' if not ws else (f"; ходьба {r['walk_command']:g} м/с: {ws['path_m'] / ws['seconds']:.2f} м/с, "
                                  f"увод {ws['yaw_rate']:+.3f} рад/с")
        fell = '' if r.get('fell_s') is None else f"; ПАДЕНИЕ на {r['fell_s']:.1f} с"
        print(f"{run.name}: {wrist_text(w)}{walk}{fell}")


def main():
    if sys.argv[1] == '--wrist':
        wrist_only(sys.argv[2:])
        return
    run = Path(sys.argv[1])
    ls, lc, sport, events, timeline, deploy = load(run)
    t0 = events['start'][0]
    fsm = fsm_timeline(lc, deploy, t0)
    velocity = first_velocity(lc, deploy, fsm, t0)
    export = run / 'export.txt'
    report = dict(run=run.name, export=export.read_text().strip() if export.exists() else None,
                  scenario=events['start'][1] or 'walk',
                  events={k: round((v[0] - t0) / 1e9, 2) for k, v in events.items()},
                  timeline=[(round((t - t0) / 1e9, 2), name, value) for t, name, value in timeline], fsm=fsm)
    t_end = events['end'][0]
    wall = (ls[-1, LS_T] - ls[0, LS_T]) / 1e9
    report['realtime'] = float((ls[-1, LS_TICK] - ls[0, LS_TICK]) / 1e3 / wall)
    report['rates_hz'] = dict(lowstate=len(ls) / wall, lowcmd=len(lc) / ((lc[-1, LC_T] - lc[0, LC_T]) / 1e9))
    if velocity is None:
        report['error'] = 'g1_ctrl did not enter Velocity'
        print(json.dumps(report, ensure_ascii=False, indent=1))
        (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        return
    t_enter, left, t_leave = velocity
    fell = falls(sport, ls, t_enter, min(t_leave, t_end))
    report['fell_s'] = None if fell is None else round((fell - t0) / 1e9, 2)
    report['left_velocity'] = left
    band_off = events['band_off'][0]
    segments = [('band', t_enter, band_off)]
    if 'walk' in events:  # the walk scenario (also every run before gamepad.py --scenario)
        walk_t, stop_t = events['walk'][0], events['stop'][0]
        segments += [('stand', band_off, walk_t), ('walk', walk_t, stop_t), ('after', stop_t, t_end)]
        report['stand'] = pose_metrics(sport, ls, band_off + 1e9, walk_t)
        report['walk'] = pose_metrics(sport, ls, walk_t, stop_t)
        report['walk_settled'] = pose_metrics(sport, ls, walk_t + SETTLE * 1e9, stop_t)
        report['walk_command'] = float(events['walk'][1])
        report['after'] = pose_metrics(sport, ls, stop_t + 1e9, t_end)
        inside = (sport[:, 0] >= walk_t) & (sport[:, 0] <= stop_t)
        report['walk']['pelvis_z_min'] = float(sport[inside, 3].min())
    scenario = scenario_segments(timeline)
    segments += [(f"{k}:{s['name']}", s['start'], s['end']) for k, s in enumerate(scenario)]
    # The stand after the band, second by second: until the walk, or to the end of the first `stand` segment
    stand_end = events['walk'][0] if 'walk' in events else next(
        (s['end'] for s in scenario[:1] if s['name'] == 'stand' and not any(s['command'])), None)
    if stand_end is not None:
        report['stand_by_second'] = stand_by_second(ls, band_off, min(stand_end, t_leave))
    report['segments'] = [segment_report(sport, ls, s, t0) for s in scenario]
    if 'passive' in events:
        report['passive'] = passive_report(sport, ls, fsm, timeline, t0)
    if 'push' in events:
        report['pushes'], report['pushes_unclaimed'] = pushes_report(run, sport, ls, timeline, t0, t_end)
    report['entries'] = velocity_entries(lc, deploy, t0)
    report['entry_holds'] = entry_holds(run)
    report['wrist'] = wrist_mode(lc, wrist_windows(events, timeline, t_leave), fixstand_wrists(run))
    rows, zero_msgs, periods, missed = latency(run, ls, lc, deploy, t_enter, t_leave, segments)
    # Timing over the drift window of the walk (after SETTLE), else over the first Velocity entry after the band
    t_from, t_to = ((events['walk'][0] + SETTLE * 1e9, events['stop'][0]) if 'walk' in events
                    else (band_off, min(t_leave, t_end)))
    steps, _ = policy_steps(lc, t_enter, t_leave)
    report['timing'], marks = timing(ls, lc, deploy, steps, t_from, t_to)
    write_targets(run, marks)
    walk_rows = [r for r in rows if r['segment'] == 'walk' and r['error'] < 1e-4]
    if walk_rows:  # gait phase of the first matched step of the walk (g1_ctrl: (step + 1) / 30 of a cycle)
        cycle = round(float(deploy['observations']['gait_phase']['params']['period']) / float(deploy['step_dt']))
        report['timing']['walk_first_step'] = walk_rows[0]['step']
        report['timing']['walk_start_phase'] = (walk_rows[0]['step'] + 1) % cycle / cycle
    good = [r for r in rows if r['error'] < 1e-4 and r['step'] > 1]
    report['latency'] = dict(steps=len(rows), matched=len(good), missed_by_probe=missed,
                             max_error_matched=max((r['error'] for r in good), default=None),
                             obs_to_cmd_ms=stats(r['obs_to_cmd_ms'] for r in good),
                             sim_delay_ms=stats(r['sim_delay_ms'] for r in good),
                             policy_period_ms=stats(periods), zero_target_msgs_at_entry=zero_msgs)
    (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    (run / 'latency_steps.json').write_text(json.dumps(rows), encoding='utf-8')
    print_report(report)


def wrist_text(w):
    """The wrist mode in Russian: «левое / правое / оба / ни одного» overall, then per window and FixStand."""
    modes = dict(left='левое', right='правое', both='оба', none='ни одного', mixed='разный')
    windows = dict(stand='стоя', moving='на ходу')
    parts = [f"{windows[name]} {modes[v['mode']]} (q20 {v['left_deg']:+.1f}°, q27 {v['right_deg']:+.1f}°)"
             for name, v in w['windows'].items()]
    fixstand = w['fixstand_deg']
    if fixstand:
        parts.append(f'поза FixStand: q20 {fixstand[0]:+.1f}°, q27 {fixstand[1]:+.1f}°')
    return (f"режим запястий {modes.get(w['mode'], '—')} (тангаж, средняя цель, согнуто при |цели| > "
            f"{w['threshold_deg']:g}°): " + '; '.join(parts))


def print_report(r):
    def f(v, digits=2):
        return '—' if v is None else f'{v:.{digits}f}'

    print(f"Запуск {r['run']}")
    print(f"Реальное время симулятора: {r['realtime']:.3f}; lowstate {r['rates_hz']['lowstate']:.0f} Гц, "
          f"lowcmd {r['rates_hz']['lowcmd']:.0f} Гц")
    print('FSM: ' + ' → '.join(f'{name} ({t:.1f} с)' for t, name in r['fsm']))
    print('События сценария: ' + ', '.join(f'{k} {v:.1f} с' for k, v in r['events'].items()))
    fell = 'нет' if r['fell_s'] is None else 'да, на %.1f с' % r['fell_s']
    states = lambda pairs: ' → '.join(f'{name} ({t:.1f} с)' for t, name in pairs)
    print(f"Сценарий {r['scenario']}. Падение в первом входе в Velocity: {fell}"
          + (f"; выход из Velocity: {states(r['left_velocity'])}" if r['left_velocity'] else ''))
    if 'stand' in r:
        s = r['stand']
        print(f"Стоя ({s['seconds']:.1f} с после снятия ленты): сместился на {s['distance_m']:.3f} м, "
              f"поворот {s['heading_deg']:+.1f}° ({s['yaw_rate']:+.3f} рад/с)")
        w, ws = r['walk'], r['walk_settled']
        print(f"Ходьба {r['walk_command']:g} м/с, {w['seconds']:.1f} с: вперёд по начальному курсу "
              f"{w['forward_m']:.2f} м, вбок {w['lateral_m']:+.2f} м, путь {w['path_m']:.2f} м, "
              f"курс {w['heading_deg']:+.1f}°, таз не ниже {w['pelvis_z_min']:.3f} м")
        print(f"  после первых {SETTLE:g} с: скорость {ws['path_m'] / ws['seconds']:.2f} м/с, "
              f"увод {ws['yaw_rate']:+.3f} рад/с (как goal_check)")
        a = r['after']
        print(f"После стопа ({a['seconds']:.1f} с): сместился на {a['distance_m']:.3f} м, "
              f"поворот {a['heading_deg']:+.1f}° ({a['yaw_rate']:+.3f} рад/с)")
    by_second = r.get('stand_by_second')
    if by_second:
        print(f"Стоя после снятия ленты по секундам, рад/с ({len(by_second)} с, курс "
              f"{math.degrees(sum(by_second)):+.1f}°): " + ' '.join(f'{v:+.3f}' for v in by_second))
    for s in r.get('segments', []):
        vx, vy, wz = s['command']
        st, wh = s['settled'], s['whole']
        fell = '' if s['fell_s'] is None else f", ПАДЕНИЕ на {s['fell_s']:.1f} с"
        # name@height: the right stick down (gamepad.py crouch / --ry, g1_ctrl-height.patch)
        crouch = f", таз после {s['settle_s']:g} с в среднем {st['pelvis_z_mean']:.3f} м" if '@' in s['name'] else ''
        print(f"{s['start_s']:5.1f} с {s['name']:<11} ({vx:+.2f}, {vy:+.2f}, {wz:+.2f}) {s['seconds']:.1f} с; "
              f"после {s['settle_s']:g} с: вперёд {st['forward_speed']:+.2f} м/с, "
              f"вбок {st['sideways_speed']:+.2f} м/с, рысканье {st['yaw_rate']:+.3f} рад/с, "
              f"смещение {st['distance_m']:.2f} м; весь отрезок: смещение {wh['distance_m']:.2f} м, "
              f"курс {wh['heading_deg']:+.1f}°, таз ≥ {wh['pelvis_z_min']:.3f} м{crouch}{fell}")
    p = r.get('passive')
    if p:
        print(f"Passive: LT + B на {p['command_s']:.1f} с, g1_ctrl в Passive на "
              f"{f(p['entered_s'], 2)} с; дальше FSM: {states(p['states_after'])}")
        if 'z_min' in p:
            fall = 'не упал' if p['fall_after_s'] is None else f"упал через {p['fall_after_s']:.2f} с"
            print(f"  таз {p['z_at_entry']:.3f} → мин {p['z_min']:.3f} м, наклон до {p['tilt_max_deg']:.0f}°, "
                  f"{fall}, сместился на {p['moved_m']:.2f} м; таз перед лентой {p['z_before_band']:.3f} м"
                  + (f", на ленте {p['z_hanging']:.3f} м" if 'z_hanging' in p else ''))
            if p.get('z_hanging', 1.) < FALL_Z:
                print('  лента не подняла робота: нажатие «9» до симулятора не дошло, возврат не проверен')
    pushes = r.get('pushes')
    if pushes:
        print(f"Толчки (сила на таз по его курсу; окно — до следующего события сценария; «успокоился» — скорость "
              f"таза и рысканье, средние за {PUSH_AVERAGE:g} с, в пределах {PUSH_SETTLE_SPEED:g} м/с и "
              f"{PUSH_SETTLE_YAW:g} рад/с от секунды до толчка, {PUSH_SETTLE_HOLD:g} с подряд):")
        for p in pushes:
            name = PUSH_DIRECTIONS.get(p['direction'], p['direction'])
            if p.get('missing'):
                print(f"  {p['key_s']:5.1f} с {name} {p['dv_command']:g} м/с: в sim.log толчка нет (клавиша не "
                      f"дошла или симулятор без патча)")
                continue
            # the key's modifiers did not give the dv asked for
            asked = ('' if abs(p['dv'] - p['dv_command']) < 1e-6
                     else f" (запрошено {p['dv_command']:g}, НЕ СОВПАЛО)")
            if p.get('lying'):
                print(f"  {p['start_s']:5.1f} с {name} {p['dv']:g} м/с{asked}: робот уже лежит, не в счёт")
                continue
            fell = 'нет' if p['fell_after_s'] is None else f"ДА, через {p['fell_after_s']:.2f} с"
            settle = (f"через {p['settle_s']:.2f} с" if p['settle_s'] is not None
                      else f"не успокоился за {p['window_s'] - PUSH_AVERAGE - PUSH_SETTLE_HOLD:.1f} с")
            forward, left = p['offset_end_m']
            print(f"  {p['start_s']:5.1f} с {name} {p['dv']:g} м/с{asked} ({p['force_n']:.0f} Н × "
                  f"{p['seconds']:.2f} с = {f(p['impulse_ns'], 1)} Н·с): падение {fell}; скорость таза по толчку до "
                  f"{p['peak_along_ms']:+.2f} м/с; {settle}; за {p['window_s']:.1f} с смещение от хода до толчка "
                  f"вперёд {forward:+.2f} / вбок {left:+.2f} м, курс {p['heading_end_deg']:+.1f}° (за вычетом "
                  f"поворота до толчка {p['heading_trend_end_deg']:+.1f}°); таз ≥ {p['pelvis_z_min']:.3f} м, "
                  f"наклон ≤ {p['tilt_max_deg']:.0f}°")
        if r.get('pushes_unclaimed'):
            print(f"  в sim.log ещё толчков без события сценария: {r['pushes_unclaimed']}")
    entries = r.get('entries')
    if entries:
        holds = r.get('entry_holds')
        print(f"Входы в Velocity: {len(entries)}; " + (f"g1_ctrl (патч) держал прежнюю цель тактов FSM: "
                                                        f"{', '.join(map(str, holds))}" if holds
                                                        else 'строк патча в g1_ctrl.log нет'))
        for e in entries:
            policy = '—' if e['policy_after_ms'] is None else f"{e['policy_after_ms']:.1f} мс"
            print(f"  {e['t_s']:5.1f} с: до первой цели политики {e['before_policy']} сообщ. (с прежней целью "
                  f"{e['held']}, с нулевой {e['zero']}), она пришла через {policy}; скачок цели в первом "
                  f"сообщении {f(e['first_jump_rad'], 3)} рад; нулевых целей за вход {e['zero_all']}"
                  + (f"; суставов с 0 в первой цели {e['torn_zero_joints']}" if e['torn_zero_joints'] else ''))
    w = r.get('wrist')
    if w:
        text = wrist_text(w)
        print(text[0].upper() + text[1:])
    tm = r.get('timing')
    if tm:
        ps, pd = tm['policy_sim_ms'], tm['pd']
        share = lambda key: '—' if pd['shares'][key] is None else f"{pd['shares'][key]:.0%}"
        print(f"Хронометраж ({'после первых 2 с ходьбы' if 'walk' in r['events'] else 'первый вход в Velocity'}): "
              f"RTF {tm['rtf']:.3f} (за 0,5 с от {tm['rtf_min_05s']:.3f} до {tm['rtf_max_05s']:.3f})"
              + (f", фаза шага на старте ходьбы {tm['walk_start_phase']:.2f}" if 'walk_start_phase' in tm else ''))
        if ps:
            print(f"  шаги политики во времени симуляции: в среднем {ps['mean']:.2f} мс, "
                  f"σ {tm['policy_sim_sd_ms']:.2f}, от {ps['min']:.0f} до {ps['max']:.0f}, "
                  f"не 20 мс — {tm['policy_sim_off']:.0%}, "
                  f"дальше ±2 мс — {tm['policy_sim_off2']:.0%}")
        print(f"  ПД моста на состоянии старше на 1 шаг физики {share(1)}, на 2 — {share(2)}, на 3 — {share(3)}, "
              f"не объяснён — {share('other')} (неизвестно {pd['counts']['unknown']} из "
              f"{sum(pd['counts'].values())} состояний)")
    lat = r['latency']
    o, d, p = lat['obs_to_cmd_ms'], lat['sim_delay_ms'], lat['policy_period_ms']
    print(f"Задержка: опознано {lat['matched']} из {lat['steps']} шагов политики "
          f"(ошибка цели ≤ {f(lat['max_error_matched'], 7)} рад; регистратор пропустил "
          f"{lat['missed_by_probe']} lowcmd с новой целью)")
    if o:
        print(f"  lowstate → lowcmd (настенные часы): медиана {o['median']:.2f} мс, 95% {o['p95']:.2f}, "
              f"макс {o['max']:.2f}")
    if d:
        print(f"  состояние → первый шаг физики с новой целью (время симуляции): "
              f"медиана {d['median']:.1f} мс, среднее {d['mean']:.2f}, 95% {d['p95']:.1f}, "
              f"мин {d['min']:.1f}, макс {d['max']:.1f}")
    if p:
        print(f"  период политики: в среднем {p['mean']:.2f} мс, от {p['min']:.2f} до {p['max']:.2f}; "
              f"lowcmd с нулевой целью при входе в Velocity: {lat['zero_target_msgs_at_entry']}")


if __name__ == '__main__':
    main()
