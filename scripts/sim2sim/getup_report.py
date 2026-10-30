"""Report of a sim2sim getting-up run (gamepad.py --scenario getup, run.sh --getup <export>).

Usage: python getup_report.py <run dir>...    (run.sh calls it in $ROOT/venv: numpy, pyyaml; writes getup.json into
       each run dir and prints the report in Russian)

Inputs: lowstate.csv / lowcmd.csv / sport.csv (sim2sim_probe), events.csv (gamepad.py), g1_ctrl.log (its state changes
and the lines of the getup and fall-passive patches), g1_ctrl_config.yaml (the config.yaml the run used: GetUp's
max_joint_speed).
- The fall: how the robot lies just before LT + down: pelvis height, torso tilt, and on what (the torso's axis closest
  to straight down: back, front, a side; upright = sitting or kneeling).
- GetUp: its entry is lowcmd's gains leaving Passive's 0 after LT + down (g1_ctrl.log's wall clock is matched to that).
  Up is the evaluation's pose without the feet (g1_tensor_math.getup_success_pose: pelvis ≥ 0.70 m, top of the head ≥
  1.20 m, torso tilt ≤ 15°); the torso and head come from the pelvis IMU, the waist angles and the model's offsets
  (unitree_mujoco g1_29dof.xml, HEAD_TOP as humanoid_sim's). Held = in that pose until GetUp ends. How it ended: RT + A
  (Velocity), its timeout, a joint past its max_joint_speed, LT + B or never. The fastest joints of the state against
  max_joint_speed, and how many would have passed one 25 rad/s for all (the first version of the patch).
- Velocity after GetUp: the target kept until its first policy step (the policy-start patch's line), a fall (pelvis
  below analyze.FALL_Z or tilted past analyze.FALL_TILT, or Velocity's tilt check), and the stand / walk / stop
  segments as analyze.py reports them (after the first second of each).
- The hand-over (RT + A): the pose GetUp hands over (its mean over the last POSE_S s of GetUp: Σ (q − home)² against
  the getting-up export's default_joint_pos, the joints farthest from home, torso and pelvis tilt) and the joint
  speeds over Velocity's first HANDOVER_S s, the arms' fastest apart.
- The getting-up chain (run.sh --getup-track, g1_ctrl-getupchain.patch): getting up is the span of GetUp, GetUpBlend
  and GetUpTrack (CHAIN), from GetUp's entry to the first change into another state; per state its time, fastest
  joints against its own max_joint_speed and cut-offs; the hand-over (its policy step, rule, angle and deviation from
  the patch's warn line); GetUpBlend's entry from lowcmd (motor 0's kp leaving GetUp's for the transition's), and
  its lowcmd targets and gains (q, kp0, kd0, kp15) against chain_model.Blend from the last target before it: the C++
  against the Python model, which tests/test_getup_chain_patch.py checks against getup_chain.py (the first
  GetUpBlend, when there are several); GetUpBlend's returns to GetUp (iteration 89: the warn line's time, angle,
  step and limit) and whether GetUp went on alone past max_retries. The hands' floor force is not in the run:
  unitree_mujoco publishes no contacts.
"""

import datetime
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

import analyze
import chain_model
from analyze import LC_KP0, LC_Q, LC_T, LS_DQ, LS_Q, LS_QUAT, LS_T

# SDK motor order (the critic's names, lab-critic/it35/over25.txt)
JOINTS = ('L_hip_p', 'L_hip_r', 'L_hip_y', 'L_knee', 'L_ank_p', 'L_ank_r',
          'R_hip_p', 'R_hip_r', 'R_hip_y', 'R_knee', 'R_ank_p', 'R_ank_r',
          'w_yaw', 'w_roll', 'w_pitch',
          'L_sh_p', 'L_sh_r', 'L_sh_y', 'L_elb', 'L_wr_r', 'L_wr_p', 'L_wr_y',
          'R_sh_p', 'R_sh_r', 'R_sh_y', 'R_elb', 'R_wr_r', 'R_wr_p', 'R_wr_y')
WAIST = (12, 13, 14)  # SDK indices of the waist yaw (z), roll (x), pitch (y) joints
# unitree_mujoco g1_29dof.xml: waist_yaw_link at the pelvis origin, waist_roll_link and torso_link offsets (no rotations)
WAIST_ROLL_POS = np.array([-0.0039635, 0., 0.035])
TORSO_POS = np.array([0., 0., 0.019])
HEAD_TOP = np.array([.005, 0., .477])  # g1_tensor_math.HEAD_TOP, torso_link frame
UP_PELVIS, UP_HEAD, UP_TILT = .70, 1.20, math.radians(15.)  # g1_tensor_math.getup_success_pose
OLD_LIMIT = 25.  # rad/s, the first getup patch's one max_joint_speed for all joints
POSE_S = .5  # s of GetUp before RT + A whose mean pose is the one handed over
HANDOVER_S = 1.  # s of Velocity after RT + A whose joint speeds are the hand-over's
ARMS = range(15, 29)  # SDK indices of the shoulder, elbow and wrist joints
LOG_LINE = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] \[(\w+)\] (.*)$')
CHANGE = re.compile(r'FSM: Change state from (\w+) to (\w+)')
TIMEOUT = re.compile(r'^(\w+): ([\d.]+) s in the state > ([\d.]+) s')
TOO_FAST = re.compile(r'^(\w+): motor (\d+) at ([\d.]+) rad/s > ([\d.]+) rad/s')
CHAIN = chain_model.STATES  # GetUp, GetUpBlend, GetUpTrack: getting up when the chain is on
HANDED = re.compile(r'^GetUp: hand-over to (\w+) at policy step (\d+) \((\w+)\), angle ([\d.]+) deg, deviation '
                    r'([\d.]+) rad, face down since policy step (\d+)')
NO_HANDOVER = re.compile(r'^GetUp: no hand-over to (\w+) within (\d+) policy steps')
# iteration 89: GetUpBlend back to GetUp (config.yaml handover retry_angle / abort_angle), and GetUp past max_retries
RETURNED = re.compile(r'^GetUpBlend: pelvis ([\d.]+) deg from frame (\d+)\'s at policy step (\d+) of (\d+) > '
                      r'(\w+) ([\d.]+) deg, back to (\w+) \(return (\d+), lowstate tick (\d+)\)')
BLOCKED = re.compile(r'^GetUp: return (\d+) from (\w+) > max_retries (\d+), no hand-over on this entry')
LC_KD0, LC_KP15 = LC_KP0 + 1, LC_KP0 + 2  # lowcmd.csv: kp of motor 0, kd of motor 0, kp of motor 15
BLEND_TOL = 1e-4  # rad (and gain units): a lowcmd target that matches chain_model.Blend's
TILTED = re.compile(r'^(\w+): pelvis tilt ([\d.]+) deg > ([\d.]+) deg')
HELD = re.compile(r'^Velocity: kept the previous target for (\d+) FSM ticks')
LYING = {(0, -1): 'спина', (0, 1): 'живот', (1, -1): 'правый бок', (1, 1): 'левый бок', (2, -1): 'сидя / на коленях',
         (2, 1): 'вниз головой'}


def g1_log(run):
    """g1_ctrl.log lines: (wall clock s, level, text)."""
    out = []
    for line in (run / 'g1_ctrl.log').read_text(encoding='utf-8', errors='replace').splitlines():
        m = LOG_LINE.match(line)
        if m:
            wall = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S.%f').timestamp()
            out.append((wall, m.group(2), m.group(3)))
    return out


def rotation(quat):
    """(n, 3, 3) world <- body of quaternions (w, x, y, z)."""
    w, x, y, z = quat.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def axis_rotation(angle, axis):
    """(n, 3, 3) rotations by angle (n,) about the body axis 0 (x), 1 (y) or 2 (z)."""
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros(angle.shape + (3, 3))
    i, j = (axis + 1) % 3, (axis + 2) % 3  # cyclic, so y gets (z, x): +s at [x, z]
    out[:, axis, axis] = 1.
    out[:, i, i], out[:, i, j], out[:, j, i], out[:, j, j] = c, -s, s, c
    return out


def torso_and_head(ls, sport):
    """Per lowstate: torso rotation (n, 3, 3), top of the head (n, 3) and pelvis position (n, 3), the pelvis from
    sportmodestate (the IMU site at the pelvis origin) interpolated to the lowstate times."""
    t = ls[:, LS_T]
    pelvis = np.stack([np.interp(t, sport[:, 0], sport[:, k]) for k in (1, 2, 3)], -1)
    q = ls[:, LS_Q]
    yaw = rotation(ls[:, LS_QUAT]) @ axis_rotation(q[:, WAIST[0]], 2)
    roll = yaw @ axis_rotation(q[:, WAIST[1]], 0)
    torso = roll @ axis_rotation(q[:, WAIST[2]], 1)
    origin = pelvis + yaw @ WAIST_ROLL_POS + roll @ TORSO_POS
    return torso, origin + torso @ HEAD_TOP, pelvis


def lying_on(torso, pelvis_z):
    """What a torso rotation (3, 3) lies on: the body axis closest to straight down and its sign (LYING); 'не упал' with
    the pelvis above analyze.FALL_Z (m)."""
    if pelvis_z > analyze.FALL_Z:
        return 'не упал'
    down = torso.T @ np.array([0., 0., -1.])  # gravity in the torso frame
    axis = int(np.argmax(np.abs(down)))
    # on its back the torso's x (forward) points up, so gravity reads -x
    return LYING[(axis, int(np.sign(down[axis])))]


def joint_limits(run, state='GetUp'):
    """A state's max_joint_speed from the run's config.yaml, (29,) rad/s; None if the state had none."""
    cfg = yaml.safe_load((run / 'g1_ctrl_config.yaml').read_text(encoding='utf-8'))
    value = (cfg['FSM'].get(state) or {}).get('max_joint_speed')
    if value is None:
        return None
    return np.asarray(value if isinstance(value, list) else [value] * len(JOINTS), float)


def speeds(ls, t_from, t_to, limits):
    """Largest |dq| per joint between two times, the joints sorted by speed / limit (limit 25 without limits)."""
    inside = (ls[:, LS_T] >= t_from) & (ls[:, LS_T] <= t_to)
    if not inside.any():
        return None
    top = np.abs(ls[inside, LS_DQ]).max(0)
    limit = limits if limits is not None else np.full(len(JOINTS), OLD_LIMIT)
    order = np.argsort(-top / limit)
    return dict(max=float(top.max()), max_joint=JOINTS[int(top.argmax())],
                top=[(JOINTS[k], float(top[k]), float(limit[k])) for k in order[:3]],
                over_limit=[JOINTS[k] for k in range(len(JOINTS)) if top[k] > limit[k]],
                over_25=[JOINTS[k] for k in range(len(JOINTS)) if top[k] > OLD_LIMIT],
                per_joint=dict(zip(JOINTS, map(float, top))))


def handover(run, ls, torso, t_exit):
    """The hand-over GetUp -> Velocity at t_exit: the mean pose over GetUp's last POSE_S s against the getting-up
    export's default_joint_pos (SDK order through joint_ids_map), torso and pelvis tilt there, and the joint speeds
    over Velocity's first HANDOVER_S s."""
    deploy = yaml.safe_load((run / 'getup' / 'deploy.yaml').read_text(encoding='utf-8'))
    home = np.zeros(len(JOINTS))
    home[deploy['joint_ids_map']] = deploy['default_joint_pos']
    t = ls[:, LS_T]
    before = (t >= t_exit - POSE_S * 1e9) & (t < t_exit)
    if not before.any():
        return None
    q = ls[before, LS_Q].mean(0)
    off = (q - home) ** 2
    window = speeds(ls, t_exit, t_exit + HANDOVER_S * 1e9, None)
    arm = None
    if window:
        k = max(ARMS, key=lambda k: window['per_joint'][JOINTS[k]])
        arm = dict(joint=JOINTS[k], speed=window['per_joint'][JOINTS[k]], q_before=float(q[k]), home=float(home[k]))
    tilt = lambda r: float(np.degrees(np.arccos(np.clip(r[:, 2, 2], -1, 1))).mean())  # noqa: E731
    return dict(pose_sq=float(off.sum()),
                farthest=[(JOINTS[k], float(q[k]), float(home[k])) for k in np.argsort(-off)[:3]],
                torso_tilt_deg=tilt(torso[before]), pelvis_tilt_deg=tilt(rotation(ls[before, LS_QUAT])),
                speeds=window, arm=arm)


def blend_check(run, lc, t_change, t_next):
    """GetUpBlend in lowcmd against chain_model.Blend (the run's config.yaml and getup_track/: the tracker's deploy.yaml
    and handover.yaml). Its entry is the first lowcmd from 50 ms before the logged change on whose motor 0 kp is no
    longer GetUp's (the transition's step 1 scales it), its start the targets of the lowcmd before; each later lowcmd
    up to the tracker (t_next) against the model's step for its time (step_at), taking the nearest of k - 1, k, k + 1
    (the probe's arrival time is late by up to a few ms): each lowcmd is matched to the model's nearest step anywhere,
    whose error says whether the C++ computes the model's numbers, its distance from the time's step (step_lag) how
    late or early it came. The tracker's first targets: the largest jump of a free
    joint's target from the transition's last over the first policy step's lowcmd that differs from it."""
    cfg = yaml.safe_load((run / 'g1_ctrl_config.yaml').read_text(encoding='utf-8'))
    _, model, frame, _ = chain_model.chain(cfg, run / 'getup_track')
    t_lc, q_lc = lc[:, LC_T], lc[:, LC_Q]
    k0 = int(np.searchsorted(t_lc, t_change - 5e7))
    own = model.stiffness[0]
    moved = np.flatnonzero(np.abs(lc[k0:, LC_KP0] - own) > 1e-3)
    if not len(moved) or k0 + moved[0] == 0:
        return None
    first = k0 + int(moved[0])
    t_blend = t_lc[first]
    model.enter(q_lc[first - 1])
    table = [model.at(k) for k in range(1, model.steps + 1)]  # every step's (targets, kp, kd)
    q_all = np.array([row[0] for row in table])
    gains = np.array([(row[1][0], row[2][0], row[1][15]) for row in table])
    worst_q = worst_gain = 0.
    exact = count = lag = torn = 0
    last = first
    for j in range(first, len(t_lc)):
        if t_lc[j] >= t_next:
            break
        k = model.step_at((t_lc[j] - t_blend) / 1e9)
        # the model's step nearest this lowcmd (targets and gains together), wherever it is: the C++ computes the
        # model's numbers if that is ~0; how far it lies from the step for the time (k) is the timing, apart. The FSM
        # thread writes lowcmd without the publisher's lock (Unitree's states all do), so a message may carry some
        # motors at one step and the rest at the next: each value then matches one of the steps around the nearest
        # (torn, counted apart).
        error_q = np.abs(q_all - q_lc[j]).max(1)
        values = lc[j, [LC_KP0, LC_KD0, LC_KP15]]
        error_gain = np.abs(gains - values).max(1)
        best = int(np.argmin(error_q + error_gain))
        near = slice(max(best - 1, 0), best + 2)
        each_q = np.abs(q_all[near] - q_lc[j]).min(0).max()
        each_gain = np.abs(gains[near] - values).min(0).max()
        torn += bool(error_q[best] >= BLEND_TOL or error_gain[best] >= 1e-3)
        worst_q, worst_gain = max(worst_q, float(each_q)), max(worst_gain, float(each_gain))
        exact += best + 1 == k
        lag = max(lag, abs(best + 1 - k))
        count += 1
        last = j
    free = frame['free']
    jump = None
    changed = np.flatnonzero(np.abs(q_lc[last + 1:] - q_lc[last]).max(1) > 1e-7)
    if len(changed):
        jump = float(np.abs(q_lc[last + 1 + changed[0]] - q_lc[last])[free].max())
    final = model.at(model.steps)[0]
    return dict(samples=count, steps=model.steps, step_dt=float(model.step_dt),
                seconds=float((t_lc[last] - t_blend) / 1e9), q_max_error=float(worst_q),
                gain_max_error=float(worst_gain), step_share=float(exact / max(count, 1)), step_lag=int(lag),
                torn=int(torn),
                start_to_goal=float(np.abs(model.goal - model.start)[free].max()),
                end_error=float(np.abs(q_lc[last] - final).max()), track_jump=jump,
                ok=bool(count > 0 and worst_q < BLEND_TOL and worst_gain < 1e-3))


def chain_report(run, ls, lc, texts, phases, t_entry, up_pose):
    """The getting-up chain's part (the module's docstring): per state its start, time, first moment up and fastest
    joints against its own max_joint_speed; the hand-over line (or the one saying none came); GetUpBlend's check."""
    t_ls = ls[:, LS_T]
    rows = []
    for state, a, b in phases:
        inside = (t_ls >= a) & (t_ls <= b)
        up = np.flatnonzero(inside & up_pose)
        limits = joint_limits(run, state)
        rows.append(dict(state=state, from_s=round((a - t_entry) / 1e9, 2), seconds=round((b - a) / 1e9, 2),
                         up_s=None if not len(up) else round((t_ls[up[0]] - t_entry) / 1e9, 2),
                         speeds=speeds(ls, a, b, limits)))
    handed = next((HANDED.match(text) for _, text in texts if HANDED.match(text)), None)
    missed = next((NO_HANDOVER.match(text) for _, text in texts if NO_HANDOVER.match(text)), None)
    out = dict(phases=rows, missed_steps=None if missed is None else int(missed.group(2)), handover=None)
    # GetUpBlend's returns to GetUp (iteration 89), each with its time from GetUp's first entry; GetUp past max_retries
    out['returns'] = [dict(at_s=round((t - t_entry) / 1e9, 2), angle_deg=float(m.group(1)), step=int(m.group(3)),
                           steps=int(m.group(4)), why=m.group(5), limit_deg=float(m.group(6)), count=int(m.group(8)))
                      for t, m in ((t, RETURNED.match(text)) for t, text in texts) if m]
    out['blocked'] = any(BLOCKED.match(text) for _, text in texts)
    if handed:
        out['handover'] = dict(to=handed.group(1), step=int(handed.group(2)), rule=handed.group(3),
                               angle_deg=float(handed.group(4)), deviation=float(handed.group(5)),
                               faced_step=int(handed.group(6)))
    names = [p[0] for p in phases]
    if 'GetUpBlend' in names:
        _, a, b = phases[names.index('GetUpBlend')]
        out['blend'] = blend_check(run, lc, a, b)
    if 'GetUpTrack' in names:  # the state the tracker starts from against its frame (getup_chain's frame_errors)
        frame = chain_model.load_yaml(run / 'getup_track' / 'handover.yaml')
        k = min(int(np.searchsorted(t_ls, phases[names.index('GetUpTrack')][1])), len(t_ls) - 1)
        gravity = -rotation(ls[k:k + 1, LS_QUAT])[0][2, :]
        out['track_start'] = dict(angle_deg=float(np.degrees(chain_model.angle_between(gravity, frame['gravity']))),
                                  deviation=float(chain_model.deviation(ls[k, LS_Q], frame)))
    return out


def report(run):
    ls, lc, sport = (analyze.table(run, name) for name in ('lowstate.csv', 'lowcmd.csv', 'sport.csv'))
    events, timeline = analyze.scenario_events(run)
    t0, t_end = events['start'][0], events['end'][0]
    s = lambda t: None if t is None else round((t - t0) / 1e9, 2)  # noqa: E731
    out = dict(run=run.name, getup_export=(run / 'getup' / 'export.txt').read_text().strip(),
               velocity_export=(run / 'export.txt').read_text().strip(),
               events={k: s(v[0]) for k, v in events.items()})
    torso, head, pelvis = torso_and_head(ls, sport)
    tilt = np.arccos(np.clip(torso[:, 2, 2], -1, 1))
    t_ls = ls[:, LS_T]
    up_pose = (pelvis[:, 2] >= UP_PELVIS) & (head[:, 2] >= UP_HEAD) & (tilt <= UP_TILT)

    # The fall: the pose 0.1 s before LT + down
    t_press = events['getup'][0]
    k = max(np.searchsorted(t_ls, t_press - 1e8) - 1, 0)
    out['lying'] = dict(on=lying_on(torso[k], pelvis[k, 2]), pelvis_z=float(pelvis[k, 2]), head_z=float(head[k, 2]),
                        tilt_deg=float(np.degrees(tilt[k])))
    if 'push' in events:
        out['lying']['push'] = events['push'][1]

    # GetUp entry: lowcmd's motor 0 gain leaves Passive's 0 after LT + down
    kp = lc[:, LC_KP0]
    after = np.where((lc[:, LC_T] >= t_press - 5e8) & (kp != 0) & np.r_[True, kp[:-1] == 0])[0]
    log = g1_log(run)
    changes = [(wall, *CHANGE.search(text).groups()) for wall, _, text in log if CHANGE.search(text)]
    entry_log = next((wall for wall, a, b in changes if a == 'Passive' and b == 'GetUp'), None)
    if not len(after) or entry_log is None:
        out['getup'] = None
        return out
    t_entry = lc[after[0], LC_T]
    mono = lambda wall: t_entry + (wall - entry_log) * 1e9  # noqa: E731  g1_ctrl.log wall clock -> CLOCK_MONOTONIC
    # Getting up ends with the first change out of GetUp, or out of the chain's states when it is on.
    exit_change = next(((wall, b) for wall, a, b in changes if a in CHAIN and b not in CHAIN and wall >= entry_log),
                       None)
    t_exit = mono(exit_change[0]) if exit_change else t_end
    texts = [(mono(wall), text) for wall, _, text in log if wall >= entry_log]
    reason = 'не вышел до конца прогона'
    cutoff = None
    if exit_change:
        reason = dict(Velocity='RT + A → Velocity', FixStand='LT + вверх → FixStand').get(exit_change[1])
        for t, text in texts:
            if t > t_exit + 5e6:
                break
            if TIMEOUT.match(text) and TIMEOUT.match(text).group(1) in CHAIN:
                m = TIMEOUT.match(text)
                cutoff = dict(kind='timeout', state=m.group(1), seconds=float(m.group(2)))
            elif TOO_FAST.match(text) and TOO_FAST.match(text).group(1) in CHAIN:
                m = TOO_FAST.match(text)
                cutoff = dict(kind='speed', state=m.group(1), joint=JOINTS[int(m.group(2))], speed=float(m.group(3)),
                              limit=float(m.group(4)))
            elif TILTED.match(text) and TILTED.match(text).group(1) in CHAIN:
                cutoff = dict(kind='tilt', state=TILTED.match(text).group(1), deg=float(TILTED.match(text).group(2)))
        if reason is None:
            reason = {None: 'Passive без строки отсечки (LT + B или нет lowstate)', 'timeout': 'тайм-аут → Passive',
                      'speed': 'скорость сустава → Passive', 'tilt': 'наклон → Passive'}[cutoff and cutoff['kind']]
    inside = (t_ls >= t_entry) & (t_ls <= t_exit)
    up = np.where(inside & up_pose)[0]
    t_up = t_ls[up[0]] if len(up) else None
    held = None
    creep = None
    after_up = None
    if t_up is not None:
        later = (t_ls >= t_up) & (t_ls <= t_exit)
        held = float(up_pose[later].mean())
        creep = float(np.linalg.norm(pelvis[later][-1, :2] - pelvis[later][0, :2]))
        out_of_pose = later & ~up_pose
        after_up = dict(seconds=float((t_exit - t_up) / 1e9), pelvis_min=float(pelvis[later, 2].min()),
                        head_min=float(head[later, 2].min()), tilt_max_deg=float(np.degrees(tilt[later].max())),
                        out_of_pose_s=float(np.diff(t_ls)[out_of_pose[:-1]].sum() / 1e9))
    limits = joint_limits(run)
    # The chain's states in turn (GetUp alone without it): (state, from, to) in CLOCK_MONOTONIC ns
    phases, state, t_from = [], 'GetUp', t_entry
    for wall, a, b in changes:
        if exit_change and wall >= exit_change[0]:
            break
        if wall > entry_log and a == state and b in CHAIN:
            phases.append((state, t_from, mono(wall)))
            state, t_from = b, mono(wall)
    phases.append((state, t_from, t_exit))
    first_to = phases[0][2]
    out['getup'] = dict(
        entered_s=s(t_entry), left_s=s(t_exit) if exit_change else None, left_to=exit_change and exit_change[1],
        reason=reason, cutoff=cutoff, up_after_s=None if t_up is None else round((t_up - t_entry) / 1e9, 2),
        held_share=held, creep_m=creep, after_up=after_up,
        pelvis_z_at_exit=float(np.interp(t_exit, t_ls, pelvis[:, 2])),
        head_z_max=float(head[inside, 2].max()) if inside.any() else None,
        speeds=speeds(ls, t_entry, first_to, limits), limits=None if limits is None else limits.tolist())
    if len(phases) > 1 or any(NO_HANDOVER.match(text) for _, text in texts):
        out['chain'] = chain_report(run, ls, lc, texts, phases, t_entry, up_pose)

    # Velocity after GetUp
    if not exit_change or exit_change[1] != 'Velocity':
        out['velocity'] = None
        return out
    leave = next(((wall, b) for wall, a, b in changes if a == 'Velocity' and wall > exit_change[0]), None)
    t_leave = mono(leave[0]) if leave else t_end
    holds = [int(HELD.match(text).group(1)) for t, text in texts if t >= t_exit - 5e6 and HELD.match(text)]
    tilted = next((float(TILTED.match(text).group(2)) for t, text in texts
                   if t_exit <= t <= t_leave + 5e6 and TILTED.match(text)
                   and TILTED.match(text).group(1) == 'Velocity'), None)
    fell = analyze.falls(sport, ls, t_exit, min(t_leave, t_end))
    segments = [seg for seg in analyze.scenario_segments(timeline) if seg['start'] >= t_exit - 1e9]
    out['handover'] = handover(run, ls, torso, t_exit)
    out['velocity'] = dict(
        entered_s=s(t_exit), held_ticks=holds[0] if holds else None, left_s=s(t_leave) if leave else None,
        left_to=leave and leave[1], tilt_check_deg=tilted, fell_s=s(fell),
        segments=[analyze.segment_report(sport, ls, seg, t0) for seg in segments],
        speeds=speeds(ls, t_exit, min(t_leave, t_end), None))
    return out


def text(r):
    def f(v, digits=2):
        return '—' if v is None else f'{v:.{digits}f}'

    lines = [f"Прогон {r['run']}", f"  вставание: {r['getup_export']}; ходьба: {r['velocity_export']}"]
    lying = r['lying']
    push = f", толчок {lying['push']}" if 'push' in lying else ''
    lines.append(f"  лежит перед LT + вниз: {lying['on']} (таз {lying['pelvis_z']:.3f} м, голова {lying['head_z']:.2f} м, "
                 f"наклон торса {lying['tilt_deg']:.0f}°{push})")
    g = r['getup']
    if g is None:
        lines.append('  GetUp: не вошёл')
        return '\n'.join(lines)
    a = g.get('after_up')
    up = 'не встал' if g['up_after_s'] is None else (
        f"встал через {g['up_after_s']:.2f} с; до выхода ({a['seconds']:.1f} с) вне позы {a['out_of_pose_s']:.2f} с, таз ≥ "
        f"{a['pelvis_min']:.3f} м, голова ≥ {a['head_min']:.2f} м, наклон торса ≤ {a['tilt_max_deg']:.1f}°, сместился на "
        f"{g['creep_m']:.2f} м")
    sp = g['speeds']
    fastest = ', '.join(f'{name} {v:.1f}/{limit:g}' for name, v, limit in sp['top']) if sp else '—'
    over = (f"; выше своего порога: {', '.join(sp['over_limit'])}" if sp and sp['over_limit'] else '')
    over25 = (f"; выше 25 рад/с: {', '.join(sp['over_25'])}" if sp and sp['over_25'] else '; выше 25 рад/с — нет')
    cutoff = g['cutoff']
    detail = ''
    if cutoff and cutoff['kind'] == 'speed':
        detail = f" ({cutoff['joint']} {cutoff['speed']:.1f} > {cutoff['limit']:g} рад/с)"
    elif cutoff and cutoff['kind'] == 'timeout':
        detail = f" ({cutoff['seconds']:.2f} с)"
    if cutoff and cutoff.get('state', 'GetUp') != 'GetUp':
        detail += f" в {cutoff['state']}"
    lines.append(f"  GetUp с {g['entered_s']:.1f} с: {up}; голова до {f(g['head_z_max'])} м; выход на {f(g['left_s'], 1)} с: "
                 f"{g['reason']}{detail}; таз при выходе {g['pelvis_z_at_exit']:.3f} м")
    lines.append(f"    быстрее всего (рад/с, к порогу): {fastest}{over}{over25}")
    c = r.get('chain')
    if c:
        h = c['handover']
        if h:
            lines.append(f"  цепочка: передача в {h['to']} на шаге политики {h['step']} ({h['rule']}; угол {h['angle_deg']:.1f}°, "
                         f"отклонение {h['deviation']:.3f} рад; лицом вниз с шага {h['faced_step']})")
        else:
            missed = '' if c['missed_steps'] is None else f" за {c['missed_steps']} шагов"
            lines.append(f"  цепочка: передачи не было{missed}, GetUp один")
        for b in c.get('returns') or ():
            lines.append(f"    возврат {b['count']} в GetUp на {b['at_s']:.2f} с: таз {b['angle_deg']:.1f}° от кадра "
                         f"на шаге ПД {b['step']} из {b['steps']} (> {b['why']} {b['limit_deg']:.1f}°)")
        if c.get('blocked'):
            lines.append('    после max_retries возвратов GetUp дальше один')
        for p in c['phases']:
            sp = p['speeds']
            top = ', '.join(f'{name} {v:.1f}/{limit:g}' for name, v, limit in sp['top']) if sp else '—'
            over = f"; выше порога: {', '.join(sp['over_limit'])}" if sp and sp['over_limit'] else ''
            up_in = '' if p['up_s'] is None else f"; встал на {p['up_s']:.2f} с"
            lines.append(f"    {p['state']:<10} с {p['from_s']:.2f} с, {p['seconds']:.2f} с{up_in}; быстрее всего {top}{over}")
        b = c.get('blend')
        if b:
            jump = '—' if b['track_jump'] is None else f"{b['track_jump']:.3f}"
            lines.append(f"    ПД-переход против chain_model: {b['samples']} отсчётов lowcmd за {b['seconds']:.2f} с "
                         f"({b['steps']} шагов), цели max |разница| {b['q_max_error']:.1e} рад, усиления "
                         f"{b['gain_max_error']:.1e}, шаг по времени совпал в {100 * b['step_share']:.0f} % (самое большее "
                         f"на {b['step_lag']} шаг.), смешанных из двух соседних шагов {b['torn']}; путь цели до "
                         f"{b['start_to_goal']:.2f} рад; скачок цели при входе в трекер {jump} рад — "
                         + ('совпадает' if b['ok'] else 'НЕ СОВПАДАЕТ'))
        elif 'GetUpBlend' in [p['state'] for p in c['phases']]:
            lines.append('    ПД-переход: вход в lowcmd не найден (kp мотора 0 не сменился)')
        if c.get('track_start'):
            lines.append(f"    вход в трекер: до кадра {c['track_start']['angle_deg']:.1f}° по тазу, отклонение суставов "
                         f"{c['track_start']['deviation']:.3f} рад")
        lines.append('    сила на кистях: в связке не измерить (unitree_mujoco не публикует контакты)')
    v = r.get('velocity')
    if v is None:
        return '\n'.join(lines)
    fell = 'нет' if v['fell_s'] is None else f"ДА, на {v['fell_s']:.1f} с"
    left = '' if v['left_s'] is None else f"; ушёл в {v['left_to']} на {v['left_s']:.1f} с"
    tilt = '' if v['tilt_check_deg'] is None else f" (проверка наклона: {v['tilt_check_deg']:.0f}°)"
    lines.append(f"  Velocity с {v['entered_s']:.1f} с: прежняя цель {f(v['held_ticks'], 0)} тактов FSM до первого шага "
                 f"политики; падение {fell}{left}{tilt}")
    h = r.get('handover')
    if h:
        far = ', '.join(f'{name} {q:+.2f} (home {home:+.2f})' for name, q, home in h['farthest'])
        lines.append(f"    передача: поза за {POSE_S:g} с до RT + A — Σ(q − home)² {h['pose_sq']:.2f} рад², дальше всех "
                     f"{far}; наклон торса {h['torso_tilt_deg']:.1f}°, таза {h['pelvis_tilt_deg']:.1f}°")
        if h['speeds']:
            arm = h['arm']
            top = ', '.join(f'{name} {speed:.1f}' for name, speed, _ in h['speeds']['top'])
            lines.append(f"    первые {HANDOVER_S:g} с Velocity: быстрее всего (рад/с) {top}; руки — {arm['joint']} "
                         f"{arm['speed']:.1f} рад/с (было {arm['q_before']:+.2f} рад, home {arm['home']:+.2f})")
    for seg in v['segments']:
        st, wh = seg['settled'], seg['whole']
        fell = '' if seg['fell_s'] is None else f", ПАДЕНИЕ на {seg['fell_s']:.1f} с"
        lines.append(f"    {seg['name']:<5} {seg['command'][0]:+.2f} м/с, {seg['seconds']:.1f} с: смещение "
                     f"{wh['distance_m']:.2f} м ({wh['distance_m'] / seg['seconds']:.2f} м/с), после {seg['settle_s']:g} с "
                     f"вперёд {st['forward_speed']:+.2f} м/с, курс {wh['heading_deg']:+.1f}°, таз ≥ "
                     f"{wh['pelvis_z_min']:.3f} м{fell}")
    return '\n'.join(lines)


def main():
    for run in map(Path, sys.argv[1:]):
        r = report(run)
        (run / 'getup.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        print(text(r))


if __name__ == '__main__':
    main()
