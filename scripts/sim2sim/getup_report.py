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
from analyze import LC_KP0, LC_T, LS_DQ, LS_Q, LS_QUAT, LS_T

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
TIMEOUT = re.compile(r'^GetUp: ([\d.]+) s in the state > ([\d.]+) s')
TOO_FAST = re.compile(r'^GetUp: motor (\d+) at ([\d.]+) rad/s > ([\d.]+) rad/s')
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


def joint_limits(run):
    """GetUp's max_joint_speed from the run's config.yaml, (29,) rad/s; None if the state had none."""
    cfg = yaml.safe_load((run / 'g1_ctrl_config.yaml').read_text(encoding='utf-8'))
    value = cfg['FSM'].get('GetUp', {}).get('max_joint_speed')
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
    exit_change = next(((wall, b) for wall, a, b in changes if a == 'GetUp' and wall >= entry_log), None)
    t_exit = mono(exit_change[0]) if exit_change else t_end
    texts = [(mono(wall), text) for wall, _, text in log if wall >= entry_log]
    reason = 'не вышел до конца прогона'
    cutoff = None
    if exit_change:
        reason = dict(Velocity='RT + A → Velocity', FixStand='LT + вверх → FixStand').get(exit_change[1])
        for t, text in texts:
            if t > t_exit + 5e6:
                break
            if TIMEOUT.match(text):
                cutoff = dict(kind='timeout', seconds=float(TIMEOUT.match(text).group(1)))
            elif TOO_FAST.match(text):
                m = TOO_FAST.match(text)
                cutoff = dict(kind='speed', joint=JOINTS[int(m.group(1))], speed=float(m.group(2)),
                              limit=float(m.group(3)))
            elif TILTED.match(text) and TILTED.match(text).group(1) == 'GetUp':
                cutoff = dict(kind='tilt', deg=float(TILTED.match(text).group(2)))
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
    out['getup'] = dict(
        entered_s=s(t_entry), left_s=s(t_exit) if exit_change else None, left_to=exit_change and exit_change[1],
        reason=reason, cutoff=cutoff, up_after_s=None if t_up is None else round((t_up - t_entry) / 1e9, 2),
        held_share=held, creep_m=creep, after_up=after_up,
        pelvis_z_at_exit=float(np.interp(t_exit, t_ls, pelvis[:, 2])),
        head_z_max=float(head[inside, 2].max()) if inside.any() else None,
        speeds=speeds(ls, t_entry, t_exit, limits), limits=None if limits is None else limits.tolist())

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
    lines.append(f"  GetUp с {g['entered_s']:.1f} с: {up}; голова до {f(g['head_z_max'])} м; выход на {f(g['left_s'], 1)} с: "
                 f"{g['reason']}{detail}; таз при выходе {g['pelvis_z_at_exit']:.3f} м")
    lines.append(f"    быстрее всего (рад/с, к порогу): {fastest}{over}{over25}")
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
