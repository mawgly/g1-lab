"""Check a recording from the real G1 against docs/robot-first-run.md (CPU: numpy, onnxruntime, pyyaml).

Usage: python record_check.py <recording dir> [--policy DIR] [--log FILE] [--patches DIR] [--json FILE]

The recording dir holds what sim2sim_probe wrote on the robot side (lowstate.csv, lowcmd.csv; sport.csv may be
empty: in debug mode nothing may publish rt/sportmodestate, and it is not used here) and, for the ONNX replay, the
policy that ran (policy.onnx + deploy.yaml; --policy DIR copies them in if they are missing and compares md5 if
not). g1_ctrl's output (`./g1_ctrl ... 2>&1 | tee g1_ctrl.log`) is looked for in the dir unless --log says
otherwise. analyze.py's main() needs the sim2sim scenario files (events.csv, sport.csv); this script reuses only its
observation replay (Replay, latency, policy_steps), unchanged.

FSM phases come from lowcmd motor 0's kp (Passive 0, FixStand 100, Velocity deploy.yaml stiffness), as in analyze.py:
  FixStand  joints |q - q_fix|, gravity and gyro at rest over [entry + 4 s, entry + 6 s] (2 s interpolation, then
            2 s to settle: in the critic's hoist |q - q_fix| is 0.05 at 2-3 s and 0.03 from 3 s on,
            reference/fixstand_air.py); after it the hand tilt test (directions in time order, gyro sign).
  Velocity  knee target >= its clip, locked wrists exactly 0, ONNX on the observation rebuilt from lowstate vs
            lowcmd, latency (lowstate arrival -> lowcmd with the answer), policy period; with no stick command
            (|command| < 0.1 nearly all the phase) also the in-the-air limits (joints at their clip other than the
            knees, leg joint speed, |delta a|) and the standing ones (tilt, heading over 30 s, leg torque, leg speed
            bursts as a stand-in for steps); with a command they are printed for reference, tilt against 15 deg.
            Motor temperature and error flags if the probe recorded them (the robot build, docs/robot-first-run.md).
  Passive   how it went limp after Velocity (leg speed, tilt): information only.
Verdicts use the protocol's thresholds; `СТОП` means: do not go on to the next step of the protocol.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / 'sim2sim'):  # the package (tools/analyze.py) or the lab repo (scripts/sim2sim)
    if (path / 'analyze.py').exists():
        sys.path.insert(0, str(path))
        break
import analyze as an  # noqa: E402

J = an.J
LEGS = list(range(12))
KNEES = (3, 9)
LOCKED_WRISTS = (20, 21, 27, 28)  # lock_wrists: wrist pitch and yaw, ONNX outputs exactly 0
# unitree_rl_mjlab deploy/robots/g1/config/config.yaml, FSM.FixStand.qs[-1] (SDK order)
Q_FIX = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0, 0, 0] + [0.35, 0.18, 0, 0.87, 0, 0, 0]
                 + [0.35, -0.18, 0, 0.87, 0, 0, 0])
FIXSTAND_KP0 = 100.0

# Thresholds (docs/robot-first-run.md: the critic's list, iteration 21)
FIX_SETTLE_S, FIX_WINDOW_S = 4.0, 2.0  # static window after the FixStand entry
FIX_Q_ERR = 0.05  # rad, |q - q_fix|
GRAVITY_TOL = 0.05  # per component around (0, 0, -1)
GYRO_BIAS = 0.02  # rad/s, |mean| per axis at rest
GYRO_STILL = 0.05  # rad/s, p95 of |gyro| above which the robot was not at rest (measure again)
TILT_DEG = dict(pitch=3.0, roll=3.0, yaw=5.0)  # tilt test: excursions counted from these
SIGN_CORR = 0.7  # correlation of d(angle)/dt with its gyro axis that counts as the same sign
MERGE_S = 0.3  # excursions of one sign closer than this are one
LEG_SPEED_P95 = 2.0  # rad/s, in the air
DELTA_A = 0.03  # mean over joints and steps of |a_t - a_t-1|, action units
ONNX_TOL = 1e-3  # rad, replayed target vs lowcmd
ONNX_SHARE = 0.8  # share of policy steps that must match (sim2sim: 82-99 % at 1e-4; the rest lost by the probe)
LATENCY_MEDIAN_MS, LATENCY_P95_MS = 5.0, 10.0
PERIOD_MS, PERIOD_TOL_MS, PERIOD_SHARE = 20.0, 2.0, 0.95
STAND_TILT_DEG, WALK_TILT_DEG = 10.0, 15.0
HEADING_DEG, HEADING_S = 30.0, 30.0
LEG_TORQUE_NM, LEG_TORQUE_S = 60.0, 2.0  # sum |tau_est| of the 12 leg motors, mean over 2 s ("долго")
BURST_SPEED, BURST_WINDOW_S, BURST_MAX = 1.0, 10.0, 1  # leg speed bursts without a command: > 1 per 10 s
ENTRY_SKIP_S = 1.0  # the Velocity entry transient (in the hoist up to 3 rad/s) is left out of the bursts
COMMAND_ZERO = 0.1  # command norm under which g1_ctrl's gait phase stops (phase_stop): a stand command
COMMAND_SHARE = 0.05  # a phase with a command longer than this share is a walking one


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ok(flag):
    return 'ок' if flag else 'СТОП'


def f(v, fmt='{:.3f}'):
    return '—' if v is None else fmt.format(v)


def prepare(run, policy):
    """policy.onnx + deploy.yaml in the recording dir (copied from --policy if missing, md5 compared if not)."""
    notes = []
    for name in ('policy.onnx', 'deploy.yaml'):
        target = run / name
        if policy is not None:
            source = policy / name
            if not source.exists():
                raise SystemExit(f'record_check: нет {source}')
            if not target.exists():
                shutil.copy2(source, target)
                notes.append(f'{name}: скопирован из {policy}')
            elif md5(target) != md5(source):
                notes.append(f'{name}: ОТЛИЧАЕТСЯ от {source} (md5 {md5(target)} против {md5(source)})')
        if not target.exists():
            raise SystemExit(f'record_check: в {run} нет {name}; укажите --policy <папка политики пакета>')
    return notes


def phases(lc, deploy):
    """Consecutive lowcmd runs with one FSM state: name, start ns, end ns (the next phase's first message)."""
    velocity_kp = round(float(deploy['stiffness'][0]), 3)
    names = {0.0: 'Passive', FIXSTAND_KP0: 'FixStand', velocity_kp: 'Velocity'}
    label = [names.get(float(k), f'kp {k:g}') for k in np.round(lc[:, an.LC_KP0], 3)]
    out, start = [], 0
    for i in range(1, len(lc) + 1):
        if i == len(lc) or label[i] != label[start]:
            out.append(dict(name=label[start], t0=lc[start, an.LC_T], t1=lc[min(i, len(lc) - 1), an.LC_T]))
            start = i
    return out


def rows_between(times, t0, t1):
    return np.where((times >= t0) & (times < t1))[0]


def angles(rows):
    """Gravity in the pelvis frame, pitch forward and roll right from it (deg), yaw (deg, unwrapped)."""
    g = an.gravity_of(rows[:, an.LS_QUAT])
    pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))  # tilted forward: g_x > 0
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2]))  # right side down: g_y < 0
    yaw = np.degrees(an.yaw_of(rows[:, an.LS_QUAT]))  # counterclockwise seen from above: +
    return g, pitch, roll, yaw


def binned(t_s, values, width=0.02):
    """Means over `width` s bins (lowstate comes at 0.5-1 kHz; the tilt test works at 50 Hz)."""
    idx = ((t_s - t_s[0]) / width).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    mean = lambda v: np.bincount(idx, weights=v)[keep] / counts[keep]
    return mean(t_s), np.column_stack([mean(values[:, c]) for c in range(values.shape[1])])


def excursions(t_s, x, threshold, names):
    """Runs of one sign beyond the threshold (merged across gaps < MERGE_S): start s, peak deg, direction."""
    out = []
    for sign in (1, -1):
        idx = np.where(sign * x > threshold)[0]
        runs = []
        for k in idx:
            if runs and t_s[k] - t_s[runs[-1][1]] <= MERGE_S:
                runs[-1][1] = k
            else:
                runs.append([k, k])
        for a, b in runs:
            seg = x[a:b + 1]
            peak = float(seg[np.argmax(sign * seg)])
            out.append(dict(t_s=round(float(t_s[a]), 2), peak_deg=round(peak, 1), direction=names[sign > 0]))
    return out


def fixstand_check(ls, phase, t_start):
    t_ls = ls[:, an.LS_T]
    out = dict(phase='FixStand', start_s=round((phase['t0'] - t_start) / 1e9, 2),
               seconds=round((phase['t1'] - phase['t0']) / 1e9, 2))
    w0 = phase['t0'] + FIX_SETTLE_S * 1e9
    w1 = w0 + FIX_WINDOW_S * 1e9
    static = None
    if phase['t1'] >= w1:
        r = ls[rows_between(t_ls, w0, w1)]
        q = r[:, an.LS_Q]
        err = np.abs(q - Q_FIX).max(axis=0)
        g = an.gravity_of(r[:, an.LS_QUAT]).mean(axis=0)
        gyro = r[:, an.LS_GYRO]
        gyro_mean = gyro.mean(axis=0)
        still = float(np.percentile(np.linalg.norm(gyro, axis=1), 95))
        finite = bool(np.all(np.isfinite(q)))
        static = dict(window_s=[round((w0 - t_start) / 1e9, 2), round((w1 - t_start) / 1e9, 2)], rows=int(len(r)),
                      finite=finite, q_err_max=float(err.max()), q_err=err.tolist(),
                      worst=[int(j) for j in np.argsort(-err)[:3]], q_ok=finite and float(err.max()) < FIX_Q_ERR,
                      gravity=g.tolist(), gravity_ok=bool(np.all(np.abs(g - [0, 0, -1]) <= GRAVITY_TOL)),
                      gyro_mean=gyro_mean.tolist(), gyro_p95=still, gyro_still=still <= GYRO_STILL,
                      gyro_ok=bool(np.all(np.abs(gyro_mean) < GYRO_BIAS)))
    out['static'] = static
    # the hand tilt test: whatever FixStand recorded after the static window
    r = ls[rows_between(t_ls, w1 if static else w0, phase['t1'])]
    if len(r) < 100:
        out['tilt_test'] = None
        return out
    t_s = (r[:, an.LS_T] - t_start) / 1e9
    g, pitch, roll, yaw = angles(r)
    ref = static['gravity'] if static else g[: max(len(g) // 20, 1)].mean(axis=0)
    pitch0 = math.degrees(math.atan2(ref[0], -ref[2]))
    roll0 = math.degrees(math.atan2(-ref[1], -ref[2]))
    tb, vb = binned(t_s, np.column_stack([pitch - pitch0, roll - roll0, yaw - yaw[0], r[:, an.LS_GYRO]]))
    test = dict(window_s=[round(float(t_s[0]), 2), round(float(t_s[-1]), 2)], moves=[], gyro_sign={})
    names = dict(pitch={True: 'вперёд', False: 'назад'}, roll={True: 'вправо', False: 'влево'},
                 yaw={True: 'влево (против часовой сверху)', False: 'вправо (по часовой сверху)'})
    for k, (axis, gyro_col) in enumerate((('pitch', 4), ('roll', 3), ('yaw', 5))):  # gyro x, y, z: columns 3-5
        x = vb[:, k]
        test['moves'] += [dict(axis=axis, **e) for e in excursions(tb, x, TILT_DEG[axis], names[axis])]
        if np.ptp(x) < TILT_DEG[axis] or len(x) < 3:
            test['gyro_sign'][axis] = None
            continue
        rate = np.gradient(np.radians(x), tb)
        test['gyro_sign'][axis] = float(np.corrcoef(rate, vb[:, gyro_col])[0, 1])
    test['moves'].sort(key=lambda m: m['t_s'])
    test['ok'] = all(c is None or c > SIGN_CORR for c in test['gyro_sign'].values())
    out['tilt_test'] = test
    return out


def mean_over(t_s, values, seconds):
    """Largest mean of `values` over any `seconds` window (by time); the whole mean if shorter."""
    if len(t_s) == 0:
        return None
    if t_s[-1] - t_s[0] < seconds:
        return float(values.mean())
    c = np.r_[0, np.cumsum(values)]
    j = np.searchsorted(t_s, t_s + seconds)
    i = np.where(j < len(t_s))[0]
    return float(((c[j[i]] - c[i]) / (j[i] - i)).max())


def motors(ls, header, rows):
    """Motor temperature (columns ta<i>: temperature[0]) and error flags (ms<i>: motorstate), written only by the
    robot build of the probe: per motor start / end / largest, the three that rose fastest, motors with flags."""
    if 'ta0' not in header or len(rows) < 2:
        return None
    t = ls[rows, an.LS_T] / 1e9
    temp = ls[rows][:, [header.index(f'ta{i}') for i in range(J)]]
    minutes = max((t[-1] - t[0]) / 60, 1e-9)
    rise = (temp[-1] - temp[0]) / minutes
    out = dict(max=float(temp.max()), fastest=[dict(motor=int(m), start=float(temp[0, m]), end=float(temp[-1, m]),
                                                    per_min=float(rise[m])) for m in np.argsort(-rise)[:3]])
    if 'ms0' in header:
        flags = ls[rows][:, [header.index(f'ms{i}') for i in range(J)]]
        out['flagged'] = {int(m): int(np.max(flags[:, m])) for m in range(J) if np.any(flags[:, m] != 0)}
    return out


def velocity_check(run, ls, lc, deploy, phase, t_start, remote, replay, header):
    t_ls, t_lc = ls[:, an.LS_T], lc[:, an.LC_T]
    t_enter, t_leave = phase['t0'], phase['t1']
    out = dict(phase='Velocity', start_s=round((t_enter - t_start) / 1e9, 2),
               seconds=round((t_leave - t_enter) / 1e9, 2))
    action = deploy['actions']['JointPositionAction']
    clip = np.array(action['clip'], float)
    scale, offset = np.array(action['scale'], float), np.array(action['offset'], float)
    steps, zero_msgs = an.policy_steps(lc, t_enter, t_leave)
    out.update(policy_steps=len(steps), zero_target_messages=zero_msgs)
    q = lc[rows_between(t_lc, t_enter, t_leave)][:, an.LC_Q]
    knee = q[:, list(KNEES)]
    out['knee_target_min'] = float(knee.min()) if len(knee) else None
    out['knee_below_clip'] = int(np.sum(knee < clip[list(KNEES), 0] - 1e-6))
    out['wrist_target_max_abs'] = float(np.abs(q[:, list(LOCKED_WRISTS)]).max()) if len(q) else None
    r_idx = rows_between(t_ls, t_enter, t_leave)
    r = ls[r_idx]
    t_s = (r[:, an.LS_T] - t_enter) / 1e9
    cmd = np.array([replay.command(float(lx), float(ly), float(rx)) for lx, ly, rx in remote[r_idx]])
    norm = np.linalg.norm(cmd, axis=1) if len(cmd) else np.zeros(0)
    out['command_share'] = float(np.mean(norm >= COMMAND_ZERO)) if len(norm) else None
    out['command_max'] = cmd[np.argmax(norm)].tolist() if len(cmd) else None
    out['moving'] = out['command_share'] is not None and out['command_share'] > COMMAND_SHARE
    if steps:
        qa = lc[rows_between(t_lc, steps[0][0], t_leave)][:, an.LC_Q]
        share = ((np.abs(qa - clip[:, 0]) <= 1e-6) | (np.abs(qa - clip[:, 1]) <= 1e-6)).mean(axis=0)
        out['at_clip_share'] = {int(j): float(share[j]) for j in range(J) if share[j] > 0}
        a = (np.array([s[1] for s in steps]) - offset) / scale  # the action back from the target (clip aside)
        da = np.abs(np.diff(a, axis=0)).mean(axis=1)
        out['delta_a_mean'] = float(da.mean()) if len(da) else None
        out['delta_a_max'] = float(da.max()) if len(da) else None
    speed = np.abs(r[:, an.LS_DQ][:, LEGS]).max(axis=1)
    out['leg_speed_p95'] = float(np.percentile(speed, 95))
    out['leg_speed_max'] = float(speed.max())
    # ONNX replay and latency: analyze.py, unchanged
    lat_rows, _, periods, missed = an.latency(run, ls, lc, deploy, t_enter, t_leave, [])
    errors = np.array([x['error'] for x in lat_rows])
    matched = [x for x in lat_rows if x['error'] < ONNX_TOL]
    out['onnx'] = dict(steps=len(lat_rows), matched=len(matched), missed_by_probe=missed,
                       share=len(matched) / len(lat_rows) if lat_rows else None,
                       error_median=float(np.median(errors)) if len(errors) else None,
                       error_max_matched=max((x['error'] for x in matched), default=None))
    out['latency_ms'] = an.stats(x['obs_to_cmd_ms'] for x in matched if x['step'] > 1)
    periods = np.asarray(periods, float)
    out['period_ms'] = an.stats(periods)
    out['period_share'] = float(np.mean(np.abs(periods - PERIOD_MS) <= PERIOD_TOL_MS)) if len(periods) else None
    # standing (walking)
    _, _, _, yaw = angles(r)
    out['tilt_max_deg'] = float(np.degrees(an.tilt_of(r)).max())
    out['heading_deg'] = float(yaw[-1] - yaw[0])
    out['heading_30s_max_deg'] = None
    if t_s[-1] >= HEADING_S:
        j = np.searchsorted(t_s, t_s + HEADING_S)
        i = np.where(j < len(t_s))[0]
        out['heading_30s_max_deg'] = float(np.abs(yaw[j[i]] - yaw[i]).max())
    torque = np.abs(r[:, an.LS_TAU][:, LEGS]).sum(axis=1)
    out.update(leg_torque_mean=float(torque.mean()), leg_torque_max=float(torque.max()),
               leg_torque_2s_max=mean_over(t_s, torque, LEG_TORQUE_S))
    fast = np.where((norm < COMMAND_ZERO) & (t_s >= ENTRY_SKIP_S) & (speed > BURST_SPEED))[0]
    bursts = []
    for k in fast:
        if bursts and t_s[k] - bursts[-1][1] <= MERGE_S:
            bursts[-1][1] = t_s[k]
        else:
            bursts.append([t_s[k], t_s[k]])
    starts = np.array([b[0] for b in bursts])
    out['bursts_s'] = [round(float(b[0]), 2) for b in bursts]
    out['bursts_10s_max'] = int(max((np.sum((starts >= s) & (starts < s + BURST_WINDOW_S)) for s in starts),
                                    default=0))
    out['motors'] = motors(ls, header, r_idx)
    # verdicts: (name, text, ok, gating)
    still = not out['moving']
    v = []
    v.append(('цель колена ≥ clip', f"мин {f(out['knee_target_min'], '{:.4f}')}, ниже — {out['knee_below_clip']} "
                                    'сообщ.', out['knee_below_clip'] == 0, True))
    v.append(('запястья SDK 20, 21, 27, 28 ровно 0', f"макс |цель| {f(out['wrist_target_max_abs'], '{:.3g}')}",
              out['wrist_target_max_abs'] == 0.0, True))
    o = out['onnx']
    v.append((f'ONNX по lowstate = lowcmd (< {ONNX_TOL:g})',
              f"{o['matched']} из {o['steps']} шагов ({f(o['share'], '{:.0%}')}, нужно ≥ {ONNX_SHARE:.0%}), медиана "
              f"ошибки {f(o['error_median'], '{:.1e}')}, шагов не дошло до регистратора {o['missed_by_probe']}",
              bool(lat_rows) and o['share'] >= ONNX_SHARE, True))
    lat = out['latency_ms']
    v.append(('задержка lowstate → lowcmd',
              '—' if lat is None else f"медиана {lat['median']:.2f} мс, 95% {lat['p95']:.2f}, макс {lat['max']:.2f}"
                                      f" (нужно < {LATENCY_MEDIAN_MS:g} / < {LATENCY_P95_MS:g})",
              lat is not None and lat['median'] < LATENCY_MEDIAN_MS and lat['p95'] < LATENCY_P95_MS, True))
    p = out['period_ms']
    v.append(('период политики 20 ± 2 мс',
              '—' if p is None else f"{out['period_share']:.1%} тактов (мин {p['min']:.1f}, макс {p['max']:.1f} мс; "
                                    f"нужно ≥ {PERIOD_SHARE:.0%})",
              out['period_share'] is not None and out['period_share'] >= PERIOD_SHARE, True))
    if 'at_clip_share' in out:
        other = {j: s for j, s in out['at_clip_share'].items() if j not in KNEES}
        knees = [f'{out["at_clip_share"].get(j, 0):.0%}' for j in KNEES]
        v.append(('на пределе clip, кроме колен', (', '.join(f'SDK {j} {s:.1%}' for j, s in other.items()) or 'нет')
                  + f" (колени SDK 3 / 9: {' / '.join(knees)} сообщений)", not other, still))
        v.append(('|Δa| среднее по суставам и шагам', f"{f(out['delta_a_mean'], '{:.4f}')} (наибольший шаг "
                                                      f"{f(out['delta_a_max'], '{:.3f}')}), порог {DELTA_A:g}",
                  out['delta_a_mean'] is not None and out['delta_a_mean'] <= DELTA_A, still))
    v.append(('скорость суставов ног p95', f"{out['leg_speed_p95']:.2f} рад/с (макс {out['leg_speed_max']:.2f}), "
                                           f"порог {LEG_SPEED_P95:g}", out['leg_speed_p95'] <= LEG_SPEED_P95, still))
    tilt_limit = STAND_TILT_DEG if still else WALK_TILT_DEG
    v.append((f'наклон корпуса (порог {tilt_limit:g}°)', f"макс {out['tilt_max_deg']:.1f}°",
              out['tilt_max_deg'] <= tilt_limit, True))
    v.append((f'поворот за {HEADING_S:g} с', f"наибольший {f(out['heading_30s_max_deg'], '{:.1f}°')} (за фазу "
                                             f"{out['heading_deg']:+.1f}°), порог {HEADING_DEG:g}°",
              out['heading_30s_max_deg'] is None or out['heading_30s_max_deg'] <= HEADING_DEG, still))
    v.append((f'Σ|τ| ног, среднее за {LEG_TORQUE_S:g} с', f"наибольшее {f(out['leg_torque_2s_max'], '{:.1f}')} Н·м "
                                                         f"(среднее {out['leg_torque_mean']:.1f}, пик "
                                                         f"{out['leg_torque_max']:.1f}), порог {LEG_TORQUE_NM:g}",
              out['leg_torque_2s_max'] is None or out['leg_torque_2s_max'] <= LEG_TORQUE_NM, still))
    v.append((f'всплески скорости ног > {BURST_SPEED:g} рад/с без команды (ориентир для шагов; считать глазами)',
              f"{len(bursts)}, за любые 10 с до {out['bursts_10s_max']} (порог {BURST_MAX})",
              out['bursts_10s_max'] <= BURST_MAX, still))
    out['verdicts'] = [dict(name=n, text=t, ok=bool(k), gating=bool(g)) for n, t, k, g in v]
    return out


def passive_info(ls, phase, t_start):
    r = ls[rows_between(ls[:, an.LS_T], phase['t0'], min(phase['t1'], phase['t0'] + 2e9))]
    out = dict(phase='Passive', start_s=round((phase['t0'] - t_start) / 1e9, 2),
               seconds=round((phase['t1'] - phase['t0']) / 1e9, 2))
    if len(r):
        out.update(leg_speed_max_2s=float(np.abs(r[:, an.LS_DQ][:, LEGS]).max()),
                   tilt_max_2s_deg=float(np.degrees(an.tilt_of(r)).max()))
    return out


def patch_log_lines(patches):
    """spdlog lines the patches add: `info` ones must be in the log (startup, each Velocity entry), `warn` ones are
    events (a fall the patch caught); with regexes ({...} = anything). The first line of a multi-line call only."""
    out = []
    for patch in sorted(patches.glob('g1_ctrl-*.patch')):
        for line in patch.read_text(encoding='utf-8').splitlines():
            m = re.match(r'^\+.*spdlog::(info|warn)\("([^"]*)"', line)
            if m:
                parts = re.split(r'\{[^}]*\}', m.group(2))
                out.append(dict(patch=patch.name, kind=m.group(1), text=m.group(2),
                                regex='.*?'.join(re.escape(p) for p in parts)))
    return out


def log_check(log, patches, velocity_entries):
    if log is None or not log.exists():
        return dict(log=None)
    text = log.read_text(encoding='utf-8', errors='replace')
    found = [dict(patch=e['patch'], kind=e['kind'], text=e['text'], found=len(re.findall(e['regex'], text)))
             for e in patch_log_lines(patches)]
    lines = [x for x in found if x['kind'] == 'info']
    return dict(log=str(log), lines=lines, events=[x for x in found if x['kind'] == 'warn'],
                ok=bool(lines) and all(x['found'] > 0 for x in lines),
                policy_dirs=re.findall(r'Policy directory: (\S+)', text), velocity_entries_lowcmd=velocity_entries)


def print_report(report):
    print(f"Запись: {report['run']}")
    for note in report['notes']:
        print('  ' + note)
    print('Фазы FSM: ' + ', '.join(f"{p['name']} {p['start_s']:.1f} с ({p['seconds']:.1f} с)"
                                    for p in report['phases']))
    stops = []
    log = report['log']
    if log.get('log'):
        print(f"Лог g1_ctrl {log['log']}: строки патчей {ok(log['ok'])}")
        for x in log['lines']:
            print(f"  {x['patch']}: «{x['text']}» — найдено {x['found']}")
        for x in log['events']:
            if x['found']:
                print(f"  СОБЫТИЕ {x['patch']}: «{x['text']}» — {x['found']} раз")
        print(f"  Policy directory: {', '.join(log['policy_dirs']) or 'строки нет'}; входов в Velocity по lowcmd "
              f"{log['velocity_entries_lowcmd']}")
        if not log['ok']:
            stops.append('лог патчей')
    else:
        print('Лог g1_ctrl не найден (--log): строки патчей не проверены')
    for c in report['checks']:
        print()
        if c['phase'] == 'FixStand':
            print(f"FixStand с {c['start_s']:.1f} с, {c['seconds']:.1f} с")
            s = c['static']
            if s is None:
                print(f'  FixStand короче {FIX_SETTLE_S + FIX_WINDOW_S:g} с: замера покоя нет')
            else:
                print(f"  покой {s['window_s'][0]:.1f}–{s['window_s'][1]:.1f} с ({s['rows']} lowstate)")
                print(f"  энкодеры: все конечны — {'да' if s['finite'] else 'НЕТ'}; |q − q_fix| макс "
                      f"{s['q_err_max']:.3f} рад (хуже всех SDK {', '.join(map(str, s['worst']))}), порог "
                      f"{FIX_Q_ERR:g}: {ok(s['q_ok'])}")
                g, gm = s['gravity'], s['gyro_mean']
                print(f"  гравитация ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}), допуск ±{GRAVITY_TOL:g}: "
                      f"{ok(s['gravity_ok'])}")
                print(f"  гироскоп, среднее ({gm[0]:+.4f}, {gm[1]:+.4f}, {gm[2]:+.4f}) рад/с, порог {GYRO_BIAS:g}: "
                      f"{ok(s['gyro_ok'])}; |gyro| p95 {s['gyro_p95']:.3f}"
                      + ('' if s['gyro_still'] else f' > {GYRO_STILL:g}: робот качался, замер повторить'))
                failed = [name for name, flag in (('энкодеры', s['q_ok']), ('гравитация', s['gravity_ok']),
                                                  ('гироскоп', s['gyro_ok'])) if not flag]
                if failed and not s['gyro_still']:
                    stops.append(f"FixStand {c['start_s']:.0f} с: робот качался — замер покоя повторить "
                                 f"({', '.join(failed)})")
                else:
                    stops += [f"FixStand {c['start_s']:.0f} с: {name}" for name in failed]
            t = c['tilt_test']
            if t is None:
                print('  наклонов рукой нет (после окна покоя FixStand кончился)')
                continue
            print(f"  наклоны рукой {t['window_s'][0]:.1f}–{t['window_s'][1]:.1f} с (от позы покоя):")
            axis = dict(pitch='тангаж', roll='крен', yaw='рысканье')
            for m in t['moves']:
                print(f"    {m['t_s']:.1f} с: {axis[m['axis']]} {m['direction']}, до {m['peak_deg']:+.1f}°")
            if not t['moves']:
                print('    наклонов больше 3° (рысканье 5°) нет')
            sign = ', '.join(f"{axis[k]} {f(v, '{:+.2f}')}" for k, v in t['gyro_sign'].items())
            print(f"  знак гироскопа (корреляция d(угол)/dt с gyro_y / gyro_x / gyro_z, нужно > {SIGN_CORR:g}): "
                  f"{sign}: {ok(t['ok'])}")
            print('  сверить с тем, что делали: наклон вперёд даёт g_x > 0, вправо (правым боком вниз) — g_y < 0')
            if not t['ok']:
                stops.append(f"FixStand {c['start_s']:.0f} с: знак гироскопа")
        elif c['phase'] == 'Velocity':
            cm = c['command_max']
            print(f"Velocity с {c['start_s']:.1f} с, {c['seconds']:.1f} с; шагов политики {c['policy_steps']}, "
                  f"сообщений с нулевой целью {c['zero_target_messages']}; команда со стика ≥ {COMMAND_ZERO:g} — "
                  f"{f(c['command_share'], '{:.0%}')} времени, наибольшая "
                  + ('—' if cm is None else f"({cm[0]:+.2f}, {cm[1]:+.2f}, {cm[2]:+.2f})")
                  + (' — фаза с командой: пределы «в воздухе» и «стоя» справочно' if c['moving'] else ''))
            for x in c['verdicts']:
                mark = ok(x['ok']) if x['gating'] else ('ок' if x['ok'] else 'выше порога') + ', справочно'
                print(f"  {x['name']}: {x['text']}: {mark}")
                if x['gating'] and not x['ok']:
                    stops.append(f"Velocity {c['start_s']:.0f} с: {x['name']}")
            m = c['motors']
            if m is None:
                print('  температура моторов: в записи нет (регистратор без правки для робота, см. протокол)')
            else:
                print(f"  температура моторов: макс {m['max']:.0f}; быстрее всех растут " + ', '.join(
                    f"SDK {x['motor']} {x['start']:.0f} → {x['end']:.0f} ({x['per_min']:+.1f} в мин)"
                    for x in m['fastest']) + ' (пороги — Unitree)')
                if m.get('flagged'):
                    print('  флаги ошибок моторов (motorstate ≠ 0): ' + ', '.join(
                        f'SDK {k}: {v}' for k, v in m['flagged'].items()) + ' — смотреть документацию Unitree')
        else:
            print(f"Passive с {c['start_s']:.1f} с: за первые 2 с скорость ног макс "
                  f"{f(c.get('leg_speed_max_2s'), '{:.2f}')} рад/с, наклон макс "
                  f"{f(c.get('tilt_max_2s_deg'), '{:.1f}')}° (справочно)")
    print()
    print('ИТОГ: ' + ('все проверки ок' if not stops else 'СТОП — ' + '; '.join(stops)))
    report['stops'] = stops


def main():
    parser = argparse.ArgumentParser(description='Проверка записи с G1 по протоколу первого запуска')
    parser.add_argument('run', type=Path, help='папка записи (lowstate.csv, lowcmd.csv)')
    parser.add_argument('--policy', type=Path, help='папка политики из пакета (policy.onnx, deploy.yaml)')
    parser.add_argument('--log', type=Path, help='вывод g1_ctrl (по умолчанию <папка записи>/g1_ctrl.log)')
    parser.add_argument('--patches', type=Path, help='папка с g1_ctrl-*.patch (по умолчанию — из пакета)')
    parser.add_argument('--json', type=Path, help='итог в JSON (по умолчанию <папка записи>/check.json)')
    args = parser.parse_args()
    run = args.run
    notes = prepare(run, args.policy)
    deploy = yaml.safe_load((run / 'deploy.yaml').read_text(encoding='utf-8'))
    ls, lc = an.table(run, 'lowstate.csv'), an.table(run, 'lowcmd.csv')
    with (run / 'lowstate.csv').open(encoding='utf-8') as fh:
        header = fh.readline().strip().split(',')
    if len(np.unique(ls[:, an.LS_TICK])) < 0.1 * len(ls):
        # analyze.latency takes a new `tick` as a new state; without a usable tick every message is one
        ls[:, an.LS_TICK] = np.arange(len(ls))
        notes.append('tick в lowstate почти не меняется: каждое сообщение считается новым состоянием')
    t_start = min(ls[0, an.LS_T], lc[0, an.LC_T])
    patches = args.patches or next((p for p in (HERE.parent / 'patches', HERE.parent / 'sim2sim' / 'patches')
                                    if p.exists()), HERE)
    replay = an.Replay(run, deploy)
    remote = an.smooth_remote(ls)
    report = dict(run=str(run), notes=notes, phases=[], checks=[])
    found = phases(lc, deploy)
    for p in found:
        report['phases'].append(dict(name=p['name'], start_s=round((p['t0'] - t_start) / 1e9, 2),
                                     seconds=round((p['t1'] - p['t0']) / 1e9, 2)))
        if p['name'] == 'FixStand':
            report['checks'].append(fixstand_check(ls, p, t_start))
        elif p['name'] == 'Velocity':
            report['checks'].append(velocity_check(run, ls, lc, deploy, p, t_start, remote, replay, header))
        elif p['name'] == 'Passive' and report['checks'] and report['checks'][-1]['phase'] == 'Velocity':
            report['checks'].append(passive_info(ls, p, t_start))
    report['log'] = log_check(args.log or run / 'g1_ctrl.log', patches, sum(p['name'] == 'Velocity' for p in found))
    print_report(report)
    out = args.json or run / 'check.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'итог: {out}')


if __name__ == '__main__':
    main()
