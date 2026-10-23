"""Knees and stops of sim2sim runs (unitree_mujoco + g1_ctrl): what analyze.py does not report.

Usage: python knees_stops.py [--print-only] <run dir>...    (run dirs of run.sh; $ROOT/venv: numpy, pyyaml)

Knees (SDK 3 left, 9 right): the joint angle from lowstate and the motor target from lowcmd, standing and
moving, within the first Velocity entry:
  stand        from 1 s after the band is off to the first command (the `stand` scenario: to the end);
  after_stop   every zero-command segment that follows a moving one, from 1 s after the stop;
  moving       the walk after its first 2 s, other moving segments after their settle time (as analyze.py).
At the stop: angle below KNEE_STOP (the joint's lower limit -0.087 in unitree_mujoco's g1_29dof.xml, within
0.012 rad: the critic's knee_stop.py measure). At the floor: target at the deploy.yaml clip's lower bound
(knee_floor 0.061 for candidate 4), i.e. g1_ctrl clipped the policy's target.

Stops: every stop after a moving command (the walk's `stop` event, zero-command `seg` events after a moving
one): displacement over the first STOP_WINDOW s, the part after the first STOP_BRAKE s (braking done), its
largest excursion from the pose at STOP_BRAKE and peak pelvis speed there, heading change, falls. A lurch is
an excursion of LURCH_M or more after the braking (the copy of the bridge showed 0.2-0.4 m with no command).
Writes knees_stops.json into each run dir (not with --print-only: earlier runs stay as they were) and prints a
summary in Russian.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import FALL_TILT, FALL_Z, SETTLE, gravity_of, scenario_events, scenario_segments, yaw_of  # noqa: E402

KNEES = (3, 9)  # SDK indices of the left / right knee
KNEE_LIMIT = -0.087267  # rad, lower range of both knees in unitree_mujoco's g1_29dof.xml
KNEE_STOP = KNEE_LIMIT + 0.012  # rad, angle below which the knee counts as at its stop (knee_stop.py: -0.075)
FLOOR_TOL = 1e-6  # rad, a target this close to the clip's lower bound is at the floor
STOP_WINDOW = 3.0  # s after a stop measured
STOP_BRAKE = 1.0  # s after a stop the robot needs to brake (the stick is smoothed 0.1-0.2 s)
LURCH_M = 0.1  # m, excursion after the braking counted as a lurch


def load(run):
    events, timeline = scenario_events(run)
    deploy = yaml.safe_load((run / 'deploy.yaml').read_text(encoding='utf-8'))
    # lowstate.csv: t_ns, tick, q0..q28 (2..30), dq, tau, quat (89..92); lowcmd.csv: t_ns, q0..q28 (1..29), kp0 (30)
    ls = np.loadtxt(run / 'lowstate.csv', delimiter=',', skiprows=1, ndmin=2,
                    usecols=(0, 2 + KNEES[0], 2 + KNEES[1], 89, 90, 91, 92))
    lc = np.loadtxt(run / 'lowcmd.csv', delimiter=',', skiprows=1, ndmin=2,
                    usecols=(0, 1 + KNEES[0], 1 + KNEES[1], 30))
    sport = np.loadtxt(run / 'sport.csv', delimiter=',', skiprows=1, ndmin=2)
    return events, timeline, deploy, ls, lc, sport


def velocity_span(lc, deploy):
    """First Velocity entry (motor 0's kp at deploy.yaml's) to its end, lowcmd ns."""
    on = np.round(lc[:, 3], 3) == round(float(deploy['stiffness'][0]), 3)
    rows = np.where(on)[0]
    if len(rows) == 0:
        return None
    s = rows[0]
    e = s
    while e < len(lc) and on[e]:
        e += 1
    return lc[s, 0], (lc[e, 0] if e < len(lc) else math.inf)


def windows(events, timeline):
    """(stand, after_stop, moving) lists of (from, to) ns, and the stops: (start, end, previous command)."""
    band_off = events['band_off'][0]
    stand, after, moving, stops = [], [], [], []
    if 'walk' in events:  # the walk scenario: stand, walk, stop
        walk, stop, end = events['walk'][0], events['stop'][0], events['end'][0]
        stand.append((band_off + 1e9, walk))
        moving.append((walk + SETTLE * 1e9, stop))
        after.append((stop + STOP_BRAKE * 1e9, end))
        stops.append((stop, end, [float(events['walk'][1]), 0., 0.]))
        return stand, after, moving, stops
    segs = scenario_segments(timeline)
    for k, s in enumerate(segs):
        still = not any(s['command'])
        if k == 0 and still and s['name'] in ('stand', 'stand_again'):
            stand.append((band_off + 1e9, s['end']))
            continue
        if still:
            previous = segs[k - 1]['command'] if k > 0 else [0., 0., 0.]
            if any(previous):
                after.append((s['start'] + STOP_BRAKE * 1e9, s['end']))
                stops.append((s['start'], s['end'], previous))
            continue
        settle = SETTLE if (s['end'] - s['start']) / 1e9 >= 8 else 1.0
        moving.append((s['start'] + settle * 1e9, s['end']))
    return stand, after, moving, stops


def mask(t, spans, limit):
    inside = np.zeros(len(t), bool)
    for a, b in spans:
        inside |= (t >= a) & (t < min(b, limit))
    return inside


def knee_stats(ls, lc, spans, floor, limit):
    """Per knee within the spans: angle (mean, min, share at the stop), target (mean, min, share at the floor)."""
    a, c = mask(ls[:, 0], spans, limit), mask(lc[:, 0], spans, limit)
    if not a.any() or not c.any():
        return None
    out = dict(seconds=sum((min(b, limit) - s) / 1e9 for s, b in spans if min(b, limit) > s))
    for k, name in enumerate(('left', 'right')):
        q, target = ls[a, 1 + k], lc[c, 1 + k]
        out[name] = dict(q_mean=float(q.mean()), q_min=float(q.min()), q_max=float(q.max()),
                         at_stop=float(np.mean(q < KNEE_STOP)), target_mean=float(target.mean()),
                         target_min=float(target.min()), at_floor=float(np.mean(target <= floor + FLOOR_TOL)))
    return out


def lying_at(sport, ls, t):
    """The robot counts as fallen at t: pelvis below FALL_Z or tilted past FALL_TILT (as analyze.py)."""
    z = np.interp(t, sport[:, 0], sport[:, 3])
    i = max(np.searchsorted(ls[:, 0], t, side='right') - 1, 0)
    tilt = math.acos(min(max(-gravity_of(ls[i:i + 1, 3:7])[0, 2], -1.), 1.))
    return bool(z < FALL_Z or tilt > FALL_TILT)


def stop_report(sport, ls, start, end, command):
    ts = sport[:, 0]
    w_end = min(start + STOP_WINDOW * 1e9, end)
    brake = start + STOP_BRAKE * 1e9

    def pos(t):
        return np.array([np.interp(t, ts, sport[:, 1]), np.interp(t, ts, sport[:, 2])])

    yaw = yaw_of(ls[:, 3:7])
    h0 = float(np.interp(start, ls[:, 0], yaw))
    d = pos(w_end) - pos(start)
    inside = (ts >= brake) & (ts <= w_end)
    excursion = np.linalg.norm(sport[inside, 1:3] - pos(brake), axis=1)
    speed = np.linalg.norm(sport[inside, 4:6], axis=1)
    whole = (ts >= brake) & (ts <= end)
    excursion_all = np.linalg.norm(sport[whole, 1:3] - pos(brake), axis=1)
    tilt = np.arccos(np.clip(-gravity_of(ls[:, 3:7])[:, 2], -1, 1))
    low = np.where((ts >= start) & (ts <= end) & (sport[:, 3] < FALL_Z))[0]
    tipped = np.where((ls[:, 0] >= start) & (ls[:, 0] <= end) & (tilt > FALL_TILT))[0]
    falls = ([ts[low[0]]] if len(low) else []) + ([ls[tipped[0], 0]] if len(tipped) else [])
    return dict(command=command, seconds=(w_end - start) / 1e9, segment_s=(end - start) / 1e9,
                moved_m=float(np.linalg.norm(d)), forward_m=float(d[0] * math.cos(h0) + d[1] * math.sin(h0)),
                left_m=float(-d[0] * math.sin(h0) + d[1] * math.cos(h0)),
                after_brake_m=float(np.linalg.norm(pos(w_end) - pos(brake))),
                excursion_m=float(excursion.max()) if len(excursion) else None,
                excursion_segment_m=float(excursion_all.max()) if len(excursion_all) else None,
                peak_speed=float(speed.max()) if len(speed) else None,
                heading_deg=math.degrees(float(np.interp(w_end, ls[:, 0], yaw)) - h0),
                pelvis_z_min=float(sport[(ts >= start) & (ts <= end), 3].min()),
                fell_after_s=None if not falls else round((min(falls) - start) / 1e9, 2))


def command_text(c):
    vx, vy, wz = c
    if vy and not vx and not wz:
        return f"шаг {'влево' if vy > 0 else 'вправо'} {abs(vy):g} м/с"
    if wz and not vy:
        return f"поворот {wz:+g} рад/с" + (f" на ходу {vx:g} м/с" if vx else ' на месте')
    return f"вперёд {vx:g} м/с"


def main():
    args = sys.argv[1:]
    write = '--print-only' not in args
    for run in (Path(a) for a in args if a != '--print-only'):
        events, timeline, deploy, ls, lc, sport = load(run)
        span = velocity_span(lc, deploy)
        if span is None:
            print(f'{run.name}: g1_ctrl не вошёл в Velocity')
            continue
        t_enter, t_leave = span
        floor = float(deploy['actions']['JointPositionAction']['clip'][KNEES[0]][0])
        in_velocity = (lc[:, 0] >= t_enter) & (lc[:, 0] < t_leave)
        targets = lc[in_velocity][:, 1:3]
        below = np.where(np.any(targets < floor - FLOOR_TOL, axis=1))[0]
        stand, after, moving, stops = windows(events, timeline)
        # a stop that finds the robot already lying (push --push-walking after a fall) is no stop
        lying = [lying_at(sport, ls, a) for a, _, _ in stops]
        after = [w for w, down in zip(after, lying) if not down]
        report = dict(run=run.name, floor=floor, knee_stop=KNEE_STOP,
                      velocity=dict(messages=int(in_velocity.sum()), target_min=targets.min(axis=0).tolist(),
                                    below_floor=int(np.sum(targets < floor - FLOOR_TOL)),
                                    # (s after the Velocity entry, left, right) of each message below the floor
                                    below_messages=[((lc[in_velocity][i, 0] - t_enter) / 1e9, *targets[i].tolist())
                                                    for i in below]),
                      knees={name: knee_stats(ls, lc, spans, floor, t_leave)
                             for name, spans in (('stand', stand), ('after_stop', after), ('moving', moving))},
                      stops=[dict(command=c, lying=True) if down else stop_report(sport, ls, a, b, c)
                             for (a, b, c), down in zip(stops, lying) if a < t_leave])
        if write:
            (run / 'knees_stops.json').write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                  encoding='utf-8')
        v = report['velocity']
        print(f"{run.name}: нижняя граница цели колена в deploy.yaml {floor:g}; в Velocity {v['messages']} lowcmd, "
              f"цель колена не ниже {v['target_min'][0]:.4f} / {v['target_min'][1]:.4f} рад (левое / правое), "
              f"ниже границы — {v['below_floor']}"
              + ''.join(f"; {t:.3f} с после входа: {a:+.3f} / {b:+.3f}" for t, a, b in v['below_messages']))
        names = dict(stand='стоя после ленты', after_stop='стоя после стопа', moving='на ходу')
        for key, k in report['knees'].items():
            if not k:
                continue
            left, right = k['left'], k['right']
            print(f"  колени {names[key]} ({k['seconds']:.1f} с): угол {math.degrees(left['q_mean']):+.1f}° / "
                  f"{math.degrees(right['q_mean']):+.1f}° (мин {left['q_min']:+.3f} / {right['q_min']:+.3f} рад), "
                  f"в упоре (< {KNEE_STOP:.3f}) {left['at_stop']:.0%} / {right['at_stop']:.0%}; цель "
                  f"{left['target_mean']:+.3f} / {right['target_mean']:+.3f}, на границе {left['at_floor']:.0%} / "
                  f"{right['at_floor']:.0%}")
        for s in report['stops']:
            if s.get('lying'):
                print(f"  стоп после «{command_text(s['command'])}»: робот уже лежит, не в счёт")
                continue
            fell ='' if s['fell_after_s'] is None else f", ПАДЕНИЕ через {s['fell_after_s']:.2f} с"
            lurch = s['excursion_m'] is not None and s['excursion_m'] >= LURCH_M
            print(f"  стоп после «{command_text(s['command'])}»: за {s['seconds']:.1f} с {s['moved_m']:.2f} м "
                  f"(вперёд {s['forward_m']:+.2f}, влево {s['left_m']:+.2f}), курс {s['heading_deg']:+.1f}°; "
                  f"после {STOP_BRAKE:g} с: {s['after_brake_m']:.3f} м, отход до {s['excursion_m']:.3f} м "
                  f"(за весь стоп {s['excursion_segment_m']:.3f} м за {s['segment_s']:.1f} с), скорость таза до "
                  f"{s['peak_speed']:.2f} м/с, таз ≥ {s['pelvis_z_min']:.3f} м"
                  + ('; РЫВОК' if lurch else '; рывка нет') + fell)


if __name__ == '__main__':
    main()
