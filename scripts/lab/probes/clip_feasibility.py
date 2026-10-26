"""CPU only. clip_feasibility.py <clip> [<clip> …] [--touch 0.01] [--timeline 0.1] [--workers 4] [--json <file>]
Is a g1_getup track clip (runs/humanoid_lab/getup_track/<clip>.npz) physically doable for the G1 before any training
(iteration 67; the critic's lab-critic/it67 feas67.py part A, static67.py, support67.py, cap67.py)? No physics run, per
50 Hz clip frame, in the getting-up model:

- inverse dynamics M(q)·q̈ + c(q, q̇) with the constraints off (q̈ the clip's central differences of qvel, averaged over
  ±ACCEL_SMOOTH frames), balanced by floor forces at the clip's own touching points: every MuJoCo contact of a robot geom
  with the floor within --touch (1 cm), a friction pyramid of FRICTION (soles 0.6, any other geom 1.0, the critic's),
  the forces ≥ 0 by non-negative least squares (Lawson-Hanson), the root's residual near hard (RESIDUAL_FORCE of m·g or
  m·g·RESIDUAL_ARM of moment costs as much as TORQUE_COST times every motor at its peak), then the joint torques;
- the residual: what a harness on the pelvis must give, its force over m·g and its moment over m·g (m);
- the centre of mass over the support: its signed distance to the convex hull of the touching points (+ inside, m);
- the torque needed against the torque available in its direction: the motor's peak at the clip's joint speed
  (g1_tensor_math.motor_torque's curve, y1 / y2 falling from x1 to x2), the PD's limit at the stop, kp·(limit − q) −
  kd·q̇ (a target cannot pass the joint range; the knee's floor 0.061), less the motor model's joint friction; the
  locked wrists' (target home) are printed, not judged;
- the floor's normal force on each hand (the wrist links' geoms: Unitree's rubber hand capsules), N;
- the noise of the accelerations: RMS of the raw q̈ about its average over ±ACCEL_SMOOTH frames (LAFAN 8, the CMU clips
  of iteration 64-66 54-76 rad/s²), and the fastest joint.

Verdict «годен» (the lead's criteria, iteration 67): the centre of mass at most COM_OUTSIDE (5 cm) outside the support,
the residual at most RESIDUAL_FORCE (0.05 m·g) and moment m·g·RESIDUAL_ARM (5 cm) in every frame, no driven joint
needing more than it has (ratio ≤ 1). Prints a summary per clip, side by side, and a timeline every --timeline seconds
(support by body group, the centre of mass margin, the residual, the worst joint, the hands' loads); --json writes
every frame. About 2 s a clip.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / 'scripts' / 'lab'))

TOUCH = .01  # m: a robot geom this close to the floor touches it
FRICTION = dict(sole=.6, other=1.)  # the friction pyramid's coefficient: the sole spheres, any other robot geom
ACCEL_SMOOTH = 2  # frames: q̈ is averaged over ±ACCEL_SMOOTH before the inverse dynamics (the critic's)
RESIDUAL_FORCE = .05  # of m·g: the largest residual force of a doable frame
RESIDUAL_ARM = .05  # m: the largest residual moment of a doable frame, over m·g
COM_OUTSIDE = .05  # m: the centre of mass at most this far outside the support
TORQUE_COST = .1  # a motor at what it has costs TORQUE_COST² against a residual at its limit's 1
TORQUE_SLACK = .02  # of the peak: a joint may need this much more than it has (its PD's error at a stop, rounding)
TORQUE_PASSES = 3  # the forces solved again with each joint's cost by what it has in the direction last asked for
FORCE_COST = 1e-3  # per m·g: the floor forces' own small cost (the solution with the least force)
HAND_LOAD = 150.  # N on one hand: the most a rubber hand should carry (the lead, iteration 67)
WAIST_FREE = ('waist',)  # joints (name parts) the second verdict leaves out of the torque rule
GROUPS = (('ankle', 'foot'), ('wrist', 'hand'), ('knee', 'knee'), ('elbow', 'elbow'), ('shoulder', 'arm'),
          ('hip', 'thigh'))


def nonnegative_least_squares(a, b, tolerance=1e-10):
    """x ≥ 0 minimizing |a x − b|² (Lawson and Hanson's active set; a few hundred unknowns at most)."""
    n = a.shape[1]
    active = np.zeros(n, bool)
    x = np.zeros(n)
    gradient = a.T @ (b - a @ x)
    for _ in range(3 * n + 10):
        if active.all() or gradient[~active].max() <= tolerance:
            break
        active[np.argmax(np.where(active, -np.inf, gradient))] = True
        while True:
            z = np.zeros(n)
            z[active] = np.linalg.lstsq(a[:, active], b, rcond=None)[0]
            if (z[active] > tolerance).all():
                break
            blocked = active & (z <= tolerance)
            step = np.min(x[blocked] / np.maximum(x[blocked] - z[blocked], 1e-300))
            x = x + step * (z - x)
            active &= x > tolerance
            x[~active] = 0.
            if not active.any():
                z = x
                break
        x = z
        gradient = a.T @ (b - a @ x)
    return x


def hull(points):
    """The convex hull (counter-clockwise, (H, 2)) of 2-D points (Andrew's monotone chain); fewer than 3 distinct
    points come back as they are."""
    points = sorted(set(map(tuple, np.round(np.asarray(points, float), 5))))
    if len(points) < 3:
        return np.array(points).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def margin(point, polygon):
    """The signed distance (m) of a 2-D point to a convex polygon (hull's): + inside, − outside; a segment or a point
    has no inside; no polygon: −inf."""
    point = np.asarray(point, float)
    if len(polygon) == 0:
        return -math.inf

    def to_segment(a, b):
        e = b - a
        t = np.clip(np.dot(point - a, e) / max(np.dot(e, e), 1e-12), 0., 1.)
        return float(np.linalg.norm(point - (a + t * e)))
    if len(polygon) == 1:
        return -float(np.linalg.norm(point - polygon[0]))
    if len(polygon) == 2:
        return -to_segment(polygon[0], polygon[1])
    inside, best = True, math.inf
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        e = b - a
        if e[0] * (point[1] - a[1]) - e[1] * (point[0] - a[0]) < 0:  # right of a counter-clockwise edge
            inside = False
        best = min(best, to_segment(a, b))
    return best if inside else -best


def group(name):
    """The body group of a body name (foot, hand, knee, elbow, arm, thigh, trunk)."""
    for key, value in GROUPS:
        if key in name:
            return value
    return 'trunk'


def available_torque(env, q, qd):
    """(up, down) ≥ 0 per joint (29,): the most torque the PD and the motor model give in the + and the − direction at
    angle q and speed qd: min(the motor curve's peak at qd, the PD's at its target's limit) less the joint friction
    (g1_tensor_math.motor_torque's terms)."""
    x1, x2, y1, y2, static, viscous = (np.asarray(row, float) for row in env.motors)
    speed = np.abs(qd)

    def peak(same):
        value = np.where(same, y1, y2)
        return np.where(speed < x1, value, np.clip(value - value / (x2 - x1) * (speed - x1), 0., None))
    friction = static * np.tanh(qd / .01) + viscous * qd
    pd_up = env.kp * (env.target_high - q) - env.kd * qd
    pd_down = env.kp * (q - env.target_low) + env.kd * qd
    up = np.minimum(peak(qd > 0), pd_up) - friction
    down = np.minimum(peak(qd < 0), pd_down) + friction
    return up, down


def accelerations(qvel, dt, frames=ACCEL_SMOOTH):
    """(raw q̈ (K, 35) by central differences of qvel, the ends 0; the same averaged over ±frames)."""
    k = len(qvel)
    raw = np.zeros_like(qvel)
    raw[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)
    index = np.clip(np.arange(k)[:, None] + np.arange(-frames, frames + 1)[None], 0, k - 1)
    return raw, raw[index].mean(1)


def frame_support(model, data, env, floor, robot, soles, touch):
    """[(world point (3,), body, friction coefficient, geom)] of the robot geoms' floor contacts within touch (after
    mj_forward with the floor's margin at touch)."""
    out = []
    for c in range(data.ncon):
        contact = data.contact[c]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if floor not in (g1, g2) or contact.dist > touch:
            continue
        g = g2 if g1 == floor else g1
        if not robot[g]:
            continue
        mu = FRICTION['sole'] if g in soles else FRICTION['other']
        out.append((contact.pos.copy(), int(model.geom_bodyid[g]), mu, g))
    return out


class FrameChecker:
    """The per-frame check on env (a G1VelocityEnv of the getting-up model): its own model copies (the floor's margin
    at touch; the constraints off for the inverse dynamics)."""

    def __init__(self, env, touch=TOUCH):
        import copy
        import mujoco
        self.env, self.touch = env, touch
        self.model = copy.deepcopy(env.model)
        self.data = mujoco.MjData(self.model)
        self.floor = self.model.geom('floor').id
        self.model.geom_margin[self.floor] = touch
        self.inverse_model = copy.deepcopy(env.model)
        self.inverse_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.inverse_data = mujoco.MjData(self.inverse_model)
        self.robot = np.asarray(env.robot_geoms, bool)
        self.soles = {int(g) for g in np.asarray(env.soles).ravel()}
        self.names = [self.model.joint(j).name[:-6] for j in env.joints]
        self.peak = np.asarray(env.motors[2], float)
        self.locked = env.wrist_mask.numpy()
        self.qadr, self.vadr = np.asarray(env.qadr), np.asarray(env.vadr)
        self.hand_sides = [{b for b in range(self.model.nbody)
                            if 'wrist' in self.model.body(b).name and self.model.body(b).name.startswith(side)}
                           for side in ('left', 'right')]
        self.weight = float(mujoco.mj_getTotalmass(self.model)) * 9.81
        self.root_scale = np.array([1 / (RESIDUAL_FORCE * self.weight)] * 3
                                   + [1 / (RESIDUAL_ARM * self.weight)] * 3)
        self.jacp = np.zeros((3, self.model.nv))

    def frame(self, qpos, qvel=None, qacc=None, lift=0.):
        """One frame: qpos (36,), qvel / qacc (35,) (None: still); the support taken with the pose lift m lower. A dict:
        com_margin, residual_force (of m·g), residual_moment (/ m·g, m), torque, have (what each joint has in the
        torque's direction), ratio (|torque| / (have + slack)), hands (N, left / right), load (N per body group),
        support [(point, body)], forces."""
        import mujoco
        model, data, env = self.model, self.data, self.env
        qvel = np.zeros(model.nv) if qvel is None else qvel
        qacc = np.zeros(model.nv) if qacc is None else qacc
        inverse = self.inverse_data
        inverse.qpos[:], inverse.qvel[:], inverse.qacc[:] = qpos, qvel, qacc
        mujoco.mj_inverse(self.inverse_model, inverse)
        needed = inverse.qfrc_inverse.copy()
        row = np.array(qpos, float)
        row[2] -= lift
        data.qpos[:], data.qvel[:] = row, 0.
        mujoco.mj_forward(model, data)
        support = frame_support(model, data, env, self.floor, self.robot, self.soles, self.touch)
        columns, owners = [], []
        for point, body, mu, _ in support:
            mujoco.mj_jac(model, data, self.jacp, None, point, body)
            for tx, ty in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                columns.append(self.jacp.T @ np.array([mu * tx, mu * ty, 1.]))  # unit normal force per column
                owners.append(body)
        q, qd = row[self.qadr], qvel[self.vadr]
        up, down = available_torque(env, q, qd)
        peak, locked = self.peak, self.locked
        slack = TORQUE_SLACK * peak
        scale = np.where(locked, peak, np.maximum(np.maximum(up, down), slack))
        if columns:
            a = np.array(columns).T
            for _ in range(TORQUE_PASSES):  # each joint's cost by what it has in the direction it was asked for
                system = np.vstack([a[:6] * self.root_scale[:, None], TORQUE_COST * a[6:] / scale[:, None],
                                    FORCE_COST / self.weight * np.eye(a.shape[1])])
                target = np.concatenate([needed[:6] * self.root_scale, TORQUE_COST * needed[6:] / scale,
                                         np.zeros(a.shape[1])])
                forces = nonnegative_least_squares(system, target)
                torque = needed[6:] - a[6:] @ forces
                scale = np.where(locked, peak, np.maximum(np.where(torque >= 0, up, down), slack))
            generalized = a @ forces
        else:
            forces, generalized = np.zeros(0), np.zeros(model.nv)
        residual = needed[:6] - generalized[:6]
        torque = needed[6:] - generalized[6:]
        have = np.where(torque >= 0, up, down)
        ratio = np.abs(torque) / (np.maximum(have, 0.) + slack)  # > 1: more than it has, beyond the slack
        load, hands = {}, [0., 0.]
        for body, f in zip(owners, forces):
            kind = group(model.body(body).name)
            load[kind] = load.get(kind, 0.) + f
            for side, bodies in enumerate(self.hand_sides):
                if body in bodies:
                    hands[side] += f
        com = data.subtree_com[env.pelvis][:2].copy()
        polygon = hull([p[:2] for p, *_ in support]) if support else np.zeros((0, 2))
        return dict(com_margin=margin(com, polygon), residual_force=float(np.linalg.norm(residual[:3]) / self.weight),
                    residual_moment=float(np.linalg.norm(residual[3:]) / self.weight), torque=torque, have=have,
                    ratio=ratio, hands=hands, load=load, support=[(p, b) for p, b, *_ in support], forces=forces,
                    qd=qd)


def check_clip(clip, env=None, touch=TOUCH, lower=True, static=False):
    """Per frame and in summary, the feasibility of a clip (g1_getup_track.load's dict or one with qpos (K, 36), qvel
    (K, 35)) in env (getup_model_env's; made when None). lower: each frame's support is taken with its lowest robot
    geom put on the floor (a clip floating a few cm, as LAFAN's after its lift, still has a support; the float is
    reported); False: as it is. static: the poses held still (q̇ = q̈ = 0; the noise is still the clip's). Returns a
    dict: rows (per frame), summary (with the verdict), names, weight, touch."""
    import cmu_mocap
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    env = getup_model_env() if env is None else env
    floor_model, floor_data, floor_geoms = cmu_mocap._floor_model(env)
    lowest = np.array([cmu_mocap.lowest_point(floor_model, floor_data, row, floor_geoms) for row in clip['qpos']])
    checker = FrameChecker(env, touch)
    names, peak, locked, weight = checker.names, checker.peak, checker.locked, checker.weight
    qpos, qvel = np.asarray(clip['qpos'], float), np.asarray(clip['qvel'], float)
    dt = env.dt
    raw, qacc = accelerations(qvel, dt)
    noise = float(np.sqrt(np.mean((raw[1:-1, 6:] - qacc[1:-1, 6:]) ** 2)))
    accel = (float(np.sqrt(np.mean(qacc[:, 6:] ** 2))), float(np.abs(qacc[:, 6:]).max()))
    if static:
        qvel, qacc = np.zeros_like(qvel), np.zeros_like(qacc)
    rows = []
    for k in range(len(qpos)):
        f = checker.frame(qpos[k], qvel[k], qacc[k], lowest[k] if lower else 0.)
        torque, have, ratio, qd = f['torque'], f['have'], f['ratio'], f['qd']
        driven = np.where(locked, 0., ratio)
        worst = int(np.argmax(driven))
        rows.append(dict(
            t=round(k * dt, 3), pelvis=float(qpos[k, 2]), com_margin=f['com_margin'],
            residual_force=f['residual_force'], residual_moment=f['residual_moment'],
            worst=names[worst], worst_ratio=float(driven.max()), worst_torque=float(torque[worst]),
            worst_have=float(have[worst]), peak_ratio=float(np.max(np.where(locked, 0., np.abs(torque) / peak))),
            locked_ratio=float(np.max(np.where(locked, np.abs(torque) / peak, 0.))), float=float(lowest[k]),
            hands=[round(h, 1) for h in f['hands']],
            load={key: round(v / weight, 2) for key, v in f['load'].items() if v > 1.},
            torque=torque.round(2).tolist(), have=have.round(2).tolist(), qd_max=float(np.abs(qd).max()),
            qd_joint=names[int(np.argmax(np.abs(qd)))]))
    summary = summarize(rows, names, noise, *accel, locked, peak)
    summary.update(float_max=float(lowest.max()), float_frames=int((lowest > touch).sum()), lowered=bool(lower),
                   static=bool(static))
    return dict(rows=rows, summary=summary, names=names, weight=weight, touch=touch)


def summarize(rows, names, noise, accel_rms, accel_max, locked, peak):
    """The clip's summary of check_clip's rows and its verdict (the lead's criteria)."""
    margin_ = np.array([r['com_margin'] for r in rows])
    force = np.array([r['residual_force'] for r in rows])
    moment = np.array([r['residual_moment'] for r in rows])
    ratio = np.array([r['worst_ratio'] for r in rows])
    times = np.array([r['t'] for r in rows])
    torque, have = np.array([r['torque'] for r in rows]), np.array([r['have'] for r in rows])
    over = (np.abs(torque) > np.maximum(have, 0.) + TORQUE_SLACK * peak[None]) & ~locked[None]
    over_joints = {}
    for j in np.flatnonzero(over.any(0)):
        frames = np.flatnonzero(over[:, j])
        over_joints[names[j]] = dict(frames=int(len(frames)), start=float(times[frames[0]]),
                                     end=float(times[frames[-1]]), need=float(np.abs(torque[frames, j]).max()),
                                     have=float(have[frames, j].min()))
    hands = np.array([r['hands'] for r in rows])
    worst = int(np.argmax(ratio))
    outside = margin_ < -COM_OUTSIDE
    summary = dict(
        seconds=float(times[-1]), frames=len(rows),
        com_margin_min=float(margin_.min()), com_margin_at=float(times[int(np.argmin(margin_))]),
        com_outside_seconds=float(outside.sum() * (times[1] - times[0]) if len(times) > 1 else 0.),
        residual_force_max=float(force.max()), residual_force_at=float(times[int(np.argmax(force))]),
        residual_force_p95=float(np.percentile(force, 95)),
        residual_moment_max=float(moment.max()), residual_moment_at=float(times[int(np.argmax(moment))]),
        ratio_max=float(ratio.max()), ratio_joint=rows[worst]['worst'], ratio_at=rows[worst]['t'],
        ratio_need=rows[worst]['worst_torque'], ratio_have=rows[worst]['worst_have'],
        ratio_frames=int((ratio > 1).sum()), over_joints=over_joints,
        peak_ratio_max=float(max(r['peak_ratio'] for r in rows)),
        locked_ratio_max=float(max(r['locked_ratio'] for r in rows)),
        hand_max=float(hands.max()), hand_both_max=float(hands.sum(1).max()),
        hand_seconds=float((hands.max(1) > 1.).sum() * (times[1] - times[0]) if len(times) > 1 else 0.),
        hand_over_seconds=float((hands.max(1) > HAND_LOAD).sum() * (times[1] - times[0]) if len(times) > 1 else 0.),
        noise=noise, accel_rms=accel_rms, accel_max=accel_max,
        qd_max=float(max(r['qd_max'] for r in rows)), qd_joint=max(rows, key=lambda r: r['qd_max'])['qd_joint'])
    reasons = []
    if summary['com_margin_min'] < -COM_OUTSIDE:
        reasons.append(f'центр масс вне опоры на {-summary["com_margin_min"] * 100:.0f} см '
                       f'({summary["com_margin_at"]:.2f} с)')
    if summary['residual_force_max'] > RESIDUAL_FORCE:
        reasons.append(f'остаток силы {summary["residual_force_max"]:.2f} mg ({summary["residual_force_at"]:.2f} с)')
    if summary['residual_moment_max'] > RESIDUAL_ARM:
        reasons.append(f'остаток момента {summary["residual_moment_max"] * 100:.0f} см·mg '
                       f'({summary["residual_moment_at"]:.2f} с)')
    balance = list(reasons)
    if summary['ratio_max'] > 1:
        reasons.append(f'момент {summary["ratio_joint"]} {abs(summary["ratio_need"]):.0f} при доступных '
                       f'{summary["ratio_have"]:.0f} Н·м ({summary["ratio_at"]:.2f} с)')
    # The same verdict without the waist's torque rule (WAIST_FREE): a waist at its stop has the PD's kp 28·(stop − q)
    # ≈ 0 there, and LAFAN A, which a tracker learned, is short there in 50-53 of its 176 frames.
    free = np.array([not any(part in name for part in WAIST_FREE) for name in names]) & ~locked
    need = np.abs(torque) / (np.maximum(have, 0.) + TORQUE_SLACK * peak[None])
    free_ratio = np.where(free[None], need, 0.).max(1)
    at = int(np.argmax(free_ratio))
    joint = int(np.argmax(np.where(free, need[at], 0.)))
    summary.update(waist_free_ratio_max=float(free_ratio.max()), waist_free_ratio_joint=names[joint],
                   waist_free_ratio_at=float(times[at]), waist_free_ratio_frames=int((free_ratio > 1).sum()))
    if free_ratio.max() > 1:
        balance.append(f'момент {names[joint]} {abs(torque[at, joint]):.0f} при доступных {have[at, joint]:.0f} Н·м '
                       f'({times[at]:.2f} с)')
    summary.update(feasible=not reasons, reasons=reasons, feasible_waist_free=not balance, reasons_waist_free=balance)
    return summary


def run(job):
    import torch
    torch.set_num_threads(1)
    from humanoid_sim.tasks import g1_getup_track
    path = Path(job['clip'])
    clip = g1_getup_track.load(path.stem, path=path) if path.suffix == '.npz' else g1_getup_track.load(job['clip'])
    out = check_clip(clip, touch=job['touch'], lower=job['lower'], static=job.get('static', False))
    out.update(clip=job['clip'], md5=clip['md5'])
    return out


def verdict_text(summary, waist_free=False):
    """«годен» or «не годен: <reasons>»; waist_free: the verdict without the waist's torque rule."""
    if waist_free:
        return 'годен' if summary['feasible_waist_free'] else 'не годен: ' + '; '.join(summary['reasons_waist_free'])
    return 'годен' if summary['feasible'] else 'не годен: ' + '; '.join(summary['reasons'])


def print_summary(results):
    lines = [
        ('длительность, с / кадров', lambda s: f'{s["seconds"]:.2f} / {s["frames"]}'),
        ('висит над полом: max, см / кадров выше касания',
         lambda s: f'{s["float_max"] * 100:.1f} / {s["float_frames"]}' + (' (опущен)' if s['lowered'] else '')),
        (f'центр масс над опорой: min, см (с) / дольше {COM_OUTSIDE * 100:.0f} см вне, с',
         lambda s: f'{s["com_margin_min"] * 100:+.0f} ({s["com_margin_at"]:.2f}) / {s["com_outside_seconds"]:.2f}'),
        ('остаток силы, mg: max (с) / p95', lambda s: f'{s["residual_force_max"]:.3f} ({s["residual_force_at"]:.2f}) / '
                                                    f'{s["residual_force_p95"]:.3f}'),
        ('остаток момента / mg, см: max (с)',
         lambda s: f'{s["residual_moment_max"] * 100:.1f} ({s["residual_moment_at"]:.2f})'),
        ('нужный / доступный момент: max (сустав, с, Н·м) / кадров > 1',
         lambda s: f'{s["ratio_max"]:.2f} ({s["ratio_joint"]}, {s["ratio_at"]:.2f}, {abs(s["ratio_need"]):.0f}/'
                   f'{s["ratio_have"]:.0f}) / {s["ratio_frames"]}'),
        ('без пояса: нужный / доступный момент max (сустав, с) / кадров > 1',
         lambda s: f'{s["waist_free_ratio_max"]:.2f} ({s["waist_free_ratio_joint"]}, {s["waist_free_ratio_at"]:.2f}) / '
                   f'{s["waist_free_ratio_frames"]}'),
        ('нужный / пик мотора: max; запертые кисти: нужный / пик',
         lambda s: f'{s["peak_ratio_max"]:.2f}; {s["locked_ratio_max"]:.2f}'),
        (f'кисти на полу: одна max / обе max, Н / на полу, с / дольше {HAND_LOAD:.0f} Н, с',
         lambda s: f'{s["hand_max"]:.0f} / {s["hand_both_max"]:.0f} / {s["hand_seconds"]:.2f} / '
                   f'{s["hand_over_seconds"]:.2f}'),
        ('шум ускорений, рад/с² (сигнал rms / max)',
         lambda s: f'{s["noise"]:.1f} ({s["accel_rms"]:.1f} / {s["accel_max"]:.0f})'),
        ('скорость суставов max, рад/с', lambda s: f'{s["qd_max"]:.1f} ({s["qd_joint"]})'),
    ]
    print('Выполнимость клипа (без физики): ' + ' | '.join(r['clip'] for r in results))
    for title, value in lines:
        print(f'   {title}: ' + ' | '.join(value(r['summary']) for r in results))
    for r in results:
        over = r['summary']['over_joints']
        if over:
            print(f'   {r["clip"]}: не хватает момента — ' + '; '.join(
                f'{j} {v["frames"]} кадров {v["start"]:.2f}–{v["end"]:.2f} с, нужно до {v["need"]:.0f} при '
                f'{v["have"]:.0f}' for j, v in sorted(over.items(), key=lambda x: -x[1]['frames'])))
        print(f'   {r["clip"]}: {verdict_text(r["summary"])}')
        print(f'   {r["clip"]}, без правила моментов пояса: {verdict_text(r["summary"], waist_free=True)}')


def print_timeline(result, every):
    rows = result['rows']
    step = max(1, int(round(every / max(rows[1]['t'] - rows[0]['t'], 1e-9)))) if len(rows) > 1 else 1
    print(f'== {result["clip"]}: t, таз, опора (доли mg), ЦМ над опорой, остаток, худший сустав, кисти Н')
    for r in rows[::step]:
        load = ' '.join(f'{k} {v:.2f}' for k, v in sorted(r['load'].items(), key=lambda x: -x[1])) or '—'
        print(f'   {r["t"]:5.2f} таз {r["pelvis"]:.2f} | {load:34s} | ЦМ {r["com_margin"] * 100:+5.0f} см | '
              f'ост. {r["residual_force"]:.2f} mg {r["residual_moment"] * 100:4.1f} см | {r["worst"]} '
              f'{r["worst_ratio"]:.2f} | кисти {r["hands"][0]:.0f}/{r["hands"][1]:.0f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('clips', nargs='+', help='клипы runs/humanoid_lab/getup_track/<имя>.npz (или путь к .npz)')
    parser.add_argument('--touch', type=float, default=TOUCH, help='м: геом ближе к полу касается его')
    parser.add_argument('--as-is', action='store_true',
                        help='кадры как есть, не опуская на пол (по умолчанию нижний геом кадра ставится на пол)')
    parser.add_argument('--static', action='store_true', help='позы неподвижно (q̇ = q̈ = 0): статика без динамики')
    parser.add_argument('--timeline', type=float, default=0., help='шаг таблицы по кадрам, с (0 — без неё)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=Path, help='все кадры и итог, JSON')
    args = parser.parse_args()
    jobs = [dict(clip=c, touch=args.touch, lower=not args.as_is, static=args.static) for c in args.clips]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(run, jobs)
    if args.timeline > 0:
        for r in results:
            print_timeline(r, args.timeline)
    print_summary(results)
    if args.json:
        args.json.write_text(json.dumps(results), encoding='utf-8')


if __name__ == '__main__':
    main()
