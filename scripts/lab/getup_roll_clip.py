"""CPU only. getup_roll_clip.py <stage> [<stage> …] [--row R …] [--rows 6] [--target <clip>] [--blend 0.5]
[--target-frame K] [--mode physics|kinematic] [--cut S] [--seconds 3] [--name <name>] [--force] [--out <file.npz>] [--dry-run]
[--workers 4] [--json <file>]
A g1_getup track clip «from the back, roll over onto the stomach, then up as a human» (iteration 82; the lead's
decision on the critic's lab-critic/it81, Pavel's idea: from any pose turn onto the stomach, then stand up as a human):
BONES-SEED has no roll from the back onto the stomach (none of its 8 back takes turns over), while the front clip A493
v3 is learnt (branch 73, exp-84b0540b0092467d @ 570 M, up and standing 10 of 10). So the roll comes from one of our own
getting-up policies (a plain g1_getup stage, not a tracker: physically doable by construction, free of any licence) and
is spliced onto the front clip (--target, by default A493 v3; --target-frame K joins it at its frame K, its frames before
K dropped: A493 v3's right hand lies inside the torso over its first 0.36 s, which the physics blend cannot reach, so
K 18 is the working value).

1. The roll (roll_run). The stage's deterministic policy runs from a bank row (--row; by default the first --rows
   training rows lying on the back: the held-out rows stay for the tracker's evaluation), no assist, pd_bound at the
   stage's lowest, the physics ours (getup_probe's: no delay, motor friction ×1, the files' friction), --seconds long;
   qpos at every control step (50 Hz, the clip's rate).
2. The hand-over (hand_over): the first step where the pose is near the target's frame 0 by the settling rule
   (settle_errors against the frame's pelvis gravity and free joints: the angle at most SETTLE_DONE_ANGLE 25°, the mean
   |q − q0| at most SETTLE_DONE_POSE 0.3 rad, every free joint slower than SETTLE_DONE_SPEED 0.5 rad/s for
   SETTLE_DONE_SECONDS 0.3 s: g1_ctrl's switch of the settling task). Our getting-up policies never lie still on the
   stomach (1164 crosses it at 0.5-0.8 s with its knees tucked, the joints at 10-15 rad/s, the angle 26-39° at best;
   823 rolls over a side straight into a squat, 52-69°), so without such a step the hand-over is the step of the least
   CUT_SCORE (angle/25° + deviation/0.3) among those after the pelvis faced down (angle below 90°) while the anchor
   (torso) is still within TRACK_NEAREST_ANCHOR of the target frame 0's height (lying); --cut S takes the step at S s.
3. The blend, --blend s from the hand-over to the target's frame 0.
   physics (the default, physical_blend): the same episode goes on from the hand-over with the policy replaced by the
   PD: its targets move from the policy's last target to the target frame 0's joints by the quintic smoothstep over
   BLEND_RAMP of the blend and stay there (pd_bound off, the tracker has none), the physics ours resolving every
   contact, so the robot itself lowers onto its front. The roll and this blend are then turned about z and moved as
   one rigid piece so that the blend's last state lies on the target's frame 0 (its heading, task.track_heading, and
   its pelvis xy), and what is left (the pelvis height, its tilt, the joints: the PD's error, a contact) is taken off
   over the blend's last CORRECT_SHARE by the quintic smoothstep (the rotation's rest as a rotation vector).
   kinematic (blend_frames, the comparison): every joint and the root position by a quintic Hermite from the roll's
   state at the hand-over (positions and velocities) to the target's frame 0, the root rotation by the same Hermite on
   the rotation vector (a slerp eased by the quintic smoothstep where the velocities are zero), the roll placed so the
   target's frame 0 lies where its pelvis would stop decelerating evenly. It swings limbs through the floor (1164's
   right upper arm 14 cm), which the lift then gives the whole body: kept to show it, not to train on.
4. The arrays (splice): the roll's and the blend's frames as g1_getup_track.build's (locked wrists home; kinematic:
   lifted by smooth_lift of their own floor depth, the target's frames left out of the smoothing; physics: not lifted,
   the physics' own contacts, raw_lift their depth: a hand under ~400 N lies 3-5 cm in the floor, and lifting the frame
   by it floats the trunk off its support; central-difference velocities
   over the whole clip; the tracked bodies' kinematics), then the target clip's frames, every one of its ARRAYS
   exactly. The header is the target's (its dataset, source and licence: the roll is ours) plus `roll` (the stage, its
   steps, the row, the hand-over, the blend, the target and its md5, the seam numbers), written by getup_track_clip's
   save_clip (clip_feasibility's verdict in the header; an existing clip not overwritten without --force).

One stage and one row: the clip is written (runs/humanoid_lab/getup_track/<name>.npz, or --out; --dry-run only
prints). Several: a table per stage and row, nothing written: the roll's time (start to hand-over), the hand-over's
rule (rule / least / cut) and pose (angle to the target's pelvis gravity, mean |q − q0|, fastest free joint, pelvis
speed, anchor height), what the blend's end correction took off (largest joint, pelvis height, tilt), the seams (each
seam's largest joint acceleration over ±SEAM_FRAMES frames against the roll's and the target's own p99, rad/s²; the
blend's fastest joint), the penetration between robot parts (clip_feasibility: frames deeper than 1 cm, deepest), the
largest load on one hand over the roll and the blend, and clip_feasibility per window around the seam (physics: the
frames as they are, not lowered onto their lowest geom, for the same reason) (the roll's
last WINDOW_SECONDS, the blend, the target's first WINDOW_SECONDS: the centre of mass's least margin over the support,
the largest residual force, the largest needed/available torque of a driven joint).
"""
import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'probes'))

TARGET = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3'
BLEND = .5  # s
MODES = ('physics', 'kinematic')
BLEND_RAMP = .7  # physics: the PD targets reach the target's frame 0 at this share of the blend, then hold
CORRECT_SHARE = .5  # physics: the end's correction runs over this last share of the blend
SECONDS = 3.  # s of the policy's run
ROWS = 6  # the training back rows a survey takes by default
SEED = 82_000
CUT_SCORE = 'angle/25° + deviation/0.3'
END_JOINTS = 4  # the joints the record names by the end correction's size
SEAM_FRAMES = 2  # frames each side of a seam its accelerations are taken over
WINDOW_SECONDS = .5  # the windows clip_feasibility is read over before and after the blend
_cache = {}


def smoothstep(x):
    """The quintic smoothstep of x clipped to [0, 1]: 0 → 1 with zero velocity and acceleration at both ends."""
    x = np.clip(x, 0., 1.)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def hermite(p0, v0, p1, v1, tau, duration):
    """(n, d) the quintic Hermite from p0 with velocity v0 to p1 with v1 (d,) over `duration` s, zero accelerations at
    both ends, at tau (n,) in [0, 1]."""
    t = np.asarray(tau, dtype=np.float64)[:, None]
    t3, t4, t5 = t ** 3, t ** 4, t ** 5
    return ((1 - 10 * t3 + 15 * t4 - 6 * t5) * p0 + (t - 6 * t3 + 8 * t4 - 3 * t5) * duration * v0
            + (10 * t3 - 15 * t4 + 6 * t5) * p1 + (-4 * t3 + 7 * t4 - 3 * t5) * duration * v1)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])


def _quat_mul(a, b):
    import mujoco
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _rotation(quat):
    import mujoco
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
    return matrix.reshape(3, 3)


def _heading(quat):
    from humanoid_sim.tasks import g1_tensor_math as task
    return float(task.track_heading(_rotation(quat)[None])[0])


def place(qpos, frame, target_qpos, lead=np.zeros(2)):
    """(qpos (S, 36) turned about z and moved as one rigid piece, yaw, shift (2,)): frame `frame`'s pelvis heading
    (task.track_heading) turned to the target frame 0's (target_qpos (36,)), its pelvis xy put at the target's less
    `lead` (2,) in the target's frame (kinematic: where the pelvis would stop)."""
    qpos = np.array(qpos, dtype=np.float64)
    yaw = _heading(target_qpos[3:7]) - _heading(qpos[frame, 3:7])
    turn = _rotation(_yaw_quat(yaw))[:2, :2]
    out = qpos.copy()
    out[:, :2] = (turn @ (qpos[:, :2] - qpos[frame, :2]).T).T + target_qpos[:2] - lead
    q = _yaw_quat(yaw)
    for k in range(len(out)):
        out[k, 3:7] = _quat_mul(q, qpos[k, 3:7])
    return out, yaw, out[frame, :2] - qpos[frame, :2]


def blend_frames(env, start, start_velocity, end, end_velocity, frames, dt):
    """(frames − 1, 36) the qpos strictly between start and end (36,) over `frames` control steps: root position and
    joints by hermite (velocities start_velocity / end_velocity (35,): the root linear in the world, angular in the
    body frame, the joints), the joints within their ranges; the root rotation by hermite on the rotation vector from
    start's rotation (the end's angular velocity carried into the start's frame)."""
    import mujoco
    model = env.model
    duration = frames * dt
    tau = np.arange(1, frames) / frames
    out = np.zeros((frames - 1, model.nq))
    out[:, :3] = hermite(start[:3], start_velocity[:3], end[:3], end_velocity[:3], tau, duration)
    out[:, 7:] = hermite(start[7:], start_velocity[6:], end[7:], end_velocity[6:], tau, duration)
    joints = np.asarray(env.joints)
    adr = np.asarray(env.qadr)
    out[:, adr] = np.clip(out[:, adr], model.jnt_range[joints, 0], model.jnt_range[joints, 1])
    q0, q1 = start[3:7] / np.linalg.norm(start[3:7]), end[3:7] / np.linalg.norm(end[3:7])
    rotation = np.zeros(3)
    mujoco.mju_subQuat(rotation, q1, q0)  # q0⁻¹·q1 as a rotation vector in q0's frame
    carried = _rotation(q0).T @ _rotation(q1) @ end_velocity[3:6]
    vectors = hermite(np.zeros(3), start_velocity[3:6], rotation, carried, tau, duration)
    for k, vector in enumerate(vectors):
        quat = q0.copy()
        mujoco.mju_quatIntegrate(quat, vector, 1.)
        out[k, 3:7] = quat
    return out


def correct_end(frames, end, target_qpos, share=CORRECT_SHARE, free=None):
    """(frames (n, 36) with the rest end → target_qpos (36,) added over their last share by the quintic smoothstep,
    weight 1 at the frame after the last, the target's; the rest (dict: largest joint (rad), pelvis height (m), tilt
    (deg))): the root position and joints by difference, the rotation by the rotation vector end⁻¹·target in the
    frames' own body frame."""
    import mujoco
    n = len(frames)
    weight = smoothstep((np.arange(1, n + 1) - (n + 1) * (1 - share)) / ((n + 1) * share))
    out = np.array(frames, dtype=np.float64)
    out[:, :3] += weight[:, None] * (target_qpos[:3] - end[:3])
    out[:, 7:] += weight[:, None] * (target_qpos[7:] - end[7:])
    rest = np.zeros(3)
    mujoco.mju_subQuat(rest, target_qpos[3:7] / np.linalg.norm(target_qpos[3:7]), end[3:7] / np.linalg.norm(end[3:7]))
    for k in range(n):
        quat = out[k, 3:7] / np.linalg.norm(out[k, 3:7])
        mujoco.mju_quatIntegrate(quat, rest * weight[k], 1.)
        out[k, 3:7] = quat
    free = np.ones(len(end) - 7, bool) if free is None else free  # the locked wrists are home in the clip anyway
    return out, dict(joint=float(np.abs(target_qpos[7:] - end[7:])[free].max()),
                     height=float(target_qpos[2] - end[2]), tilt=float(np.degrees(np.linalg.norm(rest))))


def floor_depths(env, qpos):
    """(F,) each frame's deepest floor penetration of a robot geom, m ≥ 0 (g1_getup_track.floor_depth)."""
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    model, data = env.model, env.data
    floor = model.geom('floor').id
    out = []
    for row in qpos:
        data.qpos[:], data.qvel[:] = row, 0
        mujoco.mj_forward(model, data)
        out.append(-g1_getup_track.floor_depth(model, data, env.robot_geoms, floor))
    return np.array(out)


def splice(env, roll, cut, target, blend, physical=None):
    """(clip arrays, record) of the roll's qpos (S, 36) up to frame `cut`, round(blend / dt) blend steps, then the
    target clip (g1_getup_track.load's dict), whose ARRAYS the result's last K frames are exactly. physical: the
    physics blend's qpos after the hand-over ((steps, 36), physical_blend; its last state is where the target's frame
    0 comes), the roll and it placed together on the target's frame 0 and corrected at the end (correct_end); None:
    the kinematic blend (blend_frames). The new frames as build's: locked wrists home, lifted by smooth_lift of their
    own floor depth (the target's frames, already out of the floor, not in the smoothing), central-difference
    velocities over the whole clip, the tracked bodies' kinematics; the target's frames then copied."""
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data, dt = env.model, env.data, env.dt
    steps = int(round(blend / dt))
    if steps < 2 or not 1 <= cut < len(roll) - 1:
        raise ValueError(f'splice: blend {blend} s, hand-over frame {cut} of {len(roll)}')
    target_qpos = np.asarray(target['qpos'], dtype=np.float64)
    record = {}
    if physical is not None:
        if len(physical) != steps:
            raise ValueError(f'splice: {len(physical)} physics blend frames, the blend has {steps}')
        piece = np.concatenate([np.asarray(roll[:cut + 1], np.float64), np.asarray(physical, np.float64)])
        placed, yaw, shift = place(piece, len(piece) - 1, target_qpos[0])
        middle, rest = correct_end(placed[cut + 1:-1], placed[-1], target_qpos[0], free=~env.wrist_mask.numpy())
        record['end_rest'] = {key: round(value, 4) for key, value in rest.items()}
        difference = np.where(env.wrist_mask.numpy(), 0., target_qpos[0, 7:] - placed[-1, 7:])
        names = [model.joint(j).name[:-6] for j in env.joints]
        record['end_rest_joints'] = {names[j]: round(float(difference[j]), 3)
                                     for j in np.argsort(-np.abs(difference))[:END_JOINTS]}
        new = np.concatenate([placed[:cut + 1], middle])
    else:
        piece = np.asarray(roll[:cut + 2], np.float64)
        velocity = (piece[cut + 1, :2] - piece[cut - 1, :2]) / (2 * dt)
        heading = _heading(target_qpos[0, 3:7]) - _heading(piece[cut, 3:7])
        lead = _rotation(_yaw_quat(heading))[:2, :2] @ velocity * steps * dt / 2
        placed, yaw, shift = place(piece, cut, target_qpos[0], lead)
        start_velocity = g1_getup_track.velocities(placed, dt)[cut]
        middle = blend_frames(env, placed[cut], start_velocity, target_qpos[0],
                              np.asarray(target['qvel'][0], np.float64), steps, dt)
        new = np.concatenate([placed[:cut + 1], middle])
    new[:, 3:7] /= np.linalg.norm(new[:, 3:7], axis=1, keepdims=True)
    locked = np.asarray(env.qadr)[env.wrist_mask.numpy()]
    new[:, locked] = env.home[env.wrist_mask.numpy()]
    depth = floor_depths(env, new)
    if physical is not None:  # the physics' own contacts (a loaded hand ~3 cm): lifted, the trunk would float
        lift = np.zeros(len(new))
    else:
        lift = g1_getup_track.smooth_lift(depth)
    new[:, 2] += lift
    qpos = np.concatenate([new, target_qpos])
    qvel = g1_getup_track.velocities(qpos, dt)
    bodies = [model.body(name).id for name in task.TRACK_BODIES]
    kinematics = g1_getup_track.kinematics(model, data, new, qvel[:len(new)], bodies)
    clip = dict(qpos=qpos, qvel=qvel)
    for key, values in zip(('body_pos', 'body_rot', 'body_lin', 'body_ang'), kinematics):
        clip[key] = np.concatenate([values, np.asarray(target[key], dtype=np.float64)])
    clip['lift'] = np.concatenate([lift, np.asarray(target['lift'], dtype=np.float64)])
    clip['raw_lift'] = np.concatenate([depth, np.asarray(target['raw_lift'], dtype=np.float64)])
    for key in g1_getup_track.ARRAYS:  # the target's frames exactly
        clip[key][len(new):] = np.asarray(target[key])
    g1_getup_track.check(clip)
    record.update(mode='physics' if physical is not None else 'kinematic', cut_frame=int(cut),
                  cut_seconds=round(cut * dt, 3), blend_seconds=round(steps * dt, 3), blend_frames=int(steps),
                  target_frame=int(len(new)), target_seconds=round(len(new) * dt, 3), yaw=round(float(yaw), 4),
                  shift=[round(float(v), 4) for v in shift], lift_roll=round(float(lift[:cut + 1].max()), 4),
                  lift_blend=round(float(lift[cut + 1:].max()), 4),
                  floor_depth=[round(float(depth[:cut + 1].max()), 4), round(float(depth[cut + 1:].max()), 4)])
    record.update(seams(clip, cut, len(new), dt, env))
    return clip, record


def seams(clip, cut, first_target, dt, env):
    """The seam numbers of a spliced clip: per seam (roll -> blend at `cut`, blend -> target at `first_target`) the
    largest joint acceleration (|Δq̇|/dt of the free joints) over ±SEAM_FRAMES frames, the roll's and the target's own
    p99 of it, the blend's fastest free joint (rad/s), and the largest step in the root's height between frames."""
    free = np.flatnonzero(~env.wrist_mask.numpy())
    qvel = np.asarray(clip['qvel'])[:, 6:][:, free]
    accel = np.abs(np.diff(qvel, axis=0)).max(1) / dt  # (K − 1,): between frames k and k + 1
    around = lambda k: float(accel[max(k - SEAM_FRAMES, 0):min(k + SEAM_FRAMES, len(accel))].max())
    names = [env.model.joint(j).name for j in env.joints]
    blend = np.abs(qvel[cut:first_target + 1])
    worst = int(np.unravel_index(int(np.argmax(blend)), blend.shape)[1])
    z = np.asarray(clip['qpos'])[:, 2]
    return dict(seam_accel=[round(around(cut), 1), round(around(first_target), 1)],
                roll_accel_p99=round(float(np.percentile(accel[:max(cut, 1)], 99)), 1),
                target_accel_p99=round(float(np.percentile(accel[first_target:], 99)), 1),
                blend_speed_max=round(float(blend.max()), 2), blend_speed_joint=names[free[worst]][:-6],
                seam_dz=round(float(np.abs(np.diff(z[max(cut - 1, 0):first_target + 2])).max()), 4))


def _episode(stage, row, seed):
    """(environment, policy, normalizer, config, observation) of the stage's task reset at bank row `row`: no assist,
    pd_bound at its lowest, physics ours."""
    import torch
    torch.set_num_threads(1)
    import getup_probe
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = getup_probe._policy(str(stage))
    if config.get('track_clip') or config.get('track_clips') or config.get('settle_clip'):
        raise SystemExit(f'{stage}: нужен этап обычного вставания g1_getup (без track_clip / track_clips / settle_clip)')
    key = str(stage)
    if key not in _cache:
        _cache[key] = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100_000})
    env = _cache[key]
    options = dict(assist_force=0., start=int(row), action_delay_steps=0, motor_friction_scale=1.)
    if config.get('pd_bound'):
        options['pd_bound'] = 'min'
    obs, _ = env.reset(seed=seed, options=options)
    getup_probe.apply_friction(env.unwrapped, None)
    return env, policy, norm, config, obs


def roll_run(stage, row, seconds=SECONDS, seed=SEED):
    """(qpos (S + 1, 36), qvel (S + 1, 35), the stage's steps) of the stage's deterministic policy from bank row `row`
    (_episode), `seconds`: the reset's state first, then the state after each control step."""
    env, policy, norm, _, obs = _episode(stage, row, seed)
    raw = env.unwrapped
    qpos, qvel = [raw.data.qpos.copy()], [raw.data.qvel.copy()]
    for _ in range(int(round(seconds / raw.dt))):
        obs, _, terminated, truncated, info = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if info.get('bad_state'):
            break
        qpos.append(raw.data.qpos.copy())
        qvel.append(raw.data.qvel.copy())
        if terminated or truncated:
            break
    return np.array(qpos), np.array(qvel), int(Path(stage).name.split('_')[-1])


def physical_blend(stage, row, cut, joints, steps, seed=SEED, ramp=BLEND_RAMP, check=None):
    """(steps, 36) qpos of the physics blend: roll_run's episode again up to frame `cut` (check: its qpos there, which
    it must repeat to 1e-9), then `steps` control steps of the PD alone, its targets from the policy's last one to
    `joints` (29,) by the quintic smoothstep over ramp·steps, then held; pd_bound off, the locked wrists home."""
    env, policy, norm, _, obs = _episode(stage, row, seed)
    raw = env.unwrapped
    target = None
    for _ in range(cut):
        action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        target = raw._target(action)
        obs, *_ = env.step(action)
    if check is not None and np.abs(raw.data.qpos - check).max() > 1e-9:
        raise RuntimeError(f'{stage} строка {row}: прогон не повторился к кадру {cut}')
    start = raw.data.qpos[raw.qadr].copy() if target is None else target
    if raw.bound is not None:
        raw.set_pd_bound(None)
    out = []
    for k in range(1, steps + 1):
        goal = start + smoothstep(k / (ramp * steps)) * (joints - start)
        action = (goal - raw.home) / raw.scale
        _, _, terminated, _, info = env.step(action)
        if info.get('bad_state') or terminated:
            raise RuntimeError(f'{stage} строка {row}: переход прерван на шаге {k}')
        out.append(raw.data.qpos.copy())
    return np.array(out)


def hand_over(env, qpos, qvel, target):
    """(frame, rule, pose (dict per frame: angle deg, deviation rad, speed rad/s, pelvis m/s, anchor m)) of a roll
    against the target's frame 0: the first frame where the settling rule held SETTLE_DONE_SECONDS ('rule', its last
    frame), else the least CUT_SCORE among the frames facing down and lying ('least'); None when there is none."""
    import mujoco
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data = env.model, env.data
    free = [int(j) for j in np.flatnonzero(~env.wrist_mask.numpy())]
    first = np.asarray(target['qpos'][0], dtype=np.float64)
    goal_gravity = task.projected_gravity(_rotation(first[3:7])[None])
    anchor_index = task.TRACK_BODIES.index(task.TRACK_ANCHOR)
    goal_anchor = float(target['body_pos'][0, anchor_index, 2])
    anchor = model.body(task.TRACK_ANCHOR).id
    angle, deviation, speed, pelvis, height = [], [], [], [], []
    for position, velocity in zip(qpos, qvel):
        data.qpos[:] = position
        mujoco.mj_kinematics(model, data)
        gravity = task.projected_gravity(_rotation(position[3:7])[None])
        a, _, d = task.settle_errors(gravity, position[env.qadr][None], goal_gravity, first[7:][None], free)
        angle.append(float(a[0, 0]))
        deviation.append(float(d[0, 0]))
        speed.append(float(np.abs(velocity[env.vadr][free]).max()))
        pelvis.append(float(np.linalg.norm(velocity[:3])))
        height.append(float(data.xpos[anchor, 2]))
    angle, deviation, speed, height = map(np.array, (angle, deviation, speed, height))
    near = (angle <= task.SETTLE_DONE_ANGLE) & (deviation <= task.SETTLE_DONE_POSE) & (speed < task.SETTLE_DONE_SPEED)
    hold = round(task.SETTLE_DONE_SECONDS / env.dt)
    count, frame, rule = 0, None, None
    for k, value in enumerate(near[:-1]):
        count = count + 1 if value else 0
        if count >= hold:
            frame, rule = k, 'rule'
            break
    if frame is None:
        faced = np.flatnonzero(angle < np.pi / 2)
        lying = height <= goal_anchor + task.TRACK_NEAREST_ANCHOR
        candidates = [k for k in range(int(faced[0]) if len(faced) else len(angle), len(angle) - 1) if lying[k]]
        if candidates:
            score = angle / task.SETTLE_DONE_ANGLE + deviation / task.SETTLE_DONE_POSE
            frame, rule = int(min(candidates, key=lambda k: score[k])), 'least'
    pose = dict(angle=np.degrees(angle), deviation=deviation, speed=speed, pelvis=np.array(pelvis), anchor=height)
    return frame, rule, pose


def window(rows, start, end):
    """clip_feasibility's numbers over the rows with start ≤ t < end (s): the centre of mass's least margin (m), the
    largest residual force (of m·g), the largest needed/available torque of a driven joint (its name), the largest load
    on one hand (N), the deepest penetration between robot parts (m) and the frames deeper than SELF_DEEP."""
    import clip_feasibility
    chosen = [r for r in rows if start - 1e-9 <= r['t'] < end - 1e-9]
    if not chosen:
        return None
    worst = max((float(r['worst_ratio']), r['worst']) for r in chosen)
    return dict(com_margin=min(r['com_margin'] for r in chosen),
                residual=max(r['residual_force'] for r in chosen), ratio=worst[0], ratio_joint=worst[1],
                hand=max(max(r['hands']) for r in chosen), self_depth=max(r['self_depth'] for r in chosen),
                self_frames=int(sum(r['self_depth'] > clip_feasibility.SELF_DEEP for r in chosen)))


def trim_target(clip, frame=0):
    """The target clip from its frame `frame` on (ARRAYS sliced, header and md5 kept): the splice joins there. A493 v3's
    right hand lies inside the torso over its first 0.36 s (clip_feasibility: right_wrist_yaw-torso, frames 0-18), a
    pose the physics blend cannot reach, so the end correction would push the arm into the body."""
    from humanoid_sim.tasks import g1_getup_track
    if not 0 <= frame < len(clip['qpos']) - 1:
        raise SystemExit(f'--target-frame {frame}: у цели кадры 0…{len(clip["qpos"]) - 2}')
    if frame == 0:
        return clip
    return {**clip, **{key: clip[key][frame:] for key in g1_getup_track.ARRAYS}}


def survey(job):
    """One stage and row: the roll, the hand-over, the blend, the splice and its checks (the clip itself with keep)."""
    import clip_feasibility
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    if 'env' not in _cache:
        _cache['env'] = getup_model_env()
    env = _cache['env']
    target = trim_target(g1_getup_track.load(job['target']), job.get('target_frame', 0))
    qpos, qvel, steps = roll_run(job['stage'], job['row'], job['seconds'])
    frame, rule, pose = hand_over(env, qpos, qvel, target)
    if job.get('cut') is not None:
        frame, rule = int(round(job['cut'] / env.dt)), 'cut'
    out = dict(stage=str(job['stage']), steps=steps, row=job['row'], rule=rule, frame=frame, target=job['target'])
    if frame is None or not 1 <= frame < len(qpos) - 1:
        return dict(out, rule=None)
    out.update({key: float(values[frame]) for key, values in pose.items()})
    physical = None
    if job['mode'] == 'physics':
        physical = physical_blend(job['stage'], job['row'], frame, np.asarray(target['qpos'][0, 7:], np.float64),
                                  int(round(job['blend'] / env.dt)), check=qpos[frame])
    clip, record = splice(env, qpos, frame, target, job['blend'], physical)
    # As they are: a physics frame's loaded hand lies ~3 cm in the floor, and lowered onto it (lower) the trunk floats.
    checked = clip_feasibility.check_clip(clip, env, lower=job['mode'] != 'physics')
    rows, summary = checked['rows'], checked['summary']
    cut_t, target_t = record['cut_seconds'], record['target_seconds']
    out.update(record=record, clip_seconds=round((len(clip['qpos']) - 1) * env.dt, 2),
               self_deep_frames=summary['self_deep_frames'], self_depth_max=summary['self_depth_max'],
               self_pairs=summary['self_pairs'],
               new_self=window(rows, 0., target_t), hand_roll=max(max(r['hands']) for r in rows if r['t'] < target_t),
               windows=dict(roll=window(rows, max(cut_t - WINDOW_SECONDS, 0.), cut_t + 1e-6),
                            blend=window(rows, cut_t + 1e-6, target_t),
                            target=window(rows, target_t, target_t + WINDOW_SECONDS)),
               feasible=summary['feasible'], feasible_waist_free=summary['feasible_waist_free'],
               reasons=summary['reasons_waist_free'])
    if job.get('keep'):
        out['clip'] = clip
    return out


def print_table(results):
    modes = sorted({r['record']['mode'] for r in results if 'record' in r})
    print(f'Склейка «перекат политики → {results[0]["target"]}», переход {", ".join(modes) or "—"}: '
          f'стык по правилу укладки (rule), по наименьшему {CUT_SCORE} (least) или задан (cut); окна '
          f'clip_feasibility: последние {WINDOW_SECONDS} с переката | переход | первые {WINDOW_SECONDS} с цели '
          '(ЦМ над опорой см / остаток mg / момент нужный/доступный / кисть Н)')
    head = ('этап', 'строка', 'перекат с', 'стык', 'угол°', 'откл. рад', 'сустав рад/с', 'таз м/с', 'торс м',
            'снято в конце: сустав рад / таз см / наклон°', 'ускор. на стыках', 'p99 переката/цели',
            'переход max рад/с', 'прон. до цели кадров/см', 'кисть Н', 'окна')
    print(' | '.join(head))
    for r in results:
        if r.get('rule') is None:
            print(f'{r["steps"]} | {r["row"]} | — | стыка нет (не лёг лицом вниз, пока торс внизу)')
            continue
        rec, w = r['record'], r['windows']
        cell = lambda v: '—' if v is None else (f'{v["com_margin"] * 100:+.0f}/{v["residual"]:.2f}/{v["ratio"]:.1f}/'
                                                f'{v["hand"]:.0f}')
        rest = rec.get('end_rest')
        rest = '—' if rest is None else f'{rest["joint"]:.2f} / {rest["height"] * 100:+.1f} / {rest["tilt"]:.0f}'
        print(' | '.join(str(x) for x in (
            r['steps'], r['row'], f'{rec["cut_seconds"]:.2f}', r['rule'], f'{r["angle"]:.0f}', f'{r["deviation"]:.2f}',
            f'{r["speed"]:.1f}', f'{r["pelvis"]:.2f}', f'{r["anchor"]:.2f}', rest,
            f'{rec["seam_accel"][0]:.0f} / {rec["seam_accel"][1]:.0f}',
            f'{rec["roll_accel_p99"]:.0f} / {rec["target_accel_p99"]:.0f}',
            f'{rec["blend_speed_max"]:.1f} ({rec["blend_speed_joint"]})',
            f'{r["new_self"]["self_frames"]}/{r["new_self"]["self_depth"] * 100:.1f}', f'{r["hand_roll"]:.0f}',
            f'{cell(w["roll"])} | {cell(w["blend"])} | {cell(w["target"])}')))


def main():
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    from humanoid_sim.tasks.config import GETUP_TRACK_NAME
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument('stages', nargs='+', type=Path, help='этапы обычного вставания g1_getup (папки stage_…)')
    parser.add_argument('--row', type=int, action='append', default=[],
                        help='строка банка стартов (лёжа на спине; можно несколько раз)')
    parser.add_argument('--rows', type=int, default=ROWS,
                        help=f'без --row: столько первых обучающих строк «на спине» (по умолчанию {ROWS})')
    parser.add_argument('--target', default=TARGET, help=f'клип-цель (по умолчанию {TARGET})')
    parser.add_argument('--target-frame', type=int, default=0,
                        help='стык с этим кадром цели (её кадры до него отброшены; по умолчанию 0)')
    parser.add_argument('--blend', type=float, default=BLEND, help=f'с перехода (по умолчанию {BLEND})')
    parser.add_argument('--mode', choices=MODES, default=MODES[0],
                        help='переход: physics — ПД ведёт цели к кадру 0 цели в нашей физике; kinematic — квинтика '
                             'суставов и корня (для сравнения: руки проходят сквозь пол)')
    parser.add_argument('--cut', type=float, help='стык в эту секунду переката (вместо правила)')
    parser.add_argument('--seconds', type=float, default=SECONDS, help=f'с прогона политики (по умолчанию {SECONDS})')
    parser.add_argument('--name', help='имя клипа (runs/humanoid_lab/getup_track/<имя>.npz)')
    parser.add_argument('--force', action='store_true', help='перезаписать клип с тем же именем')
    parser.add_argument('--out', type=Path, help='записать клип в этот файл (пробы)')
    parser.add_argument('--dry-run', action='store_true', help='только напечатать, не записывать')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=Path, help='итоги, JSON')
    args = parser.parse_args()
    bank = bank_module.load()
    back = bank['classes'].index('back')
    rows = args.row or [int(r) for r in np.flatnonzero(~bank['holdout'] & (bank['start_class'] == back))[:args.rows]]
    for row in rows:
        if not 0 <= row < len(bank['qpos']):
            raise SystemExit(f'строки {row} нет в банке')
        if bank['start_class'][row] != back:
            print(f'внимание: строка {row} — класс {bank["classes"][int(bank["start_class"][row])]}, не «на спине»')
        if bank['holdout'][row]:
            print(f'внимание: строка {row} отложена для проверки трекера; запись с неё сделает проверку нечестной')
    target = g1_getup_track.load(args.target)
    trim_target(target, args.target_frame)
    jobs = [dict(stage=s.resolve(), row=r, target=args.target, target_frame=args.target_frame, blend=args.blend,
                 mode=args.mode, cut=args.cut, seconds=args.seconds) for s in args.stages for r in rows]
    write = len(jobs) == 1 and not args.dry_run
    if len(jobs) == 1:
        jobs[0]['keep'] = write
        results = [survey(jobs[0])]
    else:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(survey, jobs, chunksize=1)
    print_table(results)
    if args.json:
        args.json.write_text(json.dumps([{k: v for k, v in r.items() if k != 'clip'} for r in results], indent=1,
                                        default=str), encoding='utf-8')
    if not write:
        return
    r = results[0]
    if r.get('rule') is None:
        raise SystemExit('стыка нет: клип не записан')
    import getup_track_clip
    short = 'a493v3' if args.target == TARGET else target['md5'][:8]
    short += f'f{args.target_frame}' if args.target_frame else ''
    name = args.name or f'roll-{r["steps"]}-r{r["row"]}-{args.mode[:4]}{round(args.blend * 100)}-{short}'
    if not re.fullmatch(GETUP_TRACK_NAME, name):
        raise SystemExit(f'имя клипа {name!r} не подходит ({GETUP_TRACK_NAME}): задайте --name')
    fields = {key: value for key, value in target['header'].items()
              if key not in ('version', 'bodies', 'anchor', 'rate', 'seconds', 'lift_frames', 'locked_wrists',
                             'feasibility')}
    stage = Path(r['stage'])
    fields['roll'] = dict(
        script='scripts/lab/getup_roll_clip.py',
        stage=stage.relative_to(ROOT).as_posix() if stage.is_relative_to(ROOT) else str(stage), steps=r['steps'],
        row=r['row'], row_class='back', seed=SEED, physics='ours', seconds=args.seconds, hand_over=r['rule'],
        hand_over_score=CUT_SCORE, blend_ramp=BLEND_RAMP if args.mode == 'physics' else None,
        correct_share=CORRECT_SHARE if args.mode == 'physics' else None,
        hand_over_pose=dict(angle_deg=round(r['angle'], 1), deviation=round(r['deviation'], 3),
                            joint_speed=round(r['speed'], 2), pelvis_speed=round(r['pelvis'], 3),
                            anchor=round(r['anchor'], 3)),
        target=args.target, target_md5=target['md5'], target_from_frame=args.target_frame, **r['record'],
        licence="the roll and the blend are the lab's own policy and physics (no third-party data); the rest is the "
                "target clip's")
    getup_track_clip.save_clip(_cache['env'], r['clip'], fields, name, args.out, args.force)
    rec = r['record']
    print(f'перекат 0–{rec["cut_seconds"]:.2f} с (этап {r["steps"]}, строка {r["row"]}), переход ({rec["mode"]}) до '
          f'{rec["target_seconds"]:.2f} с, дальше {args.target} с кадра {rec["target_frame"]} (бит в бит)')


if __name__ == '__main__':
    main()
