"""The human clip of g1_getup's task_config track_clip (iteration 64, motion tracking as BeyondMimic / mjlab): one human
getting up (LAFAN1 fallAndGetUp, retargeted to the G1 by Unitree, or a CMU subject-140 take retargeted by the lab's
scripts/lab/cmu_mocap.py), at the 50 Hz control step, lifted out of the floor and with the locked wrists at home, plus
the getting-up model's kinematics of g1_tensor_math.TRACK_BODIES.

The file is runs/humanoid_lab/getup_track/<name>.npz (untracked: LAFAN1 is CC BY-NC-ND 4.0, a research prototype,
never published; CMU's data may not be redistributed, even converted; scripts/lab/getup_track_clip.py makes it; its
header's dataset, lafan1 when missing, decides whether scripts/robot/package.py lets a stage trained on it into a robot
package), arrays only, read without pickle: qpos (K, 36), qvel (K, 35)
(the root's angular velocity in its body frame, as MuJoCo's free joint), body_pos (K, 14, 3), body_rot (K, 14, 3, 3)
(world <- body), body_lin (K, 14, 3), body_ang (K, 14, 3) (the world velocities of the bodies' frame origins, from qvel
through the model's cvel: g1_tensor_math.body_point_velocity), lift and raw_lift (K,) (m: the root raised by the
smoothed depth of the deepest floor contact, and that depth per frame), header (JSON text: version, bodies, rate, the
source file, its md5 and frames, the lift's smoothing). The setting names the file's md5, so a run never follows another
clip than the one it was made with.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from . import g1_tensor_math as task

ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = ROOT / 'runs' / 'humanoid_lab' / 'getup_track'
TRACK_VERSION = 'g1-getup-track-v1'
LIFT_FRAMES = 5  # the lift's smoothing: a running maximum, then a running mean, over ±5 frames (±0.1 s)
ARRAYS = ('qpos', 'qvel', 'body_pos', 'body_rot', 'body_lin', 'body_ang', 'lift', 'raw_lift')


def clip_path(name):
    """runs/humanoid_lab/getup_track/<name>.npz (name a config.GETUP_TRACK_NAME, checked by the task's validator)."""
    return TRACK_DIR / f'{name}.npz'


def load(name, md5=None, path=None):
    """The clip `name` (or the file `path`) as a dict of read-only arrays plus header (dict) and md5, read once per file
    version in a process. Raises FileNotFoundError with the command that makes it, ValueError when the file's md5 is not
    `md5` (None: not checked; tests) or its arrays do not fit."""
    path = Path(clip_path(name) if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup track clip {path} is missing: python scripts/lab/getup_track_clip.py')
    status = path.stat()
    clip = _load(str(path.resolve()), status.st_mtime_ns, status.st_size)
    if md5 is not None and clip['md5'] != md5:
        raise ValueError(f'track clip {path}: md5 {clip["md5"]}, the setting names {md5} (another file)')
    return clip


@lru_cache(maxsize=4)
def _load(path, mtime, size):
    data = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        clip = {key: arrays[key] for key in arrays.files}
    header = json.loads(str(clip.pop('header')))
    if header.get('version') != TRACK_VERSION:
        raise ValueError(f'track clip {path}: version {header.get("version")}, expected {TRACK_VERSION}')
    check(clip, f'track clip {path}')
    if tuple(header.get('bodies', ())) != task.TRACK_BODIES:
        raise ValueError(f'track clip {path}: bodies {header.get("bodies")}, expected {task.TRACK_BODIES}')
    for value in clip.values():
        value.setflags(write=False)
    clip.update(header=header, md5=hashlib.md5(data).hexdigest())
    return clip


def check(clip, what='track clip'):
    """Raises ValueError unless clip's ARRAYS have their shapes (K ≥ 2 frames of the 29-DoF G1) and are finite."""
    k = len(clip.get('qpos', ()))
    bodies = len(task.TRACK_BODIES)
    shapes = dict(qpos=(k, 36), qvel=(k, 35), body_pos=(k, bodies, 3), body_rot=(k, bodies, 3, 3),
                  body_lin=(k, bodies, 3), body_ang=(k, bodies, 3), lift=(k,), raw_lift=(k,))
    if k < 2 or any(key not in clip or np.shape(clip[key]) != shape or not np.isfinite(clip[key]).all()
                    for key, shape in shapes.items()):
        raise ValueError(f'{what}: bad arrays (expected {shapes})')


def hold_frames(clip):
    """The reference frames g1_tensor_math.clip_reference reads, float64 (K + 1, B, …): pos, rot, lin, ang of the clip,
    then its hold frame (the last pose, zero velocities)."""
    def held(key, still=False):
        values = np.asarray(clip[key], dtype=np.float64)
        return np.concatenate([values, np.zeros_like(values[-1:]) if still else values[-1:]])
    return dict(pos=held('body_pos'), rot=held('body_rot'), lin=held('body_lin', True), ang=held('body_ang', True))


def velocities(qpos, dt):
    """(K, 35) qvel of frames qpos (K, 36) dt apart, by central differences (the ends by one-sided ones): the root
    linear velocity in the world, its angular velocity in the body frame (mju_subQuat), the joints."""
    k = len(qpos)
    before, after = np.maximum(np.arange(k) - 1, 0), np.minimum(np.arange(k) + 1, k - 1)
    span = (after - before)[:, None] * dt
    qvel = np.zeros((k, 35))
    qvel[:, :3] = (qpos[after, :3] - qpos[before, :3]) / span
    qvel[:, 6:] = (qpos[after, 7:] - qpos[before, 7:]) / span
    for i in range(k):
        w = np.zeros(3)
        mujoco.mju_subQuat(w, qpos[after[i], 3:7], qpos[before[i], 3:7])
        qvel[i, 3:6] = w / span[i, 0]
    return qvel


def floor_depth(model, data, robot_geoms, floor):
    """The deepest floor contact distance (m, ≤ 0) of a robot geom in the current state (after mj_forward), 0 without
    a penetrating one."""
    contact = data.contact
    depth = 0.
    for k in range(data.ncon):
        first, second = int(contact.geom1[k]), int(contact.geom2[k])
        if (first == floor and robot_geoms[second]) or (second == floor and robot_geoms[first]):
            depth = min(depth, float(contact.dist[k]))
    return depth


def smooth_lift(depth, frames=LIFT_FRAMES):
    """(K,) the lift of depths (K,) ≥ 0: a running maximum over ±frames, then a running mean over ±frames (the ends
    clamped), so no frame is lifted by a jump and a short dip is covered."""
    k = len(depth)
    index = np.clip(np.arange(k)[:, None] + np.arange(-frames, frames + 1)[None], 0, k - 1)
    return depth[index].max(1)[index].mean(1)


def kinematics(model, data, qpos, qvel, bodies):
    """(body_pos (K, B, 3), body_rot (K, B, 3, 3), body_lin (K, B, 3), body_ang (K, B, 3)) of the bodies (ids) at frames
    qpos, qvel: mj_forward, then the bodies' frame origins and their world velocities from cvel
    (g1_tensor_math.body_point_velocity about their root's subtree_com)."""
    bodies = np.asarray(bodies)
    root = model.body_rootid[bodies]
    out = [[], [], [], []]
    for position, velocity in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = position, velocity
        mujoco.mj_forward(model, data)
        linear, angular = task.body_point_velocity(data.cvel[bodies], data.xpos[bodies], data.subtree_com[root])
        for values, value in zip(out, (data.xpos[bodies], data.xmat[bodies].reshape(-1, 3, 3), linear, angular)):
            values.append(np.array(value))
    return tuple(np.array(values) for values in out)


def build(env, qpos):
    """The clip arrays (ARRAYS) of frames qpos (K, 36) at env.dt, env a G1GetupEnv (the getting-up model, its locked
    wrists and robot geoms): the locked wrists at home, each frame raised by smooth_lift of its floor_depth, qvel by
    central differences, the kinematics of TRACK_BODIES."""
    model, data = env.model, env.data
    qpos = np.array(qpos, dtype=np.float64)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    locked = np.asarray(env.qadr)[env.wrist_mask.numpy()]
    qpos[:, locked] = env.home[env.wrist_mask.numpy()]
    floor = model.geom('floor').id
    raw = []
    for row in qpos:
        data.qpos[:] = row
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        raw.append(-floor_depth(model, data, env.robot_geoms, floor))
    raw = np.array(raw)
    lift = smooth_lift(raw)
    qpos[:, 2] += lift
    qvel = velocities(qpos, env.dt)
    bodies = [model.body(name).id for name in task.TRACK_BODIES]
    body_pos, body_rot, body_lin, body_ang = kinematics(model, data, qpos, qvel, bodies)
    clip = dict(qpos=qpos, qvel=qvel, body_pos=body_pos, body_rot=body_rot, body_lin=body_lin, body_ang=body_ang,
                lift=lift, raw_lift=raw)
    check(clip)
    return clip
