"""Splicing a roll onto a track clip (scripts/lab/getup_roll_clip.py, iteration 82) on synthetic frames: the target's
frames are its arrays exactly after the blend, the seams are smooth (no velocity jump, unlike a bare cut), the physics
blend's end lands on the target's frame 0, the rigid placement keeps the roll's shape."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DT = .02


@pytest.fixture(scope='module')
def roll_script():
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
    return importlib.import_module('getup_roll_clip')


@pytest.fixture(scope='module')
def env():
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    return getup_model_env()


def standing(env, count, phase=0., speed=0., yaw=0.):
    """(count, 36) frames standing at 0.8 m, the free joints swinging by 0.2 rad, moving along x at `speed` m/s."""
    t = np.arange(count) * DT
    qpos = np.zeros((count, env.model.nq))
    qpos[:, 0] = speed * t
    qpos[:, 2] = .8
    qpos[:, 3], qpos[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    qpos[:, 7:] = env.home + .2 * np.sin(2 * np.pi * .7 * t[:, None] + phase + np.arange(env.model.nu)[None])
    return qpos


@pytest.fixture(scope='module')
def target(env):
    from humanoid_sim.tasks import g1_getup_track
    clip = g1_getup_track.build(env, standing(env, 40, phase=1., yaw=.7))
    return {**clip, 'header': {}, 'md5': 'synthetic'}


def accel_at(clip, frame, env):
    free = ~env.wrist_mask.numpy()
    qvel = np.asarray(clip['qvel'])[:, 6:][:, free]
    return float(np.abs(np.diff(qvel, axis=0))[frame - 2:frame + 2].max() / DT)


def test_hermite_and_smoothstep_meet_their_ends(roll_script):
    tau = np.array([0., 1.])
    p = roll_script.hermite(np.array([1., 2.]), np.array([3., -1.]), np.array([-1., 5.]), np.zeros(2), tau, .5)
    np.testing.assert_allclose(p, [[1., 2.], [-1., 5.]])
    h = 1e-6  # the start velocity
    start = roll_script.hermite(np.array([1.]), np.array([3.]), np.array([-1.]), np.zeros(1), np.array([0., h]), .5)
    assert (start[1, 0] - start[0, 0]) / (h * .5) == pytest.approx(3., rel=1e-4)
    assert roll_script.smoothstep(0.) == 0. and roll_script.smoothstep(1.) == 1. and roll_script.smoothstep(.5) == .5


def test_kinematic_splice_keeps_the_target_whole_and_its_seams_smooth(roll_script, env, target):
    from humanoid_sim.tasks import g1_getup_track
    roll = standing(env, 30, speed=.3)
    clip, record = roll_script.splice(env, roll, 20, target, .5)
    first = record['target_frame']
    assert first == 21 + 24 and len(clip['qpos']) == first + len(target['qpos'])
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(clip[key][first:], target[key])  # bit for bit after the blend
    g1_getup_track.check(clip)
    # the roll's frames turned and moved as one piece: the same joints, the same distances between frames
    np.testing.assert_allclose(clip['qpos'][:21, 7:][:, ~env.wrist_mask.numpy()],
                               roll[:21, 7:][:, ~env.wrist_mask.numpy()])
    step = np.linalg.norm(np.diff(clip['qpos'][:21, :2], axis=0), axis=1)
    np.testing.assert_allclose(step, np.linalg.norm(np.diff(roll[:21, :2], axis=0), axis=1), atol=1e-12)
    # a bare cut (the roll's frame 20, then the target) jumps; the blend does not
    bare = np.concatenate([clip['qpos'][:21], target['qpos']])
    jump = accel_at(dict(qvel=g1_getup_track.velocities(bare, DT)), 21, env)
    assert record['seam_accel'][0] < .25 * jump and record['seam_accel'][1] < .25 * jump
    speed = np.abs(np.diff(clip['qpos'][:, 7:], axis=0)).max() / DT
    assert speed < 3.  # no joint swings faster than ~3 rad/s anywhere


def test_physics_splice_lands_on_the_target_frame_0(roll_script, env, target):
    from humanoid_sim.tasks import g1_getup_track
    roll = standing(env, 30, speed=.3)
    physical = standing(env, 30, phase=.4, speed=.3)[5:30]  # 25 "physics" frames after the hand-over
    physical[:, 2] -= .01
    clip, record = roll_script.splice(env, roll, 20, target, .5, physical)
    first = record['target_frame']
    assert first == 21 + 24
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(clip[key][first:], target[key])
    assert record['end_rest']['height'] == pytest.approx(.01, abs=1e-9)
    # the frame before the target is within one step's smoothstep of it
    before = clip['qpos'][first - 1]
    assert np.abs(before[7:] - target['qpos'][0, 7:]).max() < .05
    assert np.abs(before[:3] - target['qpos'][0, :3]).max() < .02
    # the correction starts at the blend's middle: its first half is the physics frames, placed rigidly
    yaw, shift = record['yaw'], record['shift']
    assert abs(roll_script._heading(clip['qpos'][first - 1, 3:7]) - roll_script._heading(target['qpos'][0, 3:7])) < .05
    np.testing.assert_allclose(clip['qpos'][21:30, 7:][:, ~env.wrist_mask.numpy()],
                               physical[:9, 7:][:, ~env.wrist_mask.numpy()])
    assert np.isfinite(yaw) and len(shift) == 2


def test_trim_target_slices_every_array(roll_script, target):
    from humanoid_sim.tasks import g1_getup_track
    trimmed = roll_script.trim_target(target, 5)
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(trimmed[key], target[key][5:])
    assert roll_script.trim_target(target, 0) is target
    with pytest.raises(SystemExit):
        roll_script.trim_target(target, len(target['qpos']) - 1)
