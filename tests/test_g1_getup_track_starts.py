"""g1_getup track_clip's training starts (iteration 68): task_config track_starts (the shares of the first frame, a
uniform frame and a bank row), track_noise (the clip starts' noise multiple), track_bank_classes (the bank classes a
bank start draws from) and track_bank_nearest (a bank start at its nearest clip frame), CPU. Missing, the starts are
bit for bit the old ones; the shares over 10 000 resets; the noise scales; the nearest frame by hand, numpy and torch
alike. Not part of the contract. CPU–GPU: tests/test_g1_gpu.py -k track_starts."""
import hashlib
import math

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OPTIONS, drop_bank
from test_g1_getup_track import FRAMES, SETTING, synthetic_clip, track_env

# starts_digest of the code before these settings (commit 6df9187), with test_g1_getup_track's synthetic clip and
# drop_bank().
OLD_STARTS_DIGEST = '6096d6a56cc2edc88fa84e8513e369ab838962bb3083c7a1f9f5a0a33287d31a'
DEFAULTS = dict(track_starts={'first': .6, 'rsi': .3, 'bank': .1}, track_noise=1., track_bank_classes=['back'],
                track_bank_nearest=False)


def starts_digest(seed=11, resets=40, steps=4, **settings):
    """sha256 of 40 training resets (each reseeded: start kind, frame, row, noise, randomization) and their first steps
    under random actions: observations, qpos, qvel, the start's info, rewards and end flags."""
    env = track_env(**settings)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    try:
        for k in range(resets):
            obs, info = env.reset(seed=seed + k)
            digest.update(obs.tobytes() + env.data.qpos.tobytes() + env.data.qvel.tobytes()
                          + f"{info['track_start']} {info['track_frame']} {info['start']}".encode())
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(rng.uniform(-3., 3., 29))
                digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated]))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return digest.hexdigest()


def test_the_settings_are_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    plain = validate_config(base)['task_config']
    assert (plain['track_starts'], plain['track_noise'], plain['track_bank_classes'], plain['track_bank_nearest']) == (
        None, 1., None, False)  # old stages: the old starts
    tracking = {'track_clip': SETTING}
    on = validate_config({**base, 'task_config': {**tracking, 'track_starts': {'first': .3, 'rsi': .3, 'bank': .4},
                                                  'track_noise': 3, 'track_bank_classes': ['back', 'front', 'side'],
                                                  'track_bank_nearest': True}})['task_config']
    assert on['track_starts'] == {'first': .3, 'rsi': .3, 'bank': .4} and on['track_noise'] == 3.
    assert on['track_bank_classes'] == ['back', 'front', 'side'] and on['track_bank_nearest'] is True
    for key, value in (('track_starts', {'first': .5, 'rsi': .3, 'bank': .1}), ('track_starts', {'first': 1.}),
                       ('track_starts', {'first': 1.2, 'rsi': -.2, 'bank': 0.}),
                       ('track_starts', {'first': .6, 'rsi': .3, 'bank': .1, 'x': 0.}), ('track_starts', [.6, .3, .1]),
                       ('track_starts', {'first': True, 'rsi': 0., 'bank': 0.}), ('track_noise', -1.),
                       ('track_noise', 6.), ('track_noise', float('nan')), ('track_noise', True),
                       ('track_bank_classes', []), ('track_bank_classes', ['chest']),
                       ('track_bank_classes', ['back', 'back']), ('track_bank_classes', 'back'),
                       ('track_bank_nearest', 1)):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {**tracking, key: value}})
    for key, value in (('track_starts', {'first': .3, 'rsi': .3, 'bank': .4}), ('track_noise', 3.),
                       ('track_bank_classes', ['front']), ('track_bank_nearest', True)):
        with pytest.raises(ValueError, match='need track_clip'):  # starts of a clip only
            validate_config({**base, 'task_config': {key: value}})
    assert task_identity('g1_29dof', 'g1_getup', {**tracking, 'track_starts': {'first': .3, 'rsi': .3, 'bank': .4},
                                                  'track_noise': 3., 'track_bank_classes': ['front'],
                                                  'track_bank_nearest': True}) == task_identity(
        'g1_29dof', 'g1_getup', tracking)
    fields = {p['key']: p for p in G1_GETUP_PARAMETERS}
    assert fields['track_starts']['default'] is None and 'первого кадра' in fields['track_starts']['label']
    assert fields['track_noise']['default'] == 1. and fields['track_noise']['max'] == 5.
    assert fields['track_bank_classes']['default'] is None and fields['track_bank_nearest']['type'] == 'bool'


def test_missing_or_the_old_values_the_starts_are_bit_for_bit_the_old_ones():
    assert starts_digest() == OLD_STARTS_DIGEST
    assert starts_digest(**DEFAULTS) == OLD_STARTS_DIGEST


def test_the_shares_over_ten_thousand_resets_and_the_bank_classes():
    """track_starts {0.3, 0.3, 0.4} over 10 000 training resets (each share within 0.015, ~3σ); a share of 0 never
    starts; the bank starts draw only track_bank_classes' rows (drop_bank: row 0 back, rows 1 and 2 front)."""
    env = track_env(track_starts={'first': .3, 'rsi': .3, 'bank': .4}, track_bank_classes=['front'])
    try:
        env.reset(seed=0)
        counts, rows = dict.fromkeys(task.TRACK_START_KINDS, 0), set()
        for _ in range(10_000):
            _, info = env.reset()
            counts[info['track_start']] += 1
            if info['track_start'] == 'bank':
                rows.add(info['start'])
                assert info['start_class'] == 'front'
        for kind, share in zip(task.TRACK_START_KINDS, (.3, .3, .4)):
            assert abs(counts[kind] / 10_000 - share) < .015, counts
        assert rows == {1, 2}
    finally:
        env.close()
    env = track_env(track_starts={'first': 0., 'rsi': 0., 'bank': 1.}, track_bank_classes=['back', 'front'])
    try:
        env.reset(seed=1)
        starts = [env.reset()[1] for _ in range(200)]
        assert {info['track_start'] for info in starts} == {'bank'}
        assert {info['start'] for info in starts} == {0, 1, 2}  # the standing row 3 is not among the classes
    finally:
        env.close()


def test_the_noise_scales_numpy_and_torch():
    """track_start_pose / track_start_speed at scale 3: root xy, z, joints (inside wide ranges) and the root's speeds
    move exactly 3 times as far on the same draws, the rotation is quat_from_euler of 3 times the angles; torch gives
    the same numbers; scale 1 is the original function, bit for bit."""
    rng = np.random.default_rng(0)
    n = 32
    qpos = rng.normal(size=(n, 36))
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qvel = rng.normal(size=(n, 35))
    pose, speed = rng.uniform(-1., 1., (n, task.TRACK_POSE_NOISE)), rng.uniform(-1., 1., (n, task.TRACK_SPEED_NOISE))
    low, high = -10 * np.ones(29), 10 * np.ones(29)
    one, three = (task.track_start_pose(qpos, pose, low, high, s) for s in (1., 3.))
    np.testing.assert_array_equal(task.track_start_pose(qpos, pose, low, high), one)
    for columns in (slice(0, 3), slice(7, 36)):
        np.testing.assert_allclose(three[:, columns] - qpos[:, columns], 3 * (one[:, columns] - qpos[:, columns]),
                                   atol=1e-12)
    width = task.TRACK_START_NOISE
    delta = task.quat_from_euler(3 * width['tilt'] * pose[:, 3], 3 * width['tilt'] * pose[:, 4],
                                 3 * width['yaw'] * pose[:, 5])
    quat = task.quat_multiply(delta, qpos[:, 3:7])
    np.testing.assert_allclose(three[:, 3:7], quat / np.linalg.norm(quat, axis=1, keepdims=True), atol=1e-12)
    slow, fast = task.track_start_speed(qvel, speed), task.track_start_speed(qvel, speed, 3.)
    np.testing.assert_allclose(fast - qvel, 3 * (slow - qvel), atol=1e-12)
    np.testing.assert_allclose(np.abs(fast[:, :3] - qvel[:, :3]).max(), 3 * width['linear'], rtol=.1)
    tensors = [torch.from_numpy(a) for a in (qpos, pose, low, high)]
    np.testing.assert_allclose(task.track_start_pose(*tensors, 3.).numpy(), three, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(task.track_start_speed(torch.from_numpy(qvel), torch.from_numpy(speed), 3.).numpy(),
                               fast, rtol=1e-12, atol=1e-12)


def test_the_task_scales_its_start_noise():
    """The same training reset seeds with track_noise 1 and 3 (all first-frame starts): the joints and the root's xy
    move 3 times as far from the clip's first frame where no joint range clips them (z: the start is raised out of
    the floor after it), within 3 times mjlab's widths."""
    first = {'first': 1., 'rsi': 0., 'bank': 0.}
    envs = [track_env(track_starts=first, track_noise=scale) for scale in (1., 3.)]
    try:
        start = envs[0].clip['qpos'][0]
        low, high = envs[0].range_low, envs[0].range_high
        for seed in range(8):
            q1, q3 = [], []
            for env, out in zip(envs, (q1, q3)):
                _, info = env.reset(seed=seed)
                assert info['track_start'] == 'clip'
                out.extend(env.data.qpos)
            q1, q3 = np.array(q1), np.array(q3)
            tripled = start[7:] + 3 * (q1[7:] - start[7:])
            free = (low < q1[7:]) & (q1[7:] < high) & (low < tripled) & (tripled < high)  # no range clipped either
            assert free.sum() > 20
            np.testing.assert_allclose((q3 - start)[7:][free], 3 * (q1 - start)[7:][free], atol=1e-9)
            np.testing.assert_allclose((q3 - start)[:2], 3 * (q1 - start)[:2], atol=1e-9)
            assert np.abs(q3[7:] - start[7:]).max() <= 3 * task.TRACK_START_NOISE['joints'] + 1e-9
            assert np.abs(q3[:2] - start[:2]).max() <= 3 * task.TRACK_START_NOISE['xy'] + 1e-9
    finally:
        for env in envs:
            env.close()


def test_the_nearest_frame_by_hand_numpy_and_torch():
    """track_nearest_frame: the largest settle kernel sum among the frames whose anchor lies within
    TRACK_NEAREST_ANCHOR of the start's; frame 0 when none does; numpy and torch alike."""
    rng = np.random.default_rng(1)
    frames, joints = 12, tuple(range(25))
    gravity_f = rng.normal(size=(frames, 3))
    gravity_f /= np.linalg.norm(gravity_f, axis=1, keepdims=True)
    position_f = rng.uniform(-1., 1., (frames, 29))
    height_f = np.linspace(.1, .9, frames)
    # Starts: frame 4's pose exactly; frame 4's pose but with the anchor 0.2 m above it (frame 4 out of reach, its
    # neighbours by height in); frame 9's pose with every frame out of reach (anchor 2 m up).
    gravity = gravity_f[[4, 4, 9]]
    position = position_f[[4, 4, 9]]
    height = np.array([height_f[4], height_f[4] + .2, 2.])
    nearest = task.track_nearest_frame(gravity, position, height, gravity_f, position_f, height_f, joints)
    angle, pose, _ = task.settle_errors(gravity, position, gravity_f, position_f, joints)
    score = np.exp(-angle ** 2 / task.SETTLE_ORI_STD ** 2) + np.exp(-pose / task.SETTLE_POSE_STD ** 2)
    reach = np.abs(height_f - height[1]) <= task.TRACK_NEAREST_ANCHOR
    assert not reach[4] and reach.any()
    assert nearest.tolist() == [4, int(np.flatnonzero(reach)[score[1, reach].argmax()]), 0]
    tensors = [torch.from_numpy(a) for a in (gravity, position, height, gravity_f, position_f, height_f)]
    assert task.track_nearest_frame(*tensors, joints).tolist() == nearest.tolist()
    assert task.TRACK_NEAREST_ANCHOR == task.TRACK_ANCHOR_Z / 2


def test_a_bank_start_at_its_nearest_frame():
    """A bank row in the synthetic clip's frame-20 pose starts with track_bank_nearest at frame 20: the clock there,
    the reference's pelvis on the robot's (track_align to that frame), the episode ends at the clip's end + the hold;
    without the setting the same row starts at frame 0."""
    clip = synthetic_clip()
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0] = clip['qpos'][20]
    for nearest, frame in ((True, 20), (False, 0)):
        env = G1GetupEnv(bank=bank, track_clip=SETTING, track_bank=clip, track_bank_nearest=nearest)
        try:
            _, info = env.reset(seed=0, options=dict(OPTIONS, start=0))
            assert info['track_start'] == 'bank' and info['track_frame'] == frame and env.clip_clock == frame
            position, rotation, _, _ = env._clip_reference()
            np.testing.assert_allclose(position[0, 0, :2], env.data.xpos[env.pelvis, :2], atol=1e-9)
            turn = float(task.track_heading(rotation[:, 0])[0]
                         - task.track_heading(env.data.xmat[env.pelvis].reshape(1, 3, 3))[0])
            assert math.cos(turn) == pytest.approx(1., abs=1e-12)
            if nearest:  # the reference is the clip's own frame 20, where the robot is
                np.testing.assert_allclose(position[0], env.data.xpos[env.track_bodies], atol=1e-6)
            *_, info = env.step(np.zeros(29))
            assert info['track_clock'] == frame + 1 and info['track_start'] == 'bank'
        finally:
            env.close()
    assert FRAMES > 21
