"""g1_getup task_config track_end_reach (iteration 78, the critic's it78), CPU: an episode of motion tracking also ends
when an ankle lies farther than the setting from its place in the clip re-anchored under the robot (3D, body_pos'
frame), a bank start's first TRACK_GRACE_SECONDS excepted. The setting (bounds, needs a clip, not part of the
contract); off (missing or null) the task is bit for bit the old one; the rule by hand, numpy and torch alike (0.39 m
does not end, 0.41 m does, the hands and the other bodies do not count, the re-anchoring); in the task with track_clip
and track_clips (info track_end_reach and track_reach, the grace). CPU–GPU: tests/test_g1_gpu.py -k track_end_reach."""
import math

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, TRACK_END_REACH_RANGE
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_track import SETTING, synthetic_clip, track_env
from test_g1_getup_track_clips import SETTINGS, front_clip
from test_g1_getup_track_sampling import OLD_EPISODES_DIGEST, episodes_digest
from test_g1_getup_track_starts import OLD_STARTS_DIGEST, starts_digest

WORKING = .4
LEFT_ANKLE, RIGHT_ANKLE = (task.TRACK_BODIES.index(name) for name in task.TRACK_REACH_BODIES)
ANCHOR = task.TRACK_BODIES.index(task.TRACK_ANCHOR)


def test_the_setting_is_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_end_reach'] is None  # old stages: off
    for tracking in ({'track_clip': SETTING}, {'track_clips': SETTINGS}):
        on = validate_config({**base, 'task_config': {**tracking, 'track_end_reach': WORKING}})['task_config']
        assert on['track_end_reach'] == WORKING
        assert validate_config({**base, 'task_config': {**tracking, 'track_end_reach': 1}})['task_config'][
            'track_end_reach'] == 1. and validate_config({**base, 'task_config': {
                **tracking, 'track_end_reach': None}})['task_config']['track_end_reach'] is None
        for value in (.05, 2.5, -1, math.inf, math.nan, True, '0.4', [.4]):
            with pytest.raises(ValueError):
                validate_config({**base, 'task_config': {**tracking, 'track_end_reach': value}})
        assert task_identity('g1_29dof', 'g1_getup', {**tracking, 'track_end_reach': WORKING}) == task_identity(
            'g1_29dof', 'g1_getup', tracking)
    with pytest.raises(ValueError, match='track_clip'):
        validate_config({**base, 'task_config': {'track_end_reach': WORKING}})  # nothing to reach
    assert TRACK_END_REACH_RANGE == (.1, 2.) and task.TRACK_REACH_BODIES == ('left_ankle_roll_link',
                                                                           'right_ankle_roll_link')
    assert set(task.TRACK_REACH_BODIES) < set(task.TRACK_END_BODIES)  # the ends' ankles, not their hands
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['track_end_reach']
    assert field['group'] is None and field['default'] is None and (field['min'], field['max']) == (.1, 2.)
    assert 'лодыжка' in field['label']


def test_off_the_task_is_bit_for_bit_the_old_one():
    """Missing or null: the old digests (getting up, track_clip's starts and its RSI episodes) and no reach info; on
    at 2 m, which no ankle reaches in these episodes, the same episodes too (observations, rewards, ends)."""
    assert getup_digest({'track_end_reach': None}, drop_bank()) == OLD_DIGESTS['plain']
    assert starts_digest(track_end_reach=None) == OLD_STARTS_DIGEST
    assert episodes_digest(track_end_reach=None) == OLD_EPISODES_DIGEST
    assert episodes_digest(track_end_reach=2.) == OLD_EPISODES_DIGEST
    env = track_env()
    try:
        assert env.track_reach is None
        env.reset(seed=0, options=dict(OPTIONS, track_frame=5, track_noise=False))
        *_, info = env.step(np.zeros(29))
        assert 'track_end_reach' not in info and 'track_reach' not in info
    finally:
        env.close()


def _pose(n=1):
    """A lying robot's 14 tracked bodies at made-up places and a reference exactly there (turned and shifted as a
    reference may be: the re-anchoring takes the robot anchor's xy and the yaw between the anchors)."""
    rng = np.random.default_rng(7)
    position = rng.uniform(-.6, .6, (n, len(task.TRACK_BODIES), 3))
    position[..., 2] = np.abs(position[..., 2]) * .3
    rotation = np.repeat(np.eye(3)[None, None], len(task.TRACK_BODIES), 1).repeat(n, 0)
    return position, rotation


def _ended(position, rotation, ref_position, ref_rotation, active=None, reach=WORKING):
    distance = task.track_reach_distance(position, rotation, ref_position, ref_rotation, ANCHOR,
                                         (LEFT_ANKLE, RIGHT_ANKLE))
    return task.track_reach_terminated(distance, reach, active), distance


def test_the_rule_by_hand_numpy_and_torch():
    """An ankle 0.39 m from its place does not end, 0.41 m does, in any direction (a 3D distance: up counts too); the
    hands and the other bodies 1 m off do not; a reference turned about z and moved in xy as a whole is the robot's
    place (re-anchored, distance 0); where active is false nothing ends; torch gives the same."""
    position, rotation = _pose()
    reference = position.copy()
    turn = task.yaw_rotation(np.array([.7]))
    shifted = np.einsum('ij,nbj->nbi', turn[0], reference) + np.array([1.5, -2., .0])  # the clip elsewhere
    turned = np.einsum('ij,nbjk->nbik', turn[0], rotation)
    for ref_position, ref_rotation in ((reference, rotation), (shifted, turned)):
        ended, distance = _ended(position, rotation, ref_position, ref_rotation)
        assert not ended[0] and distance.max() < 1e-12
    for body in (LEFT_ANKLE, RIGHT_ANKLE):
        for direction in ([1., 0., 0.], [0., -1., 0.], [0., 0., 1.], [.6, -.48, .64]):
            unit = np.array(direction) / np.linalg.norm(direction)
            for gap, expected in ((.39, False), (.41, True)):
                moved = position.copy()
                moved[0, body] += gap * unit
                ended, distance = _ended(moved, rotation, reference, rotation)
                assert bool(ended[0]) is expected and distance[0].max() == pytest.approx(gap, abs=1e-12)
                assert not _ended(moved, rotation, reference, rotation, active=np.array([False]))[0][0]
                # torch alike, the reference turned and moved
                moved_turned = np.einsum('ij,nbj->nbi', turn[0], moved) + np.array([1.5, -2., .0])
                tensors = [torch.from_numpy(a) for a in (moved_turned, turned, shifted, turned)]
                ended_t, distance_t = _ended(*tensors)
                assert bool(ended_t[0]) is expected
                np.testing.assert_allclose(distance_t.numpy(), distance, atol=1e-12)
                assert not bool(_ended(*tensors, active=torch.tensor([False]))[0][0])
    for name in ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_knee_link', 'pelvis'):
        moved = position.copy()
        moved[0, task.TRACK_BODIES.index(name)] += [1., 0., 0.]
        assert not _ended(moved, rotation, reference, rotation)[0][0], name
    # The reference's own distance (body_pos' frame): track_errors' body_pos is the mean of its squares over the bodies.
    moved = position.copy()
    moved[0, LEFT_ANKLE] += [.3, .2, 0.]
    relative, _ = task.track_relative(moved, rotation, reference, rotation, ANCHOR)
    body_pos = task.track_errors(moved, rotation, moved, moved, reference, rotation, moved, moved, ANCHOR)['body_pos']
    assert body_pos[0] == pytest.approx(((relative - moved) ** 2).sum(-1).mean(), abs=1e-15)
    assert body_pos[0] == pytest.approx(.13 / len(task.TRACK_BODIES), abs=1e-12)


def _shift_ankle(env, offset, start=0):
    """The env's reference frames (its clip_frames) with the left ankle moved by offset (m, world x) from clip frame
    start on; the tracking reward sees it too, the start poses (qpos) do not."""
    pos = env.clip_frames['pos'].copy()
    pos[start:, LEFT_ANKLE, 0] += offset
    env.clip_frames = {**env.clip_frames, 'pos': pos}


def _follow(env, steps):
    """Steps toward the clip's next frames: (step, terminated, info) of each, until an end."""
    out = []
    for step in range(1, steps + 1):
        frame = min(env.clip_clock + 1, env.clip_last)
        *_, terminated, truncated, info = env.step((env.clip['qpos'][frame, 7:] - env.home) / env.scale)
        out.append((step, terminated, info))
        if terminated or truncated:
            break
    return out


def test_in_the_task_an_ankle_too_far_ends_the_episode():
    """track_clip with track_end_reach 0.4, followed from frame 10 without noise: every step's info carries
    track_reach (the larger ankle distance) and track_end_reach; the clip followed as it is ends nothing; with the
    reference's left ankle 0.1 m away nothing ends either, 0.6 m away from frame 15 on the step that reaches it ends
    with track_failed and track_end_reach, while the heights' rules alone (the setting off) go on."""
    env = track_env(track_end_reach=WORKING)
    plain = track_env()
    try:
        env.reset(seed=1, options=dict(OPTIONS, track_frame=10, track_noise=False))
        run = _follow(env, 20)
        assert len(run) == 20 and not any(terminated for _, terminated, _ in run)
        # the PD lags the synthetic clip's roll: the ankles up to ~0.22 m off
        assert all(not info['track_end_reach'] and 0 <= info['track_reach'] < .25 for *_, info in run)
        for offset, start, ends in ((.1, 0, None), (.6, 15, 5)):
            for one in (env, plain):
                one.reset(seed=1, options=dict(OPTIONS, track_frame=10, track_noise=False))
                _shift_ankle(one, offset, start)
            run = _follow(env, 20)
            ended = [step for step, terminated, _ in run if terminated]
            assert ended == ([] if ends is None else [ends]), (offset, ended)
            for step, terminated, info in run:
                assert info['track_end_reach'] is (info['track_reach'] > WORKING) is terminated
                assert info['track_failed'] is terminated
            if ends is not None:
                assert run[-1][2]['track_reach'] == pytest.approx(offset, abs=.15)
            assert not any(terminated for _, terminated, _ in _follow(plain, 20))
    finally:
        env.close()
        plain.close()


def test_a_bank_start_waits_its_grace():
    """A bank row in the clip's first pose (the bank start aligned there, clock 0), the reference's left ankle 0.6 m
    away throughout: no end while the bank start's grace lasts (TRACK_GRACE_SECONDS, 25 steps), the end on the step
    after it, by the reach; a clip start with the same reference ends on its first step."""
    clip = synthetic_clip()
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0] = clip['qpos'][0]
    env = G1GetupEnv(bank=bank, track_clip=SETTING, track_bank=clip, track_end_reach=WORKING)
    try:
        assert env.clip_grace == round(task.TRACK_GRACE_SECONDS / env.dt) == 25
        _, info = env.reset(seed=0, options=dict(OPTIONS, start=0))
        assert info['track_start'] == 'bank' and env.clip_clock == 0
        _shift_ankle(env, .6)
        run = _follow(env, 40)
        assert [step for step, terminated, _ in run if terminated] == [env.clip_grace + 1]
        assert all(info['track_reach'] > WORKING for *_, info in run)
        assert not any(info['track_end_reach'] for *_, info in run[:-1]) and run[-1][2]['track_end_reach']
        env.reset(seed=0, options=dict(OPTIONS, track_frame=0, track_noise=False))
        _shift_ankle(env, .6)
        run = _follow(env, 5)
        assert len(run) == 1 and run[0][1] and run[0][2]['track_end_reach']
    finally:
        env.close()


def test_with_track_clips_the_rule_follows_each_episodes_clip():
    """track_clips with track_end_reach: each clip followed from a frame ends nothing; the episode's own clip's
    reference moved (its ankle 0.6 m away) ends on the first step by the reach, whichever clip it is."""
    clips = [synthetic_clip(), front_clip()]
    env = G1GetupEnv(bank=drop_bank(), track_clips=SETTINGS, track_clips_bank=clips, track_end_reach=WORKING)
    try:
        for index in (0, 1):
            env.reset(seed=2, options=dict(OPTIONS, track_clip_index=index, track_frame=5, track_noise=False))
            run = _follow(env, 10)
            assert not any(terminated for _, terminated, _ in run) and all(
                info['track_clip_index'] == index and not info['track_end_reach'] for *_, info in run)
            env.reset(seed=2, options=dict(OPTIONS, track_clip_index=index, track_frame=5, track_noise=False))
            _shift_ankle(env, .6)
            run = _follow(env, 10)
            assert len(run) == 1 and run[0][1] and run[0][2]['track_end_reach'] and run[0][2]['track_failed']
    finally:
        env.close()
