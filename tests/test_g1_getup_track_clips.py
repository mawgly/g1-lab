"""g1_getup task_config track_clips (iteration 78; Pavel 27.10 after SuperMimic: one tracker of many human rises), CPU.
The setting and its own critic contract (…-critic6-track114); off (missing or null) the task is bit for bit the old one;
a clip of the list followed from a given frame is the old track_clip with that clip, bit for bit on the actor's inputs,
the first 17 critic inputs, the rewards and the terminations; the clip and frame draws of the training starts; a bank
start's nearest clip and frame among the clips of its class, and the fallback; the critic's reference ahead by hand,
numpy and torch alike; track_sampling's bins over many clips; warm_start.insert_terms moves a track_clip stage to the
new contract acting and valuing as the source on the old columns; package.py checks every clip's licence. CPU–GPU:
tests/test_g1_gpu.py -k track_clips."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_getup_track
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import (G1_GETUP_PARAMETERS, G1_GETUP_TRACK_WEIGHTS, GETUP_CRITIC_EXTRAS,
                                       TRACK_CLIPS_CRITIC_EXTRAS, TRACK_CRITIC_EXTRAS)
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup import small_bank  # noqa: F401 (fixture)
from test_g1_getup_impact import DROP, OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_track import FRAMES, SETTING, synthetic_clip
from test_g1_getup_track_sampling import OLD_EPISODES_DIGEST, episodes_digest
from test_g1_getup_track_starts import OLD_STARTS_DIGEST, starts_digest

SETTINGS = [{'file': 'test-back', 'md5': '0' * 32}, {'file': 'test-front', 'md5': '1' * 32}]
FRONT_FRAMES = 30
OLD_COLUMNS = 490 + GETUP_CRITIC_EXTRAS + TRACK_CRITIC_EXTRAS  # 507: track_clip's observation


def front_clip(frames=FRONT_FRAMES):
    """As test_g1_getup_track.synthetic_clip, from drop_bank's front row (row 1): pressed 3 cm into the floor, moving
    0.2 m sideways while every joint swings by up to 0.2 rad."""
    env = bank_module.getup_model_env()
    start = drop_bank()['qpos'][1].copy()
    start[2] -= DROP + .03
    t = np.linspace(0., 1., frames)
    qpos = np.repeat(start[None], frames, 0)
    qpos[:, 1] += .2 * t
    qpos[:, 7:] += np.sin(np.pi * t)[:, None] * np.random.default_rng(5).uniform(-.2, .2, 29)
    return g1_getup_track.build(env, qpos)


@pytest.fixture(scope='module')
def clips():
    return [synthetic_clip(), front_clip()]


def clips_env(clips, bank=None, **settings):
    return G1GetupEnv(bank=drop_bank() if bank is None else bank, track_clips=SETTINGS, track_clips_bank=clips,
                      **settings)


def test_the_setting_is_validated_labelled_and_its_own_critic_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_clips'] is None  # old stages: off
    on = validate_config({**base, 'task_config': {'track_clips': SETTINGS}})['task_config']
    assert on['track_clips'] == SETTINGS and on['track_clip'] is None
    many = [{'file': f'c{k}', 'md5': f'{k:032x}'} for k in range(65)]
    for value in (SETTINGS[:1], many, many[:64] + many[:1], [SETTINGS[0], SETTINGS[0]],
                  [SETTINGS[0], {'file': 'x', 'md5': SETTINGS[0]['md5']}], [SETTINGS[0], {'file': '../x', 'md5': '2' * 32}],
                  [SETTINGS[0], {'file': 'x'}], [SETTINGS[0], {'file': 'x', 'md5': '2' * 32, 'w': 1}], SETTINGS[0],
                  'x', 5, True, []):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'track_clips': value}})
    assert validate_config({**base, 'task_config': {'track_clips': many[:64]}})['task_config']['track_clips'] == many[:64]
    for other in ({'track_clip': SETTING}, {'settle_clip': [SETTING]}, {'pd_bound': [1., .6]},
                  {'human_starts': {'share': .3, 'md5': '0' * 32}}, {'style_judge': True},
                  {'reward_weights': {'style': .5}}, {'reward_weights': {'amp': 1.}}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'track_clips': SETTINGS, **other}})
    sampling = {'bin': 1., 'uniform': .5, 'hard': .4, 'history': .1, 'temperature': 25}
    starts = validate_config({**base, 'task_config': {
        'track_clips': SETTINGS, 'track_starts': {'first': .4, 'rsi': .4, 'bank': .2}, 'track_noise': 2.,
        'track_bank_classes': ['back', 'front', 'side'], 'track_sampling': sampling}})['task_config']
    assert starts['track_starts']['bank'] == .2 and starts['track_sampling']['hard'] == .4
    plain = task_identity('g1_29dof', 'g1_getup', {})
    assert task_identity('g1_29dof', 'g1_getup', {'track_clips': None}) == plain
    tracking = task_identity('g1_29dof', 'g1_getup', {'track_clips': SETTINGS})
    assert tracking['observation_schema'] == plain['observation_schema'] + '-track114'
    assert TRACK_CLIPS_CRITIC_EXTRAS == 11 + 1 + 3 * (3 + 6 + 25) == 114
    assert {k: v for k, v in tracking.items() if k != 'observation_schema'} == {
        k: v for k, v in plain.items() if k != 'observation_schema'}
    assert task_identity('g1_29dof', 'g1_getup', {'track_clips': SETTINGS, 'track_starts': {
        'first': .4, 'rsi': .4, 'bank': .2}}) == tracking  # the starts are not part of the contract
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['track_clips']
    assert field['group'] is None and field['default'] is None and field['type'] == 'json'
    assert 'многим записям' in field['label']


def test_off_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('a track clip read with the setting off')
    assert starts_digest(track_clips=None) == OLD_STARTS_DIGEST
    assert episodes_digest(track_clips=None) == OLD_EPISODES_DIGEST
    monkeypatch.setattr(g1_getup_track, 'load', refuse)
    assert getup_digest({'track_clips': None}, drop_bank()) == OLD_DIGESTS['plain']
    env = G1GetupEnv(bank=drop_bank())
    try:
        assert env.clips is None and env.observation_space.shape == (490 + GETUP_CRITIC_EXTRAS,)
    finally:
        env.close()
    monkeypatch.undo()
    env = G1GetupEnv(bank=drop_bank(), track_clip=SETTING, track_bank=synthetic_clip())
    try:
        assert env.clips is None
        with pytest.raises(ValueError, match='track_clip_index'):
            env.reset(seed=0, options=dict(OPTIONS, track_clip_index=0))  # one clip: no index
    finally:
        env.close()


def _same_episode(many, one, options, steps=12, seed=3):
    """The track_clips task `many` and the track_clip task `one` from the same reset (options: many's track_clip_index
    besides) under the same random actions: qpos, the actor's inputs and the first 17 critic inputs, rewards, reward
    terms, ends and the track info bit for bit; returns many's last observation."""
    rng = np.random.default_rng(seed)
    first = {key: value for key, value in options.items() if key != 'track_clip_index'}
    a, info_a = many.reset(seed=seed, options=options)
    b, info_b = one.reset(seed=seed, options=first)
    assert np.array_equal(many.data.qpos, one.data.qpos) and np.array_equal(a[:OLD_COLUMNS], b)
    for key in ('track_start', 'track_frame', 'start', 'start_class', 'track_lift'):
        assert info_a[key] == info_b[key], key
    for _ in range(steps):
        action = rng.uniform(-2., 2., 29)
        a, reward_a, terminated_a, truncated_a, info_a = many.step(action)
        b, reward_b, terminated_b, truncated_b, info_b = one.step(action)
        assert np.array_equal(a[:OLD_COLUMNS], b) and reward_a == reward_b
        assert (terminated_a, truncated_a) == (terminated_b, truncated_b)
        assert info_a['reward_components'] == info_b['reward_components']
        for key in ('track_clock', 'track_done', 'track_failed', 'track_standing', 'hand_force', 'track_errors'):
            assert info_a[key] == info_b[key], key
        if terminated_a or truncated_a:
            break
    return a


def test_a_clip_of_the_list_is_track_clip_with_that_clip(clips):
    """From a given clip frame (track_clip_index + track_frame, with and without noise; an evaluation reset's noisy
    first frame) each clip of the list is the old track_clip task with that clip, bit for bit on the old 507 inputs;
    a bank row lying on the back takes the back clip at the frame track_bank_nearest gives (the front clip is another
    class); the observation is 114 critic inputs longer."""
    many = clips_env(clips)
    singles = [G1GetupEnv(bank=drop_bank(), track_clip=SETTING, track_bank=clip, track_bank_nearest=True)
               for clip in clips]
    try:
        assert many.observation_space.shape == (490 + GETUP_CRITIC_EXTRAS + TRACK_CLIPS_CRITIC_EXTRAS,)
        assert many.clip_classes.tolist() == [0, 1]  # back, front
        for index, one in enumerate(singles):
            for frame, noise in ((0, False), (7, False), (12, True)):
                obs = _same_episode(many, one, dict(OPTIONS, track_clip_index=index, track_frame=frame,
                                                    track_noise=noise))
                assert obs.shape == many.observation_space.shape
            _same_episode(many, one, dict(OPTIONS, track_clip_index=index))  # the noisy first frame
            assert many.clip_index == index
        _same_episode(many, singles[0], dict(OPTIONS, start=0))
        assert many.clip_index == 0 and many.clip_start == 'bank'
    finally:
        many.close()
        for one in singles:
            one.close()


def test_the_training_draws_of_clip_and_frame(clips):
    """Over 6000 training resets (track_starts 0.4 / 0.4 / 0.2, the bank's back and front rows): a first-frame start's
    clip is uniform over the clips; an RSI frame is uniform over every clip's frames (the back clip 40 of 70), every
    frame of each clip drawn; a bank row lying on the back follows the back clip, on the front the front one; the
    episode's end is its own clip's (last frame + the hold); an evaluation reset draws its clip uniformly too."""
    env = clips_env(clips, track_starts={'first': .4, 'rsi': .4, 'bank': .2}, track_bank_classes=['back', 'front'])
    try:
        assert env.rsi_frames == FRAMES + FRONT_FRAMES and env.rsi_starts.tolist() == [0, FRAMES]
        env.reset(seed=0)
        kinds, first, rsi, frames = {}, np.zeros(2), np.zeros(2), [set(), set()]
        for _ in range(6000):
            _, info = env.reset()
            kind, index = info['track_start'], info['track_clip_index']
            kinds[kind] = kinds.get(kind, 0) + 1
            assert env.clip_end == env.clips[index]['last'] + 150 and env.clip_last == env.clips[index]['last']
            if kind == 'clip':
                first[index] += 1
            elif kind == 'frame':
                rsi[index] += 1
                frames[index].add(info['track_frame'])
            else:
                assert index == {'back': 0, 'front': 1}[info['start_class']] and info['track_clip_class'] == \
                    info['start_class']
        for kind, share in (('clip', .4), ('frame', .4), ('bank', .2)):
            assert abs(kinds[kind] / 6000 - share) < .025, kinds
        assert abs(first[0] / first.sum() - .5) < .04
        assert abs(rsi[0] / rsi.sum() - FRAMES / (FRAMES + FRONT_FRAMES)) < .04
        assert frames[0] == set(range(FRAMES)) and frames[1] == set(range(FRONT_FRAMES))
        picked = [env.reset(seed=k, options=dict(OPTIONS))[1]['track_clip_index'] for k in range(200)]
        assert 70 < sum(picked) < 130
    finally:
        env.close()


def test_the_nearest_clip_by_hand_numpy_and_torch():
    """track_nearest_clip: the largest settle kernel sum among the frames of the clips of the start's class whose
    anchor lies within TRACK_NEAREST_ANCHOR of the start's; every frame where no clip has that class; (0, not found)
    when none is within reach. track_start_class, track_locate, track_clip_frames and track_clips_index by hand."""
    rng = np.random.default_rng(2)
    frames, joints = 20, tuple(range(25))
    gravity_f = rng.normal(size=(frames, 3))
    gravity_f /= np.linalg.norm(gravity_f, axis=1, keepdims=True)
    position_f = rng.uniform(-1., 1., (frames, 29))
    height_f = np.full(frames, .3)
    kind_f = np.array([0] * 8 + [1] * 12)  # clip 0 (8 frames) on the back, clip 1 (12) on the front
    # Starts: frame 12's pose as a back row (only clip 0's frames count), frame 12's pose as a front row (frame 12),
    # frame 3's pose as a side row (no side clip: every frame), anything 1 m up (none within reach).
    gravity = gravity_f[[12, 12, 3, 3]]
    position = position_f[[12, 12, 3, 3]]
    height = np.array([.3, .3, .3, 1.3])
    start_class = np.array([0, 1, 2, 0])
    frame, found = task.track_nearest_clip(gravity, position, height, start_class, gravity_f, position_f, height_f,
                                           kind_f, joints)
    angle, pose, _ = task.settle_errors(gravity, position, gravity_f, position_f, joints)
    score = np.exp(-angle ** 2 / task.SETTLE_ORI_STD ** 2) + np.exp(-pose / task.SETTLE_POSE_STD ** 2)
    assert frame.tolist() == [int(score[0, :8].argmax()), 12, 3, 0] and found.tolist() == [True, True, True, False]
    tensors = [torch.from_numpy(a) for a in (gravity, position, height, start_class, gravity_f, position_f, height_f,
                                             kind_f)]
    frame_t, found_t = task.track_nearest_clip(*tensors, joints)
    assert frame_t.tolist() == frame.tolist() and found_t.tolist() == found.tolist()
    g = np.array([[-1., 0., 0.], [.9, .1, -.1], [.1, -.8, -.2], [.1, .2, -.9], [-.5, .6, 0.]])
    assert task.track_start_class(g).tolist() == [0, 1, 2, -1, 2]
    assert task.track_start_class(torch.from_numpy(g)).tolist() == [0, 1, 2, -1, 2]
    starts, total = task.track_clip_frames([40, 30, 5])
    assert starts.tolist() == [0, 40, 70] and total == 75
    clip, local = task.track_locate(np.array([0, 39, 40, 69, 70, 74]), starts)
    assert clip.tolist() == [0, 0, 1, 1, 2, 2] and local.tolist() == [0, 39, 0, 29, 0, 4]
    clip_t, local_t = task.track_locate(torch.tensor([0, 39, 40, 69, 70, 74]), torch.from_numpy(starts))
    assert clip_t.tolist() == clip.tolist() and local_t.tolist() == local.tolist()
    # Clip 1 (frames 41..71 with its hold row at 71 when laid with their hold rows): the clock held at last + 1.
    index = task.track_clips_index(np.array([41, 41, 41]), np.array([-3, 10, 99]), np.array([29, 29, 29]))
    assert index.tolist() == [41, 51, 71]
    assert task.track_clips_index(torch.tensor([41]), torch.tensor([99]), torch.tensor([29])).tolist() == [71]


def test_a_bank_row_beyond_every_clip_starts_a_clip_of_its_class_at_frame_0(clips):
    """drop_bank's standing row (bank class stand: no clip class) lies far above every lying frame: no nearest frame,
    so frame 0 of a clip drawn from every clip (the stand class has none); a back row whose pose is lifted 1 m (out of
    reach) starts frame 0 of the back clip."""
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0, 2] += 1.
    env = clips_env(clips, bank=bank)
    try:
        _, _, _, kinds = env.bank_row_poses([0, 1, 3])
        assert kinds.tolist() == [0, 1, -1]
        seen = set()
        for seed in range(30):
            _, info = env.reset(seed=seed, options=dict(OPTIONS, start=3))
            assert info['track_frame'] == 0 and env.clip_clock == 0
            seen.add(info['track_clip_index'])
            _, info = env.reset(seed=seed, options=dict(OPTIONS, start=0))
            assert info['track_frame'] == 0 and info['track_clip_index'] == 0
        assert seen == {0, 1}
        with pytest.raises(ValueError, match='start'):
            env.reset(seed=0, options=dict(OPTIONS, start=0, track_clip_index=1))
        for bad in (2, -1, True, 1.):
            with pytest.raises(ValueError, match='track_clip_index'):
                env.reset(seed=0, options=dict(OPTIONS, track_clip_index=bad))
    finally:
        env.close()


def test_the_critic_sees_the_reference_ahead_by_hand_and_its_normalization_is_fixed(clips):
    """track_clips_critic_extras against a hand calculation (numpy and torch alike); in the task the 114 inputs after
    the six are it of the clip's own frames (the reference turned and moved as the episode's), the seconds left to the
    hold's end, the anchor ahead at 5, 25 and 50 steps held at the hold frame; fixed_normalization lists the 103 new
    columns at task.track_clips_normalization's numbers."""
    rng = np.random.default_rng(4)
    n, horizons = 6, 3
    position, ref_position = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    rotation = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    ref_rotation = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    future_position = rng.normal(size=(n, horizons, 3))
    future_rotation = np.stack([[np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(horizons)] for _ in range(n)])
    future_joints = rng.normal(size=(n, horizons, 25))
    clock, last, end = np.arange(n) * 30, np.array([39, 29, 39, 29, 39, 29]), np.array([189, 179, 189, 179, 189, 179])
    extras = task.track_clips_critic_extras(clock, last, end, .02, position, rotation, ref_position, ref_rotation,
                                            future_position, future_rotation, future_joints)
    assert extras.shape == (n, TRACK_CLIPS_CRITIC_EXTRAS)
    np.testing.assert_allclose(extras[:, :11], task.track_critic_extras(clock, last, position, rotation, ref_position,
                                                                        ref_rotation), atol=1e-12)
    np.testing.assert_allclose(extras[:, 11], np.maximum(end - clock, 0) * .02, atol=1e-12)
    for w in range(n):
        for h in range(horizons):
            block = extras[w, 12 + 34 * h:12 + 34 * (h + 1)]
            np.testing.assert_allclose(block[:3], rotation[w].T @ (future_position[w, h] - position[w]), atol=1e-12)
            np.testing.assert_allclose(block[3:9], (rotation[w].T @ future_rotation[w, h])[:, :2].ravel(), atol=1e-12)
            np.testing.assert_allclose(block[9:], future_joints[w, h], atol=1e-12)
    tensors = [torch.from_numpy(a) for a in (clock, last, end)]
    rest = [torch.from_numpy(a) for a in (position, rotation, ref_position, ref_rotation, future_position,
                                          future_rotation, future_joints)]
    np.testing.assert_allclose(task.track_clips_critic_extras(*tensors, .02, *rest).numpy(), extras, atol=1e-12)
    assert task.track_future_steps(.02) == (5, 25, 50)
    env = clips_env(clips)
    try:
        obs, _ = env.reset(seed=0, options=dict(OPTIONS, track_clip_index=1, track_frame=FRONT_FRAMES - 10,
                                                track_noise=False))
        extras = obs[490 + GETUP_CRITIC_EXTRAS:]
        assert extras.shape == (TRACK_CLIPS_CRITIC_EXTRAS,)
        assert extras[11] == pytest.approx((FRONT_FRAMES - 1 + 150 - (FRONT_FRAMES - 10)) * .02, rel=1e-6)
        frames = g1_getup_track.hold_frames(clips[1])
        anchor = env.track_anchor
        body = env.track_bodies[anchor]
        inverse = env.data.xmat[body].reshape(3, 3).T
        for h, steps in enumerate((5, 25, 50)):
            ahead = min(FRONT_FRAMES - 10 + steps, FRONT_FRAMES)  # the hold frame from the 10th step on
            block = extras[12 + 34 * h:12 + 34 * (h + 1)]
            np.testing.assert_allclose(block[:3], inverse @ (frames['pos'][ahead, anchor] - env.data.xpos[body]),
                                       atol=1e-5)
            free = list(env.nearest_joints)
            q = clips[1]['qpos'][min(ahead, FRONT_FRAMES - 1), 7:]
            np.testing.assert_allclose(block[9:], (q - env.home)[free], atol=1e-6)
        columns, mean, variance = env.fixed_normalization
        first = 490 + GETUP_CRITIC_EXTRAS + TRACK_CRITIC_EXTRAS
        assert columns[-103:].tolist() == list(range(first, first + 103))
        expected_mean, expected_variance = task.track_clips_normalization()
        np.testing.assert_array_equal(mean[-103:], expected_mean)
        np.testing.assert_array_equal(variance[-103:], expected_variance)
        assert expected_mean[0] == 4. and expected_variance[0] == 9. and len(columns) == 25 + 103
    finally:
        env.close()


def test_track_sampling_bins_lie_within_each_clip():
    """TrackBinSampler over clips of 40 and 30 frames at bins of 0.5 s: [0, 25) [25, 40) of the first, [40, 65)
    [65, 70) of the second, none across the two; its draws locate back into each clip's own frames; one clip (a
    number) as before."""
    starts, lengths = task.track_clip_bins([40, 30], 25)
    assert starts.tolist() == [0, 25, 40, 65] and lengths.tolist() == [25, 15, 25, 5]
    setting = {'bin': .5, 'uniform': 1., 'hard': 0., 'history': 0., 'temperature': 25.}
    kernel = sum(G1_GETUP_TRACK_WEIGHTS[key] for key in task.TRACK_SIGMA)
    sampler = task.TrackBinSampler(setting, [40, 30], .02, kernel, 4, 'cpu', 0)
    assert sampler.bins == 4
    _, bins, frames = sampler.draw(20_000)
    clip, local = task.track_locate(frames, torch.tensor([0, 40]))
    assert bool(((bins < 2) == (clip == 0)).all()) and int(local[clip == 1].max()) == 29
    assert int(local[clip == 0].max()) == 39 and int(frames.max()) == 69
    one = task.TrackBinSampler(setting, 40, .02, kernel, 4, 'cpu', 0)
    assert one.starts.tolist() == [0, 25] and one.lengths.tolist() == [25, 15]


def _write_clip(folder, name, clip, dataset='bones_seed'):
    """A track clip file as getup_track_clip.py writes it (arrays, the header's version, bodies and dataset); its md5."""
    header = dict(version=g1_getup_track.TRACK_VERSION, bodies=list(task.TRACK_BODIES), rate=50, dataset=dataset,
                  source=f'{name}.csv', frames=[0, len(clip['qpos']) - 1])
    path = folder / f'{name}.npz'
    np.savez(path, **{key: clip[key] for key in g1_getup_track.ARRAYS}, header=np.array(json.dumps(header)))
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_insert_terms_moves_a_track_clip_stage_to_track_clips(tmp_path, monkeypatch, clips, small_bank):  # noqa: F811
    """The branch from a one-clip tracker (branch 73 / 77): insert_terms takes a track_clip stage to track_clips with
    the same clip first; stage 0 passes the new contract, keeps steps, the old columns' statistics and the
    training_state, has zero weights on the 103 new critic columns and the fixed normalization there, and acts and
    values exactly as the source on the old inputs (the actor never sees the critic's)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.checkpoint_schema import checkpoint_identity, validate_resume
    from humanoid_sim.core import load_stage, read_json, save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    from humanoid_sim.tasks.identity import task_identity
    from humanoid_sim.warm_start import check_insert_terms, insert_terms
    monkeypatch.setattr(g1_getup_track, 'TRACK_DIR', tmp_path)
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    back = {'file': 'clip-back', 'md5': _write_clip(tmp_path, 'clip-back', clips[0])}
    front = {'file': 'clip-front', 'md5': _write_clip(tmp_path, 'clip-front', clips[1])}

    def run(name, task_config):
        folder = tmp_path / name
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', n_steps=8, batch_size=8,
                                      task_config=task_config))
        write_json(folder / 'run_config.json', config)
        return folder, config
    source_run, config = run('old', {'track_clip': back, 'assist_force': 0.})
    env = VecNormalize(DummyVecEnv([lambda: G1GetupEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        rng = np.random.default_rng(0)  # statistics that matter, as a trained stage's
        env.obs_rms.mean = rng.normal(0., .3, OLD_COLUMNS)
        env.obs_rms.var = rng.uniform(.2, 2., OLD_COLUMNS)
        env.obs_rms.count = 5.7e8
        columns, mean, variance = env.venv.envs[0].fixed_normalization
        env.obs_rms.mean[columns], env.obs_rms.var[columns] = mean, variance
        stage = save_stage(source_run, model, env, {'assist_force': [0.]})
    finally:
        env.close()
    target, config = run('new', {'track_clips': [back, front], 'assist_force': 0.})
    saved = read_json(stage / 'metadata.json')
    check_insert_terms(saved, task_identity('g1_29dof', 'g1_getup', config['task_config']))
    moved = insert_terms(stage, target)
    metadata = read_json(moved / 'metadata.json')
    validate_resume(checkpoint_identity(metadata), task_identity('g1_29dof', 'g1_getup', config['task_config']))
    assert metadata['observation_schema'].endswith('-critic6-track114') and metadata['steps'] == saved['steps']
    assert metadata['training_state'] == {'assist_force': [0.]}
    assert read_json(moved / 'warm_start.json')['observation_size'] == [OLD_COLUMNS, OLD_COLUMNS + 103]
    (old_model, old_norm), (new_model, new_norm) = load_stage(stage), load_stage(moved)
    try:
        old_rms, new_rms = old_norm.obs_rms, new_norm.obs_rms
        assert new_rms.count == old_rms.count
        np.testing.assert_array_equal(new_rms.mean[:OLD_COLUMNS], old_rms.mean)
        np.testing.assert_array_equal(new_rms.var[:OLD_COLUMNS], old_rms.var)
        expected_mean, expected_variance = task.track_clips_normalization()
        np.testing.assert_array_equal(new_rms.mean[OLD_COLUMNS:], expected_mean)
        np.testing.assert_array_equal(new_rms.var[OLD_COLUMNS:], expected_variance)
        for name, value in new_model.policy.state_dict().items():
            if value.dim() == 2 and value.shape[1] == OLD_COLUMNS + 103:
                assert not value[:, OLD_COLUMNS:].any(), name
                torch.testing.assert_close(value[:, :OLD_COLUMNS], old_model.policy.state_dict()[name], rtol=0, atol=0)
        raw = new_norm.venv.envs[0].unwrapped
        for options in (dict(OPTIONS, track_clip_index=0, track_frame=5), dict(OPTIONS, track_clip_index=1),
                        dict(OPTIONS, start=int(np.flatnonzero(raw.bank['start_class'] == 0)[0]))):
            obs, _ = raw.reset(seed=2, options=options)
            for _ in range(3):
                new_obs = torch.as_tensor(new_norm.normalize_obs(obs)[None])
                old_obs = torch.as_tensor(old_norm.normalize_obs(obs[:OLD_COLUMNS])[None])
                with torch.no_grad():
                    np.testing.assert_allclose(new_model.policy.predict_values(new_obs).numpy(),
                                               old_model.policy.predict_values(old_obs).numpy(), rtol=0, atol=1e-5)
                    np.testing.assert_allclose(new_model.policy.get_distribution(new_obs).distribution.mean.numpy(),
                                               old_model.policy.get_distribution(old_obs).distribution.mean.numpy(),
                                               rtol=0, atol=1e-6)
                obs, *_ = raw.step(new_model.predict(new_norm.normalize_obs(obs), deterministic=True)[0])
    finally:
        old_norm.close(), new_norm.close()


def test_package_checks_every_clip_of_the_list(tmp_path, monkeypatch, clips):
    """scripts/robot/package.py: a track_clips stage goes into a robot package only if every clip is from a commercial
    dataset (one LAFAN1 clip refuses it), each clip's md5 checked."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'robot'))
    package = importlib.import_module('package')
    monkeypatch.setattr(package, 'TRACK_DIR', tmp_path)
    bones = {'file': 'b1', 'md5': _write_clip(tmp_path, 'b1', clips[0])}
    cmu = {'file': 'c1', 'md5': _write_clip(tmp_path, 'c1', clips[1], 'cmu')}
    lafan = {'file': 'l1', 'md5': _write_clip(tmp_path, 'l1', clips[1], 'lafan1')}
    for name, (entries, refused) in enumerate((([bones, cmu], False), ([bones, lafan], True),
                                                ([bones, {**cmu, 'md5': '0' * 32}], True))):
        stage = tmp_path / f'stage-{name}'
        stage.mkdir()
        (stage / 'metadata.json').write_text(json.dumps({'task_config': {'track_clips': entries}}), encoding='utf-8')
        folder = tmp_path / f'export-{name}'
        folder.mkdir()
        (folder / 'deploy.yaml').write_text(f'# humanoid_sim.export_g1: {stage} (100 steps)\n', encoding='utf-8')
        if refused:
            with pytest.raises(SystemExit):
                package.check_licence(folder)
        else:
            assert set(package.check_licence(folder)['datasets']) == {'bones_seed', 'cmu'}


def test_the_reference_of_each_clip_is_its_own(clips):
    """With a world on the front clip the reference is that clip's frames (clip_reference of its hold frames), held at
    its own hold frame after its last frame, the episode truncated at its own end; the back clip's end is later."""
    env = clips_env(clips)
    try:
        env.reset(seed=0, options=dict(OPTIONS, track_clip_index=1, track_frame=FRONT_FRAMES - 2, track_noise=False))
        frames = g1_getup_track.hold_frames(clips[1])
        position, _, linear, _ = env._clip_reference()
        np.testing.assert_allclose(position[0], frames['pos'][FRONT_FRAMES - 2], atol=1e-12)
        steps = 0
        while True:
            *_, terminated, truncated, info = env.step(
                (clips[1]['qpos'][min(env.clip_clock + 1, FRONT_FRAMES - 1), 7:] - env.home) / env.scale)
            steps += 1
            if terminated or truncated:
                break
        assert truncated and not terminated and env.clip_clock == FRONT_FRAMES - 1 + 150 and info['track_done']
        _, _, linear, _ = env._clip_reference()
        np.testing.assert_array_equal(linear, 0.)
        assert info['track_clip_index'] == 1 and env.clips[0]['end'] == FRAMES - 1 + 150
    finally:
        env.close()


def test_the_bank_rows_pose_is_the_resets_state(clips):
    """bank_row_poses gives the state a bank reset puts: the pelvis gravity, the joints and the anchor height after
    mj_forward of the row (the quaternion normalized), without touching the episode's data."""
    env = clips_env(clips)
    try:
        env.reset(seed=0, options=dict(OPTIONS, track_clip_index=0, track_frame=3, track_noise=False))
        before = env.data.qpos.copy()
        gravity, joints, anchor, kinds = env.bank_row_poses([1])
        assert np.array_equal(env.data.qpos, before)
        env.reset(seed=0, options=dict(OPTIONS, start=1))
        body = env.track_bodies[env.track_anchor]
        np.testing.assert_allclose(gravity[0], task.projected_gravity(env.data.xmat[env.pelvis].reshape(1, 3, 3))[0],
                                   atol=1e-12)
        np.testing.assert_allclose(joints[0], env.data.qpos[env.qadr], atol=1e-12)
        assert anchor[0] == pytest.approx(float(env.data.xpos[body, 2]), abs=1e-12) and kinds.tolist() == [1]
    finally:
        env.close()


def test_a_bank_start_at_its_nearest_clip_and_frame(clips):
    """Bank rows in the back clip's frame-20 pose and the front clip's frame-10 pose (its frames 14 and 15 are alike) start there: the clip of the row's
    class and that frame, the clock there, the reference's pelvis on the robot's; the back row as the old track_clip
    task with track_bank_nearest, bit for bit on the old 507 inputs."""
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0], bank['qpos'][1] = clips[0]['qpos'][20], clips[1]['qpos'][10]
    many = clips_env(clips, bank=bank)
    one = G1GetupEnv(bank=bank, track_clip=SETTING, track_bank=clips[0], track_bank_nearest=True)
    try:
        for row, (index, frame) in ((0, (0, 20)), (1, (1, 10))):
            _, info = many.reset(seed=0, options=dict(OPTIONS, start=row))
            assert (info['track_clip_index'], info['track_frame'], many.clip_clock) == (index, frame, frame)
            position, _, _, _ = many._clip_reference()
            np.testing.assert_allclose(position[0], many.data.xpos[many.track_bodies], atol=1e-6)
        _same_episode(many, one, dict(OPTIONS, start=0))
        assert many.clip_index == 0
    finally:
        many.close()
        one.close()
