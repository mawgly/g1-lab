"""g1_getup track_clip's task_config track_sampling (iteration 77; SuperMimic, arXiv 2610.28480, appendix D.1): the RSI
starts favour the hard bins of the clip. CPU: the validator; off (missing or null), and on in the CPU task, which does
not act on it, the starts and episodes are bit for bit the old ones; g1_tensor_math's bins, mixture, G_b update and
TrackBinSampler (torch on the CPU device) against hand calculations; its training_state there and back. Not part of
the contract. The CUDA task: tests/test_g1_gpu.py -k track_sampling."""
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_TRACK_WEIGHTS
from test_g1_getup_track import SETTING, track_env
from test_g1_getup_track_starts import OLD_STARTS_DIGEST, starts_digest

WORKING = {'bin': 1.0, 'uniform': .5, 'hard': .4, 'history': .1, 'temperature': 25}
# episodes_digest of the code before this setting (commit 620d858), with test_g1_getup_track's synthetic clip and
# drop_bank().
OLD_EPISODES_DIGEST = '80b36445c74a2f6ec8066155c0934b54db0070317a1c724b7b232ce37a38cd84'
KERNEL = sum(G1_GETUP_TRACK_WEIGHTS[key] for key in task.TRACK_SIGMA)


def episodes_digest(seed=21, resets=12, steps=60, **settings):
    """sha256 of 12 training resets mostly from RSI (track_starts 0.2 / 0.7 / 0.1) and up to 60 steps each under random
    actions: observations, qpos, qvel, the start's kind, frame and row, rewards and end flags."""
    env = track_env(track_starts={'first': .2, 'rsi': .7, 'bank': .1}, **settings)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    try:
        env.reset(seed=seed)
        for _ in range(resets):
            obs, info = env.reset()
            digest.update(obs.tobytes() + env.data.qpos.tobytes() + env.data.qvel.tobytes()
                          + f"{info['track_start']} {info['track_frame']} {info['start']}".encode())
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(rng.uniform(-1., 1., 29))
                digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated]))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return digest.hexdigest()


def sampler(frames=250, n=8, setting=WORKING, seed=0):
    return task.TrackBinSampler(setting, frames, .02, KERNEL, n, 'cpu', seed)


def test_the_setting_is_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_sampling'] is None  # old stages: off
    tracking = {'track_clip': SETTING}
    on = validate_config({**base, 'task_config': {**tracking, 'track_sampling': WORKING}})['task_config']
    assert on['track_sampling'] == {key: float(value) for key, value in WORKING.items()}
    assert all(type(value) is float for value in on['track_sampling'].values())
    assert validate_config({**base, 'task_config': {**tracking, 'track_sampling': None}})['task_config'][
        'track_sampling'] is None
    for value in ({**WORKING, 'uniform': .6}, {**WORKING, 'bin': 0.}, {**WORKING, 'bin': -1.}, {**WORKING, 'bin': 11.},
                  {**WORKING, 'hard': -.1, 'uniform': .6}, {**WORKING, 'temperature': -1.},
                  {**WORKING, 'temperature': float('inf')}, {**WORKING, 'bin': float('nan')},
                  {**WORKING, 'uniform': True}, {**WORKING, 'x': 0.},
                  {key: value for key, value in WORKING.items() if key != 'history'}, [1., .5, .4, .1, 25.], 'on'):
        with pytest.raises(ValueError, match='track_sampling'):
            validate_config({**base, 'task_config': {**tracking, 'track_sampling': value}})
    with pytest.raises(ValueError, match='needs track_clip'):
        validate_config({**base, 'task_config': {'track_sampling': WORKING}})
    with pytest.raises(ValueError, match='RSI'):  # nothing to sample without RSI starts
        validate_config({**base, 'task_config': {**tracking, 'track_sampling': WORKING,
                                                 'track_starts': {'first': .9, 'rsi': 0., 'bank': .1}}})
    assert task_identity('g1_29dof', 'g1_getup', {**tracking, 'track_sampling': WORKING}) == task_identity(
        'g1_29dof', 'g1_getup', tracking)
    fields = {p['key']: p for p in G1_GETUP_PARAMETERS}
    assert fields['track_sampling']['default'] is None and fields['track_sampling']['type'] == 'json'
    assert 'трудных' in fields['track_sampling']['label']


def test_off_and_on_the_cpu_task_the_starts_and_episodes_are_bit_for_bit_the_old_ones():
    """Missing or null, the task is the old one; the CPU task keeps its uniform RSI with the setting too (the full-GPU
    task acts on it)."""
    assert starts_digest(track_sampling=None) == OLD_STARTS_DIGEST
    assert starts_digest(track_sampling=WORKING) == OLD_STARTS_DIGEST
    assert episodes_digest() == OLD_EPISODES_DIGEST
    assert episodes_digest(track_sampling=None) == OLD_EPISODES_DIGEST
    assert episodes_digest(track_sampling=WORKING) == OLD_EPISODES_DIGEST


def test_the_bins_cover_the_clip_frames():
    starts, lengths = task.track_bins(230, 50)
    assert starts.tolist() == [0, 50, 100, 150, 200] and lengths.tolist() == [50, 50, 50, 50, 30]
    starts, lengths = task.track_bins(250, 50)
    assert lengths.tolist() == [50] * 5 and lengths.sum() == 250
    starts, lengths = task.track_bins(40, 50)  # the synthetic clip: one bin of every frame
    assert starts.tolist() == [0] and lengths.tolist() == [40]
    assert sampler(frames=176).bins == 4  # 3.5 s at bin 1.0: the last bin 26 frames


def test_the_mixture_on_synthetic_returns():
    """Bins with G_b/G_max (0.8, 0.8, 0.2, 0.8, 0.8): over 400 000 draws the parts come 50 / 40 / 10 %; the uniform
    part spreads evenly, the hard part goes to the hard bin (softmax at temperature 25: e^15 against the others), the
    history part follows its weights; the frame is uniform within its bin; the history weights take one EMA step toward
    the softmax row per draw; numpy and torch give the same draws."""
    s = sampler()
    s.g.copy_(torch.tensor([.8, .8, .2, .8, .8], dtype=torch.float64))
    history = torch.tensor([.1, .1, .1, .1, .6], dtype=torch.float64)
    s.history.copy_(history)
    rows = s.weights()
    soft = np.exp(-25 * np.array([.8, .8, .2, .8, .8]))
    np.testing.assert_allclose(rows[1].numpy(), soft / soft.sum(), rtol=1e-12)
    assert float(rows[1, 2]) == pytest.approx(1 / (1 + 4 * math.exp(-15)), rel=1e-12)
    np.testing.assert_allclose(rows[0].numpy(), .2)
    np.testing.assert_allclose(rows[2].numpy(), history.numpy() / history.sum().item())
    n = 400_000
    part, bin_, frame = s.draw(n)
    np.testing.assert_allclose(s.history.numpy(), history.numpy() + task.TRACK_SAMPLING_HISTORY_RATE * (
        rows[1].numpy() - history.numpy()), rtol=1e-12)
    shares = part.bincount(minlength=3).numpy() / n
    np.testing.assert_allclose(shares, (.5, .4, .1), atol=.004)
    for index, expected in enumerate(rows.numpy()):
        mine = bin_[part == index].bincount(minlength=5).numpy()
        np.testing.assert_allclose(mine / mine.sum(), expected, atol=.006)
    overall = bin_.bincount(minlength=5).numpy() / n
    np.testing.assert_allclose(overall, .5 * .2 + .4 * rows[1].numpy() + .1 * rows[2].numpy(), atol=.004)
    assert overall[2] > .5  # 0.1 + 0.4 + 0.01: the hard bin's share, from 0.2 uniform
    assert bool((frame // 50 == bin_).all()) and int(frame.min()) == 0 and int(frame.max()) == 249
    inside = (frame[bin_ == 2] - 100).bincount(minlength=50).numpy()
    assert len(inside) == 50 and inside.min() > .8 * inside.mean()
    rng = np.random.default_rng(0)
    draws = [rng.uniform(size=1000) for _ in range(3)]
    starts, lengths = task.track_bins(230, 50)
    expected = task.track_bin_draw(rows.numpy(), (.5, .4, .1), *draws, starts, lengths)
    mine = task.track_bin_draw(rows, (.5, .4, .1), *(torch.from_numpy(d) for d in draws), torch.from_numpy(starts),
                               torch.from_numpy(lengths))
    for a, b in zip(expected, mine):
        np.testing.assert_array_equal(a, b.numpy())
    assert expected[2].max() <= 229 and (expected[2][expected[1] == 4] >= 200).all()
    one_hot = torch.zeros((3, 5), dtype=torch.float64)
    one_hot[:, 2] = 1.  # every part on bin 2: its edges by the draws' edges
    zero = task.track_bin_draw(one_hot, (0., 1., 0.), torch.zeros(3, dtype=torch.float64),
                               torch.tensor([0., .5, 1 - 1e-12], dtype=torch.float64),
                               torch.tensor([0., .5, 1 - 1e-12], dtype=torch.float64), torch.from_numpy(starts),
                               torch.from_numpy(lengths))
    assert zero[0].tolist() == [1, 1, 1] and zero[1].tolist() == [2, 2, 2] and zero[2].tolist() == [100, 125, 149]


def test_the_temperature_and_equal_returns():
    """At temperature 0, or with every bin alike, the hard part is uniform too; a high temperature puts it on the
    smallest G."""
    g = torch.tensor([.3, .5, .9], dtype=torch.float64)
    history = torch.ones(3, dtype=torch.float64)
    np.testing.assert_allclose(task.track_bin_weights(g, history, 0.)[1].numpy(), 1 / 3)
    np.testing.assert_allclose(task.track_bin_weights(torch.full((3,), .7, dtype=torch.float64), history, 25.)[1].numpy(),
                               1 / 3)
    assert task.track_bin_weights(g, history, 1000.)[1].argmax().item() == 0
    np.testing.assert_allclose(task.track_bin_weights(g.numpy(), history.numpy(), 25.),
                               task.track_bin_weights(g, history, 25.).numpy(), rtol=1e-12)


def test_the_running_estimate():
    """track_bin_update: the plain mean of a bin's first 1/rate episodes, then an EMA of rate per episode; a bin without
    new episodes stays; numpy and torch alike."""
    g, count = np.zeros(3), np.zeros(3)
    g, count = task.track_bin_update(g, count, np.array([1.2, 0., 0.]), np.array([2., 0., 0.]))
    np.testing.assert_allclose(g, [.6, 0., 0.])
    np.testing.assert_allclose(count, [2., 0., 0.])
    values = np.random.default_rng(2).uniform(size=98)
    for value in values:
        g, count = task.track_bin_update(g, count, np.array([value, 0., 0.]), np.array([1., 0., 0.]))
    assert count[0] == 100 and g[0] == pytest.approx((1.2 + values.sum()) / 100, abs=1e-12)
    before = g[0]
    g, count = task.track_bin_update(g, count, np.array([3., 0., 0.]), np.array([1., 0., 0.]))
    assert g[0] == pytest.approx(before + task.TRACK_SAMPLING_RATE * (3. - before), abs=1e-12)
    before = g[0]
    g, count = task.track_bin_update(g, count, np.array([2.5, 0., 0.]), np.array([5., 0., 0.]))  # five of mean 0.5
    assert g[0] == pytest.approx(before + 5 * task.TRACK_SAMPLING_RATE * (.5 - before), abs=1e-12)
    assert g[1] == 0 and g[2] == 0 and count.tolist() == [106., 0., 0.]
    torch_g, torch_count = task.track_bin_update(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                                                 torch.tensor([1.2, 0., 0.], dtype=torch.float64),
                                                 torch.tensor([2., 0., 0.], dtype=torch.float64))
    np.testing.assert_allclose(torch_g.numpy(), [.6, 0., 0.])
    np.testing.assert_allclose(torch_count.numpy(), [2, 0, 0])
    assert task.track_bin_return_max(KERNEL, 1) == pytest.approx(KERNEL)
    assert task.track_bin_return_max(KERNEL, 150) == pytest.approx(KERNEL * (1 - .99 ** 150) / .01)


def test_the_sampler_counts_its_episodes():
    """Four worlds: two RSI starts in bins 0 and 3 (the second from the hard part), a first-frame start (bin −1) and a
    discarded one. Rewards: world 0 earns every kernel term in full for its whole limit (normalized 1), world 1 half of
    that for 5 of its 20 steps, then terminates; only they count, each in its bin; the hard share is 1 of the 3 RSI
    draws (the discarded episode's start was drawn too)."""
    s = sampler(n=4)
    everyone = torch.ones(4, dtype=torch.bool)
    s.start(everyone, torch.tensor([0, 3, -1, 2]), torch.tensor([0, 1, 1, 0]), torch.tensor([10, 20, 10, 10]))
    s.discard(torch.tensor([False, False, False, True]))
    for step in range(10):
        reward = torch.tensor([KERNEL, KERNEL / 2 if step < 5 else 0., 1., 1.])
        s.add(reward)
        done = torch.tensor([step == 9, step == 4, step == 9, step == 9])
        s.finish(done)
        s.discard(done)
    expected = (KERNEL / 2) * (1 - .99 ** 5) / .01 / task.track_bin_return_max(KERNEL, 20)
    np.testing.assert_allclose(s.g.numpy(), [1., 0., 0., expected, 0.], rtol=1e-9)
    np.testing.assert_allclose(s.count.numpy(), [1., 0., 0., 1., 0.])
    assert bool((s.world_bin == -1).all())
    metrics = s.metrics()
    assert metrics['rollout/track_bin_hard_share'] == pytest.approx(1 / 3)  # a discarded start was still drawn
    assert metrics['rollout/track_bin_g_min'] == pytest.approx(expected)
    assert metrics['rollout/track_bin_g_mean'] == pytest.approx((1 + expected) / 2)
    assert metrics['rollout/track_bin_hardest'] == 3.
    again = s.metrics()
    assert math.isnan(again['rollout/track_bin_hard_share'])  # the counts start again; the bins stay
    assert again['rollout/track_bin_g_min'] == metrics['rollout/track_bin_g_min']
    fresh = sampler().metrics()
    assert math.isnan(fresh['rollout/track_bin_g_min']) and fresh['rollout/track_bin_hardest'] == -1.


def test_the_training_state_there_and_back():
    """state() goes through JSON into another sampler of the same bins exactly; another bin count, missing keys or bad
    numbers start afresh."""
    s = sampler()
    s.g.copy_(torch.tensor([.1, .2, .3, .4, .5], dtype=torch.float64))
    s.count.copy_(torch.tensor([1., 2., 300., 4., 0.], dtype=torch.float64))
    s.draw(16)
    state = json.loads(json.dumps(s.state()))
    assert set(state) == {'track_bin_g', 'track_bin_count', 'track_bin_history'}
    other = sampler(seed=5)
    assert other.restore({'assist_force': [0.], **state})
    for name in ('g', 'count', 'history'):
        assert torch.equal(getattr(other, name), getattr(s, name)), name
    for state_ in ({'assist_force': [0.]}, {**state, 'track_bin_g': state['track_bin_g'][:4]},
                   {**state, 'track_bin_count': [-1.] * 5}, {**state, 'track_bin_history': [0.] * 5},
                   {**state, 'track_bin_g': [float('nan')] * 5}):
        fresh = sampler()
        assert not fresh.restore(state_)
        assert bool((fresh.g == 0).all()) and bool((fresh.count == 0).all()) and bool((fresh.history == .2).all())
    longer = sampler(frames=400)  # 8 bins: the five saved ones do not fit
    assert not longer.restore(state) and longer.bins == 8
