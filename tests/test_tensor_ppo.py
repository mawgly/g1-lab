import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

from humanoid_sim.tensor_ppo import RunningMoments, TensorNormalize, generalized_advantage


def test_running_moments_match_sb3_across_batches():
    source = np.random.default_rng(42)
    expected = RunningMeanStd(shape=(3,))
    actual = RunningMoments((3,), 'cpu')
    for count in (1, 7, 256):
        values = source.normal(size=(count, 3))
        expected.update(values)
        actual.update(torch.tensor(values))
    np.testing.assert_allclose(actual.mean.numpy(), expected.mean, atol=1e-12)
    np.testing.assert_allclose(actual.var.numpy(), expected.var, atol=1e-12)
    restored = RunningMeanStd(shape=(3,))
    actual.export_rms(restored)
    assert restored.count == expected.count


def test_reward_normalization_resets_returns_on_both_done_types():
    norm = TensorNormalize((3,), 2, 'cpu', gamma=.9)
    norm.reward(torch.tensor([1., 2.]), torch.tensor([False, True]))
    torch.testing.assert_close(norm.returns, torch.tensor([1., 0.], dtype=torch.float64))
    norm.reward(torch.tensor([3., 4.]), torch.tensor([True, False]))
    torch.testing.assert_close(norm.returns, torch.tensor([0., 4.], dtype=torch.float64))
    expected = RunningMeanStd(shape=())
    expected.update(np.array([1., 2.]))
    expected.update(np.array([3.9, 4.]))
    np.testing.assert_allclose(norm.ret.var.numpy(), expected.var)


def test_gae_does_not_leak_across_reset_and_bootstraps_timeout_only_once():
    # World0 falls on step0; world1 times out with terminal value10.
    rewards = torch.tensor([[1., 2. + .9 * 10], [3., 4.]])
    values = torch.tensor([[.5, 1.], [2., 3.]])
    dones = torch.tensor([[True, True], [False, False]])
    advantage, returns = generalized_advantage(rewards, values, dones, torch.tensor([5., 6.]), .9, .95)
    torch.testing.assert_close(advantage, torch.tensor([[.5, 10.], [5.5, 6.4]]))
    torch.testing.assert_close(returns, torch.tensor([[1., 11.], [7.5, 9.4]]))
