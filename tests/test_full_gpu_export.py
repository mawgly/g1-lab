"""Exercise tensor PPO archives on CPU; CUDA physics has separate tests."""
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.core import list_stages, load_stage, read_json, save_stage
from humanoid_sim.lab import validate_config
from humanoid_sim.tensor_ppo import TensorNormalize, TensorPPO


class FakeTensorEnv:
    """Deterministic batched task with both falls and time-limit resets."""

    def __init__(self, n_envs, seed=0):
        self.device = torch.device('cpu')
        self.n_envs = n_envs
        self.reference = gym.make('Humanoid-v5')
        self.observation_space = self.reference.observation_space
        self.action_space = self.reference.action_space
        self.lengths = torch.zeros(n_envs, dtype=torch.long)
        self.last_episode_lengths = torch.zeros(n_envs)
        self.last_episode_returns = torch.zeros(n_envs)
        self.returns = torch.zeros(n_envs)
        self.initial = torch.linspace(-.2, .2, self.observation_space.shape[0]).repeat(n_envs, 1)
        self.state = self.initial.clone()
        self.checked = False

    def reset(self):
        self.state.copy_(self.initial)
        return self.state.clone()

    def step(self, actions):
        self.state[:, :17] += .03 * actions
        self.lengths += 1
        self.state[:, 17] = self.lengths.float() * .1
        reward = 1.0 - actions.square().mean(1)
        self.returns += reward
        terminated = (self.lengths >= 3) & (torch.arange(self.n_envs) % 2 == 0)
        truncated = self.lengths >= 5
        done = terminated | truncated
        terminal = self.state.clone()
        self.last_episode_lengths.copy_(self.lengths)
        self.last_episode_returns.copy_(self.returns)
        self.state[done] = self.initial[done]
        self.lengths[done] = 0
        self.returns[done] = 0
        return self.state.clone(), reward, terminated, truncated, terminal

    def check_health(self):
        assert torch.isfinite(self.state).all()
        self.checked = True

    def close(self):
        self.reference.close()


def test_bounds_loss_pulls_saturated_means_back_and_std_is_capped():
    torch.set_num_threads(2)
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    env = FakeTensorEnv(2)
    try:
        means = {}
        for coef in (0., .1):
            config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=4, steps=1024, target_kl=None,
                                          learning_rate=.001, bounds_coef=coef, max_std=1. if coef else None))
            model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8, batch_size=8,
                        policy_kwargs={'net_arch': [32, 32]})
            with torch.no_grad():  # a policy that drifted far outside the action box, as seen on G1
                model.policy.action_net.bias.fill_(20.)
                model.policy.log_std.fill_(10.)
            learner = TensorPPO(model, env, config, TensorNormalize(env.observation_space.shape, 2, 'cpu'))
            learner.collect()
            learner.update()
            obs = learner.obs_buffer.flatten(0, 1)
            means[coef] = float(model.policy.get_distribution(obs).distribution.mean.mean())
            if coef:
                assert float(model.policy.log_std.max()) <= np.log(1.) + 1e-6
            else:
                assert float(model.policy.log_std.max()) > 9.  # off by default: older runs train unchanged
        assert means[.1] < means[0.] - 1e-3  # Adam moves a few lr per step; the direction is what matters
    finally:
        env.close()


def test_range_bounds_pull_means_back_inside_the_joint_range_equivalent_box():
    torch.set_num_threads(2)
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    env = FakeTensorEnv(2)
    try:
        means = {}
        for bounds in (None, (-np.full(17, .1), np.full(17, .1))):
            config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=4, steps=1024, target_kl=None,
                                          learning_rate=.001, bounds_coef=.1, range_bounds=bounds is not None))
            model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8, batch_size=8,
                        policy_kwargs={'net_arch': [32, 32]})
            with torch.no_grad():  # inside the ±0.4 action box (no loss there), outside a ±0.1 joint-range box
                model.policy.action_net.bias.fill_(.3)
            learner = TensorPPO(model, env, config, TensorNormalize(env.observation_space.shape, 2, 'cpu'),
                                bounds=bounds)
            learner.collect()
            learner.update()
            with torch.no_grad():
                means[bounds is None] = float(model.policy.get_distribution(learner.obs_buffer.flatten(0, 1))
                                              .distribution.mean.mean())
        assert means[False] < means[True] - 1e-3
    finally:
        env.close()


class SwapMirror:
    """Stands in for symmetry.Mirror: reverses observations, swaps neighbouring actions (both involutions)."""

    def __init__(self, observations, actions):
        self.obs_index = torch.arange(observations - 1, -1, -1)
        self.action_index = torch.arange(actions)
        self.action_index[:actions // 2 * 2] = self.action_index[:actions // 2 * 2].view(-1, 2).flip(1).flatten()

    def to(self, device):
        return self

    def observation(self, values):
        return values[..., self.obs_index]

    def action(self, values):
        return values[..., self.action_index]


def test_mirror_loss_makes_the_policy_answer_mirrored_states_with_mirrored_actions():
    torch.set_num_threads(2)
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    env = FakeTensorEnv(2)
    try:
        mirror = SwapMirror(env.observation_space.shape[0], env.action_space.shape[0])
        asymmetry = {}
        for coef in (0., 1.):
            config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=10, steps=1024, target_kl=None,
                                          learning_rate=.001, mirror_coef=coef))
            model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8, batch_size=8,
                        policy_kwargs={'net_arch': [32, 32]})
            learner = TensorPPO(model, env, config, TensorNormalize(env.observation_space.shape, 2, 'cpu'),
                                mirror=mirror)

            def error():
                obs, moments = learner.obs_buffer.flatten(0, 1), learner.normalizer.obs
                scale = torch.sqrt(moments.var + 1e-8)
                mirrored = ((mirror.observation(obs.double() * scale + moments.mean) - moments.mean) / scale)
                with torch.no_grad():
                    mean = model.policy.get_distribution(obs).distribution.mean
                    other = model.policy.get_distribution(mirrored.clamp(-10, 10).float()).distribution.mean
                return float((other - mirror.action(mean)).square().mean())

            learner.collect()
            before = error()
            stats = learner.update()
            asymmetry[coef] = error() / before
            assert (stats['train/mirror_loss'] > 0) == bool(coef)  # off by default: older runs train unchanged
        assert asymmetry[1.] < .8 * asymmetry[0.]
    finally:
        env.close()


class MetricTensorEnv(FakeTensorEnv):
    """A task that reports its own means once per rollout, as TensorG1Env.rollout_metrics does."""

    def __init__(self, n_envs):
        super().__init__(n_envs)
        self.reads = 0

    def rollout_metrics(self):
        self.reads += 1
        return {'rollout/yaw_drift_abs': .042}


def test_collect_logs_the_task_rollout_metrics():
    torch.set_num_threads(2)
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=1, steps=1024, target_kl=None,
                                  learning_rate=.0003))
    try:
        for env in (FakeTensorEnv(2), MetricTensorEnv(2)):
            try:
                model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8, batch_size=8,
                            policy_kwargs={'net_arch': [32, 32]})
                learner = TensorPPO(model, env, config, TensorNormalize(env.observation_space.shape, 2, 'cpu'))
                metrics = learner.collect()
                assert {'rollout/ep_rew_mean', 'rollout/ep_len_mean'} <= set(metrics)
                if isinstance(env, MetricTensorEnv):  # read once per rollout, after it
                    assert metrics['rollout/yaw_drift_abs'] == .042 and env.reads == 1
                else:
                    assert 'rollout/yaw_drift_abs' not in metrics
            finally:
                env.close()
    finally:
        export.close()


@pytest.fixture
def trained_tensor_learner():
    torch.set_num_threads(2)
    config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=2,
                                 steps=1024, target_kl=None, learning_rate=.0003))
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8,
                batch_size=8, policy_kwargs={'net_arch': [32, 32]})
    env = FakeTensorEnv(2)
    normalizer = TensorNormalize(env.observation_space.shape, 2, 'cpu')
    learner = TensorPPO(model, env, config, normalizer)
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    metrics = learner.collect()
    metrics.update(learner.update())
    assert env.checked and model.num_timesteps == 16
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[name], value) for name, value in model.policy.state_dict().items())
    normalizer.export_normalizer(export)
    try:
        yield learner, export, config
    finally:
        env.close()
        export.close()


def test_tensor_update_exports_identical_cpu_predictions_and_statistics(tmp_path, trained_tensor_learner):
    learner, export, _ = trained_tensor_learner
    stage = save_stage(tmp_path, learner.model, export)
    loaded, frozen = load_stage(stage)
    try:
        raw = learner.env.reset().numpy()
        tensor_obs = learner.normalizer.observation(torch.from_numpy(raw), update=False).numpy()
        np.testing.assert_allclose(frozen.normalize_obs(raw), tensor_obs, atol=1e-6)
        np.testing.assert_allclose(loaded.predict(frozen.normalize_obs(raw), deterministic=True)[0],
                                   learner.model.predict(tensor_obs, deterministic=True)[0], atol=1e-7)
        assert loaded.num_timesteps == 16
        assert not frozen.training and not frozen.norm_reward
        np.testing.assert_array_equal(frozen.ret_rms.var, export.ret_rms.var)
        assert frozen.ret_rms.count == export.ret_rms.count
        expected = learner.policy.optimizer.state_dict()
        actual = loaded.policy.optimizer.state_dict()
        assert actual['param_groups'] == expected['param_groups']
        assert actual['state']
        for param, state in expected['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(actual['state'][param][key], value)
    finally:
        frozen.close()


def test_make_learner_resume_applies_new_optimizer_rate(tmp_path, monkeypatch, trained_tensor_learner):
    """Run the real resume orchestration, substituting CPU for CUDA devices only."""
    from humanoid_sim import full_gpu_train
    learner, export, config = trained_tensor_learner
    stage = save_stage(tmp_path, learner.model, export)
    fake_module = ModuleType('humanoid_sim.tensor_env')
    fake_module.TensorHumanoidEnv = FakeTensorEnv

    class CPUArchiveLoader:
        @staticmethod
        def load(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return PPO.load(*args, **kwargs)

    changed = {**config, 'learning_rate': .00005, 'gamma': .95}
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'humanoid_sim.tensor_env', fake_module)
        patch.setattr(torch.cuda, 'is_available', lambda: True)
        patch.setattr(full_gpu_train, 'PPO', CPUArchiveLoader)
        resumed, resumed_export = full_gpu_train.make_learner(changed, stage)
    try:
        assert all(group['lr'] == .00005 for group in resumed.policy.optimizer.param_groups)
        assert resumed.policy.optimizer.state
        assert resumed.normalizer.gamma == .95
        assert resumed.normalizer.ret.count.item() == pytest.approx(1e-4)
        assert resumed.normalizer.obs.count.item() > export.obs_rms.count
        resumed.collect()
        resumed.update()
        assert resumed.model.num_timesteps == 32
        resumed.normalizer.export_normalizer(resumed_export)
        assert resumed_export.gamma == .95
        reexport = save_stage(tmp_path, resumed.model, resumed_export)
        loaded, frozen = load_stage(reexport)
        try:
            assert loaded.gamma == .95 and frozen.gamma == .95
            assert loaded.learning_rate == .00005
        finally:
            frozen.close()
    finally:
        resumed.env.close()
        resumed_export.close()


@pytest.mark.parametrize('stop_before_start', [False, True])
def test_full_gpu_training_lifecycle_exports_and_respects_parent_stop(
        tmp_path, monkeypatch, trained_tensor_learner, stop_before_start):
    from humanoid_sim import full_gpu_train

    learner, export, original_config = trained_tensor_learner
    initial_steps = learner.model.num_timesteps
    # Two actual tensor rollouts keep this orchestration check fast on CPU.
    config = {**original_config, 'steps': 32, 'every': 1024}
    parent_stop = tmp_path / 'queue.stop'
    if stop_before_start:
        parent_stop.touch()
    run = tmp_path / 'run'
    monkeypatch.setattr(full_gpu_train, 'make_learner', lambda *_: (learner, export))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU test substitute')
    # Keep real snapshot/save/export behavior; replace only the expensive physics evaluation.
    import humanoid_sim.train as cpu_train
    monkeypatch.setattr(cpu_train, 'evaluate_stage', lambda *_: {
        'mean_return': 0.0, 'mean_seconds': 0.0, 'mean_forward_m': 0.0, 'episodes': []})
    closed = []
    original_close = learner.env.close
    monkeypatch.setattr(learner.env, 'close', lambda: (closed.append(True), original_close()))

    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=run, parent_stop=parent_stop), config)

    status = read_json(run / 'status.json')
    assert status['state'] == ('stopped' if stop_before_start else 'completed')
    assert closed
    expected_steps = initial_steps if stop_before_start else initial_steps + 32
    assert status['steps'] == expected_steps
    stages = list_stages(run)
    assert [int(stage.name[6:]) for stage in stages] == (
        [initial_steps] if stop_before_start else [initial_steps, expected_steps])
    loaded, frozen = load_stage(stages[-1])
    try:
        assert loaded.num_timesteps == expected_steps
        np.testing.assert_allclose(frozen.obs_rms.mean, learner.normalizer.obs.mean.numpy())
        np.testing.assert_allclose(frozen.ret_rms.var, learner.normalizer.ret.var.numpy())
        assert frozen.obs_rms.count == learner.normalizer.obs.count.item()
    finally:
        frozen.close()
