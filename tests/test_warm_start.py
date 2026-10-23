import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.core import load_stage, read_json, save_stage, write_json
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.warm_start import carry_over, widen_history

REAL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=True)


def _run(path, history):
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config={**REAL, 'history': history}))
    write_json(path / 'run_config.json', config)
    return config


def test_history_warm_start_acts_and_values_like_the_single_frame_stage(tmp_path):
    (tmp_path / 'old').mkdir(), (tmp_path / 'new').mkdir()
    config = _run(tmp_path / 'old', 1)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 98})
        model.learn(16)  # optimizer moments and normalization statistics that are not the defaults
        env.obs_rms.mean[:] = np.random.default_rng(1).normal(0, 1, 102)
        stage = save_stage(tmp_path / 'old', model, env)
    finally:
        env.close()
    _run(tmp_path / 'new', 5)
    wide = widen_history(stage, tmp_path / 'new')
    assert read_json(wide / 'metadata.json')['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-critic4'
    assert read_json(wide / 'metadata.json')['steps'] == model.num_timesteps
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(wide)
    try:
        assert after.observation_space.shape == (494,) and after.policy.actor_inputs == 490
        frames = torch.randn(6, 5, 98, dtype=torch.float64)
        extras = torch.randn(6, 4, dtype=torch.float64)
        single = torch.cat((frames[:, -1], extras), dim=1).numpy()
        history = torch.cat((task.history_observation(frames, 29), extras), dim=1).numpy()
        a = torch.as_tensor(old_norm.normalize_obs(single), dtype=torch.float32)
        b = torch.as_tensor(new_norm.normalize_obs(history), dtype=torch.float32)
        with torch.no_grad():  # older frames have zero weights: only the newest frame and the extras matter
            torch.testing.assert_close(after.policy.get_distribution(b).distribution.mean,
                                       before.policy.get_distribution(a).distribution.mean)
            torch.testing.assert_close(after.policy.predict_values(b), before.policy.predict_values(a))
        moments = after.policy.optimizer.state_dict()['state'][1]  # first actor layer
        oldest, newest = (torch.tensor(i) for i in task.history_index(29, 5)[::4])
        assert moments['exp_avg'].shape == (32, 494) and torch.count_nonzero(moments['exp_avg'][:, oldest]) == 0
        torch.testing.assert_close(moments['exp_avg_sq'][:, oldest], moments['exp_avg_sq'][:, newest])
        assert float(moments['exp_avg_sq'][:, newest].max()) > 0
    finally:
        old_norm.close()
        new_norm.close()
    with pytest.raises(ValueError):  # history to history is not a widening
        widen_history(wide, tmp_path / 'new')


def test_carry_over_keeps_the_weights_under_the_phase_stop_contract(tmp_path):
    (tmp_path / 'old').mkdir(), (tmp_path / 'new').mkdir()
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config={**REAL, 'history': 5}))
    write_json(tmp_path / 'old' / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        model.learn(16)
        stage = save_stage(tmp_path / 'old', model, env)
    finally:
        env.close()
    with pytest.raises(ValueError):  # same contract: an ordinary branch, nothing to carry over
        write_json(tmp_path / 'new' / 'run_config.json', config)
        carry_over(stage, tmp_path / 'new')
    write_json(tmp_path / 'new' / 'run_config.json',
               {**config, 'task_config': {**config['task_config'], 'phase_stop': True}})
    moved = carry_over(stage, tmp_path / 'new')
    assert read_json(moved / 'metadata.json')['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-critic4'
    (before, _), (after, _) = load_stage(stage), load_stage(moved)
    for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
        torch.testing.assert_close(a, b, msg=name)


def test_carry_over_moves_a_stage_to_the_lock_wrists_action_contract_and_refuses_other_changes(tmp_path):
    (tmp_path / 'old').mkdir()
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config={**REAL, 'history': 5, 'phase_stop': True}))
    write_json(tmp_path / 'old' / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        model.learn(16)
        stage = save_stage(tmp_path / 'old', model, env)
    finally:
        env.close()

    def run(name, **changes):
        (tmp_path / name).mkdir()
        write_json(tmp_path / name / 'run_config.json',
                   {**config, 'task_config': {**config['task_config'], **changes}})
        return tmp_path / name

    moved = carry_over(stage, run('locked', lock_wrists=True))
    metadata, warm = read_json(moved / 'metadata.json'), read_json(moved / 'warm_start.json')
    old = read_json(stage / 'metadata.json')
    assert metadata['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock'
    assert metadata['observation_schema'] == old['observation_schema']
    assert (warm['from_action_schema'], warm['to_action_schema']) == (old['action_schema'], metadata['action_schema'])
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        assert new_norm.venv.envs[0].unwrapped.lock_wrists
    finally:
        old_norm.close(), new_norm.close()
    # The same layout but a different action box (action_limit 10 -> 4), a different model (the simulation profile)
    # or a different observation layout (history 5 -> 1) is not a carry-over.
    for name, changes in (('box', dict(action_limit=4.)), ('limit', dict(action_limit=4., lock_wrists=True)),
                          ('physics', dict(real_robot=False, critic_extras=False, history=1, phase_stop=False)),
                          ('history', dict(history=1, lock_wrists=True))):
        with pytest.raises(ValueError):
            carry_over(stage, run(name, **changes))
        assert not any((tmp_path / name).glob('stage_*'))
