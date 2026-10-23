import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from humanoid_sim.core import save_stage, load_stage, list_stages, StageSelection, RunLock, evaluate_stage

torch.set_num_threads(2)


def small_model():
    env = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1)
    model.learn(16)
    return model, env


def test_snapshot_roundtrip_and_ignores_incomplete(tmp_path):
    model, env = small_model()
    stage = save_stage(tmp_path, model, env)
    (tmp_path / '.pending-000000099').mkdir()
    assert list_stages(tmp_path) == [stage]
    loaded, normalizer = load_stage(stage)
    raw = np.arange(348, dtype=np.float32) / 100
    np.testing.assert_array_equal(env.normalize_obs(raw), normalizer.normalize_obs(raw))
    obs = env.normalize_obs(raw)
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    assert not normalizer.training and not normalizer.norm_reward
    assert loaded.num_timesteps == 16
    normalizer.close()
    env.close()


def test_selection_history_and_follow():
    selection = StageSelection()
    assert selection.select([0, 100, 200]) == 200
    selection.move(-1, [0, 100, 200])
    assert selection.select([0, 100, 200, 300]) == 100
    selection.move(-20, [0, 100, 200, 300])
    assert selection.select([0, 100, 200, 300]) == 0
    selection.follow = True
    assert selection.select([0, 100, 200, 300]) == 300


def test_left_starts_from_displayed_stage_when_new_stages_arrive():
    selection = StageSelection()
    assert selection.select([0]) == 0
    selection.move(-1, [0, 100, 200])
    assert selection.current == 0


def test_training_finalization_failure_publishes_failed_status(tmp_path, monkeypatch):
    import argparse
    import json
    from humanoid_sim.train import StageCallback, train
    def failed_snapshot(self):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(StageCallback, 'snapshot', failed_snapshot)
    with pytest.raises(OSError, match='simulated disk failure'):
        train(argparse.Namespace(run_dir=tmp_path, envs=1, seed=1, every=100000, steps=8))
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert 'simulated disk failure' in status['error']
    with RunLock(tmp_path):
        pass


def test_lock_prevents_two_writers_and_releases(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match='already'):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass


def test_evaluation_is_repeatable(tmp_path):
    model, env = small_model()
    stage = save_stage(tmp_path, model, env)
    a = evaluate_stage(stage, episodes=2, max_steps=10)
    b = evaluate_stage(stage, episodes=2, max_steps=10)
    assert a == b
    assert len(a['episodes']) == 2
    assert all(0 < x['seconds'] <= 0.15 + 1e-9 for x in a['episodes'])
    env.close()
