import numpy as np
import pytest
from stable_baselines3 import PPO
from arm_sim.env import ArmReachEnv
from arm_sim.cli import evaluate_policy, load_policy, positive_int


def test_report_identifies_exact_checkpoint(tmp_path):
    import argparse
    import hashlib
    import json
    from arm_sim.cli import evaluate
    env = ArmReachEnv()
    path = tmp_path / 'policy.zip'
    PPO('MlpPolicy', env, n_steps=8, batch_size=8, device='cpu').save(path)
    output = tmp_path / 'report.json'
    evaluate(argparse.Namespace(model=path, episodes=1, seed=100, output=output))
    report = json.loads(output.read_text())
    assert report['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['training_timesteps'] == 0


def test_save_load_and_reproducible_evaluation(tmp_path):
    env = ArmReachEnv(max_steps=8)
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=9)
    model.learn(16)
    model.save(tmp_path / 'policy')
    loaded = load_policy(tmp_path / 'policy.zip')
    obs, _ = env.reset(seed=7)
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    a = evaluate_policy(loaded, episodes=2, seed=100, max_steps=8)
    b = evaluate_policy(loaded, episodes=2, seed=100, max_steps=8)
    assert a == b
    assert len(a['episodes']) == 2
    assert 0 <= a['success_rate'] <= 1
    assert a['mean_final_distance_m'] >= 0


def test_missing_checkpoint_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match='train'):
        load_policy(tmp_path / 'missing.zip')


def test_random_baseline_is_reproducible():
    a = evaluate_policy(None, episodes=2, seed=100, max_steps=5)
    assert a == evaluate_policy(None, episodes=2, seed=100, max_steps=5)


def test_positive_cli_counts():
    for value in ('0', '-5'):
        with pytest.raises(ValueError):
            positive_int(value)
    assert positive_int('5') == 5
