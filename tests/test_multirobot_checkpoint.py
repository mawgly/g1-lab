import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_sim.tasks.catalog import make_task
from humanoid_sim.core import save_stage, load_stage, write_json, read_json


@pytest.mark.parametrize('robot,task', [('g1_29dof', 'g1_velocity'), ('so101', 'arm_reach')])
def test_policy_roundtrip_for_different_robots(tmp_path, robot, task):
    write_json(tmp_path/'run_config.json', {'robot_id':robot, 'task_id':task})
    env = VecNormalize(DummyVecEnv([lambda: make_task(task, robot, {})]))
    try:
        model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu')
        stage = save_stage(tmp_path, model, env)
        metadata = read_json(stage/'metadata.json')
        assert metadata['robot_id'] == robot
        policy, norm = load_stage(stage)
        try:
            obs = env.reset()
            np.testing.assert_allclose(model.predict(obs, deterministic=True)[0],
                                       policy.predict(obs, deterministic=True)[0])
            assert norm.observation_space.shape == env.observation_space.shape
            from humanoid_sim.diagnostics import evaluate_new_seeds
            diagnostics = evaluate_new_seeds(stage, episodes=1, seed_start=85000)
            assert diagnostics['episodes'][0]['seed']==85000
            assert diagnostics['settings_consistency']['evaluation_normalization_frozen']
            if robot=='so101':
                assert diagnostics['fall_rate'] is None
            from humanoid_sim.reports import build_report, render_report
            write_json(tmp_path/'diagnostics.json',{'created_at':1,'checkpoints':{'last':diagnostics}})
            assert 'ROBOT LAB' in render_report(build_report(tmp_path))
        finally:
            norm.close()
    finally:
        env.close()
