import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_sim.tasks import catalog
from humanoid_sim.tasks.registry import TaskRegistry, TaskSpec
from humanoid_sim.robots.registry import RobotRegistry, RobotSpec
from humanoid_sim.core import save_stage, load_stage, evaluate_stage, write_json
from humanoid_sim.checkpoint_schema import LEGACY


def test_new_one_motor_task_trains_saves_and_evaluates_without_locomotion_fields(tmp_path, monkeypatch):
    robots, tasks = RobotRegistry(), TaskRegistry()
    robots.register(RobotSpec('pendulum', 'Pendulum', 'fixed', ('hinge',), ('motor',), frozenset({'balance'})))
    identity = dict(LEGACY, robot_id='pendulum', task_id='balance', model_revision='pendulum-v1',
                    observation_schema='pendulum-obs-v1', action_schema='pendulum-torque-v1', control_dt=.05)
    tasks.register(TaskSpec('balance', frozenset({'balance'}), frozenset({'cpu'}),
                            lambda robot, config: gym.make('Pendulum-v1', **config),
                            identity_factory=lambda robot: identity))
    monkeypatch.setattr(catalog, 'registries', lambda: (robots, tasks))
    write_json(tmp_path/'run_config.json', {'robot_id':'pendulum', 'task_id':'balance'})
    env = VecNormalize(DummyVecEnv([lambda: catalog.make_task('balance', 'pendulum', {})]))
    try:
        model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, seed=4, device='cpu')
        model.learn(16)
        stage = save_stage(tmp_path, model, env)
        restored, norm = load_stage(stage)
        norm.close()
        assert restored.action_space.shape == (1,)
        result = evaluate_stage(stage, episodes=2, max_steps=4)
        assert result['robot_id'] == 'pendulum'
        assert result['mean_seconds'] == .2
        assert result['mean_forward_m'] is None
        assert all(r['fell'] is None for r in result['episodes'])
    finally:
        env.close()
