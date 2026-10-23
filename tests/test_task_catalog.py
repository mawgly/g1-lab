import numpy as np
import pytest
from humanoid_sim.tasks.catalog import make_task, catalog


@pytest.mark.parametrize('robot,task,actions', [('humanoid', 'humanoid_v5', 17), ('so101', 'arm_reach', 5), ('g1_29dof', 'g1_velocity', 29), ('g1_12dof', 'g1_velocity', 12)])
def test_builtin_tasks_reset_step_and_close(robot, task, actions):
    env = make_task(task, robot, {})
    try:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (actions,)
        obs, reward, terminated, truncated, info = env.step(np.zeros(actions, dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    finally:
        env.close()


def test_incompatible_robot_task_is_rejected():
    with pytest.raises(ValueError):
        make_task('humanoid_v5', 'so101', {})
    assert {r['robot_id'] for r in catalog()} >= {'humanoid', 'so101'}
