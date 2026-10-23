import numpy as np
import pytest
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv


def test_g1_reset_seed_action_limits_and_timeout():
    env = G1VelocityEnv(max_episode_steps=3)
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        assert env.model.nu == 29
        assert env.observation_space.contains(obs)
        again, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs, again)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(np.full(29, 5.0))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert np.all(env.data.ctrl <= env.model.actuator_ctrlrange[:, 1])
            assert np.all(env.data.ctrl >= env.model.actuator_ctrlrange[:, 0])
        assert truncated
        assert sum(info['reward_components'].values()) == pytest.approx(reward)
        with pytest.raises(ValueError):
            env.step(np.full(29, np.nan))
        with pytest.raises(ValueError):
            env.set_command([float('inf'), 0, 0])
    finally:
        env.close()


def test_fallen_state_terminates_and_command_is_observed():
    env = G1VelocityEnv()
    try:
        env.reset(seed=3)
        env.set_command([0.5, 0, 0.2])
        np.testing.assert_allclose(env._observation()[9:12], [0.5, 0, 0.2])
        env.data.qpos[2] = .1
        _, _, terminated, _, info = env.step(np.zeros(29))
        assert terminated and info['fell']
    finally:
        env.close()
