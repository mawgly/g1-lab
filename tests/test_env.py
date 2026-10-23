import numpy as np
import pytest
import mujoco
from stable_baselines3.common.env_checker import check_env
from arm_sim.env import ArmReachEnv


def test_gym_contract_and_seed():
    env = ArmReachEnv()
    check_env(env)
    a, _ = env.reset(seed=42)
    target = env.target.copy()
    b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(target, env.target)
    env.close()


def test_targets_have_collision_free_witness():
    env = ArmReachEnv()
    for seed in range(20):
        env.reset(seed=seed)
        probe = mujoco.MjData(env.model)
        probe.qpos[:] = env.goal_qpos
        mujoco.mj_forward(env.model, probe)
        np.testing.assert_allclose(probe.site_xpos[env.tip_id], env.target, atol=1e-7)
        assert probe.ncon == 0
        assert env.target[2] > 0.08


def test_actions_move_arm_and_physics_remains_finite():
    env = ArmReachEnv()
    env.reset(seed=1)
    start = env.data.qpos.copy()
    for _ in range(20):
        obs, reward, _, _, info = env.step(np.array([0.5, 0, 0, 0, 0]))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    assert abs(env.data.qpos[0] - start[0]) > 0.1
    assert info['distance'] >= 0


def test_bad_action_does_not_mutate_state():
    env = ArmReachEnv()
    env.reset(seed=1)
    original = env.data.qpos.copy()
    for action in ([0, 0], [np.nan] * 5, [np.inf] * 5):
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(original, env.data.qpos)


def test_horizon_is_truncation():
    env = ArmReachEnv(max_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(np.zeros(5))
    assert truncated and not terminated


def test_success_requires_holding_target():
    env = ArmReachEnv()
    env.reset(seed=1)
    env.target[:] = env.data.site_xpos[env.tip_id]
    for i in range(5):
        _, _, terminated, _, info = env.step(np.zeros(5))
        assert terminated == (i == 4)
    assert info['is_success']


def test_observation_and_distance_match_current_physics_state():
    env = ArmReachEnv()
    obs, info = env.reset(seed=10)
    for _ in range(10):
        probe = mujoco.MjData(env.model)
        probe.qpos[:] = env.data.qpos
        mujoco.mj_forward(env.model, probe)
        tip = probe.site_xpos[env.tip_id]
        np.testing.assert_allclose(obs[18:21], tip * 3, atol=1e-7)
        assert info['distance'] == pytest.approx(np.linalg.norm(env.target - tip), abs=1e-7)
        obs, _, _, _, info = env.step(np.ones(5))
