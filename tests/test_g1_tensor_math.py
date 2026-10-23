import mujoco
import numpy as np
import torch
import pytest
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_tensor_math import observation, reward_components, pd_torque


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).unsqueeze(0)


def test_tensor_pd_matches_cpu_controller():
    env = G1VelocityEnv()
    try:
        env.reset(seed=12, options={'command': [.7, .1, .2]})
        rng = np.random.default_rng(18)
        for _ in range(8):
            action = rng.uniform(-1.3, 1.3, 29)
            clipped = np.clip(action, -1, 1)
            target = np.clip(env.home+env.scale*clipped, *env.model.jnt_range[env.joints].T)
            expected = np.clip(env.kp*(target-env.data.qpos[env.qadr])-env.kd*env.data.qvel[env.vadr],
                               *env.model.actuator_ctrlrange.T)
            torque = pd_torque(tensor(action), tensor(env.data.qpos[env.qadr]), tensor(env.data.qvel[env.vadr]),
                               tensor(env.home), tensor(env.scale), tensor(env.kp), tensor(env.kd),
                               tensor(env.model.jnt_range[env.joints, 0]), tensor(env.model.jnt_range[env.joints, 1]),
                               tensor(env.model.actuator_ctrlrange[:, 0]), tensor(env.model.actuator_ctrlrange[:, 1]))
            np.testing.assert_allclose(torque[0], expected, atol=1e-4, rtol=2e-5)
            env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize('weights', [None, {'velocity': 1., 'action_change': 0., 'gait': 2.}])
def test_cpu_step_reports_shared_reward_components(weights):
    env = G1VelocityEnv(reward_weights=weights)
    try:
        obs, _ = env.reset(seed=3, options={'command': [.7, 0, 0]})
        assert obs.shape == (101,)
        np.testing.assert_allclose(obs[-2:], [0., 1.], atol=1e-6)  # gait clock starts at phase 0
        for _ in range(5):
            obs, reward, terminated, _, info = env.step(np.zeros(29))
            assert set(info['reward_components']) == set(env.reward_weights)
            assert sum(info['reward_components'].values()) == pytest.approx(reward)
        np.testing.assert_allclose(obs[9:12], [.7, 0, 0])
        np.testing.assert_allclose(obs[-31:-2], 0., atol=1e-7)
    finally:
        env.close()


def _rewards(command, foot_height, phase, height=.78, position=None, torso=None, weights=None):
    n = len(command)
    position = torch.zeros(n, 29) if position is None else position
    torso = torch.eye(3).repeat(n, 1, 1) if torso is None else torso
    return reward_components(torch.zeros(n, 6), torch.eye(3).repeat(n, 1, 1), torch.tensor(height).expand(n),
                             torch.tensor(command), torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29),
                             torch.tensor(foot_height), torch.tensor(phase), position, torch.zeros(29), torso,
                             weights=weights)


def _pitch(angle):
    """Rotation about y: positive angle tilts the up axis towards +x (forward lean)."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=torch.float32)


def _yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float32)


def test_heading_hold_steers_back_and_keeps_explicit_turns():
    rotation = torch.stack([_yaw(.6), _yaw(-3.), _yaw(.6)])
    command = torch.tensor([[1., 0, 0], [1., 0, 0], [1., 0, .4]])
    effective = task.effective_command(command, rotation, torch.zeros(3))
    torch.testing.assert_close(effective[:, 2], torch.tensor([-.3, 1., .4]))  # clipped to 1 rad/s; explicit kept
    torch.testing.assert_close(effective[:, :2], command[:, :2])


def test_torso_lean_in_heading_frame_and_reward():
    torso = torch.stack([_yaw(1.2) @ _pitch(.3), _pitch(task.TORSO_LEAN), _pitch(-.5)])
    pelvis = torch.stack([_yaw(1.2), torch.eye(3), torch.eye(3)])
    forward, side = task.torso_lean(torso, pelvis)
    torch.testing.assert_close(forward, torch.tensor([.3, task.TORSO_LEAN, -.5]))
    torch.testing.assert_close(side, torch.zeros(3), atol=1e-6, rtol=0)
    weights = {**task.G1_REWARD_WEIGHTS, 'torso': 2.}
    components, _ = _rewards([[1., 0, 0]]*2, [[.034, .034]]*2, [.25, .25], torso=torso[1:], weights=weights)
    torch.testing.assert_close(components['torso'], torch.tensor([0., -2*(.5+task.TORSO_LEAN)**2]))
    components, _ = _rewards([[1., 0, 0]], [[.034, .034]], [.25], torso=torso[2:])
    assert components['torso'][0] == 0  # absent unless enabled: older checkpoints keep their reward


def test_cpu_heading_hold_changes_observed_yaw_command():
    import mujoco as mj
    for hold, expected in ((False, 0.), (True, -.3)):
        env = G1VelocityEnv(heading_hold=hold)
        try:
            env.reset(seed=2, options={'command': [.5, 0, 0]})
            env.data.qpos[3:7] = [np.cos(.3), 0, 0, np.sin(.3)]  # turned 0.6 rad left of the start heading
            mj.mj_forward(env.model, env.data)
            assert env._observation()[11] == pytest.approx(expected, abs=1e-5)
            np.testing.assert_allclose(env.command, [.5, 0, 0])  # the requested command is unchanged
        finally:
            env.close()


def test_gait_rewards_follow_clock_and_standing_command():
    planted, left_up = [.034, .034], [.10, .034]
    components, fell = _rewards([[0., 0, 0], [1., 0, 0], [1., 0, 0], [1., 0, 0]],
                                [planted, planted, left_up, [.034, .10]], [.0, .25, .25, .25])
    assert not fell.any()
    # Standing: both feet down is the target. Walking at phase 0.25: left stance, right swing.
    torch.testing.assert_close(components['gait'], torch.tensor([.5, .25, 0., .5]))
    assert components['swing'][0] == 0 and components['swing'][1] == 0
    torch.testing.assert_close(components['swing'][2:], torch.full((2,), -10*(.10-task.SWING_HEIGHT)**2))


def test_fall_posture_and_batch_independence():
    position = torch.zeros(3, 29)
    position[1, 15] = .5  # arm joint away from default pose
    position[2, 0] = .5   # hip pitch is free for walking
    components, fell = _rewards([[0., 0, 0]]*3, [[.034, .034]]*3, [0., 0., 0.], height=[.8, .8, .5],
                                position=position)
    assert fell.tolist() == [False, False, True]
    torch.testing.assert_close(components['posture'], torch.tensor([0., -.1*.25, 0.]))
    torch.testing.assert_close(components['fall'], torch.tensor([0., 0., -5.]))


def test_command_sampling_and_resampling():
    generator = torch.Generator().manual_seed(1)
    commands = task.sample_commands(20000, lambda n: torch.rand(n, generator=generator))
    assert commands[:, 1:].abs().max() == 0
    assert 0 <= commands[:, 0].min() and commands[:, 0].max() <= task.MAX_FORWARD_SPEED
    assert abs(float((commands[:, 0] == 0).float().mean()) - task.STAND_PROBABILITY) < .02
    env = G1VelocityEnv(max_episode_steps=2000)
    try:
        env.reset(seed=5)
        seen = {tuple(env.command)}
        for _ in range(1000):  # the robot falls; only the command schedule matters here
            env.step(np.zeros(29))
            seen.add(tuple(env.command))
        assert len(seen) > 1
        env.reset(seed=5, options={'command': [.5, 0, 0]})
        for _ in range(500):
            env.step(np.zeros(29))
        np.testing.assert_allclose(env.command, [.5, 0, 0])
    finally:
        env.close()


def test_body_velocity_matches_mujoco_at_rotated_moving_states():
    from humanoid_sim.tasks.g1_tensor_math import body_velocity
    env = G1VelocityEnv()
    rng = np.random.default_rng(123)
    try:
        env.reset(seed=42)
        for _ in range(12):
            quat = rng.normal(size=4)
            env.data.qpos[3:7] = quat/np.linalg.norm(quat)
            env.data.qvel[:] = rng.normal(size=env.model.nv)
            mujoco.mj_forward(env.model, env.data)
            expected = np.zeros(6)
            mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.pelvis, expected, 1)
            root = env.model.body_rootid[env.pelvis]
            actual = body_velocity(tensor(env.data.cvel[env.pelvis]), tensor(env.data.xipos[env.pelvis]),
                                   tensor(env.data.subtree_com[root]), tensor(env.data.ximat[env.pelvis].reshape(3, 3)))
            np.testing.assert_allclose(actual[0], expected, atol=5e-6, rtol=2e-5)
    finally:
        env.close()
