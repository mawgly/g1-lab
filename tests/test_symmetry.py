import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.symmetry import Mirror, joint_mirror
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

REAL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=False,
            gait={'period': .6, 'stance_fraction': .55})


def test_g1_mirror_pairs_left_and_right_joints_by_their_axes():
    env = G1VelocityEnv(**REAL)
    try:
        names = [env.model.joint(j).name for j in env.joints]
        permutation, signs = joint_mirror(env)
        sign = dict(zip(names, signs))
        assert names[permutation[names.index('left_hip_pitch_joint')]] == 'right_hip_pitch_joint'
        assert (sign['left_hip_pitch_joint'], sign['left_knee_joint']) == (1, 1)  # pitch joints keep their sign
        assert (sign['left_hip_roll_joint'], sign['left_hip_yaw_joint'], sign['waist_yaw_joint']) == (-1, -1, -1)
        np.testing.assert_allclose(env.home[permutation] * signs, env.home, atol=1e-9)  # symmetric default pose
        np.testing.assert_allclose(env.scale[permutation], env.scale)
        mirror = Mirror(env)
        x, a = torch.randn(4, env.observation_space.shape[0]), torch.randn(4, 29)
        torch.testing.assert_close(mirror.observation(mirror.observation(x)), x)  # mirroring twice is identity
        torch.testing.assert_close(mirror.action(mirror.action(a)), a)
    finally:
        env.close()


CROUCH = dict(REAL, history=5, height_command=True, upper_body_slot=True)


def test_mirror_covers_the_height_command_and_the_upper_body_slot():
    """height_command / upper_body_slot (iteration 28): 589 entries, an involution; the height and the slot's flag
    keep their place and sign, the slot's waist and arm targets follow the joint map (left shoulder ↔ right shoulder,
    roll and yaw joints change sign), in every history frame."""
    from humanoid_sim.tasks import g1_tensor_math as task
    env = G1VelocityEnv(**CROUCH)
    try:
        mirror = Mirror(env)
        assert len(mirror.obs_index) == env.observation_space.shape[0] == 589
        x = torch.randn(4, 589)
        torch.testing.assert_close(mirror.observation(mirror.observation(x)), x)
        names = [env.model.joint(j).name for j in env.joints]
        permutation, signs = joint_mirror(env)
        height = task.term_positions(29, 5, env.extra_terms, task.HEIGHT_TERM)
        slot = task.term_positions(29, 5, env.extra_terms, task.UPPER_BODY_TERM)
        assert mirror.obs_index[height].tolist() == height and mirror.obs_sign[height].tolist() == [1.] * 5
        for frame in range(5):
            entries = slot[18 * frame:18 * (frame + 1)]
            assert int(mirror.obs_index[entries[0]]) == entries[0] and float(mirror.obs_sign[entries[0]]) == 1
            for joint in range(12, 29):
                position = entries[1 + joint - 12]
                assert int(mirror.obs_index[position]) == entries[1 + permutation[joint] - 12]
                assert float(mirror.obs_sign[position]) == signs[joint]
        left = names.index('left_shoulder_roll_joint')
        assert names[permutation[left]] == 'right_shoulder_roll_joint' and signs[left] == -1
    finally:
        env.close()


def test_mirrored_state_and_action_give_the_mirrored_observation_with_a_height_command():
    """The physics check below with the extra terms and a crouch command (0.70) in both environments."""
    first, second = G1VelocityEnv(**CROUCH), G1VelocityEnv(**CROUCH)
    try:
        mirror = Mirror(first)
        permutation, signs = mirror.action_index.numpy(), mirror.action_sign.numpy().astype(float)
        rng = np.random.default_rng(0)
        first.reset(seed=3, options={'command': [.6, .2, .3], 'height': .7})
        for _ in range(7):
            first.step(rng.normal(0, .5, 29))
        second.reset(seed=3, options={'command': [.6, -.2, -.3], 'height': .7})
        qpos, qvel = first.data.qpos.copy(), first.data.qvel.copy()
        second.data.qpos[:7] = qpos[0], -qpos[1], qpos[2], qpos[3], -qpos[4], qpos[5], -qpos[6]
        second.data.qpos[second.qadr] = qpos[first.qadr][permutation] * signs
        second.data.qvel[:6] = qvel[0], -qvel[1], qvel[2], -qvel[3], qvel[4], -qvel[5]
        second.data.qvel[second.vadr] = qvel[first.vadr][permutation] * signs
        second.previous_action[:] = first.previous_action[permutation] * signs
        second.steps = first.steps + round(first.gait['period'] / first.dt) // 2
        mujoco.mj_forward(second.model, second.data)
        second._observation('fill')
        first.frames[:, :-1] = first.frames[:, -1:]  # one frame repeated in both: the history mirrors as a whole
        action = rng.normal(0, .5, 29)
        observed = first.step(action)[0]
        mirrored = second.step(action[permutation] * signs)[0]
        assert observed[494] == pytest.approx(-.08, abs=1e-6)
        np.testing.assert_allclose(mirrored, mirror.observation(torch.tensor(observed)).numpy(), rtol=1e-2, atol=2e-3)
    finally:
        first.close()
        second.close()


def test_mirrored_state_and_action_give_the_mirrored_observation():
    """The whole map against the physics, gait phase included: mirror a state and step with the mirrored action."""
    first, second = G1VelocityEnv(**REAL), G1VelocityEnv(**REAL)
    try:
        mirror = Mirror(first)
        permutation, signs = mirror.action_index.numpy(), mirror.action_sign.numpy().astype(float)
        rng = np.random.default_rng(0)
        first.reset(seed=3, options={'command': [.6, .2, .3]})
        for _ in range(7):
            first.step(rng.normal(0, .5, 29))
        second.reset(seed=3, options={'command': [.6, -.2, -.3]})
        qpos, qvel = first.data.qpos.copy(), first.data.qvel.copy()
        second.data.qpos[:7] = qpos[0], -qpos[1], qpos[2], qpos[3], -qpos[4], qpos[5], -qpos[6]
        second.data.qpos[second.qadr] = qpos[first.qadr][permutation] * signs
        second.data.qvel[:6] = qvel[0], -qvel[1], qvel[2], -qvel[3], qvel[4], -qvel[5]
        second.data.qvel[second.vadr] = qvel[first.vadr][permutation] * signs
        second.previous_action[:] = first.previous_action[permutation] * signs
        second.steps = first.steps + round(first.gait['period'] / first.dt) // 2  # right foot: half a cycle later
        mujoco.mj_forward(second.model, second.data)
        action = rng.normal(0, .5, 29)
        observed = first.step(action)[0]
        mirrored = second.step(action[permutation] * signs)[0]
        # The model itself is symmetric to ~0.2% (a waist joint velocity differs by that much); a wrong sign or
        # pair would be off by 100%.
        np.testing.assert_allclose(mirrored, mirror.observation(torch.tensor(observed)).numpy(), rtol=1e-2, atol=2e-3)
    finally:
        first.close()
        second.close()
