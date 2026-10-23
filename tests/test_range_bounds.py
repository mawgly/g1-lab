import numpy as np

from humanoid_sim.full_gpu_train import range_bounds
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tensor_ppo import BOUND


def test_range_bounds_start_the_loss_at_the_box_edge_for_joints_whose_range_lies_beyond_it():
    env = G1VelocityEnv(real_robot=True, action_limit=10.)
    try:
        low, high = range_bounds(env)
        names = [env.model.joint(j).name for j in env.joints]
        limits = env.model.jnt_range[env.joints]
        reach_low, reach_high = ((limits[:, k] - env.home) / env.scale for k in (0, 1))
        wrist = names.index('left_wrist_pitch_joint')
        assert reach_high[wrist] > 10  # the joint range lies beyond the action box
        # The loss starts where the action is clipped: no gap between the clip (10) and the loss (BOUND * bound).
        assert np.isclose(BOUND * high[wrist], 10.) and np.isclose(BOUND * low[wrist], -10.)
        inside = (np.abs(reach_low) < 10 / BOUND) & (np.abs(reach_high) < 10 / BOUND)
        knee = names.index('left_knee_joint')
        assert inside[knee]
        np.testing.assert_allclose(low[inside], reach_low[inside])  # joints inside the box keep their bound
        np.testing.assert_allclose(high[inside], reach_high[inside])
        assert np.all(BOUND * np.abs(np.concatenate([low, high])) <= 10 + 1e-9)
    finally:
        env.close()


def test_range_bounds_start_the_knee_loss_at_the_knee_floor():
    """With knee_floor the knee's lower bound is where the target reaches the floor, (0.061 − home)/scale = −0.683,
    not the stop (−1.106, the loss from −1.22): otherwise a knee mean in −0.68…−1.22 would move neither the floored
    target nor the loss. Everything else is the bound without the floor, on both G1 models."""
    knees = [3, 9]
    for legs_only, limit in ((False, 10.), (True, 4.)):  # boxes wide enough that the knee range lies inside
        plain = G1VelocityEnv(legs_only=legs_only, real_robot=not legs_only, action_limit=limit)
        floored = G1VelocityEnv(legs_only=legs_only, real_robot=not legs_only, action_limit=limit, knee_floor=.061)
        try:
            (low, high), (floor_low, floor_high) = range_bounds(plain), range_bounds(floored)
            np.testing.assert_allclose(floor_low[knees], (.061 - floored.home[knees]) / floored.scale[knees],
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(floor_low[knees], -.683, atol=5e-4)
            np.testing.assert_allclose(low[knees], -1.106, atol=5e-4)  # the joint's stop, without the floor
            others = [i for i in range(plain.model.nu) if i not in knees]
            np.testing.assert_array_equal(floor_low[others], low[others])
            np.testing.assert_array_equal(floor_high, high)
        finally:
            plain.close(), floored.close()
