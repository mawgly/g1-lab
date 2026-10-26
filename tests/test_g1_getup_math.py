"""g1_getup equations (iteration 33): the head, the reward and the assist force, the same numbers in numpy and torch."""
import math

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS, G1_GETUP_TASK_WEIGHTS
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv


def rotations(rng, n, tilt=None):
    """(n, 3, 3) rotations: random ones, or with the z axis tilted by `tilt` (rad, (n,)) in a random direction."""
    out = np.zeros((n, 3, 3))
    for k in range(n):
        if tilt is None:
            quat = rng.normal(size=4)
        else:
            axis = np.r_[rng.normal(size=2), 0.]
            axis /= np.linalg.norm(axis)
            quat = np.r_[math.cos(tilt[k] / 2), math.sin(tilt[k] / 2) * axis]
        quat /= np.linalg.norm(quat)
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, quat)
        out[k] = flat.reshape(3, 3)
    return out


def batch(n=64, seed=0):
    """Random g1_getup reward inputs, every term in play: lying, half up and standing worlds."""
    rng = np.random.default_rng(seed)
    env = G1VelocityEnv()
    try:
        low, high = env.model.jnt_range[env.joints].T
        home = env.home.copy()
    finally:
        env.close()
    soft_low, soft_high = task.soft_limits(low, high)
    standing = np.arange(n) % 3 == 0
    tilt = np.where(standing, rng.uniform(0, .2, n), rng.uniform(0, math.pi, n))
    inputs = dict(
        head=np.where(standing, rng.uniform(1.1, 1.35, n), rng.uniform(.05, 1.3, n)),
        height=np.where(standing, rng.uniform(.65, .8, n), rng.uniform(.05, .8, n)),
        torso_rotation=rotations(rng, n, tilt), velocity=rng.normal(0, .5, (n, 6)),
        position=np.where(standing[:, None], home + rng.normal(0, .1, (n, 29)), rng.uniform(low, high, (n, 29))),
        home=home, soft_low=soft_low, soft_high=soft_high, limit_joints=range(task.LEGS),
        sole_height=np.where(standing[:, None], rng.uniform(-.001, .012, (n, 2)), rng.uniform(-.001, .3, (n, 2))),
        torque=rng.normal(0, 20, (n, 29)), joint_velocity=rng.normal(0, 3, (n, 29)),
        action=rng.normal(0, 2, (n, 29)), previous_action=rng.normal(0, 2, (n, 29)))
    return inputs


def as_torch(inputs, dtype=torch.float64):
    return {key: torch.as_tensor(value, dtype=dtype) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()}


def test_the_head_top_is_the_head_meshs_highest_point():
    """HEAD_TOP in the torso frame is the highest vertex of the head_link mesh standing in the home pose (the critic's
    mechanics.py: 1.324 m with the pelvis at 0.793), and head_position follows the torso in any pose."""
    env = G1VelocityEnv()
    try:
        env.reset(seed=0, options={'command': [0., 0., 0.]})
        env.data.qpos[env.qadr] = env.home
        env.data.qpos[2] = .793
        mujoco.mj_forward(env.model, env.data)
        torso = env.torso
        head = task.head_position(env.data.xpos[torso][None], env.data.xmat[torso].reshape(1, 3, 3))[0]
        meshes = [g for g in range(env.model.ngeom) if env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and env.model.mesh(env.model.geom_dataid[g]).name == 'head_link']
        vertices = np.vstack([env.data.geom_xpos[g] + env.model.mesh_vert[
            env.model.mesh_vertadr[env.model.geom_dataid[g]]:][:env.model.mesh_vertnum[env.model.geom_dataid[g]]]
            @ env.data.geom_xmat[g].reshape(3, 3).T for g in meshes])
        top = vertices[vertices[:, 2].argmax()]
        np.testing.assert_allclose(head, top, atol=6e-4)
        assert head[2] == pytest.approx(1.324, abs=1e-3)
        rng = np.random.default_rng(1)  # any pose: the same torso-frame point
        for _ in range(5):
            env.data.qpos[3:7] = rng.normal(size=4)
            env.data.qpos[3:7] /= np.linalg.norm(env.data.qpos[3:7])
            env.data.qpos[env.qadr] = rng.uniform(*env.model.jnt_range[env.joints].T)
            mujoco.mj_forward(env.model, env.data)
            R, p = env.data.xmat[torso].reshape(3, 3), env.data.xpos[torso]
            np.testing.assert_allclose(task.head_height(p[None], R[None])[0], (p + R @ task.HEAD_TOP)[2], atol=1e-12)
    finally:
        env.close()


def test_numpy_and_torch_give_the_same_numbers():
    inputs = batch()
    expected, standing = task.getup_reward_components(**inputs)
    assert set(expected) == set(G1_GETUP_TASK_WEIGHTS) and all(v.shape == (64,) for v in expected.values())
    assert isinstance(standing, np.ndarray) and 5 < standing.sum() < 30  # a share of the worlds stands
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-4)):
        got, got_standing = task.getup_reward_components(**as_torch(inputs, dtype))
        np.testing.assert_array_equal(got_standing.numpy(), standing)
        for key in expected:
            np.testing.assert_allclose(got[key].numpy(), expected[key], rtol=tolerance, atol=tolerance, err_msg=key)
    rotation, torso = inputs['torso_rotation'], np.random.default_rng(3).normal(size=(64, 3))
    np.testing.assert_allclose(task.head_height(torch.as_tensor(torso), torch.as_tensor(rotation)).numpy(),
                               task.head_height(torso, rotation), rtol=0, atol=1e-12)
    force = np.random.default_rng(4).uniform(0, 200, 64)
    np.testing.assert_array_equal(task.assist_force(torch.as_tensor(rotation), torch.as_tensor(force)).numpy(),
                                  task.assist_force(rotation, force))
    mean = np.where(np.arange(64) % 4 == 0, np.nan, np.random.default_rng(5).uniform(.5, 1.5, 64))
    np.testing.assert_array_equal(task.assist_update(torch.as_tensor(force), torch.as_tensor(mean)).numpy(),
                                  task.assist_update(force, mean))
    np.testing.assert_array_equal(task.getup_success_pose(*as_torch(dict(
        a=inputs['head'], b=inputs['height'], c=inputs['torso_rotation'], d=inputs['sole_height'])).values()).numpy(),
        task.getup_success_pose(inputs['head'], inputs['height'], inputs['torso_rotation'], inputs['sole_height']))


def test_the_terms_lying_and_standing():
    """Lying (head 0.1 m, pelvis 0.1 m, torso on its back) earns nothing of the task; standing in the home pose, still,
    feet on the floor, earns head 1, upright 0.5, the bonus 1 and the pose 0.5; regularization as specified."""
    weights = G1_GETUP_REWARD_WEIGHTS
    supine = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])  # the torso's z axis along world x
    inputs = batch(n=2)
    inputs.update(head=np.array([.1, 1.33]), height=np.array([.1, .79]), torso_rotation=np.stack([supine, np.eye(3)]),
                  velocity=np.zeros((2, 6)), position=np.stack([inputs['home']] * 2), sole_height=np.array(
                      [[.2, .2], [-.0006, -.0006]]), torque=np.zeros((2, 29)), joint_velocity=np.zeros((2, 29)),
                  action=np.zeros((2, 29)), previous_action=np.zeros((2, 29)))
    terms, standing = task.getup_reward_components(**inputs)
    assert standing.tolist() == [False, True]
    assert {key: float(value[0]) for key, value in terms.items()} == {key: 0. for key in terms}
    assert {key: float(value[1]) for key, value in terms.items()} == dict(
        head=1., upright=.5, stand=1., stand_pose=.5, stand_still=0., effort=0., joint_speed=0., action_change=0.,
        joint_limits=0., impact=0., sole_impact=0., stance_width=0., arm_press=0., style=0., head_rise=0.,
        sole_slip=0.)
    # Head term linear between 0.15 and 1.20; upright only with the pelvis above 0.4.
    inputs.update(head=np.array([.15 + 1.05 / 2, 2.]), height=np.array([.39, .41]),
                  torso_rotation=np.stack([np.eye(3), np.eye(3)]))
    terms, _ = task.getup_reward_components(**inputs)
    assert terms['head'].tolist() == pytest.approx([.5, 1.]) and terms['upright'].tolist() == [0., .5]
    # Regularization: τ², q̇², Δa², and the legs beyond their soft limits only.
    inputs.update(torque=np.full((2, 29), 10.), joint_velocity=np.full((2, 29), 2.), action=np.ones((2, 29)))
    position = np.stack([inputs['home']] * 2)
    knee, wrist = 3, 20
    position[1, knee] = inputs['soft_high'][knee] + .1  # a knee 0.1 rad into its last 5 %
    position[1, wrist] = inputs['soft_high'][wrist] + .3  # an arm joint: not a leg, not charged
    inputs.update(position=position)
    terms, _ = task.getup_reward_components(**inputs)
    assert terms['effort'][0] == pytest.approx(-weights['effort'] * 29 * 100)
    assert terms['joint_speed'][0] == pytest.approx(-weights['joint_speed'] * 29 * 4)
    assert terms['action_change'][0] == pytest.approx(-weights['action_change'] * 29)
    assert terms['joint_limits'].tolist() == pytest.approx([0., -weights['joint_limits'] * .1])
    # Moving while standing costs ‖v‖²; the pose term falls with the joints' mean squared offset.
    inputs.update(velocity=np.array([[0.] * 6, [0., 0., 0., .3, .4, 0.]]), position=np.stack(
        [inputs['home'], inputs['home'] + .1]), torque=np.zeros((2, 29)), joint_velocity=np.zeros((2, 29)),
        action=np.zeros((2, 29)), head=np.array([1.33, 1.33]), height=np.array([.79, .79]),
        sole_height=np.full((2, 2), -.0006))
    terms, standing = task.getup_reward_components(**inputs)
    assert standing.tolist() == [True, True]
    assert terms['stand_still'].tolist() == pytest.approx([0., -weights['stand_still'] * .25])
    assert terms['stand_pose'].tolist() == pytest.approx([.5, .5 * math.exp(-1)])
    # Not standing: a foot in the air, the head too low, the torso tilted 16°, the pelvis below 0.70.
    for change in (dict(sole_height=np.array([[-.0006, .02]] * 2)), dict(head=np.array([1.14] * 2)),
                   dict(torso_rotation=rotations(np.random.default_rng(0), 2, np.radians([16., 16.]))),
                   dict(height=np.array([.69] * 2))):
        assert not task.getup_reward_components(**{**inputs, **change})[1].any(), change


def test_the_soft_limits_are_unitrees_and_the_knee_floor():
    env = G1VelocityEnv()
    try:
        low, high = env.model.jnt_range[env.joints].T
        soft_low, soft_high = task.soft_limits(low, high)
        np.testing.assert_allclose(soft_high - soft_low, .9 * (high - low))
        np.testing.assert_allclose(soft_low - low, .05 * (high - low))
        assert soft_low[3] == pytest.approx(.061, abs=1e-3)  # the knee: task_config knee_floor's working value
    finally:
        env.close()


def test_the_assist_pulls_nearly_upright_and_steps_down():
    rng = np.random.default_rng(0)
    tilt = np.radians([0., 30., 36.5, 37.5, 60., 90., 180.])
    force = np.full(7, 200.)
    pulled = task.assist_force(rotations(rng, 7, tilt), force)
    assert pulled.tolist() == [200., 200., 200., 0., 0., 0., 0.]  # projected gravity z < −0.8: tilt < 36.87°
    assert task.GETUP_ASSIST_FORCE == 200 and task.GETUP_ASSIST_STEP == 20
    force = np.array([200., 200., 10., 0., 100.])
    mean = np.array([1.01, .99, 1.2, 1.3, np.nan])
    assert task.assist_update(force, mean).tolist() == [180., 200., 0., 0., 100.]


def test_assist_forces_carry_over_to_another_number_of_worlds():
    forces = np.array([200., 0., 120., 40., 180., 60., 100., 20.])
    assert task.assist_resample(forces, 8) is forces  # the same worlds: as they are
    wider = task.assist_resample(forces, 4096)
    assert wider.shape == (4096,) and wider.min() == 0 and wider.max() == 200
    assert wider.mean() == pytest.approx(forces.mean(), abs=.5)
    np.testing.assert_array_equal(task.assist_resample(forces, 4), [20., 60., 120., 200.])  # sorted, every second
    torch.testing.assert_close(task.assist_resample(torch.as_tensor(forces), 4), torch.tensor([20., 60., 120., 200.],
                                                                                              dtype=torch.float64))
