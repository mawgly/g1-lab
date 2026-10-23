from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

ROOT = Path(__file__).resolve().parents[1]
# task_config of exp-e4fe1c0b88bb4629 (iteration 3, feet_flat), the branch iteration 4 would start from.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
              stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
              commands=dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                            straight_probability=.5),
              reward_weights=dict(velocity=2., yaw=2., alive=.15, orientation=2., height=10., torso=1., still=3.,
                                  stand_pose=1., slip=1., feet_flat=.5, posture=.3, hips=1., action_change=.05,
                                  fall=100.))


def _counted(foot_height, sole_height=None):
    """(n, 2) bool: which feet the reward counts as on the ground, read back through the slip term."""
    n = len(foot_height)
    args = (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.tensor([[1., 0, 0]] * n, dtype=torch.float64),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29), foot_height, torch.full((n,), .25),
            torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    weights = {**task.G1_REWARD_WEIGHTS, 'slip': 1.}
    speeds = torch.eye(2, dtype=torch.float64)
    return torch.stack([-task.reward_components(*args, weights=weights, foot_speed=speeds[k].repeat(n, 1),
                                                sole_height=sole_height)[0]['slip'] > .5 for k in range(2)], dim=1)


def _touching(env):
    """Per foot: one of its sole spheres touches the floor in MuJoCo (data.contact)."""
    floor = env.model.geom('floor').id
    pairs = [(int(c.geom1), int(c.geom2)) for c in env.data.contact[:env.data.ncon]]
    return [any(floor in pair and set(pair) & set(soles.tolist()) for pair in pairs) for soles in env.soles]


def _pose(env, toe_up=0., lift=0.):
    """Kinematic pose: home joints, the left foot turned toe up by toe_up (rad, ankle pitch), the robot moved
    so that its lowest sole point is 0.5 mm into the floor, then lifted by lift (m)."""
    env.reset(seed=0, options={'command': [1., 0., 0.]})
    names = [env.model.joint(j).name for j in env.joints]
    env.data.qpos[env.qadr] = env.home
    env.data.qpos[env.qadr[names.index('left_ankle_pitch_joint')]] -= toe_up
    env.data.qvel[:] = 0
    mujoco.mj_kinematics(env.model, env.data)
    env.data.qpos[2] += lift - .0005 - float(env._sole_height().min())
    mujoco.mj_forward(env.model, env.data)
    return env._sole_height(), torch.from_numpy(env.data.xpos[env.feet, 2].copy())[None]


def test_sole_points_are_the_foot_spheres_and_sole_height_is_their_lowest_bottom():
    env = G1VelocityEnv()
    try:
        assert env.soles.shape == (2, 4)  # heel and toe, inside and outside, on each foot
        assert all(env.model.geom_bodyid[g] == foot for foot, soles in zip(env.feet, env.soles) for g in soles)
        torch.testing.assert_close(env.sole_radius, torch.full((2, 4), .005, dtype=torch.float64))
    finally:
        env.close()
    spheres = torch.tensor([[[.03, .02, .1, .2], [.5, .5, .5, .0045]]], dtype=torch.float64)
    torch.testing.assert_close(task.sole_height(spheres, torch.full((2, 4), .005)),
                               torch.tensor([[.015, -.0005]], dtype=torch.float64))
    assert task.SOLE_CONTACT_HEIGHT == .01


def test_reward_contact_by_sole_sees_a_heel_stance_the_ankle_height_misses():
    env = G1VelocityEnv(sole_contact=True)
    try:
        sole, origin = _pose(env)  # flat feet on the floor
        assert _touching(env) == [True, True]
        assert _counted(origin, sole).tolist() == [[True, True]] == _counted(origin).tolist()

        sole, origin = _pose(env, toe_up=np.radians(50))  # left foot on its heel, toe up 50°
        assert np.degrees(np.arcsin(env.data.xmat[env.feet[0]][6])) == pytest.approx(50, abs=.01)
        assert float(origin[0, 0]) > task.CONTACT_HEIGHT  # the ankle_roll origin rises above the old threshold
        assert _touching(env) == [True, False]  # the heel bears; the flat right foot hangs about 2 cm up
        assert _counted(origin, sole).tolist() == [[True, False]]
        assert _counted(origin).tolist() == [[False, False]]  # the old rule misses the heel stance

        sole, origin = _pose(env, lift=.02)  # both feet 2 cm above the floor
        assert _touching(env) == [False, False]
        torch.testing.assert_close(sole, torch.full((1, 2), .0195, dtype=torch.float64))
        assert _counted(origin, sole).tolist() == [[False, False]]
    finally:
        env.close()


def test_without_sole_contact_the_reward_is_unchanged_and_physics_never_changes(monkeypatch):
    weights = {'gait': 2., 'slip': 1., 'swing': 10.}
    envs = [G1VelocityEnv(reward_weights=weights), G1VelocityEnv(reward_weights=weights, sole_contact=False),
            G1VelocityEnv(reward_weights=weights, sole_contact=True)]
    given = []
    original = task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs.get('sole_height'))
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    try:
        for env in envs:
            env.reset(seed=3, options={'command': [.5, 0., 0.]})
        rng = np.random.default_rng(1)
        differs = False
        for _ in range(20):
            action = rng.normal(0, .3, 29)
            (obs, reward, *_, info), (obs_off, reward_off, *_, info_off), (obs_sole, reward_sole, *_, info_sole) = (
                env.step(action) for env in envs)
            assert given[-3] is None and given[-2] is None and given[-1] is not None
            np.testing.assert_array_equal(obs, obs_off)
            np.testing.assert_array_equal(obs, obs_sole)  # a reward setting only: same physics and observations
            assert reward == reward_off and info['reward_components'] == info_off['reward_components']
            differs |= reward_sole != reward
        assert differs  # the first steps drop the feet onto the floor: sole 13 mm up, ankle origin 48 mm
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize('real_robot', [False, True])
def test_cpu_sole_contact_matches_mujoco_contacts(real_robot):
    """Zero and small random actions: the robot drops onto its feet, stands, then falls."""
    profile = dict(real_robot=True, action_limit=10.) if real_robot else {}
    agree, caught = [], []
    for scale in (0., .1, .3):
        env = G1VelocityEnv(sole_contact=True, **profile)
        try:
            env.reset(seed=0, options={'command': [0., 0., 0.]})
            rng = np.random.default_rng(0)
            for _ in range(300):
                terminated = env.step(rng.normal(0, scale, 29))[2]
                counted = (env._sole_height()[0] < task.SOLE_CONTACT_HEIGHT).tolist()
                for real, sole in zip(_touching(env), counted):
                    agree.append(real == sole)
                    if real:
                        caught.append(sole)
                if terminated:
                    break
        finally:
            env.close()
    assert len(agree) > 200 and np.mean(agree) >= .9  # measured 0.97-0.98
    assert np.mean(caught) == 1  # every real contact is counted (a touching sphere is below the floor)


def test_sole_contact_is_validated_off_by_default_and_keeps_the_checkpoint_contract():
    from humanoid_sim.lab import validate_config
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['sole_contact'] is False  # old branches keep their reward
    assert validate_config({**base, 'task_config': {'sole_contact': True}})['task_config']['sole_contact'] is True
    for value in (1, 'true', None):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'sole_contact': value}})
    for config in ({}, {'legs_only': True}, RECIPE):
        robot = 'g1_12dof' if config.pop('legs_only', False) else 'g1_29dof'
        plain = task_identity(robot, 'g1_velocity', config)
        assert task_identity(robot, 'g1_velocity', {**config, 'sole_contact': True}) == plain
        validate_resume(plain, task_identity(robot, 'g1_velocity', {**config, 'sole_contact': True}))


def test_the_iteration_3_stage_can_be_branched_with_sole_contact():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    run = ROOT / 'runs' / 'humanoid_lab' / 'exp-e4fe1c0b88bb4629'
    stages = list_stages(run) if run.is_dir() else []
    if not stages:
        pytest.skip('local run exp-e4fe1c0b88bb4629 is not here')
    metadata = read_json(stages[-1] / 'metadata.json')
    assert 'sole_contact' not in metadata['task_config']
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'sole_contact': True})
    validate_resume(checkpoint_identity(metadata), requested)
