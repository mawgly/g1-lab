import json
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
# Iteration 22 (stage 2, commands x −1…2 from candidate 6): the branch the flag would start from.
PARENT = ROOT / 'runs' / 'humanoid_lab' / 'exp-94b331444b3b4788'
# task_config of exp-94b331444b3b4788.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, random_phase=True,
              action_limit=10., heading_hold=True, stand_heading_hold=False, lock_wrists=True, knee_floor=.061,
              arm_perturb=.1, force_push=1.25, gait=dict(period=.6, stance_fraction=.55),
              commands=dict(min_speed=-1., max_speed=2., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                            straight_probability=.5, side_share=.2, side_stop=1.),
              reward_weights=dict(velocity=2., yaw=2., yaw_drift=5., alive=.15, tilt=0., orientation=2., height=10.,
                                  torso=1., still=3., stand_pose=1., slip=1., feet_flat=2., self_contact=1.,
                                  arm_press=.1, posture=.3, hips=1., action_change=.05, fall=100.))


def _rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'pitch':
        return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=torch.float64)
    return torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]], dtype=torch.float64)


def _args(foot_height):
    n = len(foot_height)
    return (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.tensor([[1., 0, 0]] * n, dtype=torch.float64),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29), foot_height, torch.full((n,), .25),
            torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))


def _slip_contact(args, **kwargs):
    """(n, 2) the contact the slip term uses, read back with foot speeds 1 (left) and 2 (right)."""
    weights = {**task.G1_REWARD_WEIGHTS, 'slip': 1.}
    speed = torch.tensor([[1., 2.]] * len(args[0]), dtype=torch.float64)
    slip = -task.reward_components(*args, weights=weights, foot_speed=speed, **kwargs)[0]['slip']
    right = slip > 1.5
    return torch.stack([(slip - 2 * right.to(slip.dtype)) > .5, right], dim=1)


def test_the_flag_charges_only_the_feet_in_contact_by_the_rule_of_gait_and_slip():
    flat, heel, swing = torch.eye(3, dtype=torch.float64), _rotation('pitch', np.radians(-50)), \
        _rotation('pitch', np.radians(-34))  # toe up 50° (heel stance), toe up 34° (v0's mean swing roll)
    roll = _rotation('roll', .3)
    # Rows: heel stance + flat stance; foot rolling in swing + flat stance; both in the air, tilted; heel stance +
    # tilted foot in the air. Sole heights (m) as sole_height(): −0.5 mm on the floor.
    feet = torch.stack([torch.stack([heel, flat]), torch.stack([swing, flat]), torch.stack([swing, roll]),
                        torch.stack([heel, roll])])
    sole = torch.tensor([[-.0005, -.0005], [.05, -.0005], [.03, .03], [-.0005, .02]], dtype=torch.float64)
    # Ankle_roll origins: a heel stance toe up 50° lifts it to 6 cm, above CONTACT_HEIGHT (the old rule's blind spot).
    origin = torch.tensor([[.06, .034], [.09, .034], [.07, .07], [.06, .054]], dtype=torch.float64)
    args = _args(origin)
    weights = {**task.G1_REWARD_WEIGHTS, 'feet_flat': 2.}
    tilt = feet[..., 2, :2].square().sum(dim=-1)  # sin^2 of each foot's tilt
    for sole_contact in (True, False):
        given = dict(foot_rotation=feet, sole_height=sole, sole_contact=sole_contact)
        old = task.reward_components(*args, weights=weights, **given)[0]
        off = task.reward_components(*args, weights=weights, feet_flat_stance=False, **given)[0]
        on = task.reward_components(*args, weights=weights, feet_flat_stance=True, **given)[0]
        assert all(torch.equal(old[key], off[key]) for key in old)  # off: bit for bit the old reward
        assert all(torch.equal(old[key], on[key]) for key in old if key != 'feet_flat')  # only this term changes
        torch.testing.assert_close(old['feet_flat'], -2 * tilt.sum(dim=1), rtol=0, atol=1e-12)  # both feet, always
        contact = _slip_contact(args, foot_rotation=feet, sole_height=sole, sole_contact=sole_contact)
        expected = (sole < task.SOLE_CONTACT_HEIGHT) if sole_contact else (origin < task.CONTACT_HEIGHT)
        assert torch.equal(contact, expected)
        torch.testing.assert_close(on['feet_flat'], -2 * (tilt * contact).sum(dim=1), rtol=0, atol=1e-12)
    # By the sole: the heel stance is charged as before, the foot rolling in swing is not.
    on = task.reward_components(*args, weights=weights, foot_rotation=feet, sole_height=sole, sole_contact=True,
                                feet_flat_stance=True)[0]['feet_flat']
    sin2 = np.sin(np.radians(50)) ** 2
    torch.testing.assert_close(on, torch.tensor([-2 * sin2, 0., 0., -2 * sin2], dtype=torch.float64))
    # Not weighted or no foot frames: no term, flag or not.
    assert task.reward_components(*args, foot_rotation=feet, feet_flat_stance=True)[0]['feet_flat'].abs().max() == 0
    assert task.reward_components(*args, weights=weights, feet_flat_stance=True)[0]['feet_flat'].abs().max() == 0


def test_without_the_flag_the_cpu_reward_is_bitwise_the_old_one_and_physics_never_changes(monkeypatch):
    """The iteration 22 recipe: no flag, flag false and flag true step identically; the reward differs only in the
    feet_flat term, which with the flag counts the feet whose sole is below SOLE_CONTACT_HEIGHT."""
    given = []
    original = task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs.get('feet_flat_stance'))
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    envs = [G1VelocityEnv(**RECIPE), G1VelocityEnv(feet_flat_stance=False, **RECIPE),
            G1VelocityEnv(feet_flat_stance=True, **RECIPE)]
    try:
        for env in envs:
            env.reset(seed=3, options={'command': [1., 0., 0.]})
        rng = np.random.default_rng(1)
        spared = 0
        for _ in range(25):
            action = rng.normal(0, .3, 29)
            (obs, reward, *_, info), (obs_off, reward_off, *_, info_off), (obs_on, reward_on, *_, info_on) = (
                env.step(action) for env in envs)
            assert given[-3:] == [False, False, True]
            np.testing.assert_array_equal(obs, obs_off)
            np.testing.assert_array_equal(obs, obs_on)  # a reward setting only: same physics and observations
            assert reward == reward_off and info['reward_components'] == info_off['reward_components']
            terms, terms_on = info['reward_components'], info_on['reward_components']
            assert {k: v for k, v in terms.items() if k != 'feet_flat'} == \
                   {k: v for k, v in terms_on.items() if k != 'feet_flat'}
            env = envs[2]
            tilt = np.array([env.data.xmat[foot][6] ** 2 + env.data.xmat[foot][7] ** 2 for foot in env.feet])
            contact = env._sole_height()[0].numpy() < task.SOLE_CONTACT_HEIGHT
            assert terms['feet_flat'] == pytest.approx(-2 * tilt.sum(), rel=1e-12, abs=1e-15)
            assert terms_on['feet_flat'] == pytest.approx(-2 * (tilt * contact).sum(), rel=1e-12, abs=1e-15)
            spared += terms_on['feet_flat'] > terms['feet_flat'] + 1e-6
        assert spared  # the reset drops the robot from 13 mm, and random actions lift and tilt the feet
    finally:
        for env in envs:
            env.close()


def _pose(env, toe_up=0., lift=0.):
    """Home joints, the left foot turned toe up by toe_up (rad, ankle pitch), the robot moved so that its lowest sole
    point is 0.5 mm into the floor, then lifted by lift (m)."""
    env.reset(seed=0, options={'command': [1., 0., 0.]})
    names = [env.model.joint(j).name for j in env.joints]
    env.data.qpos[env.qadr] = env.home
    env.data.qpos[env.qadr[names.index('left_ankle_pitch_joint')]] -= toe_up
    env.data.qvel[:] = 0
    mujoco.mj_kinematics(env.model, env.data)
    env.data.qpos[2] += lift - .0005 - float(env._sole_height().min())
    mujoco.mj_forward(env.model, env.data)


def test_a_tilted_foot_in_the_air_is_spared_and_a_heel_stance_is_charged_as_before():
    profile = dict(real_robot=True, action_limit=10., sole_contact=True, reward_weights={'feet_flat': 2.})
    envs = {flag: G1VelocityEnv(feet_flat_stance=flag, **profile) for flag in (False, True)}
    try:
        def step(toe_up, lift):
            for env in envs.values():
                _pose(env, toe_up, lift)
            terms = {flag: env.step(np.zeros(29))[-1]['reward_components']['feet_flat'] for flag, env in envs.items()}
            env = envs[True]
            np.testing.assert_array_equal(env.data.qpos, envs[False].data.qpos)
            tilt = np.array([env.data.xmat[foot][6] ** 2 + env.data.xmat[foot][7] ** 2 for foot in env.feet])
            return terms, tilt, env._sole_height()[0].numpy()

        # Lifted 5 cm, left toe up 34° (v0's mean swing roll): both feet in the air, the left one still tilted 27°.
        terms, tilt, sole = step(np.radians(34), .05)
        assert (sole > task.SOLE_CONTACT_HEIGHT).all() and tilt[0] > .15
        assert terms[False] == pytest.approx(-2 * tilt.sum()) and terms[False] < -.3
        assert terms[True] == 0
        # Left heel on the floor, toe up 50°: the heel bears (the ankle_roll origin is above CONTACT_HEIGHT), the
        # flat right foot hangs about 2 cm up. The heel stance costs what it did; only the right foot's tilt is spared.
        terms, tilt, sole = step(np.radians(50), 0.)
        assert sole[0] < task.SOLE_CONTACT_HEIGHT < sole[1] and tilt[0] > .4
        assert envs[True].data.xpos[envs[True].feet[0], 2] > task.CONTACT_HEIGHT
        assert terms[True] == pytest.approx(-2 * tilt[0]) and terms[True] < -.8
        assert terms[True] - terms[False] == pytest.approx(2 * tilt[1], abs=1e-12)
    finally:
        for env in envs.values():
            env.close()


def test_the_flag_is_validated_off_by_default_labelled_and_keeps_the_checkpoint_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['feet_flat_stance'] is False  # old branches keep their reward
    accepted = validate_config({**base, 'task_config': {'feet_flat_stance': True}})
    assert accepted['task_config']['feet_flat_stance'] is True
    for value in (1, 'true', None, 0.):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'feet_flat_stance': value}})
    field = next(p for p in G1_PARAMETERS if p['key'] == 'feet_flat_stance')
    assert field['type'] == 'bool' and field['default'] is False and field['group'] is None
    assert 'опорной' in field['label']
    for config in ({}, {'legs_only': True}, dict(RECIPE)):
        robot = 'g1_12dof' if config.pop('legs_only', False) else 'g1_29dof'
        plain = task_identity(robot, 'g1_velocity', config)
        flagged = task_identity(robot, 'g1_velocity', {**config, 'feet_flat_stance': True})
        assert flagged == plain
        validate_resume(plain, flagged)


def test_the_iteration_22_branch_can_be_branched_with_the_flag():
    """What scripts/lab/task_branch.py <parent> feet_flat_stance true … does before creating the branch: the parent's
    effective value, the new config validated, its identity against the parent's last stage, the CPU task built."""
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.tasks.config import g1_config
    if not (PARENT / 'run_config.json').exists():
        pytest.skip('local run exp-94b331444b3b4788 is not here')
    parent = read_json(PARENT / 'run_config.json')
    assert 'feet_flat_stance' not in parent['task_config']
    assert g1_config(parent['task_config'])['feet_flat_stance'] is False
    value = json.loads('true')  # task_branch.py reads the value as JSON
    task_config = {**parent['task_config'], 'feet_flat_stance': value}
    config = validate_config({'robot_id': parent['robot_id'], 'task_id': parent['task_id'], 'task_config': task_config})
    assert config['task_config']['feet_flat_stance'] is True
    stages = list_stages(PARENT)
    validate_resume(checkpoint_identity(read_json(stages[-1] / 'metadata.json')),
                    task_identity(parent['robot_id'], parent['task_id'], task_config))
    env = make_task(parent['task_id'], parent['robot_id'], config['task_config'])
    try:
        assert env.unwrapped.feet_flat_stance is True and env.unwrapped.sole_contact is True
    finally:
        env.close()
