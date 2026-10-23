from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import checkpoint_identity, validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

from test_g1_self_contact import CANDIDATE, PROFILE, WEIGHTS, arms_pressed

LAB = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab'
SELF_CONTACT_BRANCH = LAB / 'exp-0de9104930294a18'  # iteration 10 (self_contact 1), still training when written
ARMS = tuple(range(15, 29))


def _args(n, position):
    return (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.zeros(n, 3, dtype=torch.float64), torch.zeros(n, 29),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.full((n, 2), .034, dtype=torch.float64),
            torch.full((n,), .25), position, torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))


def _target(env, action):
    """The CPU controller's target as it was written before task.pd_target (numpy)."""
    return np.clip(env.home + env.scale * np.clip(action, -env.action_limit, env.action_limit),
                   env.model.jnt_range[env.joints, 0], env.model.jnt_range[env.joints, 1])


def test_pd_target_is_the_set_point_of_both_controllers():
    rng = np.random.default_rng(5)
    for limit in (1., 10.):
        env = G1VelocityEnv(action_limit=limit)
        try:
            low, high = (torch.from_numpy(env.model.jnt_range[env.joints, k]) for k in (0, 1))
            home, scale, kp, kd = (torch.from_numpy(value) for value in (env.home, env.scale, env.kp, env.kd))
            for _ in range(20):
                action = np.clip(rng.normal(0, 2 * limit, 29), -limit, limit)
                np.testing.assert_array_equal(env._target(action), _target(env, action))  # bit for bit the old one
                position, velocity = (torch.from_numpy(rng.normal(0, .5, 29)) for _ in range(2))
                torque = task.pd_torque(torch.from_numpy(action), position, velocity, home, scale, kp, kd, low, high,
                                        -torch.inf, torch.inf, limit)
                target = task.pd_target(torch.from_numpy(action), home, scale, low, high, limit)
                torch.testing.assert_close(torque, kp * (target - position) - kd * velocity, rtol=0, atol=0)
        finally:
            env.close()


def test_arm_press_counts_the_arms_only_and_the_term_is_off_by_default():
    position = torch.zeros(4, 29, dtype=torch.float64)
    target = position.clone()
    target[1, 15] = .3  # left shoulder pitch target 0.3 rad from its angle
    target[2, [17, 28]] = torch.tensor([.3, -.4], dtype=torch.float64)  # left shoulder yaw, right wrist yaw
    target[3, :15] = 2.  # legs and waist do not count
    press = task.arm_press(target, position, ARMS)
    torch.testing.assert_close(press, torch.tensor([0., .09, .25, 0.], dtype=torch.float64))
    assert task.arm_press(target[:, :12], position[:, :12], ()).abs().max() == 0  # legs-only G1: no arms
    weights = {**task.G1_REWARD_WEIGHTS, 'arm_press': .05}
    charged = task.reward_components(*_args(4, position), weights=weights, arm_press=press)[0]
    torch.testing.assert_close(charged['arm_press'], -.05 * press)
    plain = task.reward_components(*_args(4, position), weights=weights)[0]
    assert plain['arm_press'].abs().max() == 0  # not given: zero
    assert all(torch.equal(plain[key], charged[key]) for key in plain if key != 'arm_press')
    assert task.G1_REWARD_WEIGHTS['arm_press'] == 0
    assert task.reward_components(*_args(4, position), arm_press=press)[0]['arm_press'].abs().max() == 0


def test_arm_press_off_keeps_the_reward_bit_for_bit(monkeypatch):
    plain = G1VelocityEnv(reward_weights=WEIGHTS, **PROFILE)
    weighted = G1VelocityEnv(reward_weights={**WEIGHTS, 'arm_press': .05}, **PROFILE)
    given, original = [], task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    try:
        for env in (plain, weighted):
            env.reset(seed=3, options={'command': [.5, 0., 0.]})
        rng = np.random.default_rng(1)
        for _ in range(12):
            action = arms_pressed() + rng.normal(0, .5, 29)
            (obs, reward, *_, info), (obs_w, reward_w, *_, info_w) = (env.step(action) for env in (plain, weighted))
            assert given[-2]['arm_press'] is None and given[-1]['arm_press'] is not None
            np.testing.assert_array_equal(obs, obs_w)  # a reward weight only: same physics and observations
            components, components_w = info['reward_components'], info_w['reward_components']
            assert components['arm_press'] == 0
            assert reward == sum(value for key, value in components.items() if key != 'arm_press')
            assert {k: v for k, v in components.items() if k != 'arm_press'} == {
                k: v for k, v in components_w.items() if k != 'arm_press'}
            assert reward_w == pytest.approx(reward + components_w['arm_press'], rel=1e-12, abs=1e-12)
            # The policy's target of this action against the joint angles after the step.
            expected = ((_target(weighted, action) - weighted.data.qpos[weighted.qadr])[list(ARMS)] ** 2).sum()
            assert components_w['arm_press'] == pytest.approx(-.05 * expected, rel=1e-12)
    finally:
        for env in (plain, weighted):
            env.close()


def test_cpu_arm_press_is_near_zero_in_the_home_pose_and_large_against_the_body():
    env = G1VelocityEnv(reward_weights={**WEIGHTS, 'arm_press': 1.}, **PROFILE)
    try:
        assert env.arm_joints == ARMS
        env.reset(seed=4, options={'command': [0., 0., 0.]})
        free = [-env.step(np.zeros(29))[-1]['reward_components']['arm_press'] for _ in range(10)]
        assert max(free) < .02  # target home, the arms hang there (~0.0004 rad² at the start, 0.009 after 10 steps)
        for _ in range(10):
            press = -env.step(arms_pressed())[-1]['reward_components']['arm_press']
        assert press > 4.  # shoulder roll and yaw targets at their limits, the torso holds the upper arms
    finally:
        env.close()
    legs = G1VelocityEnv(reward_weights={'arm_press': 1.}, legs_only=True)
    try:
        assert legs.arm_joints == ()
        legs.reset(seed=4, options={'command': [0., 0., 0.]})
        assert legs.step(np.ones(12))[-1]['reward_components']['arm_press'] == 0
    finally:
        legs.close()


def test_cpu_arm_press_charges_the_policy_target_under_action_delay():
    """With a delay of 15 substeps the motors follow the previous (zero) action for the whole first step, so the arm
    stays at home while the policy's target is Δ away: the term is −w·Δ², from the policy's target."""
    delta, weight = .5, 2.
    env = G1VelocityEnv(reward_weights={**WEIGHTS, 'arm_press': weight}, action_delay=.03, **PROFILE)
    try:
        env.reset(seed=4, options={'command': [0., 0., 0.], 'action_delay_steps': 15})
        action = np.zeros(29)
        action[15] = delta / env.scale[15]  # left shoulder pitch target Δ from home
        term = env.step(action)[-1]['reward_components']['arm_press']
        assert term == pytest.approx(-weight * delta ** 2, abs=weight * .01)
        expected = ((_target(env, action) - env.data.qpos[env.qadr])[list(ARMS)] ** 2).sum()
        assert term == pytest.approx(-weight * expected, rel=1e-12)
    finally:
        env.close()


def _latest_stage(run_dir):
    from humanoid_sim.core import list_stages
    stages = list_stages(run_dir) if run_dir.exists() else []
    if not stages:
        pytest.skip(f'{run_dir.name} has no stage on this machine')
    return stages[-1]


def test_the_candidate_presses_its_arms_and_keeps_its_contract_with_the_term():
    from humanoid_sim.core import load_stage, read_json, stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    if not (CANDIDATE / 'metadata.json').exists():
        pytest.skip('candidate stage is not on this machine')
    for stage in (CANDIDATE, _latest_stage(SELF_CONTACT_BRANCH)):
        config = stage_task_config(stage)
        weighted = {**config, 'reward_weights': {**config['reward_weights'], 'arm_press': .05}}
        validate_resume(checkpoint_identity(read_json(stage / 'metadata.json')),
                        task_identity('g1_29dof', 'g1_velocity', weighted))
        assert task_identity('g1_29dof', 'g1_velocity', weighted) == task_identity('g1_29dof', 'g1_velocity', config)
    model, norm = load_stage(CANDIDATE)
    config = stage_task_config(CANDIDATE)
    assert 'arm_press' not in read_json(CANDIDATE / 'metadata.json')['task_config']['reward_weights']
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'reward_weights': {**config['reward_weights'],
                                                                            'arm_press': .05}})
    try:
        obs, _ = env.reset(seed=73000, options={'command': [0., 0., 0.]})
        press = []
        for _ in range(40):  # standing: the shoulders lean on the torso, the hands on the hips
            obs, _, fell, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            assert not fell
            press.append(-info['reward_components']['arm_press'] / .05)
        # 1.9 rad² at the first step, 14-15 once the press has settled (exp-0de9 @ 4995.4 M: 13.3).
        assert press[0] < 3 and 12. < np.mean(press[10:]) < 17.
    finally:
        env.close()


def test_arm_press_weight_is_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['reward_weights']['arm_press'] == 0  # old branches keep theirs
    accepted = validate_config({**base, 'task_config': {'reward_weights': {'arm_press': .05}}})
    assert accepted['task_config']['reward_weights']['arm_press'] == .05
    for value in (-.1, 101., float('nan'), '1'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'arm_press': value}}})
    assert 'рук' in G1_LABELS['arm_press']
    assert [p['default'] for p in G1_PARAMETERS if p['key'] == 'arm_press'] == [0.]
