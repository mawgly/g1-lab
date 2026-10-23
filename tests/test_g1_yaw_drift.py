import math
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
# task_config of exp-4e4a49003563481f (iteration 5, feet_flat x4), the branch iteration 6 would start from.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, action_limit=10.,
              heading_hold=True, stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
              commands=dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                            straight_probability=.5),
              reward_weights=dict(velocity=2., yaw=2., alive=.15, tilt=0., orientation=2., height=10., torso=1.,
                                  still=3., stand_pose=1., slip=1., feet_flat=2., posture=.3, hips=1., action_change=.05,
                                  fall=100.))
DT = .02


def _yaw_inputs(omega, yaw_command):
    """velocity and command batches where only ω_z and the yaw command are what the update may read."""
    n = len(omega)
    velocity = torch.full((n, 6), 5., dtype=torch.float64)
    command = torch.ones((n, 3), dtype=torch.float64)
    velocity[:, 2], command[:, 2] = torch.tensor(omega, dtype=torch.float64), torch.tensor(yaw_command,
                                                                                            dtype=torch.float64)
    return velocity, command


def test_yaw_drift_mean_settles_on_a_steady_error_within_a_few_tau():
    assert task.YAW_DRIFT_SECONDS == 1.2
    # A steady arc of 0.073 rad/s with no turn asked; a turn of 0.25 tracked at 0.30.
    velocity, command = _yaw_inputs([.073, .30], [0., .25])
    target = torch.tensor([.073, .05], dtype=torch.float64)
    alpha = DT / task.YAW_DRIFT_SECONDS
    error = torch.zeros(2, dtype=torch.float64)
    for k in range(1, 5 * 60 + 1):  # 60 control steps per tau
        error = task.yaw_drift_update(error, velocity, command, DT)
        torch.testing.assert_close(error, target * (1 - (1 - alpha) ** k))
        if k == 60:
            assert torch.all((error > .6 * target) & (error < .7 * target))  # 1 - 1/e after one tau
    assert torch.all((error - target).abs() < .01 * target)  # within 1% after 5 tau
    # The gait's own sway (±0.12 rad/s once per 0.6 s cycle) on top of an arc of 0.04 averages out.
    error, seen = torch.zeros(1, dtype=torch.float64), []
    for k in range(400):
        sway = .04 + .12 * math.sin(2 * math.pi * k * DT / .6)
        error = task.yaw_drift_update(error, *_yaw_inputs([sway], [0.]), DT)
        if k >= 300:
            seen.append(float(error[0]))
    assert max(abs(value - .04) for value in seen) < .015


def _args(command):
    n = len(command)
    return (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.tensor(command, dtype=torch.float64),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29), torch.full((n, 2), .034, dtype=torch.float64),
            torch.full((n,), .25), torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))


def test_yaw_drift_charges_the_mean_error_on_straight_commands_only():
    # Requested: straight forward, straight diagonal, straight backwards, a turn, told to stand (twice).
    requested = torch.tensor([[1., 0, 0], [.5, .3, 0], [-.3, 0, 0], [1., 0, .5], [0., 0, 0], [0., 0, 0]],
                             dtype=torch.float64)
    # Rewarded command under heading hold: the zero yaw rate of a straight command is steered, so it is not zero.
    steered = [[1., 0, .2], [.5, .3, -.1], [-.3, 0, .05], [1., 0, .5], [0., 0, 0], [0., 0, 0]]
    assert not task.straight_walking(torch.tensor(steered))[:3].any()  # judged by it nothing would be straight
    error = torch.tensor([.07, -.04, .02, .07, .07, 0.], dtype=torch.float64)
    args = _args(steered)
    weights = {**task.G1_REWARD_WEIGHTS, 'yaw_drift': 5.}

    def components(weights=weights, **kwargs):
        return task.reward_components(*args, weights=weights, **kwargs)[0]

    charged = components(yaw_drift_error=error, requested_command=requested)
    torch.testing.assert_close(charged['yaw_drift'],
                               torch.tensor([-5 * .07, -5 * .04, -5 * .02, 0., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(components(yaw_drift_error=2 * error, requested_command=requested)['yaw_drift'],
                               2 * charged['yaw_drift'])  # linear in the error
    plain = components()
    assert plain['yaw_drift'].abs().max() == 0  # no state given
    assert all(torch.equal(plain[key], charged[key]) for key in plain if key != 'yaw_drift')  # nothing else moves
    assert task.G1_REWARD_WEIGHTS['yaw_drift'] == 0
    assert components(task.G1_REWARD_WEIGHTS, yaw_drift_error=error,
                      requested_command=requested)['yaw_drift'].abs().max() == 0  # absent unless weighted
    with pytest.raises(ValueError):  # the steered command cannot stand in for the requested one
        components(yaw_drift_error=error)


def _turned(env, yaw):
    env.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)


def test_cpu_yaw_drift_follows_the_shared_update_and_restarts_with_every_command(monkeypatch):
    env = G1VelocityEnv(reward_weights={'yaw_drift': 5.}, heading_hold=True)
    try:
        env.reset(seed=3, options={'command': [.6, 0., 0.]})
        assert env.yaw_drift_error.tolist() == [0.]
        _turned(env, .4)  # 0.4 rad off the held heading: the rewarded command steers at -0.2 rad/s
        env._observation()
        assert env.observed_command[2] == pytest.approx(-.2, abs=1e-6)
        rng = np.random.default_rng(2)
        for _ in range(10):
            before, steered = float(env.yaw_drift_error[0]), env.observed_command.copy()
            info = env.step(rng.normal(0, .3, 29))[-1]
            # omega_z: MuJoCo pelvis angular velocity (local frame), as the yaw term reads it.
            expected = before + env.dt / task.YAW_DRIFT_SECONDS * (env._velocity()[2] - steered[2] - before)
            assert float(env.yaw_drift_error[0]) == pytest.approx(expected, rel=1e-12, abs=1e-15)
            assert info['reward_components']['yaw_drift'] == pytest.approx(-5 * abs(expected), rel=1e-12, abs=1e-15)
        assert float(env.yaw_drift_error[0]) > .02  # turning back towards the heading slower than steered
        env.set_command([.6, 0., 0.])  # a command set from outside starts a new mean
        assert env.yaw_drift_error.tolist() == [0.]
        env.set_command([.6, 0., .3])  # a turn: the mean runs, but is not charged
        for _ in range(3):
            assert env.step(np.zeros(29))[-1]['reward_components']['yaw_drift'] == 0
        assert float(env.yaw_drift_error[0]) != 0
        env.reset(seed=4, options={'command': [.6, 0., 0.]})
        assert env.yaw_drift_error.tolist() == [0.]
        env.step(np.zeros(29))
        env.reset(seed=4)  # a sampled command
        assert env.yaw_drift_error.tolist() == [0.]
    finally:
        env.close()
    # Training resamples after the reward: the step is charged with its mean, the next command starts from zero.
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)
    env = G1VelocityEnv(reward_weights={'yaw_drift': 5.}, heading_hold=True,
                        commands={'min_speed': .5, 'max_speed': 1., 'stand_probability': 0.})
    try:
        env.reset(seed=5)
        _turned(env, .4)
        env._observation()
        for _ in range(5):
            steered = env.observed_command.copy()
            info = env.step(np.zeros(29))[-1]
            assert env.yaw_drift_error.tolist() == [0.]
            charged = -5 * abs(env.dt / task.YAW_DRIFT_SECONDS * (env._velocity()[2] - steered[2]))
            assert info['reward_components']['yaw_drift'] == pytest.approx(charged, rel=1e-12, abs=1e-15)
            assert charged < 0
    finally:
        env.close()


def _spy(monkeypatch):
    given, original = [], task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    return given


def test_yaw_drift_off_keeps_the_reward_bit_for_bit_and_keeps_no_state(monkeypatch):
    profile = {key: value for key, value in RECIPE.items() if key != 'reward_weights'}
    weights = RECIPE['reward_weights']
    plain = G1VelocityEnv(reward_weights=weights, **profile)
    weighted = G1VelocityEnv(reward_weights={**weights, 'yaw_drift': 5.}, **profile)
    given = _spy(monkeypatch)
    try:
        for env in (plain, weighted):
            env.reset(seed=3, options={'command': [.8, 0., 0.]})
            _turned(env, .3)
            env._observation()
        rng = np.random.default_rng(1)
        charged = []
        for _ in range(30):
            action = rng.normal(0, .5, 29)
            (obs, reward, *_, info), (obs_w, reward_w, *_, info_w) = (env.step(action) for env in (plain, weighted))
            assert given[-2]['yaw_drift_error'] is None and given[-1]['yaw_drift_error'] is not None
            np.testing.assert_array_equal(obs, obs_w)  # a reward weight only: same physics and observations
            components, components_w = info['reward_components'], info_w['reward_components']
            # Off (the default): the term is zero and the sum is bit for bit the one without it.
            assert components['yaw_drift'] == 0
            assert reward == sum(value for key, value in components.items() if key != 'yaw_drift')
            assert {k: v for k, v in components.items() if k != 'yaw_drift'} == {
                k: v for k, v in components_w.items() if k != 'yaw_drift'}
            assert reward_w == pytest.approx(reward + components_w['yaw_drift'], rel=1e-12, abs=1e-12)
            assert plain.yaw_drift_error.tolist() == [0.]  # no state kept while off
            charged.append(components_w['yaw_drift'])
        assert min(charged) < -.01
    finally:
        for env in (plain, weighted):
            env.close()


def test_yaw_drift_weight_is_validated_labelled_and_keeps_the_checkpoint_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['reward_weights']['yaw_drift'] == 0  # old branches keep their reward
    accepted = validate_config({**base, 'task_config': {'reward_weights': {'yaw_drift': 5.}}})
    assert accepted['task_config']['reward_weights']['yaw_drift'] == 5
    for value in (-.1, 101., float('nan'), '5'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'yaw_drift': value}}})
    assert '«прямо»' in G1_LABELS['yaw_drift']
    assert [p['default'] for p in G1_PARAMETERS if p['key'] == 'yaw_drift'] == [0.]
    with_term = {**RECIPE, 'reward_weights': {**RECIPE['reward_weights'], 'yaw_drift': 5.}}
    saved = task_identity('g1_29dof', 'g1_velocity', RECIPE)
    assert task_identity('g1_29dof', 'g1_velocity', with_term) == saved
    validate_resume(saved, task_identity('g1_29dof', 'g1_velocity', with_term))


@pytest.mark.parametrize('branch', ['exp-4e4a49003563481f', 'exp-89d499d1c441484c'])
def test_the_latest_stages_can_be_branched_with_yaw_drift(branch):
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    run = ROOT / 'runs' / 'humanoid_lab' / branch
    stages = list_stages(run) if run.is_dir() else []
    if not stages:
        pytest.skip(f'local run {branch} is not here')
    metadata = read_json(stages[-1] / 'metadata.json')
    assert 'yaw_drift' not in metadata['task_config']['reward_weights']
    weights = {**metadata['task_config']['reward_weights'], 'yaw_drift': 5.}
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'reward_weights': weights})
    validate_resume(checkpoint_identity(metadata), requested)
