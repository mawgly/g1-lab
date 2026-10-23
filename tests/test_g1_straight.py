import math

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

COMMANDS = dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.)


def _sample(n, seed, **options):
    generator = torch.Generator().manual_seed(seed)
    return task.sample_commands(n, lambda k: torch.rand(k, generator=generator), **{**COMMANDS, **options})


def test_straight_share_of_walking_commands_and_unchanged_stream_without_it():
    commands = _sample(20000, 3, straight_probability=.3)
    standing = commands[:, :2].abs().sum(dim=1) == 0
    straight = task.straight_walking(commands)
    assert abs(float(standing.float().mean()) - .1) < .02
    assert abs(float(straight.sum()) / float((~standing).sum()) - .3) < .02
    assert not straight[standing].any()  # standing is not «walking straight»
    torch.testing.assert_close(_sample(50, 7, straight_probability=0.), _sample(50, 7))  # old runs keep their stream


def turned(env, yaw):
    env.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)


def test_straight_command_holds_the_heading_it_starts_at():
    commands = {**COMMANDS, 'stand_probability': 0., 'straight_probability': 1.}
    env = G1VelocityEnv(real_robot=True, action_limit=10., heading_hold=True, commands=commands)
    try:
        env.reset(seed=1)
        assert env.heading_target == 0. and env.command[2] == 0.
        turned(env, .5)
        env._sample_command()  # a fresh straight command: keep walking where the robot faces now
        assert abs(env.heading_target - .5) < 1e-9
        env._observation()
        assert abs(env.observed_command[2]) < 1e-6  # no steering back to the episode's start
    finally:
        env.close()
    env = G1VelocityEnv(real_robot=True, action_limit=10., heading_hold=True,
                        commands={**COMMANDS, 'stand_probability': 1.})
    try:  # without the share nothing changes: standing still steers back to the episode's heading
        env.reset(seed=1)
        turned(env, .5)
        env._sample_command()
        env._observation()
        assert env.heading_target == 0. and env.observed_command[2] < -.2
    finally:
        env.close()


def test_stand_command_stays_zero_without_stand_heading_hold():
    """Isaac Lab zeroes standing envs after heading control; a turned robot told to stand is still standing."""
    still = []
    for stand_heading_hold in (True, False):
        env = G1VelocityEnv(real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=stand_heading_hold,
                            commands={**COMMANDS, 'stand_probability': 1.}, reward_weights={'still': 1.})
        try:
            env.reset(seed=1)
            turned(env, .5)
            env._observation()
            steering = float(env.observed_command[2])
            info = env.step(np.zeros(29))[4]
            still.append(info['reward_components']['still'])
        finally:
            env.close()
        assert (steering < -.2) if stand_heading_hold else steering == 0.
    assert still[0] == 0. and still[1] < 0.  # turned 0.5 rad: «turn in place» switched the standing terms off


def _old_effective_command(command, rotation, heading_target, standing=True):
    """effective_command as it was before side_heading_hold (iteration 25), for the bit-for-bit check."""
    error = torch.remainder(heading_target - task.heading(rotation) + math.pi, 2 * math.pi) - math.pi
    hold = torch.clamp(task.HEADING_GAIN * error, -task.MAX_YAW_COMMAND, task.MAX_YAW_COMMAND)
    steer = command[:, 2] == 0
    if not standing:
        steer = steer & (command[:, :2].abs().sum(dim=1) > 0)
    result = command.clone()
    result[:, 2] = torch.where(steer, hold.to(command.dtype), command[:, 2])
    return result


# Side steps both ways (the slowest one too), forward, diagonal, backward, stand, a side step with a turn.
SIDE_CASES = [[0., 1., 0.], [0., -1., 0.], [0., .1, 0.], [1., 0., 0.], [.5, .5, 0.], [-.5, 0., 0.], [0., 0., 0.],
              [0., 1., .3]]


def _yawed(yaws):
    yaws = torch.as_tensor(yaws, dtype=torch.float64)
    rotation = torch.zeros((len(yaws), 3, 3), dtype=torch.float64)
    rotation[:, 0, 0], rotation[:, 0, 1] = torch.cos(yaws), -torch.sin(yaws)
    rotation[:, 1, 0], rotation[:, 1, 1] = torch.sin(yaws), torch.cos(yaws)
    rotation[:, 2, 2] = 1.
    return rotation


def test_side_command_turn_stays_zero_without_side_heading_hold():
    """Iteration 26: heading hold no longer steers a purely sideways command, so the reward sees the whole drift."""
    command = torch.tensor(SIDE_CASES, dtype=torch.float64).repeat(2, 1)
    yaws = [.5] * len(SIDE_CASES) + [-2.5] * len(SIDE_CASES)  # steering −0.25, and clamped to +1 past the wrap
    rotation, target = _yawed(yaws), torch.zeros(len(yaws), dtype=torch.float64)
    side = task.pure_sideways(command)
    assert side.tolist() == [True, True, True, False, False, False, False, False] * 2
    for standing in (True, False):  # missing and True: bit for bit the old command
        old = _old_effective_command(command, rotation, target, standing)
        assert torch.equal(task.effective_command(command, rotation, target, standing), old)
        assert torch.equal(task.effective_command(command, rotation, target, standing, True), old)
        assert bool((old[side, 2].abs() > .2).all())  # the old hold steered the side steps
        new = task.effective_command(command, rotation, target, standing, False)
        assert bool((new[side, 2] == 0).all())  # exactly zero, however far the heading went
        assert torch.equal(new[~side], old[~side])  # forward, diagonal, backward, stand, a set turn: as before
        assert torch.equal(new[:, :2], command[:, :2])


def test_cpu_task_passes_side_heading_hold_as_the_shared_function():
    """The CPU task's observed and rewarded command is effective_command on CPU tensors with its own setting;
    missing and True give the same observations and rewards bit for bit."""
    results = {}
    for setting in (None, True, False):
        options = {} if setting is None else {'side_heading_hold': setting}
        env = G1VelocityEnv(real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=False,
                            commands={**COMMANDS, 'stand_probability': 0.}, reward_weights={'yaw_drift': 1.},
                            **options)
        runs = []
        try:
            for command in SIDE_CASES:
                env.reset(seed=1, options={'command': command})
                turned(env, .5)
                observation = env._observation()
                rotation = torch.from_numpy(env.data.xmat[env.pelvis].reshape(1, 3, 3).copy())
                expected = task.effective_command(torch.tensor([command], dtype=torch.float64), rotation,
                                                  torch.tensor([env.heading_target], dtype=torch.float64), False,
                                                  setting is not False)
                assert np.array_equal(env.observed_command, expected[0].numpy())
                observed, reward, _, _, info = env.step(np.zeros(29))
                assert np.array_equal(info['command'], expected[0].numpy())  # rewarded for what it observed
                velocity = torch.from_numpy(env._velocity())[None]
                drift = task.yaw_drift_update(torch.zeros(1, dtype=torch.float64), velocity, expected, env.dt)
                assert torch.equal(env.yaw_drift_error, drift)  # ω_z − c with c = 0 on a side step without the hold
                runs.append((observation, observed, reward, info['reward_components']))
        finally:
            env.close()
        results[setting] = runs
    for missing, held in zip(results[None], results[True]):
        assert all(np.array_equal(a, b) for a, b in zip(missing[:2], held[:2]))
        assert missing[2] == held[2] and missing[3] == held[3]
    for k, command in enumerate(SIDE_CASES):  # without the hold only the side steps' observation and reward change
        same = not bool(task.pure_sideways(torch.tensor([command]))[0])
        assert np.array_equal(results[True][k][0], results[False][k][0]) == same
        assert (results[True][k][3]['yaw'] == results[False][k][3]['yaw']) == same


def test_side_heading_hold_setting_keeps_old_configs_and_the_checkpoint_identity():
    from humanoid_sim.tasks.config import g1_config
    from humanoid_sim.tasks.identity import task_identity
    assert g1_config({})['side_heading_hold'] is True  # saved configs without it: steered as before
    assert g1_config({'side_heading_hold': False})['side_heading_hold'] is False
    for value in (0, 1, 'false', None):
        with pytest.raises(ValueError):
            g1_config({'side_heading_hold': value})
    profile = dict(real_robot=True, action_limit=10., heading_hold=True, history=5, phase_stop=True,
                   critic_extras=True)
    assert (task_identity('g1_29dof', 'g1_velocity', {**profile, 'side_heading_hold': False})
            == task_identity('g1_29dof', 'g1_velocity', profile))  # commands and reward only
