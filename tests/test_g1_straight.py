import math

import mujoco
import numpy as np
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
