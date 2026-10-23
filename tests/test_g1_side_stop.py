from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import SIDE_MIN_SPEED, g1_commands
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

# Commands of exp-902b8958ee504780 (iteration 18), the branch iteration 19 starts from.
COMMANDS = dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                straight_probability=.5)
SIDE = dict(side_share=.2, side_stop=1.)  # the working pair of the docstring
PROFILE = dict(real_robot=True, phase_stop=True, history=5, action_limit=10., heading_hold=True,
               stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55))


def _old_sample_commands(n, uniform, min_speed=0., max_speed=1.5, stand_probability=.2, lateral_speed=0.,
                         yaw_rate=0., straight_probability=0.):
    """sample_commands as it was before side_share / side_stop (iteration 18), kept verbatim."""
    speed, stand = uniform(n), uniform(n)
    command = torch.zeros((n, 3), dtype=speed.dtype, device=speed.device)
    standing = stand < stand_probability
    command[:, 0] = torch.where(standing, 0., min_speed + speed * (max_speed - min_speed))
    if lateral_speed:
        command[:, 1] = torch.where(standing, 0., (2 * uniform(n) - 1) * lateral_speed)
    if yaw_rate:
        command[:, 2] = torch.where(standing, 0., (2 * uniform(n) - 1) * yaw_rate)
        if straight_probability:
            command[:, 2] = torch.where(uniform(n) < straight_probability, 0., command[:, 2])
    return command


def _sample(n, seed, sampler=task.sample_commands, **options):
    generator = torch.Generator().manual_seed(seed)
    return sampler(n, lambda k: torch.rand(k, generator=generator), **{**COMMANDS, **options})


def _old_drop_in(n, uniform, previous=None, side_share=0., side_stop=0., **commands):
    assert not side_share and not side_stop
    return _old_sample_commands(n, uniform, **commands)


def test_without_the_settings_the_draw_is_bit_for_bit_the_old_one():
    assert g1_commands(COMMANDS) == {**COMMANDS, 'side_share': 0., 'side_stop': 0.}  # missing keys: off
    previous = _sample(500, 1, side_share=.2)
    for seed in range(3):
        old = _sample(500, seed, _old_sample_commands)
        assert torch.equal(_sample(500, seed), old)
        assert torch.equal(_sample(500, seed, side_share=0., side_stop=0., previous=previous), old)
    new, old = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)  # the defaults too
    assert torch.equal(task.sample_commands(200, lambda k: torch.rand(k, generator=new)),
                       _old_sample_commands(200, lambda k: torch.rand(k, generator=old)))


def test_without_the_settings_the_cpu_task_resamples_its_old_commands(monkeypatch):
    """Training episode with a resample at every step: commands, held headings and observations as before."""
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)

    def run():
        env = G1VelocityEnv(commands=COMMANDS, random_phase=True, **PROFILE)
        try:
            observations, commands, headings = [env.reset(seed=4)[0]], [env.command.copy()], [env.heading_target]
            for _ in range(5):
                observations.append(env.step(np.zeros(29))[0])
                commands.append(env.command.copy())
                headings.append(env.heading_target)
            return np.array(observations), np.array(commands), np.array(headings)
        finally:
            env.close()

    new = run()
    assert len({tuple(c) for c in new[1]}) == 6  # a fresh command at every step
    monkeypatch.setattr(task, 'sample_commands', _old_drop_in)
    for now, before in zip(new, run()):
        np.testing.assert_array_equal(now, before)


def test_side_commands_come_from_the_backward_ones_and_are_real_sideways_steps():
    old = _sample(20000, 3)
    new = _sample(20000, 3, side_share=.2)
    side = task.pure_sideways(new)
    walking = old.abs().sum(dim=1) > 0
    assert torch.equal(walking, new.abs().sum(dim=1) > 0)  # the stand share is untouched
    assert abs(float(side.sum()) / float(walking.sum()) - .2) < .01
    assert not task.pure_sideways(old).any() and bool((old[side, 0] < 0).all())  # taken from backward commands
    assert torch.equal(new[~side], old[~side])  # forward, the rest of backward and stand: the old draws
    assert bool((new[side, 0] == 0).all()) and bool((new[side, 2] == 0).all())
    speed = new[side, 1].abs()
    assert float(speed.min()) >= SIDE_MIN_SPEED - 1e-6 and float(speed.max()) <= .5 + 1e-6
    assert abs(float(speed.mean()) - .3) < .01  # uniform over [0.1, 0.5]
    assert torch.equal(torch.sign(new[side, 1]), torch.sign(old[side, 1]))  # both directions, as drawn
    backward = new[:, 0] < 0
    assert float(new[backward, 0].min()) < -.49  # the backward ones keep their whole speed range
    assert SIDE_MIN_SPEED == task.STAND_SPEED  # each one asks for motion (gait, phase): never a stand command
    everything = _sample(20000, 3, side_share=1 / 3)  # the whole backward share
    assert not bool((everything[:, 0] < 0).any())


def test_side_stop_turns_the_next_non_forward_command_into_stand():
    sideways = torch.tensor([[0., .3, 0.]]).expand(5000, -1)
    plain = _sample(5000, 7, side_share=.2)
    stopped = _sample(5000, 7, side_share=.2, side_stop=1., previous=sideways)
    forward = plain[:, 0] > 0
    assert torch.equal(stopped[forward], plain[forward])  # forward draws are left alone
    assert float(stopped[~forward].abs().max()) == 0  # everything else: stand
    half = _sample(5000, 7, side_share=.2, side_stop=.5, previous=sideways)
    share = float((half[~forward].abs().sum(dim=1) == 0).float().mean())
    standing = float((plain[~forward].abs().sum(dim=1) == 0).float().mean())
    assert abs(share - (standing + .5 * (1 - standing))) < .03
    walking = torch.tensor([[.5, 0., 0.], [0., 0., 0.], [-.3, .2, 0.], [0., .3, .4]]).repeat(1250, 1)
    assert torch.equal(_sample(5000, 7, side_share=.2, side_stop=1., previous=walking), plain)  # only after sideways
    assert torch.equal(_sample(5000, 7, side_share=.2, side_stop=1.), plain)  # a reset has no previous command


def _chain(commands, worlds=4096, episodes=2, steps=1000, seed=0):
    """Commands as the CUDA training env draws them (per-step resampling, a fresh draw at each episode start):
    shares of the command changes and of the time spent under each kind of command."""
    generator = torch.Generator().manual_seed(seed)
    uniform = lambda n: torch.rand(n, generator=generator)
    probability = .02 / task.COMMAND_RESAMPLE_SECONDS
    changes, side_stand, walk_stand = 0, 0, 0
    time = torch.zeros(4)
    for _ in range(episodes):
        command = task.sample_commands(worlds, uniform, **commands)
        for _ in range(steps):
            stand = command.abs().sum(dim=1) == 0
            side = task.pure_sideways(command)
            time += torch.stack((command[:, 0] > 0, command[:, 0] < 0, side, stand)).float().mean(dim=1)
            resample = uniform(worlds) < probability
            fresh = task.sample_commands(worlds, uniform, previous=command, **commands)
            stops = resample & (fresh.abs().sum(dim=1) == 0)
            changes += int(resample.sum())
            side_stand += int((side & stops).sum())
            walk_stand += int((~stand & stops).sum())
            command = torch.where(resample[:, None], fresh, command)
    forward, backward, sideways, standing = (time / (episodes * steps)).tolist()
    return dict(changes=changes, side_stand=side_stand / changes, walk_stand=walk_stand / changes, forward=forward,
                backward=backward, side=sideways, stand=standing)


def test_long_run_shares_of_sideways_then_stand():
    """The docstring's numbers (4096 worlds × 10 episodes: 6.4% / 1.8% / 0) on a run of 3·10^4 changes."""
    on = _chain(g1_commands({**COMMANDS, **SIDE}))
    assert on['changes'] > 30000
    assert .058 < on['side_stand'] < .070  # the goal: at least ~5% of the command changes
    assert abs(on['forward'] - .6) < .01 and abs(on['side'] - .16) < .01
    assert abs(on['backward'] - .11) < .01 and abs(on['stand'] - .14) < .01
    assert .12 < on['walk_stand'] < .15
    share = _chain(g1_commands({**COMMANDS, 'side_share': .2}))
    assert .014 < share['side_stand'] < .022  # drawn independently, the stop rarely follows a sideways step
    assert abs(share['forward'] - .6) < .01 and abs(share['side'] - .18) < .01
    before = _chain(g1_commands(COMMANDS))
    assert before['side_stand'] == 0 and abs(before['forward'] - .6) < .01 and abs(before['backward'] - .3) < .01
    assert .085 < before['walk_stand'] < .1


def test_cpu_task_alternates_sideways_and_stand_and_holds_the_heading_of_a_side_step(monkeypatch):
    """Every walking command sideways (the speed range all backward) and side_stop 1: a resample at every step
    goes sideways, stand, sideways, …; a sideways step keeps the heading the robot has when it starts."""
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)
    commands = {**COMMANDS, 'max_speed': 0., 'stand_probability': 0., 'side_share': 1., 'side_stop': 1.}
    env = G1VelocityEnv(commands=commands, **PROFILE)
    try:
        env.reset(seed=2)
        assert abs(env.heading_target) < .01
        env.data.qpos[3:7] = [np.cos(.25), 0., 0., np.sin(.25)]  # turned 0.5 rad off the episode's heading
        mujoco.mj_forward(env.model, env.data)
        kinds = [bool(task.pure_sideways(torch.from_numpy(env.command)[None])[0])]
        for _ in range(6):
            env.step(np.zeros(29))
            if kinds[-1]:
                assert not env.command.any()
            kinds.append(bool(task.pure_sideways(torch.from_numpy(env.command)[None])[0]))
            if kinds[-1]:
                assert .1 <= abs(env.command[1]) <= .5 and env.command[0] == 0 and env.command[2] == 0
                rotation = env.data.xmat[env.pelvis].reshape(3, 3)  # sideways from where the robot faces now
                assert env.heading_target == pytest.approx(np.arctan2(rotation[1, 0], rotation[0, 0]))
                assert env.heading_target > .4
        assert kinds == [True, False, True, False, True, False, True]
        env.reset(seed=2, options={'command': [0., .3, 0.]})  # a set command is never resampled
        env.step(np.zeros(29))
        np.testing.assert_array_equal(env.command, [0., .3, 0.])
    finally:
        env.close()


def test_settings_are_bounded_and_outside_the_checkpoint_identity():
    for commands in ({**COMMANDS, 'side_share': .34},  # beyond the backward share 1/3
                     {**COMMANDS, 'min_speed': 0., 'side_share': .1},  # nothing backward to take from
                     {**COMMANDS, 'lateral_speed': .1, 'side_share': .1},  # no sideways step above STAND_SPEED
                     {**COMMANDS, 'side_stop': .5},  # nothing to stop after
                     {**COMMANDS, 'side_share': -.1}, {**COMMANDS, 'side_share': .2, 'side_stop': 1.5}):
        with pytest.raises(ValueError):
            g1_commands(commands)
    assert g1_commands({**COMMANDS, 'side_share': 1 / 3, 'side_stop': 1})['side_share'] == pytest.approx(1 / 3)
    config = {**PROFILE, 'commands': COMMANDS}
    saved = task_identity('g1_29dof', 'g1_velocity', config)
    with_side = {**config, 'commands': {**COMMANDS, **SIDE}}
    assert task_identity('g1_29dof', 'g1_velocity', with_side) == saved
    validate_resume(saved, task_identity('g1_29dof', 'g1_velocity', with_side))


def test_last_iteration18_stage_resumes_with_side_commands():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    run = Path(__file__).resolve().parents[1] / 'runs' / 'humanoid_lab' / 'exp-902b8958ee504780'
    if not run.exists() or not list_stages(run):
        pytest.skip('iteration 18 run is not on this machine')
    metadata = read_json(list_stages(run)[-1] / 'metadata.json')
    config = metadata['task_config']
    requested = task_identity('g1_29dof', 'g1_velocity', {**config, 'commands': {**config['commands'], **SIDE}})
    validate_resume(checkpoint_identity(metadata), requested)
