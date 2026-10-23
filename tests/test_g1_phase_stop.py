import math

import numpy as np
import pytest

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

REAL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=True)
PHASE = slice(9, 11)  # sin, cos in a deploy frame (deploy_terms: 3 + 3 + 3, then 2)


def test_phase_reads_zero_while_told_to_stand_as_on_the_robot():
    env = G1VelocityEnv(**REAL, phase_stop=True)
    old = G1VelocityEnv(**REAL)
    try:
        for command, moving in (([0., 0., 0.], False), ([.05, .05, .05], False), ([.5, 0., 0.], True), ([0., 0., .2], True)):
            obs, _ = env.reset(seed=1, options={'command': command})
            assert (abs(obs[PHASE]).sum() > 0) == moving, command
            obs, _ = old.reset(seed=1, options={'command': command})
            assert abs(obs[PHASE]).sum() > 0  # without phase_stop the phase always ticks
    finally:
        env.close()
        old.close()


def test_reset_can_start_the_gait_clock_later_without_shortening_the_episode():
    clock = lambda env, steps: (lambda a: [math.sin(a), math.cos(a)])(
        2 * math.pi * (steps * env.dt / env.gait['period'] % 1.))
    env = G1VelocityEnv(max_episode_steps=4, gait={'period': .6, 'stance_fraction': .55})
    try:
        half = round(env.gait['period'] / env.dt / 2)
        assert half == 15
        obs, _ = env.reset(seed=1, options={'command': [.5, 0., 0.], 'phase_steps': half})
        np.testing.assert_allclose(obs[-2:], clock(env, half), atol=1e-6)  # the first observation is shifted
        ends = [env.step(np.zeros(29))[3] for _ in range(4)]
        assert ends == [False, False, False, True]  # four steps, as without the shift
        np.testing.assert_allclose(env._observation()[-2:], clock(env, half + 4), atol=1e-6)
        obs, _ = env.reset(seed=1, options={'command': [.5, 0., 0.]})  # back to the default
        np.testing.assert_allclose(obs[-2:], clock(env, 0), atol=1e-6)
        for bad in (-1, 1.5, True, '3'):
            with pytest.raises(ValueError):
                env.reset(seed=1, options={'phase_steps': bad})
    finally:
        env.close()
    env = G1VelocityEnv(**REAL, history=3, phase_stop=True)
    try:  # every history frame filled at reset carries the shifted phase
        obs, _ = env.reset(seed=1, options={'command': [.5, 0., 0.], 'phase_steps': 7})
        index = np.array(task.history_index(env.model.nu, 3))[:, PHASE]
        np.testing.assert_allclose(obs[index], [clock(env, 7)] * 3, atol=1e-6)
    finally:
        env.close()


def test_phase_stop_is_its_own_contract_and_needs_the_real_robot():
    plain = task_identity('g1_29dof', 'g1_velocity', {**REAL, 'history': 5})
    stop = task_identity('g1_29dof', 'g1_velocity', {**REAL, 'history': 5, 'phase_stop': True})
    assert stop['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-critic4'
    with pytest.raises(ValueError):
        validate_resume(plain, stop)
    with pytest.raises(ValueError):
        G1VelocityEnv(phase_stop=True)
    assert task.PHASE_STOP_NORM == .1  # unitree_rl_mjlab gait_phase: cmd_norm < 0.1
