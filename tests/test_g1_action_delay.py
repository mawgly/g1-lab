from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

PROFILES = ({}, dict(real_robot=True, action_limit=10., phase_stop=True, history=5))
CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-facb20de4bfe472c/stage_004975362048'


def actions(count, nu, seed=0):
    return np.random.default_rng(seed).uniform(-.4, .4, (count, nu))


def manual_physics(raw, targets):
    """The task's PD (and motor model) by hand, one target per physics substep."""
    for target in targets:
        joint_velocity = raw.data.qvel[raw.vadr]
        torque = raw.kp*(target-raw.data.qpos[raw.qadr])-raw.kd*joint_velocity
        if raw.motors is not None:
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                       *raw._motor_tensors).numpy()
        raw.data.ctrl[:] = np.clip(torque, raw.model.actuator_ctrlrange[:, 0], raw.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(raw.model, raw.data)
    mujoco.mj_forward(raw.model, raw.data)


def target_of(raw, action):
    action = np.clip(action, -raw.action_limit, raw.action_limit)
    return np.clip(raw.home + raw.scale*action, raw.model.jnt_range[raw.joints, 0], raw.model.jnt_range[raw.joints, 1])


@pytest.mark.parametrize('profile', PROFILES)
def test_zero_delay_keeps_the_original_step_bitwise(profile):
    plain, zero, unused = G1VelocityEnv(**profile), G1VelocityEnv(action_delay=0., **profile), \
        G1VelocityEnv(action_delay=.02, **profile)
    try:
        # A training reset without delay draws nothing new: the same random stream as before the setting.
        np.testing.assert_array_equal(plain.reset(seed=5)[0], zero.reset(seed=5)[0])
        assert zero.delay_steps == 0 and not zero.delay_lags.any() and zero.delay_actions == []
        command = dict(options={'command': [.6, 0., 0.]})
        plain.reset(seed=8, **command), zero.reset(seed=8, **command), unused.reset(seed=8, **command)
        assert unused.delay_steps == 0  # evaluation resets (a command given) have no delay
        reference = G1VelocityEnv(**profile)  # the original step's physics, by hand
        reference.reset(seed=8, **command)
        for action in actions(15, plain.model.nu):
            expected = plain.step(action)
            for env in (zero, unused):
                result = env.step(action)
                np.testing.assert_array_equal(result[0], expected[0])
                assert result[1] == expected[1]
                np.testing.assert_array_equal(env.data.qpos, plain.data.qpos)
            manual_physics(reference, [target_of(reference, action)] * reference.frame_skip)
            np.testing.assert_array_equal(reference.data.qpos, plain.data.qpos)
            np.testing.assert_array_equal(reference.data.qvel, plain.data.qvel)
        reference.close()
    finally:
        plain.close(), zero.close(), unused.close()


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('delay', (3, 10, 15))
def test_targets_reach_the_motors_delay_substeps_late(profile, delay):
    env, twin = G1VelocityEnv(**profile), G1VelocityEnv(**profile)
    try:
        options = {'command': [.6, 0., 0.], 'action_delay_steps': delay}
        env.reset(seed=9, options=options)
        twin.reset(seed=9, options={'command': [.6, 0., 0.]})
        assert env.delay_steps == delay
        n = env.frame_skip
        # Every substep's target in time order, as scripts/lab/probes/robust_probe.py: the home pose before the start.
        substeps = [target_of(twin, np.zeros(twin.model.nu))] * delay
        for action in actions(12, env.model.nu, seed=delay):
            observation = env.step(action)[0]
            substeps += [target_of(twin, action)] * n
            manual_physics(twin, substeps[-n - delay:len(substeps) - delay])
            np.testing.assert_array_equal(env.data.qpos, twin.data.qpos)
            np.testing.assert_array_equal(env.data.qvel, twin.data.qvel)
            np.testing.assert_array_equal(env.previous_action, np.clip(action, -env.action_limit, env.action_limit))
            assert np.isfinite(observation).all()
    finally:
        env.close(), twin.close()


def test_a_full_step_of_delay_shifts_the_actions_by_one_step():
    delayed, plain = G1VelocityEnv(), G1VelocityEnv()
    try:
        delayed.reset(seed=4, options={'command': [.5, 0., 0.], 'action_delay_steps': delayed.frame_skip})
        plain.reset(seed=4, options={'command': [.5, 0., 0.]})
        assert delayed.delay_lags.tolist() == [1] * delayed.frame_skip
        sequence = actions(8, plain.model.nu, seed=3)
        plain.step(np.zeros(plain.model.nu))
        for action in sequence:
            delayed.step(action)
            np.testing.assert_array_equal(delayed.data.qpos, plain.data.qpos)  # moved by the previous action
            plain.step(action)
        assert np.abs(delayed.data.qpos - plain.data.qpos).max() > 1e-4
    finally:
        delayed.close(), plain.close()


def test_training_resets_draw_the_delay_within_bounds():
    env = G1VelocityEnv(real_robot=True, action_limit=10., phase_stop=True, history=5, random_phase=True,
                        gait=dict(period=.6), action_delay=.02)
    try:
        assert env.max_delay_steps == 10 and env.frame_skip == 10
        drawn = set()
        for seed in range(60):
            env.reset(seed=seed)
            assert 0 <= env.delay_steps <= 10
            assert env.delay_lags.tolist() == [1] * env.delay_steps + [0] * (10 - env.delay_steps)
            assert len(env.delay_actions) == (env.delay_steps > 0)
            drawn.add(env.delay_steps)
        assert len(drawn) > 7
        env.reset(seed=3, options={'command': [.5, 0., 0.]})
        assert env.delay_steps == 0 and not env.delay_lags.any()
        env.reset(seed=3, options={'command': [.5, 0., 0.], 'action_delay_steps': 7})
        assert env.delay_steps == 7 and env.delay_lags.tolist() == [1] * 7 + [0] * 3
        env.reset(seed=3, options={'action_delay_steps': 25})  # beyond two control steps: two older actions
        assert env.delay_lags.tolist() == [3] * 5 + [2] * 5 and len(env.delay_actions) == 3
        for bad in (-1, 1.5, True):
            with pytest.raises(ValueError):
                env.reset(seed=3, options={'action_delay_steps': bad})
    finally:
        env.close()
    for bad in (-.01, .06, float('nan')):
        with pytest.raises(ValueError):
            G1VelocityEnv(action_delay=bad)


def test_action_delay_config_is_bounded_and_outside_the_checkpoint_identity():
    from humanoid_sim.lab import validate_config
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['action_delay'] == 0
    assert validate_config({**base, 'task_config': {'action_delay': .02}})['task_config']['action_delay'] == .02
    for bad in (-.01, .051, float('nan'), True, '0.02'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'action_delay': bad}})
    config = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.)
    saved = task_identity('g1_29dof', 'g1_velocity', config)
    delayed = task_identity('g1_29dof', 'g1_velocity', {**config, 'action_delay': .02})
    assert delayed == saved
    validate_resume(saved, delayed)


def test_candidate_stage_resumes_with_action_delay():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import read_json
    if not (CANDIDATE / 'metadata.json').exists():
        pytest.skip('candidate stage is not on this machine')
    metadata = read_json(CANDIDATE / 'metadata.json')
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'action_delay': .02})
    validate_resume(checkpoint_identity(metadata), requested)
