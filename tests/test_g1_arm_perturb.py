from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

PROFILES = ({}, dict(real_robot=True, action_limit=10., phase_stop=True, history=5))
# Candidate 2, whose wrist latch iteration 14 targets.
CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-e4da6963dbc7432b/stage_005300944896'
ARMS = list(range(15, 29))
BODY = list(range(15))  # legs and waist


def actions(count, nu, seed=0):
    return np.random.default_rng(seed).uniform(-.4, .4, (count, nu))


def previous_action(env, obs):
    """The newest observed previous action: the last 29 entries of the deploy history, before the phase otherwise."""
    return obs[-env.model.nu:] if env.real_robot else obs[-env.model.nu - 2:-2]


@pytest.mark.parametrize('share', (.1, .5))
def test_segments_start_at_the_onset_that_gives_the_share(share):
    env = G1VelocityEnv(real_robot=True, action_limit=10.)
    scale = torch.from_numpy(env.scale)
    env.close()
    shortest, longest = task.arm_perturb_lengths(.02)
    assert (shortest, longest) == (25, 100)
    assert task.arm_perturb_onset(share, .02) == pytest.approx(share / (share + (1 - share) * 62.5), rel=1e-12)
    generator = torch.Generator().manual_seed(7)
    uniform = lambda k: torch.rand(k, generator=generator, dtype=torch.float64)
    n = 2048
    steps, held = torch.zeros(n, dtype=torch.long), torch.zeros((n, 29), dtype=torch.float64)
    perturbed, lengths, drawn = 0, [], []
    for step in range(3000):
        before, kept = steps.clone(), held.clone()
        steps, held = task.arm_perturb_draw(steps, held, uniform, share, .02, scale, 10.)
        running = before > 0
        assert torch.equal(steps[running], before[running]) and torch.equal(held[running], kept[running])
        started = (before == 0) & (steps > 0)
        lengths.append(steps[started])
        drawn.append(held[started])
        on = steps > 0
        if step >= 1000:  # past the start, which is outside every segment
            perturbed += int(on.sum())
        steps = steps - on.long()  # what the environments do after applying the step
    assert perturbed / (n * 2000) == pytest.approx(share, abs=.006)
    lengths, drawn = torch.cat(lengths), torch.cat(drawn)
    assert int(lengths.min()) == shortest and int(lengths.max()) == longest
    assert float(lengths.double().mean()) == pytest.approx(62.5, abs=1.)
    home = (drawn == 0).all(dim=1)
    assert float(home.double().mean()) == pytest.approx(task.ARM_PERTURB_HOME, abs=.03)
    bound = task.ARM_PERTURB_RADIANS / scale
    random = drawn[~home]
    assert bool((random.abs() <= bound).all()) and bool((random.abs().amax(dim=0) > .95 * bound).all())
    # Zero share: no segment ever starts.
    steps, held = task.arm_perturb_draw(torch.zeros(n, dtype=torch.long), torch.zeros((n, 29), dtype=torch.float64),
                                        uniform, 0., .02, scale, 10.)
    assert int(steps.max()) == 0 and float(held.abs().max()) == 0


def test_random_perturbations_are_clipped_to_the_action_box():
    env = G1VelocityEnv()  # action_limit 1: the wrists' ±0.4 rad (±5.7 at scale 0.07) exceed the box, not the rest
    scale = torch.from_numpy(env.scale)
    env.close()
    rng = np.random.default_rng(3)
    uniform = lambda k: torch.from_numpy(rng.random(k))
    _, held = task.arm_perturb_draw(torch.zeros(20000, dtype=torch.long), torch.zeros((20000, 29), dtype=torch.float64),
                                    uniform, .5, .02, scale, 1.)
    held, wide = held[:, ARMS], scale[ARMS] < task.ARM_PERTURB_RADIANS
    assert int(wide.sum()) == 4  # the wrist pitch and yaw joints
    assert float(held[:, wide].abs().max()) == 1. and float(held[:, ~wide].abs().max()) < 1.


@pytest.mark.parametrize('profile', PROFILES)
def test_zero_share_and_evaluation_resets_keep_the_original_step_bitwise(profile):
    plain, zero, unused = G1VelocityEnv(**profile), G1VelocityEnv(arm_perturb=0., **profile), \
        G1VelocityEnv(arm_perturb=.5, **profile)
    try:
        nu = plain.model.nu
        # A training reset without the setting draws nothing: the same random stream as before it existed.
        np.testing.assert_array_equal(plain.reset(seed=5)[0], zero.reset(seed=5)[0])
        assert not zero.perturb_live
        for action in actions(10, nu, seed=2):
            expected, result = plain.step(action), zero.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1]
            np.testing.assert_array_equal(zero.data.qpos, plain.data.qpos)
        # Evaluation resets (a command given) have no perturbation, even with the setting on.
        command = dict(options={'command': [.6, 0., 0.]})
        for env in (plain, zero, unused):
            env.reset(seed=8, **command)
        assert not unused.perturb_live and unused.perturb_steps == 0
        for action in actions(15, nu, seed=3):
            expected = plain.step(action)
            for env in (zero, unused):
                result = env.step(action)
                np.testing.assert_array_equal(result[0], expected[0])
                assert result[1] == expected[1]
                np.testing.assert_array_equal(env.data.qpos, plain.data.qpos)
                np.testing.assert_array_equal(env.previous_action, plain.previous_action)
        # A training episode switches it off at set_command (scenarios set the command after a reset).
        np.testing.assert_array_equal(plain.reset(seed=11)[0], unused.reset(seed=11)[0])
        assert unused.perturb_live
        for env in (plain, unused):
            env.set_command([.5, 0., 0.])
        assert not unused.perturb_live
        for action in actions(10, nu, seed=4):
            expected, result = plain.step(action), unused.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1]
    finally:
        plain.close(), zero.close(), unused.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_a_segment_replaces_the_arm_actions_in_physics_reward_and_observation(profile):
    """A perturbed robot given the policy's actions moves, observes and is rewarded exactly as an unperturbed one
    given the applied actions (policy legs and waist, held arms), also under a 15-substep delay, whose queue still
    holds the held actions after the segment ends."""
    weights = {'arm_press': .1, 'action_change': .05}
    env = G1VelocityEnv(reward_weights=weights, action_delay=.03, arm_perturb=.1, **profile)
    twin = G1VelocityEnv(reward_weights=weights, action_delay=.03, **profile)
    try:
        nu, limit = env.model.nu, env.action_limit
        options = {'command': [.5, 0., 0.], 'action_delay_steps': 15}
        np.testing.assert_array_equal(env.reset(seed=6, options=options)[0], twin.reset(seed=6, options=options)[0])
        held = np.full(nu, 5.)  # the body entries of the held action are never used
        held[ARMS] = np.random.default_rng(9).uniform(-.8, .8, len(ARMS))
        env.set_arm_perturb(4, held)
        for step, action in enumerate(actions(8, nu, seed=4)):
            applied = np.clip(action, -limit, limit)
            if step < 4:
                applied[ARMS] = held[ARMS]
            obs, reward, _, _, info = env.step(action)
            expected, expected_reward, _, _, expected_info = twin.step(applied)
            np.testing.assert_array_equal(env.data.qpos, twin.data.qpos)
            np.testing.assert_array_equal(env.data.qvel, twin.data.qvel)
            np.testing.assert_array_equal(obs, expected)
            assert reward == expected_reward and info['reward_components'] == expected_info['reward_components']
            np.testing.assert_array_equal(env.previous_action, applied)
            np.testing.assert_array_equal(env.previous_action[BODY], np.clip(action, -limit, limit)[BODY])
            np.testing.assert_array_equal(previous_action(env, obs), applied.astype(np.float32))
            if step < 4:  # the arm targets are the held action's (arm_press compares them with the angles)
                target = env._target(applied)
                press = ((target - env.data.qpos[env.qadr])[ARMS] ** 2).sum()
                assert info['reward_components']['arm_press'] == pytest.approx(-.1 * press, rel=1e-12)
                assert not np.allclose(target[ARMS], env._target(np.clip(action, -limit, limit))[ARMS])
        assert env.perturb_steps == 0
    finally:
        env.close(), twin.close()


def test_training_episodes_spend_the_share_under_perturbation():
    env = G1VelocityEnv(max_episode_steps=10 ** 6, arm_perturb=.5)
    try:
        env.reset(seed=1)
        assert env.perturb_live and env.perturb_steps == 0
        rng = np.random.default_rng(0)
        perturbed, lengths, count = 0, [], 4000
        for _ in range(count):  # the robot falls after ~1.3 s; the arms' segments do not depend on it (no reset here)
            before, kept = env.perturb_steps, env.perturb_action.copy()
            action = rng.uniform(-.05, .05, env.model.nu)
            env.step(action)
            on = not np.array_equal(env.previous_action[ARMS], action[ARMS])
            assert on == (before > 0 or env.perturb_steps > 0)
            np.testing.assert_array_equal(env.previous_action[BODY], action[BODY])
            if on:
                np.testing.assert_array_equal(env.previous_action[ARMS], env.perturb_action[ARMS])
                if before:
                    np.testing.assert_array_equal(env.perturb_action, kept)  # one held action per segment
                else:
                    lengths.append(env.perturb_steps + 1)
            perturbed += on
        assert perturbed / count == pytest.approx(.5, abs=.12)
        assert len(lengths) >= 20 and min(lengths) >= 25 and max(lengths) <= 100
        env.reset(seed=2)  # a reset ends the segment
        assert env.perturb_steps == 0 and not env.perturb_action.any()
    finally:
        env.close()


def test_the_legs_only_g1_has_no_arms_to_perturb():
    env = G1VelocityEnv(legs_only=True, arm_perturb=.5, max_episode_steps=10 ** 6)
    try:
        assert env.arm_joints == () and not env.arm_mask.any()
        env.reset(seed=2)
        segments = 0
        for action in actions(300, env.model.nu, seed=5):
            env.step(action)
            np.testing.assert_array_equal(env.previous_action, action)
            segments += env.perturb_steps > 0
        assert segments > 0  # the segments run, with nothing to hold
    finally:
        env.close()


def test_arm_perturb_config_is_bounded_labelled_and_outside_the_checkpoint_identity():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['arm_perturb'] == 0  # old branches keep theirs
    for value in (.1, .5):
        assert validate_config({**base, 'task_config': {'arm_perturb': value}})['task_config']['arm_perturb'] == value
    for bad in (-.01, .51, float('nan'), True, '0.1'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'arm_perturb': bad}})
    for bad in (-.01, .6, float('nan')):
        with pytest.raises(ValueError):
            G1VelocityEnv(arm_perturb=bad)
    field = next(p for p in G1_PARAMETERS if p['key'] == 'arm_perturb')
    assert field['default'] == 0 and field['min'] == 0 and field['max'] == .5 and 'рук' in field['label']
    config = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.)
    saved = task_identity('g1_29dof', 'g1_velocity', config)
    perturbed = task_identity('g1_29dof', 'g1_velocity', {**config, 'arm_perturb': .1})
    assert perturbed == saved
    validate_resume(saved, perturbed)


def test_candidate_stage_resumes_with_arm_perturb():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import read_json
    if not (CANDIDATE / 'metadata.json').exists():
        pytest.skip('candidate stage is not on this machine')
    metadata = read_json(CANDIDATE / 'metadata.json')
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'arm_perturb': .1})
    validate_resume(checkpoint_identity(metadata), requested)
