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

PROFILES = ({}, dict(real_robot=True, action_limit=10., phase_stop=True, history=5,
                     gait={'period': .6, 'stance_fraction': .55}))
# Candidate 4 (knee_floor), the parent line of iteration 20.
CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-902b8958ee504780/stage_005631836160'
MASS = 35.112142  # the G1's total mass, as the sim2sim push patch prints it


def actions(count, nu, seed=0, size=.4):
    return np.random.default_rng(seed).uniform(-size, size, (count, nu))


def record_pelvis_forces(monkeypatch, env):
    """The pelvis xfrc_applied (force xyz, torque xyz) as each mj_step of env sees it, appended to the returned list,
    plus whether any other body was ever pushed."""
    seen, original = [], mujoco.mj_step

    def step(model, data):
        if data is env.data:
            others = np.delete(data.xfrc_applied, env.pelvis, axis=0)
            seen.append((data.xfrc_applied[env.pelvis].copy(), bool(others.any())))
        original(model, data)

    monkeypatch.setattr(mujoco, 'mj_step', step)
    return seen


def test_draws_space_the_pushes_and_spread_their_size_direction_and_length():
    """The shared push process on 2048 worlds for 30 s: starts 100-200 control steps apart (2-4 s, start to start,
    the first one that long after a reset), lengths 25-100 substeps (0.05-0.2 s), Δv = |F|·T/M uniform in [0, 1.25],
    the direction uniform relative to the heading, and force_push_substeps spreading a push over its substeps."""
    generator = torch.Generator().manual_seed(7)
    uniform = lambda k: torch.rand(k, generator=generator, dtype=torch.float64)
    n, dt, substep, frame_skip = 2048, .02, .002, 10
    assert task.force_push_lengths(substep) == (25, 100)
    timer = task.force_push_interval(uniform, n, dt)  # as a reset draws it
    assert int(timer.min()) == 100 and int(timer.max()) == 200
    first = timer.clone()
    heading = (2 * uniform(n) - 1) * math.pi
    left, force = torch.zeros(n, dtype=torch.long), torch.zeros((n, 3), dtype=torch.float64)
    last = torch.full((n,), -1, dtype=torch.long)
    gaps, lengths, dvs, angles = [], [], [], []
    for step in range(1500):
        before, running, kept = timer.clone(), left.clone(), force.clone()
        timer, left, force = task.force_push_draw(timer, left, force, heading, uniform, 1.25, MASS, dt, substep)
        started = before == 0
        assert torch.equal(timer[~started], before[~started] - 1)  # every timer counts the step off
        assert torch.equal(left[~started], running[~started]) and torch.equal(force[~started], kept[~started])
        if started.any():
            fresh = last[started] < 0
            assert bool((step == first[started][fresh]).all())  # the first push: the reset's interval on
            gaps.append(step - last[started][~fresh])
            last[started] = step
            assert bool((timer[started] >= 99).all() and (timer[started] <= 199).all())
            length, push = left[started], force[started]
            assert bool((push[:, 2] == 0).all())  # horizontal
            lengths.append(length)
            dvs.append(push[:, :2].norm(dim=1) * length * substep / MASS)
            angles.append(torch.atan2(push[:, 1], push[:, 0]) - heading[started])
        # The force at each substep: the push's while it lasts, then 0; the substeps left go down by a step.
        pushes, after = task.force_push_substeps(force, left, frame_skip)
        on = left[:, None] > torch.arange(frame_skip)
        assert torch.equal(pushes, torch.where(on[..., None], force[:, None], torch.zeros((), dtype=torch.float64)))
        assert torch.equal(after, (left - frame_skip).clamp(min=0))
        left = after
    gaps, lengths, dvs, angles = (torch.cat(x) for x in (gaps, lengths, dvs, angles))
    assert len(gaps) > 15000
    assert int(gaps.min()) == 100 and int(gaps.max()) == 200 and float(gaps.double().mean()) == pytest.approx(150, abs=1)
    assert int(lengths.min()) == 25 and int(lengths.max()) == 100
    assert float(lengths.double().mean()) == pytest.approx(62.5, abs=.6)
    assert float(dvs.min()) >= 0 and float(dvs.max()) <= 1.25 + 1e-12 and float(dvs.max()) > 1.24
    assert float(dvs.mean()) == pytest.approx(.625, abs=.01)
    assert float(torch.cos(angles).mean()) == pytest.approx(0, abs=.02)
    assert float(torch.sin(angles).mean()) == pytest.approx(0, abs=.02)
    quadrant = torch.remainder(angles, 2 * math.pi) // (math.pi / 2)
    assert [float((quadrant == k).double().mean()) for k in range(4)] == pytest.approx([.25] * 4, abs=.015)


def test_a_push_force_is_m_dv_over_t_turned_by_the_heading():
    heading = torch.tensor([0., math.pi / 2, .3], dtype=torch.float64)
    velocity = torch.tensor([[1., 0.], [1., 0.], [-1.25, .5]], dtype=torch.float64)
    force = task.force_push_force(velocity, heading, MASS, torch.tensor([50, 50, 25]), .002)
    np.testing.assert_allclose(force[0].numpy(), [MASS / .1, 0, 0], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(force[1].numpy(), [0, MASS / .1, 0], rtol=1e-12, atol=1e-9)
    world = np.array([-1.25 * math.cos(.3) - .5 * math.sin(.3), -1.25 * math.sin(.3) + .5 * math.cos(.3), 0])
    np.testing.assert_allclose(force[2].numpy(), MASS / .05 * world, rtol=1e-12)


def test_a_long_training_run_pushes_the_pelvis_every_2_to_4_s_for_0_05_to_0_2_s(monkeypatch):
    """A 60 s training episode at the working 1.25 m/s, read from what mj_step gets: the pelvis alone is pushed, by
    a horizontal force constant over each push, pushes start with a control step 2-4 s apart (the first one the
    reset's interval on), last 0.05-0.2 s and carry an impulse of at most M·1.25; between steps xfrc_applied is 0.
    The robot falls after about a second and lies on; the pushes do not depend on it."""
    env = G1VelocityEnv(max_episode_steps=10 ** 6, force_push=1.25)
    try:
        seen = record_pelvis_forces(monkeypatch, env)
        env.reset(seed=1)
        assert env.push_live and 100 <= env.push_timer <= 200 and env.push_left == 0
        assert env.mass == pytest.approx(MASS, abs=1e-6)
        first = env.push_timer
        for action in actions(3000, env.model.nu, seed=0, size=.05):
            env.step(action)
            assert not env.data.xfrc_applied.any()
        wrench = np.array([w for w, _ in seen])
        assert len(wrench) == 30000 and not any(other for _, other in seen)
        assert not wrench[:, 2:].any()  # no vertical force, no torque
        on = np.abs(wrench[:, :2]).sum(axis=1) > 0
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        starts, ends = edges[::2], edges[1::2]
        assert len(starts) >= 14 and starts[0] == 10 * first and not (starts % 10).any()  # 14: all gaps 4 s
        gaps = np.diff(starts) // 10
        assert gaps.min() >= 100 and gaps.max() <= 200
        lengths = ends - starts
        assert lengths.min() >= 25 and lengths.max() <= 100 and len(set(lengths.tolist())) > 5
        for start, end in zip(starts, ends):
            np.testing.assert_array_equal(wrench[start:end], np.repeat(wrench[start:start + 1], end - start, axis=0))
            assert np.linalg.norm(wrench[start, :2]) * (end - start) * .002 <= MASS * 1.25 * (1 + 1e-12)
        env.reset(seed=2)  # a reset ends a running push and draws the next one
        assert env.push_left == 0 and not env.data.xfrc_applied.any() and 100 <= env.push_timer <= 200
    finally:
        env.close()


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('direction', ((-1., 0.), (0., 1.)))
def test_a_set_push_is_the_bridge_patch_force_and_speeds_the_standing_pelvis_up_by_as_much(profile, direction,
                                                                                         monkeypatch):
    """set_force_push((Δv forward, left), 0.1) on a robot standing still (command 0, zero action) is the push of
    scripts/sim2sim/patches/unitree_mujoco-push.patch, as scripts/lab/probes/bridge_kick.py copies it: M·Δv/0.1 s
    along the pelvis heading, from xquat, written before each of 50 mj_steps, then 0. A twin pushed that way moves
    identically, and at the end of the push the pelvis runs 1.4-2 m/s faster along it than an unpushed twin (Δv 1;
    bridge_kick on candidate 4 in our physics: peak 1.41-1.63 m/s; set_force_push on it standing: 1.56-1.61)."""
    pushed, patch, still = (G1VelocityEnv(force_push=1.25, **profile), G1VelocityEnv(**profile),
                            G1VelocityEnv(**profile))
    try:
        forces = record_pelvis_forces(monkeypatch, pushed)
        recording = mujoco.mj_step
        state = dict(left=0, force=np.zeros(3))

        def bridge(model, data):  # the patch's Push::Apply before every mj_step of the twin
            if data is patch.data:
                if state['left'] > 0:
                    data.xfrc_applied[patch.pelvis, :3] = state['force']
                    state['left'] -= 1
                else:
                    data.xfrc_applied[patch.pelvis, :3] = 0.
            recording(model, data)

        monkeypatch.setattr(mujoco, 'mj_step', bridge)
        for env in (pushed, patch, still):
            env.reset(seed=3, options={'command': [0., 0., 0.]})
        assert not pushed.push_live  # a set command: no pushes of its own
        zero = np.zeros(pushed.model.nu)
        for _ in range(5):  # land on the floor
            for env in (pushed, patch, still):
                env.step(zero)
        np.testing.assert_array_equal(pushed.data.qvel, still.data.qvel)
        q = patch.data.xquat[patch.pelvis]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))
        size = patch.mass * 1. / .1
        forward, left = direction
        state.update(left=round(.1 / patch.model.opt.timestep), force=np.array(
            [size * (forward * math.cos(yaw) - left * math.sin(yaw)),
             size * (forward * math.sin(yaw) + left * math.cos(yaw)), 0.]))
        forces.clear()
        pushed.set_force_push(direction, .1)
        assert pushed.push_left == 50
        along = state['force'][:2] / size
        for step in range(12):
            for env in (pushed, patch, still):
                env.step(zero)
            np.testing.assert_allclose(pushed.data.qvel, patch.data.qvel, rtol=0, atol=1e-8)
            np.testing.assert_allclose(pushed.data.qpos, patch.data.qpos, rtol=0, atol=1e-8)
            if step == 4:  # the push's last step
                gain = float((pushed.data.qvel[:2] - still.data.qvel[:2]) @ along)
                assert 1.4 < gain < 2.
        wrench = np.array([w for w, _ in forces])
        np.testing.assert_allclose(wrench[:50, :3], np.repeat(state['force'][None], 50, axis=0), rtol=1e-12, atol=1e-9)
        assert np.linalg.norm(wrench[0, :2]) == pytest.approx(MASS / .1, rel=1e-6)
        assert not wrench[50:].any() and not wrench[:, 2:].any()
        assert wrench[:, :2].sum(axis=0) @ along * pushed.model.opt.timestep == pytest.approx(MASS * 1., rel=1e-9)
        assert pushed.push_left == 0 and not pushed.data.xfrc_applied.any()
    finally:
        pushed.close(), patch.close(), still.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_and_on_set_commands_the_step_is_bitwise_the_old_one(profile):
    """No setting (missing or 0) draws nothing and pushes nothing; with it, resets with a command and set_command
    (evaluation, scenarios) push nothing either, past the 2 s the first push could come."""
    plain, zero, unused = G1VelocityEnv(**profile), G1VelocityEnv(force_push=0., **profile), \
        G1VelocityEnv(force_push=1.25, **profile)
    try:
        nu = plain.model.nu
        np.testing.assert_array_equal(plain.reset(seed=5)[0], zero.reset(seed=5)[0])
        assert not zero.push_live and zero.push_timer == 0
        for action in actions(30, nu, seed=2):
            expected, result = plain.step(action), zero.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1]
            np.testing.assert_array_equal(zero.data.qpos, plain.data.qpos)
        command = dict(options={'command': [.6, 0., 0.]})
        for env in (plain, zero, unused):
            env.reset(seed=8, **command)
        assert not unused.push_live and unused.push_timer == 0
        for action in actions(220, nu, seed=3, size=.1):
            expected = plain.step(action)
            for env in (zero, unused):
                result = env.step(action)
                np.testing.assert_array_equal(result[0], expected[0])
                assert result[1] == expected[1]
                np.testing.assert_array_equal(env.data.qvel, plain.data.qvel)
        # A training episode (drawn last at its reset, so the same pose and command) stops pushing at set_command.
        np.testing.assert_array_equal(plain.reset(seed=11)[0], unused.reset(seed=11)[0])
        assert unused.push_live and unused.push_timer >= 100
        unused.push_timer = 0  # a push would start at the next step
        for env in (plain, unused):
            env.set_command([.5, 0., 0.])
        assert not unused.push_live
        for action in actions(10, nu, seed=4):
            expected, result = plain.step(action), unused.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1]
        assert unused.push_left == 0
    finally:
        plain.close(), zero.close(), unused.close()


def test_set_command_ends_a_running_push_and_a_set_push_needs_a_velocity_and_a_substep():
    env = G1VelocityEnv()
    try:
        env.reset(seed=1, options={'command': [0., 0., 0.]})
        env.set_force_push((1., 0.), .05)
        env.step(np.zeros(env.model.nu))
        assert env.push_left == 15  # 25 substeps, 10 of them this step
        env.set_command([0., 0., 0.])
        assert env.push_left == 0
        for velocity, seconds in (((1.,), .1), ((1., float('nan')), .1), ((1., 0.), 0.), ((1., 0.), .0009),
                                  ((1., 0.), float('inf')), ((1., 0.), True)):
            with pytest.raises(ValueError):
                env.set_force_push(velocity, seconds)
    finally:
        env.close()


def test_the_legs_only_g1_is_pushed_too():
    env = G1VelocityEnv(legs_only=True, force_push=1., max_episode_steps=10 ** 6)
    try:
        assert env.mass == pytest.approx(MASS, abs=1e-6)  # welded links keep their mass
        env.reset(seed=2)
        pushed = 0
        for action in actions(250, env.model.nu, seed=5, size=.05):
            env.step(action)
            pushed += env.push_left > 0 or not np.array_equal(env.push_force, np.zeros(3))
        assert pushed > 0 and np.linalg.norm(env.push_force[:2]) > 0
    finally:
        env.close()


def test_force_push_config_is_bounded_labelled_and_outside_the_checkpoint_identity():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['force_push'] == 0  # old branches keep theirs
    for value in (1.25, 2, 0):
        assert validate_config({**base, 'task_config': {'force_push': value}})['task_config']['force_push'] == value
    for bad in (-.01, 2.01, float('nan'), True, '1.25', {'max_dv': 1.25}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'force_push': bad}})
    for bad in (-.01, 2.5, float('nan')):
        with pytest.raises(ValueError):
            G1VelocityEnv(force_push=bad)
    field = next(p for p in G1_PARAMETERS if p['key'] == 'force_push')
    assert field['default'] == 0 and field['min'] == 0 and field['max'] == 2 and 'Толчки' in field['label']
    config = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., lock_wrists=True,
                  knee_floor=.061)
    saved = task_identity('g1_29dof', 'g1_velocity', config)
    pushed = task_identity('g1_29dof', 'g1_velocity', {**config, 'force_push': 1.25})
    assert pushed == saved
    validate_resume(saved, pushed)


def test_candidate_stage_resumes_with_force_push():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import read_json
    if not (CANDIDATE / 'metadata.json').exists():
        pytest.skip('candidate stage is not on this machine')
    metadata = read_json(CANDIDATE / 'metadata.json')
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'force_push': 1.25})
    validate_resume(checkpoint_identity(metadata), requested)
