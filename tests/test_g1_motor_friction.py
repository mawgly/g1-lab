from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

PROFILES = (dict(real_robot=True), dict(real_robot=True, action_limit=10., phase_stop=True, history=5,
                                        gait={'period': .6, 'stance_fraction': .55}))
# The parent of iteration 27: iteration 26 (side_heading_hold false), the stage its branch starts from.
PARENT = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-1357d79b90f94477/stage_007121240064'


def actions(count, nu, seed=0, size=.4):
    return np.random.default_rng(seed).uniform(-size, size, (count, nu))


def record_substeps(monkeypatch, env):
    """(joint angles, joint velocities, ctrl) as each mj_step of env gets them, appended to the returned list."""
    seen, original = [], mujoco.mj_step

    def step(model, data):
        if data is env.data:
            seen.append((data.qpos[env.qadr].copy(), data.qvel[env.vadr].copy(), data.ctrl.copy()))
        original(model, data)

    monkeypatch.setattr(mujoco, 'mj_step', step)
    return seen


def test_the_draw_is_uniform_between_zero_and_the_setting():
    generator = torch.Generator().manual_seed(5)
    uniform = lambda k: torch.rand(k, generator=generator, dtype=torch.float64)
    scale = task.motor_friction_draw(uniform, 20000, 1.25)
    assert scale.shape == (20000,) and float(scale.min()) >= 0 and float(scale.max()) < 1.25
    assert float(scale.min()) < .01 and float(scale.max()) > 1.24
    assert float(scale.mean()) == pytest.approx(.625, abs=.01)
    shares = [float(((scale >= .25 * k) & (scale < .25 * (k + 1))).double().mean()) for k in range(5)]
    assert shares == pytest.approx([.2] * 5, abs=.015)


def test_the_motor_torque_friction_is_scaled_by_exactly_m_per_world():
    """Both friction parts (static·tanh(v/0.01) + viscous·v) and nothing else: the clipped PD torque minus m times the
    friction, bitwise, per row; m = 1 is the original model bit for bit, m = 0 the frictionless one."""
    env = G1VelocityEnv(real_robot=True)
    try:
        x1, x2, y1, y2, static, viscous = env._motor_tensors
    finally:
        env.close()
    rng = np.random.default_rng(1)
    torque = torch.from_numpy(rng.uniform(-150, 150, (4, 29)))
    velocity = torch.from_numpy(rng.uniform(-30, 30, (4, 29)))
    velocity[:, :6] = torch.from_numpy(rng.uniform(-.02, .02, (4, 6)))  # the steep part of tanh
    plain = task.motor_torque(torque, velocity, x1, x2, y1, y2, static, viscous)
    clipped = task.motor_torque(torque, velocity, x1, x2, y1, y2, 0 * static, 0 * viscous)
    friction = static * torch.tanh(velocity / .01) + viscous * velocity
    assert torch.equal(plain, clipped - friction) and float((plain - clipped).abs().max()) > 5
    m = torch.tensor([0., .25, 1., 1.25], dtype=torch.float64)
    scaled = task.motor_torque(torque, velocity, x1, x2, y1, y2, static, viscous, friction_scale=m)
    assert torch.equal(scaled, clipped - m[:, None] * friction)
    assert torch.equal(scaled[0], clipped[0]) and torch.equal(scaled[2], plain[2])
    assert torch.equal(task.motor_torque(torque, velocity, x1, x2, y1, y2, static, viscous,
                                         friction_scale=torch.ones(4, dtype=torch.float64)), plain)
    # An unbatched torque (the CPU task's) is one world; a float32 multiplier is cast to the torque's dtype.
    one = task.motor_torque(torque[3], velocity[3], x1, x2, y1, y2, static, viscous,
                            friction_scale=torch.tensor([1.25], dtype=torch.float32))
    assert one.shape == (29,) and torch.equal(one, scaled[3])


@pytest.mark.parametrize('profile', PROFILES)
def test_the_cpu_task_applies_the_shared_motor_model_with_its_multiplier(profile, monkeypatch):
    """Every substep's ctrl of the CPU task is the actuator range clip of motor_torque with this episode's m, exactly
    (float64), and the CUDA task's computation of it (pd_torque without torque bounds, motor_torque, the clamp) in
    float32 on CPU tensors gives the same torque; m changes the torque by m times the friction."""
    env = G1VelocityEnv(motor_friction=1.25, **profile)
    try:
        seen = record_substeps(monkeypatch, env)
        env.reset(seed=3)  # a training reset draws m
        m = env.friction_scale
        assert 0 <= m < 1.25 and m != 1 and torch.equal(env._friction_scale, torch.tensor([m], dtype=torch.float64))
        steps = actions(6, env.model.nu, seed=1)
        for action in steps:
            env.step(action)
        assert len(seen) == 60
        f32 = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)[None]
        motors64 = env._motor_tensors
        motors32 = tuple(torch.as_tensor(row, dtype=torch.float32) for row in env.motors)
        low, high = env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1]
        unclamped = torch.full((env.model.nu,), float('inf'))
        friction_seen = 0.
        for k, (position, velocity, ctrl) in enumerate(seen):
            action = np.clip(steps[k // 10], -env.action_limit, env.action_limit)
            pd = env.kp * (env._target(action) - position) - env.kd * velocity
            exact = task.motor_torque(torch.from_numpy(pd), torch.from_numpy(velocity), *motors64,
                                      friction_scale=torch.tensor([m], dtype=torch.float64)).numpy()
            np.testing.assert_array_equal(ctrl, np.clip(exact, low, high))
            torque = task.pd_torque(f32(action), f32(position), f32(velocity), f32(env.home), f32(env.scale),
                                    f32(env.kp), f32(env.kd), f32(env.target_low), f32(env.target_high),
                                    -unclamped, unclamped, env.action_limit)
            torque = task.motor_torque(torque, f32(velocity), *motors32,
                                       friction_scale=torch.tensor([m], dtype=torch.float32)).clamp(
                min=f32(low), max=f32(high))
            np.testing.assert_allclose(torque[0].numpy(), ctrl, rtol=1e-5, atol=1e-3)
            nominal = task.motor_torque(torch.from_numpy(pd), torch.from_numpy(velocity), *motors64).numpy()
            frictionless = task.motor_torque(torch.from_numpy(pd), torch.from_numpy(velocity),
                                             *motors64[:4], 0 * motors64[4], 0 * motors64[5]).numpy()
            np.testing.assert_allclose(frictionless - exact, m * (frictionless - nominal), rtol=1e-12, atol=1e-12)
            friction_seen = max(friction_seen, float(np.abs(frictionless - nominal).max()))
        assert friction_seen > 1  # the joints moved: the friction acted
    finally:
        env.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_and_on_command_resets_the_step_is_bitwise_the_old_one(profile):
    """No setting (missing or 0) draws nothing and scales nothing; with it, a reset with a command keeps m = 1 and
    draws nothing either; a training reset draws m after every other draw, so the pose, command and random stream
    up to it are the old ones."""
    plain, zero, unused = (G1VelocityEnv(**profile), G1VelocityEnv(motor_friction=0., **profile),
                           G1VelocityEnv(motor_friction=1.25, **profile))
    try:
        nu = plain.model.nu
        np.testing.assert_array_equal(plain.reset(seed=5)[0], zero.reset(seed=5)[0])
        assert zero.friction_scale == 1 and zero._friction_scale is None
        for action in actions(30, nu, seed=2):
            expected, result = plain.step(action), zero.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1]
            np.testing.assert_array_equal(zero.data.qpos, plain.data.qpos)
        assert zero.np_random.bit_generator.state == plain.np_random.bit_generator.state
        for env in (plain, zero, unused):
            env.reset(seed=8, options={'command': [.6, 0., 0.]})
        assert unused.friction_scale == 1 and unused._friction_scale is None
        for action in actions(60, nu, seed=3, size=.2):
            expected = plain.step(action)
            for env in (zero, unused):
                result = env.step(action)
                np.testing.assert_array_equal(result[0], expected[0])
                assert result[1] == expected[1]
                np.testing.assert_array_equal(env.data.qvel, plain.data.qvel)
        assert unused.np_random.bit_generator.state == plain.np_random.bit_generator.state
        # A training reset: the same pose and command, then one more draw, m = 1.25·U.
        np.testing.assert_array_equal(plain.reset(seed=11)[0], unused.reset(seed=11)[0])
        np.testing.assert_array_equal(unused.command, plain.command)
        assert unused.friction_scale == 1.25 * plain.np_random.random()
        assert unused.np_random.bit_generator.state == plain.np_random.bit_generator.state
        for action in actions(10, nu, seed=4):
            plain.step(action), unused.step(action)
        assert np.abs(unused.data.qvel - plain.data.qvel).max() > .01  # another friction, another motion
    finally:
        plain.close(), zero.close(), unused.close()


def test_training_resets_draw_m_and_command_resets_options_and_the_setter_fix_it():
    env = G1VelocityEnv(motor_friction=1.25, **PROFILES[1])
    try:
        env.reset(seed=0)
        scales = [env.friction_scale]
        for _ in range(399):
            env.reset()
            scales.append(env.friction_scale)
        scales = np.array(scales)
        assert scales.min() >= 0 and scales.max() < 1.25 and len(set(scales.tolist())) == 400
        assert scales.mean() == pytest.approx(.625, abs=.06)
        assert (scales < .25).mean() > .1 and (scales > 1.).mean() > .1
        env.reset(seed=1, options={'command': [0., 0., 0.]})
        assert env.friction_scale == 1 and env._friction_scale is None
        state = env.np_random.bit_generator.state
        env.reset(seed=1, options={'command': [0., 0., 0.], 'motor_friction_scale': .25})
        assert env.friction_scale == .25 and env.np_random.bit_generator.state == state  # set, not drawn
        env.reset(seed=1, options={'motor_friction_scale': 0})  # a training reset with a set m: no draw either
        assert env.friction_scale == 0 and torch.equal(env._friction_scale, torch.zeros(1, dtype=torch.float64))
        env.set_motor_friction(1.5)
        env.set_command([.3, 0., 0.])  # an episode property, as action_delay: set_command keeps it
        assert env.friction_scale == 1.5
        env.set_motor_friction(1)
        assert env._friction_scale is None
        for bad in (-.1, float('nan'), float('inf'), True, '1', None, [1.]):
            with pytest.raises(ValueError):
                env.set_motor_friction(bad)
            with pytest.raises(ValueError):
                env.reset(seed=1, options={'motor_friction_scale': bad})
    finally:
        env.close()
    simulation = G1VelocityEnv()  # no motor model, nothing to scale
    try:
        simulation.reset(seed=1)
        assert simulation.friction_scale == 1 and simulation._friction_scale is None
        with pytest.raises(ValueError):
            simulation.set_motor_friction(.5)
        with pytest.raises(ValueError):
            simulation.reset(seed=1, options={'motor_friction_scale': .5})
    finally:
        simulation.close()
    for bad in (dict(motor_friction=1.), dict(real_robot=True, motor_friction=2.5),
                dict(real_robot=True, motor_friction=-.1)):
        with pytest.raises(ValueError):
            G1VelocityEnv(**bad)


def test_motor_friction_config_is_bounded_labelled_real_robot_only_and_outside_the_checkpoint_identity():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.tasks.config import G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    real = {'real_robot': True, 'action_limit': 10.}
    assert validate_config(base)['task_config']['motor_friction'] == 0  # old branches keep theirs
    assert validate_config({**base, 'task_config': {'motor_friction': 0}})['task_config']['motor_friction'] == 0
    assert validate_config({**base, 'task_config': {**real, 'motor_friction': None}})['task_config'][
        'motor_friction'] == 0
    for value in (1.25, 2, 0, .5):
        config = validate_config({**base, 'task_config': {**real, 'motor_friction': value}})
        assert config['task_config']['motor_friction'] == value
    for bad in (-.01, 2.01, float('nan'), float('inf'), True, '1.25', [1.25], {'max': 1.25}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {**real, 'motor_friction': bad}})
    with pytest.raises(ValueError, match='real_robot'):  # the simulation profile has no motor friction
        validate_config({**base, 'task_config': {'motor_friction': 1.25}})
    field = next(p for p in G1_PARAMETERS if p['key'] == 'motor_friction')
    assert field['default'] == 0 and field['min'] == 0 and field['max'] == 2 and 'трения моторов' in field['label']
    checked = validate_config({**base, 'task_config': {**real, 'motor_friction': 1.25}})
    env = make_task(checked['task_id'], checked['robot_id'], checked['task_config'])
    try:
        assert env.unwrapped.motor_friction == 1.25
    finally:
        env.close()
    config = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., lock_wrists=True,
                  knee_floor=.061, foot_collision=True)
    saved = task_identity('g1_29dof', 'g1_velocity', config)
    scaled = task_identity('g1_29dof', 'g1_velocity', {**config, 'motor_friction': 1.25})
    assert scaled == saved
    validate_resume(saved, scaled)


def test_the_iteration_26_stage_resumes_with_motor_friction():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import read_json
    if not (PARENT / 'metadata.json').exists():
        pytest.skip('iteration 26 stage is not on this machine')
    metadata = read_json(PARENT / 'metadata.json')
    assert 'motor_friction' not in metadata['task_config']  # saved before the setting: 0
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'motor_friction': 1.25})
    validate_resume(checkpoint_identity(metadata), requested)
