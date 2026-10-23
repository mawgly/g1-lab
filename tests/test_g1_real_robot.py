"""Real-robot profile of the G1 task (docs/g1-real-robot-plan.md)."""
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.catalog import make_task
from humanoid_sim.tasks.config import g1_config
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv, MOTORS, motor_parameters
from humanoid_sim.tasks.identity import task_identity

REAL = dict(real_robot=True, action_limit=10.)


def unitree_clip(effort, velocity, x1, x2, y1, y2):
    """unitree_rl_lab UnitreeActuator._clip_effort, transcribed for one joint."""
    peak = y1 if velocity * effort > 0 else y2
    if abs(velocity) >= x1:
        peak = max(0., -peak / (x2 - x1) * (abs(velocity) - x1) + peak)
    return float(np.clip(effort, -peak, peak))


def test_motor_torque_follows_unitree_curve_and_friction():
    x1, x2, y1, y2, fs, fd = MOTORS['7520-22']
    cases = [(200., 1.), (200., -1.), (-200., 18.), (200., 18.), (50., 30.), (-20., 0.)]
    torque = torch.tensor([c[0] for c in cases], dtype=torch.float64)
    velocity = torch.tensor([c[1] for c in cases], dtype=torch.float64)
    params = [torch.full((len(cases),), value, dtype=torch.float64) for value in (x1, x2, y1, y2, fs, fd)]
    result = task.motor_torque(torque, velocity, *params)
    for (effort, speed), value in zip(cases, result.tolist()):
        expected = unitree_clip(effort, speed, x1, x2, y1, y2) - (fs * np.tanh(speed / .01) + fd * speed)
        assert value == pytest.approx(expected)
    assert result[0] == pytest.approx(111 - 2.4 * np.tanh(100) - .24)  # same direction: y1 = 111 N·m
    assert result[1] == pytest.approx(131 + 2.4 * np.tanh(100) + .24)  # braking: y2 = 131 N·m


def test_real_robot_env_matches_unitree_deploy_layout():
    env = G1VelocityEnv(**g1_config(REAL))
    try:
        obs, _ = env.reset(seed=4, options={'command': [.6, -.2, .3]})
        assert obs.shape == (11 + 3 * 29,)  # deploy.yaml velocity v0: 98 values
        np.testing.assert_allclose(obs[6:9], [.6, -.2, .3], atol=1e-6)  # command after angular velocity and gravity
        np.testing.assert_allclose(obs[9:11], [0., 1.], atol=1e-6)  # phase (sin, cos) at t = 0
        np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=2e-2)  # projected gravity, upright
        assert env.gait['period'] == pytest.approx(.6)
        names = [env.model.joint(j).name for j in env.joints]
        armature = env.model.dof_armature[env.vadr]
        assert armature[names.index('left_knee_joint')] == pytest.approx(.0251, abs=1e-4)
        assert armature[names.index('left_hip_pitch_joint')] == pytest.approx(.0102, abs=1e-4)
        assert armature[names.index('left_ankle_pitch_joint')] == pytest.approx(.0072, abs=1e-4)
        assert env.model.dof_frictionloss[env.vadr].max() == 0
        assert np.all(env.model.geom_friction[env.foot_geoms, 0] == .6) and np.all(env.model.geom_priority[env.foot_geoms] == 1)
        assert env.action_space.high[0] == 10
        for _ in range(5):
            obs, reward, terminated, _, info = env.step(np.zeros(29))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    finally:
        env.close()
    plain = G1VelocityEnv()
    try:
        assert plain.observation_space.shape == (101,) and plain.model.dof_frictionloss[plain.vadr].max() == .2
    finally:
        plain.close()
    params = motor_parameters(['left_knee_joint', 'right_wrist_yaw_joint', 'waist_pitch_joint'])
    np.testing.assert_allclose(params[2], [111., 4.8, 49.6])  # y1: 7520-22, 4010, two 5020


def test_critic_extras_reach_the_critic_but_never_the_actor(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import load_stage, save_stage, write_json
    from humanoid_sim.policies import CriticExtrasPolicy
    config = g1_config({**REAL, 'critic_extras': True})
    env = G1VelocityEnv(**config)
    try:
        obs, _ = env.reset(seed=1, options={'command': [1., 0, 0]})
        assert obs.shape == (98 + 4,) and env.actor_inputs == 98
        np.testing.assert_allclose(obs[-1], env.data.qpos[2], atol=1e-6)  # pelvis height
    finally:
        env.close()
    assert task_identity('g1_29dof', 'g1_velocity', config)['observation_schema'] == \
        'g1_velocity:obs-unitree-deploy-v0-critic4'
    with pytest.raises(ValueError):
        g1_config({'critic_extras': True})
    with pytest.raises(ValueError):
        g1_config({**REAL, 'training_realism': 1.5})
    # the curriculum knob shapes training only, not the policy contract
    assert task_identity('g1_29dof', 'g1_velocity', {**config, 'training_realism': 0.}) == \
        task_identity('g1_29dof', 'g1_velocity', config)
    write_json(tmp_path / 'run_config.json', dict(robot_id='g1_29dof', task_id='g1_velocity', task_config=config))
    vec = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config)]))
    model = PPO(CriticExtrasPolicy, vec, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=2,
                policy_kwargs=dict(net_arch=[32, 32], actor_inputs=98))
    torch.manual_seed(0)
    a, b = torch.randn(4, 102), torch.randn(4, 102)
    b[:, :98] = a[:, :98]  # same robot inputs, different critic extras
    policy = model.policy
    with torch.no_grad():
        assert torch.equal(policy.get_distribution(a).distribution.mean, policy.get_distribution(b).distribution.mean)
        assert not torch.equal(policy.predict_values(a), policy.predict_values(b))
    model.learn(16)
    stage = save_stage(tmp_path, model, vec)
    vec.close()
    loaded, norm = load_stage(stage)
    try:
        assert isinstance(loaded.policy, CriticExtrasPolicy) and loaded.policy.actor_inputs == 98
        with torch.no_grad():
            torch.testing.assert_close(loaded.policy.predict_values(b), model.policy.predict_values(b))
    finally:
        norm.close()


def test_real_robot_contract_and_joystick_commands():
    base = task_identity('g1_29dof', 'g1_velocity')
    real = task_identity('g1_29dof', 'g1_velocity', REAL)
    assert real['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0'
    assert real['model_revision'] != base['model_revision']
    with pytest.raises(ValueError):
        validate_resume(base, real)
    with pytest.raises(ValueError):
        G1VelocityEnv(real_robot=True, legs_only=True)
    generator = torch.Generator().manual_seed(5)
    commands = task.sample_commands(20000, lambda n: torch.rand(n, generator=generator), -.5, 1., .1, .5, 1.)
    standing = (commands.abs().sum(dim=1) == 0).float().mean()
    assert abs(float(standing) - .1) < .02
    assert commands[:, 0].min() >= -.5 and commands[:, 0].max() <= 1.
    assert commands[:, 1].abs().max() <= .5 and commands[:, 1].abs().max() > .45
    assert commands[:, 2].abs().max() <= 1. and commands[:, 2].abs().max() > .95
    old = task.sample_commands(5, lambda n: torch.full((n,), .5, dtype=torch.float64))
    np.testing.assert_allclose(old[:, 1:].numpy(), 0)  # forward-only configs draw no extra numbers
    env = make_task('g1_velocity', 'g1_29dof', g1_config({**REAL, 'commands': {'min_speed': -.5, 'max_speed': 1.,
                                                                                    'lateral_speed': .5, 'yaw_rate': 1.}}))
    try:
        seen = []
        for seed in range(6):
            env.reset(seed=seed)
            seen.append(env.unwrapped.command.copy())
        assert np.abs(np.array(seen)[:, 1:]).max() > 0
    finally:
        env.close()
