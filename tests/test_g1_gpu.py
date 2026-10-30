"""Run explicitly in a free compute slot: RUN_G1_GPU_TESTS=1, .venv-fullgpu."""
import math
import os
import numpy as np
import pytest
import torch
from humanoid_sim.g1_tensor_env import TensorG1Env


def test_invalid_batch_fails_before_cuda_setup():
    with pytest.raises(ValueError):
        TensorG1Env(0)


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_timeout_preserves_terminal_observation_and_resets():
    env = TensorG1Env(2, seed=7, max_episode_steps=1)
    try:
        obs=env.reset()
        assert obs.shape==(2,101) and obs.is_cuda
        nxt,reward,terminated,truncated,terminal=env.step(torch.full((2,29),.2,device=env.device))
        env.check_health()
        assert truncated.all() and not terminated.any()
        torch.testing.assert_close(terminal[:,-31:-2],torch.full((2,29),.2,device=env.device))
        assert torch.count_nonzero(nxt[:,-31:-2])==0
        assert env.lengths.tolist()==[0,0]
        assert env.last_episode_lengths.tolist()==[1,1]
        assert torch.isfinite(reward).all()
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_pushes_only_when_enabled(monkeypatch):
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'PUSH_INTERVAL_SECONDS', .02)  # push every step
    speeds = []
    for push in (1., 0.):
        env = TensorG1Env(64, seed=3, resample_commands=False, push_velocity=push)
        try:
            env.reset()
            env.step(torch.zeros((64,29),device=env.device))
            env.check_health()
            speeds.append(float(env.qvel[:,:2].norm(dim=1).mean()))
        finally:
            env.close()
    assert speeds[0] > .2 > speeds[1]


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_real_robot_randomizes_worlds_and_noises_observations():
    import warp as wp
    from humanoid_sim.g1_tensor_env import FOOT_FRICTION_RANGE
    env = TensorG1Env(16, seed=7, real_robot=True, action_limit=10.)
    try:
        assert env.observation_space.shape == (98,)
        friction = wp.to_torch(env.model.geom_friction)[:, env.reference.foot_geoms, 0]
        assert friction.shape[0] == 16 and torch.all(friction == friction[:, :1])  # shared by both feet
        assert FOOT_FRICTION_RANGE[0] <= float(friction.min()) and float(friction.max()) <= FOOT_FRICTION_RANGE[1]
        assert float(friction[:, 0].std()) > .05  # differs between worlds
        com = wp.to_torch(env.model.body_ipos)[:, env.pelvis]
        assert float(com.std(dim=0).min()) > 1e-3
        first = env.reset()
        clean = TensorG1Env(16, seed=7, real_robot=True, action_limit=10., observation_noise=False, randomize=False)
        try:
            quiet = clean.reset()
            assert float((first[:, 11:40] - quiet[:, 11:40]).abs().max()) > 1e-3  # encoder bias and noise
        finally:
            clean.close()
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((16, 29), device=env.device))
        env.check_health()
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    finally:
        env.close()
    calm = TensorG1Env(16, seed=7, real_robot=True, action_limit=10., training_realism=0.)
    try:  # realism 0: no noise, no friction or COM spread, no pushes (random draws still happen, scaled by 0)
        friction = wp.to_torch(calm.model.geom_friction)[:, calm.reference.foot_geoms, 0]
        torch.testing.assert_close(friction, torch.full_like(friction, .6))
        assert float(wp.to_torch(calm.model.body_ipos)[:, calm.pelvis].std(dim=0).max()) < 1e-7
        calm.reset()
        assert float(calm.encoder_bias.abs().max()) == 0
        assert torch.equal(calm._obs(), calm._obs())  # noisy observations would differ between two reads
        before = calm.qvel.clone()
        calm.step(torch.zeros((16, 29), device=calm.device))
        assert float((calm.qvel[:, :6] - before[:, :6]).abs().max()) < 5.  # no 6-DoF kicks on top of the physics
    finally:
        calm.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_history_shifts_each_step_and_refills_reset_worlds():
    from humanoid_sim.tasks import g1_tensor_math as task
    env = TensorG1Env(4, seed=5, real_robot=True, action_limit=10., critic_extras=True, history=5, max_episode_steps=3,
                      resample_commands=False, push_velocity=0, observation_noise=False, randomize=False)
    try:
        obs = env.reset()
        assert obs.shape == (4, 5 * 98 + 4)
        start = env.frames.clone()
        assert torch.equal(start, start[:, :1].expand(-1, 5, -1))  # filled with the first frame
        nxt, _, _, _, terminal = env.step(torch.zeros((4, 29), device=env.device))
        env.check_health()
        torch.testing.assert_close(env.frames[:, :-1], start[:, 1:])  # shifted by one, newest last
        assert torch.equal(nxt[:, :490], task.history_observation(env.frames, 29)) and torch.equal(nxt, terminal)
        for _ in range(2):
            nxt, _, _, truncated, terminal = env.step(torch.zeros((4, 29), device=env.device))
        assert truncated.all()
        assert torch.equal(env.frames, env.frames[:, :1].expand(-1, 5, -1))  # fresh history after the reset
        assert not torch.equal(terminal[:, :490], nxt[:, :490])  # the terminal observation keeps the old one
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_straight_command_holds_the_heading_it_starts_at(monkeypatch):
    import warp as wp
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)  # resample every step
    commands = {'min_speed': .5, 'max_speed': 1., 'stand_probability': 0., 'yaw_rate': 1., 'straight_probability': 1.}
    env = TensorG1Env(4, seed=2, real_robot=True, action_limit=10., heading_hold=True, commands=commands,
                      push_velocity=0, observation_noise=False, randomize=False)
    try:
        env.reset()
        yaw = .5
        env.qpos[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], device=env.device)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device))):
            wp.capture_launch(env.forward_graph)
        env.step(torch.zeros((4, 29), device=env.device))
        env.check_health()
        here = task.heading(env.xmat[:, env.pelvis].reshape(-1, 3, 3))
        torch.testing.assert_close(env.heading_target, here, atol=1e-4, rtol=0)
        assert float(env.heading_target.min()) > .4
    finally:
        env.close()
    for phase_stop in (False, True):  # told to stand, the phase input reads (0, 0) only with phase_stop
        env = TensorG1Env(2, seed=2, real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=False,
                          phase_stop=phase_stop, commands={'stand_probability': 1.}, push_velocity=0,
                          observation_noise=False, randomize=False)
        try:
            phase = env.reset()[:, 9:11]
            assert bool((phase == 0).all()) == phase_stop
        finally:
            env.close()
    for stand_heading_hold in (True, False):  # a turned robot told to stand: turn in place, or a zero command
        env = TensorG1Env(2, seed=2, real_robot=True, action_limit=10., heading_hold=True,
                          stand_heading_hold=stand_heading_hold, commands={'stand_probability': 1.},
                          push_velocity=0, observation_noise=False, randomize=False)
        try:
            env.reset()
            env.qpos[:, 3:7] = torch.tensor([math.cos(.25), 0, 0, math.sin(.25)], device=env.device)
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device))):
                wp.capture_launch(env.forward_graph)
            env._obs()
            yaw = env.observed_command[:, 2]
            assert bool((yaw < -.2).all()) if stand_heading_hold else bool((yaw == 0).all())
        finally:
            env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_side_command_turn_stays_zero_without_side_heading_hold(monkeypatch):
    """side_heading_hold through the shared effective_command: four worlds turned 0.5 rad off the held heading, told
    to step left, right, forward and diagonally. Missing and true: heading hold steers all four (−0.25), as before;
    false: the side steps' turn is exactly zero in the observed and in the rewarded command, the other two as before.
    The same command as effective_command on CPU tensors and as the CPU task with the same setting."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):  # the command the CUDA env rewards
        seen.append(args[3].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    commands = torch.tensor([[0., 1., 0.], [0., -1., 0.], [1., 0., 0.], [.5, .5, 0.]])
    side = task.pure_sideways(commands)
    assert side.tolist() == [True, True, False, False]
    profile = dict(real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=False)
    steered = {}
    for setting in (None, True, False):
        options = {} if setting is None else {'side_heading_hold': setting}
        hold = setting is not False
        env = TensorG1Env(4, seed=2, resample_commands=False, push_velocity=0, observation_noise=False,
                          randomize=False, **profile, **options)
        cpu = G1VelocityEnv(**profile, **options)
        try:
            assert env.reference.side_heading_hold is hold
            env.reset()
            env.command.copy_(commands.to(env.device))
            env.qpos[:, 3:7] = torch.tensor([math.cos(.25), 0, 0, math.sin(.25)], device=env.device)
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device))):
                wp.capture_launch(env.forward_graph)
            env._obs()
            observed = env.observed_command.cpu()
            yaw = observed[:, 2]
            assert bool((yaw[~side] < -.2).all())  # forward and diagonal: steered back in every case
            assert bool((yaw[side] < -.2).all()) if hold else bool((yaw[side] == 0).all())
            torch.testing.assert_close(observed[:, :2], commands[:, :2])
            rotation = env.xmat[:, env.pelvis].reshape(-1, 3, 3).cpu().double()
            shared = task.effective_command(commands.double(), rotation, env.heading_target.cpu().double(), False, hold)
            torch.testing.assert_close(observed.double(), shared, atol=1e-6, rtol=0)
            for k, command in enumerate(commands.tolist()):  # the CPU task with the same setting
                cpu.reset(seed=1, options={'command': command})
                cpu.data.qpos[3:7] = [math.cos(.25), 0, 0, math.sin(.25)]
                mujoco.mj_forward(cpu.model, cpu.data)
                cpu._observation()
                np.testing.assert_allclose(cpu.observed_command, observed[k].numpy(), atol=1e-5, rtol=0)
                if not hold and side[k]:
                    assert cpu.observed_command[2] == 0.
            env.step(torch.zeros((4, 29), device=env.device))
            env.check_health()
            assert torch.equal(seen[-1].cpu(), observed)  # rewarded for the command it observed
            steered[setting] = observed
        finally:
            env.close()
            cpu.close()
    assert torch.equal(steered[None], steered[True])
    assert torch.equal(steered[False][~side], steered[True][~side])


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_side_commands_stop_after_a_sideways_step_and_keep_the_old_draw_without_them(monkeypatch):
    """side_share / side_stop through the shared sample_commands, a resample at every step: (1) every walking command
    sideways and side_stop 1: each world goes sideways, stand, sideways, …, a sideways step holding the heading it
    starts at; (2) the iteration 18 commands with side_share 0.2 / side_stop 1 over 4096 worlds: the shares of
    sideways commands and of "sideways → stand" changes as the same chain drawn on the CPU; (3) without the settings
    every world's commands are bit for bit those of the old draw (same CUDA random stream)."""
    import warp as wp
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_side_stop import COMMANDS, _old_drop_in
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)  # resample every step
    profile = dict(real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=False, phase_stop=True,
                   push_velocity=0, observation_noise=False, randomize=False)
    sideways = {**COMMANDS, 'max_speed': 0., 'stand_probability': 0., 'side_share': 1., 'side_stop': 1.}
    env = TensorG1Env(64, seed=2, commands=sideways, **profile)
    try:
        env.reset()
        assert bool(task.pure_sideways(env.command).all())
        env.qpos[:, 3:7] = torch.tensor([math.cos(.25), 0, 0, math.sin(.25)], device=env.device)  # turned 0.5 rad
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device))):
            wp.capture_launch(env.forward_graph)
        for step in range(4):
            env.step(torch.zeros((64, 29), device=env.device))
            env.check_health()
            if step % 2 == 0:
                assert float(env.command.abs().max()) == 0  # stand after the sideways step
                continue
            assert bool(task.pure_sideways(env.command).all())
            speed = env.command[:, 1].abs()
            assert float(speed.min()) >= .1 - 1e-6 and float(speed.max()) <= .5 + 1e-6
            assert bool((env.command[:, 1] > 0).any()) and bool((env.command[:, 1] < 0).any())
            here = task.heading(env.xmat[:, env.pelvis].reshape(-1, 3, 3))
            torch.testing.assert_close(env.heading_target, here, atol=1e-4, rtol=0)  # sideways from here
            assert float(env.heading_target.min()) > .4
    finally:
        env.close()
    recipe = {**COMMANDS, 'side_share': .2, 'side_stop': 1.}
    n, steps = 4096, 10
    env = TensorG1Env(n, seed=3, commands=recipe, **profile)
    try:
        env.reset()
        side_start = float(task.pure_sideways(env.command).float().mean())
        side_stand = 0
        for _ in range(steps):
            before = env.command.clone()
            env.step(torch.zeros((n, 29), device=env.device))
            env.check_health()
            side_stand += int((task.pure_sideways(before) & (env.command.abs().sum(dim=1) == 0)).sum())
        gpu = side_stand / (n * steps)
    finally:
        env.close()
    generator = torch.Generator().manual_seed(0)
    uniform = lambda k: torch.rand(k, generator=generator)
    command = task.sample_commands(16 * n, uniform, **recipe)
    cpu_start = float(task.pure_sideways(command).float().mean())
    cpu = 0
    for _ in range(steps):
        fresh = task.sample_commands(16 * n, uniform, previous=command, **recipe)
        cpu += int((task.pure_sideways(command) & (fresh.abs().sum(dim=1) == 0)).sum())
        command = fresh
    cpu /= 16 * n * steps
    assert abs(side_start - .18) < .02 and abs(cpu_start - .18) < .01
    assert cpu > .05 and abs(gpu - cpu) < .008
    drawn = []
    for sampler in (task.sample_commands, _old_drop_in):
        monkeypatch.setattr(task, 'sample_commands', sampler)
        env = TensorG1Env(256, seed=4, commands=COMMANDS, **profile)
        try:
            env.reset()
            commands = [env.command.clone()]
            for _ in range(5):
                env.step(torch.zeros((256, 29), device=env.device))
                commands.append(env.command.clone())
            drawn.append(torch.stack(commands))
        finally:
            env.close()
    assert torch.equal(drawn[0], drawn[1])


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_side_speed_widens_only_the_purely_sideways_commands(monkeypatch):
    """side_speed through the shared sample_commands on the CUDA stream, a resample at every step (zero actions, so
    the physics and the stream do not depend on the commands): with the same seed every world draws what it draws
    without side_speed, except the purely sideways commands, whose |vy| is the same uniform number on [0.1, 1]
    instead of [0.1, 0.5] (no new draw, so the later resamples match too); the other walking commands stay within
    ±lateral_speed 0.5."""
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_side_stop import COMMANDS
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)  # resample every step
    profile = dict(real_robot=True, action_limit=10., heading_hold=True, stand_heading_hold=False, phase_stop=True,
                   push_velocity=0, observation_noise=False, randomize=False)
    n, drawn = 4096, []
    for side_speed in (0., 1.):
        commands = {**COMMANDS, 'side_share': .2, 'side_stop': 1., 'side_speed': side_speed}
        env = TensorG1Env(n, seed=5, commands=commands, **profile)
        try:
            env.reset()
            frames = [env.command.clone()]
            for _ in range(3):
                env.step(torch.zeros((n, 29), device=env.device))
                env.check_health()
                frames.append(env.command.clone())
            drawn.append(torch.stack(frames).reshape(-1, 3))
        finally:
            env.close()
    plain, wide = drawn
    side = task.pure_sideways(wide)
    assert torch.equal(side, task.pure_sideways(plain)) and int(side.sum()) > 1000
    assert torch.equal(wide[~side], plain[~side])
    assert torch.equal(torch.sign(wide[side, 1]), torch.sign(plain[side, 1]))
    speed = wide[side, 1].abs()
    assert float(speed.min()) >= .1 - 1e-6 and float(speed.max()) <= 1. + 1e-6 and float(speed.max()) > .9
    torch.testing.assert_close((speed - .1) / .9, (plain[side, 1].abs() - .1) / .4)  # the same uniform number
    walking = wide.abs().sum(dim=1) > 0
    assert float(wide[walking & ~side, 1].abs().max()) <= .5 + 1e-6


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
@pytest.mark.parametrize('options', [{}, {'heading_hold': True, 'reward_weights': {'torso': 2.}},
                                     {'legs_only': True, 'heading_hold': True,
                                      'reward_weights': {'orientation': 2., 'height': 20., 'tilt': 0.},
                                      'commands': {'min_speed': .3, 'max_speed': 1., 'stand_probability': 0.}},
                                     {'heading_hold': True, 'reward_weights': {'velocity': 0., 'forward': 3., 'gait': .5,
                                                                               'sideways': 3., 'posture': .1,
                                                                               'hips': 1., 'heading': 2.},
                                      'commands': {'min_speed': 1., 'max_speed': 1., 'stand_probability': 0.},
                                      'gait': {'period': .7, 'stance_fraction': .4}, 'action_limit': 4.},
                                     {'real_robot': True, 'action_limit': 10., 'heading_hold': True,
                                      'gait': {'period': .6, 'stance_fraction': .55}, 'critic_extras': True,
                                      'reward_weights': {'slip': 5., 'stand_pose': 1., 'still': 1.,
                                                         'feet_flat': 2.}},
                                     # Contact by the lowest sole point: the three steps drop the feet onto the floor,
                                     # sole 12.9 / 6.8 / -2.5 mm on CPU, so contact switches on at the second step.
                                     {'real_robot': True, 'action_limit': 10., 'heading_hold': True,
                                      'gait': {'period': .6, 'stance_fraction': .55}, 'sole_contact': True,
                                      'reward_weights': {'slip': 5., 'gait': 2., 'swing': 10., 'feet_flat': .5}},
                                     # feet_flat on the feet in contact only: at the first step both soles are in the
                                     # air (12.9 / 12.0 mm on CPU), so the term is 0; then both stand, nearly flat.
                                     {'real_robot': True, 'action_limit': 10., 'heading_hold': True,
                                      'gait': {'period': .6, 'stance_fraction': .55}, 'sole_contact': True,
                                      'feet_flat_stance': True,
                                      'reward_weights': {'slip': 5., 'gait': 2., 'swing': 10., 'feet_flat': 2.}},
                                     # Swing clearance without sole_contact: the sole height goes to the reward for
                                     # this term alone, contact stays by the ankle_roll origin. The feet settle at
                                     # 4-17 mm/s: clearance -0.008 / -0.019 / -0.024 on CPU at weight 10.
                                     {'real_robot': True, 'action_limit': 10., 'heading_hold': True,
                                      'gait': {'period': .6, 'stance_fraction': .55},
                                      'reward_weights': {'clearance': 10., 'slip': 5., 'gait': 2., 'swing': 10.}},
                                     # Yaw drift: 0.4 rad off the held heading the steered command is -0.2 rad/s, so
                                     # the mean error grows from zero by about 0.2/60 per step (term ~ -0.05 at the
                                     # third step at weight 5).
                                     {'real_robot': True, 'action_limit': 10., 'heading_hold': True,
                                      'stand_heading_hold': False, 'gait': {'period': .6, 'stance_fraction': .55},
                                      'reward_weights': {'yaw_drift': 5., 'yaw': 2.}}])
def test_g1_cpu_gpu_short_trajectory(options, monkeypatch):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    clearance = options.get('reward_weights', {}).get('clearance')
    yaw_drift = options.get('reward_weights', {}).get('yaw_drift')
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):  # what the CUDA env hands to the shared reward
        components, fell = original(*args, **kwargs)
        seen.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    cpu=G1VelocityEnv(**options)
    gpu=TensorG1Env(1,seed=4,resample_commands=False,push_velocity=0,observation_noise=False,randomize=False,**options)
    try:
        cpu.reset(seed=4,options={'command':[.6,0,0]})
        if options:  # start 0.4 rad off the held heading so the steering command is non-zero
            cpu.data.qpos[3:7]=[np.cos(.2),0,0,np.sin(.2)]
            mujoco.mj_forward(cpu.model,cpu.data)
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(cpu.data.qpos,device=gpu.device,dtype=torch.float32)[None])
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.6,0,0]],device=gpu.device))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        np.testing.assert_allclose(gpu._obs().cpu()[0],cpu._observation(),atol=2e-5,rtol=2e-5)
        actions=cpu.action_space.shape[0]
        for _ in range(3):
            expected, reward, _, _, info = cpu.step(np.zeros(actions))
            _, actual_reward, _, _, terminal = gpu.step(torch.zeros((1,actions),device=gpu.device))
            gpu.check_health()
            np.testing.assert_allclose(terminal.cpu()[0],expected,atol=.03,rtol=.02)
            np.testing.assert_allclose(actual_reward.cpu()[0],reward,atol=.03,rtol=.02)
            kwargs, components = seen[-1]
            assert kwargs['sole_contact'] is bool(options.get('sole_contact'))
            assert kwargs['feet_flat_stance'] is bool(options.get('feet_flat_stance'))
            assert (kwargs['sole_height'] is None) == (not options.get('sole_contact') and not clearance)
            if options.get('sole_contact') or clearance:  # Warp geom_xpos read as on the CPU task
                gpu_sole=task.sole_height(gpu.geom_xpos[:,gpu.soles,2],gpu.sole_radius)
                np.testing.assert_allclose(gpu_sole.cpu().numpy(),cpu._sole_height().numpy(),atol=1e-3,rtol=0)
                np.testing.assert_allclose(kwargs['sole_height'].cpu().numpy(),gpu_sole.cpu().numpy(),atol=1e-6,rtol=0)
            if options.get('feet_flat_stance'):  # Σ sin² tilt over the feet whose Warp sole is in contact
                tilt=gpu.xmat[:,gpu.feet].reshape(1,len(gpu.feet),3,3)[...,2,:2].square().sum(dim=-1)
                contact=kwargs['sole_height']<task.SOLE_CONTACT_HEIGHT
                torch.testing.assert_close(components['feet_flat'],-2.*(tilt*contact).sum(dim=1))
                np.testing.assert_allclose(components['feet_flat'].cpu()[0],info['reward_components']['feet_flat'],
                                           atol=2e-3,rtol=0)
            if clearance:
                assert info['reward_components']['clearance'] < 0
                np.testing.assert_allclose(components['clearance'].cpu()[0],info['reward_components']['clearance'],
                                           atol=.01,rtol=.1)
            assert (kwargs['yaw_drift_error'] is None) == (not yaw_drift)
            torch.testing.assert_close(kwargs['requested_command'].cpu(),torch.tensor([[.6,0,0]]))
            if yaw_drift:  # the same mean as on the CPU task, from Warp's pelvis velocity
                torch.testing.assert_close(kwargs['yaw_drift_error'],gpu.yaw_drift_error)
                np.testing.assert_allclose(gpu.yaw_drift_error.cpu().numpy(),cpu.yaw_drift_error.numpy(),
                                           atol=2e-3,rtol=0)
                assert info['reward_components']['yaw_drift'] < 0
                np.testing.assert_allclose(components['yaw_drift'].cpu()[0],info['reward_components']['yaw_drift'],
                                           atol=.01,rtol=.1)
    finally:
        cpu.close();gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_feet_flat_stance(monkeypatch):
    """Two worlds with feet_flat_stance and sole_contact, each against the CPU task with the same setting, from a
    kinematic pose with the left foot toe up: world 0 lifted 5 cm (toe up 34°, v0's mean swing roll), both feet in
    the air for the three steps, so the term is 0 while the left foot is still tilted 27…15° (sin² 0.20 / 0.12 / 0.07
    on CPU); world 1 on its left heel (toe up 50°, the ankle_roll origin 6 cm up), the right foot hanging 19 / 15 /
    9 mm, charged as before (-0.92 / -0.60 / -0.35 on CPU at weight 2). The CUDA term is the shared formula on its own
    Warp sole contact."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., sole_contact=True, feet_flat_stance=True,
                   reward_weights={'feet_flat': 2.})
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        assert gpu.reference.feet_flat_stance is True
        for cpu, (toe_up, lift) in zip(cpus, ((np.radians(34), .05), (np.radians(50), 0.))):
            cpu.reset(seed=0, options={'command': [1., 0., 0.]})
            names = [cpu.model.joint(j).name for j in cpu.joints]
            cpu.data.qpos[cpu.qadr] = cpu.home
            cpu.data.qpos[cpu.qadr[names.index('left_ankle_pitch_joint')]] -= toe_up
            cpu.data.qvel[:] = 0
            mujoco.mj_kinematics(cpu.model, cpu.data)
            cpu.data.qpos[2] += lift - .0005 - float(cpu._sole_height().min())  # lowest sole 0.5 mm in the floor
            mujoco.mj_forward(cpu.model, cpu.data)
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[1., 0, 0]], device=gpu.device).expand(2, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()  # refreshes the observed (rewarded) command
        for step in range(3):
            results = [cpu.step(np.zeros(29)) for cpu in cpus]
            gpu.step(torch.zeros((2, 29), device=gpu.device))
            gpu.check_health()
            kwargs, components = seen[-1]
            assert kwargs['feet_flat_stance'] is True and kwargs['sole_contact'] is True
            sole = task.sole_height(gpu.geom_xpos[:, gpu.soles, 2], gpu.sole_radius)
            torch.testing.assert_close(kwargs['sole_height'], sole)
            tilt = gpu.xmat[:, gpu.feet].reshape(2, len(gpu.feet), 3, 3)[..., 2, :2].square().sum(dim=-1)
            contact = sole < task.SOLE_CONTACT_HEIGHT
            torch.testing.assert_close(components['feet_flat'], -2. * (tilt * contact).sum(dim=1))
            # World 0: both feet in the air on both backends, the tilted left foot spared.
            assert not bool(contact[0].any()) and float(tilt[0, 0]) > .03
            assert bool((cpus[0]._sole_height() > task.SOLE_CONTACT_HEIGHT).all())
            assert float(components['feet_flat'][0]) == 0 and results[0][-1]['reward_components']['feet_flat'] == 0
            # World 1: the heel stance is a contact on both backends and costs what it did without the flag.
            assert bool(contact[1, 0]) and bool(cpus[1]._sole_height()[0, 0] < task.SOLE_CONTACT_HEIGHT)
            assert float(components['feet_flat'][1]) < -.2
            np.testing.assert_allclose(float(components['feet_flat'][1]),
                                       results[1][-1]['reward_components']['feet_flat'], atol=.05, rtol=.05)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
@pytest.mark.parametrize('delay', (5, 15))
def test_g1_cpu_gpu_short_trajectory_with_action_delay(delay):
    """Two worlds in one batch, no delay and `delay` substeps (half a step / one and a half steps), each against
    the CPU task with the same fixed delay, under changing actions."""
    import warp as wp
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55})
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      action_delay=.03, **profile)
    try:
        assert gpu.reference.max_delay_steps == 15 and gpu.delay_depth == 2
        for cpu, steps in zip(cpus, (0, delay)):
            cpu.reset(seed=4, options={'command': [.6, 0, 0], 'action_delay_steps': steps})
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, delay]))
        torch.testing.assert_close(gpu.delay_lags.cpu(), torch.from_numpy(np.stack([cpus[0].delay_lags,
                                                                                      cpus[1].delay_lags])))
        gpu.qpos.copy_(torch.tensor(np.stack([cpus[0].data.qpos, cpus[1].data.qpos]), device=gpu.device,
                                    dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.6, 0, 0]], device=gpu.device).expand(2, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()  # also refreshes the observed (rewarded) command
        for world in range(2):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        joints = cpus[0].action_space.shape[0]
        for step in range(4):
            action = .4 * np.sin(np.arange(joints) + 1.7 * step)
            results = [cpu.step(action) for cpu in cpus]  # world 0: no delay, world 1: `delay` substeps
            _, reward, _, _, terminal = gpu.step(torch.tensor(action, device=gpu.device,
                                                              dtype=torch.float32)[None].expand(2, -1).contiguous())
            gpu.check_health()
            for world in range(2):
                expected, expected_reward = results[world][:2]
                np.testing.assert_allclose(terminal.cpu()[world], expected, atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
            torch.testing.assert_close(gpu.delay_actions[:, 0].cpu(), torch.tensor(action, dtype=torch.float32)
                                       .expand(2, -1))
        # The delay matters far beyond the CPU-GPU tolerance: the delayed CPU robot is not where the undelayed one is.
        assert float(np.abs(cpus[1].data.qvel - cpus[0].data.qvel).max()) > .5
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_action_delay_differs_between_worlds_and_redraws_at_reset():
    from humanoid_sim.tasks import g1_tensor_math as task
    env = TensorG1Env(256, seed=3, max_episode_steps=2, resample_commands=False, push_velocity=0, real_robot=True,
                      action_limit=10., observation_noise=False, randomize=False, action_delay=.02)
    try:
        assert env.reference.max_delay_steps == 10 and env.delay_depth == 1
        env.reset()
        first = env.delay_steps.clone()
        assert int(first.min()) >= 0 and int(first.max()) <= 10 and len(first.unique()) > 7
        assert torch.equal(env.delay_lags, task.action_lags(first, env.reference.frame_skip))
        action = torch.full((256, 29), .3, device=env.device)
        env.step(action)
        torch.testing.assert_close(env.delay_actions[:, 0], action)  # the motors may still follow it next step
        assert torch.equal(env.delay_steps, first)  # fixed within an episode
        _, _, _, truncated, _ = env.step(action)
        assert truncated.all()
        assert float(env.delay_actions.abs().max()) == 0  # a new episode starts from the home pose
        assert int(env.delay_steps.max()) <= 10 and float((env.delay_steps != first).float().mean()) > .5
        assert torch.equal(env.delay_lags, task.action_lags(env.delay_steps, env.reference.frame_skip))
    finally:
        env.close()
    plain = TensorG1Env(2, seed=3)
    try:
        assert plain.delay_depth == 0 and plain.delay_actions.shape == (2, 0, 29)
        plain.reset()
        assert int(plain.delay_steps.abs().max()) == 0
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_random_phase_starts_worlds_anywhere_in_the_cycle_and_redraws_at_reset():
    from humanoid_sim.tasks import g1_tensor_math as task
    env = TensorG1Env(256, seed=3, max_episode_steps=2, resample_commands=False, push_velocity=0, real_robot=True,
                      action_limit=10., observation_noise=False, randomize=False, random_phase=True)
    try:
        assert env.period_steps == round(env.reference.gait['period'] / env.dt)
        env.reset()
        first = env.phase_offset.clone()
        assert int(first.min()) >= 0 and int(first.max()) < env.period_steps and len(first.unique()) > env.period_steps // 2
        # The observation and the gait reward read one clock: episode length plus the world's offset.
        torch.testing.assert_close(env._phase(), task.gait_phase(env.lengths + first, env.dt, env.reference.gait['period']))
        action = torch.zeros((256, 29), device=env.device)
        env.step(action)
        assert torch.equal(env.phase_offset, first)  # fixed within an episode
        _, _, _, truncated, _ = env.step(action)
        assert truncated.all() and float((env.phase_offset != first).float().mean()) > .5  # redrawn at the reset
    finally:
        env.close()
    plain = TensorG1Env(2, seed=3)
    try:
        plain.reset()
        assert int(plain.phase_offset.abs().max()) == 0  # off: every episode starts at phase 0
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_arm_perturb_runs_segments_per_world_and_ends_them_at_reset(monkeypatch):
    """1024 worlds at the working share 0.1, 1000-step episodes with falls switched off (a fallen robot lies on; the
    segments do not depend on it): each world runs its own segments of 25-100 steps holding home or a random arm pose,
    the observed previous action of the arms is the held one and the rest stays the policy's, the share of world-steps
    is 0.098 (an episode starts outside a segment) and is logged; the reset ends every segment."""
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'FALL_HEIGHT', -1e9)
    monkeypatch.setattr(task, 'FALL_UPRIGHT', -2.)
    n = 1024
    env = TensorG1Env(n, seed=3, max_episode_steps=1000, resample_commands=False, push_velocity=0, real_robot=True,
                      action_limit=10., observation_noise=False, randomize=False, arm_perturb=.1)
    try:
        arms = env.arm_mask
        assert env.perturb_arms and arms.nonzero().flatten().tolist() == list(range(15, 29))
        env.reset()
        assert int(env.perturb_steps.abs().max()) == 0
        policy = torch.full((n, 29), .05, device=env.device)
        shortest, longest = task.arm_perturb_lengths(env.dt)
        bound = (task.ARM_PERTURB_RADIANS / env.scale)[arms]
        perturbed, starts, homes, busy = 0, 0, 0, []
        for _ in range(999):
            before, kept = env.perturb_steps.clone(), env.perturb_action.clone()
            env.step(policy)
            on = (env.previous_action[:, arms] != .05).any(dim=1)  # the arms followed a segment
            assert torch.equal(on, (before > 0) | (env.perturb_steps > 0))
            assert bool((env.previous_action[:, ~arms] == .05).all())  # legs and waist: always the policy's
            assert torch.equal(env.previous_action[on][:, arms], env.perturb_action[on][:, arms])
            running = before > 0  # one held action per segment, one step off its length per step
            assert torch.equal(env.perturb_steps[running], before[running] - 1)
            assert torch.equal(env.perturb_action[running], kept[running])
            started = on & ~running
            length = env.perturb_steps[started] + 1
            assert bool(((length >= shortest) & (length <= longest)).all())
            held = env.perturb_action[started][:, arms]
            home = (held == 0).all(dim=1)
            assert bool((held[~home].abs() <= bound * (1 + 1e-6)).all())
            starts, homes = starts + int(started.sum()), homes + int(home.sum())
            busy.append(int(started.sum()))
            perturbed += int(on.sum())
        env.check_health()
        share = perturbed / (n * 999)
        assert share == pytest.approx(.098, abs=.01)
        assert env.rollout_metrics()['rollout/arm_perturb'] == pytest.approx(share, rel=1e-6)
        assert math.isnan(env.rollout_metrics()['rollout/arm_perturb'])  # cleared by each read
        assert starts > 1000 and homes / starts == pytest.approx(task.ARM_PERTURB_HOME, abs=.06)
        assert max(busy) < 20 and sum(k > 0 for k in busy) > 500  # independent worlds: starts spread over steps
        _, _, terminated, truncated, _ = env.step(policy)
        assert truncated.all() and not terminated.any()
        assert int(env.perturb_steps.abs().max()) == 0 and float(env.perturb_action.abs().max()) == 0
        assert float(env.previous_action.abs().max()) == 0
    finally:
        env.close()
    plain, zero = TensorG1Env(2, seed=3), TensorG1Env(2, seed=3, arm_perturb=0.)  # off: nothing drawn or logged
    try:
        for env in (plain, zero):
            env.reset()
            env.step(torch.full((2, 29), .3, device=env.device))
        assert not zero.perturb_arms and int(zero.perturb_steps.abs().max()) == 0
        assert torch.equal(zero.generator.get_state(), plain.generator.get_state())
        assert float((zero.previous_action - .3).abs().max()) == 0 and zero.rollout_metrics() == {}
    finally:
        plain.close(), zero.close()
    legs = TensorG1Env(2, seed=3, legs_only=True, arm_perturb=.5)  # no arms: segments run with nothing to hold
    try:
        assert not bool(legs.arm_mask.any())
        legs.reset()
        legs.step(torch.full((2, 12), .3, device=legs.device))
        assert float((legs.previous_action - .3).abs().max()) == 0
    finally:
        legs.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_short_trajectory_with_arm_perturb(monkeypatch):
    """Two worlds with segments set by hand, each against the CPU task with the same segment: world 0 holds the arms
    at home for 3 steps, world 1 a random arm pose for 3 steps with the motors 15 substeps late, so its delay queue
    still holds the held actions after the segment. The applied action (policy legs and waist, held arms) is what
    the PD, the queue, the reward and the observed previous action get on both backends."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((args, kwargs))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    arms = list(range(15, 29))
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, action_delay=.03,
                   reward_weights={'arm_press': .1, 'action_change': .05})
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        held = np.zeros((2, 29))
        held[1, arms] = np.random.default_rng(9).uniform(-.8, .8, len(arms))
        for cpu, steps, pose in zip(cpus, (0, 15), held):
            cpu.reset(seed=4, options={'command': [.6, 0, 0], 'action_delay_steps': steps})
            cpu.set_arm_perturb(3, pose)
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, 15]))
        gpu.set_arm_perturb(torch.tensor([3, 3]), torch.tensor(held, dtype=torch.float32))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.6, 0, 0]], device=gpu.device).expand(2, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()  # also refreshes the observed (rewarded) command
        for world in range(2):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        for step in range(5):
            action = .4 * np.sin(np.arange(29) + 1.7 * step)
            applied = np.stack([action, action])
            if step < 3:
                applied[:, arms] = held[:, arms]
            results = [cpu.step(action) for cpu in cpus]
            _, reward, _, _, terminal = gpu.step(torch.tensor(action, device=gpu.device,
                                                              dtype=torch.float32)[None].expand(2, -1).contiguous())
            gpu.check_health()
            args, kwargs = seen_gpu[-1]
            expected = torch.tensor(applied, dtype=torch.float32)
            torch.testing.assert_close(args[5].cpu(), expected)  # the action the reward reads (action_change)
            torch.testing.assert_close(gpu.previous_action.cpu(), expected)
            torch.testing.assert_close(gpu.delay_actions[:, 0].cpu(), expected)  # the queue the motors follow
            target = task.pd_target(args[5], gpu.home, gpu.scale, gpu.joint_low, gpu.joint_high, 10.)
            torch.testing.assert_close(kwargs['arm_press'], task.arm_press(target, gpu.qpos[:, gpu.qadr],
                                                                           gpu.arm_joints))
            for world in range(2):
                np.testing.assert_allclose(cpus[world].previous_action, applied[world], rtol=0, atol=0)
                expected_obs, expected_reward, *_, info = results[world]
                np.testing.assert_allclose(terminal.cpu()[world], expected_obs, atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
                assert float(kwargs['arm_press'][world]) == pytest.approx(
                    -info['reward_components']['arm_press'] / .1, rel=.1, abs=.02)
        assert int(gpu.perturb_steps.abs().max()) == 0 and all(cpu.perturb_steps == 0 for cpu in cpus)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


WRISTS = [20, 21, 27, 28]  # left/right wrist pitch and yaw, the joints lock_wrists holds at home


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_short_trajectory_with_lock_wrists(monkeypatch):
    """Two worlds with lock_wrists, each against the CPU task with the same setting, under actions that ask the wrists
    for ±8 (±32°): world 0 plain, world 1 in a 3-step arm perturbation segment whose held pose bends the wrists, with
    the motors 15 substeps late. The applied action (the policy's, held arms in the segment, wrist pitch and yaw at 0)
    is what the PD, the queue, the reward and the observed previous action get on both backends."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((args, kwargs))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    arms = list(range(15, 29))
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, action_delay=.03,
                   reward_weights={'arm_press': .1, 'action_change': .05}, lock_wrists=True)
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        assert gpu.lock_wrists and gpu.wrist_mask.nonzero().flatten().tolist() == WRISTS
        held = np.zeros((2, 29))
        held[1, arms] = np.random.default_rng(9).uniform(-.8, .8, len(arms))  # wrists bent too
        for cpu, steps, pose in zip(cpus, (0, 15), held):
            cpu.reset(seed=4, options={'command': [.6, 0, 0], 'action_delay_steps': steps})
            cpu.set_arm_perturb(3, pose)
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, 15]))
        gpu.set_arm_perturb(torch.tensor([3, 3]), torch.tensor(held, dtype=torch.float32))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.6, 0, 0]], device=gpu.device).expand(2, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()  # also refreshes the observed (rewarded) command
        for world in range(2):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        for step in range(5):
            action = .4 * np.sin(np.arange(29) + 1.7 * step)
            action[WRISTS] = 8 * np.sign(np.sin(np.arange(4) + step) + .1)
            applied = np.stack([action, action])
            if step < 3:
                applied[:, arms] = held[:, arms]
            applied[:, WRISTS] = 0.
            results = [cpu.step(action) for cpu in cpus]
            _, reward, _, _, terminal = gpu.step(torch.tensor(action, device=gpu.device,
                                                              dtype=torch.float32)[None].expand(2, -1).contiguous())
            gpu.check_health()
            args, kwargs = seen_gpu[-1]
            expected = torch.tensor(applied, dtype=torch.float32)
            torch.testing.assert_close(args[5].cpu(), expected)  # the action the reward reads (action_change)
            torch.testing.assert_close(gpu.previous_action.cpu(), expected)
            torch.testing.assert_close(gpu.delay_actions[:, 0].cpu(), expected)  # the queue the motors follow
            target = task.pd_target(args[5], gpu.home, gpu.scale, gpu.joint_low, gpu.joint_high, 10.)
            torch.testing.assert_close(target[:, WRISTS], gpu.home[WRISTS].expand(2, -1), rtol=0, atol=0)
            torch.testing.assert_close(kwargs['arm_press'], task.arm_press(target, gpu.qpos[:, gpu.qadr],
                                                                           gpu.arm_joints))
            assert float(terminal[:, -29:][:, WRISTS].abs().max()) == 0  # observed previous action
            for world in range(2):
                np.testing.assert_allclose(cpus[world].previous_action, applied[world], rtol=0, atol=0)
                expected_obs, expected_reward, *_, info = results[world]
                np.testing.assert_allclose(terminal.cpu()[world], expected_obs, atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
                assert float(kwargs['arm_press'][world]) == pytest.approx(
                    -info['reward_components']['arm_press'] / .1, rel=.1, abs=.02)
        # The wrists held home on both backends (the unlocked targets would be up to ±0.56 rad off).
        for world in range(2):
            wrist = gpu.qpos[world, gpu.qadr][WRISTS].cpu().numpy() - gpu.home[WRISTS].cpu().numpy()
            assert float(np.abs(wrist).max()) < .05
            np.testing.assert_allclose(cpus[world].data.qpos[cpus[world].qadr][WRISTS] - cpus[world].home[WRISTS],
                                       wrist, atol=.02, rtol=0)  # CPU 0.000-0.008 rad
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_lock_wrists_holds_every_world_and_draws_nothing(monkeypatch):
    """256 worlds in training (resets without a command, arm perturbation at share 0.5, random delay, 5-frame
    history), policy wrist actions uniform in the whole ±10 box: in every world the applied wrist pitch and yaw are 0 in
    the observed previous action (all history frames), the delay queue and the PD target, the legs and the waist
    follow the policy. The lock draws nothing: the random stream is the one without it. Off, the wrists follow the
    policy. The legs-only G1 refuses it."""
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'FALL_HEIGHT', -1e9)  # falls switched off: the lock does not depend on them
    monkeypatch.setattr(task, 'FALL_UPRIGHT', -2.)
    n = 256
    profile = dict(max_episode_steps=1000, resample_commands=False, push_velocity=0, real_robot=True, action_limit=10.,
                   history=5, phase_stop=True, observation_noise=False, randomize=False, arm_perturb=.5,
                   action_delay=.02)
    locked, free = TensorG1Env(n, seed=3, lock_wrists=True, **profile), TensorG1Env(n, seed=3, **profile)
    try:
        start, end = task.deploy_terms(29)[-1]  # last_action, 5 frames
        policy_rng = torch.Generator(device=locked.device).manual_seed(11)
        for env in (locked, free):
            env.reset()
        for _ in range(60):
            # ±1 on the body (no thrashing on the floor), the whole ±10 box on the wrists.
            policy = 2 * torch.rand((n, 29), device=locked.device, generator=policy_rng) - 1
            policy[:, WRISTS] *= 10
            before = free.perturb_steps.clone()
            obs, *_ = locked.step(policy)
            free.step(policy)
            # Worlds whose arms no segment held this step (one that ended now has 0 left too): free wrists = policy.
            outside = (before == 0) & (free.perturb_steps == 0)
            torch.testing.assert_close(free.previous_action[outside][:, WRISTS], policy[outside][:, WRISTS], rtol=0,
                                       atol=0)
            assert float(locked.previous_action[:, WRISTS].abs().max()) == 0
            assert float(obs[:, 5 * start:5 * end].reshape(n, 5, 29)[:, :, WRISTS].abs().max()) == 0
            assert float(locked.delay_actions[:, :, WRISTS].abs().max()) == 0
            target = task.pd_target(locked.previous_action, locked.home, locked.scale, locked.joint_low,
                                    locked.joint_high, 10.)
            assert bool((target[:, WRISTS] == locked.home[WRISTS]).all())
            torch.testing.assert_close(locked.previous_action[:, :15], policy[:, :15], rtol=0, atol=0)
            torch.testing.assert_close(free.previous_action[:, :15], policy[:, :15], rtol=0, atol=0)
            assert torch.equal(locked.generator.get_state(), free.generator.get_state())
            assert torch.equal(locked.perturb_steps, free.perturb_steps)  # the same segments, wrists aside
        assert float(free.previous_action[:, WRISTS].abs().max()) > 5
        locked.check_health()
        free.check_health()
    finally:
        locked.close(), free.close()
    with pytest.raises(ValueError):
        TensorG1Env(2, seed=3, legs_only=True, lock_wrists=True)


KNEES = [3, 9]  # left/right knee, the joints knee_floor bounds
KNEE_FLOOR = .061


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_short_trajectory_with_knee_floor(monkeypatch):
    """Two worlds with knee_floor 0.061, each against the CPU task with the same setting, under actions that ask both
    knees for the bottom of the box (target −3.2 rad): world 0 without delay, world 1 with the motors 15 substeps
    late. The knee PD target is the floor on both backends (the bounds pd_torque and arm_press use), while the
    reward's action, the observed previous action and the delay queue keep the policy's knee action, as g1_ctrl's
    last_action (the raw ONNX output)."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((args, kwargs))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, action_delay=.03,
                   reward_weights={'arm_press': .1, 'action_change': .05}, knee_floor=KNEE_FLOOR)
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        others = [i for i in range(29) if i not in KNEES]
        low = torch.tensor(cpus[0].model.jnt_range[cpus[0].joints, 0], dtype=torch.float32)
        torch.testing.assert_close(gpu.joint_low[KNEES].cpu(), torch.full((2,), KNEE_FLOOR), rtol=0, atol=0)
        torch.testing.assert_close(gpu.joint_low[others].cpu(), low[others], rtol=0, atol=0)
        for cpu, steps in zip(cpus, (0, 15)):
            cpu.reset(seed=4, options={'command': [0., 0., 0.], 'action_delay_steps': steps})
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, 15]))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()  # also refreshes the observed (rewarded) command
        for world in range(2):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        for step in range(8):
            action = .3 * np.sin(np.arange(29) + 1.7 * step)
            action[KNEES] = -10.
            results = [cpu.step(action) for cpu in cpus]
            _, reward, _, _, terminal = gpu.step(torch.tensor(action, device=gpu.device,
                                                              dtype=torch.float32)[None].expand(2, -1).contiguous())
            gpu.check_health()
            args, kwargs = seen_gpu[-1]
            expected = torch.tensor(action, dtype=torch.float32)[None].expand(2, -1)
            torch.testing.assert_close(args[5].cpu(), expected)  # the action the reward reads: knees not raised
            torch.testing.assert_close(gpu.previous_action.cpu(), expected)
            torch.testing.assert_close(gpu.delay_actions[:, 0].cpu(), expected)  # the queue the motors follow
            target = task.pd_target(args[5], gpu.home, gpu.scale, gpu.joint_low, gpu.joint_high, 10.)
            assert bool((target[:, KNEES] == gpu.joint_low[KNEES]).all())  # the floor, whatever the action asks
            torch.testing.assert_close(kwargs['arm_press'], task.arm_press(target, gpu.qpos[:, gpu.qadr],
                                                                           gpu.arm_joints))
            assert float(terminal[:, -29:][:, KNEES].max()) == -10.  # the observed previous action
            for world in range(2):
                np.testing.assert_array_equal(cpus[world].previous_action[KNEES], action[KNEES])
                assert (cpus[world]._target(cpus[world].previous_action)[KNEES] == KNEE_FLOOR).all()
                expected_obs, expected_reward, *_, info = results[world]
                np.testing.assert_allclose(terminal.cpu()[world], expected_obs, atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
                assert float(kwargs['arm_press'][world]) == pytest.approx(
                    -info['reward_components']['arm_press'] / .1, rel=.1, abs=.02)
        # The knees stay out of the stop on both backends (without the floor they reach −0.087).
        for world in range(2):
            knee = gpu.qpos[world, gpu.qadr][KNEES].cpu().numpy()
            assert float(knee.min()) > .03
            np.testing.assert_allclose(cpus[world].data.qpos[cpus[world].qadr][KNEES], knee, atol=.02, rtol=0)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_knee_floor_holds_every_world_and_changes_no_action(monkeypatch):
    """256 worlds in training (resets without a command, arm perturbation at share 0.5, random delay, 5-frame
    history), knee actions down to −3 (targets to −0.75 rad): with knee_floor every PD target of every substep, the
    delayed ones included, has its knees at the floor or above, while without it they go below; the actions (observed
    previous action, all history frames, delay queue, perturbation segments) and the random stream are exactly those
    without it: only the physics differs. Both G1 models take it; a floor outside the knee range is refused."""
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'FALL_HEIGHT', -1e9)  # falls switched off: the floor does not depend on them
    monkeypatch.setattr(task, 'FALL_UPRIGHT', -2.)
    n = 256
    profile = dict(max_episode_steps=1000, resample_commands=False, push_velocity=0, real_robot=True, action_limit=10.,
                   history=5, phase_stop=True, observation_noise=False, randomize=False, arm_perturb=.5,
                   action_delay=.02)
    floored, free = TensorG1Env(n, seed=3, knee_floor=KNEE_FLOOR, **profile), TensorG1Env(n, seed=3, **profile)
    lowest, original = {'floored': [], 'free': []}, tensor_env.pd_torque

    def pd_torque(action, position, velocity, home, scale, kp, kd, joint_low, joint_high, *rest):
        name = 'floored' if joint_low is floored.joint_low else 'free'
        target = task.pd_target(action, home, scale, joint_low, joint_high, rest[-1])
        lowest[name].append(float(target[:, KNEES].min()))
        return original(action, position, velocity, home, scale, kp, kd, joint_low, joint_high, *rest)

    monkeypatch.setattr(tensor_env, 'pd_torque', pd_torque)
    try:
        start, end = task.deploy_terms(29)[-1]  # last_action, 5 frames
        policy_rng = torch.Generator(device=floored.device).manual_seed(11)
        for env in (floored, free):
            env.reset()
        for _ in range(40):
            policy = 2 * torch.rand((n, 29), device=floored.device, generator=policy_rng) - 1
            policy[:, KNEES] = 4 * torch.rand((n, 2), device=floored.device, generator=policy_rng) - 3
            obs, *_ = floored.step(policy)
            free_obs, *_ = free.step(policy)
            assert torch.equal(floored.previous_action, free.previous_action)
            torch.testing.assert_close(floored.previous_action[:, :15], policy[:, :15], rtol=0, atol=0)
            assert torch.equal(obs[:, 5 * start:5 * end], free_obs[:, 5 * start:5 * end])  # every history frame
            assert torch.equal(floored.delay_actions, free.delay_actions)
            assert torch.equal(floored.perturb_steps, free.perturb_steps)
            assert torch.equal(floored.generator.get_state(), free.generator.get_state())
        assert len(lowest['floored']) == len(lowest['free']) == 40 * 10
        assert min(lowest['floored']) >= np.float32(KNEE_FLOOR) and min(lowest['free']) < 0
        floored.check_health()
        free.check_health()
    finally:
        floored.close(), free.close()
    legs = TensorG1Env(2, seed=3, legs_only=True, knee_floor=KNEE_FLOOR)
    try:
        torch.testing.assert_close(legs.joint_low[KNEES].cpu(), torch.full((2,), KNEE_FLOOR), rtol=0, atol=0)
    finally:
        legs.close()
    with pytest.raises(ValueError):
        TensorG1Env(2, seed=3, knee_floor=3.)


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_self_contact(monkeypatch):
    """World 0 presses the upper arms on the torso, world 1 stays in the home pose (feet on the floor only), each
    against the CPU task: the same largest robot-robot contact force per substep (Warp's contact forces), the same
    share above 10 N, the same reward. The press starts in the second step (substep 6 on both: 91 / 94 N)."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_self_contact import _record, arms_pressed
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55},
                   reward_weights={'self_contact': 4.})
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    seen_cpu = [_record(cpu) for cpu in cpus]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        for cpu in cpus:
            cpu.reset(seed=4, options={'command': [0., 0., 0.]})
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device,
                                    dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()  # refreshes the observed (rewarded) command
        actions = np.stack([arms_pressed(), np.zeros(29)])
        shares = []
        for step in range(8):
            results = [cpu.step(action) for cpu, action in zip(cpus, actions)]
            _, reward, _, _, _ = gpu.step(torch.tensor(actions, device=gpu.device, dtype=torch.float32))
            gpu.check_health()
            force = gpu.self_contact_force.cpu().numpy()
            kwargs, components = seen_gpu[-1]
            torch.testing.assert_close(kwargs['self_contact'], task.self_contact_share(gpu.self_contact_force))
            for world in range(2):
                expected, expected_reward, *_, info = results[world]
                cpu_force = np.array(seen_cpu[world][-10:])
                share = float(np.mean(cpu_force > task.SELF_CONTACT_FORCE))
                assert float(kwargs['self_contact'][world]) == share  # no substep near the threshold here
                assert info['reward_components']['self_contact'] == -4 * share
                assert float(components['self_contact'][world]) == -4 * share
                if share:  # contact forces chatter from substep to substep; their mean agrees
                    assert np.mean(force[world][cpu_force > 0]) == pytest.approx(np.mean(cpu_force[cpu_force > 0]),
                                                                                 rel=.3)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
            assert float(force[1].max()) == 0  # home: the feet press on the floor, which never counts
            shares.append(float(kwargs['self_contact'][0]))
        assert shares[0] == 0 and shares[1] == .5 and shares[2:] == [1.] * 6
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/self_contact'] == pytest.approx(np.mean(shares) / 2, rel=1e-6)
        assert math.isnan(gpu.rollout_metrics()['rollout/self_contact'])  # cleared by each read
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    plain = TensorG1Env(2, seed=4, real_robot=True, action_limit=10.)  # off: nothing read, nothing logged
    try:
        assert plain.self_contact_force is None
        plain.reset()
        plain.step(torch.zeros((2, 29), device=plain.device))
        assert seen_gpu[-1][0]['self_contact'] is None and plain.rollout_metrics() == {}
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_foot_collision(monkeypatch):
    """foot_collision on the CUDA task: Warp gets the CPU task's model (Unitree's foot capsules with contype and
    conaffinity 2 and Unitree's contact, condim 3, friction 0.6, priority 1; the same sole spheres) and the default
    contact buffers hold. World 0 drives its feet into each other in the air (test_g1_foot_collision.cross_in_the_air:
    on the CPU task ~27 N, self_contact from the 4th step), world 1 stands in the home pose; each against the CPU
    task: the capsules touch only each other, above 10 N, self_contact charges the same steps, the same reward;
    standing, nothing touches them. The foot-foot forces are the robot-robot ones here (no other contact), and
    rollout/foot_contact is the share of world-steps with one above 10 N. Without the setting the same crossing passes
    through on both backends (no contact, no charge, no foot_contact)."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_foot_collision import crossed
    from test_g1_self_contact import _record
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55},
                   reward_weights={'self_contact': 4.})
    for flag in (True, False):
        cpus = [G1VelocityEnv(foot_collision=flag, **profile) for _ in range(2)]
        seen_cpu = [_record(cpu) for cpu in cpus]
        gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False,
                          randomize=False, foot_collision=flag, **profile)
        try:
            ref = cpus[0]
            assert gpu.model.ngeom == ref.model.ngeom == 74 + (14 if flag else 0)
            for name in ('geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_priority'):
                np.testing.assert_array_equal(wp.to_torch(getattr(gpu.model, name)).cpu().numpy(),
                                              getattr(ref.model, name), err_msg=name)
            np.testing.assert_allclose(wp.to_torch(gpu.model.geom_friction).cpu().numpy()[0], ref.model.geom_friction,
                                       rtol=1e-6)
            assert (gpu.foot_contact_force is None) != flag
            assert gpu.soles.cpu().tolist() == ref.soles.tolist() and gpu.reference.foot_capsules.shape == (
                2, 7 if flag else 0)
            capsules = set(ref.foot_capsules.ravel().tolist())
            left, right = (set(foot.tolist()) for foot in ref.foot_capsules)
            for cpu in cpus:
                cpu.reset(seed=4, options={'command': [0., 0., 0.]})
            cpus[0].data.qpos[cpus[0].qadr] = cpus[0].home + crossed(cpus[0], .12)
            cpus[0].data.qpos[2] = .9  # in the air
            mujoco.mj_forward(cpus[0].model, cpus[0].data)
            gpu.reset()
            gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device,
                                        dtype=torch.float32))
            gpu.qvel.zero_()
            gpu.command.zero_()
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
                wp.capture_launch(gpu.forward_graph)
            gpu._obs()  # refreshes the observed (rewarded) command
            actions = np.stack([crossed(ref, .25) / ref.scale, np.zeros(29)])
            shares, touched, stepped = {'cpu': [], 'gpu': []}, [], []
            gpu.rollout_metrics()  # from here on
            for step in range(6):
                results = [cpu.step(action) for cpu, action in zip(cpus, actions)]
                _, reward, _, _, _ = gpu.step(torch.tensor(actions, device=gpu.device, dtype=torch.float32))
                gpu.check_health()  # no contact or constraint buffer overflow
                kwargs, components = seen_gpu[-1]
                force = gpu.self_contact_force.cpu().numpy()
                if flag:  # the only robot-robot contacts here are the capsules'
                    feet = gpu.foot_contact_force.cpu().numpy()
                    np.testing.assert_array_equal(feet, force)
                    stepped.append(task.contact_step(gpu.foot_contact_force).float().mean().item())
                count = int(gpu.nacon.item())  # the contacts of the final state, all worlds
                geoms = gpu.contact_geom[:count].long().cpu().numpy()
                worlds = gpu.contact_world[:count].long().cpu().numpy()
                pairs = [tuple(int(g) for g in pair) for pair, world in zip(geoms, worlds) if capsules & set(pair)]
                assert all((a in left and b in right) or (a in right and b in left) for a, b in pairs)
                assert all(world == 0 for pair, world in zip(geoms, worlds) if capsules & set(pair.tolist()))
                touched += pairs
                for world in range(2):
                    expected_reward, info = results[world][1], results[world][-1]
                    cpu_force = np.array(seen_cpu[world][-10:])
                    if world == 0:
                        shares['cpu'].append(-info['reward_components']['self_contact'] / 4)
                        shares['gpu'].append(float(kwargs['self_contact'][0]))
                        if cpu_force.max() > 0 and force[0].max() > 0:  # contact forces chatter; their mean agrees
                            assert np.mean(force[0][force[0] > 0]) == pytest.approx(np.mean(cpu_force[cpu_force > 0]),
                                                                                    rel=.5)
                    else:  # standing: nothing robot-robot, the floor never counts
                        assert float(force[1].max()) == 0 and cpu_force.max() == 0
                        assert float(components['self_contact'][1]) == 0
                    # The rest of the reward (the onset substep of the contact may differ, so its term is apart).
                    rest = float(reward.cpu()[world]) - float(components['self_contact'][world])
                    assert rest == pytest.approx(expected_reward - info['reward_components']['self_contact'],
                                                 abs=.05, rel=.03)
            metrics = gpu.rollout_metrics()
            if flag:  # CPU: [0, 0, 0, 1, 1, 1]; Warp may start a substep or two apart
                assert shares['cpu'][0] == shares['gpu'][0] == 0 and min(shares['cpu'][-2:] + shares['gpu'][-2:]) >= .8
                assert abs(sum(shares['gpu']) - sum(shares['cpu'])) <= .3 and touched
                # world 1 never, world 0 from the step its contact starts: a half per such step
                assert stepped[0] == 0 and stepped[-1] == .5
                assert metrics['rollout/foot_contact'] == pytest.approx(np.mean(stepped), rel=1e-6)
                assert math.isnan(gpu.rollout_metrics()['rollout/foot_contact'])  # cleared by each read
            else:  # the feet pass through each other
                assert shares['cpu'] == shares['gpu'] == [0.] * 6 and not touched
                assert 'rollout/foot_contact' not in metrics
            np.testing.assert_allclose(gpu.qpos[0].cpu().numpy(), cpus[0].data.qpos, atol=.05, rtol=0)
        finally:
            for cpu in cpus:
                cpu.close()
            gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_hand_collision():
    """hand_collision on the CUDA task: Warp gets the CPU task's model (Unitree's hand capsules: contype and conaffinity
    1, condim 1, priority 0, MuJoCo's default friction) and, with the robot lowered until its left hand capsule is 5 mm
    into the floor (test_g1_hand_collision; a forward pass only, the legs are deep in the floor there), both backends
    find the hand-floor contact. Without the setting nothing of the hands touches the floor there."""
    import mujoco
    import warp as wp
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_hand_collision import capsule_bottom
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, foot_collision=True,
                   reward_weights={'self_contact': 4.})  # weighted: the CUDA task decodes its contacts
    drop = None
    for flag in (True, False):
        cpu = G1VelocityEnv(hand_collision=flag, **profile)
        gpu = TensorG1Env(1, seed=4, resample_commands=False, push_velocity=0, observation_noise=False,
                          randomize=False, hand_collision=flag, **profile)
        try:
            assert gpu.model.ngeom == cpu.model.ngeom == 74 + 14 + (2 if flag else 0)
            for name in ('geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_priority'):
                np.testing.assert_array_equal(wp.to_torch(getattr(gpu.model, name)).cpu().numpy(),
                                              getattr(cpu.model, name), err_msg=name)
            np.testing.assert_allclose(wp.to_torch(gpu.model.geom_friction).cpu().numpy()[0], cpu.model.geom_friction,
                                       rtol=1e-6)
            cpu.reset(seed=4, options={'command': [0., 0., 0.]})
            if drop is None:
                drop = capsule_bottom(cpu, cpu.hand_capsules[0]) + .005
            cpu.data.qpos[2] -= drop
            mujoco.mj_forward(cpu.model, cpu.data)
            gpu.reset()
            gpu.qpos.copy_(torch.tensor(cpu.data.qpos[None], device=gpu.device, dtype=torch.float32))
            gpu.qvel.zero_()
            gpu.command.zero_()
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
                wp.capture_launch(gpu.forward_graph)
            floor, hands = cpu.model.geom('floor').id, set(cpu.hand_capsules.tolist())
            count = int(gpu.nacon.item())
            pairs = {tuple(sorted(int(g) for g in pair)) for pair in gpu.contact_geom[:count].long().cpu().numpy()}
            cpu_pairs = {tuple(sorted(int(g) for g in pair)) for pair in cpu.data.contact.geom[:cpu.data.ncon]}
            on_floor = lambda found: {g for pair in found if floor in pair for g in pair} & hands
            assert on_floor(pairs) == on_floor(cpu_pairs) == hands  # both hands (the right one is 2 mm lower)
            gpu.check_health()  # no contact or constraint buffer overflow
        finally:
            cpu.close()
            gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_foot_collision_fits_the_default_contact_buffers():
    """256 worlds in training (resets without a command, force pushes, arm perturbation) under bang-bang leg actions
    that throw the feet about and cross them: the default nconmax / njmax (128 / 512 per world) never overflow with
    the 14 foot capsules (on the CPU task with condim 3: at most 20 contacts and 81 constraint rows in side steps of
    ±1 m/s, 42 and 168 in random leg poses)."""
    env = TensorG1Env(256, seed=11, real_robot=True, action_limit=10., history=5, phase_stop=True,
                      critic_extras=True, force_push=1.25, arm_perturb=.1, foot_collision=True,
                      reward_weights={'self_contact': 1.})
    try:
        env.reset()
        generator = torch.Generator(device=env.device).manual_seed(5)
        most = 0
        for _ in range(60):
            actions = (torch.rand((256, 29), device=env.device, generator=generator) * 2 - 1) * 10.
            env.step(actions)
            env.check_health()  # raises on a contact or constraint buffer overflow
            most = max(most, int(env.nacon.item()))
        assert most < 128 * 256
        metrics = env.rollout_metrics()
        assert 0 <= metrics['rollout/foot_contact'] <= 1 and 'rollout/self_contact' in metrics
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_foot_contact_is_logged_without_the_self_contact_weight(monkeypatch):
    """foot_collision with self_contact at 0: the contact forces are decoded for rollout/foot_contact only, the reward
    gets no self_contact (nothing charged, as without the setting) and rollout/self_contact is not logged."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_foot_collision import crossed
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, foot_collision=True)
    cpu = G1VelocityEnv(**profile)
    gpu = TensorG1Env(1, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        assert gpu.self_contact_force is None and gpu.foot_contact_force is not None
        cpu.reset(seed=4, options={'command': [0., 0., 0.]})
        cpu.data.qpos[cpu.qadr] = cpu.home + crossed(cpu, .12)
        cpu.data.qpos[2] = .9  # in the air
        mujoco.mj_forward(cpu.model, cpu.data)
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(cpu.data.qpos[None], device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()
        action = torch.tensor(crossed(cpu, .25) / cpu.scale, device=gpu.device, dtype=torch.float32)[None]
        for _ in range(6):
            gpu.step(action)
            gpu.check_health()
            assert seen[-1]['self_contact'] is None
        metrics = gpu.rollout_metrics()
        assert set(metrics) == {'rollout/foot_contact'} and metrics['rollout/foot_contact'] >= 2 / 6
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_self_contact_with_arm_contact_mask(monkeypatch):
    """arm_perturb_contact_mask on both backends, three worlds each against the CPU task with the same segment (set by
    hand; arm_perturb 0, so nothing else is drawn): world 0 a 6-step segment holds the upper arms pressed on the torso
    (test_g1_self_contact.arms_pressed, the policy's arms at home), world 1 the policy presses them itself (no
    segment), world 2 drives its feet into each other in the air (foot_collision's capsules) while a segment holds the
    arms at home. Worlds 0 and 1 move alike (the same applied action), but in the segment world 0's arm contacts leave
    the share (0 on both backends) while world 1's count (0, 0.5, then 1, as without the mask); the foot contacts of
    world 2 count in its segment, as on the CPU task (Warp may start them a substep or two apart). The segment over,
    world 0 pressing by itself is charged again. rollout/self_contact is the masked share."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_foot_collision import crossed
    from test_g1_self_contact import _record, arms_pressed
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55}, foot_collision=True,
                   reward_weights={'self_contact': 4.}, arm_perturb_contact_mask=True)
    cpus = [G1VelocityEnv(**profile) for _ in range(3)]
    seen_cpu = [_record(cpu) for cpu in cpus]
    gpu = TensorG1Env(3, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        assert gpu.arm_geoms is not None and torch.equal(gpu.arm_geoms.cpu(), torch.from_numpy(cpus[0].arm_geoms))
        for cpu in cpus:
            cpu.reset(seed=4, options={'command': [0., 0., 0.]})
        cpus[2].data.qpos[cpus[2].qadr] = cpus[2].home + crossed(cpus[2], .12)
        cpus[2].data.qpos[2] = .9  # in the air
        mujoco.mj_forward(cpus[2].model, cpus[2].data)
        held = np.stack([arms_pressed(), np.zeros(29), np.zeros(29)])
        cpus[0].set_arm_perturb(6, held[0])
        cpus[2].set_arm_perturb(6, held[2])
        gpu.reset()
        gpu.set_arm_perturb(torch.tensor([6, 0, 6]), torch.tensor(held, dtype=torch.float32))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()  # refreshes the observed (rewarded) command
        gpu.rollout_metrics()  # from here on
        cross = crossed(cpus[2], .25) / cpus[2].scale
        shares = {'cpu': [], 'gpu': []}
        for step in range(8):
            actions = np.stack([np.zeros(29) if step < 6 else arms_pressed(), arms_pressed(), cross])
            results = [cpu.step(action) for cpu, action in zip(cpus, actions)]
            _, reward, _, _, _ = gpu.step(torch.tensor(actions, device=gpu.device, dtype=torch.float32))
            gpu.check_health()
            in_segment = step < 6
            assert gpu.arm_contacts_masked.tolist() == [in_segment, False, in_segment]
            assert [cpu.arm_contacts_masked for cpu in cpus] == [in_segment, False, in_segment]
            kwargs, components = seen_gpu[-1]
            force = gpu.self_contact_force.cpu().numpy()
            torch.testing.assert_close(kwargs['self_contact'], task.self_contact_share(gpu.self_contact_force))
            for world in range(3 if in_segment else 2):
                expected_reward, info = results[world][1], results[world][-1]
                cpu_force = np.array(seen_cpu[world][-10:])
                cpu_share = -info['reward_components']['self_contact'] / 4
                assert cpu_share == float(np.mean(cpu_force > task.SELF_CONTACT_FORCE))
                if world < 2:  # no substep near the threshold here: the same share on both
                    assert float(kwargs['self_contact'][world]) == cpu_share
                    np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
                else:  # the onset substep of the foot contact may differ: the rest of the reward apart
                    shares['cpu'].append(cpu_share)
                    shares['gpu'].append(float(kwargs['self_contact'][2]))
                    rest = float(reward.cpu()[2]) - float(components['self_contact'][2])
                    assert rest == pytest.approx(expected_reward - info['reward_components']['self_contact'],
                                                 abs=.05, rel=.03)
            if in_segment:  # world 0 presses as world 1 does, its arm contacts left out: nothing else touches
                assert float(force[0].max()) == 0 and max(seen_cpu[0][-10:]) == 0
                np.testing.assert_allclose(gpu.qpos[0].cpu().numpy(), gpu.qpos[1].cpu().numpy(), atol=1e-4, rtol=0)
            else:  # over: world 0 pressing by itself is charged
                assert float(kwargs['self_contact'][0]) == 1.
        world_1 = [float(kwargs['self_contact'][1]) for kwargs, _ in seen_gpu[-8:]]
        assert world_1[0] == 0 and 0 < world_1[1] < 1 and world_1[2:] == [1.] * 6  # as without the mask (0.5 there)
        assert [float(kwargs['self_contact'][0]) for kwargs, _ in seen_gpu[-8:]] == [0.] * 6 + [1.] * 2
        # World 2: CPU [0, 0, 0, 1, 1, 1]; Warp may start a substep or two apart.
        assert shares['cpu'][0] == shares['gpu'][0] == 0 and min(shares['cpu'][-2:] + shares['gpu'][-2:]) >= .8
        assert abs(sum(shares['gpu']) - sum(shares['cpu'])) <= .3
        metrics = gpu.rollout_metrics()
        everything = [float(kwargs['self_contact'].sum()) for kwargs, _ in seen_gpu[-8:]]
        assert metrics['rollout/self_contact'] == pytest.approx(np.sum(everything) / 24, rel=1e-6)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    plain = TensorG1Env(2, seed=4, real_robot=True, action_limit=10., reward_weights={'self_contact': 4.})
    try:  # without the setting: no arm geoms, nothing masked (the original rule)
        assert plain.arm_geoms is None
        plain.reset()
        plain.set_arm_perturb(torch.tensor([3, 3]), torch.tensor(np.stack([arms_pressed()] * 2), dtype=torch.float32))
        plain.step(torch.zeros((2, 29), device=plain.device))
        assert not bool(plain.arm_contacts_masked.any())
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_arm_perturb_radians_scales_the_held_poses(monkeypatch):
    """1024 worlds at arm_perturb 0.5, amplitude 0.4 and 1.0, one seed, falls switched off (so no reset changes a
    segment): the same segments start at the same steps in the same worlds (the draws do not depend on the amplitude),
    each holding the same pose scaled by 2.5 within the action box (the wrist pitch and yaw joints, scale 0.07, clip at
    ±10); the poses of the wide one reach ±1.0 rad per arm joint, never beyond."""
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'FALL_HEIGHT', -1e9)
    monkeypatch.setattr(task, 'FALL_UPRIGHT', -2.)
    n = 1024
    envs = [TensorG1Env(n, seed=3, max_episode_steps=1000, resample_commands=False, push_velocity=0, real_robot=True,
                        action_limit=10., observation_noise=False, randomize=False, arm_perturb=.5,
                        arm_perturb_radians=radians) for radians in (.4, 1.)]
    try:
        narrow, wide = envs
        arms = wide.arm_mask
        for env in envs:
            env.reset()
        policy = torch.full((n, 29), .05, device=wide.device)
        largest, segments = torch.zeros(int(arms.sum()), device=wide.device), 0
        for _ in range(300):
            for env in envs:
                env.step(policy)
            assert torch.equal(wide.perturb_steps, narrow.perturb_steps)
            torch.testing.assert_close(wide.perturb_action[:, arms], (2.5 * narrow.perturb_action[:, arms]).clamp(-10, 10),
                                       rtol=1e-5, atol=1e-6)
            on = (wide.previous_action[:, arms] != .05).any(dim=1)
            assert torch.equal(on, (narrow.previous_action[:, arms] != .05).any(dim=1))
            radians = (wide.perturb_action[on][:, arms] * wide.scale[arms]).abs()
            if len(radians):
                largest = torch.maximum(largest, radians.amax(dim=0))
            segments += int(on.sum())
            assert not bool(wide.arm_contacts_masked.any())  # the mask is off
        wrists = wide.scale[arms] < .1
        assert int(wrists.sum()) == 4 and segments > 1000
        assert bool((largest[~wrists] <= 1. + 1e-5).all()) and bool((largest[~wrists] > .95).all())
        torch.testing.assert_close(largest[wrists], 10 * wide.scale[arms][wrists])  # clipped at the box edge
        assert wide.rollout_metrics()['rollout/arm_perturb'] == narrow.rollout_metrics()['rollout/arm_perturb']
    finally:
        for env in envs:
            env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_arm_press(monkeypatch):
    """World 0 presses the upper arms on the torso (no delay), world 1 sets the left shoulder pitch target 0.5 rad
    from home with the motors 15 substeps late, so its whole first step follows the zero action: each against the
    CPU task, the same Σ over the arm joints of (policy target − angle)², the same term and reward; the policy's
    own target counts under the delay (0.25 rad² at the first step). Logged as rollout/arm_press while weighted."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from test_g1_self_contact import arms_pressed
    seen_gpu, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen_gpu.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    weight = .05
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55},
                   reward_weights={'arm_press': weight}, action_delay=.03)
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        assert gpu.arm_joints.tolist() == list(range(15, 29))
        for cpu, steps in zip(cpus, (0, 15)):
            cpu.reset(seed=4, options={'command': [0., 0., 0.], 'action_delay_steps': steps})
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, 15]))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device,
                                    dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()  # refreshes the observed (rewarded) command
        shoulder = np.zeros(29)
        shoulder[15] = .5 / cpus[1].scale[15]
        actions = np.stack([arms_pressed(), shoulder])
        action = torch.tensor(actions, device=gpu.device, dtype=torch.float32)
        presses = []
        for step in range(8):
            results = [cpu.step(a) for cpu, a in zip(cpus, actions)]
            _, reward, _, _, _ = gpu.step(action)
            gpu.check_health()
            kwargs, components = seen_gpu[-1]
            target = task.pd_target(action, gpu.home, gpu.scale, gpu.joint_low, gpu.joint_high, 10.)
            torch.testing.assert_close(kwargs['arm_press'], task.arm_press(target, gpu.qpos[:, gpu.qadr],
                                                                           gpu.arm_joints))
            for world in range(2):
                expected, expected_reward, *_, info = results[world]
                cpu_press = -info['reward_components']['arm_press'] / weight
                assert float(kwargs['arm_press'][world]) == pytest.approx(cpu_press, rel=.1, abs=.02)
                assert float(components['arm_press'][world]) == pytest.approx(-weight * float(kwargs['arm_press'][world]),
                                                                              rel=1e-6)
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
            presses.append(kwargs['arm_press'].cpu().numpy())
        assert presses[0][1] == pytest.approx(.25, abs=.01)  # world 1: the arm still at home, the target 0.5 away
        assert presses[-1][0] > 4.  # world 0: the torso holds the upper arms off their targets
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/arm_press'] == pytest.approx(float(np.mean(presses)), rel=1e-5)
        assert math.isnan(gpu.rollout_metrics()['rollout/arm_press'])  # cleared by each read
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    plain = TensorG1Env(2, seed=4, real_robot=True, action_limit=10.)  # off: nothing computed, nothing logged
    try:
        plain.reset()
        plain.step(torch.zeros((2, 29), device=plain.device))
        assert seen_gpu[-1][0]['arm_press'] is None and plain.rollout_metrics() == {}
    finally:
        plain.close()
    legs = TensorG1Env(2, seed=4, legs_only=True, reward_weights={'arm_press': 1.})  # no arms: the term is 0
    try:
        assert legs.arm_joints.numel() == 0
        legs.reset()
        legs.step(torch.ones((2, 12), device=legs.device))
        assert float(seen_gpu[-1][1]['arm_press'].abs().max()) == 0
        assert legs.rollout_metrics()['rollout/arm_press'] == 0
    finally:
        legs.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_yaw_drift_follows_the_shared_update_restarts_and_is_logged(monkeypatch):
    from humanoid_sim.tasks import g1_tensor_math as task
    commands = {'min_speed': .5, 'max_speed': 1., 'stand_probability': 0., 'yaw_rate': 1., 'straight_probability': 1.}
    profile = dict(real_robot=True, action_limit=10., heading_hold=True, commands=commands, push_velocity=0,
                   observation_noise=False, randomize=False)
    env = TensorG1Env(4, seed=3, reward_weights={'yaw_drift': 5.}, max_episode_steps=4, resample_commands=False,
                      **profile)
    try:
        env.reset()
        assert float(env.yaw_drift_error.abs().max()) == 0
        means = []
        for _ in range(3):
            before, steered = env.yaw_drift_error.clone(), env.observed_command.clone()
            env.step(torch.zeros((4, 29), device=env.device))
            env.check_health()
            torch.testing.assert_close(env.yaw_drift_error, task.yaw_drift_update(before, env._velocity(), steered,
                                                                                  env.dt))
            means.append(float(env.yaw_drift_error.abs().mean()))  # every command is straight here
        metrics = env.rollout_metrics()
        assert metrics['rollout/yaw_drift_abs'] == pytest.approx(np.mean(means), rel=1e-4)
        assert math.isnan(env.rollout_metrics()['rollout/yaw_drift_abs'])  # cleared by each read
        env.step(torch.zeros((4, 29), device=env.device))  # the fourth step ends every episode
        assert float(env.yaw_drift_error.abs().max()) == 0
    finally:
        env.close()
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)  # resample every step
    env = TensorG1Env(4, seed=3, reward_weights={'yaw_drift': 5.}, **profile)
    try:
        env.reset()
        for _ in range(3):
            env.step(torch.zeros((4, 29), device=env.device))
            assert float(env.yaw_drift_error.abs().max()) == 0  # charged this step, then a new command starts at 0
    finally:
        env.close()
    env = TensorG1Env(2, seed=3, **profile)  # off: no state, nothing logged
    try:
        env.reset()
        env.step(torch.zeros((2, 29), device=env.device))
        assert float(env.yaw_drift_error.abs().max()) == 0 and env.rollout_metrics() == {}
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_short_trajectory_with_force_push(monkeypatch):
    """Three worlds pushed by hand, each against the CPU task with the same push: back 1 m/s, left 1.25 m/s, none,
    all for 0.05 s (25 substeps: the push ends in the third step). Warp's captured step graph reads the pelvis
    xfrc_applied written before each launch: every substep gets the force the CPU task's mj_step gets (same first and
    last substep), and the worlds keep the CPU pelvis velocity (at the push's end 2.05 / 2.25 m/s faster than the
    unpushed one on CPU), positions and orientation strictly, joint velocities within the CPU-GPU contact gap (see
    below), and the reward. The simulation profile: with the motor model a 1e-6 nudge of a pushed CPU trajectory
    grows to 0.26 in four steps (friction tanh), beyond any CPU-GPU tolerance."""
    import mujoco
    import warp as wp
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    pushes = [(-1., 0.), (0., 1.25), (0., 0.)]
    cpus = [G1VelocityEnv() for _ in pushes]
    gpu = TensorG1Env(3, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False)
    cpu_forces, gpu_forces = [[] for _ in pushes], []
    original_step, original_launch = mujoco.mj_step, wp.capture_launch

    def cpu_step(model, data):  # what each CPU substep gets
        for world, cpu in enumerate(cpus):
            if data is cpu.data:
                cpu_forces[world].append(data.xfrc_applied.copy())
        original_step(model, data)

    def launch(graph, *args, **kwargs):  # what each Warp substep reads
        if graph is gpu.step_graph:
            gpu_forces.append(gpu.xfrc_applied.cpu().numpy().copy())
        return original_launch(graph, *args, **kwargs)

    monkeypatch.setattr(mujoco, 'mj_step', cpu_step)
    monkeypatch.setattr(wp, 'capture_launch', launch)
    try:
        for cpu in cpus:
            cpu.reset(seed=4, options={'command': [0., 0., 0.]})
        gpu.reset()
        assert not gpu.force_pushes  # off until a push is set
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()
        for world in range(3):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        for cpu, velocity in zip(cpus, pushes):
            cpu.set_force_push(velocity, .05)
        gpu.set_force_push(torch.tensor(pushes), .05)
        assert gpu.force_pushes and gpu.push_left.tolist() == [25, 25, 25]
        torch.testing.assert_close(gpu.push_force.cpu().double(),
                                   torch.from_numpy(np.stack([cpu.push_force for cpu in cpus])), rtol=1e-5, atol=1e-3)
        joints = cpus[0].action_space.shape[0]
        # Observation blocks of the simulation profile: velocities (pelvis angular and linear, joints), positions and
        # orientation (projected gravity, joint angles), the rest (command, previous action, gait phase).
        velocities, positions = np.r_[0:6, 41:70], np.r_[6:9, 12:41]
        rest = np.setdiff1d(np.arange(len(start[0])), np.r_[velocities, positions])
        previous = [cpu._observation() for cpu in cpus]
        for step in range(6):
            results = [cpu.step(np.zeros(joints)) for cpu in cpus]
            _, reward, _, _, terminal = gpu.step(torch.zeros((3, joints), device=gpu.device))
            gpu.check_health()
            assert gpu.push_left.tolist() == [cpu.push_left for cpu in cpus]
            velocity = gpu.qvel[:, :2].cpu().numpy()
            for world in range(3):
                expected, expected_reward = results[world][:2]
                observed = terminal.cpu().numpy()[world]
                # Positions and orientation strictly (at most 6e-4 seen, in world 1 after its arm hit the torso).
                np.testing.assert_allclose(observed[positions], expected[positions], atol=2e-3, rtol=0)
                np.testing.assert_allclose(gpu.qpos[world].cpu().numpy(), cpus[world].data.qpos, atol=2e-3, rtol=0)
                np.testing.assert_allclose(observed[rest], expected[rest], atol=1e-5, rtol=0)
                # Velocities: the usual tolerance plus 2% of the step's own change. In world 1 the push swings the
                # left arm (shoulder roll -7.7 rad/s) into the torso at step 2: a mesh-mesh contact, for which MuJoCo
                # makes 1 point in two substeps and MJWarp 4, so the impact's impulse is shared differently; the
                # shoulder velocities then differ by 0.05-0.07 rad/s (0.6% / 0.14% of their change in the step
                # beyond the usual tolerance), fading to 0.007 by step 5. Elsewhere they agree within 5e-4.
                change = np.abs(expected[velocities] - previous[world][velocities])
                error = np.abs(observed[velocities] - expected[velocities])
                assert (error <= .03 + .02 * np.abs(expected[velocities]) + .02 * change).all(), (step, world)
                previous[world] = expected
                np.testing.assert_allclose(reward.cpu()[world], expected_reward, atol=.03, rtol=.02)
                np.testing.assert_allclose(gpu.qvel[world, :3].cpu().numpy(), cpus[world].data.qvel[:3], atol=.02,
                                           rtol=0)
            if step == 2:  # the push is over: the pushed pelvis runs ~2 m/s faster along it
                cpu_velocity = np.array([cpu.data.qvel[:2] for cpu in cpus])
                for world, along in ((0, (-1., 0.)), (1, (0., 1.))):
                    gain = (velocity[world] - velocity[2]) @ along
                    assert gain > 1.5 and gain == pytest.approx((cpu_velocity[world] - cpu_velocity[2]) @ along,
                                                                abs=.05)
        forces = np.array(gpu_forces)
        assert forces.shape[:2] == (60, 3)
        for world in range(3):
            np.testing.assert_allclose(forces[:, world], np.array(cpu_forces[world]), rtol=1e-5, atol=1e-3)
        pelvis = gpu.push_body
        assert not np.delete(forces, pelvis, axis=2).any() and not forces[:, :, pelvis, 2:].any()
        pushed = np.abs(forces[:, :, pelvis, :2]).sum(axis=-1) > 0
        assert pushed[:25, :2].all() and not pushed[25:].any() and not pushed[:, 2].any()
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_force_push_runs_per_world_on_top_of_the_velocity_pushes_and_ends_at_reset(monkeypatch):
    """1024 worlds of the training profile (velocity pushes, randomisation, noise) at the working 1.25 m/s, 1000-step
    episodes with falls switched off (a fallen robot lies on; the pushes do not depend on it): each world starts its
    pushes 100-200 steps apart, the first one its reset's interval on; the pelvis xfrc_applied each step graph reads
    is the world's push force for 25-100 substeps from a control step on and 0 otherwise, no other body is pushed,
    Δv = |F|·T/M lies in [0, 1.25] (mean 0.625); the reset ends every push. Off, nothing is drawn or written."""
    import warp as wp
    from humanoid_sim.tasks import g1_tensor_math as task
    monkeypatch.setattr(task, 'FALL_HEIGHT', -1e9)
    monkeypatch.setattr(task, 'FALL_UPRIGHT', -2.)
    n, steps = 1024, 999
    env = TensorG1Env(n, seed=3, max_episode_steps=1000, real_robot=True, action_limit=10., force_push=1.25)
    on = torch.zeros((steps * 10, n), dtype=torch.bool, device=env.device)
    wrong = torch.zeros((), dtype=torch.bool, device=env.device)
    launches, original = [0], wp.capture_launch

    def launch(graph, *args, **kwargs):  # what each substep's step graph reads, without a host sync
        if graph is env.step_graph and launches[0] < len(on):
            pelvis = env.xfrc_applied[:, env.push_body]
            pushed = pelvis[:, :3].abs().sum(dim=1) > 0
            on[launches[0]] = pushed
            others = torch.cat((env.xfrc_applied[:, :env.push_body], env.xfrc_applied[:, env.push_body + 1:]), dim=1)
            wrong.logical_or_(((pelvis[:, :3] - env.push_force) * pushed[:, None]).abs().amax() > 0)
            wrong.logical_or_((pelvis[:, 2:].abs().amax() > 0) | (others.abs().amax() > 0))
            launches[0] += 1
        return original(graph, *args, **kwargs)

    monkeypatch.setattr(wp, 'capture_launch', launch)
    try:
        assert env.force_pushes and env.push_body == env.pelvis
        assert env.reference.mass == pytest.approx(35.112142, abs=1e-6)
        env.reset()
        first = env.push_timer.clone()
        assert int(first.min()) >= 100 and int(first.max()) <= 200 and len(first.unique()) > 50
        policy = torch.zeros((n, 29), device=env.device)
        started = torch.zeros((steps, n), dtype=torch.bool, device=env.device)
        dvs = []
        for step in range(steps):
            before = env.push_timer.clone()
            env.step(policy)
            start = before == 0
            started[step] = start
            assert torch.equal(env.push_timer[~start], before[~start] - 1)
            length = env.push_left[start] + 10  # every push outlasts its first step (25 substeps or more)
            dvs.append(env.push_force[start, :2].norm(dim=1) * length * .002 / env.reference.mass)
        env.check_health()
        assert launches[0] == steps * 10 and not bool(wrong)
        dvs = torch.cat(dvs)
        assert len(dvs) > 3000 and float(dvs.min()) >= 0 and float(dvs.max()) <= 1.25 * (1 + 1e-5)
        assert float(dvs.mean()) == pytest.approx(.625, abs=.03)
        started, on, first = started.cpu().numpy(), on.cpu().numpy(), first.cpu().numpy()
        assert started.sum(axis=1).max() < 40  # independent worlds: starts spread over the steps
        for world in range(n):
            starts = np.flatnonzero(started[:, world])
            gaps = np.diff(starts)
            assert starts[0] == first[world] and gaps.min() >= 100 and gaps.max() <= 200
            edges = np.flatnonzero(np.diff(np.r_[0, on[:, world].astype(int), 0]))
            begin, end = edges[::2], edges[1::2]
            assert not (begin % 10).any() and set((begin // 10).tolist()) <= set(starts.tolist())
            lengths = (end - begin)[end < len(on)]  # the last push may run past the recording
            assert lengths.min() >= 25 and lengths.max() <= 100
        _, _, terminated, truncated, _ = env.step(policy)
        assert truncated.all() and not terminated.any()
        assert int(env.push_left.abs().max()) == 0 and float(env.push_force.abs().max()) == 0
        assert float(env.xfrc_applied.abs().max()) == 0  # reset_data zeroed it
        assert int(env.push_timer.min()) >= 100 and int(env.push_timer.max()) <= 200
    finally:
        env.close()
    monkeypatch.setattr(wp, 'capture_launch', original)
    plain, zero = TensorG1Env(2, seed=3), TensorG1Env(2, seed=3, force_push=0.)
    try:
        for other in (plain, zero):
            other.reset()
            other.step(torch.full((2, 29), .3, device=other.device))
        assert not zero.force_pushes and int(zero.push_timer.abs().max()) == 0
        assert torch.equal(zero.generator.get_state(), plain.generator.get_state())
        assert float(zero.xfrc_applied.abs().max()) == 0
        torch.testing.assert_close(zero.qpos, plain.qpos, rtol=0, atol=0)
    finally:
        plain.close(), zero.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_motor_friction_scales_the_motor_torque_per_world(monkeypatch):
    """Three worlds with the motor friction set by hand to 0, 0.5 and 1.25 (set_motor_friction, no setting), each
    against the CPU task reset with the same motor_friction_scale, from one pose with the joints moving at
    0.5·sin(j + 1) rad/s, so the friction acts from the first substep (up to 3.1 N·m between m 0 and 1.25 on CPU).
    The first substep's torque is the CPU task's in every world; every substep's torque is the shared motor model
    with the world's m on its own Warp state (m = 0: no friction at all). Trajectories are compared at m = 0 only:
    with the friction's tanh(v/0.01) a float32-rounded start drifts from the float64 one on CPU alone by 0.04-0.18 in
    the observation within 1-3 steps at m 0.5-1.25 (by 4e-5 at m = 0), beyond any CPU-GPU tolerance."""
    import mujoco
    import warp as wp
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    scales = (0., .5, 1.25)
    profile = dict(real_robot=True, action_limit=10., gait={'period': .6, 'stance_fraction': .55})
    cpus = [G1VelocityEnv(**profile) for _ in scales]
    gpu = TensorG1Env(3, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    cpu_ctrl, gpu_seen = [[] for _ in scales], []
    original_step, original_launch = mujoco.mj_step, wp.capture_launch

    def cpu_step(model, data):  # what each CPU substep gets
        for world, cpu in enumerate(cpus):
            if data is cpu.data:
                cpu_ctrl[world].append(data.ctrl.copy())
        original_step(model, data)

    def launch(graph, *args, **kwargs):  # the Warp state and torque each substep starts from
        if graph is gpu.step_graph:
            gpu_seen.append((gpu.qpos[:, gpu.qadr].clone(), gpu.qvel[:, gpu.vadr].clone(), gpu.ctrl.clone()))
        return original_launch(graph, *args, **kwargs)

    monkeypatch.setattr(mujoco, 'mj_step', cpu_step)
    monkeypatch.setattr(wp, 'capture_launch', launch)
    try:
        assert not gpu.motor_frictions and gpu.reference.motor_friction == 0
        for cpu, m in zip(cpus, scales):
            cpu.reset(seed=4, options={'command': [.6, 0., 0.], 'motor_friction_scale': m})
            assert cpu.friction_scale == m
            cpu.data.qvel[cpu.vadr] = .5 * np.sin(np.arange(cpu.model.nu) + 1.)
            mujoco.mj_forward(cpu.model, cpu.data)
        gpu.reset()
        assert torch.equal(gpu.friction_scale.cpu(), torch.ones(3))
        gpu.set_motor_friction(torch.tensor(scales))
        assert gpu.motor_frictions
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.copy_(torch.tensor(np.stack([cpu.data.qvel for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.command.copy_(torch.tensor([[.6, 0., 0.]], device=gpu.device).expand(3, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs().cpu()  # also refreshes the observed (rewarded) command
        for world in range(3):
            np.testing.assert_allclose(start[world], cpus[world]._observation(), atol=2e-5, rtol=2e-5)
        joints = cpus[0].action_space.shape[0]
        for step in range(3):
            results = [cpu.step(np.zeros(joints)) for cpu in cpus]
            _, reward, _, _, terminal = gpu.step(torch.zeros((3, joints), device=gpu.device))
            gpu.check_health()
            np.testing.assert_allclose(terminal.cpu()[0], results[0][0], atol=.03, rtol=.02)
            np.testing.assert_allclose(reward.cpu()[0], results[0][1], atol=.03, rtol=.02)
        assert len(gpu_seen) == 30 and all(len(ctrl) == 30 for ctrl in cpu_ctrl)
        # The first substep starts from the same state on both backends: the same torque in every world.
        first, expected = gpu_seen[0][2].cpu().numpy(), np.array([ctrl[0] for ctrl in cpu_ctrl])
        np.testing.assert_allclose(first, expected, rtol=1e-4, atol=2e-3)
        assert np.abs(expected[2] - expected[0]).max() > 2  # the friction acted
        # Every substep: the shared motor model with each world's m on the Warp state it starts from.
        for position, velocity, ctrl in gpu_seen:
            pd = task.pd_torque(torch.zeros_like(ctrl), position, velocity, gpu.home, gpu.scale, gpu.kp, gpu.kd,
                                gpu.joint_low, gpu.joint_high, -gpu.unclamped, gpu.unclamped, 10.)
            torque = task.motor_torque(pd, velocity, *gpu.motors, friction_scale=gpu.friction_scale)
            torch.testing.assert_close(ctrl, torque.clamp(min=gpu.torque_low, max=gpu.torque_high), rtol=1e-6,
                                       atol=1e-5)
            x1, x2, y1, y2, static, viscous = gpu.motors
            frictionless = task.motor_torque(pd, velocity, x1, x2, y1, y2, 0 * static, 0 * viscous)
            torch.testing.assert_close(ctrl[0], frictionless[0].clamp(min=gpu.torque_low, max=gpu.torque_high),
                                       rtol=1e-6, atol=1e-5)
        assert 'rollout/motor_friction' not in gpu.rollout_metrics()  # logged only with the setting
        gpu.reset()  # without the setting a reset puts m back to 1
        assert torch.equal(gpu.friction_scale.cpu(), torch.ones(3))
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_motor_friction_draws_per_world_redraws_at_reset_and_is_logged():
    """256 worlds at the working 1.25: each world's m is drawn at its reset in [0, 1.25] (mean 0.625), fixed within
    the episode, redrawn at the next reset and logged as rollout/motor_friction; off, nothing is drawn or scaled."""
    env = TensorG1Env(256, seed=3, max_episode_steps=2, resample_commands=False, push_velocity=0, real_robot=True,
                      action_limit=10., observation_noise=False, randomize=False, motor_friction=1.25)
    try:
        assert env.motor_frictions and env.reference.motor_friction == 1.25
        env.reset()
        first = env.friction_scale.clone()
        assert float(first.min()) >= 0 and float(first.max()) < 1.25 and len(first.unique()) > 250
        assert float(first.mean()) == pytest.approx(.625, abs=.1)
        action = torch.zeros((256, 29), device=env.device)
        env.step(action)
        assert torch.equal(env.friction_scale, first)  # fixed within an episode
        assert env.rollout_metrics()['rollout/motor_friction'] == pytest.approx(float(first.mean()), rel=1e-5)
        _, _, _, truncated, _ = env.step(action)
        env.check_health()
        assert truncated.all()
        again = env.friction_scale
        assert float(again.min()) >= 0 and float(again.max()) < 1.25 and float((again != first).float().mean()) > .99
    finally:
        env.close()
    plain = TensorG1Env(2, seed=3, real_robot=True, action_limit=10.)
    zero = TensorG1Env(2, seed=3, real_robot=True, action_limit=10., motor_friction=0.)
    try:
        for other in (plain, zero):
            other.reset()
            other.step(torch.full((2, 29), .3, device=other.device))
        assert not zero.motor_frictions and torch.equal(zero.friction_scale.cpu(), torch.ones(2))
        assert torch.equal(zero.generator.get_state(), plain.generator.get_state())
        torch.testing.assert_close(zero.qpos, plain.qpos, rtol=0, atol=0)
        assert 'rollout/motor_friction' not in zero.rollout_metrics()
        for bad in (-.1, float('nan'), [1., float('inf')]):
            with pytest.raises(ValueError):
                zero.set_motor_friction(bad)
    finally:
        plain.close(), zero.close()
    simulation = TensorG1Env(2, seed=3)  # no motor model, nothing to scale
    try:
        with pytest.raises(ValueError):
            simulation.set_motor_friction(.5)
    finally:
        simulation.close()


# height_command / upper_body_slot (iteration 28): candidate 8's layout with the two terms, the crouch draws' recipe.
HEIGHT_PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.,
                      gait={'period': .6, 'stance_fraction': .55}, height_command=True, upper_body_slot=True,
                      reward_weights={'height': 10., 'height_ceiling': 2., 'stand_pose': 1., 'still': 3.})
CROUCH_COMMANDS = {'crouch_share': [.5, .33], 'crouch_min': [.64, .70], 'crouch_speed': 1.}


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_height_command(monkeypatch):
    """Two standing worlds told 0.70 and 0.78 (set_height), each against the CPU task reset with the same height: the
    observation (the height term c − 0.78 and the slot's zeros in every history frame) matches, and over three steps
    the CUDA task hands the shared reward the world's c and stand_target(home, c) on the crouch joints, so the
    height, height_ceiling and stand_pose terms match the CPU ones; at 0.78 the ceiling is 0 and the stand pose home."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    heights = (.70, .78)
    cpus = [G1VelocityEnv(**HEIGHT_PROFILE) for _ in heights]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **HEIGHT_PROFILE)
    try:
        assert gpu.observation_space.shape == (589,) and gpu.crouch and gpu.fixed_normalization is not None
        for cpu, height in zip(cpus, heights):
            cpu.reset(seed=4, options={'command': [0., 0., 0.], 'height': height})
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        gpu.set_height(torch.tensor(heights))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        start = gpu._obs('fill').cpu()
        for world, cpu in enumerate(cpus):
            np.testing.assert_allclose(start[world], cpu._observation(), atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(start[world, 490:495], heights[world] - .78, atol=1e-6, rtol=0)
            assert not start[world, 495:585].any()
        for step in range(3):
            results = [cpu.step(np.zeros(29)) for cpu in cpus]
            _, reward, terminated, _, terminal = gpu.step(torch.zeros((2, 29), device=gpu.device))
            gpu.check_health()
            kwargs, components = seen[-1]
            torch.testing.assert_close(kwargs['height_command'].cpu(), torch.tensor(heights))
            torch.testing.assert_close(kwargs['stand_home'],
                                       task.stand_target(gpu.home, gpu.height_command, gpu.crouch_joints))
            assert torch.equal(kwargs['stand_home'][1], gpu.home)  # 0.78: home
            for world, result in enumerate(results):
                np.testing.assert_allclose(terminal.cpu()[world], result[0], atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], result[1], atol=.03, rtol=.02)
                parts = result[4]['reward_components']
                for name in ('height', 'height_ceiling', 'stand_pose'):
                    np.testing.assert_allclose(float(components[name][world]), parts[name], atol=2e-3, rtol=1e-2)
            assert float(components['height_ceiling'][0]) < -.3 and float(components['height_ceiling'][1]) == 0
            assert not terminated.any()
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_height_draws_per_world_redraws_at_command_changes_freezes_its_columns_and_logs(monkeypatch):
    """4096 worlds with the crouch recipe: at the reset standing worlds crouch about half the time to [0.64, 0.78),
    walking ones up to 1 m/s a third of the time to [0.70, 0.78), faster ones never; a resample at every step draws c
    for each new command; rollout/crouch_share and rollout/height_excess are logged; TensorNormalize keeps the fixed
    columns; without the settings nothing is drawn, the observation is the old 494 and no crouch log appears."""
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tensor_ppo import TensorNormalize
    commands = {'min_speed': -1., 'max_speed': 2., 'stand_probability': .3, 'lateral_speed': .5, 'yaw_rate': 1.,
                'straight_probability': .5, **CROUCH_COMMANDS}
    profile = dict(push_velocity=0, observation_noise=False, randomize=False)
    n = 4096
    env = TensorG1Env(n, seed=5, commands=commands, **profile, **HEIGHT_PROFILE)

    def kinds(command):
        moving = task.command_moving(command)
        fast = command[:, :2].norm(dim=1) > 1
        return ~moving, moving & ~fast, moving & fast

    try:
        obs = env.reset()
        assert obs.shape == (n, 589)
        torch.testing.assert_close(obs[:, 494], env.height_command - .78, atol=1e-6, rtol=0)
        stand, walk, fast = kinds(env.command)
        c = env.height_command
        for mask, share, lowest in ((stand, .5, .64), (walk, .33, .70), (fast, 0., None)):
            crouched = c[mask] < task.BASE_HEIGHT
            assert float(crouched.float().mean()) == pytest.approx(share, abs=.06)
            assert bool((c[mask][~crouched] == np.float32(task.BASE_HEIGHT)).all())
            if lowest is not None:
                assert float(c[mask][crouched].min()) >= lowest - 1e-6 and float(c[mask].max()) <= .78 + 1e-6
        norm = TensorNormalize(env.observation_space.shape, n, env.device, fixed=env.fixed_normalization)
        norm.observation(obs)
        positions = torch.as_tensor(env.fixed_normalization[0], device=env.device)
        assert not norm.obs.mean[positions].any() and float(norm.obs.var[490]) == .0025
        monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .001)  # every world changes its command every step
        action = torch.zeros((n, 29), device=env.device)
        for _ in range(3):
            obs, _, _, _, _ = env.step(action)
            env.check_health()
            stand, walk, fast = kinds(env.command)
            c = env.height_command
            assert bool((c[fast] == np.float32(.78)).all()) and float(c[walk].min()) >= .70 - 1e-6
            assert float(c[stand].min()) >= .64 - 1e-6 and float((c[stand] < .78).float().mean()) > .35
            torch.testing.assert_close(obs[:, 494], c - .78, atol=1e-6, rtol=0)
            norm.observation(obs)
            assert not norm.obs.mean[positions].any()
        metrics = env.rollout_metrics()
        assert .1 < metrics['rollout/crouch_share'] < .4 and metrics['rollout/height_excess'] > 0
    finally:
        env.close()
    plain_profile = {key: value for key, value in HEIGHT_PROFILE.items() if key not in ('height_command',
                                                                                       'upper_body_slot')}
    plain_commands = {key: value for key, value in commands.items() if key not in CROUCH_COMMANDS}
    plain = TensorG1Env(8, seed=5, commands=plain_commands, **profile, **plain_profile)
    off = TensorG1Env(8, seed=5, commands=plain_commands, height_command=False, upper_body_slot=False, **profile,
                      **plain_profile)
    try:
        assert plain.observation_space.shape == off.observation_space.shape == (494,)
        for other in (plain, off):
            other.reset()
            other.step(torch.full((8, 29), .3, device=other.device))
        assert torch.equal(off.generator.get_state(), plain.generator.get_state())
        torch.testing.assert_close(off.qpos, plain.qpos, rtol=0, atol=0)
        assert 'rollout/crouch_share' not in off.rollout_metrics() and off.fixed_normalization is None
        with pytest.raises(ValueError):
            off.set_height(.7)
    finally:
        plain.close(), off.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_arm_blind_freezes_the_arm_columns_and_normalizes_as_the_cpu_path():
    """arm_blind (iteration 31) on the height layout, 512 worlds with arm perturbations of 1.0 rad half the time and
    sensor noise: the CUDA task's fixed_normalization is the CPU task's (95 term columns, then the 210 arm columns with
    NaN means); its reset and first step are those of the task without arm_blind, bit for bit (the blindness is the
    normalization's; the CPU test checks 200 steps). TensorNormalize on CUDA, imported from blind statistics (mean
    −z·1e6, var 1e12 on the arm columns), keeps them through 30 updates and gives z there within 3e-5 in every world
    and step; the same statistics on the CPU (train.FrozenMoments, VecNormalize's formula) normalize those
    observations as TensorNormalize does."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from humanoid_sim.tensor_ppo import TensorNormalize
    from humanoid_sim.train import FrozenMoments
    profile = dict(push_velocity=0, observation_noise=True, randomize=False, arm_perturb=.5, arm_perturb_radians=1.)
    n = 512
    blind = TensorG1Env(n, seed=6, arm_blind=True, **profile, **HEIGHT_PROFILE)
    seen = TensorG1Env(n, seed=6, **profile, **HEIGHT_PROFILE)
    cpu = G1VelocityEnv(**HEIGHT_PROFILE, arm_blind=True, arm_perturb=.5, arm_perturb_radians=1.)
    try:
        positions, mean, variance = blind.fixed_normalization
        expected = cpu.fixed_normalization
        np.testing.assert_array_equal(positions, expected[0])
        np.testing.assert_array_equal(mean, expected[1])  # NaN where NaN
        np.testing.assert_array_equal(variance, expected[2])
        columns = np.asarray(cpu.arm_columns)
        assert len(columns) == 210 and np.array_equal(positions[-210:], columns) and np.isnan(mean[-210:]).all()
        rng = np.random.default_rng(0)
        stats = RunningMeanStd(shape=(589,))
        stats.mean, stats.var, stats.count = rng.normal(0, .3, 589), rng.uniform(.05, 4, 589), 8e9
        z = rng.uniform(-1.5, 1.5, 210)
        stats.mean[columns], stats.var[columns] = -z * task.ARM_BLIND_SCALE, task.ARM_BLIND_SCALE ** 2
        norm = TensorNormalize((589,), n, blind.device, fixed=blind.fixed_normalization)
        norm.import_normalizer(type('Stats', (), dict(obs_rms=stats, ret_rms=RunningMeanStd(shape=()), gamma=.99))())
        frozen = FrozenMoments(stats, blind.fixed_normalization)
        arms = torch.as_tensor(columns, device=blind.device)
        reference = torch.as_tensor(z, dtype=torch.float32, device=blind.device)
        obs, other = blind.reset(), seen.reset()
        assert torch.equal(obs, other)
        action = torch.zeros((n, 29), device=blind.device)
        (_, reward, *_), (_, other_reward, *_) = blind.step(action), seen.step(action)
        # Two CUDA envs step in parallel contact solves whose atomics are not bitwise reproducible across instances,
        # so the physics after one step agrees to float noise, not bit for bit (the CPU test checks bit for bit).
        torch.testing.assert_close(blind.qpos, seen.qpos, rtol=0, atol=1e-5)
        torch.testing.assert_close(reward, other_reward, rtol=0, atol=1e-4)
        assert torch.equal(blind.generator.get_state(), seen.generator.get_state())
        for _ in range(30):
            normalized = norm.observation(obs)
            frozen.update(obs.double().cpu().numpy())
            assert float((normalized[:, arms] - reference).abs().max()) < 3e-5
            on_cpu = np.clip((obs.double().cpu().numpy() - frozen.mean) / np.sqrt(frozen.var + 1e-8), -10, 10)
            np.testing.assert_allclose(normalized.cpu().numpy(), on_cpu.astype(np.float32), rtol=1e-6, atol=1e-6)
            obs, *_ = blind.step(action)
            blind.check_health()
        assert blind.rollout_metrics()['rollout/arm_perturb'] > .1
        assert float(obs[:, arms].std(dim=0).max()) > .1  # the raw arm inputs did move
        torch.testing.assert_close(norm.obs.mean[arms].cpu(), torch.from_numpy(-z * task.ARM_BLIND_SCALE), rtol=0, atol=0)
        assert bool((norm.obs.var[arms] == task.ARM_BLIND_SCALE ** 2).all())
        np.testing.assert_array_equal(frozen.mean[columns], -z * task.ARM_BLIND_SCALE)
    finally:
        blind.close(), seen.close(), cpu.close()


# g1_getup (iteration 33, part B): the CUDA task g1_getup_tensor_env.TensorG1GetupEnv against the CPU task G1GetupEnv.
# It reads the start bank (runs/humanoid_lab/getup_bank/bank.npz, untracked): skipped without it.
def _getup_bank():
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    if not bank_module.BANK.is_file():
        pytest.skip('the g1_getup start bank is not on this machine')
    return bank_module.load()


def _getup_rows(bank, classes, holdout=False):
    """The first row of each class (training rows unless holdout)."""
    return [int(np.flatnonzero((bank['start_class'] == bank['classes'].index(kind)) & (bank['holdout'] == holdout))[0])
            for kind in classes]


GETUP_CLASSES = ('back', 'front', 'side', 'fall', 'sit', 'stand')


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_from_the_same_bank_poses(monkeypatch):
    """One world per start class, each against the CPU task reset from the same bank row with the same settings (assist
    200 N, delay 0, motor friction 1, body friction 0.6; no sensor noise or COM offset on CUDA): the first observation
    (the walking actor inputs with the command and phase at 0, then the six critic extras: pelvis velocity, pelvis
    height, head height, assist) matches; over three steps of the same actions the observation, the reward and each of
    its terms (task.getup_reward_components on both, numpy and torch), the lift (only the standing world's pelvis is
    upright enough: 200 N, the lying ones 0) and the head height match within float32-vs-float64 and contact noise
    (lying, many mesh-plane contacts; the walking tests' tolerances)."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    bank = _getup_bank()
    rows = _getup_rows(bank, GETUP_CLASSES)
    seen, original = [], task.getup_reward_components

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append((isinstance(args[0], torch.Tensor), {key: np.asarray(value.cpu() if isinstance(value, torch.Tensor)
                                                                        else value, dtype=float).copy()
                                                         for key, value in components.items()}))
        return components, standing

    monkeypatch.setattr(task, 'getup_reward_components', spy)
    options = dict(assist_force=200., action_delay_steps=0, motor_friction_scale=1., body_friction=.6)
    cpus = [G1GetupEnv() for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False)
    try:
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(options, start=row))[0] for cpu, row in zip(cpus, rows)])
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(200.)
        obs = gpu._obs('fill').cpu().numpy()
        assert obs.shape == (len(rows), 496) and gpu.start_row.tolist() == rows
        np.testing.assert_allclose(obs, cpu_obs, atol=1e-4, rtol=1e-4)
        np.testing.assert_array_equal(obs[:, 490:][:, -1], 200.)
        # Velocities (base angular, joints, every frame) against the rest: lying, contacts of meshes with the floor and
        # with each other differ a little between MuJoCo C and MJWarp, which shows in the small links' speeds first
        # (the force_push test: up to 0.07 rad/s at the shoulders).
        speeds = np.zeros(496, dtype=bool)
        speeds[task.joint_positions(29, 5, (), (0,), range(3)) + task.joint_positions(29, 5, (), (5,), range(29))] = True
        rng = np.random.default_rng(0)
        for step in range(3):
            action = rng.normal(0., .5, 29)
            seen.clear()
            results = [cpu.step(action) for cpu in cpus]
            nxt, reward, terminated, truncated, terminal = gpu.step(
                torch.tensor(action, dtype=torch.float32, device=gpu.device).expand(len(rows), -1).contiguous())
            gpu.check_health()
            assert not terminated.any() and not truncated.any()
            got, expected = terminal.cpu().numpy(), np.stack([r[0] for r in results])
            np.testing.assert_allclose(got[:, ~speeds], expected[:, ~speeds], atol=.03, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, speeds], expected[:, speeds], atol=.1, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.03, rtol=.02)
            np.testing.assert_allclose(gpu.lift.cpu().numpy(), [r[4]['lift'] for r in results], atol=1e-3)
            np.testing.assert_allclose(gpu._head().cpu().numpy(), [r[4]['head_height'] for r in results], atol=2e-3)
            cpu_terms = [terms for on_gpu, terms in seen if not on_gpu]
            (gpu_terms,) = [terms for on_gpu, terms in seen if on_gpu]
            for key in gpu_terms:
                np.testing.assert_allclose(gpu_terms[key], [terms[key][0] for terms in cpu_terms], atol=.03, rtol=.02,
                                           err_msg=key)
        lifts = gpu.lift.cpu().numpy()
        assert lifts[GETUP_CLASSES.index('stand')] == 200. and not lifts[:4].any()
        assert not gpu.xfrc_applied.abs().max()  # only while the step runs
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_resets_bank_rows_assist_and_body_friction_per_world():
    """256 worlds: resets draw training rows only, every class in roughly its bank share, and one body friction per
    world in [0.3, 1.0] on every colliding robot geom and the floor (the other geoms keep the model's), redrawn at the
    next reset; the observation's command and phase columns read 0 and the task's fixed normalization is the CPU
    task's; the assist steps 20 N down at a reset after 2 s with the head above 1 m (not for a shorter episode, not
    below 0), training_state / restore_training_state carry the forces (spread by quantiles over another world count);
    the contact demand and the assist are logged per rollout."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    import warp as wp
    bank = _getup_bank()
    n = 256
    env = TensorG1GetupEnv(n, seed=8, observation_noise=False, randomize=False)
    cpu = G1GetupEnv()
    try:
        obs = env.reset()
        rows = env.start_row.cpu().numpy()
        assert not bank['holdout'][rows].any()
        shares = np.bincount(bank['start_class'][rows], minlength=6) / n
        np.testing.assert_allclose(shares, [.30, .30, .10, .15, .05, .10], atol=.1)
        friction = env.friction.cpu().numpy()
        assert friction.min() >= .3 and friction.max() <= 1. and len(np.unique(friction)) > 250
        model = wp.to_torch(env.model.geom_friction).cpu().numpy()[:, :, 0]
        geoms = cpu.friction_geoms
        np.testing.assert_allclose(model[:, geoms], np.repeat(friction[:, None], len(geoms), 1), rtol=0, atol=0)
        others = np.setdiff1d(np.arange(cpu.model.ngeom), geoms)
        np.testing.assert_allclose(model[:, others], np.repeat(cpu.nominal_friction[None, others], n, 0), atol=1e-6)
        columns, mean, variance = env.fixed_normalization
        expected = cpu.fixed_normalization
        np.testing.assert_array_equal(columns, expected[0])
        assert len(columns) == 25 and (mean == 0).all() and (variance == 1).all()
        assert not obs[:, torch.as_tensor(columns, device=env.device)].any()
        action = torch.zeros((n, 29), device=env.device)
        for _ in range(3):
            obs, *_ = env.step(action)
            env.check_health()
        assert not obs[:, torch.as_tensor(columns, device=env.device)].any()
        metrics = env.rollout_metrics()
        assert set(metrics) == {'rollout/assist', 'rollout/getup_success', 'rollout/nefc_max', 'rollout/ncon_max',
                                'rollout/bad_states'}
        # The first episodes are cut at random steps (episode_limit), a few may end within these 3 steps.
        success = metrics['rollout/getup_success']
        assert metrics['rollout/assist'] == 200. and (math.isnan(success) or 0 <= success <= 1)
        assert 0 < metrics['rollout/ncon_max'] < 128 and 0 < metrics['rollout/nefc_max'] < 512
        assert metrics['rollout/bad_states'] == 0
        # Assist: the head above 1 m over the last 2 s of an episode of at least 2 s steps it down.
        half = n // 2
        stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
        with stream:
            env.head_window[:half], env.head_window[half:] = 1.05, .95
            env.lengths.fill_(150)
            env._masked_reset(torch.ones(n, dtype=torch.bool, device=env.device))
        assist = env.assist.cpu().numpy()
        assert (assist[:half] == 180.).all() and (assist[half:] == 200.).all()
        assert not np.array_equal(env.friction.cpu().numpy(), friction)  # redrawn
        with stream:
            env.head_window.fill_(1.2)
            env.lengths.fill_(99)  # shorter than 2 s: no step
            env._masked_reset(torch.ones(n, dtype=torch.bool, device=env.device))
            assert (env.assist.cpu().numpy() == assist).all()
            env.set_assist(10.)
            env.head_window.fill_(1.2)
            env.lengths.fill_(100)
            env._masked_reset(torch.ones(n, dtype=torch.bool, device=env.device))
        assert not env.assist.any()  # not below 0
        env.set_assist(torch.linspace(0., 200., n))
        state = env.training_state()
        assert len(state['assist_force']) == n and state['assist_force'][-1] == 200.
        env.restore_training_state({'assist_force': [200., 100., 0., 50.]})
        assert sorted(np.unique(env.assist.cpu().numpy()).tolist()) == [0., 50., 100., 200.]
        assert np.bincount(env.assist.cpu().numpy().astype(int) // 50).tolist() == [64, 64, 64, 0, 64]
        with pytest.raises(ValueError):
            env.restore_training_state({'assist_force': [-1.]})
        with pytest.raises(ValueError):
            env.set_starts(len(bank['qpos']))
    finally:
        env.close(), cpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_episodes_end_by_time_out_of_step_and_the_assist_steps_down_after_one_up():
    """Episodes end by the time limit: from set_starts every world runs exactly max_episode_steps (500) and is
    truncated together, its episode length 500 (last_episode_lengths, what TensorPPO.collect averages as
    rollout/ep_len_mean); one world held standing by hand (its bank pose written back after every step, the head at
    ~1.3 m) gets 20 N less assist at that reset, the lying one keeps 200 N, and rollout/getup_success is 1 of 2.
    After a full reset() the first episodes are cut at random steps of 1..500 (episode_limit), so the worlds end out of
    step (the first root kept 4096 worlds in lockstep: an episode end in 1 rollout of ~21); later ones last 500."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    import warp as wp
    bank = _getup_bank()
    stand, back = _getup_rows(bank, ('stand', 'back'))
    env = TensorG1GetupEnv(2, seed=3, observation_noise=False, randomize=False)
    try:
        assert env.max_episode_steps == 500
        env.reset()
        env.set_starts([stand, back])
        assert env.episode_limit.tolist() == [500, 500]
        pose = env.qpos[0].clone()
        stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
        action = torch.zeros((2, 29), device=env.device)
        env.rollout_metrics()
        ends = []
        for step in range(1, 501):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated.any()
            if truncated.any():
                ends.append((step, truncated.tolist()))
                break
            with stream:  # the standing world held by hand
                env.qpos[0], env.qvel[0] = pose, 0.
                wp.capture_launch(env.forward_graph)
            assert float(env._head()[0]) > 1.2
        assert ends == [(500, [True, True])]
        assert env.last_episode_lengths.tolist() == [500, 500] and env.lengths.tolist() == [0, 0]
        assert env.assist.tolist() == [180., 200.]
        metrics = env.rollout_metrics()
        assert metrics['rollout/getup_success'] == .5 and metrics['rollout/assist'] == 190.
    finally:
        env.close()
    env = TensorG1GetupEnv(64, seed=5, max_episode_steps=50, observation_noise=False, randomize=False)
    try:
        env.reset()
        limits = env.episode_limit.cpu().numpy()
        assert limits.min() >= 1 and limits.max() <= 50 and len(np.unique(limits)) > 25
        lengths = []
        action = torch.zeros((64, 29), device=env.device)
        for _ in range(120):
            _, _, terminated, truncated, _ = env.step(action)
            done = (terminated | truncated).cpu().numpy()
            lengths += env.last_episode_lengths.cpu().numpy()[done].tolist()
        first, later = lengths[:64], lengths[64:]
        assert len(set(first)) > 25  # the first episodes end spread over the steps
        assert later and set(later) <= {50} | set(range(1, 50))  # the rest run the whole length (or end early)
        assert np.mean(np.array(later) == 50) > .9 and (env.episode_limit == 50).all()
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_body_friction_reaches_the_contacts_of_each_world():
    """Warp's collision reads the batched geom_friction written at a reset in place (no recapture): two worlds lying on
    the same bank pose with body friction 0.3 and 1.2 get exactly those sliding frictions on their robot-floor
    contacts (the soles take theirs by priority, which is the same value)."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    import warp as wp
    bank = _getup_bank()
    row = _getup_rows(bank, ('back',))[0]
    env = TensorG1GetupEnv(2, seed=2, observation_noise=False, randomize=False)
    try:
        env.reset()
        env.set_starts(row)
        env.set_body_friction(torch.tensor([.3, 1.2]))
        env.step(torch.zeros((2, 29), device=env.device))
        env.check_health()
        count = int(env.nacon.item())
        world = env.contact_world[:count].long().cpu().numpy()
        friction = wp.to_torch(env.data.contact.friction)[:count, 0].cpu().numpy()
        geom = wp.to_torch(env.data.contact.geom)[:count].cpu().numpy()
        floor = env.reference.model.geom('floor').id
        on_floor = (geom == floor).any(axis=1)
        for w, value in enumerate((.3, 1.2)):
            mine = on_floor & (world == w)
            assert mine.sum() >= 3
            np.testing.assert_allclose(friction[mine], value, rtol=1e-6)
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_4096_worlds_fit_the_contact_buffers_stay_finite_and_report_speed():
    """The training batch: 4096 worlds from the bank with the training randomization (sensor noise, COM, delay, motor
    and body friction), 200 control steps of σ 1 action noise around home (an untrained policy's exploration), then
    100 of bang-bang targets over the whole ±10 box held 0.1-0.3 s (flailing): no contact or constraint buffer overflow
    (default nconmax / njmax 128 / 512 per world), no non-finite state; prints the most contacts and constraint rows of
    one world and the steps per second (walking: ~50 thousand)."""
    import time
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    _getup_bank()
    n = 4096
    env = TensorG1GetupEnv(n, seed=11)
    try:
        env.reset()
        generator = torch.Generator(device=env.device).manual_seed(5)
        for _ in range(10):  # warm-up
            env.step(torch.randn((n, 29), device=env.device, generator=generator))
        env.check_health()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            env.step(torch.randn((n, 29), device=env.device, generator=generator))
            env.check_health()
        torch.cuda.synchronize()
        speed = 200 * n / (time.perf_counter() - started)
        noise = env.rollout_metrics()
        held, hold = torch.zeros((n, 29), device=env.device), torch.zeros(n, dtype=torch.long, device=env.device)
        for _ in range(100):
            fresh = hold <= 0
            draw = (torch.rand((n, 29), device=env.device, generator=generator) * 2 - 1) * 10.
            held = torch.where(fresh[:, None], draw, held)
            hold = torch.where(fresh, torch.randint(5, 16, (n,), device=env.device, generator=generator), hold) - 1
            env.step(held)
            env.check_health()
        flail = env.rollout_metrics()
        print(f'g1_getup 4096 worlds: {speed:,.0f} steps/s (σ 1 noise); most per world: σ 1 noise '
              f'{noise["rollout/ncon_max"]} contacts / {noise["rollout/nefc_max"]} rows, flailing '
              f'{flail["rollout/ncon_max"]} / {flail["rollout/nefc_max"]}; bad states {noise["rollout/bad_states"]} / '
              f'{flail["rollout/bad_states"]}; episodes ended up {noise["rollout/getup_success"]}')
        for metrics in (noise, flail):
            assert metrics['rollout/ncon_max'] < 128 and metrics['rollout/nefc_max'] < 400
            assert metrics['rollout/bad_states'] == 0
        assert speed > 10000
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_full_gpu_learner_keeps_and_restores_the_assist(tmp_path):
    """full_gpu_train on the CUDA task: make_learner gives g1_getup the asymmetric policy (actor on 490 of 496 inputs)
    and TensorNormalize the command and phase columns fixed at 0 / 1; one rollout and update; a stage saved with the
    worlds' assist forces (training_state) gives them back to the next learner's worlds before their first reset."""
    from humanoid_sim import full_gpu_train
    from humanoid_sim.core import save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    from humanoid_sim.tasks import g1_tensor_math as task
    _getup_bank()
    config = {**validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='full_gpu', envs=64, n_steps=8,
                                     batch_size=512, n_epochs=1, steps=1024)), 'steps': 512}
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    learner, export = full_gpu_train.make_learner(config)
    try:
        assert isinstance(learner.model.policy, CriticExtrasPolicy) and learner.model.policy.actor_inputs == 490
        columns = torch.as_tensor(task.zero_term_columns(29, 5), device=learner.device)
        learner.collect()
        learner.update()
        assert bool((learner.normalizer.obs.mean[columns] == 0).all()) and bool((learner.normalizer.obs.var[columns]
                                                                                 == 1).all())
        learner.env.set_assist(torch.linspace(100., 200., 64))
        learner.normalizer.export_normalizer(export)
        stage = save_stage(run, learner.model, export, learner.env.training_state())
    finally:
        learner.env.close(), export.close()
    resumed, export = full_gpu_train.make_learner({**config, 'envs': 32, 'batch_size': 256}, stage)
    try:
        expected = task.assist_resample(np.linspace(100., 200., 64, dtype=np.float32).astype(float), 32)
        np.testing.assert_allclose(resumed.env.assist.cpu().numpy(), expected, rtol=1e-6)
        assert float(resumed.obs[:, -1].std()) > 0  # the first observation (after the reset) shows each world's own
    finally:
        resumed.env.close(), export.close()


# g1_getup task_config pd_bound (iteration 34): the PD error bound with its curriculum on CUDA.
GETUP_PD_BOUND = [1., .6]


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_pd_bound():
    """One world per start class against the CPU task reset from the same bank row, both at c = 0.15 under a 10 ms
    delay (5 substeps; no sensor noise or COM offset on CUDA), actions over the whole ±10 box: the CUDA applied action
    is exactly task.pd_bound of the clipped action around the newest frame of each world's history, then the wrists
    at 0, and the CPU task's within float32 (its angles drift with the contacts after the first step); the first
    step's actions in flight are the bounded zero on both (an unbounded home target would jerk the lying worlds apart);
    observations and rewards match within the tolerances of the test without the bound, but for the locked wrists'
    speeds (light links whose hand capsules touch the floor: MuJoCo C and MJWarp contact points, 0.6 rad/s apart on
    the third step lying on the side, while every angle stays within 0.03); the bound and the share of bounded
    joint-steps are logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    bank = _getup_bank()
    rows = _getup_rows(bank, GETUP_CLASSES)
    options = dict(assist_force=200., action_delay_steps=5, motor_friction_scale=1., body_friction=.6, pd_bound=.15)
    cpus = [G1GetupEnv(pd_bound=GETUP_PD_BOUND) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, pd_bound=GETUP_PD_BOUND)
    try:
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(options, start=row))[0] for cpu, row in zip(cpus, rows)])
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(5)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(200.)
        gpu.set_pd_bound(.15)
        obs = gpu._obs('fill').cpu().numpy()
        np.testing.assert_allclose(obs, cpu_obs, atol=1e-4, rtol=1e-4)
        gpu.rollout_metrics()
        speeds = np.zeros(496, dtype=bool)
        speeds[task.joint_positions(29, 5, (), (0,), range(3))] = True
        speeds[task.joint_positions(29, 5, (), (5,), range(29))] = True
        wrists = np.zeros(496, dtype=bool)
        wrists[task.joint_positions(29, 5, (), (5,), np.flatnonzero(cpus[0].wrist_mask.numpy()))] = True
        speeds &= ~wrists
        rng = np.random.default_rng(0)
        for step in range(3):
            action = rng.uniform(-10., 10., 29)
            actions = torch.tensor(action, dtype=torch.float32, device=gpu.device).expand(len(rows), -1).contiguous()
            position = gpu.frames[:, -1, gpu.position_entries].clone()
            expected = task.lock_joints(task.pd_bound(actions, position, gpu.bound, gpu.peak, gpu.kp, gpu.scale,
                                                      gpu.bound_free), gpu.wrist_mask)
            results = [cpu.step(action) for cpu in cpus]
            nxt, reward, terminated, truncated, terminal = gpu.step(actions)
            gpu.check_health()
            assert not terminated.any() and not truncated.any()
            torch.testing.assert_close(gpu.previous_action, expected, rtol=0, atol=0)
            np.testing.assert_allclose(gpu.previous_action.cpu().numpy(), np.stack([c.previous_action for c in cpus]),
                                       atol=1e-3 if step == 0 else .1, err_msg=f'step {step}')
            got, wanted = terminal.cpu().numpy(), np.stack([r[0] for r in results])
            rest = ~speeds & ~wrists
            np.testing.assert_allclose(got[:, rest], wanted[:, rest], atol=.03, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, speeds], wanted[:, speeds], atol=.1, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, wrists], wanted[:, wrists], atol=1., err_msg=f'step {step}')
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.03, rtol=.02)
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/pd_bound'] == pytest.approx(.15) and metrics['rollout/pd_bound_share'] > .5
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_pd_bound_steps_down_keeps_its_state_and_is_off_without_the_setting(monkeypatch):
    """256 worlds with pd_bound [1, 0.6]: every world starts at 1; at a reset after 2 s with the head above 1 m its
    bound steps 0.02 down with the assist (not after a shorter episode), to exactly the lowest and not below;
    training_state / restore_training_state carry the bounds (quantiles over another world count, within [lowest,
    start]; a stage without them leaves the worlds' bounds); rollout/pd_bound and rollout/pd_bound_share are logged.
    Without the setting the bound's functions never run and nothing is logged or kept for it."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    import warp as wp
    _getup_bank()
    n = 256
    lowest = float(np.float32(.6))
    env = TensorG1GetupEnv(n, seed=8, observation_noise=False, randomize=False, pd_bound=GETUP_PD_BOUND)
    try:
        env.reset()
        assert bool((env.bound == 1.).all())
        generator = torch.Generator(device=env.device).manual_seed(1)
        for _ in range(3):
            env.step(torch.randn((n, 29), device=env.device, generator=generator) * 5)
            env.check_health()
        metrics = env.rollout_metrics()
        assert metrics['rollout/pd_bound'] == 1. and 0 < metrics['rollout/pd_bound_share'] < 1
        everyone = torch.ones(n, dtype=torch.bool, device=env.device)
        half = n // 2
        stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
        with stream:
            env.head_window[:half], env.head_window[half:] = 1.05, .95
            env.lengths.fill_(150)
            env._masked_reset(everyone)
        bound = env.bound.cpu().numpy()
        np.testing.assert_allclose(bound[:half], .98, rtol=0, atol=1e-6)
        assert (bound[half:] == 1.).all() and (env.assist[:half] == 180.).all()
        with stream:
            env.head_window.fill_(1.2)
            env.lengths.fill_(99)  # shorter than 2 s: no step
            env._masked_reset(everyone)
        assert (env.bound.cpu().numpy() == bound).all()
        env.set_pd_bound(.61)
        for _ in range(2):  # to the lowest itself, then no lower
            with stream:
                env.head_window.fill_(1.2)
                env.lengths.fill_(100)
                env._masked_reset(everyone)
            assert (env.bound.cpu().numpy() == lowest).all()
        state = env.training_state()
        assert set(state) == {'assist_force', 'pd_bound'} and state['pd_bound'] == [lowest] * n
        env.restore_training_state({'assist_force': [0.] * 4, 'pd_bound': [1., .8, .6, .3]})
        values, counts = np.unique(env.bound.cpu().numpy(), return_counts=True)
        np.testing.assert_allclose(values, [.6, .8, 1.], rtol=1e-6)
        assert counts.tolist() == [128, 64, 64]  # 0.3 lifted to the lowest
        env.set_pd_bound(.9)
        env.restore_training_state({'assist_force': [0.]})  # a carried-over stage without bounds
        np.testing.assert_allclose(env.bound.cpu().numpy(), .9, rtol=1e-6)
        with pytest.raises(ValueError):
            env.restore_training_state({'assist_force': [0.], 'pd_bound': [0.]})
        with pytest.raises(ValueError):
            env.set_pd_bound(-1.)
    finally:
        env.close()

    def refuse(*args, **kwargs):
        raise AssertionError('pd_bound used without the setting')
    monkeypatch.setattr(task, 'pd_bound', refuse)
    monkeypatch.setattr(task, 'pd_bound_update', refuse)
    plain = TensorG1GetupEnv(64, seed=8, observation_noise=False, randomize=False)
    try:
        plain.reset()
        generator = torch.Generator(device=plain.device).manual_seed(1)
        for _ in range(3):
            plain.step(torch.randn((64, 29), device=plain.device, generator=generator) * 5)
        stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(plain.device)))
        with stream:
            plain.head_window.fill_(1.2)
            plain.lengths.fill_(150)
            plain._masked_reset(torch.ones(64, dtype=torch.bool, device=plain.device))
        assert set(plain.rollout_metrics()) == {'rollout/assist', 'rollout/getup_success', 'rollout/nefc_max',
                                                'rollout/ncon_max', 'rollout/bad_states'}
        assert set(plain.training_state()) == {'assist_force'} and not plain.bounding
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_full_gpu_learner_keeps_and_restores_the_pd_bound(tmp_path):
    """full_gpu_train with pd_bound: the worlds start at the setting's first bound, a rollout logs rollout/pd_bound and
    rollout/pd_bound_share, a stage (its action contract …-pdbound0.6) keeps the worlds' bounds next to the assist
    forces and gives them back to the next learner's worlds (quantiles over another world count)."""
    from humanoid_sim import full_gpu_train
    from humanoid_sim.core import read_json, save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks import g1_tensor_math as task
    _getup_bank()
    config = {**validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='full_gpu', envs=64, n_steps=8,
                                     batch_size=512, n_epochs=1, steps=1024,
                                     task_config={'pd_bound': GETUP_PD_BOUND})), 'steps': 512}
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    learner, export = full_gpu_train.make_learner(config)
    try:
        assert bool((learner.env.bound == 1.).all())
        metrics = learner.collect()
        assert metrics['rollout/pd_bound'] == 1. and 0 <= metrics['rollout/pd_bound_share'] <= 1
        learner.update()
        learner.env.set_pd_bound(torch.linspace(.6, 1., 64))
        learner.normalizer.export_normalizer(export)
        stage = save_stage(run, learner.model, export, learner.env.training_state())
    finally:
        learner.env.close(), export.close()
    metadata = read_json(stage / 'metadata.json')
    assert metadata['action_schema'].endswith('-pdbound0.6') and len(metadata['training_state']['pd_bound']) == 64
    resumed, export = full_gpu_train.make_learner({**config, 'envs': 32, 'batch_size': 256}, stage)
    try:
        expected = task.assist_resample(np.linspace(.6, 1., 64, dtype=np.float32).astype(float), 32)
        np.testing.assert_allclose(resumed.env.bound.cpu().numpy(), expected, rtol=1e-6)
    finally:
        resumed.env.close(), export.close()


# g1_getup reward weight impact (iteration 35): the floor approach speeds on CUDA. The drop rows of
# tests/test_g1_getup_impact.py (no start bank needed): the home pose on its back, on its front, on its front spinning,
# the lowest geom but the soles 0.3 m up, then standing in home.
GETUP_IMPACT = {'impact': 2.}


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_impact(monkeypatch):
    """The three drops on CUDA and on the CPU task (zero action holds the pose; no assist, delay, sensor noise or COM
    offset): the first floor contact comes in the same control step and physics substep, a body touches first on both,
    and those touching on both close at the same speed within 10 % (√(2gh) = 2.43 m/s for the two still drops); that
    step's impact term (task.getup_reward_components on both) agrees within 10 %, and every step before it charges 0."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 1, 2]
    seen, original = [], task.getup_reward_components

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        value = components['impact']
        seen.append((isinstance(value, torch.Tensor), np.asarray(value.cpu() if isinstance(value, torch.Tensor)
                                                                 else value, dtype=float).copy()))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    cpus = [G1GetupEnv(bank=bank, reward_weights=GETUP_IMPACT) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                           reward_weights=GETUP_IMPACT)
    try:
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu._obs('fill')
        first = {}
        expected = math.sqrt(2 * 9.81 * .3)
        for step in range(30):
            seen.clear()
            for cpu in cpus:
                cpu.step(np.zeros(29))
            _, _, terminated, _, _ = gpu.step(torch.zeros((len(rows), 29), device=gpu.device))
            gpu.check_health()
            assert not terminated.any()
            cpu_terms = np.array([terms[0] for on_gpu, terms in seen if not on_gpu])
            (gpu_terms,) = [terms for on_gpu, terms in seen if on_gpu]
            got = gpu.impact_substeps.cpu().numpy()
            for world, cpu in enumerate(cpus):
                if world in first:
                    continue
                wanted = cpu.impact_substeps
                touched = np.flatnonzero(wanted.max(1) > 0), np.flatnonzero(got[world].max(1) > 0)
                if not len(touched[0]) and not len(touched[1]):
                    assert cpu_terms[world] == 0 and gpu_terms[world] == 0, (world, step)
                    continue
                assert len(touched[0]) and len(touched[1]) and touched[0][0] == touched[1][0], (world, step, touched)
                substep = touched[0][0]
                both = np.flatnonzero((wanted[substep] > 0) & (got[world, substep] > 0))
                assert both.size, (world, step)
                np.testing.assert_allclose(got[world, substep, both], wanted[substep, both], rtol=.1)
                if world < 2:
                    np.testing.assert_allclose(wanted[substep, both], expected, rtol=.05)
                assert cpu_terms[world] < -1 and gpu_terms[world] == pytest.approx(cpu_terms[world], rel=.1)
                first[world] = step, substep
            if len(first) == len(rows):
                break
        assert sorted(first) == [0, 1, 2], first
        assert first[0][0] == first[1][0] > first[2][0]  # the still drops together, the spinning one sooner
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_impact_is_logged_per_rollout_and_off_without_the_weight(monkeypatch):
    """64 worlds, 16 on each drop row, zero actions for 20 steps (the drops hit at steps 6-13): rollout/impact is the
    mean unweighted term per world-step and rollout/impact_hit the share of world-steps with a body above 1.5 m/s, as
    the steps' own speeds give them, counted since the last call; every dropped world hits, the standing ones are never
    charged. Without the weight the impact functions never run, no buffer is made and nothing is logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_getup_impact import drop_bank
    bank = drop_bank()
    n = 64
    rows = torch.arange(n) % 4
    standing = rows == 3
    env = TensorG1GetupEnv(n, seed=6, observation_noise=False, randomize=False, bank=bank, reward_weights=GETUP_IMPACT)
    try:
        assert tuple(env.impact_substeps.shape) == (n, 10, env.reference.model.nbody)
        env.reset()
        env.set_starts(rows)
        env.set_action_delay_steps(0)
        env.set_assist(0.)
        env.rollout_metrics()
        zero = torch.zeros((n, 29), device=env.device)
        penalties, hits = [], []
        for _ in range(20):
            env.step(zero)
            env.check_health()
            speed = env.impact_substeps.amax(dim=1)
            penalties.append(task.impact_penalty(speed).cpu())
            hits.append(task.impact_hit(speed).cpu())
        penalties, hits = torch.stack(penalties), torch.stack(hits)
        metrics = env.rollout_metrics()
        assert metrics['rollout/impact'] == pytest.approx(float(penalties.mean()), rel=1e-5)
        assert metrics['rollout/impact_hit'] == pytest.approx(float(hits.float().mean()), rel=1e-6)
        assert not penalties[:, standing].any() and bool(hits[:, ~standing].any(dim=0).all())
        assert 0 < metrics['rollout/impact_hit'] < .5 and metrics['rollout/impact'] > 0
        after = env.rollout_metrics()  # nothing stepped since the last call
        assert math.isnan(after['rollout/impact']) and math.isnan(after['rollout/impact_hit'])
    finally:
        env.close()

    def refuse(*args, **kwargs):
        raise AssertionError('impact computed with its weight at 0')
    for name in ('impact_contacts', 'approach_speed', 'impact_penalty', 'impact_hit'):
        monkeypatch.setattr(task, name, refuse)
    plain = TensorG1GetupEnv(n, seed=6, observation_noise=False, randomize=False, bank=bank)
    try:
        assert plain.impact_substeps is None
        plain.reset()
        plain.set_starts(rows)
        for _ in range(15):
            plain.step(torch.zeros((n, 29), device=plain.device))
            plain.check_health()
        assert set(plain.rollout_metrics()) == {'rollout/assist', 'rollout/getup_success', 'rollout/nefc_max',
                                                'rollout/ncon_max', 'rollout/bad_states'}
    finally:
        plain.close()


# g1_getup reward weight sole_impact (iteration 44): the soles' floor approach speeds on CUDA, in the pass impact's
# speeds come from. The rows of tests/test_g1_getup_sole_impact.py's sole_bank (no start bank needed): the home pose
# upright with the soles 5 cm and 1 cm up, standing on the floor, and drop_bank's back and front drops 0.3 m up.
GETUP_SOLE_IMPACT = {'impact': 2., 'sole_impact': 2.}


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_sole_impact(monkeypatch):
    """Both weights on, the standing drops from 5 and 1 cm and the back and front drops on CUDA and on the CPU task
    (zero action holds the pose; no assist, delay, sensor noise or COM offset). For the standing drops sole_impact
    (per foot), for the lying ones impact (per body): the first floor contact comes in the same control step and
    physics substep on both, the speeds of the entries touching on both agree within 10 % (the soles at √(2gh) within
    5 %: 0.99 and 0.44 m/s), and that step's term (task.getup_reward_components on both) agrees within 10 % (the 1 cm
    drop, below 0.5 m/s: 0 on the CPU, within 0.01 on CUDA); every step before charges 0. The standing drops never
    charge impact."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS
    from test_g1_getup_sole_impact import SOFT, SOFT_DROP, SOLE_DROP, STAND_DROP, sole_bank
    bank = sole_bank()
    rows = [STAND_DROP, SOFT, 0, 1]
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), impact=host(components['impact']),
                         sole_impact=host(components['sole_impact'])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    cpus = [G1GetupEnv(bank=bank, reward_weights=GETUP_SOLE_IMPACT) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                           reward_weights=GETUP_SOLE_IMPACT)
    try:
        assert tuple(gpu.sole_impact_substeps.shape) == (len(rows), 10, 2)
        assert tuple(gpu.impact_substeps.shape) == (len(rows), 10, gpu.reference.model.nbody)
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu._obs('fill')
        first = {}
        free_fall = {STAND_DROP: math.sqrt(2 * 9.81 * SOLE_DROP), SOFT: math.sqrt(2 * 9.81 * SOFT_DROP)}
        terms = ['sole_impact', 'sole_impact', 'impact', 'impact']  # per world, the term compared
        for step in range(40):
            seen.clear()
            for cpu in cpus:
                cpu.step(np.zeros(29))
            _, _, terminated, _, _ = gpu.step(torch.zeros((len(rows), 29), device=gpu.device))
            gpu.check_health()
            assert not terminated.any()
            cpu_calls = [call for call in seen if not call['gpu']]
            (gpu_call,) = [call for call in seen if call['gpu']]
            speeds = dict(sole_impact=gpu.sole_impact_substeps.cpu().numpy(), impact=gpu.impact_substeps.cpu().numpy())
            for world, (cpu, row, term) in enumerate(zip(cpus, rows, terms)):
                if term == 'sole_impact':  # the standing drops touch with the soles only
                    assert not cpu.impact_substeps.any() and not speeds['impact'][world].any(), (world, step)
                    assert cpu_calls[world]['impact'][0] == 0 and gpu_call['impact'][world] == 0
                if world in first:
                    continue
                wanted = cpu.sole_impact_substeps if term == 'sole_impact' else cpu.impact_substeps
                got = speeds[term][world]
                cpu_term, gpu_term = cpu_calls[world][term][0], gpu_call[term][world]
                touched = np.flatnonzero(wanted.max(1) > 0), np.flatnonzero(got.max(1) > 0)
                if not len(touched[0]) and not len(touched[1]):
                    assert cpu_term == 0 and gpu_term == 0, (world, step)
                    continue
                assert len(touched[0]) and len(touched[1]) and touched[0][0] == touched[1][0], (world, step, touched)
                substep = touched[0][0]
                both = np.flatnonzero((wanted[substep] > 0) & (got[substep] > 0))
                assert both.size, (world, step)
                np.testing.assert_allclose(got[substep, both], wanted[substep, both], rtol=.1)
                if row == SOFT:
                    assert cpu_term == 0 and gpu_term == pytest.approx(0, abs=.01)
                else:
                    assert cpu_term < -.8 and gpu_term == pytest.approx(cpu_term, rel=.1)
                if row in free_fall:
                    np.testing.assert_allclose(wanted[substep, both], free_fall[row], rtol=.05)
                first[world] = step, substep
            if len(first) == len(rows):
                break
        assert sorted(first) == [0, 1, 2, 3], first
        assert first[1][0] < first[0][0] < first[2][0]  # from 1 cm, then 5 cm, then the lying ones from 0.3 m
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_sole_impact_is_logged_per_rollout_and_off_without_the_weight(monkeypatch):
    """64 worlds, 16 on each of the standing drops from 5 and 1 cm, standing on the floor and the back drop, sole_impact
    alone (no impact buffer), zero actions for 25 steps: rollout/sole_impact is the mean unweighted term per world-step,
    rollout/sole_impact_hit the share of world-steps with a foot above 0.5 m/s and rollout/flight the share of
    world-steps 40 ms or more into a stretch without any floor contact (the host's count from each step's touching), as
    the steps' own speeds give them, counted since the last call. The 5 cm drops hit and fly (0.10 s), the 1 cm ones fly
    (0.045 s: two steps without contact) but land below 0.5 m/s, the standing ones do neither, the back drops fly. With
    impact alone no sole buffer is made and nothing of it logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_getup_sole_impact import SOFT, STAND_DROP, STANDING, sole_bank
    bank = sole_bank()
    n = 64
    kinds = torch.tensor([STAND_DROP, SOFT, STANDING, 0])
    rows = kinds[torch.arange(n) % 4]
    env = TensorG1GetupEnv(n, seed=6, observation_noise=False, randomize=False, bank=bank,
                           reward_weights={'sole_impact': 2.})
    try:
        assert env.impact_substeps is None and tuple(env.sole_impact_substeps.shape) == (n, 10, 2)
        assert env.flight_steps == 2
        env.reset()
        env.set_starts(rows)
        env.set_action_delay_steps(0)
        env.set_assist(0.)
        env.rollout_metrics()
        zero = torch.zeros((n, 29), device=env.device)
        penalties, hits, flights = [], [], []
        airborne = torch.zeros(n, dtype=torch.long)
        for _ in range(25):
            env.step(zero)
            env.check_health()
            speed = env.sole_impact_substeps.amax(dim=1)
            penalties.append(task.impact_penalty(speed, task.GETUP_SOLE_IMPACT_SPEED).cpu())
            hits.append(task.impact_hit(speed, task.GETUP_SOLE_IMPACT_SPEED).cpu())
            airborne = torch.where(env.touching.cpu(), 0, airborne + 1)
            torch.testing.assert_close(env.airborne.cpu(), airborne, rtol=0, atol=0)
            flights.append(airborne >= 2)
        penalties, hits, flights = torch.stack(penalties), torch.stack(hits), torch.stack(flights)
        metrics = env.rollout_metrics()
        assert 'rollout/impact' not in metrics
        assert metrics['rollout/sole_impact'] == pytest.approx(float(penalties.mean()), rel=1e-5)
        assert metrics['rollout/sole_impact_hit'] == pytest.approx(float(hits.float().mean()), rel=1e-6)
        assert metrics['rollout/flight'] == pytest.approx(float(flights.float().mean()), rel=1e-6)
        for kind, hit, flew in ((STAND_DROP, True, True), (SOFT, False, True), (STANDING, False, False),
                                (0, None, True)):
            worlds = rows == kind
            flew_each = flights[:, worlds].any(dim=0)
            assert bool(flew_each.all()) if flew else not bool(flew_each.any()), kind
            if hit is not None:
                hit_each = hits[:, worlds].any(dim=0)
                assert bool(hit_each.all()) if hit else not bool(hit_each.any()), kind
        after = env.rollout_metrics()  # nothing stepped since the last call
        assert all(math.isnan(after[key]) for key in ('rollout/sole_impact', 'rollout/sole_impact_hit',
                                                       'rollout/flight'))
    finally:
        env.close()
    impact = TensorG1GetupEnv(n, seed=6, observation_noise=False, randomize=False, bank=bank,
                              reward_weights={'impact': 2.})
    try:
        assert impact.sole_impact_substeps is None and impact.impact_substeps is not None
        impact.reset()
        impact.set_starts(rows)
        for _ in range(10):
            impact.step(torch.zeros((n, 29), device=impact.device))
            impact.check_health()
        logged = impact.rollout_metrics()
        assert 'rollout/impact' in logged and not {'rollout/sole_impact', 'rollout/sole_impact_hit',
                                                   'rollout/flight'} & set(logged)
    finally:
        impact.close()


# g1_getup reward weight stance_width (after branch 46): the feet's width on CUDA. The rows of
# tests/test_g1_getup_stance_width.py's stance_bank: standing in home and in the wide stance on the floor, the standing
# drop from 5 cm and drop_bank's back drop.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_stance_width(monkeypatch):
    """The weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset), 20 steps: the wide
    stance held by its PD targets, the others with zero action. Per world and step both pass
    task.getup_reward_components the same W (task.stance_width; within 2 mm, the lying back drop too until it lands:
    its heading from the pelvis y axis, GETUP_STANCE_HEADING), count the term in the same steps
    (task.stance_active: home and the wide stance always, the back drop never, the drop once landed) and charge it
    alike (within 0.005; −2·(W − 0.30) in the wide stance, 0 standing in home); rollout/stance_width is the mean W over
    the world-steps that counted, as the steps' own inputs give it. Without the weight nothing is read or logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS
    from test_g1_getup_sole_impact import STAND_DROP, STANDING
    from test_g1_getup_stance_width import WIDE, WORKING, hold, stance_bank
    bank = stance_bank()
    rows = [STANDING, WIDE, STAND_DROP, 0]
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), width=host(kwargs['stance_width']),
                         active=host(task.stance_active(args[1], args[9])).astype(bool),
                         term=host(components['stance_width'])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    cpus = [G1GetupEnv(bank=bank, reward_weights=WORKING) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                           reward_weights=WORKING)
    try:
        assert gpu.stance_weighted
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu._obs('fill')
        gpu.rollout_metrics()
        actions = np.stack([hold(cpus[0]) if row == WIDE else np.zeros(29) for row in rows])
        widths, counted = [], []
        for step in range(20):
            seen.clear()
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            _, _, terminated, _, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not terminated.any()
            cpu_calls = [call for call in seen if not call['gpu']]
            (gpu_call,) = [call for call in seen if call['gpu']]
            cpu_width = np.array([call['width'][0] for call in cpu_calls])
            cpu_active = np.array([call['active'][0] for call in cpu_calls])
            # W everywhere while the back drop falls (it lands at step 13; lying, its pelvis x axis is vertical and the
            # heading comes from the y axis), then where the term counts on both
            compared = np.ones(len(rows), bool) if step < 10 else gpu_call['active'] & cpu_active
            np.testing.assert_allclose(gpu_call['width'][compared], cpu_width[compared], rtol=0, atol=.002,
                                       err_msg=f'step {step}')
            # the landing drop may bounce over the 1 cm sole threshold a step apart: compared where both count
            np.testing.assert_array_equal(gpu_call['active'][[0, 1, 3]], cpu_active[[0, 1, 3]], err_msg=f'step {step}')
            same = gpu_call['active'] == cpu_active
            np.testing.assert_allclose(gpu_call['term'][same], np.array([call['term'][0] for call in cpu_calls])[same],
                                       rtol=0, atol=.005, err_msg=f'step {step}')
            assert gpu_call['active'][:2].all() and not gpu_call['active'][3], step
            assert gpu_call['term'][0] == 0 and gpu_call['term'][1] < -.15, step
            widths.append(gpu_call['width'])
            counted.append(gpu_call['active'])
        widths, counted = np.array(widths), np.array(counted)
        assert counted[-5:, 2].all()  # the drop landed and stands
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/stance_width'] == pytest.approx(widths[counted].mean(), rel=1e-5)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()

    def refuse(*args, **kwargs):
        raise AssertionError('stance_width computed with its weight at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    for name in ('stance_width', 'stance_active'):
        monkeypatch.setattr(task, name, refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank)
    try:
        assert not plain.stance_weighted
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
            plain.check_health()
        assert 'rollout/stance_width' not in plain.rollout_metrics()
    finally:
        plain.close()


# g1_getup task_config stand_pose_std (iteration 36): the standing pose term's σ on CUDA. The drop rows of
# tests/test_g1_getup_impact.py: the home pose on its back and on its front 0.3 m up, then standing in home.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_stand_pose_std(monkeypatch):
    """σ 0.7 on the CUDA and the CPU task, then the CUDA task without the setting (σ 0.1), zero actions for 15 steps
    (no assist, delay, sensor noise or COM offset): both pass their σ to task.getup_reward_components; on CUDA the pose
    term is w·[standing]·exp(−mean((q − home)²)/σ²) of its own joint angles (torch, float32), the standing world earns
    it on both (CPU and CUDA within 0.01; > 0.49 at 0.7), the lying ones nothing."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 1, 3]
    weight = G1_GETUP_REWARD_WEIGHTS['stand_pose']
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), std=kwargs.get('stand_pose_std'),
                         pose=host(components['stand_pose']), standing=host(standing).astype(bool),
                         position=host(args[4]), home=host(args[5])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    for std, settings in ((.7, {'stand_pose_std': .7}), (.1, {})):
        cpus = [G1GetupEnv(bank=bank, stand_pose_std=std) for _ in rows]
        gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
        try:
            assert gpu.stand_pose_std == std
            for cpu, row in zip(cpus, rows):
                cpu.reset(seed=4, options=dict(OPTIONS, start=row))
            gpu.reset()
            gpu.set_starts(rows)
            gpu.set_action_delay_steps(0)
            gpu.set_motor_friction(1.)
            gpu.set_body_friction(.6)
            gpu.set_assist(0.)
            gpu._obs('fill')
            for step in range(15):
                seen.clear()
                for cpu in cpus:
                    cpu.step(np.zeros(29))
                _, _, terminated, _, _ = gpu.step(torch.zeros((len(rows), 29), device=gpu.device))
                gpu.check_health()
                assert not terminated.any()
                cpu_calls = [call for call in seen if not call['gpu']]
                (gpu_call,) = [call for call in seen if call['gpu']]
                assert gpu_call['std'] == std and [call['std'] for call in cpu_calls] == [std] * len(rows)
                mean = ((gpu_call['position'] - gpu_call['home']) ** 2).mean(-1)
                expected = weight * gpu_call['standing'] * np.exp(-mean / std ** 2)
                np.testing.assert_allclose(gpu_call['pose'], expected, rtol=1e-5, atol=1e-7, err_msg=f'step {step}')
                cpu_pose = np.array([call['pose'][0] for call in cpu_calls])
                np.testing.assert_allclose(gpu_call['pose'], cpu_pose, rtol=0, atol=.01, err_msg=f'step {step}')
                assert gpu_call['standing'].tolist() == [False, False, True], step
                assert [bool(call['standing'][0]) for call in cpu_calls] == [False, False, True], step
                assert gpu_call['pose'][2] > (.49 if std == .7 else 0) and not gpu_call['pose'][:2].any()
        finally:
            for cpu in cpus:
                cpu.close()
            gpu.close()


# g1_getup task_config action_change_unbounded (iteration 37): the action_change term on the action before pd_bound on
# CUDA. The drop rows of tests/test_g1_getup_impact.py: the home pose on its back, on its front, on its front spinning,
# 0.3 m up, then standing in home.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_action_change_unbounded(monkeypatch):
    """The CUDA and the CPU task with pd_bound at c 0.15 under a 10 ms delay (no assist, sensor noise or COM offset),
    actions over and beyond the ±10 box, first with the setting, then without. With it each world's previous_unbounded
    is exactly lock_joints of its box-clipped action and previous_action the bounded one (the PD, the delay queue and
    the observation keep it); task.getup_reward_components gets the actions before the bound and those of the step
    before (zero after a reset), its action_change term is −w·Σ(a − a_prev)² of them and the CPU task's within float32
    (it depends on the actions alone, not on the physics); a reset of one world zeroes its previous one only. Without
    it the term gets the bounded actions and there is no such state. The other terms match the CPU task's within the
    tolerances of the test without the setting."""
    import warp as wp
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 1, 2, 3]
    weight = G1_GETUP_REWARD_WEIGHTS['action_change']
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), action=host(args[12]), previous=host(args[13]),
                         terms={key: host(value) for key, value in components.items()}))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    options = dict(OPTIONS, action_delay_steps=5, pd_bound=.15)
    for flag in (True, False):
        settings = dict(pd_bound=GETUP_PD_BOUND, action_change_unbounded=flag)
        cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
        gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
        try:
            assert (gpu.previous_unbounded is not None) is flag
            for cpu, row in zip(cpus, rows):
                cpu.reset(seed=4, options=dict(options, start=row))
            gpu.reset()
            gpu.set_starts(rows)
            gpu.set_action_delay_steps(5)
            gpu.set_motor_friction(1.)
            gpu.set_body_friction(.6)
            gpu.set_assist(0.)
            gpu.set_pd_bound(.15)
            gpu._obs('fill')
            if flag:
                assert not gpu.previous_unbounded.any()
            rng = np.random.default_rng(0)
            before = torch.zeros((len(rows), 29), device=gpu.device)
            for step in range(3):
                action = rng.uniform(-12., 12., 29)
                actions = torch.tensor(action, dtype=torch.float32, device=gpu.device).expand(len(rows), -1)
                actions = actions.contiguous()
                clipped = actions.clamp(-10., 10.)
                position = gpu.frames[:, -1, gpu.position_entries].clone()
                pre = task.lock_joints(clipped, gpu.wrist_mask)
                bounded = task.lock_joints(task.pd_bound(clipped, position, gpu.bound, gpu.peak, gpu.kp, gpu.scale,
                                                         gpu.bound_free), gpu.wrist_mask)
                seen.clear()
                for cpu in cpus:
                    cpu.step(action)
                _, _, terminated, _, _ = gpu.step(actions)
                gpu.check_health()
                assert not terminated.any()
                torch.testing.assert_close(gpu.previous_action, bounded, rtol=0, atol=0)
                charged = pre if flag else bounded
                cpu_calls = [call for call in seen if not call['gpu']]
                (gpu_call,) = [call for call in seen if call['gpu']]
                np.testing.assert_array_equal(gpu_call['action'], charged.cpu().numpy())
                np.testing.assert_array_equal(gpu_call['previous'], before.cpu().numpy())
                expected = -weight * ((charged.double() - before.double()) ** 2).sum(-1).cpu().numpy()
                np.testing.assert_allclose(gpu_call['terms']['action_change'], expected, rtol=1e-5,
                                           err_msg=f'step {step}')
                if flag:
                    torch.testing.assert_close(gpu.previous_unbounded, pre, rtol=0, atol=0)
                    np.testing.assert_allclose(np.stack([cpu.previous_unbounded for cpu in cpus]), pre.cpu().numpy(),
                                               rtol=0, atol=2e-6)
                    cpu_change = np.array([call['terms']['action_change'][0] for call in cpu_calls])
                    np.testing.assert_allclose(gpu_call['terms']['action_change'], cpu_change, rtol=1e-5,
                                               err_msg=f'step {step}')  # the actions alone: float32 against float64
                    assert bool((pre != bounded).any())  # the bound acted: not the bounded actions' term
                for key in (key for key in gpu_call['terms'] if key != 'action_change'):
                    np.testing.assert_allclose(gpu_call['terms'][key], [call['terms'][key][0] for call in cpu_calls],
                                               atol=.03, rtol=.02, err_msg=f'{key} step {step}')
                before = charged
            if flag:  # one world's reset zeroes its previous action before the bound, the others keep theirs
                kept = gpu.previous_unbounded[1:].clone()
                assert bool(gpu.previous_unbounded[0].any())
                mask = torch.zeros(len(rows), dtype=torch.bool, device=gpu.device)
                mask[0] = True
                with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
                    gpu._masked_reset(mask)
                assert not gpu.previous_unbounded[0].any()
                torch.testing.assert_close(gpu.previous_unbounded[1:], kept, rtol=0, atol=0)
        finally:
            for cpu in cpus:
                cpu.close()
            gpu.close()


# task_config arm_targets (iteration 40, stage 3): the arm command in the upper-body slot, the residual arm action on home
# moved by the targets, per-world modes, targets and hand loads (batched body masses), the load curve.
ARM_TARGET_PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.,
                          gait={'period': .6, 'stance_fraction': .55}, height_command=True, upper_body_slot=True,
                          lock_wrists=True, arm_targets=True,
                          reward_weights={'arm_track': 2., 'posture': .3, 'arm_press': .1, 'action_change': .05})
ARM_TRACKED = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]  # shoulders, elbows, wrist roll: the joints arm_track follows


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_arm_targets_and_hand_loads(monkeypatch):
    """Two worlds against the CPU task with the same arm command (set_arm_targets: world 0 a pose off home with both
    shoulders rolled outward, world 1 a carry pose of the library, both in a 0.2 s quintic) and hand loads
    (set_hand_load: 1 / 2 kg and 0.5 / 0 kg), world 1 with the motors 15 substeps late: the batched body masses, COMs and
    subtree masses are the CPU model's and so is the subtree COM; over 12 steps the targets follow the same quintic, the
    slot (every frame) and the observed last action (clip(a) + d/scale) match, the delay queue keeps each step's
    offsets, and the reward's arm_track, posture and arm_press (0 with flag 1) match the CPU ones. The poses keep the
    arms off the body (checked on the CPU every step): with the left shoulder rolled inward instead (0.18 − 0.39 rad) the
    left wrist_yaw mesh goes into the left hip at step 9 (5-31 N), a mesh-mesh contact for which MuJoCo C and MJWarp make
    different contact points, and from there the left arm's joint speeds differ by up to 0.4 rad/s, loads or not."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen.append((args, kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = {**ARM_TARGET_PROFILE, 'action_delay': .03}
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        poses = np.stack([cpus[0].home, cpus[0].home + cpus[0].arm_carry[5].numpy()])
        # Left: pitch, roll (outward), yaw, elbow, wrist roll up; right: the same parts, its roll outward (negative).
        poses[0, ARM_TRACKED] += np.abs(np.linspace(-.5, .5, 10)) * np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, -1])
        loads = np.array([[1., 2.], [.5, 0.]])
        for cpu, steps, pose, load in zip(cpus, (0, 15), poses, loads):
            cpu.reset(seed=4, options={'command': [0., 0., 0.], 'action_delay_steps': steps})
            cpu.set_arm_targets(pose, 1, seconds=.2)
            cpu.set_hand_load(load)
        gpu.reset()
        gpu.set_action_delay_steps(torch.tensor([0, 15]))
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        gpu.set_arm_targets(torch.tensor(poses, dtype=torch.float32), 1, seconds=.2)
        gpu.set_hand_load(torch.tensor(loads, dtype=torch.float32))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        root = gpu.root
        for world, cpu in enumerate(cpus):
            np.testing.assert_allclose(gpu.body_mass[world].cpu().numpy(), cpu.model.body_mass, rtol=1e-6)
            np.testing.assert_allclose(gpu.body_ipos[world].cpu().numpy(), cpu.model.body_ipos, atol=1e-7)
            np.testing.assert_allclose(gpu.body_subtreemass[world].cpu().numpy(), cpu.model.body_subtreemass, rtol=1e-6)
            mujoco.mj_forward(cpu.model, cpu.data)  # the same state with the loads
            np.testing.assert_allclose(gpu.subtree_com[world, root].cpu().numpy(), cpu.data.subtree_com[root],
                                       atol=2e-5)
        start = gpu._obs('fill').cpu()
        for world, cpu in enumerate(cpus):
            np.testing.assert_allclose(start[world], cpu._observation('fill'), atol=2e-5, rtol=2e-5)
        slot = np.array(task.term_positions(29, 5, cpus[0].extra_terms, task.UPPER_BODY_TERM)).reshape(5, 18)
        last = np.array(task.newest_frame_columns(29, 5, cpus[0].extra_sizes, term=6))
        for step in range(12):
            action = .4 * np.sin(np.arange(29) + 1.7 * step)
            results = [cpu.step(action) for cpu in cpus]
            actions = torch.tensor(action, device=gpu.device, dtype=torch.float32)[None].expand(2, -1).contiguous()
            _, reward, terminated, _, terminal = gpu.step(actions)
            gpu.check_health()
            args, kwargs, components = seen[-1]
            assert bool(kwargs['arm_flag'].all()) and not terminated.any()
            for world, (obs, value, _, _, info) in enumerate(results):
                cpu = cpus[world]
                geom = cpu.data.contact.geom
                assert not (cpu.robot_geoms[geom].all(axis=1) & cpu.arm_geoms[geom].any(axis=1)).any()  # arms clear
                np.testing.assert_allclose(gpu.previous_offset[world].cpu().numpy(), cpu.previous_offset, atol=1e-6)
                np.testing.assert_allclose(gpu.arm_offset[world].cpu().numpy(), cpu.arm_offset, atol=1e-6)
                np.testing.assert_allclose(terminal.cpu()[world, slot.ravel()], obs[slot.ravel()], atol=1e-6)
                np.testing.assert_allclose(terminal.cpu()[world, last], obs[last], atol=.03, rtol=.02)
                np.testing.assert_allclose(terminal.cpu()[world], obs, atol=.03, rtol=.02)
                np.testing.assert_allclose(reward.cpu()[world], value, atol=.03, rtol=.02)
                parts = info['reward_components']
                for name in ('arm_track', 'posture'):
                    np.testing.assert_allclose(float(components[name][world]), parts[name], atol=2e-3, rtol=1e-2)
                assert float(components['arm_press'][world]) == 0 == parts['arm_press']
            if gpu.delay_depth:
                np.testing.assert_allclose(gpu.delay_offsets[1, 0].cpu().numpy(), cpus[1].delay_offsets[0], atol=1e-6)
        assert float(gpu.arm_offset[0, 15]) == pytest.approx(poses[0, 15] - cpus[0].home[15], abs=1e-6)  # held
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_arm_targets_draw_per_world_run_the_curve_and_leave_the_task_stream_alone():
    """4096 worlds in training (arm perturbation at share 0.5, random delay, COM spread, command resampling): the modes
    in ARM_MODES' shares, the slot's newest frame upper_body_command(flag, d), no perturbation segment in a flag-1
    world, the targets moving in the targets mode; a restored ceiling of 1 kg loads each flag-1 world's hands with its
    shares into the batched masses (the pelvis COM spread kept); rollout_metrics logs the arm numbers and moves the
    ceiling by task.load_curve_update, the loads following it; the task's own generator is that of the same task
    without arm_targets, step for step. Without the setting: no arm logs, no training_state, no arm API."""
    from humanoid_sim.tasks import g1_tensor_math as task
    n = 4096
    plain_profile = {key: value for key, value in ARM_TARGET_PROFILE.items() if key != 'arm_targets'}
    profile = dict(push_velocity=0, observation_noise=False, randomize=True, arm_perturb=.5, arm_perturb_radians=1.,
                   action_delay=.02, **plain_profile)
    arms, plain = TensorG1Env(n, seed=5, arm_targets=True, **profile), TensorG1Env(n, seed=5, **profile)
    try:
        obs, _ = arms.reset(), plain.reset()
        shares = torch.bincount(arms.arm_mode, minlength=3).double() / n
        np.testing.assert_allclose(shares.cpu().numpy(), task.ARM_MODES, atol=.03)
        assert torch.equal(arms.arm_flag, arms.arm_mode != task.ARM_FREE)
        slot = torch.as_tensor(task.term_positions(29, 5, arms.reference.extra_terms, task.UPPER_BODY_TERM),
                               device=arms.device).reshape(5, 18)
        torch.testing.assert_close(obs[:, slot[-1]][:, 0], arms.arm_flag.float())
        assert not arms.hand_load.any()  # the ceiling starts at 0
        arms.restore_training_state({'load_max': [1.]})
        flag = arms.arm_flag
        assert arms.load_max == 1. and float(arms.hand_load[~flag].abs().max()) == 0
        torch.testing.assert_close(arms.hand_load[flag], arms.load_share[flag])
        hands = arms.hand_bodies
        torch.testing.assert_close(arms.body_mass[:, hands], arms.hand_mass + arms.hand_load)
        assert float(arms.body_ipos[:, arms.pelvis].std(dim=0).min()) > 1e-3  # the COM spread is still there
        action = torch.zeros((n, 29), device=arms.device)
        for _ in range(30):
            obs, *_ = arms.step(action)
            plain.step(action)
            assert not bool((arms.perturb_steps[arms.arm_flag] > 0).any())
            torch.testing.assert_close(obs[:, slot[-1]], task.upper_body_command(arms.arm_flag, arms.arm_offset))
            assert torch.equal(arms.generator.get_state(), plain.generator.get_state())
        moving = arms.arm_mode == task.ARM_TARGET
        assert float(arms.arm_offset[moving].abs().max()) > .05
        assert float(arms.arm_offset[~arms.arm_flag].abs().max()) == 0
        assert bool((plain.perturb_steps > 0).any())
        metrics = arms.rollout_metrics()
        assert {'rollout/arm_track', 'rollout/arm_error', 'rollout/arm_falls', 'rollout/load_max'} <= set(metrics)
        assert 0 < metrics['rollout/arm_track'] <= 1 and metrics['rollout/arm_error'] > 0
        level = arms.load_max
        arms._arm_steps.fill_(75000.), arms._arm_track_sum.fill_(.9 * 75000), arms._arm_falls.zero_()
        assert arms.rollout_metrics()['rollout/load_max'] == pytest.approx(min(level + .1, 2.))
        torch.testing.assert_close(arms.hand_load, arms.load_share * arms.load_max)
        torch.testing.assert_close(arms.body_mass[:, hands], arms.hand_mass + arms.hand_load)
        arms._arm_steps.fill_(75000.), arms._arm_track_sum.fill_(.9 * 75000), arms._arm_falls.fill_(3.)
        assert arms.rollout_metrics()['rollout/load_max'] == pytest.approx(max(min(level + .1, 2.) - .1, 0.))
        assert arms.training_state() == {'load_max': [arms.load_max]} and plain.training_state() is None
        new_logs = {'rollout/arm_track', 'rollout/arm_error', 'rollout/arm_falls', 'rollout/load_max'}
        assert not new_logs & set(plain.rollout_metrics())  # its rollout/arm_perturb and rollout/arm_press stay
        for call in (lambda: plain.set_arm_targets(None, 1), lambda: plain.set_hand_load(1.)):
            with pytest.raises(ValueError):
                call()
        arms.check_health()
        plain.check_health()
    finally:
        arms.close(), plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_arm_targets_full_gpu_learner_freezes_the_arm_columns_and_keeps_the_load_ceiling(tmp_path):
    """full_gpu_train with arm_targets: TensorNormalize keeps the arm columns at mean 0 and their fixed variances through
    a rollout and an update (with the mirror loss); a stage saved with the ceiling (training_state) gives it back to the
    next learner's worlds before their first reset, their loads its shares."""
    from humanoid_sim import full_gpu_train
    from humanoid_sim.core import read_json, save_stage, write_json
    from humanoid_sim.lab import validate_config
    config = {**validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', backend='full_gpu', envs=64, n_steps=8,
                                     batch_size=512, n_epochs=1, steps=1024, mirror_coef=.3,
                                     task_config=ARM_TARGET_PROFILE)), 'steps': 512}
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    learner, export = full_gpu_train.make_learner(config)
    try:
        positions, mean, variance = learner.env.fixed_normalization
        columns = torch.as_tensor(positions[-210:], device=learner.device)
        learner.collect()
        learner.update()
        assert not learner.normalizer.obs.mean[columns].any()
        torch.testing.assert_close(learner.normalizer.obs.var[columns].cpu(), torch.as_tensor(variance[-210:]),
                                   rtol=0, atol=0)
        learner.env.restore_training_state({'load_max': [1.3]})
        learner.normalizer.export_normalizer(export)
        stage = save_stage(run, learner.model, export, learner.env.training_state())
        assert read_json(stage / 'metadata.json')['training_state'] == {'load_max': [1.3]}
    finally:
        learner.env.close(), export.close()
    resumed, export = full_gpu_train.make_learner({**config, 'envs': 32, 'batch_size': 256}, stage)
    try:
        assert resumed.env.load_max == pytest.approx(1.3)
        torch.testing.assert_close(resumed.env.hand_load, resumed.env.load_share * 1.3)
        assert float(resumed.env.hand_load.max()) > 0
    finally:
        resumed.env.close(), export.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_load_curve_decides_once_a_block_and_logs_the_free_falls():
    """task_config load_curve (iteration 40b) on CUDA: 64 worlds stepping 5 times, then rollout windows with set flag-1
    counts. With block 3 the ceiling holds through two windows (no rollout/arm_falls_block) and moves at the third on the
    three windows' sums (73728 steps each: 3, 3 and 2 falls are 3.6 % per episode, below 4 %: +0.1 kg, the loads
    following it); rollout/arm_falls stays the window's own. Without the setting the per-window rule moves it at once
    (3 falls of a window, 4.1 %: -0.1 kg). rollout/free_falls is logged with arm_targets either way, from the flag-0
    episodes' own counts."""
    from humanoid_sim.tasks import g1_tensor_math as task
    profile = dict(push_velocity=0, observation_noise=False, randomize=False, **ARM_TARGET_PROFILE)
    curve = {'block': 3, 'raise_falls': .04, 'lower_falls': .06}
    curved, plain = TensorG1Env(64, seed=6, load_curve=curve, **profile), TensorG1Env(64, seed=6, **profile)
    try:
        assert curved.load_block is not None and plain.load_block is None
        for env in (curved, plain):
            env.reset()
            env.restore_training_state({'load_max': [1.]})
            for _ in range(5):
                env.step(torch.zeros((64, 29), device=env.device))
            env.check_health()
            steps = env._free_steps.item()
            assert steps + env._arm_steps.item() == 5 * 64  # every world-step counted on one side
            metrics = env.rollout_metrics()  # the real counts of these 5 steps: the block's first window for curved
            assert 'rollout/free_falls' in metrics and 'rollout/arm_falls_block' not in metrics
        assert curved.load_max == 1.  # no decision before the block is full
        curved.load_block = task.LoadCurveBlock(**curve)  # a block of the windows below only
        blocks = []
        for falls in (3., 3., 2.):
            curved._arm_steps.fill_(73728.), curved._arm_track_sum.fill_(.8 * 73728), curved._arm_falls.fill_(falls)
            curved._free_steps.fill_(24576.), curved._free_falls.fill_(1.)
            metrics = curved.rollout_metrics()
            blocks.append((metrics['rollout/load_max'], metrics.get('rollout/arm_falls_block')))
            assert metrics['rollout/arm_falls'] == pytest.approx(falls * 1000 / 73728)
            assert metrics['rollout/free_falls'] == pytest.approx(1000 / 24576)
        assert blocks[0] == (1., None) and blocks[1] == (1., None)
        assert blocks[2][0] == pytest.approx(1.1) and blocks[2][1] == pytest.approx(8 * 1000 / (3 * 73728))
        torch.testing.assert_close(curved.hand_load, curved.load_share * curved.load_max)
        plain._arm_steps.fill_(73728.), plain._arm_track_sum.fill_(.8 * 73728), plain._arm_falls.fill_(3.)
        before = plain.load_max
        assert plain.rollout_metrics()['rollout/load_max'] == pytest.approx(max(before - .1, 0.))
        assert curved.training_state() == {'load_max': [curved.load_max]}  # the block's counts are not kept
        curved.check_health()
        plain.check_health()
    finally:
        curved.close(), plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_arm_error_on_holds(monkeypatch):
    """Reward weight arm_error (iteration 43) at 6: two worlds against the CPU task, both arms to the poses of
    test_g1_cpu_gpu_arm_targets_and_hand_loads in a 0.2 s quintic (10 steps), 1 / 2 kg and 0.5 / 0 kg on the hands. Over
    16 steps the holds (task.arm_holding) are the CPU task's, none in the quintic and all from its 11th step; arm_error
    is 0 while the targets move and −6·mean |q − target| on a hold, the CPU one's; rollout/arm_error_hold is the mean
    error over the hold world-steps. The arms stay off the body (checked on the CPU every step)."""
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen.append((kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = {**ARM_TARGET_PROFILE, 'reward_weights': {**ARM_TARGET_PROFILE['reward_weights'], 'arm_error': 6.}}
    cpus = [G1VelocityEnv(**profile) for _ in range(2)]
    gpu = TensorG1Env(2, seed=4, resample_commands=False, push_velocity=0, observation_noise=False, randomize=False,
                      **profile)
    try:
        poses = np.stack([cpus[0].home, cpus[0].home + cpus[0].arm_carry[5].numpy()])
        poses[0, ARM_TRACKED] += np.abs(np.linspace(-.5, .5, 10)) * np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, -1])
        loads = np.array([[1., 2.], [.5, 0.]])
        for cpu, pose, load in zip(cpus, poses, loads):
            cpu.reset(seed=4, options={'command': [0., 0., 0.]})
            cpu.set_arm_targets(pose, 1, seconds=.2)
            cpu.set_hand_load(load)
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.zero_()
        gpu.set_arm_targets(torch.tensor(poses, dtype=torch.float32), 1, seconds=.2)
        gpu.set_hand_load(torch.tensor(loads, dtype=torch.float32))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs('fill')
        gpu.rollout_metrics()  # a fresh window
        held, held_cpu = [], []
        for step in range(16):
            action = .4 * np.sin(np.arange(29) + 1.7 * step)
            results = [cpu.step(action) for cpu in cpus]
            actions = torch.tensor(action, device=gpu.device, dtype=torch.float32)[None].expand(2, -1).contiguous()
            _, reward, terminated, _, _ = gpu.step(actions)
            gpu.check_health()
            kwargs, components = seen[-1]
            assert not terminated.any() and bool(kwargs['arm_flag'].all())
            assert kwargs['arm_hold'].tolist() == [info['arm_hold'] for *_, info in results] == [step >= 10] * 2
            for world, (_, value, _, _, info) in enumerate(results):
                cpu = cpus[world]
                geom = cpu.data.contact.geom
                assert not (cpu.robot_geoms[geom].all(axis=1) & cpu.arm_geoms[geom].any(axis=1)).any()  # arms clear
                part = info['reward_components']['arm_error']
                assert part == (pytest.approx(-6 * info['arm_error'], rel=1e-12) if step >= 10 else 0)
                np.testing.assert_allclose(float(components['arm_error'][world]), part, atol=1e-2, rtol=2e-2)
                np.testing.assert_allclose(reward.cpu()[world], value, atol=.03, rtol=.02)
                if step >= 10:
                    held.append(float(kwargs['arm_error'][world]))
                    held_cpu.append(info['arm_error'])
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/arm_error_hold'] == pytest.approx(np.mean(held), rel=1e-5)
        assert metrics['rollout/arm_error_hold'] == pytest.approx(np.mean(held_cpu), rel=2e-2)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


# g1_getup reward weight arm_press (iteration 52): standing, the free arm joints' target before pd_bound against their
# angles, on CUDA. The rows of tests/test_g1_getup_stance_width.py's stance_bank: standing in home and in the wide
# stance, the standing drop from 5 cm and drop_bank's back drop; a PD error bound c 0.15 clamps the arms' targets.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_arm_press(monkeypatch):
    """The weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset; pd_bound at c 0.15),
    15 steps of the same actions (the arms random, now and then beyond the box; the legs at home or holding the wide
    stance): per world both pass task.getup_reward_components the same Σ (within 1 %) and standing, and charge
    −w·[standing]·Σ alike; the back drop is never charged; rollout/arm_press is the mean Σ over the standing
    world-steps, as the steps' own inputs give it. Without the weight nothing is read or logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS
    from test_g1_getup_sole_impact import STAND_DROP, STANDING
    from test_g1_getup_stance_width import WIDE, hold, stance_bank
    bank = stance_bank()
    rows = [STANDING, WIDE, STAND_DROP, 0]
    settings = dict(reward_weights={'arm_press': .05}, pd_bound=[1., .6])
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), press=host(kwargs['arm_press']),
                         standing=host(standing).astype(bool), term=host(components['arm_press'])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
    rng = np.random.default_rng(0)
    try:
        assert gpu.press_weighted and gpu.press_joints.tolist() == [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row, pd_bound=.15))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu.set_pd_bound(.15)
        gpu._obs('fill')
        gpu.rollout_metrics()
        presses, standings = [], []
        for step in range(15):
            actions = np.stack([hold(cpus[0]) if row == WIDE else np.zeros(29) for row in rows])
            actions[:, 15:] += rng.normal(0., 12. if step % 5 == 0 else 2., (len(rows), 14))
            seen.clear()
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            _, _, terminated, _, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not terminated.any()
            cpu_calls = [call for call in seen if not call['gpu']]
            (gpu_call,) = [call for call in seen if call['gpu']]
            cpu_press = np.array([call['press'][0] for call in cpu_calls])
            cpu_standing = np.array([call['standing'][0] for call in cpu_calls])
            np.testing.assert_allclose(gpu_call['press'], cpu_press, rtol=.01, atol=.002, err_msg=f'step {step}')
            np.testing.assert_array_equal(gpu_call['standing'], cpu_standing, err_msg=f'step {step}')
            np.testing.assert_allclose(gpu_call['term'], -.05 * cpu_standing * cpu_press, rtol=.01, atol=1e-4,
                                       err_msg=f'step {step}')
            assert not gpu_call['standing'][3] and gpu_call['term'][3] == 0, step
            presses.append(gpu_call['press'])
            standings.append(gpu_call['standing'])
        presses, standings = np.array(presses), np.array(standings)
        assert standings[:, 0].sum() >= 5
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/arm_press'] == pytest.approx(presses[standings].mean(), rel=1e-5)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()

    def refuse(*args, **kwargs):
        raise AssertionError('arm_press computed with its weight at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    monkeypatch.setattr(task, 'arm_press', refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                             pd_bound=[1., .6])
    try:
        assert not plain.press_weighted
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
            plain.check_health()
        assert 'rollout/arm_press' not in plain.rollout_metrics()
    finally:
        plain.close()


# g1_getup reward weight style (iteration 53): each world's distance to the nearest human frame of the style library,
# paid with the head in task.STYLE_HEAD_BAND, on CUDA. drop_bank's rows: back, front and spinning drops (the head starts
# ~0.4 m up, then lies below 0.3 m) and standing in home (above the band); a library of tests/test_g1_getup_style.py.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_style(monkeypatch, tmp_path):
    """The weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset), 15 steps of the same
    random actions: per world both pass task.getup_reward_components the same d (within 1 %) and pay
    0.5·[band]·exp(−d/2) alike (the band from each one's head); the standing world is never paid; rollout/style_dist and
    rollout/style_band are the mean d over the band world-steps and their share, as the steps' own inputs give them.
    Without the weight nothing is loaded, computed or logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_getup_style
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_style import write_library
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    bank = drop_bank()
    rows = [0, 1, 2, 3]
    probe = G1GetupEnv(bank=bank)
    try:
        centres = []
        for row in rows[:3]:
            probe.reset(seed=1, options=dict(OPTIONS, start=row))
            centres.append(probe.style_features()[0])
    finally:
        probe.close()
    rng = np.random.default_rng(3)
    write_library(tmp_path, 'gpu', np.concatenate([c + rng.normal(0., .4, (300, len(c))) for c in centres]))
    settings = dict(reward_weights={'style': .5}, style_library='gpu')
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), style=host(kwargs['style']), head=host(args[0]),
                         term=host(components['style'])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
    try:
        assert gpu.style_weighted and gpu.style_joints.tolist() == list(cpus[0].style_joints)
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu._obs('fill')
        gpu.rollout_metrics()
        styles, bands = [], []
        for step in range(15):
            actions = rng.uniform(-10., 10., (len(rows), 29)) * (.3 if step % 4 else 1.)
            actions[3] = 0.  # the standing world holds home
            seen.clear()
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            _, _, terminated, _, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not terminated.any()
            cpu_calls = [call for call in seen if not call['gpu']]
            (gpu_call,) = [call for call in seen if call['gpu']]
            cpu_style = np.array([call['style'][0] for call in cpu_calls])  # +inf outside the band (not computed)
            cpu_head = np.array([call['head'][0] for call in cpu_calls])
            band = (gpu_call['head'] > .3) & (gpu_call['head'] < 1.1)
            np.testing.assert_array_equal(band, (cpu_head > .3) & (cpu_head < 1.1), err_msg=f'step {step}')
            assert np.isinf(cpu_style[~band]).all() and np.isfinite(gpu_call['style']).all(), step
            np.testing.assert_allclose(gpu_call['style'][band], cpu_style[band], rtol=.01, atol=.005,
                                       err_msg=f'step {step}')
            np.testing.assert_allclose(gpu_call['term'], .5 * band * np.exp(-gpu_call['style'] / 2), rtol=1e-5,
                                       atol=1e-7, err_msg=f'step {step}')
            cpu_term = np.array([call['term'][0] for call in cpu_calls])
            np.testing.assert_allclose(gpu_call['term'], cpu_term, rtol=.02, atol=1e-4, err_msg=f'step {step}')
            assert not band[3] and gpu_call['term'][3] == 0, step
            styles.append(gpu_call['style'])
            bands.append(band)
        styles, bands = np.array(styles), np.array(bands)
        assert bands.sum() >= 5 and (~bands[:, :3]).sum() >= 5
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/style_dist'] == pytest.approx(styles[bands].mean(), rel=1e-5)
        assert metrics['rollout/style_band'] == pytest.approx(bands.mean(), rel=1e-6)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()

    def refuse(*args, **kwargs):
        raise AssertionError('style computed or loaded with its weight at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    for name in ('style_features', 'style_distance'):
        monkeypatch.setattr(task, name, refuse)
    monkeypatch.setattr(g1_getup_style, 'load', refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                             style_library='gpu')
    try:
        assert not plain.style_weighted
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
            plain.check_health()
        assert not any(key.startswith('rollout/style') for key in plain.rollout_metrics())
    finally:
        plain.close()


# g1_getup reward weight amp (iteration 55): each world's state pair (getup_amp.amp_state before and after the step)
# and its band on CUDA, for the learner's discriminator; with it the style library judges (rollout/style_dist) without
# paying the style term. drop_bank's rows and a style library of tests/test_g1_getup_style.py, as the style test above.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_amp_pairs_and_the_style_judge(monkeypatch, tmp_path):
    """The amp weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset), 15 steps of the
    same random actions: per world the pair's halves are amp_state of the CPU task's state before and after the step
    (within 0.03; float32 and Warp against float64: 0.024 after 13 steps of random ±10 actions), the first half the
    second half of the step before exactly, the
    band the head band of the CPU task's head; the style term is not paid (style=None) but rollout/style_dist and
    rollout/style_band are logged. style_judge alone judges too; without either nothing is loaded or computed."""
    from humanoid_sim import getup_amp
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_getup_style
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_style import write_library
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    bank = drop_bank()
    rows = [0, 1, 2, 3]
    rng = np.random.default_rng(3)
    probe = G1GetupEnv(bank=bank)
    try:
        probe.reset(seed=1, options=dict(OPTIONS, start=0))
        centre = probe.style_features()[0]
    finally:
        probe.close()
    write_library(tmp_path, 'gpu', centre + rng.normal(0., .4, (300, len(centre))))
    settings = dict(reward_weights={'amp': 1.}, style_library='gpu')
    seen, original = [], task.getup_reward_components

    def spy(*args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            seen.append(kwargs['style'])
        return original(*args, **kwargs)
    monkeypatch.setattr(task, 'getup_reward_components', spy)

    def cpu_state(env):
        d = env.data
        bodies, offsets = getup_amp.point_bodies(env.model)
        head = task.head_height(d.xpos[env.torso][None], d.xmat[env.torso].reshape(1, 3, 3))
        return getup_amp.amp_state(d.qpos[None, 7:], d.qpos[None, :3], d.xmat[env.pelvis].reshape(1, 3, 3), head,
                                   getup_amp.amp_points(d.xpos[None], d.xmat[None], bodies, offsets),
                                   env.style_joints)[0], head[0]
    cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
    try:
        assert gpu.amp_weight == 1. and gpu.amp_library == 'gpu' and gpu.amp_joints == cpus[0].style_joints
        assert gpu.style_judged and not gpu.style_weighted
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        gpu._obs('fill')
        gpu.rollout_metrics()
        start = np.array([cpu_state(cpu)[0] for cpu in cpus])
        np.testing.assert_allclose(gpu.amp_state.cpu().numpy(), start, atol=2e-3)
        previous, bands = None, []
        for step in range(15):
            actions = rng.uniform(-10., 10., (len(rows), 29)) * (.3 if step % 4 else 1.)
            actions[3] = 0.
            before = np.array([cpu_state(cpu)[0] for cpu in cpus])
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            after, heads = zip(*(cpu_state(cpu) for cpu in cpus))
            seen.clear()
            _, _, terminated, truncated, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not (terminated | truncated).any()
            assert seen == [None]  # judged, not paid
            pair = gpu.amp_pair.cpu().numpy()
            assert pair.shape == (len(rows), 108)
            np.testing.assert_allclose(pair[:, :54], before, atol=.03, err_msg=f'step {step}')
            np.testing.assert_allclose(pair[:, 54:], np.array(after), atol=.03, err_msg=f'step {step}')
            if previous is not None:
                np.testing.assert_array_equal(pair[:, :54], previous[:, 54:])
            band = gpu.amp_band.cpu().numpy()
            np.testing.assert_array_equal(band, (np.array(heads) > .3) & (np.array(heads) < 1.1), err_msg=f'{step}')
            previous = pair
            bands.append(band)
        assert np.array(bands).sum() >= 5 and (~np.array(bands)).sum() >= 5
        metrics = gpu.rollout_metrics()
        assert math.isfinite(metrics['rollout/style_dist']) and 0 < metrics['rollout/style_band'] < 1
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    judge = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                             style_library='gpu', style_judge=True)
    try:
        assert judge.style_judged and judge.amp_pair is None and judge.amp_state is None
        judge.reset()
        judge.set_starts(rows)
        for _ in range(5):
            judge.step(torch.zeros((len(rows), 29), device=judge.device))
        assert 'rollout/style_dist' in judge.rollout_metrics() and judge.amp_pair is None
    finally:
        judge.close()

    def refuse(*args, **kwargs):
        raise AssertionError('amp or style computed or loaded with their weights at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    for name in ('amp_state', 'amp_points'):
        monkeypatch.setattr(getup_amp, name, refuse)
    monkeypatch.setattr(g1_getup_style, 'load', refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank,
                             style_library='gpu')
    try:
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
        assert plain.amp_pair is None and not any(key.startswith('rollout/style') for key in plain.rollout_metrics())
    finally:
        plain.close()


# The AMP discriminator's warm-up from the getting-up stage of branch 53 (exp-588d1616e95c4567 @ 1235091456, style 0.5
# there; the AMP branch's parent): the run's real learner (full_gpu_train.make_learner) with amp 1 and style 0, 1024
# worlds, ~2 M steps, all within the 5 M warm-up (reward weight 0). The lab's stage and AMP library are local.
AMP_PARENT = os.path.join(os.path.dirname(__file__), '..', 'runs', 'humanoid_lab', 'exp-588d1616e95c4567')


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_amp_warmup_separates_the_held_out_subject_from_the_rollouts():
    """After the warm-up's first ~2 M steps D gives the held-out subject 5's pairs a mean above 0.5 and the policy's
    in-band pairs of the last rollout one below −0.5; the reward's weight is still 0, the judge's rollout/style_dist is
    logged, and the stage's own files are untouched (nothing is saved)."""
    from pathlib import Path
    from humanoid_sim import full_gpu_train, getup_amp
    from humanoid_sim.core import read_json
    from humanoid_sim.lab import validate_config
    stage = Path(AMP_PARENT) / 'stage_001235091456'
    if not (stage / 'policy.zip').is_file() or not getup_amp.library_path('lafan1').is_file():
        pytest.skip('the lab stage or runs/humanoid_lab/getup_style/lafan1-amp.npz is missing')
    files = sorted(path.name for path in stage.iterdir())
    parent = read_json(Path(AMP_PARENT) / 'run_config.json')
    weights = {**parent['task_config']['reward_weights'], 'amp': 1., 'style': 0.}
    config = validate_config({**{key: parent[key] for key in (
        'backend', 'seed', 'learning_rate', 'n_steps', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'clip_range',
        'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')},
        'envs': 1024, 'batch_size': 6144, 'steps': 1_000_000_000,
        'task_config': {**parent['task_config'], 'reward_weights': weights}})
    learner, export = full_gpu_train.make_learner(config, stage)
    try:
        amp = learner.amp
        assert amp is not None and not amp.restored and amp.weight_now == 0.
        for _ in range(80):
            metrics = learner.collect()
            metrics.update(learner.update())
        assert 0 < amp.trained_steps <= 80 * 1024 * 24 < getup_amp.AMP_WARMUP_STEPS and amp.weight_now == 0.
        assert math.isfinite(metrics['rollout/style_dist']) and math.isfinite(metrics['rollout/amp_reward'])
        with torch.no_grad():
            heldout = float(amp.score(amp.heldout).mean())
            rollout = float(amp.score(amp._pairs[amp._band]).mean())
        print(f'D held out {heldout:.3f}, rollouts {rollout:.3f}, {metrics}')
        assert heldout > .5 and rollout < -.5
        assert metrics['train/disc_heldout'] > .5 and metrics['train/disc_policy'] < -.5
    finally:
        learner.env.close()
        export.close()
    assert sorted(path.name for path in stage.iterdir()) == files


# g1_getup task_config human_starts (iteration 56): a share of training resets from human rows, drawn per world with a
# generator of its own; human episodes step neither the assist nor the bound. tests/test_g1_getup_human.py's rows.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_human_starts():
    """The same human row starts the CUDA world as the CPU task's reset option human_start (qpos, qvel and the
    observation); 3 × 4096 training resets at share 0.3 start ~0.3 from human rows (each row drawn); after an episode
    up the bank worlds' assist and bound step down and the human ones keep theirs; rollout/human_start_share and
    rollout/getup_success_bank are logged. Without the setting nothing is drawn or logged."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_human import MD5, human_bank
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank, human = drop_bank(), human_bank()
    settings = dict(bank=bank, human_bank=human, human_starts={'share': .3, 'md5': MD5}, pd_bound=[1., .6])
    rows = list(range(len(human['qpos'])))
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, **settings)
    cpu = G1GetupEnv(**settings)
    try:
        gpu.reset()
        obs = gpu.set_human_starts(rows).cpu().numpy()
        assert gpu.human_episode.all() and (gpu.start_row == -1).all()
        for row in rows:
            cpu_obs, info = cpu.reset(seed=4, options=dict(OPTIONS, human_start=row))
            assert info['human_start'] == row
            np.testing.assert_allclose(gpu.qpos[row].cpu().numpy(), cpu.data.qpos, rtol=0, atol=1e-6)
            np.testing.assert_allclose(gpu.qvel[row].cpu().numpy(), cpu.data.qvel, rtol=0, atol=1e-6)
            # the last critic input is the assist force: the CPU reset sets 0 (OPTIONS), the CUDA world keeps its own
            np.testing.assert_allclose(obs[row, :-1], cpu_obs[:-1], rtol=1e-5, atol=1e-4, err_msg=f'row {row}')
    finally:
        gpu.close()
        cpu.close()
    gpu = TensorG1GetupEnv(4096, seed=5, **settings)
    try:
        gpu.reset()
        everyone = torch.ones(gpu.n_envs, dtype=torch.bool, device=gpu.device)
        stream = gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device)))
        shares, counts = [], torch.zeros(len(rows), device=gpu.device)
        with stream:
            for _ in range(3):
                gpu._masked_reset(everyone)
                shares.append(gpu.human_episode.float().mean().item())
                human_qpos = torch.as_tensor(human['qpos'][:, 0], dtype=torch.float32, device=gpu.device)
                match = (gpu.qpos[:, 0:1] == human_qpos[None]).float()
                counts += match[gpu.human_episode].sum(0)
                assert torch.equal(gpu.human_episode, gpu.start_row == -1)
            assert abs(np.mean(shares) - .3) < .015 and bool((counts > 100).all()), (shares, counts)
            started = gpu.human_episode.clone()
            assist, bound = gpu.assist.clone(), gpu.bound.clone()
            gpu.lengths.fill_(gpu.head_steps)
            gpu.head_window.fill_(1.3)  # every world up for the whole window
            gpu._masked_reset(everyone)
        assert torch.equal(gpu.assist[started], assist[started]) and torch.equal(gpu.bound[started], bound[started])
        torch.testing.assert_close(gpu.assist[~started], assist[~started] - 20.)
        torch.testing.assert_close(gpu.bound[~started], bound[~started] - .02)
        gpu.reset()  # the first episodes after reset() end at random steps: ~80 of them within 10 steps
        gpu.rollout_metrics()
        for _ in range(10):
            gpu.step(torch.zeros((gpu.n_envs, 29), device=gpu.device))
        metrics = gpu.rollout_metrics()
        assert 0 <= metrics['rollout/human_start_share'] <= 1 and 'rollout/getup_success_bank' in metrics
    finally:
        gpu.close()
    plain = TensorG1GetupEnv(64, seed=5, bank=bank)
    try:
        assert plain.human_share is None and not hasattr(plain, 'human_generator')
        plain.reset()
        plain.step(torch.zeros((64, 29), device=plain.device))
        assert not any('human' in key or 'bank' in key for key in plain.rollout_metrics())
    finally:
        plain.close()


# g1_getup reward weight head_rise (iteration 57): the head top's vertical speed over the step, charged above 0.8 m/s,
# on CUDA. drop_bank's rows (back, front, spinning drops, standing) and a standing world thrown upwards.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_head_rise(monkeypatch):
    """The weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset), 15 steps of the same
    random actions: per world the speed passed to task.getup_reward_components is (head − head a step before)/dt of
    its own heads, the first step's from the reset's head, and the CPU task's within 1 % (atol 0.01 m/s); the terms
    alike; rollout/head_rise and rollout/head_speed_p90 as the steps give them. Without the weight nothing is
    computed or logged."""
    import mujoco
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 1, 2, 3, 3]
    settings = dict(reward_weights={'head_rise': 3.})
    seen, original = [], task.getup_reward_components

    def host(value):
        return np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        seen.append(dict(gpu=isinstance(args[0], torch.Tensor), speed=host(kwargs['head_speed']), head=host(args[0]),
                         term=host(components['head_rise'])))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    rng = np.random.default_rng(6)
    cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
    try:
        assert gpu.rise_weighted
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        torch.testing.assert_close(gpu.previous_head, gpu._head())  # a reset starts from its own head
        cpus[4].data.qvel[2] = 3.  # the last world thrown upwards at the same pose
        mujoco.mj_forward(cpus[4].model, cpus[4].data)
        gpu.qvel[4, 2] = 3.
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu.wp.capture_launch(gpu.forward_graph)
        gpu._obs('fill')
        gpu.rollout_metrics()
        previous = gpu._head().cpu().numpy()
        speeds, bands, terms = [], [], []
        for step in range(15):
            actions = rng.uniform(-10., 10., (len(rows), 29)) * (.3 if step % 4 else 1.)
            seen.clear()
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            _, _, terminated, truncated, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not (terminated | truncated).any()
            cpu_calls = [call for call in seen if not call['gpu']]
            (gpu_call,) = [call for call in seen if call['gpu']]
            np.testing.assert_allclose(gpu_call['speed'], (gpu_call['head'] - previous) / gpu.dt, rtol=1e-4,
                                       atol=1e-4, err_msg=f'step {step}')
            # within 1 % for 10 steps; later float32 drifts from float64 under the random ±10 actions (0.02 m/s seen at
            # step 13, as the AMP pairs' parity)
            tolerance = dict(rtol=.01, atol=.01) if step < 10 else dict(rtol=.01, atol=.03)
            cpu_speed = np.array([call['speed'][0] for call in cpu_calls])
            np.testing.assert_allclose(gpu_call['speed'], cpu_speed, **tolerance, err_msg=f'step {step}')
            cpu_term = np.array([call['term'][0] for call in cpu_calls])
            np.testing.assert_allclose(gpu_call['term'], cpu_term, rtol=.01, atol=3 * tolerance['atol'],
                                       err_msg=f'step {step}')
            np.testing.assert_allclose(gpu_call['term'], -3. * np.maximum(0., gpu_call['speed'] - .8), rtol=1e-5,
                                       atol=1e-6)
            previous = gpu_call['head']
            speeds.append(gpu_call['speed'])
            bands.append((gpu_call['head'] > .3) & (gpu_call['head'] < 1.1))
            terms.append(gpu_call['term'])
        speeds, bands, terms = np.array(speeds), np.array(bands), np.array(terms)
        assert (terms < 0).sum() >= 1  # the thrown world at least
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/head_rise'] == pytest.approx(-terms.mean() / 3., rel=1e-4, abs=1e-6)
        if bands.any():
            assert metrics['rollout/head_speed_p90'] == pytest.approx(np.percentile(speeds[bands], 90), rel=1e-4,
                                                                      abs=1e-5)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()

    def refuse(*args, **kwargs):
        raise AssertionError('head_rise computed with its weight at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    monkeypatch.setattr(task, 'head_rise', refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank)
    try:
        assert not plain.rise_weighted and plain.previous_head is None
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
        assert not any(key.startswith('rollout/head') for key in plain.rollout_metrics())
    finally:
        plain.close()


# g1_getup task_config stance_pelvis (iteration 60): the stance_width term's pelvis threshold on CUDA. The wide stance and
# home standing of tests/test_g1_getup_stance_width.py (pelvis ~0.75 m).
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_stance_pelvis(monkeypatch):
    """stance_width 2 with stance_pelvis 0.25 and 0.8, on CUDA and on the CPU task, 10 steps of the wide stance held and
    home standing: at 0.25 the wide stance is charged −2·(W − 0.30) on both (within 0.005), home not (W 0.24); at 0.8
    (above the pelvis) nothing and rollout/stance_width has no world-step (NaN); the CUDA task passes the threshold to
    task.getup_reward_components as a float."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS
    from test_g1_getup_sole_impact import STANDING
    from test_g1_getup_stance_width import WIDE, hold, stance_bank
    bank = stance_bank()
    rows = [STANDING, WIDE]
    seen, original = [], task.getup_reward_components

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        if isinstance(args[0], torch.Tensor):
            seen.append((kwargs['stance_pelvis'], components['stance_width'].cpu().numpy().copy()))
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    for pelvis in (.25, .8):
        settings = dict(reward_weights={'stance_width': 2.}, stance_pelvis=pelvis)
        cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
        gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
        try:
            assert gpu.stance_pelvis == pelvis and isinstance(gpu.stance_pelvis, float)
            for cpu, row in zip(cpus, rows):
                cpu.reset(seed=4, options=dict(OPTIONS, start=row))
            gpu.reset()
            gpu.set_starts(rows)
            gpu.set_action_delay_steps(0)
            gpu.set_motor_friction(1.)
            gpu.set_body_friction(.6)
            gpu.set_assist(0.)
            gpu._obs('fill')
            gpu.rollout_metrics()
            actions = np.stack([hold(cpus[0]) if row == WIDE else np.zeros(29) for row in rows])
            for step in range(10):
                cpu_terms = [cpu.step(action)[-1]['reward_components']['stance_width']
                             for cpu, action in zip(cpus, actions)]
                seen.clear()
                gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
                gpu.check_health()
                ((threshold, terms),) = seen
                assert threshold == pelvis and isinstance(threshold, float)
                if pelvis == .8:
                    assert cpu_terms == [0., 0.] and not terms.any(), step
                else:
                    assert cpu_terms[0] == 0. and cpu_terms[1] < -.15, step
                np.testing.assert_allclose(terms, cpu_terms, rtol=0, atol=.005, err_msg=f'{pelvis} step {step}')
            metrics = gpu.rollout_metrics()
            if pelvis == .8:
                assert np.isnan(metrics['rollout/stance_width'])
            else:  # the two worlds' mean W: home 0.24, the wide stance ~0.41
                assert .3 < metrics['rollout/stance_width'] < .35
        finally:
            for cpu in cpus:
                cpu.close()
            gpu.close()


# g1_getup reward weight sole_slip (iteration 62): the feet in contact charged their horizontal speed², on CUDA.
# drop_bank's rows (back, front, spinning drops, standing) and a standing world sliding sideways at 2 m/s.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_sole_slip(monkeypatch):
    """The weight on, on CUDA and on the CPU task (no assist, delay, sensor noise or COM offset), 10 steps of the same
    random actions: per world the Σ passed to task.getup_reward_components is task.sole_slip of its own feet's speeds
    over the step (the first from the reset's pose: the slid world is charged at once). The feet's speeds match the CPU
    task's within 1.5 % (atol 0.01 m/s: the sliding world, sliding friction under PGS and Newton, reaches 1.4 % by step
    9 and parts after it), the Σ within 3 % (quadratic) where both backends count the same feet; rollout/sole_slip and
    rollout/sole_slip_speed as the steps give them. Without the weight nothing is computed or logged."""
    import mujoco
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 1, 2, 3, 3]
    settings = dict(reward_weights={'sole_slip': .1})
    seen, original = [], task.getup_reward_components

    def spy(*args, **kwargs):
        components, standing = original(*args, **kwargs)
        if isinstance(args[0], torch.Tensor):
            seen.append(kwargs['sole_slip'].cpu().numpy().copy())
        return components, standing
    monkeypatch.setattr(task, 'getup_reward_components', spy)
    rng = np.random.default_rng(7)
    cpus = [G1GetupEnv(bank=bank, **settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, **settings)
    try:
        assert gpu.slip_weighted
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.set_assist(0.)
        cpus[4].data.qvel[1] = 2.  # the last world slides sideways at the same pose
        mujoco.mj_forward(cpus[4].model, cpus[4].data)
        gpu.qvel[4, 1] = 2.
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu.wp.capture_launch(gpu.forward_graph)
        gpu._obs('fill')
        gpu.rollout_metrics()
        sums, speeds, disagreed = [], [], 0
        for step in range(10):
            actions = rng.uniform(-10., 10., (len(rows), 29)) * (.3 if step % 4 else 1.)
            actions[4] = 0.
            before = gpu.xpos[:, gpu.feet, :2].clone()
            cpu_infos = [cpu.step(action)[-1] for cpu, action in zip(cpus, actions)]
            seen.clear()
            _, _, terminated, truncated, _ = gpu.step(torch.tensor(actions, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert not (terminated | truncated).any()
            (slip,) = seen
            speed = ((gpu.xpos[:, gpu.feet, :2] - before).norm(dim=-1) / gpu.dt)
            sole = task.sole_height(gpu.geom_xpos[:, gpu.soles, 2], gpu.sole_radius)
            np.testing.assert_allclose(slip, task.sole_slip(speed, sole).cpu().numpy(), rtol=1e-5, atol=1e-6)
            cpu_slip = np.array([info['sole_slip'] for info in cpu_infos])
            cpu_speed = np.array([info['foot_speed'] for info in cpu_infos])
            cpu_contact = np.array([cpu._sole_height().numpy()[0] < .01 for cpu in cpus])
            # the speeds everywhere; the Σ where both count the same feet (a foot at the 1 cm threshold may count on one
            # backend and not the other, as stance_width's landing drop)
            np.testing.assert_allclose(speed.cpu().numpy(), cpu_speed, rtol=.015, atol=.01, err_msg=f'step {step}')
            same = (cpu_contact == (sole < .01).cpu().numpy()).all(1)
            disagreed += int((~same).sum())
            np.testing.assert_allclose(slip[same], cpu_slip[same], rtol=.03, atol=.01, err_msg=f'step {step}')
            if step == 0:
                assert slip[4] > 1. and cpu_slip[4] > 1.  # charged from the reset's own pose
            sums.append(slip)
            contact = (sole < .01).cpu().numpy()
            speeds.append(speed.cpu().numpy()[contact])
        assert disagreed <= 6, disagreed  # of 50 world-steps
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/sole_slip'] == pytest.approx(np.mean(sums), rel=1e-4)
        assert metrics['rollout/sole_slip_speed'] == pytest.approx(np.median(np.concatenate(speeds)), rel=1e-4,
                                                                   abs=1e-5)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()

    def refuse(*args, **kwargs):
        raise AssertionError('sole_slip computed with its weight at 0')
    monkeypatch.setattr(task, 'getup_reward_components', original)
    monkeypatch.setattr(task, 'sole_slip', refuse)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank)
    try:
        assert not plain.slip_weighted
        plain.reset()
        plain.set_starts(rows)
        for _ in range(5):
            plain.step(torch.zeros((len(rows), 29), device=plain.device))
        assert not any(key.startswith('rollout/sole_slip') for key in plain.rollout_metrics())
    finally:
        plain.close()


# g1_getup task_config hip_roll_out (iteration 63): the hip roll PD target's outward limit, on CUDA through the CPU
# task's target_low / target_high (joint_low / joint_high), as knee_floor.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_hip_roll_out():
    """hip_roll_out 0.7: the CUDA task's PD bounds are the CPU task's (left hip roll high 0.7, right low −0.7, the rest
    the joint ranges); lying on the back with both hips driven outwards to the box edge, 10 steps: the motor torques
    and hip roll angles match the CPU task's (torque within 2 % of the peak, angles within 0.01 rad) and the angles stay
    near the limit, while the unlimited CUDA task's go far past it."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    bank = drop_bank()
    rows = [0, 0]
    rng = np.random.default_rng(3)
    cpus = [G1GetupEnv(bank=bank, hip_roll_out=.7) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank, hip_roll_out=.7)
    plain = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, bank=bank)
    try:
        np.testing.assert_array_equal(gpu.joint_low.cpu().numpy(), cpus[0].target_low.astype(np.float32))
        np.testing.assert_array_equal(gpu.joint_high.cpu().numpy(), cpus[0].target_high.astype(np.float32))
        assert gpu.joint_high[1].item() == pytest.approx(.7) and gpu.joint_low[7].item() == pytest.approx(-.7)
        assert plain.joint_high[1].item() > 2.9 and plain.joint_low[7].item() < -2.9
        for cpu in cpus:
            cpu.reset(seed=4, options=dict(OPTIONS, start=0))
        for env in (gpu, plain):
            env.reset()
            env.set_starts(rows)
            env.set_action_delay_steps(0)
            env.set_motor_friction(1.)
            env.set_body_friction(.6)
            env.set_assist(0.)
        peak = np.asarray(cpus[0].peak)
        for step in range(10):
            actions = rng.uniform(-1., 1., (len(rows), 29))
            actions[:, 1], actions[:, 7] = 10., -10.
            for cpu, action in zip(cpus, actions):
                cpu.step(action)
            tensor = torch.tensor(actions, dtype=torch.float32, device=gpu.device)
            gpu.step(tensor)
            plain.step(tensor)
            gpu.check_health()
            torque = np.array([cpu.data.ctrl.copy() for cpu in cpus])
            np.testing.assert_allclose(gpu.ctrl.cpu().numpy(), torque, rtol=0, atol=.02 * peak.max(),
                                       err_msg=f'step {step}')
            angles = np.array([cpu.data.qpos[cpu.qadr][[1, 7]] for cpu in cpus])
            np.testing.assert_allclose(gpu.qpos[:, gpu.qadr][:, [1, 7]].cpu().numpy(), angles, rtol=0, atol=.01,
                                       err_msg=f'step {step}')
        outward = lambda env: (env.qpos[:, env.qadr][:, [1, 7]] * torch.tensor([1., -1.], device=env.device)).max()
        assert outward(gpu).item() < .7 + .15 and outward(plain).item() > 1.2
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
        plain.close()



# g1_getup task_config track_clip (iteration 64): motion tracking of one human clip on CUDA, against the CPU task.
# tests/test_g1_getup_track.py's synthetic clip (drop_bank's back row moving), the lab clip where it is on disk.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_track_clip(monkeypatch):
    """Worlds started from clip frames 0, 10, 20, 30 and the last without noise (and one from the bank's back row) are
    the CPU task's resets track_frame / start: qpos, qvel, the reference's alignment and the observation with its 17
    critic extras match; over five steps of the same actions the observation, the reward, every tracking term
    (task.track_reward_components, numpy and torch: errors, self_contact, impact, sole_impact), the hands' floor force
    and the terminations match within float32-vs-float64 and contact noise; the clock counts on in both."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    bank, clip = drop_bank(), synthetic_clip()
    settings = dict(bank=bank, track_clip=SETTING, track_bank=clip, reward_weights={'impact': 2., 'sole_impact': 1.})
    frames = [0, 10, 20, 30, len(clip['qpos']) - 1]
    seen, original = [], task.track_reward_components

    def spy(*args, **kwargs):
        components = original(*args, **kwargs)
        seen.append((isinstance(args[0]['anchor_pos'], torch.Tensor), {
            key: np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value, dtype=float).copy()
            for key, value in components.items()}))
        return components

    monkeypatch.setattr(task, 'track_reward_components', spy)
    cpus = [G1GetupEnv(**settings) for _ in frames]
    gpu = TensorG1GetupEnv(len(frames), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(OPTIONS, track_frame=f, track_noise=False))[0]
                            for cpu, f in zip(cpus, frames)])
        gpu.reset()
        obs = gpu.set_clip_starts(frames).cpu().numpy()
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert obs.shape == (len(frames), 507) and gpu.clip_clock.tolist() == frames
        assert (gpu.clip_kind == 1).all() and not gpu.clip_yaw.any() and not gpu.clip_offset.any()
        np.testing.assert_allclose(gpu.qpos.cpu().numpy(), np.stack([c.data.qpos for c in cpus]), atol=1e-3)
        np.testing.assert_allclose(gpu.qvel.cpu().numpy(), np.stack([c.data.qvel for c in cpus]), atol=1e-4)
        np.testing.assert_allclose(obs, cpu_obs, atol=2e-3, rtol=1e-3)
        np.testing.assert_allclose(gpu.episode_limit.cpu().numpy(), [min(gpu.max_episode_steps, gpu.clip_end - f)
                                                                     for f in frames])
        speeds = np.zeros(507, dtype=bool)
        speeds[task.joint_positions(29, 5, (), (0,), range(3)) + task.joint_positions(29, 5, (), (5,), range(29))] = True
        rng = np.random.default_rng(0)
        for step in range(5):
            action = np.stack([(c.clip['qpos'][min(c.clip_clock + 1, c.clip_last), 7:] - c.home) / c.scale
                               for c in cpus]) + rng.normal(0., .3, (len(frames), 29))
            seen.clear()
            results = [cpu.step(a) for cpu, a in zip(cpus, action)]
            _, reward, terminated, truncated, terminal = gpu.step(torch.tensor(action, dtype=torch.float32,
                                                                               device=gpu.device))
            gpu.check_health()
            assert terminated.cpu().tolist() == [r[2] for r in results], step
            got, expected = terminal.cpu().numpy(), np.stack([r[0] for r in results])
            np.testing.assert_allclose(got[:, ~speeds], expected[:, ~speeds], atol=.03, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, speeds], expected[:, speeds], atol=.1, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.05, rtol=.02)
            np.testing.assert_allclose(gpu.hand_substeps.amax(dim=1).cpu().numpy(), [r[4]['hand_force'] for r in results],
                                       atol=5., rtol=.1)
            cpu_terms = [terms for on_gpu, terms in seen if not on_gpu]
            (gpu_terms,) = [terms for on_gpu, terms in seen if on_gpu]
            for key in gpu_terms:
                np.testing.assert_allclose(gpu_terms[key], [terms[key][0] for terms in cpu_terms], atol=.03, rtol=.02,
                                           err_msg=f'{key} step {step}')
            live = ~(terminated | truncated).cpu().numpy()
            assert gpu.clip_clock.cpu().numpy()[live].tolist() == [c.clip_clock for c, k in zip(cpus, live) if k]
        cpus[0].reset(seed=4, options=dict(OPTIONS, start=0))
        obs = gpu.set_starts([0] * len(frames)).cpu().numpy()
        assert (gpu.clip_kind == 2).all() and (gpu.clip_clock == 0).all() and (gpu.start_row == 0).all()
        np.testing.assert_allclose(gpu.clip_yaw[0].item(), cpus[0].clip_yaw, atol=1e-5)
        np.testing.assert_allclose(gpu.clip_offset[0].cpu().numpy(), cpus[0].clip_offset, atol=1e-5)
        np.testing.assert_allclose(obs[0, -task.TRACK_CRITIC_EXTRAS:], cpus[0]._observation()[-task.TRACK_CRITIC_EXTRAS:],
                                   atol=1e-4)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


# The BONES-SEED clips chosen 27.10 (scripts/lab/getup_track_clip.py --source bones: back, side, front), made locally.
# Since iteration 71 the clips of fit v2 (bounded steps, palms on the floor, header bones_fit): front, back, side and
# its mirror, the roots' clips.
BONES_CLIPS = ('bones-faint_stand_up_lying_R_001__A495-906-1246', 'bones-faint_stand_up_lying_side_R_001__A495-889-1224',
               'bones-faint_stand_up_lying_stomach_R_001__A495-962-1347',
               'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v2',
               'bones-faint_stand_up_lying_R_001__A495-785-1234-v2',
               'bones-faint_stand_up_lying_side_R_001__A492-710-1031-v2',
               'bones-faint_stand_up_lying_side_R_001__A492_M-710-1031-v2')


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
@pytest.mark.parametrize('name', BONES_CLIPS)
def test_g1_cpu_gpu_getup_track_clip_bones(name):
    """A BONES-SEED clip from disk (its fitted floor, the hand and foot capsules out of it): worlds started without
    noise from frames 0, a quarter, a half and the last are the CPU task's track_frame resets (qpos, qvel, the
    observation with its critic extras); over three steps toward the clip's next frames the observation, the reward
    and the terminations match within float32 and contact noise, and nothing is cut at once."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    if not g1_getup_track.clip_path(name).is_file():
        pytest.skip(f'{name} is made locally (getup_track_clip.py --source bones)')
    clip = g1_getup_track.load(name)
    assert clip['header']['dataset'] == 'bones_seed'
    setting = {'file': name, 'md5': clip['md5']}
    settings = dict(bank=drop_bank(), track_clip=setting, reward_weights={'impact': 2., 'sole_impact': 1.})
    last = len(clip['qpos']) - 1
    frames = [0, last // 4, last // 2, last]
    cpus = [G1GetupEnv(**settings) for _ in frames]
    gpu = TensorG1GetupEnv(len(frames), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(OPTIONS, track_frame=f, track_noise=False))[0]
                            for cpu, f in zip(cpus, frames)])
        gpu.reset()
        obs = gpu.set_clip_starts(frames).cpu().numpy()
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert gpu.clip_clock.tolist() == frames
        np.testing.assert_allclose(gpu.qpos.cpu().numpy(), np.stack([c.data.qpos for c in cpus]), atol=1e-3)
        np.testing.assert_allclose(gpu.qvel.cpu().numpy(), np.stack([c.data.qvel for c in cpus]), atol=1e-4)
        np.testing.assert_allclose(obs, cpu_obs, atol=2e-3, rtol=1e-3)
        for step in range(3):
            action = np.stack([(c.clip['qpos'][min(c.clip_clock + 1, c.clip_last), 7:] - c.home) / c.scale
                               for c in cpus])
            results = [cpu.step(a) for cpu, a in zip(cpus, action)]
            _, reward, terminated, _, terminal = gpu.step(torch.tensor(action, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            assert terminated.cpu().tolist() == [r[2] for r in results] == [False] * len(frames), step
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.05, rtol=.02)
            expected = np.stack([r[0] for r in results])
            np.testing.assert_allclose(terminal.cpu().numpy(), expected, atol=.1, rtol=.02, err_msg=f'step {step}')
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_clip_starts_ends_and_logs():
    """4096 worlds: training resets start 0.6 / 0.3 / 0.1 from the clip's first frame, a uniform frame and the bank's
    back rows; the clip starts carry mjlab's noise within its widths and nothing is left in the floor; each episode
    ends within what is left of the clip and its hold; a full reset() cuts the first episodes within that too; the
    logs carry track_done, the errors, the hand force and self_contact; standing worlds against a lying reference end
    by the tracking rules. Off: no clip and no track logs."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_getup_impact import drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    bank, clip = drop_bank(), synthetic_clip()
    n = 4096
    gpu = TensorG1GetupEnv(n, seed=7, bank=bank, track_clip=SETTING, track_bank=clip)
    try:
        gpu.reset()
        assert bool((gpu.episode_limit <= (gpu.clip_end - gpu.clip_clock)).all())
        everyone = torch.ones(n, dtype=torch.bool, device=gpu.device)
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu._masked_reset(everyone)
        kinds = gpu.clip_kind.bincount(minlength=3).cpu().numpy() / n
        np.testing.assert_allclose(kinds, task.TRACK_START_SHARES, atol=.025)
        frames = gpu.clip_clock[gpu.clip_kind == 1]
        assert len(frames.unique()) == len(clip['qpos'])
        assert bool((gpu.clip_clock[gpu.clip_kind != 1] == 0).all())
        assert bool((gpu.start_row[gpu.clip_kind == 2] == 0).all()) and bool((gpu.start_row[gpu.clip_kind != 2] == -1).all())
        first = gpu.clip_kind == 0
        start = torch.as_tensor(clip['qpos'][0], dtype=torch.float32, device=gpu.device)
        assert float((gpu.qpos[first, :2] - start[:2]).abs().max()) <= task.TRACK_START_NOISE['xy'] + 1e-5
        joints = (gpu.qpos[first, 7:] - start[7:]).abs()
        assert float(joints.max()) <= task.TRACK_START_NOISE['joints'] + 1e-5 and float(joints.max()) > .09
        assert float(gpu._floor_depth()[gpu.clip_kind != 2].min()) > -2e-3
        np.testing.assert_array_equal(gpu.episode_limit.cpu().numpy(),
                                      np.minimum(gpu.max_episode_steps, gpu.clip_end - gpu.clip_clock.cpu().numpy()))
        gpu.rollout_metrics()
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        for _ in range(40):
            gpu.step(torch.randn((n, 29), device=gpu.device, generator=generator) * .5)
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        for key in ('track_done', 'track_done_clip', 'track_failed', 'track_standing', 'hand_force', 'self_contact',
                    *(f'track_err_{k}' for k in task.TRACK_SIGMA)):
            assert f'rollout/{key}' in metrics, key
        assert metrics['rollout/bad_states'] == 0 and metrics['rollout/track_err_anchor_pos'] > 0
        assert 0. <= metrics['rollout/track_standing'] <= metrics['rollout/track_done'] + 1e-9 or math.isnan(
            metrics['rollout/track_standing'])
        gpu.set_clip_starts(0)
        gpu.qpos[:, 2:7] = torch.tensor([.8, 1., 0., 0., 0.], device=gpu.device)  # standing while the clip lies
        gpu.qpos[:, gpu.qadr] = gpu.home
        gpu.qvel.zero_()
        gpu.wp.capture_launch(gpu.forward_graph)
        _, _, terminated, _, _ = gpu.step(torch.zeros((n, 29), device=gpu.device))
        metrics = gpu.rollout_metrics()
        assert bool(terminated.all()) and metrics['rollout/track_failed'] == 1.
        assert metrics['rollout/track_standing'] == 0.  # up, but ended by the tracking rules, not by the hold's end
    finally:
        gpu.close()
    plain = TensorG1GetupEnv(64, seed=5, bank=bank)
    try:
        assert plain.clip is None
        plain.reset()
        plain.step(torch.zeros((64, 29), device=plain.device))
        assert not any('track' in key or 'hand_force' in key for key in plain.rollout_metrics())
    finally:
        plain.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_starts():
    """track_clip's start settings (iteration 68) on CUDA. The old values spelled out are the missing ones bit for bit
    (same seed: the same kinds, clocks, rows, alignment and qpos over four training resets). 4096 worlds with track_starts
    {0.3, 0.3, 0.4}, track_noise 3 and track_bank_classes front: the shares within 0.025, the bank starts on the front
    rows only, the first-frame starts' joints and xy within 3 times mjlab's widths and reaching near them. With
    track_bank_nearest a bank row in the synthetic clip's frame-20 pose starts at frame 20 with the CPU task's
    alignment and critic inputs. The per-kind logs: track_done_first / rsi / bank, track_standing_bank."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    bank, clip = drop_bank(), synthetic_clip()
    old = dict(track_starts={'first': .6, 'rsi': .3, 'bank': .1}, track_noise=1., track_bank_classes=['back'],
               track_bank_nearest=False)
    envs = [TensorG1GetupEnv(256, seed=3, bank=bank, track_clip=SETTING, track_bank=clip, **settings)
            for settings in ({}, old)]
    try:
        for env in envs:
            env.reset()
        everyone = torch.ones(256, dtype=torch.bool, device=envs[0].device)
        for repeat in range(4):  # the reset of reset(), then three more training resets of every world
            if repeat:
                for env in envs:
                    with env.wp.ScopedStream(env.wp.stream_from_torch(torch.cuda.current_stream(env.device))):
                        env._masked_reset(everyone)
            for name in ('qpos', 'qvel', 'clip_kind', 'clip_clock', 'start_row', 'clip_yaw', 'clip_offset',
                         'episode_limit'):
                assert torch.equal(getattr(envs[0], name), getattr(envs[1], name)), (name, repeat)
    finally:
        for env in envs:
            env.close()
    n = 4096
    gpu = TensorG1GetupEnv(n, seed=7, bank=bank, track_clip=SETTING, track_bank=clip,
                           track_starts={'first': .3, 'rsi': .3, 'bank': .4}, track_noise=3.,
                           track_bank_classes=['front'])
    try:
        gpu.reset()
        kinds = gpu.clip_kind.bincount(minlength=3).cpu().numpy() / n
        np.testing.assert_allclose(kinds, (.3, .3, .4), atol=.025)
        assert set(gpu.start_row[gpu.clip_kind == 2].unique().tolist()) == {1, 2}
        first = gpu.clip_kind == 0
        start = torch.as_tensor(clip['qpos'][0], dtype=torch.float32, device=gpu.device)
        joints = (gpu.qpos[first, 7:] - start[7:]).abs()
        assert .27 < float(joints.max()) <= 3 * task.TRACK_START_NOISE['joints'] + 1e-5
        xy = (gpu.qpos[first, :2] - start[:2]).abs()
        assert .13 < float(xy.max()) <= 3 * task.TRACK_START_NOISE['xy'] + 1e-5
        gpu.rollout_metrics()
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        for _ in range(40):
            gpu.step(torch.randn((n, 29), device=gpu.device, generator=generator) * .5)
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        for key in ('track_done_first', 'track_done_rsi', 'track_done_bank', 'track_standing_bank'):
            assert f'rollout/{key}' in metrics, key
        assert metrics['rollout/track_done_first'] == metrics['rollout/track_done_clip'] or math.isnan(
            metrics['rollout/track_done_clip'])
        assert metrics['rollout/bad_states'] == 0
    finally:
        gpu.close()
    near = {key: value.copy() for key, value in bank.items() if isinstance(value, np.ndarray)}
    near['classes'] = bank['classes']
    near['qpos'][0] = clip['qpos'][20]
    settings = dict(bank=near, track_clip=SETTING, track_bank=clip, track_bank_nearest=True)
    cpu = G1GetupEnv(**settings)
    gpu = TensorG1GetupEnv(4, seed=4, observation_noise=False, randomize=False, **settings)
    try:
        cpu_obs, info = cpu.reset(seed=4, options=dict(OPTIONS, start=0))
        assert info['track_frame'] == 20
        gpu.reset()
        obs = gpu.set_starts([0] * 4).cpu().numpy()
        assert (gpu.clip_kind == 2).all() and (gpu.clip_clock == 20).all()
        np.testing.assert_array_equal(gpu.episode_limit.cpu().numpy(), min(gpu.max_episode_steps, gpu.clip_end - 20))
        np.testing.assert_allclose(gpu.clip_yaw.cpu().numpy(), cpu.clip_yaw, atol=1e-5)
        np.testing.assert_allclose(gpu.clip_offset.cpu().numpy(), np.repeat(cpu.clip_offset[None], 4, 0), atol=1e-5)
        np.testing.assert_allclose(obs[:, -task.TRACK_CRITIC_EXTRAS:],
                                   np.repeat(cpu_obs[None, -task.TRACK_CRITIC_EXTRAS:], 4, 0), atol=1e-4)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_sampling():
    """track_sampling (iteration 77) on CUDA. Its draws come from its own generator: with it, the same seed gives the
    same start kinds, rows and clocks of the first-frame and bank starts and the same qpos of every non-RSI world over
    four training resets as without it. 4096 worlds, bins of 0.2 s over the synthetic clip's 40 frames (4 bins) with
    G_b/G_max (0.9, 0.9, 0.1, 0.9): the RSI starts come from the hard part 40 % of the time, bin 2 gets ~55 % of them,
    each clock inside its bin; the logs; the training_state there and back (another bin count: afresh); a probe's clip
    starts do not count. Off: no sampler and no track_bin logs."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from test_g1_getup_impact import drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    bank, clip = drop_bank(), synthetic_clip()
    working = {'bin': 1.0, 'uniform': .5, 'hard': .4, 'history': .1, 'temperature': 25}
    envs = [TensorG1GetupEnv(256, seed=3, bank=bank, track_clip=SETTING, track_bank=clip, **settings)
            for settings in ({}, {'track_sampling': working})]
    try:
        assert envs[0].sampler is None and envs[1].sampler.bins == 1
        for env in envs:
            env.reset()
        everyone = torch.ones(256, dtype=torch.bool, device=envs[0].device)
        for repeat in range(4):
            if repeat:
                for env in envs:
                    with env.wp.ScopedStream(env.wp.stream_from_torch(torch.cuda.current_stream(env.device))):
                        env._masked_reset(everyone)
            other = envs[0].clip_kind != 1
            for name in ('clip_kind', 'start_row', 'clip_yaw', 'clip_offset'):
                assert torch.equal(getattr(envs[0], name), getattr(envs[1], name)), (name, repeat)
            assert torch.equal(envs[0].clip_clock[other], envs[1].clip_clock[other]), repeat
            assert torch.equal(envs[0].qpos[other], envs[1].qpos[other]), repeat
        envs[0].step(torch.zeros((256, 29), device=envs[0].device))
        assert not any('track_bin' in key for key in envs[0].rollout_metrics())
    finally:
        for env in envs:
            env.close()
    n = 4096
    setting = {**working, 'bin': .2}
    gpu = TensorG1GetupEnv(n, seed=7, bank=bank, track_clip=SETTING, track_bank=clip, track_sampling=setting)
    try:
        sampler = gpu.sampler
        assert sampler.bins == 4 and sampler.starts.tolist() == [0, 10, 20, 30]
        gpu.reset()
        gpu.rollout_metrics()
        sampler.g.copy_(torch.tensor([.9, .9, .1, .9], dtype=torch.float64, device=gpu.device))
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu._masked_reset(torch.ones(n, dtype=torch.bool, device=gpu.device))
        rsi = gpu.clip_kind == 1
        bins = sampler.world_bin[rsi]
        assert bool((bins >= 0).all()) and bool((sampler.world_bin[~rsi] == -1).all())
        assert bool((gpu.clip_clock[rsi] // 10 == bins).all())
        shares = bins.bincount(minlength=4).cpu().numpy() / len(bins)
        soft = np.exp(-25 * np.array([.9, .9, .1, .9]))
        history = sampler.history.cpu().numpy()  # near uniform after a few draws
        expected = .5 / 4 + .4 * soft / soft.sum() + .1 * history / history.sum()
        np.testing.assert_allclose(shares, expected, atol=.04)
        metrics = gpu.rollout_metrics()
        assert abs(metrics['rollout/track_bin_hard_share'] - .4) < .05
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        gpu.episode_limit.clamp_(max=20)  # episodes end within the loop (the clip's hold alone lasts 3 s), so bins count
        for _ in range(40):
            gpu.step(torch.randn((n, 29), device=gpu.device, generator=generator) * .5)
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        for key in ('track_bin_hard_share', 'track_bin_g_min', 'track_bin_g_mean', 'track_bin_hardest'):
            assert f'rollout/{key}' in metrics, key
        assert metrics['rollout/bad_states'] == 0 and float(sampler.count.sum()) > 0
        assert metrics['rollout/track_bin_g_min'] <= metrics['rollout/track_bin_g_mean'] <= 1 + 1e-6
        state = gpu.training_state()
        assert len(state['track_bin_g']) == 4 and len(state['assist_force']) == n
        gpu.set_clip_starts(5)
        assert bool((sampler.world_bin == -1).all())  # a probe's start does not count
    finally:
        gpu.close()
    again = TensorG1GetupEnv(64, seed=8, bank=bank, track_clip=SETTING, track_bank=clip, track_sampling=setting)
    finer = TensorG1GetupEnv(64, seed=8, bank=bank, track_clip=SETTING, track_bank=clip,
                             track_sampling={**setting, 'bin': .1})
    try:
        again.restore_training_state(state)
        np.testing.assert_array_equal(again.sampler.g.cpu().numpy(), state['track_bin_g'])
        np.testing.assert_array_equal(again.sampler.history.cpu().numpy(), state['track_bin_history'])
        finer.restore_training_state(state)  # 8 bins against 4: afresh
        assert bool((finer.sampler.g == 0).all()) and bool((finer.sampler.count == 0).all())
    finally:
        again.close()
        finer.close()


# g1_getup task_config track_clips (iteration 78): one tracker of many clips on CUDA, against the CPU task.
# tests/test_g1_getup_track_clips.py's two synthetic clips (drop_bank's back row and front row moving), and a bank whose
# back and front rows lie in the clips' frames 20 and 10 (their nearest frames).
def _clips_bank(clips):
    from test_g1_getup_impact import drop_bank
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0], bank['qpos'][1] = clips[0]['qpos'][20], clips[1]['qpos'][10]
    return bank


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_track_clips():
    """Worlds started without noise from frames of both clips (set_clip_starts with clips) are the CPU task's resets
    track_clip_index / track_frame: qpos, qvel, each world's clip, last frame and end, the episode limit and the
    observation with its 120 critic inputs (the reference ahead held at each clip's own hold frame) match; over five
    steps toward each clip's next frames the observation, the reward and the terminations match within float32 and
    contact noise. Bank starts (set_starts): the rows in the clips' frames 20 and 10 take that clip and frame as on the
    CPU (the nearest table), with the CPU task's alignment and critic inputs; the standing row (no clip class, out of
    reach) frame 0 of either clip."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS
    from test_g1_getup_track import synthetic_clip
    from test_g1_getup_track_clips import SETTINGS, front_clip
    clips = [synthetic_clip(), front_clip()]
    settings = dict(bank=_clips_bank(clips), track_clips=SETTINGS, track_clips_bank=clips,
                    reward_weights={'impact': 2., 'sole_impact': 1.})
    starts = [(0, 0), (0, 20), (0, 39), (1, 0), (1, 10), (1, 29)]
    cpus = [G1GetupEnv(**settings) for _ in starts]
    gpu = TensorG1GetupEnv(len(starts), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        assert gpu.clips_start.tolist() == [0, 41] and gpu.clips_last.tolist() == [39, 29]
        assert gpu.clips_class.tolist() == [0, 1] and gpu.rsi_starts.tolist() == [0, 40]
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(OPTIONS, track_clip_index=c, track_frame=f,
                                                           track_noise=False))[0] for cpu, (c, f) in zip(cpus, starts)])
        gpu.reset()
        obs = gpu.set_clip_starts([f for _, f in starts], clips=[c for c, _ in starts]).cpu().numpy()
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert obs.shape == (len(starts), 610) and gpu.clip_index.tolist() == [c for c, _ in starts]
        assert gpu.clip_last.tolist() == [cpu.clip_last for cpu in cpus]
        assert gpu.clip_end.tolist() == [cpu.clip_end for cpu in cpus]
        np.testing.assert_array_equal(gpu.episode_limit.cpu().numpy(),
                                      [min(gpu.max_episode_steps, cpu.clip_end - f) for cpu, (_, f) in zip(cpus, starts)])
        np.testing.assert_allclose(gpu.qpos.cpu().numpy(), np.stack([c.data.qpos for c in cpus]), atol=1e-3)
        np.testing.assert_allclose(gpu.qvel.cpu().numpy(), np.stack([c.data.qvel for c in cpus]), atol=1e-4)
        np.testing.assert_allclose(obs, cpu_obs, atol=2e-3, rtol=1e-3)
        speeds = np.zeros(610, dtype=bool)
        speeds[task.joint_positions(29, 5, (), (0,), range(3)) + task.joint_positions(29, 5, (), (5,), range(29))] = True
        rng = np.random.default_rng(0)
        for step in range(5):
            action = np.stack([(c.clip['qpos'][min(c.clip_clock + 1, c.clip_last), 7:] - c.home) / c.scale
                               for c in cpus]) + rng.normal(0., .3, (len(starts), 29))
            results = [cpu.step(a) for cpu, a in zip(cpus, action)]
            _, reward, terminated, truncated, terminal = gpu.step(torch.tensor(action, dtype=torch.float32,
                                                                               device=gpu.device))
            gpu.check_health()
            assert terminated.cpu().tolist() == [r[2] for r in results], step
            got, expected = terminal.cpu().numpy(), np.stack([r[0] for r in results])
            np.testing.assert_allclose(got[:, ~speeds], expected[:, ~speeds], atol=.03, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, speeds], expected[:, speeds], atol=.1, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.05, rtol=.02)
            live = ~(terminated | truncated).cpu().numpy()
            assert gpu.clip_clock.cpu().numpy()[live].tolist() == [c.clip_clock for c, k in zip(cpus, live) if k]
        rows = [0, 1, 0, 1, 3, 3]
        obs = gpu.set_starts(rows).cpu().numpy()
        assert (gpu.clip_kind == 2).all() and gpu.start_row.tolist() == rows
        for w, row in enumerate(rows):
            cpu_obs, info = cpus[w].reset(seed=4, options=dict(OPTIONS, start=row))
            if row == 3:  # no clip within reach: frame 0 of a clip drawn (each task its own draw)
                assert gpu.clip_clock[w].item() == 0 and info['track_frame'] == 0
                continue
            assert (gpu.clip_index[w].item(), gpu.clip_clock[w].item()) == ({0: 0, 1: 1}[row], {0: 20, 1: 10}[row])
            assert (info['track_clip_index'], info['track_frame']) == (gpu.clip_index[w].item(), gpu.clip_clock[w].item())
            np.testing.assert_allclose(gpu.clip_yaw[w].item(), cpus[w].clip_yaw, atol=1e-5)
            np.testing.assert_allclose(gpu.clip_offset[w].cpu().numpy(), cpus[w].clip_offset, atol=1e-5)
            np.testing.assert_allclose(obs[w, -task.TRACK_CLIPS_CRITIC_EXTRAS:],
                                       cpu_obs[-task.TRACK_CLIPS_CRITIC_EXTRAS:], atol=1e-4)
        assert bool(gpu.nearest_found[[0, 1]].all()) and not bool(gpu.nearest_found[3])
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_clips_starts_logs_and_sampling():
    """track_clips' training starts on CUDA, 4096 worlds (track_starts 0.4 / 0.4 / 0.2, the bank's back and front rows):
    the kinds' shares; a first-frame start's clip uniform over the clips; RSI uniform over every clip's frames (the back
    clip 40 of 70), each world's clock within its own clip; the bank rows in the clips' frames 20 and 10 at that clip
    and frame, the spinning front row (out of reach) at frame 0 of the front clip; each world's last frame, end and
    episode limit its own clip's. The logs: track_done_first and track_standing per start class (side: no clip, NaN),
    track_clips_done of track_clips_started. With track_sampling at bins of 0.5 s the bins lie within each clip
    ([0, 25) [25, 40) [40, 65) [65, 70) of the RSI frames) and every RSI world's clock inside its bin."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from test_g1_getup_track import synthetic_clip
    from test_g1_getup_track_clips import SETTINGS, front_clip
    clips = [synthetic_clip(), front_clip()]
    bank = _clips_bank(clips)
    settings = dict(bank=bank, track_clips=SETTINGS, track_clips_bank=clips,
                    track_starts={'first': .4, 'rsi': .4, 'bank': .2}, track_bank_classes=['back', 'front'])
    n = 4096
    gpu = TensorG1GetupEnv(n, seed=7, **settings)
    try:
        gpu.reset()
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu._masked_reset(torch.ones(n, dtype=torch.bool, device=gpu.device))
        kinds = gpu.clip_kind.bincount(minlength=3).cpu().numpy() / n
        np.testing.assert_allclose(kinds, (.4, .4, .2), atol=.025)
        first, rsi, bank_kind = gpu.clip_kind == 0, gpu.clip_kind == 1, gpu.clip_kind == 2
        assert abs(gpu.clip_index[first].float().mean().item() - .5) < .05
        assert abs((gpu.clip_index[rsi] == 0).float().mean().item() - 40 / 70) < .05
        assert bool((gpu.clip_clock[first] == 0).all()) and bool((gpu.clip_clock <= gpu.clip_last).all())
        assert len(gpu.clip_clock[rsi & (gpu.clip_index == 1)].unique()) == 30
        assert torch.equal(gpu.clip_last, gpu.clips_last[gpu.clip_index])
        assert torch.equal(gpu.clip_end, gpu.clips_end[gpu.clip_index])
        np.testing.assert_array_equal(gpu.episode_limit.cpu().numpy(), np.minimum(
            gpu.max_episode_steps, (gpu.clip_end - gpu.clip_clock).cpu().numpy()))
        expected = {0: (0, 20), 1: (1, 10), 2: (1, 0)}
        for row, (clip, frame) in expected.items():
            mine = bank_kind & (gpu.start_row == row)
            assert bool(mine.any()), row
            assert bool((gpu.clip_index[mine] == clip).all()) and bool((gpu.clip_clock[mine] == frame).all()), row
        gpu.rollout_metrics()
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        gpu.episode_limit.clamp_(max=20)
        for _ in range(40):
            gpu.step(torch.randn((n, 29), device=gpu.device, generator=generator) * .5)
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        for key in ('track_done_first', 'track_standing', *(f'track_{kind}_{cls}' for kind in ('done_first', 'standing')
                                                             for cls in task.TRACK_CLASSES),
                    'track_clips_done', 'track_clips_started'):
            assert f'rollout/{key}' in metrics, key
        assert math.isnan(metrics['rollout/track_done_first_side']) and math.isnan(metrics['rollout/track_standing_side'])
        assert 0 <= metrics['rollout/track_clips_done'] <= metrics['rollout/track_clips_started'] == 2
        assert metrics['rollout/bad_states'] == 0
    finally:
        gpu.close()
    sampling = {'bin': .5, 'uniform': .5, 'hard': .4, 'history': .1, 'temperature': 25}
    gpu = TensorG1GetupEnv(n, seed=8, track_sampling=sampling, **settings)
    try:
        sampler = gpu.sampler
        assert sampler.bins == 4 and sampler.starts.tolist() == [0, 25, 40, 65]
        assert sampler.lengths.tolist() == [25, 15, 25, 5]
        gpu.reset()
        sampler.g.copy_(torch.tensor([.9, .1, .9, .9], dtype=torch.float64, device=gpu.device))
        with gpu.wp.ScopedStream(gpu.wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            gpu._masked_reset(torch.ones(n, dtype=torch.bool, device=gpu.device))
        rsi = gpu.clip_kind == 1
        bins = sampler.world_bin[rsi]
        frames = gpu.rsi_starts[gpu.clip_index[rsi]] + gpu.clip_clock[rsi]
        assert bool((sampler.starts[bins] <= frames).all()) and bool((frames < sampler.starts[bins]
                                                                      + sampler.lengths[bins]).all())
        assert (bins == 1).float().mean().item() > .4  # the hard bin of the back clip
        assert bool((gpu.clip_index[rsi] == (bins >= 2).long()).all())
    finally:
        gpu.close()


# g1_getup task_config track_end_reach (iteration 78, the critic's it78): an ankle farther than the setting from its
# place in the re-anchored clip ends the episode, on CUDA as on the CPU task (tests/test_g1_getup_track_reach.py).
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_track_end_reach(monkeypatch):
    """track_end_reach 0.4 with the reference's left ankle moved 0.6 m from clip frame 15 on (both tasks' frames alike):
    worlds started without noise from frames 0, 5, 12 and 20 end on the step their clock reaches 15 (never, 10, 3, 1),
    on CUDA as on the CPU task, by the reach (task.track_reach_terminated's flags per world against the CPU info
    track_end_reach, every step until each world's end); rollout/track_end_reach is logged, within track_failed. Bank
    starts in the clip's first pose with the ankle moved throughout end on step TRACK_GRACE_SECONDS/dt + 1
    in both. track_clips: only the worlds on the clip whose reference moved end, on the same step as on the CPU."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    from test_g1_getup_track_clips import SETTINGS, front_clip
    ankle = task.TRACK_BODIES.index('left_ankle_roll_link')
    seen, original = [], task.track_reach_terminated

    def spy(*args, **kwargs):
        ended = original(*args, **kwargs)
        if isinstance(ended, torch.Tensor):
            seen.append(ended.cpu().numpy().copy())
        return ended

    def shifted(frames, rows):  # frames' pos with the left ankle 0.6 m along world x at rows (numpy or torch)
        pos = frames['pos'].clone() if isinstance(frames['pos'], torch.Tensor) else frames['pos'].copy()
        pos[rows, ankle, 0] += .6
        return {**frames, 'pos': pos}

    def follow(cpus, gpu, steps, expected):
        """Steps toward each CPU world's next clip frames (zeros for a world already ended): the CPU and CUDA ends
        and the CUDA reach flags against the CPU info, world by world until its end; expected: each world's end step
        (None: no end within steps)."""
        live = np.ones(len(cpus), dtype=bool)
        ends = [None] * len(cpus)
        for step in range(1, steps + 1):
            action = np.stack([(c.clip['qpos'][min(c.clip_clock + 1, c.clip_last), 7:] - c.home) / c.scale if k
                               else np.zeros(29) for c, k in zip(cpus, live)])
            seen.clear()
            results = {w: cpus[w].step(action[w]) for w in np.flatnonzero(live)}
            _, _, terminated, _, _ = gpu.step(torch.tensor(action, dtype=torch.float32, device=gpu.device))
            gpu.check_health()
            (flags,) = seen
            for w, (_, _, cpu_terminated, _, info) in results.items():
                assert bool(terminated[w]) is cpu_terminated, (step, w)
                assert bool(flags[w]) is info['track_end_reach'], (step, w)
                if cpu_terminated:
                    assert info['track_end_reach'] and info['track_failed']
                    ends[w], live[w] = step, False
        assert ends == expected
        return ends

    monkeypatch.setattr(task, 'track_reach_terminated', spy)
    clip = synthetic_clip()
    bank = drop_bank()
    bank['qpos'] = bank['qpos'].copy()
    bank['qpos'][0] = clip['qpos'][0]
    settings = dict(bank=bank, track_clip=SETTING, track_bank=clip, track_end_reach=.4)
    frames = [0, 5, 12, 20]
    cpus = [G1GetupEnv(**settings) for _ in frames]
    gpu = TensorG1GetupEnv(len(frames), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        assert gpu.track_reach == .4 and gpu.track_reach_bodies == cpus[0].track_reach_bodies
        for cpu, f in zip(cpus, frames):
            cpu.reset(seed=4, options=dict(OPTIONS, track_frame=f, track_noise=False))
            cpu.clip_frames = shifted(cpu.clip_frames, slice(15, None))
        gpu.reset()
        gpu.clip = shifted(gpu.clip, slice(15, None))
        gpu.set_clip_starts(frames)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        gpu.rollout_metrics()
        follow(cpus, gpu, 12, [None, 10, 3, 1])
        metrics = gpu.rollout_metrics()
        # the three ends, and whatever the ended worlds' training starts did after them under zero actions
        assert 0. < metrics['rollout/track_end_reach'] <= metrics['rollout/track_failed'] <= 1.
        # Bank starts in the clip's first pose (frame 0, aligned there), the ankle moved throughout: the grace.
        gpu.clip = shifted(gpu.clip, slice(0, 15))  # now every row
        for cpu in cpus:
            cpu.clip_frames = shifted(cpu.clip_frames, slice(0, 15))
            cpu.reset(seed=4, options=dict(OPTIONS, start=0))
            assert cpu.clip_start == 'bank' and cpu.clip_clock == 0
        gpu.set_starts([0] * len(frames))
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert (gpu.clip_kind == 2).all() and (gpu.clip_clock == 0).all()
        grace = cpus[0].clip_grace
        follow(cpus, gpu, grace + 3, [grace + 1] * len(frames))
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    # track_clips: the front clip's reference moved from its frame 15 on, the back clip's not.
    clips = [synthetic_clip(), front_clip()]
    settings = dict(bank=drop_bank(), track_clips=SETTINGS, track_clips_bank=clips, track_end_reach=.4)
    starts = [(0, 5), (0, 20), (1, 5), (1, 20)]
    cpus = [G1GetupEnv(**settings) for _ in starts]
    gpu = TensorG1GetupEnv(len(starts), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        for cpu, (c, f) in zip(cpus, starts):
            cpu.clips[1]['frames'] = shifted(cpu.clips[1]['frames'], slice(15, None))
            cpu.reset(seed=4, options=dict(OPTIONS, track_clip_index=c, track_frame=f, track_noise=False))
        gpu.reset()
        first = int(gpu.clips_start[1])
        gpu.clip = shifted(gpu.clip, slice(first + 15, first + int(gpu.clips_last[1]) + 2))  # with its hold row
        gpu.set_clip_starts([f for _, f in starts], clips=[c for c, _ in starts])
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        follow(cpus, gpu, 12, [None, None, 10, 1])
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_clip_4096_worlds_speed():
    """The training batch of the lab clip (or the synthetic one): 4096 worlds with the training randomization, σ 1
    action noise around the clip's targets for 200 steps: no contact buffer overflow or non-finite state; prints the
    steps per second (the getting-up task: ~38-50 thousand) and the logs."""
    import time
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_getup_track
    from test_g1_getup_track import REAL, SETTING, synthetic_clip
    if g1_getup_track.clip_path(REAL).is_file() and bank_file_present():
        clip = g1_getup_track.load(REAL)
        settings = dict(track_clip={'file': REAL, 'md5': clip['md5']})
    else:
        from test_g1_getup_impact import drop_bank
        settings = dict(bank=drop_bank(), track_clip=SETTING, track_bank=synthetic_clip())
    n = 4096
    env = TensorG1GetupEnv(n, seed=11, reward_weights={'impact': 2., 'sole_impact': 1.}, **settings)
    try:
        env.reset()
        generator = torch.Generator(device=env.device).manual_seed(5)

        def actions():
            frame = (env.clip_clock + 1).clamp(max=env.clip_last)
            return (env.clip_qpos[frame, 7:] - env.home) / env.scale + torch.randn((n, 29), device=env.device,
                                                                                 generator=generator)
        for _ in range(10):
            env.step(actions())
        env.check_health()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            env.step(actions())
            env.check_health()
        torch.cuda.synchronize()
        speed = 200 * n / (time.perf_counter() - started)
        metrics = env.rollout_metrics()
        print(f'g1_getup track_clip 4096 worlds: {speed:,.0f} steps/s; ' + ', '.join(
            f'{key[8:]} {value:.3g}' for key, value in metrics.items()))
        assert metrics['rollout/ncon_max'] < 128 and metrics['rollout/bad_states'] == 0
        assert speed > 10000
    finally:
        env.close()


def bank_file_present():
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    return bank_module.BANK.is_file()


# g1_getup task_config settle_clip (iteration 66): settling into the nearest clip start on CUDA, against the CPU task.
# tests/test_g1_getup_settle.py's two-frame clips (drop_bank's home pose lying on the back and on the front).
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_settle_clip(monkeypatch):
    """Worlds started from drop_bank's rows (back, front, spinning front, back) are the CPU task's resets start: qpos,
    qvel and the observation match; over ten steps of the same actions (each row's target held, noise on top) the
    observation, the reward, every settling term (task.settle_reward_components, numpy and torch: the errors, the
    nearest target, stillness, self_contact, impact, sole_impact) and the hands' floor force match within
    float32-vs-float64 and contact noise; then, holding the targets without noise, both settle into the same target
    within a few steps of each other."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_settle import SETTING, lying_clip
    bank = drop_bank()
    settings = dict(bank=bank, settle_clip=SETTING, settle_bank=[lying_clip(0), lying_clip(1)],
                    reward_weights={'impact': 2., 'sole_impact': 1.})
    rows = [0, 1, 2, 0]
    seen, original = [], task.settle_reward_components

    def spy(*args, **kwargs):
        components, nearest, near = original(*args, **kwargs)
        on_gpu = isinstance(args[0], torch.Tensor)
        seen.append((on_gpu, {key: np.asarray(value.cpu() if on_gpu else value, dtype=float).copy()
                              for key, value in components.items()},
                     np.asarray(nearest.cpu() if on_gpu else nearest).copy()))
        return components, nearest, near

    monkeypatch.setattr(task, 'settle_reward_components', spy)
    cpus = [G1GetupEnv(**settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        assert gpu.max_episode_steps == 350 and not gpu.assist.any()
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(OPTIONS, start=row))[0] for cpu, row in zip(cpus, rows)])
        gpu.reset()
        obs = gpu.set_starts(rows).cpu().numpy()
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert obs.shape == (len(rows), 496)
        np.testing.assert_allclose(gpu.qpos.cpu().numpy(), np.stack([c.data.qpos for c in cpus]), atol=1e-5)
        np.testing.assert_allclose(obs, cpu_obs, atol=2e-3, rtol=1e-3)
        targets = np.stack([(c.settle['position'][min(row, 1)] - c.home) / c.scale for c, row in zip(cpus, rows)])
        speeds = np.zeros(496, dtype=bool)
        speeds[task.joint_positions(29, 5, (), (0,), range(3)) + task.joint_positions(29, 5, (), (5,), range(29))] = True
        rng = np.random.default_rng(0)
        for step in range(10):
            action = targets + rng.normal(0., .3, targets.shape)
            seen.clear()
            results = [cpu.step(a) for cpu, a in zip(cpus, action)]
            _, reward, terminated, truncated, terminal = gpu.step(torch.tensor(action, dtype=torch.float32,
                                                                               device=gpu.device))
            gpu.check_health()
            assert terminated.cpu().tolist() == [r[2] for r in results], step
            got, expected = terminal.cpu().numpy(), np.stack([r[0] for r in results])
            np.testing.assert_allclose(got[:, ~speeds], expected[:, ~speeds], atol=.03, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(got[:, speeds], expected[:, speeds], atol=.1, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(reward.cpu().numpy(), [r[1] for r in results], atol=.05, rtol=.02)
            np.testing.assert_allclose(gpu.hand_substeps.amax(dim=1).cpu().numpy(),
                                       [r[4]['hand_force'] for r in results], atol=5., rtol=.1)
            cpu_seen = [(terms, nearest) for on_gpu, terms, nearest in seen if not on_gpu]
            ((gpu_terms, gpu_nearest),) = [(terms, nearest) for on_gpu, terms, nearest in seen if on_gpu]
            assert gpu_nearest.tolist() == [int(nearest[0]) for _, nearest in cpu_seen]
            for key in gpu_terms:
                np.testing.assert_allclose(gpu_terms[key], [terms[key][0] for terms, _ in cpu_seen], atol=.03, rtol=.02,
                                           err_msg=f'{key} step {step}')
        # Holding the targets: both backends settle each world into the same target at nearly the same step.
        cpu_at = [None] * len(rows)
        for step in range(200):
            results = [cpu.step(a) for cpu, a in zip(cpus, targets)]
            gpu.step(torch.tensor(targets, dtype=torch.float32, device=gpu.device))
            for w, result in enumerate(results):
                if cpu_at[w] is None and result[4]['settle_done']:
                    cpu_at[w] = (cpus[w].steps, result[4]['settle_target'])
        gpu.check_health()
        assert all(at is not None for at in cpu_at)
        gpu_at, gpu_target = gpu.settle_at.cpu().tolist(), gpu.settle_target.cpu().tolist()
        for w, (steps, name) in enumerate(cpu_at):
            assert gpu_at[w] >= 0 and abs(gpu_at[w] - steps) <= 10 and gpu.settle_names[gpu_target[w]] == name, w
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_settle_clip_starts_logs_and_speed():
    """4096 worlds with the training randomization: training resets draw the bank's rows of the settling classes only
    (never the standing row), no assist, episodes within 350 steps (a full reset() cuts the first ones within that);
    random actions for 40 steps give the settle logs and no overflow or non-finite state; worlds lying on the back with
    their motors holding the back target settle into it (settle_at, settle_target) and an ended episode counts in
    rollout/settle_done; prints the steps per second of σ 1 action noise around the targets. Off: no settle logs."""
    import time
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from test_g1_getup_impact import drop_bank
    from test_g1_getup_settle import SETTING, lying_clip
    bank = drop_bank()
    n = 4096
    gpu = TensorG1GetupEnv(n, seed=7, bank=bank, settle_clip=SETTING, settle_bank=[lying_clip(0), lying_clip(1)],
                           reward_weights={'impact': 2., 'sole_impact': 1.})
    try:
        gpu.reset()
        assert not gpu.assist.any() and bool((gpu.episode_limit <= 350).all())
        assert set(gpu.start_row.unique().cpu().tolist()) == {0, 1, 2}  # row 3 stands
        gpu.rollout_metrics()
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        for _ in range(40):
            gpu.step(torch.randn((n, 29), device=gpu.device, generator=generator))
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        for key in ('settle_done', 'settle_seconds', 'settle_target0', 'settle_target1', 'settle_angle',
                    'settle_deviation', 'hand_force', 'self_contact', 'impact', 'sole_impact'):
            assert f'rollout/{key}' in metrics, key
        assert metrics['rollout/bad_states'] == 0 and metrics['rollout/ncon_max'] < 128
        assert metrics['rollout/settle_angle'] > 0
        gpu.set_starts(0)
        hold = ((gpu.settle['position'][0] - gpu.home) / gpu.scale).expand(n, -1).contiguous()
        for _ in range(150):
            gpu.step(hold)
        gpu.check_health()
        settled = gpu.settle_at >= 0
        assert settled.float().mean().item() > .9 and bool((gpu.settle_target[settled] == 0).all())
        gpu.episode_limit.copy_(gpu.lengths + 1)  # end every episode on the next step
        gpu.step(hold)
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/settle_done'] > .9 and metrics['rollout/settle_target0'] == 1.
        assert 0 < metrics['rollout/settle_seconds'] < 3
        gpu.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(200):
            gpu.step(hold + torch.randn((n, 29), device=gpu.device, generator=generator))
            gpu.check_health()
        torch.cuda.synchronize()
        speed = 200 * n / (time.perf_counter() - started)
        metrics = gpu.rollout_metrics()
        print(f'g1_getup settle_clip 4096 worlds: {speed:,.0f} steps/s; ' + ', '.join(
            f'{key[8:]} {value:.3g}' for key, value in metrics.items()))
        assert metrics['rollout/ncon_max'] < 128 and metrics['rollout/bad_states'] == 0 and speed > 10000
    finally:
        gpu.close()
    plain = TensorG1GetupEnv(64, seed=5, bank=bank)
    try:
        assert plain.settle is None
        plain.reset()
        plain.step(torch.zeros((64, 29), device=plain.device))
        assert not any('settle' in key or 'hand_force' in key for key in plain.rollout_metrics())
    finally:
        plain.close()


# g1_getup task_config settle_success and settle_starts (iteration 85): the success by joints and the starts from the
# target on CUDA, against the CPU task, with tests/test_g1_getup_settle.py's two-frame clips.
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_settle_success_and_target_starts():
    """settle_success: worlds from the back and front rows holding their targets (one with the left shoulder roll
    0.5 rad off) settle by the new rule at nearly the CPU task's steps into the same target, the one off never, and
    the worst arm joint matches the CPU info. settle_starts over 1024 worlds: the target share within 0.06, the bank
    starts of `classes` only, every target start within the noise of a target's joints and out of the floor; its own
    generator, so a world that does not start from the target has the start of the same seed without the setting bit
    for bit; the logs per start class and the worst arm joint; set_target_starts."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_settle import SETTING, lying_clip
    from test_g1_getup_settle_success import SUCCESS
    bank = drop_bank()
    settings = dict(bank=bank, settle_clip=SETTING, settle_bank=[lying_clip(0), lying_clip(1)],
                    reward_weights={'impact': 2., 'sole_impact': 1.}, settle_success=SUCCESS)
    rows = [0, 0, 1, 2]
    cpus = [G1GetupEnv(**settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        assert gpu.settle_hold == 15 and gpu.settle_arms == cpus[0].settle_arms
        for cpu, row in zip(cpus, rows):
            cpu.reset(seed=4, options=dict(OPTIONS, start=row))
        gpu.reset()
        gpu.set_starts(rows)
        gpu.set_action_delay_steps(0)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        offset = np.zeros(29)
        offset[16] = .5
        targets = np.stack([(c.settle['position'][min(row, 1)] + (offset if w == 1 else 0.) - c.home) / c.scale
                            for w, (c, row) in enumerate(zip(cpus, rows))])
        cpu_at, info = [None] * len(rows), [None] * len(rows)
        for _ in range(200):
            results = [cpu.step(a) for cpu, a in zip(cpus, targets)]
            gpu.step(torch.tensor(targets, dtype=torch.float32, device=gpu.device))
            for w, result in enumerate(results):
                info[w] = result[4]
                if cpu_at[w] is None and result[4]['settle_done']:
                    cpu_at[w] = (cpus[w].steps, result[4]['settle_target'])
        gpu.check_health()
        assert cpu_at[1] is None and all(cpu_at[w] is not None for w in (0, 2, 3))
        gpu_at, gpu_target = gpu.settle_at.cpu().tolist(), gpu.settle_target.cpu().tolist()
        assert gpu_at[1] < 0
        for w in (0, 2, 3):
            steps, name = cpu_at[w]
            assert gpu_at[w] >= 0 and abs(gpu_at[w] - steps) <= 10 and gpu.settle_names[gpu_target[w]] == name, w
        worst = task.settle_arm_worst(gpu.qpos[:, gpu.qadr], gpu.settle['position'], gpu.settle_arms)
        nearest = [min(row, 1) for row in rows]
        np.testing.assert_allclose(worst.cpu().numpy()[np.arange(len(rows)), nearest],
                                   [i['settle_arm_worst'] for i in info], atol=.02)
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    n = 1024  # three tasks at once: 4096 worlds each run out of the 16 GB in Warp's collision buffers
    common = dict(bank=bank, settle_clip=SETTING, settle_bank=[lying_clip(0), lying_clip(1)],
                  reward_weights={'impact': 2., 'sole_impact': 1.})
    old = TensorG1GetupEnv(n, seed=7, **common)
    new = TensorG1GetupEnv(n, seed=7, settle_starts={'target': .5, 'noise': 1}, **common)
    front = TensorG1GetupEnv(n, seed=7, settle_starts={'target': .5, 'classes': ['front']}, settle_success=SUCCESS,
                             **common)
    try:
        old.reset()
        new.reset()
        front.reset()
        bank_start = new.start_row >= 0
        share = 1 - bank_start.float().mean().item()
        assert abs(share - .5) < .06  # 4σ at 1024 worlds
        assert torch.equal(new.start_row[bank_start], old.start_row[bank_start])
        assert torch.equal(new.qpos[bank_start], old.qpos[bank_start])
        assert torch.equal(new.friction, old.friction) and torch.equal(new.delay_steps, old.delay_steps)
        assert torch.equal(new.episode_limit, old.episode_limit)
        target = ~bank_start
        joints = new.qpos[target][:, new.qadr]
        nearest = (joints[:, None] - new.settle['position'][None]).abs().amax(-1).amin(-1)
        assert nearest.max().item() <= .1 + 1e-5
        assert new._floor_depth()[target].min().item() > -2e-3  # raised out of the floor
        assert set(front.start_row.unique().cpu().tolist()) == {-1, 1, 2}  # the front rows and the target
        assert bool((front.settle_class[front.start_row < 0] == len(front.settle_class_names) - 1).all())
        front.rollout_metrics()
        generator = torch.Generator(device=front.device).manual_seed(1)
        for _ in range(30):
            front.step(torch.randn((n, 29), device=front.device, generator=generator))
            front.check_health()
        front.episode_limit.copy_(front.lengths + 1)
        front.step(torch.zeros((n, 29), device=front.device))
        metrics = front.rollout_metrics()
        for key in ('settle_done_front', 'settle_done_target', 'settle_done_back', 'settle_arm_worst'):
            assert f'rollout/{key}' in metrics, key
        assert math.isnan(metrics['rollout/settle_done_back']) and metrics['rollout/settle_arm_worst'] > 0
        assert 'rollout/settle_arm_worst' not in new.rollout_metrics()
        assert not any(key.startswith('rollout/settle_done_') for key in old.rollout_metrics())
        obs = old.set_target_starts(1)
        assert obs.shape == (n, 496) and bool((old.start_row == -1).all())
        assert (old.qpos[:, old.qadr] - old.settle['position'][1]).abs().amax().item() <= .1 + 1e-5
    finally:
        for env in (old, new, front):
            env.close()


# g1_getup task_config track_handover_starts (iteration 83): track_clip's starts from the hand-over bank on CUDA, with
# tests/test_g1_getup_track_handover.py's synthetic bank (the synthetic clip's task under random actions).
@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_getup_track_handover_starts():
    """Worlds started from hand-over rows 0-3 (set_handover_starts) are the CPU task's resets track_handover: qpos,
    qvel, the clock at the bank's frame, kind 3, the reference's alignment, the observation with its critic extras, and
    under a delay of 10 substeps the row's newest action in flight; over three steps of the same actions the
    observation, reward and terminations match within float32 and contact noise. Training resets with share 0.3 over
    4096 worlds: the hand-over share within 0.025, the other kinds 0.7 of TRACK_START_SHARES, every hand-over world at
    its row's state and the bank's frame; its own generator, so the worlds that do not start there have the old start
    of the same seed bit for bit (kind, clock, row, qpos, alignment). The logs carry track_done_handover and
    track_standing_handover with the setting only; set_handover_starts without it is refused."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_g1_getup_impact import OPTIONS, drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    from test_g1_getup_track_handover import FRAME, HANDOVER, handover_bank
    bank, clip, handover = drop_bank(), synthetic_clip(), handover_bank()
    # self_contact off: the synthetic rows press limbs together at ~10 N, where MuJoCo C and MJWarp differ by 10-30 %
    # (row 1: 7 against 5 substeps above the threshold, −7 against −5), contact noise, not the starts this tests.
    settings = dict(bank=bank, track_clip=SETTING, track_bank=clip, track_handover_starts=HANDOVER,
                    handover_bank=handover, reward_weights={'impact': 2., 'sole_impact': 1.},
                    track_weights={'self_contact': 0.})
    rows = [0, 1, 2, 3]
    cpus = [G1GetupEnv(**settings) for _ in rows]
    gpu = TensorG1GetupEnv(len(rows), seed=4, observation_noise=False, randomize=False, **settings)
    try:
        cpu_obs = np.stack([cpu.reset(seed=4, options=dict(OPTIONS, action_delay_steps=10, track_handover=row))[0]
                            for cpu, row in zip(cpus, rows)])
        gpu.reset()
        obs = gpu.set_handover_starts(rows).cpu().numpy()
        gpu.set_action_delay_steps(10)
        gpu.set_motor_friction(1.)
        gpu.set_body_friction(.6)
        assert (gpu.clip_kind == 3).all() and (gpu.clip_clock == FRAME).all() and (gpu.start_row == -1).all()
        np.testing.assert_allclose(gpu.qpos.cpu().numpy(), handover['qpos'][rows], atol=1e-6)
        np.testing.assert_allclose(gpu.qvel.cpu().numpy(), handover['qvel'][rows], atol=1e-5)
        np.testing.assert_allclose(gpu.delay_actions[:, 0].cpu().numpy(), handover['actions'][rows, 0], atol=1e-6)
        for cpu in cpus:
            assert len(cpu.delay_actions) == 1
        np.testing.assert_allclose(gpu.clip_yaw.cpu().numpy(), [c.clip_yaw for c in cpus], atol=1e-5)
        np.testing.assert_allclose(gpu.clip_offset.cpu().numpy(), np.stack([c.clip_offset for c in cpus]), atol=1e-5)
        np.testing.assert_allclose(obs, cpu_obs, atol=2e-3, rtol=1e-3)
        np.testing.assert_array_equal(gpu.episode_limit.cpu().numpy(), min(gpu.max_episode_steps, gpu.clip_end - FRAME))
        rng = np.random.default_rng(0)
        live = np.ones(len(rows), dtype=bool)  # a world that ended is reset on CUDA and not on the CPU: left out
        for step in range(3):
            action = np.stack([(c.clip['qpos'][min(c.clip_clock + 1, c.clip_last), 7:] - c.home) / c.scale
                               for c in cpus]) + rng.normal(0., .3, (len(rows), 29))
            results = [cpu.step(a) for cpu, a in zip(cpus, action)]
            _, reward, terminated, truncated, terminal = gpu.step(torch.tensor(action, dtype=torch.float32,
                                                                               device=gpu.device))
            gpu.check_health()
            assert terminated.cpu().numpy()[live].tolist() == [r[2] for r, k in zip(results, live) if k], step
            np.testing.assert_allclose(reward.cpu().numpy()[live], [r[1] for r, k in zip(results, live) if k],
                                       atol=.05, rtol=.02, err_msg=f'step {step}')
            np.testing.assert_allclose(terminal.cpu().numpy()[live], np.stack([r[0] for r in results])[live], atol=.1,
                                       rtol=.02, err_msg=f'step {step}')
            live &= ~(terminated | truncated).cpu().numpy()
        assert live.sum() >= 2  # the synthetic rows move fast: some end at once by the anchor's rules
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
    n = 4096
    plain = TensorG1GetupEnv(n, seed=7, bank=bank, track_clip=SETTING, track_bank=clip)
    gpu = TensorG1GetupEnv(n, seed=7, bank=bank, track_clip=SETTING, track_bank=clip,
                           track_handover_starts={**HANDOVER, 'share': .3}, handover_bank=handover)
    try:
        with pytest.raises(ValueError, match='track_handover_starts'):
            plain.set_handover_starts(0)
        for env in (plain, gpu):
            env.reset()
        other = gpu.clip_kind != 3
        for name in ('clip_kind', 'clip_clock', 'start_row', 'clip_yaw', 'clip_offset', 'qpos', 'qvel'):
            assert torch.equal(getattr(plain, name)[other], getattr(gpu, name)[other]), name
        kinds = gpu.clip_kind.bincount(minlength=4).cpu().numpy() / n
        np.testing.assert_allclose(kinds, (*(.7 * s for s in task.TRACK_START_SHARES), .3), atol=.025)
        handing = ~other
        assert bool((gpu.clip_clock[handing] == FRAME).all()) and bool((gpu.start_row[handing] == -1).all())
        states = torch.as_tensor(handover['qpos'], dtype=torch.float32, device=gpu.device)
        distance = (gpu.qpos[handing][:, None] - states[None]).abs().amax(dim=2).amin(dim=1)
        assert float(distance.max()) < 1e-5  # each one some row's state, exactly
        gpu.rollout_metrics()
        plain.rollout_metrics()
        generator = torch.Generator(device=gpu.device).manual_seed(1)
        for _ in range(60):
            actions = torch.randn((n, 29), device=gpu.device, generator=generator) * .5
            gpu.step(actions)
            plain.step(actions)
            gpu.check_health()
        metrics = gpu.rollout_metrics()
        assert 'rollout/track_done_handover' in metrics and 'rollout/track_standing_handover' in metrics
        assert metrics['rollout/bad_states'] == 0
        assert not any('handover' in key for key in plain.rollout_metrics())
    finally:
        plain.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_getup_track_handover_grace():
    """track_handover_starts' grace on CUDA, as the CPU task's test: hand-over worlds in the clip's first pose under a
    reference lifted 0.6 m end on their first step without grace and on step 101 with grace 2 s (the anchor's rules
    wait too)."""
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from test_g1_getup_impact import drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    from test_g1_getup_track_handover import HANDOVER, handover_bank
    clip = synthetic_clip()
    bank = handover_bank(frame=0)
    bank['qpos'], bank['qvel'] = bank['qpos'].copy(), np.zeros_like(bank['qvel'])
    bank['qpos'][0] = clip['qpos'][0]
    for grace, ends_at in ((None, 1), (2., 101)):
        setting = {**HANDOVER, **({} if grace is None else {'grace': grace})}
        gpu = TensorG1GetupEnv(4, seed=4, observation_noise=False, randomize=False, bank=drop_bank(),
                               track_clip=SETTING, track_bank=clip, track_handover_starts=setting, handover_bank=bank)
        try:
            gpu.reset()
            gpu.set_handover_starts(0)
            gpu.set_action_delay_steps(0)
            gpu.clip['pos'][..., 2] += .6
            ended = None
            for step in range(1, 121):
                frame = min(int(gpu.clip_clock[0]) + 1, gpu.clip_last)
                target = torch.as_tensor((clip['qpos'][frame, 7:] - gpu.reference.home) / gpu.reference.scale,
                                         dtype=torch.float32, device=gpu.device).expand(4, -1)
                _, _, terminated, _, _ = gpu.step(target.contiguous())
                if bool(terminated.any()):
                    assert bool(terminated.all())
                    ended = step
                    break
            assert ended == ends_at, grace
        finally:
            gpu.close()

@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_terrain(monkeypatch):
    """task_config terrain on both backends: four worlds on the hardest stairs, slopes and blocks, each against the CPU
    task on the same tile. The same start (the tile's centre, the platform's height), the same heights above the ground
    under the body handed to the shared reward (pelvis, ankle_roll origins, soles; feet_flat against the ground's
    normal), close rewards over ten steps; then the curriculum: a path walked well moves a world up, a short one down,
    each respawns on its new tile; training_state keeps the levels, rollout_metrics logs them."""
    import mujoco
    import warp as wp
    import humanoid_sim.g1_tensor_env as tensor_env
    from humanoid_sim.tasks import g1_terrain, g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    seen, original = [], tensor_env.reward_components

    def spy(*args, **kwargs):
        components, fell = original(*args, **kwargs)
        seen.append((args, kwargs, components))
        return components, fell

    monkeypatch.setattr(tensor_env, 'reward_components', spy)
    profile = dict(real_robot=True, action_limit=10., sole_contact=True, lock_wrists=True, knee_floor=.061,
                   foot_collision=True, terrain={},
                   reward_weights={'feet_flat': 2., 'height': 10., 'clearance': 1., 'fall': 100.})
    tiles = (('stairs_up', 9), ('stairs_down', 9), ('slope_down', 9), ('boxes', 6))
    cpus = [G1VelocityEnv(**profile) for _ in tiles]
    gpu = TensorG1Env(4, seed=4, max_episode_steps=12, resample_commands=False, push_velocity=0,
                      observation_noise=False, randomize=False, **profile)
    try:
        torch.testing.assert_close(gpu.terrain_heights.cpu().double(), cpus[0].terrain_heights, atol=1e-7, rtol=0)
        kinds = [g1_terrain.TERRAIN_TYPES.index(kind) for kind, _ in tiles]
        gpu.terrain_type.copy_(torch.tensor(kinds, device=gpu.device))
        gpu.set_terrain_level([level for _, level in tiles])
        gpu.reset()
        for world, (cpu, (kind, level)) in enumerate(zip(cpus, tiles)):
            cpu.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': kind, 'terrain_level': level})
            np.testing.assert_allclose(gpu.qpos[world, :3].cpu().numpy(), cpu.data.qpos[:3], atol=1e-5, rtol=0)
            assert float(gpu.pelvis_height()[world]) == pytest.approx(.8, abs=1e-5)
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.5, 0., 0.]], device=gpu.device).expand(4, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()
        for step in range(10):
            results = [cpu.step(np.zeros(29)) for cpu in cpus]
            gpu.step(torch.zeros((4, 29), device=gpu.device))
            gpu.check_health()
            args, kwargs, components = seen[-1]
            ground = gpu._ground_terms()
            torch.testing.assert_close(args[2], ground['height'])
            torch.testing.assert_close(args[7], ground['foot_height'])
            torch.testing.assert_close(kwargs['sole_height'], task.sole_height(ground['spheres'], gpu.sole_radius))
            torch.testing.assert_close(kwargs['floor_normal'], ground['floor_normal'])
            for world, (cpu, (_, _, fell, _, info)) in enumerate(zip(cpus, results)):
                cpu_terms = cpu._ground_terms()
                np.testing.assert_allclose(float(args[2][world]), float(cpu_terms['height'][0]), atol=2e-3, rtol=0)
                np.testing.assert_allclose(ground['foot_height'][world].cpu().numpy(),
                                           cpu_terms['foot_height'][0].numpy(), atol=2e-3, rtol=0)
                np.testing.assert_allclose(kwargs['sole_height'][world].cpu().numpy(), cpu._sole_height()[0].numpy(),
                                           atol=2e-3, rtol=0)
                assert not fell and not bool(gpu.lengths[world] == 0)
                for name in ('height', 'feet_flat', 'gait', 'clearance', 'alive'):
                    np.testing.assert_allclose(float(components[name][world]), info['reward_components'][name],
                                               atol=.02, rtol=.05, err_msg=f'{name} world {world} step {step}')
        # The curriculum at the episodes' end (12 steps): world 0 walked 0.9 of 10 m (up), world 1 0.1 (down), world
        # 2 fell (down), world 3 commanded too little to be judged (kept).
        gpu.terrain_path.copy_(torch.tensor([[10., 9.], [10., 1.], [10., 9.], [1., 0.]], device=gpu.device))
        gpu.qpos[2, 3:7] = torch.tensor([.70710678, .70710678, 0., 0.], device=gpu.device)  # world 2 on its side
        _, _, terminated, _, _ = gpu.step(torch.zeros((4, 29), device=gpu.device))  # step 11: world 2 falls
        assert terminated.tolist() == [False, False, True, False]
        _, _, _, truncated, _ = gpu.step(torch.zeros((4, 29), device=gpu.device))  # step 12: the others end
        gpu.check_health()
        assert truncated.tolist() == [True, True, False, True]
        levels = gpu.terrain_level.tolist()
        assert levels[1:] == [8, 8, 6] and 0 <= levels[0] < 10  # world 0 past the top: a random level
        centres = torch.as_tensor(g1_terrain.tile_centres(), device=gpu.device, dtype=torch.float32)
        respawned = centres[gpu.terrain_level, gpu.terrain_type]
        torch.testing.assert_close(gpu.qpos[[0, 1, 3], :2], respawned[[0, 1, 3]])
        torch.testing.assert_close(gpu.qpos[2, :2], respawned[2], atol=.05, rtol=0)  # one step after its respawn
        assert bool((gpu.terrain_path[[0, 1, 3]] == 0).all())
        state = gpu.training_state()
        assert state['terrain_level'] == gpu.terrain_level.tolist() and state['terrain_type'] == kinds
        metrics = gpu.rollout_metrics()
        assert metrics['rollout/terrain_level'] == pytest.approx(float(gpu.terrain_level.float().mean()))
        assert metrics['rollout/terrain_share_stairs_up'] == .25 and metrics['rollout/terrain_drop'] > 0
        other = TensorG1Env(8, seed=5, observation_noise=False, randomize=False, **profile)
        try:
            other.restore_training_state(state)  # another world count keeps each type's levels
            for kind, level in zip(kinds, state['terrain_level']):
                worlds = other.terrain_type == kind
                if bool(worlds.any()):
                    assert set(other.terrain_level[worlds].tolist()) == {level}
        finally:
            other.close()
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_terrain_seed(monkeypatch):
    """task_config terrain's seed on both backends: the CUDA task's heights are the CPU task's map of that seed (not
    the default map), four worlds start on the same tiles at the new map's platform heights and see the same ground
    terms over ten steps; a stage's terrain levels restore as on the default map."""
    import warp as wp
    from humanoid_sim.tasks import g1_terrain
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    profile = dict(real_robot=True, action_limit=10., sole_contact=True, lock_wrists=True, knee_floor=.061,
                   foot_collision=True, terrain={'seed': 6},
                   reward_weights={'feet_flat': 2., 'height': 10., 'clearance': 1., 'fall': 100.})
    tiles = (('stairs_up', 9), ('stairs_down', 9), ('rough', 9), ('boxes', 9))
    cpus = [G1VelocityEnv(**profile) for _ in tiles]
    gpu = TensorG1Env(4, seed=4, max_episode_steps=50, resample_commands=False, push_velocity=0,
                      observation_noise=False, randomize=False, **profile)
    try:
        heights = gpu.terrain_heights.cpu().double()
        torch.testing.assert_close(heights, cpus[0].terrain_heights, atol=1e-7, rtol=0)
        torch.testing.assert_close(heights, torch.from_numpy(g1_terrain.terrain_grid(6)), atol=1e-7, rtol=0)
        assert not np.allclose(heights.numpy(), g1_terrain.terrain_grid(), atol=1e-3)
        kinds = [g1_terrain.TERRAIN_TYPES.index(kind) for kind, _ in tiles]
        gpu.terrain_type.copy_(torch.tensor(kinds, device=gpu.device))
        gpu.set_terrain_level([level for _, level in tiles])
        gpu.reset()
        for world, (cpu, (kind, level)) in enumerate(zip(cpus, tiles)):
            cpu.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': kind, 'terrain_level': level})
            np.testing.assert_allclose(gpu.qpos[world, :3].cpu().numpy(), cpu.data.qpos[:3], atol=1e-5, rtol=0)
            assert float(gpu.pelvis_height()[world]) == pytest.approx(.8, abs=1e-5)
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.5, 0., 0.]], device=gpu.device).expand(4, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()
        for step in range(10):
            for cpu in cpus:
                cpu.step(np.zeros(29))
            gpu.step(torch.zeros((4, 29), device=gpu.device))
            gpu.check_health()
            ground = gpu._ground_terms()
            for world, cpu in enumerate(cpus):
                cpu_terms = cpu._ground_terms()
                np.testing.assert_allclose(float(ground['height'][world]), float(cpu_terms['height'][0]), atol=2e-3,
                                           rtol=0, err_msg=f'world {world} step {step}')
                np.testing.assert_allclose(ground['foot_height'][world].cpu().numpy(),
                                           cpu_terms['foot_height'][0].numpy(), atol=2e-3, rtol=0)
        state = {'terrain_level': [9, 9, 9, 9], 'terrain_type': kinds}
        other = TensorG1Env(8, seed=5, observation_noise=False, randomize=False, **profile)
        try:
            other.restore_training_state(state)
            for kind in kinds:
                worlds = other.terrain_type == kind
                if bool(worlds.any()):
                    assert set(other.terrain_level[worlds].tolist()) == {9}
        finally:
            other.close()
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_terrain_generator(monkeypatch):
    """task_config terrain's generator 3 on both backends: the CUDA task's heights are the CPU task's 2.5 cm grid of
    nine types (not generator 2's), four worlds on the top level's 20 cm stairs up and down, 15 cm curbs and 20 cm pits
    (box tiles: their heightfields collide with nothing, g1_terrain.box_tiles) start on the same tiles and see the
    same ground terms over ten steps, no ground contact (boxes or heightfield) deeper than 5 cm and no buffer overflow
    (plane's 128 / 512); the tile types are fixed by the generator's shares, and a generator-2
    stage carried here (pits and curbs in terrain_new_types) restores its levels with pits and curbs at 0."""
    import warp as wp
    from humanoid_sim.tasks import g1_terrain
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv, carried_terrain_state
    profile = dict(real_robot=True, action_limit=10., sole_contact=True, lock_wrists=True, knee_floor=.061,
                   foot_collision=True, terrain={'generator': 3},
                   reward_weights={'feet_flat': 2., 'height': 10., 'clearance': 1., 'fall': 100.})
    spec = g1_terrain.generator(3)
    tiles = (('stairs_up', 9), ('stairs_down', 9), ('curbs', 9), ('pits', 9))
    cpus = [G1VelocityEnv(**profile) for _ in tiles]
    gpu = TensorG1Env(4, seed=4, max_episode_steps=50, resample_commands=False, push_velocity=0,
                      observation_noise=False, randomize=False, **profile)
    try:
        heights = gpu.terrain_heights.cpu().double()
        # float32 heights up to ELEVATION_3 3 m: rounding up to ~3.6e-7
        torch.testing.assert_close(heights, cpus[0].terrain_heights, atol=4e-7, rtol=0)
        torch.testing.assert_close(heights, torch.from_numpy(g1_terrain.terrain_grid(5, 3)), atol=4e-7, rtol=0)
        assert gpu.terrain_names == spec.types and len(gpu.tile_centres[0]) == 9
        model = gpu.reference.model
        boxes = g1_terrain.terrain_boxes(model)  # the four tiles collide through boxes, the pits' ramp tilted ones
        assert 1000 < len(boxes) < 2000 and sum(model.geom_quat[i][0] != 1. for i in boxes) == 12 * 10
        for kind, level in tiles:
            geom = model.geom(g1_terrain.tile_name(level, spec.types.index(kind)))
            assert geom.contype[0] == geom.conaffinity[0] == 0
        terrain = torch.zeros(model.ngeom, dtype=torch.bool, device=gpu.device)
        terrain[g1_terrain.ground_geoms(model)] = True
        geom, dist = gpu.wp.to_torch(gpu.data.contact.geom), gpu.wp.to_torch(gpu.data.contact.dist)
        nacon, slots = gpu.wp.to_torch(gpu.data.nacon), torch.arange(gpu.data.naconmax, device=gpu.device)
        deepest = torch.zeros((), device=gpu.device)
        original = gpu.wp.capture_launch

        def launch(graph):  # each physics substep's ground contacts (boxes and heightfields), read on the device
            nonlocal deepest
            original(graph)
            if graph is gpu.step_graph:
                live = (slots < nacon[0]) & terrain[geom.long()].any(dim=1)
                deepest = torch.minimum(deepest, torch.where(live, dist, 0.).min())

        monkeypatch.setattr(gpu.wp, 'capture_launch', launch)
        kinds = [spec.types.index(kind) for kind, _ in tiles]
        gpu.terrain_type.copy_(torch.tensor(kinds, device=gpu.device))
        gpu.set_terrain_level([level for _, level in tiles])
        gpu.reset()
        for world, (cpu, (kind, level)) in enumerate(zip(cpus, tiles)):
            cpu.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': kind, 'terrain_level': level})
            np.testing.assert_allclose(gpu.qpos[world, :3].cpu().numpy(), cpu.data.qpos[:3], atol=1e-5, rtol=0)
            assert float(gpu.pelvis_height()[world]) == pytest.approx(.8, abs=1e-5)
        gpu.qpos.copy_(torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32))
        gpu.qvel.zero_()
        gpu.command.copy_(torch.tensor([[.5, 0., 0.]], device=gpu.device).expand(4, -1))
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        gpu._obs()
        for step in range(10):
            for cpu in cpus:
                cpu.step(np.zeros(29))
            gpu.step(torch.zeros((4, 29), device=gpu.device))
            gpu.check_health()
            ground = gpu._ground_terms()
            for world, cpu in enumerate(cpus):
                cpu_terms = cpu._ground_terms()
                np.testing.assert_allclose(float(ground['height'][world]), float(cpu_terms['height'][0]), atol=2e-3,
                                           rtol=0, err_msg=f'world {world} step {step}')
                np.testing.assert_allclose(ground['foot_height'][world].cpu().numpy(),
                                           cpu_terms['foot_height'][0].numpy(), atol=2e-3, rtol=0)
        assert float(deepest) > -.05, float(deepest)
        metrics = gpu.rollout_metrics()
        assert 'rollout/terrain_level_pits' in metrics and 'rollout/terrain_level_curbs' in metrics
        old = {'terrain_level': [9, 9, 4, 4], 'terrain_type': [3, 4, 0, 1]}  # generator 2's numbering
        state = carried_terrain_state(old, g1_terrain.TERRAIN_TYPES, spec.types)
        other = TensorG1Env(64, seed=5, observation_noise=False, randomize=False, **profile)
        try:
            from humanoid_sim.tasks import g1_tensor_math as task
            assert other.terrain_type.cpu().tolist() == task.terrain_types(64, spec.shares).tolist()  # by world index
            other.restore_training_state(state)
            for kind, level in ((3, 9), (4, 9), (0, 4), (1, 4), (7, 0), (8, 0)):
                worlds = other.terrain_type == kind
                assert bool(worlds.any()) and set(other.terrain_level[worlds].tolist()) == {level}, kind
        finally:
            other.close()
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_terrain_pelvis_support():
    """task_config terrain pelvis_support on both backends: two worlds standing in home on a flat tile, the reward's
    ground (terrain_heights, not the physics) raised 0.25 m under the pelvis and ahead of it, between the feet
    (tests/test_g1_terrain_pelvis_support.py). Both measure the pelvis 0.78 m above the feet's mean ground (above the
    ground under it 0.53, below the fall height), in the reward, the critic's last column and the fall: no world falls
    over ten steps, and the CPU task agrees."""
    import mujoco
    import warp as wp
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    profile = dict(real_robot=True, critic_extras=True, action_limit=10., sole_contact=True, lock_wrists=True,
                   knee_floor=.061, foot_collision=True, terrain={'pelvis_support': True},
                   reward_weights={'feet_flat': 2., 'height': 10., 'fall': 100.})
    cpu = G1VelocityEnv(**profile)
    gpu = TensorG1Env(2, seed=4, max_episode_steps=50, resample_commands=False, push_velocity=0,
                      observation_noise=False, randomize=False, **profile)
    try:
        cpu.reset(seed=0, options={'command': [0., 0., 0.], 'terrain_type': 'flat', 'terrain_level': 0})
        cpu.data.qpos[cpu.qadr] = cpu.home
        mujoco.mj_kinematics(cpu.model, cpu.data)
        cpu.data.qpos[2] += -.0005 - float(cpu._sole_height().min())
        mujoco.mj_forward(cpu.model, cpu.data)
        cell, (left, bottom) = cpu.terrain_cell, cpu.terrain_corner
        rows, columns = cpu.terrain_heights.shape
        x = left + cell * torch.arange(columns, dtype=torch.float64)
        y = bottom + cell * torch.arange(rows, dtype=torch.float64)
        step = ((x[None, :] >= cpu.data.qpos[0] - cell) & ((y[:, None] - cpu.data.qpos[1]).abs() <= .06)).double()
        cpu.terrain_heights = cpu.terrain_heights + .25 * step
        gpu.terrain_heights = gpu.terrain_heights + .25 * step.to(gpu.terrain_heights)
        gpu.terrain_type.zero_()  # flat
        gpu.set_terrain_level(0)
        gpu.reset()
        gpu.qpos.copy_(torch.tensor(cpu.data.qpos, device=gpu.device, dtype=torch.float32).expand(2, -1))
        gpu.qvel.zero_()
        gpu.command.zero_()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(gpu.device))):
            wp.capture_launch(gpu.forward_graph)
        observation = gpu._obs()
        point = gpu.qpos[:, 2] - task.pelvis_ground(gpu.qpos[:, :2], gpu.xpos[:, gpu.feet, :2], gpu.terrain_heights,
                                                    cpu.terrain_corner, cpu.terrain_cell, cpu.terrain_lift)
        assert point.tolist() == pytest.approx([.53, .53], abs=.02)  # what the old rule saw: a fall
        torch.testing.assert_close(observation[:, -1], gpu.pelvis_height())
        assert gpu.pelvis_height().tolist() == pytest.approx([cpu.pelvis_height()] * 2, abs=1e-4)
        assert cpu.pelvis_height() == pytest.approx(.78, abs=.02)
        for _ in range(10):
            _, _, cpu_fell, _, info = cpu.step(np.zeros(29))
            observation, _, terminated, _, _ = gpu.step(torch.zeros((2, 29), device=gpu.device))
            gpu.check_health()
            assert not cpu_fell and not bool(terminated.any())
            torch.testing.assert_close(observation[:, -1], gpu.pelvis_height())
            np.testing.assert_allclose(gpu.pelvis_height().cpu().numpy(), info['pelvis_height'], atol=2e-3, rtol=0)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_cpu_gpu_height_scan():
    """task_config height_scan on both backends: four worlds on the top stairs up and down, rough and boxes tiles, off
    the platform and turned, each against the CPU task in the same state. Without the training noise the actor's scan
    (columns 490-654) and the critic's (659-883) are the CPU task's over ten steps, and the layout, the observation size
    and the fixed normalization are the reference's; with the noise (observation_noise on) only the actor's scan moves,
    by at most noise, per world and point, the critic's stays exact."""
    import mujoco
    import warp as wp
    from humanoid_sim.tasks import g1_terrain, g1_tensor_math as task
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    profile = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.,
                   sole_contact=True, lock_wrists=True, knee_floor=.061, foot_collision=True,
                   terrain={'pelvis_support': True}, height_scan={},
                   reward_weights={'feet_flat': 2., 'height': 10., 'fall': 100.})
    tiles = (('stairs_up', 9), ('stairs_down', 9), ('rough', 9), ('boxes', 9))
    cpus = [G1VelocityEnv(**profile) for _ in tiles]
    gpu = TensorG1Env(4, seed=4, max_episode_steps=50, resample_commands=False, push_velocity=0,
                      observation_noise=False, randomize=False, **profile)
    noisy = TensorG1Env(4, seed=5, max_episode_steps=50, resample_commands=False, push_velocity=0,
                        observation_noise=True, randomize=False, **profile)
    actor, critic = slice(490, 655), slice(659, 884)
    try:
        assert gpu.observation_space.shape == (884,) and gpu.reference.actor_inputs == 655
        positions, mean, variance = gpu.fixed_normalization
        assert positions.tolist() == list(range(490, 655)) + list(range(659, 884))
        kinds = [g1_terrain.TERRAIN_TYPES.index(kind) for kind, _ in tiles]
        for env in (gpu, noisy):
            env.terrain_type.copy_(torch.tensor(kinds, device=env.device))
            env.set_terrain_level([level for _, level in tiles])
            env.reset()
        for world, (cpu, (kind, level)) in enumerate(zip(cpus, tiles)):
            cpu.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': kind, 'terrain_level': level})
            cpu.data.qpos[0] += 1.4
            yaw = .3 * (world - 1.5)
            cpu.data.qpos[3:7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
            cpu.data.qpos[2] += float(cpu._ground(cpu.data.qpos[:2])) - float(cpu._ground(cpu.data.qpos[:2] - [1.4, 0]))
            mujoco.mj_forward(cpu.model, cpu.data)
            cpu._observation('fill')
        qpos = torch.tensor(np.stack([cpu.data.qpos for cpu in cpus]), device=gpu.device, dtype=torch.float32)
        for env in (gpu, noisy):
            env.qpos.copy_(qpos)
            env.qvel.zero_()
            env.command.copy_(torch.tensor([[.5, 0., 0.]], device=env.device).expand(4, -1))
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device))):
                wp.capture_launch(env.forward_graph)
        observation = gpu._obs('fill')
        for step in range(10):
            expected = np.stack([cpu._observation() for cpu in cpus])
            got = observation.cpu().numpy()
            if step == 0:  # the same state: CPU and CUDA alike
                np.testing.assert_allclose(got[:, actor], expected[:, actor], atol=5e-3, rtol=0)
                np.testing.assert_allclose(got[:, critic], expected[:, critic], atol=5e-3, rtol=0)
            # later the two physics part a little, and near a 20 cm riser a millimetre of pelvis moves a point by a
            # centimetre: each backend's scan must then be the formula of its own state
            own = task.height_scan(gpu.qpos[:, :3], gpu.xmat[:, gpu.pelvis].reshape(-1, 3, 3), gpu.scan_grid,
                                   gpu.terrain_heights, cpus[0].terrain_corner, cpus[0].terrain_cell,
                                   cpus[0].terrain_lift).cpu().numpy()
            np.testing.assert_allclose(got[:, actor], own, atol=1e-5, rtol=0, err_msg=f'step {step}')
            assert np.abs(expected[:, actor]).max() > .02  # the steps, the blocks and the rough ground are seen
            for cpu in cpus:
                cpu.step(np.zeros(29))
            observation = gpu.step(torch.zeros((4, 29), device=gpu.device))[0]
            gpu.check_health()
        first = noisy._obs()
        exact = task.height_scan(noisy.qpos[:, :3], noisy.xmat[:, noisy.pelvis].reshape(-1, 3, 3), noisy.scan_grid,
                                 noisy.terrain_heights, cpus[0].terrain_corner, cpus[0].terrain_cell,
                                 cpus[0].terrain_lift)
        noise = first[:, actor] - exact
        assert float(noise.abs().max()) <= .1 + 1e-5 and float(noise.abs().max()) > .05
        assert float((noise[0] - noise[1]).abs().max()) > .05  # per world
        second = noisy._obs()
        assert float((second[:, actor] - first[:, actor]).abs().max()) > .05  # fresh every observation
        torch.testing.assert_close(second[:, critic], first[:, critic], rtol=0, atol=0)  # the critic's: exact
    finally:
        for cpu in cpus:
            cpu.close()
        gpu.close()
        noisy.close()


@pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')
def test_g1_gpu_terrain_candidate_walks(monkeypatch):
    """task_config terrain with a real policy: candidate 8 (exp-28691193718e4d0e @ 7221510144, trained on the plane)
    walks 300 steps deterministically in 64 worlds of every tile type at level 0, with its task's training resets,
    command resampling and randomization. terrain1's one heightfield for the whole grid gave MJWarp sole-sphere
    contacts 1.4-4 m deep (g1_terrain's docstring) and ~1100 falls in 300 steps over 256 worlds, within ~10 steps of
    a reset, flat tiles too; zero actions for ten steps (test_g1_cpu_gpu_terrain) never showed it. Now at most one
    flat world and four worlds in all fall (Warp is not deterministic here: 0-1 a run, a flat one only where the
    robot had walked 4 m to the grid's outer edge), and no heightfield contact in any substep is deeper than 5 cm (with this
    randomization the soles press up to 2.1 cm into the plane, and Warp measures a heightfield's depth vertically, up to
    3.2 cm at a stair edge). Skipped without the lab's stage (runs/, untracked)."""
    from pathlib import Path
    from humanoid_sim.core import read_json
    from humanoid_sim.export_g1 import load_policy
    from humanoid_sim.tasks import g1_terrain
    stage = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-28691193718e4d0e/stage_007221510144'
    if not stage.exists():
        pytest.skip('candidate 8 stage is missing')
    policy, norm, _ = load_policy(stage)
    config = {**read_json(stage / 'metadata.json')['task_config'], 'terrain': {}}
    gpu = TensorG1Env(64, seed=3, **config)
    try:
        terrain = torch.zeros(gpu.reference.model.ngeom, dtype=torch.bool, device=gpu.device)
        terrain[g1_terrain.terrain_geoms(gpu.reference.model)] = True
        geom, dist = gpu.wp.to_torch(gpu.data.contact.geom), gpu.wp.to_torch(gpu.data.contact.dist)
        nacon, slots = gpu.wp.to_torch(gpu.data.nacon), torch.arange(gpu.data.naconmax, device=gpu.device)
        deepest = torch.zeros((), device=gpu.device)
        original = gpu.wp.capture_launch

        def launch(graph):  # each physics substep's heightfield contacts, read on the device
            nonlocal deepest
            original(graph)
            if graph is gpu.step_graph:
                live = (slots < nacon[0]) & terrain[geom.long()].any(dim=1)
                deepest = torch.minimum(deepest, torch.where(live, dist, 0.).min())

        monkeypatch.setattr(gpu.wp, 'capture_launch', launch)
        obs = gpu.reset()
        kinds = gpu.terrain_type.clone()
        assert set(kinds.tolist()) == set(range(len(g1_terrain.TERRAIN_TYPES)))
        falls = torch.zeros(64, dtype=torch.long, device=gpu.device)
        for _ in range(300):
            action = policy.predict(norm.normalize_obs(obs.cpu().numpy()), deterministic=True)[0]
            obs, _, terminated, _, _ = gpu.step(torch.as_tensor(action, device=gpu.device, dtype=torch.float32))
            falls += terminated
        gpu.check_health()
        assert int(falls[kinds == 0].sum()) <= 1, falls.tolist()
        assert int(falls.sum()) <= 4, falls.tolist()
        assert float(deepest) > -.05, float(deepest)
    finally:
        gpu.close()