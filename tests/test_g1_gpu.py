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
