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
