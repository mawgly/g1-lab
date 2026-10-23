import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA PyTorch runtime required')


def test_tensor_gpu_task_matches_hybrid_and_resets_on_device():
    from humanoid_sim.tensor_env import TensorHumanoidEnv
    if not torch.cuda.is_available():
        pytest.skip('CUDA PyTorch runtime required')
    from humanoid_sim.gpu_env import create_gpu_env
    import warp as wp
    env = TensorHumanoidEnv(2, 42, max_episode_steps=3)
    hybrid = create_gpu_env(2, 42, max_episode_steps=3)
    try:
        env.reset()
        hybrid.reset()
        # Match initial physics state; CUDA RNG is intentionally different.
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            env.qpos.copy_(torch.as_tensor(hybrid.data.qpos.numpy(), device='cuda'))
            env.qvel.copy_(torch.as_tensor(hybrid.data.qvel.numpy(), device='cuda'))
            wp.capture_launch(env.forward_graph)
            env._centers.copy_(env._center())
        for _ in range(3):
            actions = torch.full((2, 17), .03, device='cuda')
            obs, reward, terminated, truncated, terminal_obs = env.step(actions)
            expected, expected_reward, done, infos = hybrid.step(actions.cpu().numpy())
            for value in (obs, reward, terminated, truncated, terminal_obs):
                assert value.device.type == 'cuda'
            np.testing.assert_allclose(reward.cpu(), expected_reward, atol=.002)
            np.testing.assert_allclose(terminal_obs.cpu(), np.stack([x.get('terminal_observation', expected[i]) for i,x in enumerate(infos)]), atol=.02, rtol=.01)
        assert truncated.all().item()
        assert not terminated.any().item()
        assert (obs[:,0] > 1.3).all().item()
        env.check_health()
    finally:
        env.close()
        hybrid.close()


def test_tensor_step_uses_zero_copy_views_without_host_tensor_reads(monkeypatch):
    from humanoid_sim.tensor_env import TensorHumanoidEnv
    env = TensorHumanoidEnv(4, 12, max_episode_steps=2)
    try:
        env.reset()
        assert env.qpos.data_ptr() == env.data.qpos.ptr
        actions = torch.zeros((4, 17), device='cuda')
        def forbidden(*args, **kwargs):
            raise AssertionError('Per-step host tensor read is forbidden')
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'cpu', forbidden)
            patch.setattr(torch.Tensor, 'numpy', forbidden)
            patch.setattr(torch.Tensor, 'item', forbidden)
            patch.setattr(torch.Tensor, '__bool__', forbidden)
            env.step(actions)
            outputs = env.step(actions)
        assert outputs[3].all().item()
        assert all(value.is_cuda for value in outputs)
        env.check_health()
    finally:
        env.close()


def test_tensor_selective_reset_and_custom_stream_preserve_survivor():
    from humanoid_sim.tensor_env import TensorHumanoidEnv
    env = TensorHumanoidEnv(2, 19)
    control = TensorHumanoidEnv(2, 19)
    try:
        env.reset()
        control.reset()
        actions = torch.full((2, 17), .1, device='cuda')
        env.lengths[0] = 999
        env.step(actions)
        control.step(actions)
        # Switch stream with an explicit dependency, as a consumer must.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            obs, reward, _, _, _ = env.step(actions)
            expected, expected_reward, _, _, _ = control.step(actions)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(obs[1], expected[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reward[1], expected_reward[1], atol=1e-5, rtol=1e-5)
        assert env.lengths[0].item() == 1
        assert env.lengths[1].item() == 2
    finally:
        env.close()
        control.close()

def test_tensor_sustained_motion_matches_cpu_newton_reward():
    from humanoid_sim.tensor_env import TensorHumanoidEnv
    import gymnasium as gym
    import mujoco
    import warp as wp
    env = TensorHumanoidEnv(1, 42)
    cpu = gym.make('Humanoid-v5')
    cpu.unwrapped.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    try:
        env.reset()
        cpu.reset(seed=42)
        cpu.unwrapped.data.qvel[0] = 2.0
        mujoco.mj_forward(cpu.unwrapped.model, cpu.unwrapped.data)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            env.qpos.copy_(torch.as_tensor(cpu.unwrapped.data.qpos, dtype=torch.float32, device='cuda'))
            env.qvel.copy_(torch.as_tensor(cpu.unwrapped.data.qvel, dtype=torch.float32, device='cuda'))
            wp.capture_launch(env.forward_graph)
            env._centers.copy_(env._center())
        rng = np.random.default_rng(1042)
        steps = 0
        for _ in range(50):
            action = rng.uniform(-.05, .05, 17).astype(np.float32)
            _, reward, done, _, terminal_obs = env.step(torch.as_tensor(action[None], device='cuda'))
            expected, expected_reward, expected_done, _, _ = cpu.step(action)
            np.testing.assert_allclose(terminal_obs[0, :22].cpu(), expected[:22], atol=.005, rtol=.005)
            assert abs(reward[0].item() - expected_reward) < .01
            assert done[0].item() == expected_done
            steps += 1
            if expected_done:
                break
        assert steps >= 20
        env.check_health()
    finally:
        env.close()
        cpu.close()
