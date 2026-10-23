import importlib.util
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('mujoco_warp') is None, reason='isolated GPU dependencies required')

def test_gpu_matches_gym_short_trajectory_and_autoresets():
    from humanoid_sim.gpu_env import create_gpu_env
    import gymnasium as gym
    cpu = gym.make('Humanoid-v5')
    import mujoco
    cpu.unwrapped.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    gpu = create_gpu_env(2, 42, max_episode_steps=3)
    try:
        expected, _ = cpu.reset(seed=42)
        actual = gpu.reset()
        np.testing.assert_allclose(actual[0], expected, atol=2e-4)
        for i in range(3):
            action = np.full(17, 0.03, dtype=np.float32)
            expected, rew, _, _, _ = cpu.step(action)
            actual, rewards, dones, infos = gpu.step(np.tile(action, (2, 1)))
            compared = infos[0]['terminal_observation'] if dones[0] else actual[0]
            np.testing.assert_allclose(compared, expected, atol=0.04, rtol=0.01)
            np.testing.assert_allclose(rewards[0], rew, atol=0.02)
        assert dones.all()
        assert infos[0]['TimeLimit.truncated']
        assert infos[0]['episode']['l'] == 3
        assert actual[0, 0] > 1.3
    finally:
        gpu.close()
        cpu.close()


def test_gpu_contacts_and_selective_reset_preserve_other_worlds():
    # Detects stale postconstraint forces, wrong reward timing, and cross-world resets.
    from humanoid_sim.gpu_env import create_gpu_env
    import gymnasium as gym
    import mujoco
    env = create_gpu_env(2, 42)
    cpu = [gym.make('Humanoid-v5') for _ in range(2)]
    try:
        env.reset()
        for i, ref in enumerate(cpu):
            ref.unwrapped.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            ref.reset(seed=42 + i)
        rng = np.random.default_rng(1042)
        saw_contact = False
        saw_done = False
        for _ in range(40):
            actions = rng.uniform(-0.05, 0.05, (2, 17)).astype(np.float32)
            obs, rewards, dones, infos = env.step(actions)
            for i, ref in enumerate(cpu):
                expected, reward, terminated, _, _ = ref.step(actions[i])
                actual = infos[i].get('terminal_observation', obs[i])
                np.testing.assert_allclose(actual[:22], expected[:22], atol=0.005, rtol=0.005)
                assert abs(float(rewards[i]) - reward) < 0.01
                assert bool(dones[i]) == terminated
                saw_contact |= bool(ref.unwrapped.data.ncon)
                if terminated:
                    saw_done = True
                    ref.reset()
        assert saw_contact and saw_done
    finally:
        env.close()
        for ref in cpu:
            ref.close()



def test_reset_of_one_world_keeps_other_world_reward_history():
    from humanoid_sim.gpu_env import create_gpu_env
    env = create_gpu_env(2, 71)
    control = create_gpu_env(2, 71)
    try:
        env.reset()
        control.reset()
        actions = np.full((2, 17), 0.1, dtype=np.float32)
        env.lengths[0] = 999
        env.step(actions)
        control.step(actions)
        _, reward, _, _ = env.step(actions)
        _, expected, _, _ = control.step(actions)
        assert abs(float(reward[1]) - float(expected[1])) < 1e-5
    finally:
        env.close()
        control.close()

def test_gpu_policy_normalization_checkpoint_loads_on_cpu(tmp_path):
    from humanoid_sim.gpu_env import create_gpu_env
    from humanoid_sim.core import save_stage, load_stage
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import gymnasium as gym
    env = VecNormalize(create_gpu_env(4, 4))
    try:
        policy = PPO('MlpPolicy', env, n_steps=8, batch_size=32, n_epochs=1, device='cpu')
        policy.learn(32)
        stage = save_stage(tmp_path, policy, env)
        restored, norm = load_stage(stage)
        cpu = gym.make('Humanoid-v5')
        try:
            obs, _ = cpu.reset(seed=1)
            action, _ = restored.predict(norm.normalize_obs(obs), deterministic=True)
            assert action.shape == (17,)
            assert np.isfinite(cpu.step(action)[0]).all()
        finally:
            cpu.close()
            norm.close()
    finally:
        env.close()
