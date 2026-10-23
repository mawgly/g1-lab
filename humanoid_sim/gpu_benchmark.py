"""Measure real end-to-end VecEnv throughput, including host copies and resets."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import gymnasium as gym
import mujoco
import numpy as np
import warp as wp
from .gpu_env import create_gpu_env


def comparison(seed=42, steps=50):
    gpu = create_gpu_env(1, seed)
    cpu = gym.make('Humanoid-v5')
    original = gym.make('Humanoid-v5')
    cpu.unwrapped.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    try:
        obs = gpu.reset()[0]
        expected = cpu.reset(seed=seed)[0]
        original.reset(seed=seed)
        reset_error = float(np.max(np.abs(obs - expected)))
        obs_errors, reward_errors, pgs_errors = [], [], []
        rng = np.random.default_rng(seed + 1000)
        for _ in range(steps):
            action = rng.uniform(-0.05, 0.05, size=17).astype(np.float32)
            obs, rew, done, info = gpu.step(action[None])
            expected, expected_rew, terminated, _, _ = cpu.step(action)
            pgs, _, pgs_done, _, _ = original.step(action)
            actual = info[0].get('terminal_observation', obs[0])
            obs_errors.append(float(np.max(np.abs(actual - expected))))
            reward_errors.append(float(abs(rew[0] - expected_rew)))
            pgs_errors.append(float(np.max(np.abs(pgs - expected))))
            if done[0] or terminated or pgs_done:
                break
        return dict(seed=seed, steps=len(obs_errors), reset_max_abs_error=reset_error,
                    newton_obs_max_abs_error=max(obs_errors), newton_reward_max_abs_error=max(reward_errors),
                    cpu_pgs_vs_newton_obs_max_abs_error=max(pgs_errors))
    finally:
        gpu.close()
        cpu.close()
        original.close()


def benchmark(sizes=(128, 256, 512, 1024), warmup=30, steps=100, output=None):
    report = {'backend': 'mujoco-warp', 'solver': 'Newton (Gym default PGS unsupported)',
              'integrator': 'RK4', 'frame_skip': 5, 'includes': 'physics, observation/reward, CPU copies, automatic resets; excludes PPO',
              'versions': {p: importlib.metadata.version(p) for p in ('mujoco', 'mujoco-warp', 'warp-lang', 'gymnasium', 'stable-baselines3', 'torch')},
              'results': []}
    def save():
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            from .core import write_json
            write_json(path, report)
    report['correctness'] = comparison()
    from stable_baselines3.common.vec_env import DummyVecEnv
    cpu = DummyVecEnv([lambda: gym.make('Humanoid-v5') for _ in range(128)])
    cpu.seed(123)
    cpu.reset()
    action = np.zeros((128, 17), dtype=np.float32)
    for _ in range(warmup):
        cpu.step(action)
    started = time.perf_counter()
    for _ in range(steps):
        cpu.step(action)
    elapsed = time.perf_counter() - started
    report['cpu_baseline'] = dict(backend='Gymnasium DummyVecEnv serial CPU', solver='default PGS',
                                  n_envs=128, vector_steps=steps, elapsed_seconds=elapsed,
                                  environment_steps_per_second=128 * steps / elapsed)
    cpu.close()
    save()
    for n in sizes:
        started = time.perf_counter()
        env = create_gpu_env(n, 123)
        report['device'] = env.device.name
        env.reset()
        action = np.zeros((n, 17), dtype=np.float32)
        for _ in range(warmup):
            env.step(action)
        wp.synchronize()
        setup = time.perf_counter() - started
        started = time.perf_counter()
        episodes = 0
        for _ in range(steps):
            _, _, done, _ = env.step(action)
            episodes += int(done.sum())
        wp.synchronize()
        elapsed = time.perf_counter() - started
        row = dict(n_envs=n, vector_steps=steps, warmup_steps=warmup, elapsed_seconds=elapsed,
                   environment_steps_per_second=n * steps / elapsed, completed_episodes=episodes,
                   setup_and_warmup_seconds=setup)
        report['results'].append(row)
        print(json.dumps(row), flush=True)
        save()
        env.close()
        del env
        gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='lab/gpu_benchmark.json')
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.warmup < 1 or args.steps < 1:
        parser.error('Sizes, warmup and steps must be positive')
    benchmark(args.sizes, args.warmup, args.steps, args.output)


if __name__ == '__main__':
    main()
