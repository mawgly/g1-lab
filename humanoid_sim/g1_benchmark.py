"""Measure complete full-GPU PPO iterations for G1 (physics, rollout, updates).

Excludes checkpoint export, CPU evaluation and browser rendering. Run in .venv-fullgpu
while no other training uses the GPU:
    .venv-fullgpu\\Scripts\\python.exe -m humanoid_sim.g1_benchmark
"""
import argparse
import gc
import time
from importlib.metadata import version

import torch

from .core import write_json
from .full_gpu_train import make_learner
from .lab import DEFAULTS, PRESETS

# (envs, n_steps, batch_size)
CONFIGS = ((64, 128, 256), (256, 32, 256), (256, 32, 2048), (1024, 32, 8192), (2048, 32, 16384), (4096, 24, 24576))


def measure(envs, n_steps, batch_size, window_seconds):
    # Not validate_config: the benchmark must run before full_gpu is public for G1.
    config = {**DEFAULTS, **PRESETS['careful'], 'robot_id': 'g1_29dof', 'task_id': 'g1_velocity',
              'backend': 'full_gpu', 'envs': envs, 'n_steps': n_steps, 'batch_size': batch_size}
    learner = export = None
    try:
        started = time.perf_counter()
        learner, export = make_learner(config)
        learner.collect()
        learner.update()
        torch.cuda.synchronize()
        setup = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        iterations, collect_seconds, started = 0, 0., time.perf_counter()
        while time.perf_counter() - started < window_seconds:
            begin = time.perf_counter()
            learner.collect()
            torch.cuda.synchronize()
            collect_seconds += time.perf_counter() - begin
            learner.update()
            torch.cuda.synchronize()
            iterations += 1
        elapsed = time.perf_counter() - started
        transitions = envs * n_steps * iterations
        return dict(n_envs=envs, n_steps=n_steps, batch_size=batch_size, iterations=iterations,
                    setup_seconds=setup, training_seconds=elapsed,
                    training_steps_per_second=transitions / elapsed,
                    rollout_share=collect_seconds / elapsed,
                    torch_peak_memory_mb=torch.cuda.max_memory_allocated() / 1024**2)
    finally:
        if learner is not None:
            learner.env.close()
        if export is not None:
            export.close()
        del learner, export
        gc.collect()
        torch.cuda.empty_cache()


def benchmark(configs=CONFIGS, window_seconds=20., output=None):
    report = {'backend': 'full_gpu', 'robot_id': 'g1_29dof', 'task_id': 'g1_velocity',
              'device': torch.cuda.get_device_name(), 'created_at': time.time(),
              'versions': {name: version(name) for name in ('torch', 'mujoco', 'mujoco-warp', 'stable-baselines3')},
              'includes': 'physics, observation/reward, normalization, rollout and PPO updates',
              'excludes': 'checkpoint export, CPU evaluation and browser rendering',
              'preset': 'careful', 'window_seconds': window_seconds, 'results': []}
    for envs, n_steps, batch_size in configs:
        row = measure(envs, n_steps, batch_size, window_seconds)
        report['results'].append(row)
        print(row, flush=True)
        if output:
            write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='lab/g1_full_gpu_benchmark.json')
    parser.add_argument('--window', type=float, default=20.)
    args = parser.parse_args()
    benchmark(window_seconds=args.window, output=args.output)
