"""Measure complete CUDA PPO iterations, excluding checkpoint evaluation."""
import argparse
import gc
import time
from pathlib import Path
from importlib.metadata import version

import torch

from .core import write_json
from .full_gpu_train import make_learner
from .lab import DEFAULTS, PRESETS


def hybrid_local(config, iterations):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from .gpu_env import create_gpu_env
    started = time.perf_counter()
    env = VecNormalize(create_gpu_env(config['envs'], config['seed']), gamma=config['gamma'])
    try:
        parameters = {key: config[key] for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs',
                      'gamma', 'gae_lambda', 'ent_coef', 'clip_range', 'target_kl')}
        model = PPO('MlpPolicy', env, **parameters, seed=config['seed'], device='cpu',
                    policy_kwargs={'net_arch': [256, 256], 'log_std_init': -1})
        model.learn(8192)
        setup = time.perf_counter() - started
        started = time.perf_counter()
        model.learn(8192 * iterations, reset_num_timesteps=False)
        elapsed = time.perf_counter() - started
        return {'training_steps_per_second': 8192 * iterations / elapsed,
                'training_seconds': elapsed, 'setup_seconds': setup}
    finally:
        env.close()


def benchmark(sizes=(256, 512, 1024), iterations=8, output=None):
    report = {'backend': 'full_gpu', 'device': torch.cuda.get_device_name(),
              'created_at': time.time(),
              'versions': {name: version(name) for name in ('torch', 'mujoco', 'mujoco-warp', 'stable-baselines3')},
              'includes': 'physics, observation/reward, normalization, rollout and PPO updates',
              'excludes': 'checkpoint export, CPU evaluation and browser rendering',
              'rollout_size': 8192, 'preset': 'careful', 'seed': 42,
              'parameters': {**PRESETS['careful'], 'batch_size': 256, 'gamma': .99, 'gae_lambda': .95, 'clip_range': .2},
              'comparison_note': 'Same work budget; CUDA and NumPy reset RNGs produce different trajectories. Short throughput test, not a learning-quality comparison.',
              'results': []}
    for count in sizes:
        if count not in (256, 512, 1024):
            raise ValueError('Benchmark supports 256, 512 or 1024 worlds')
        config = {**DEFAULTS, **PRESETS['careful'], 'backend': 'full_gpu', 'envs': count,
                  'n_steps': 8192 // count, 'preset': 'careful'}
        torch.set_num_threads(2)
        # Both backends now share one interpreter, so the hybrid baseline runs in-process.
        hybrid = {**hybrid_local(config, iterations), 'torch_version': torch.__version__}
        learner = export = None
        try:
            started = time.perf_counter()
            learner, export = make_learner(config)
            learner.collect()
            learner.update()
            torch.cuda.synchronize()
            setup = time.perf_counter() - started
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(iterations):
                learner.collect()
                learner.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {'n_envs': count, 'n_steps': config['n_steps'], 'iterations': iterations,
                   'setup_seconds': setup, 'training_seconds': elapsed,
                   'training_steps_per_second': 8192 * iterations / elapsed,
                   'hybrid_steps_per_second': hybrid['training_steps_per_second'],
                   'speedup_vs_hybrid': (8192 * iterations / elapsed) / hybrid['training_steps_per_second'],
                   'hybrid': hybrid,
                   'torch_peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2}
            report['results'].append(row)
            print(row, flush=True)
            if output:
                write_json(Path(output), report)
        finally:
            if learner is not None:
                learner.env.close()
            if export is not None:
                export.close()
            del learner, export
            gc.collect()
            torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='lab/full_gpu_benchmark.json')
    parser.add_argument('--iterations', type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    benchmark(iterations=args.iterations, output=args.output)
