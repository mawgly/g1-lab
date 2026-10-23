import os
from importlib.metadata import version
import time
from pathlib import Path
from functools import partial

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .core import RunLock, list_stages, save_stage, evaluate_stage, write_json, read_json
from .night import SleepGuard


class StageCallback(BaseCallback):
    def __init__(self, run_dir, every, target, duration_seconds=None, parent_stop=None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every = every
        self.target = target
        self.last_stage = -every
        self.last_status = 0.0
        self.start = time.monotonic()
        self.initial_steps = 0
        self.duration_seconds = duration_seconds
        self.timed_out = False
        self.parent_stop = parent_stop

    def status(self, state, error=None):
        elapsed = time.monotonic() - self.start
        try:
            self._write_status(state, error, elapsed)
        except PermissionError as failure:  # status.json is advisory (locks say what runs): never stop training
            print(f'status.json not updated ({state}): {failure}', flush=True)

    def _write_status(self, state, error, elapsed):
        write_json(self.run_dir / 'status.json', {
            'state': state, 'pid': os.getpid(), 'steps': self.model.num_timesteps,
            'target_steps': self.target, 'elapsed_seconds': elapsed,
            'steps_per_second': (self.model.num_timesteps - self.initial_steps) / max(elapsed, 0.001),
            'updated_at': time.time(), 'error': error,
            'night_mode': self.duration_seconds is not None,
            'remaining_seconds': max(0, self.duration_seconds - elapsed) if self.duration_seconds else None,
            'sleep_protected': self.duration_seconds is not None and state in ('training', 'saving'),
            'stop_reason': 'timer' if self.timed_out else None})

    def snapshot(self):
        self.status('saving')
        stage = save_stage(self.run_dir, self.model, self.training_env)
        if not (stage / 'evaluation.json').exists():
            metrics = evaluate_stage(stage)
            write_json(stage / 'evaluation.json', metrics)
            print(f'Stage {self.model.num_timesteps:,}: return={metrics["mean_return"]:.1f}, '
                  f'duration={metrics["mean_seconds"]:.2f}s, forward={metrics["mean_forward_m"]}m', flush=True)
        self.last_stage = self.model.num_timesteps
        try:
            from .reports import write_report
            write_report(self.run_dir)
        except Exception as exc:
            print(f'Report refresh failed (checkpoint is saved): {exc}', flush=True)
        self.status('training')

    def _on_training_start(self):
        self.initial_steps = self.model.num_timesteps
        self.snapshot()

    def _on_rollout_start(self):
        if self.model.num_timesteps - self.last_stage >= self.every:
            self.snapshot()

    def _on_step(self):
        if time.monotonic() - self.last_status > 2:
            self.status('training')
            self.last_status = time.monotonic()
        self.timed_out = self.duration_seconds is not None and time.monotonic() - self.start >= self.duration_seconds
        return not self.timed_out and not (self.run_dir / 'stop.request').exists() and not (self.parent_stop and self.parent_stop.exists())


def train(args):
    torch.set_num_threads(2)
    run = args.run_dir
    duration = getattr(args, 'duration_seconds', None)
    config = None
    if getattr(args, 'config', None):
        from .lab import validate_config
        config = validate_config(read_json(args.config))
        args.envs, args.seed, args.steps, args.every = config['envs'], config['seed'], config['steps'], config['every']
        duration = int(config['hours'] * 3600) if config['hours'] else None
    if config and config['backend'] == 'full_gpu':
        from .full_gpu_train import train_full_gpu
        return train_full_gpu(args, config)
    with RunLock(run), SleepGuard(enabled=duration is not None):
        (run / 'stop.request').unlink(missing_ok=True)
        stages = list_stages(run)
        base = env = callback = model = None
        state = 'failed'
        failure = None
        write_json(run / 'status.json', {'state': 'starting', 'pid': os.getpid(), 'updated_at': time.time()})
        try:
            from .tasks.catalog import make_task
            from .tasks.identity import task_identity
            from .checkpoint_schema import validate_resume
            robot_id = config.get('robot_id', 'humanoid') if config else 'humanoid'
            task_id = config.get('task_id', 'humanoid_v5') if config else 'humanoid_v5'
            if stages:
                validate_resume(read_json(stages[-1] / 'metadata.json'),
                                task_identity(robot_id, task_id, (config or {}).get('task_config') or {}))
            if config and config['backend'] == 'gpu':
                from .gpu_env import create_gpu_env
                base = create_gpu_env(args.envs, args.seed)
            else:
                base = make_vec_env(partial(make_task, task_id, robot_id, (config or {}).get('task_config') or {}), n_envs=args.envs, seed=args.seed,
                                    vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
            parameters = dict(learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
                              gamma=.99, gae_lambda=.95, ent_coef=0, clip_range=.2, target_kl=None)
            if config:
                parameters.update({k: config[k] for k in parameters})
            if stages:
                latest = stages[-1]
                env = VecNormalize.load(latest / 'normalize.pkl', base)
                if env.gamma != parameters['gamma']:
                    from stable_baselines3.common.running_mean_std import RunningMeanStd
                    env.ret_rms = RunningMeanStd(shape=())
                env.gamma = parameters['gamma']
                env.training = True
                env.norm_reward = True
                model = PPO.load(latest / 'policy.zip', env=env, device='cpu', **parameters)
                model.set_random_seed(args.seed)
            else:
                env = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10, gamma=parameters['gamma'])
                model = PPO('MlpPolicy', env, **parameters,
                            policy_kwargs={'net_arch': [256, 256], 'log_std_init': -1},
                            device='cpu', seed=args.seed, verbose=1)
            session_dir = run / f'session_{time.time_ns()}'
            model.set_logger(configure(str(session_dir), ['stdout', 'csv']))
            callback = StageCallback(run, args.every, model.num_timesteps + args.steps, duration, getattr(args, 'parent_stop', None))
            write_json(session_dir / 'config.json', {'environment': 'Humanoid-v5' if robot_id == 'humanoid' else task_id,
                       'robot_id': robot_id, 'task_id': task_id, 'algorithm': 'PPO',
                       'task_config': (config or {}).get('task_config') or {},
                       'seed': args.seed, 'additional_steps': args.steps, 'envs': args.envs,
                       'checkpoint_interval': args.every, 'resume': str(stages[-1]) if stages else None,
                       'parameters': parameters, 'backend': config['backend'] if config else 'cpu',
                       'physics_variant': 'MuJoCo Warp Newton' if config and config['backend'] == 'gpu' else 'Gym MuJoCo PGS',
                       'versions': {name: version(name) for name in ('mujoco', 'gymnasium', 'stable-baselines3', 'numpy', 'torch')},
                       'duration_seconds': duration})
            model.learn(args.steps, callback=callback, log_interval=10, reset_num_timesteps=not bool(stages))
            state = 'stopped' if callback.timed_out or (run / 'stop.request').exists() or (callback.parent_stop and callback.parent_stop.exists()) else 'completed'
        except KeyboardInterrupt:
            state = 'stopped'
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                if model is not None and env is not None and callback is not None and hasattr(callback, 'model'):
                    try:
                        callback.snapshot()
                    except BaseException as save_error:
                        try:
                            callback.status('failed', error=str(failure or save_error))
                        except OSError:
                            pass  # Disk failure may also prevent publishing status.
                        if failure is None:
                            raise
                    else:
                        callback.status(state, error=str(failure) if failure else None)
                else:
                    write_json(run / 'status.json', {'state': 'failed', 'pid': os.getpid(),
                                                     'updated_at': time.time(), 'error': str(failure)})
            finally:
                if env is not None:
                    env.close()
                elif base is not None:
                    base.close()
        print(f'Training {state}. Saved stages: {run}', flush=True)
