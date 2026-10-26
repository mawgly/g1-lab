import copyreg
import os
from importlib.metadata import version
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .core import RunLock, list_stages, save_stage, evaluate_stage, write_json, read_json
from .night import SleepGuard


class FrozenMoments(RunningMeanStd):
    """SB3 observation statistics whose fixed columns stay put through every update, as TensorNormalize's fixed= on the
    full-GPU path. fixed: (positions, mean, variance), a task's fixed_normalization; a NaN mean keeps the column's mean
    as `rms` has it (G1 arm_blind). Pickled as a plain RunningMeanStd, so a saved normalize.pkl is the ordinary one."""

    def __init__(self, rms, fixed):
        super().__init__(shape=rms.mean.shape)
        self.mean, self.var, self.count = rms.mean.copy(), rms.var.copy(), rms.count
        positions, mean, variance = fixed
        self.positions = np.asarray(positions, dtype=np.int64)
        mean = np.asarray(mean, dtype=np.float64)
        self.fixed_mean = np.where(np.isnan(mean), self.mean[self.positions], mean)
        self.fixed_var = np.asarray(variance, dtype=np.float64)
        self._freeze()

    def _freeze(self):
        self.mean[self.positions] = self.fixed_mean
        self.var[self.positions] = self.fixed_var

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        super().update_from_moments(batch_mean, batch_var, batch_count)
        self._freeze()

    def __reduce_ex__(self, protocol):
        # copyreg's own rebuild of a plain object (__newobj__ insists on the instance's class): no reference to this
        # module in the pickle.
        return copyreg._reconstructor, (RunningMeanStd, object, None), {
            'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}


def freeze_normalization(env, task_id, robot_id, task_config):
    """Keeps the task's fixed_normalization columns (G1 height_command / upper_body_slot terms, arm_blind's arm
    columns) frozen in the CPU path's VecNormalize `env`, as full_gpu_train does with TensorNormalize; nothing for a
    task without them. The columns are read from one environment built here (the vectorized ones may be
    subprocesses)."""
    from .tasks.catalog import make_task
    probe = make_task(task_id, robot_id, task_config)
    try:
        fixed = getattr(probe.unwrapped, 'fixed_normalization', None)
    finally:
        probe.close()
    if fixed is not None:
        env.obs_rms = FrozenMoments(env.obs_rms, fixed)


def keeps_training_state(env):
    """Whether a vectorized environment's tasks keep a training state (G1GetupEnv.keeps_training_state). has_attr
    first: a subprocess environment dies on get_attr of a missing attribute; the batched Humanoid one answers None."""
    return env.has_attr('keeps_training_state') and all(env.get_attr('keeps_training_state'))


def vec_training_state(env):
    """The environments' training state (G1GetupEnv.training_state: g1_getup's assist forces), merged world by world
    into {key: [one value per environment]}; None for tasks without one."""
    if not keeps_training_state(env):
        return None
    merged = {}
    for state in env.env_method('training_state'):
        for key, values in state.items():
            merged.setdefault(key, []).extend(values)
    return merged


def restore_vec_training_state(env, stage):
    """A resumed run's environments take the training state the stage saved (save_stage's training_state), spread
    over however many there are now (each environment's restore_training_state); nothing without one."""
    state = read_json(Path(stage) / 'metadata.json', {}).get('training_state')
    if state and keeps_training_state(env):
        for index in range(env.num_envs):
            env.env_method('restore_training_state', state, index, env.num_envs, indices=[index])


class StageCallback(BaseCallback):
    def __init__(self, run_dir, every, target, duration_seconds=None, parent_stop=None, training_state=None,
                 extras=None):
        super().__init__()
        # training_state(): the environments' state each stage keeps (save_stage), None: nothing (most tasks).
        # extras(folder): more files each stage keeps (save_stage; full GPU g1_getup with amp: the discriminator).
        self.training_state = training_state
        self.extras = extras
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
        stage = save_stage(self.run_dir, self.model, self.training_env,
                           None if self.training_state is None else self.training_state(), extras=self.extras)
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
                restore_vec_training_state(env, latest)  # g1_getup: the worlds' assist forces
            else:
                env = VecNormalize(base, norm_obs=True, norm_reward=True, clip_obs=10, gamma=parameters['gamma'])
                policy, policy_kwargs = 'MlpPolicy', {'net_arch': [256, 256], 'log_std_init': -1}
                if base.has_attr('asymmetric_critic') and all(base.get_attr('asymmetric_critic')):
                    # g1_getup: the critic's extras (head height, assist force) are not the robot's to see
                    from .policies import CriticExtrasPolicy
                    policy = CriticExtrasPolicy
                    policy_kwargs['actor_inputs'] = base.get_attr('actor_inputs')[0]
                model = PPO(policy, env, **parameters, policy_kwargs=policy_kwargs, device='cpu', seed=args.seed,
                            verbose=1)
            if not (config and config['backend'] == 'gpu'):  # the gpu backend's env is the Humanoid's Warp one
                freeze_normalization(env, task_id, robot_id, (config or {}).get('task_config') or {})
            session_dir = run / f'session_{time.time_ns()}'
            model.set_logger(configure(str(session_dir), ['stdout', 'csv']))
            callback = StageCallback(run, args.every, model.num_timesteps + args.steps, duration,
                                     getattr(args, 'parent_stop', None), training_state=lambda: vec_training_state(env))
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
