"""CUDA PPO training with checkpoint/export compatibility with the browser."""
import os
import time
from importlib.metadata import version

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .core import RunLock, list_stages, write_json
from .night import SleepGuard
from .tensor_ppo import BOUND, TensorNormalize, TensorPPO
from .train import StageCallback


def range_bounds(raw):
    """(low, high) per action for the range_bounds loss: the action where the motor target reaches its bounds, the
    joint's range (raw.target_low / target_high: with task_config knee_floor the knee's lower bound is the floor, so
    (0.061 − 0.30)/0.35 = −0.683 there instead of −1.106, else a knee mean in −0.68…−1.22 would move neither the
    floored target nor the loss). The loss starts at BOUND times these. A joint whose range lies beyond the
    ±action_limit box is bounded by the box instead, at ±action_limit/BOUND, so its loss starts right where the action
    is clipped: bounded at ±action_limit, its loss began only at 1.1·action_limit, and a mean between the two moved
    neither the clipped target (no reward gradient) nor the loss. The G1 left wrist pitch sat there at +10.3…+11.5 for
    400 M steps (target stuck at +40°, 22.10). Joints whose range lies inside the box keep their bound."""
    box = raw.action_limit / BOUND
    return tuple(np.clip((bound - raw.home) / raw.scale, -box, box) for bound in (raw.target_low, raw.target_high))


def make_learner(config, stage=None):
    from .backends.registry import backends
    from .tasks.catalog import make_task
    from .tasks.identity import task_identity
    from .checkpoint_schema import validate_resume
    from .core import read_json
    robot_id, task_id = config.get('robot_id', 'humanoid'), config.get('task_id', 'humanoid_v5')
    identity = task_identity(robot_id, task_id, config.get('task_config') or {})
    if stage:
        validate_resume(read_json(stage / 'metadata.json'), identity)
    if not torch.cuda.is_available():
        raise RuntimeError('Полный GPU требует CUDA PyTorch: запустите setup_fullgpu.cmd')
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    parameters = {k: config[k] for k in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                                       'gae_lambda', 'ent_coef', 'clip_range', 'target_kl')}
    # CPU Gym exists only for SB3 archive compatibility and stage evaluation.
    task_config = config.get('task_config') or {}
    base = DummyVecEnv([lambda: make_task(task_id, robot_id, task_config)])
    env = None
    try:
        export = VecNormalize.load(stage / 'normalize.pkl', base) if stage else VecNormalize(base, gamma=config['gamma'])
        export.training = export.norm_reward = True
        if stage:
            model = PPO.load(stage / 'policy.zip', env=export, device='cuda', **parameters)
            model.set_random_seed(config['seed'])
        else:
            policy, policy_kwargs = 'MlpPolicy', {'net_arch': [256, 256], 'log_std_init': -1}
            # Asymmetric actor-critic when the task's observation ends with critic-only inputs (G1 critic_extras,
            # g1_getup's six): the actor sees only the robot's own, the first actor_inputs.
            actor_inputs = getattr(base.envs[0].unwrapped, 'actor_inputs', None)
            if actor_inputs is not None and actor_inputs < base.observation_space.shape[0]:
                from .policies import CriticExtrasPolicy
                policy = CriticExtrasPolicy
                policy_kwargs['actor_inputs'] = actor_inputs
            model = PPO(policy, export, **parameters, device='cuda', seed=config['seed'], policy_kwargs=policy_kwargs)
        # PPO.load restores optimizer state, including its previous learning rate.
        for group in model.policy.optimizer.param_groups:
            group['lr'] = config['learning_rate']
        env = backends().make(robot_id, task_id, 'full_gpu', n_envs=config['envs'], seed=config['seed'], **task_config)
        # What the stage kept of the training beyond the weights (save_stage's training_state: g1_getup's assist
        # forces), back into the worlds before their first reset (TensorPPO's), as train.restore_vec_training_state.
        saved = (read_json(stage / 'metadata.json', {}) or {}).get('training_state') if stage else None
        if saved and hasattr(env, 'restore_training_state'):
            env.restore_training_state(saved)
        # Columns with set statistics (G1 height_command / upper_body_slot terms, g1_getup's command and phase) stay
        # frozen.
        norm = TensorNormalize(env.observation_space.shape, config['envs'], env.device, config['gamma'],
                               fixed=getattr(base.envs[0].unwrapped, 'fixed_normalization', None))
        if stage:
            norm.import_normalizer(export)
        model.n_envs = config['envs']
        mirror = None
        if config.get('mirror_coef'):  # PPO mirror loss: left–right symmetric policy (symmetry.py)
            from .symmetry import Mirror
            mirror = Mirror(base.envs[0])
        bounds = range_bounds(base.envs[0].unwrapped) if config.get('range_bounds') else None
        learner = TensorPPO(model, env, config, norm, mirror=mirror, bounds=bounds)
        return learner, export
    except BaseException:
        base.close()
        if env is not None:
            env.close()
        raise


def train_full_gpu(args, config):
    run = args.run_dir
    duration = int(config['hours'] * 3600) if config['hours'] else getattr(args, 'duration_seconds', None)
    learner = export = callback = None
    failure = None
    state = 'failed'
    with RunLock(run), SleepGuard(enabled=duration is not None):
        (run / 'stop.request').unlink(missing_ok=True)
        write_json(run / 'status.json', {'state': 'starting', 'pid': os.getpid(), 'updated_at': time.time()})
        try:
            stages = list_stages(run)
            learner, export = make_learner(config, stages[-1] if stages else None)
            model = learner.model
            session = run / f'session_{time.time_ns()}'
            logger = configure(str(session), ['stdout', 'csv'])
            model.set_logger(logger)
            # training_state: the worlds' state each stage keeps (g1_getup: assist forces); None for other tasks.
            callback = StageCallback(run, config['every'], model.num_timesteps + config['steps'], duration,
                                     getattr(args, 'parent_stop', None),
                                     training_state=getattr(learner.env, 'training_state', None))
            callback.init_callback(model)
            callback.initial_steps = model.num_timesteps
            parameters = {key: config.get(key) for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs',
                          'gamma', 'gae_lambda', 'ent_coef', 'clip_range', 'target_kl', 'bounds_coef', 'max_std',
                          'mirror_coef', 'range_bounds')}
            write_json(session / 'config.json', {'environment': config.get('task_id', 'humanoid_v5'), 'algorithm': 'PPO',
                'robot_id': config.get('robot_id', 'humanoid'), 'task_id': config.get('task_id', 'humanoid_v5'),
                'task_config': config.get('task_config') or {},
                'seed': config['seed'], 'additional_steps': config['steps'], 'envs': config['envs'],
                'checkpoint_interval': config['every'], 'resume': str(stages[-1]) if stages else None,
                'parameters': parameters, 'backend': 'full_gpu', 'physics_variant': 'MuJoCo Warp Newton',
                'pipeline': 'CUDA physics, normalization, rollout buffers and PPO; CPU checkpoint evaluation',
                'versions': {name: version(name) for name in ('mujoco', 'gymnasium', 'stable-baselines3', 'numpy', 'torch')},
                'duration_seconds': duration, 'device': torch.cuda.get_device_name()})

            def snapshot():
                learner.normalizer.export_normalizer(export)
                callback.snapshot()

            snapshot()
            while model.num_timesteps < callback.target:
                if not callback._on_step():
                    state = 'stopped'
                    break
                started = time.perf_counter()
                metrics = learner.collect()
                metrics.update(learner.update())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                metrics['time/training_steps_per_second'] = config['envs'] * config['n_steps'] / elapsed
                metrics['time/total_timesteps'] = model.num_timesteps
                metrics['train/n_updates'] = model._n_updates
                metrics['train/learning_rate'] = config['learning_rate']
                for key, value in metrics.items():
                    logger.record(key, value)
                logger.dump(model.num_timesteps)
                if model.num_timesteps - callback.last_stage >= config['every']:
                    snapshot()
            else:
                state = 'completed'
        except KeyboardInterrupt:
            state = 'stopped'
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                if callback is not None:
                    try:
                        learner.normalizer.export_normalizer(export)
                        callback.snapshot()
                    except BaseException as save_error:
                        callback.status('failed', str(failure or save_error))
                        if failure is None:
                            raise
                    else:
                        callback.status(state, str(failure) if failure else None)
                else:
                    write_json(run / 'status.json', {'state': 'failed', 'pid': os.getpid(),
                               'updated_at': time.time(), 'error': str(failure)})
            finally:
                if learner is not None:
                    learner.env.close()
                if export is not None:
                    export.close()
