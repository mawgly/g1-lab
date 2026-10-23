"""Read-only checkpoint diagnostics; new evaluations never replace original results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from .core import list_stages, load_stage, read_json, write_json


def evaluate_new_seeds(stage, episodes=30, seed_start=80000):
    from .checkpoint_schema import checkpoint_identity
    identity=checkpoint_identity(read_json(Path(stage)/'metadata.json'))
    if identity['robot_id']!='humanoid':
        return evaluate_task_seeds(stage,episodes,seed_start)
    policy, norm = load_stage(stage)
    env = gym.make('Humanoid-v5')
    records = []
    def episode(seed):
        obs, initial = env.reset(seed=seed)
        total, sums, digest = 0.0, {}, hashlib.sha256()
        for step in range(env.spec.max_episode_steps):
            action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
            digest.update(np.asarray(action).tobytes())
            obs, reward, terminated, truncated, info = env.step(action)
            digest.update(np.asarray(obs).tobytes())
            total += float(reward)
            for key, value in info.items():
                if key.startswith('reward_'):
                    sums[key] = sums.get(key, 0.0) + float(value)
            if terminated or truncated:
                break
        return {'seed': seed, 'return': total, 'seconds': (step + 1) * env.unwrapped.dt,
                'forward_m': float(info['x_position'] - initial['x_position']), 'fell': bool(terminated),
                'truncated': bool(truncated), 'components': sums, 'trajectory_sha256': digest.hexdigest()}
    try:
        records = [episode(seed_start + i) for i in range(episodes)]
        replay = episode(seed_start)
        raw = env.unwrapped
        environment = {key: getattr(raw, key, None) for key in
                       ('dt', '_forward_reward_weight', '_ctrl_cost_weight', '_contact_cost_weight',
                        '_healthy_reward', '_healthy_z_range', '_terminate_when_unhealthy', '_reset_noise_scale')}
        environment['max_episode_steps'] = env.spec.max_episode_steps
        normalization = {key: getattr(norm, key, None) for key in
                         ('norm_obs', 'norm_reward', 'training', 'clip_obs', 'clip_reward', 'gamma', 'epsilon')}
        normalization['observation_count'] = float(norm.obs_rms.count)
        normalization['return_count'] = float(norm.ret_rms.count)
        return {'stage': Path(stage).name, 'episodes': records, 'episode_count': episodes,
                'mean_return': float(np.mean([r['return'] for r in records])),
                'std_return': float(np.std([r['return'] for r in records], ddof=1)) if episodes > 1 else None,
                'mean_seconds': float(np.mean([r['seconds'] for r in records])),
                'mean_forward_m': float(np.mean([r['forward_m'] for r in records])),
                'fall_rate': float(np.mean([r['fell'] for r in records])),
                'mean_components': {k: float(np.mean([r['components'][k] for r in records])) for k in records[0]['components']},
                'reward_components_max_abs_error': max(abs(r['return']-sum(r['components'].values())) for r in records),
                'settings_consistency': {'normalization_gamma_matches_ppo': norm.gamma == policy.gamma,
                    'observation_shape_matches': env.observation_space.shape == policy.observation_space.shape,
                    'action_shape_matches': env.action_space.shape == policy.action_space.shape,
                    'evaluation_normalization_frozen': not norm.training and not norm.norm_reward},
                'deterministic_replay_equal': replay == records[0], 'environment': environment,
                'evaluation_normalization': normalization,
                'actual_model': {'clip_range': float(policy.clip_range(1.0)),
                                 'clip_range_vf': float(policy.clip_range_vf(1.0)) if policy.clip_range_vf else None,
                                 'policy': str(policy.policy), 'optimizer': str(policy.policy.optimizer),
                                 'observation_shape': list(policy.observation_space.shape),
                                 'action_shape': list(policy.action_space.shape)},
                'policy_log_std': {'min': float(policy.policy.log_std.min().detach()),
                                   'max': float(policy.policy.log_std.max().detach())}}
    finally:
        env.close()
        norm.close()


def evaluate_task_seeds(stage,episodes,seed_start):
    from .core import evaluate_stage
    from .checkpoint_schema import checkpoint_identity
    identity=checkpoint_identity(read_json(Path(stage)/'metadata.json'))
    length=150 if identity['robot_id']=='so101' else 750
    result=evaluate_stage(stage,episodes,length,seed_start)
    replay=evaluate_stage(stage,1,length,seed_start)
    rows=result['episodes']
    result.update(stage=Path(stage).name,episode_count=episodes,
                  std_return=float(np.std([r['return'] for r in rows],ddof=1)) if episodes>1 else None,
                  fall_rate=None if identity['robot_id']=='so101' else float(np.mean([r['fell'] for r in rows])),
                  deterministic_replay_equal=rows[0]==replay['episodes'][0],
                  mean_components={},reward_components_max_abs_error=None,
                  environment={'robot_id':identity['robot_id'],'task_id':identity['task_id'],'dt':identity['control_dt'],
                               'max_episode_steps':length},
                  settings_consistency={'evaluation_normalization_frozen':True},
                  actual_model={},evaluation_normalization={'training':False,'norm_reward':False})
    return result


def run_diagnostics(run_dir, episodes=30):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    run_dir = Path(run_dir)
    stages = list_stages(run_dir)
    if not stages:
        raise ValueError('No complete checkpoints')
    evaluated = [(s, read_json(s / 'evaluation.json')) for s in stages]
    evaluated = [(s, v) for s, v in evaluated if v and v.get('mean_return') is not None]
    best = max(evaluated, key=lambda pair: pair[1]['mean_return'])[0] if evaluated else stages[-1]
    torch.set_num_threads(2)
    result = {'schema_version': 1, 'created_at': time.time(), 'seed_start': 80000,
              'selection': 'best selected by original fixed-seed mean return; fresh seeds shared across checkpoints',
              'versions': {n: importlib.metadata.version(n) for n in ('gymnasium', 'mujoco', 'stable-baselines3', 'torch', 'numpy')},
              'checkpoints': {}}
    cache = {}
    for role, stage in [('baseline', stages[0]), ('best', best), ('last', stages[-1])]:
        if stage not in cache:
            cache[stage] = evaluate_new_seeds(stage, episodes)
        result['checkpoints'][role] = cache[stage]
    write_json(run_dir / 'diagnostics.json', result)
    return result
