import json
import os
import time
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs' / 'humanoid'
# One CUDA environment serves both GPU backends (hybrid PPO simply runs on its CPU kernels).
GPU_ENVIRONMENT = '.venv-fullgpu'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    try:
        # A reader holding the target open blocks os.replace on Windows (sharing violation); readers come and go,
        # so retry for up to ~5 s (1 s was not enough once, 22.10.2026: a G1 trainer died on status.json).
        for attempt in range(100):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 99:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


class RunLock:
    """Windows byte-range lock: released by the OS even if a trainer crashes."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'trainer.lock'
        self.file = None

    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.file.close()
            raise RuntimeError('Training is already running in this folder. Open humanoid_watch.cmd.') from exc
        return self

    def __exit__(self, *_):
        import msvcrt
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()


def list_stages(run_dir):
    return sorted(p for p in Path(run_dir).glob('stage_*')
                  if p.is_dir() and all((p / name).is_file() for name in ('policy.zip', 'normalize.pkl', 'metadata.json')))


def save_stage(run_dir, model, normalizer):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f'stage_{model.num_timesteps:012d}'
    if destination.exists():
        return destination
    pending = run_dir / f'.pending-{uuid.uuid4().hex}'
    pending.mkdir()
    model.save(pending / 'policy.zip')
    normalizer.save(pending / 'normalize.pkl')
    from .tasks.identity import task_identity
    config = read_json(run_dir / 'run_config.json', {})
    identity = task_identity(config.get('robot_id', 'humanoid'), config.get('task_id', 'humanoid_v5'),
                             config.get('task_config') or {})
    write_json(pending / 'metadata.json', {**identity, 'steps': model.num_timesteps,
                                          'task_config': config.get('task_config') or {},
                                          'environment': 'Humanoid-v5' if identity['robot_id'] == 'humanoid' else identity['task_id'],
                                          'created_at': time.time(), 'algorithm': 'PPO'})
    pending.rename(destination)
    return destination


def stage_task_config(stage):
    from .tasks.catalog import registries
    from .checkpoint_schema import checkpoint_identity
    metadata = read_json(Path(stage)/'metadata.json')
    identity = checkpoint_identity(metadata)
    robots, tasks = registries()
    task = tasks.resolve(identity['task_id'], robots.get(identity['robot_id']))
    return task.config_validator(metadata.get('task_config') or {})


def load_stage(stage):
    stage = Path(stage)
    from .checkpoint_schema import checkpoint_identity, validate_resume
    from .tasks.identity import task_identity
    from .tasks.catalog import make_task
    metadata = read_json(stage / 'metadata.json')
    identity = checkpoint_identity(metadata)
    validate_resume(identity, task_identity(identity['robot_id'], identity['task_id'], metadata.get('task_config') or {}))
    config = stage_task_config(stage)
    base = DummyVecEnv([lambda: make_task(identity['task_id'], identity['robot_id'], config)])
    try:
        norm = VecNormalize.load(stage / 'normalize.pkl', base)
        norm.training = False
        norm.norm_reward = False
        return PPO.load(stage / 'policy.zip', device='cpu'), norm
    except BaseException:
        base.close()
        raise


class StageSelection:
    def __init__(self):
        self.follow = True
        self.current = None

    def select(self, stages):
        if not stages:
            raise ValueError('No saved stages yet. Run humanoid_train.cmd first.')
        if self.follow or self.current not in stages:
            self.current = stages[-1]
        return self.current

    def move(self, offset, stages):
        self.follow = False
        current = self.select(stages)
        self.current = stages[max(0, min(len(stages) - 1, stages.index(current) + offset))]


def evaluate_stage(stage, episodes=5, max_steps=1000, seed_start=60000):
    if episodes < 1 or max_steps < 1:
        raise ValueError('Episode count and length must be positive')
    policy, norm = load_stage(stage)
    from .checkpoint_schema import checkpoint_identity
    from .tasks.catalog import make_task
    identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
    from .tasks.catalog import registries
    robots, tasks = registries()
    task = tasks.resolve(identity['task_id'], robots.get(identity['robot_id']))
    options = stage_task_config(stage)
    if task.episode_limit_key:
        options[task.episode_limit_key] = max_steps
    env = None
    records = []
    try:
        env = make_task(identity['task_id'], identity['robot_id'], options)
        for index in range(episodes):
            obs, initial = env.reset(seed=seed_start + index)
            total = 0.0
            for step in range(max_steps):
                action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            metrics = dict(forward_m=None, fell=None, distance_m=None, success=None)
            if task.evaluation_metrics is not None:
                metrics.update(task.evaluation_metrics(initial, info, terminated))
            records.append({**metrics, 'seed': seed_start + index, 'return': total,
                            'seconds': (step + 1) * identity['control_dt']})
    finally:
        if env is not None:
            env.close()
        norm.close()
    def mean_metric(key):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.mean(values)) if values else None

    return {'robot_id': identity['robot_id'], 'task_id': identity['task_id'],
            'mean_return': float(np.mean([r['return'] for r in records])),
            'mean_seconds': float(np.mean([r['seconds'] for r in records])),
            'mean_forward_m': mean_metric('forward_m'),
            'mean_distance_m': mean_metric('distance_m'),
            'success_rate': mean_metric('success'),
            'episodes': records}
