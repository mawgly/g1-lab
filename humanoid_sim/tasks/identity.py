"""Versioned contracts for built-in tasks and asset-content fingerprints."""
import hashlib
from pathlib import Path
from ..checkpoint_schema import checkpoint_identity

def asset_identity(robot, *, directory: Path, task_id: str, control_dt: float, observation_version: int = 1) -> dict:
    """Preserve the original fingerprint for existing saved policies."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.xml', '.stl', '.obj', '.json'):
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
    return dict(schema_version=1, robot_id=robot.robot_id, task_id=task_id,
                model_revision=digest.hexdigest(), observation_schema=f'{task_id}:obs-v{observation_version}',
                action_schema=f'{task_id}:action-v1', normalization_schema='sb3-vecnormalize-default',
                control_dt=control_dt)


def task_identity(robot_id='humanoid', task_id='humanoid_v5', task_config=None) -> dict:
    """Checkpoint contract of a robot/task; task_config (a run's or a stage's) refines it where the task says so."""
    from .catalog import registries
    robots, tasks = registries()
    robot = robots.get(robot_id)
    task = tasks.resolve(task_id, robot)
    if task.identity_factory is None:
        raise ValueError(f'Task has no checkpoint contract: {task_id}')
    identity = checkpoint_identity(task.identity_factory(robot))
    if task.config_identity is not None:
        identity = checkpoint_identity(task.config_identity(identity, task.config_validator(task_config or {})))
    if identity['robot_id'] != robot_id or identity['task_id'] != task_id:
        raise ValueError('Adapter checkpoint identity does not match requested robot/task')
    return identity
