"""Explicit policy compatibility; never infer a new robot from tensor sizes."""

FIELDS = ('robot_id', 'task_id', 'model_revision', 'observation_schema',
          'action_schema', 'normalization_schema', 'control_dt')

LEGACY = dict(schema_version=1, robot_id='humanoid', task_id='humanoid_v5',
              model_revision='gymnasium-humanoid-v5-legacy',
              observation_schema='gymnasium-humanoid-v5-default',
              action_schema='gymnasium-humanoid-v5-17-torques',
              normalization_schema='sb3-vecnormalize-default', control_dt=0.015)


def checkpoint_identity(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint metadata must be an object')
    if 'schema_version' not in metadata:
        if metadata.get('environment') != 'Humanoid-v5':
            raise ValueError('Unknown legacy checkpoint environment')
        return dict(LEGACY)
    if type(metadata['schema_version']) is not int or metadata['schema_version'] != 1:
        raise ValueError('Unsupported checkpoint schema_version')
    for field in FIELDS:
        if field not in metadata or metadata[field] is None:
            raise ValueError(f'Missing checkpoint {field}')
    return {key: metadata[key] for key in ('schema_version', *FIELDS)}


def validate_resume(saved: dict, requested: dict) -> None:
    source, target = checkpoint_identity(saved), checkpoint_identity(requested)
    for field in FIELDS:
        if source[field] != target[field]:
            raise ValueError(f'Incompatible checkpoint {field}: {source[field]!r} != {target[field]!r}')
