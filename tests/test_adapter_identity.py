from dataclasses import replace
import pytest
from humanoid_sim.tasks import catalog as catalog_module
from humanoid_sim.tasks.registry import TaskSpec, TaskRegistry
from humanoid_sim.robots.registry import RobotSpec, RobotRegistry
from humanoid_sim.tasks.identity import task_identity
from humanoid_sim.checkpoint_schema import LEGACY


def test_registered_adapter_resolves_identity_without_builtin_pair(monkeypatch):
    robots, tasks = RobotRegistry(), TaskRegistry()
    robot = RobotSpec('custom', 'Custom', 'fixed', ('hinge',), ('motor',), frozenset({'reach'}))
    robots.register(robot)
    identity = dict(LEGACY, robot_id='custom', task_id='reach', model_revision='custom-v1')
    tasks.register(TaskSpec('reach', frozenset({'reach'}), frozenset({'cpu'}),
                            lambda *_: None, identity_factory=lambda r: dict(identity)))
    monkeypatch.setattr(catalog_module, 'registries', lambda: (robots, tasks))
    assert task_identity('custom', 'reach') == identity
    from humanoid_sim.lab import validate_config
    assert validate_config({'robot_id': 'custom', 'task_id': 'reach'})['robot_id'] == 'custom'
    with pytest.raises(ValueError, match='backend'):
        validate_config({'robot_id': 'custom', 'task_id': 'reach', 'backend': 'full_gpu'})
    with pytest.raises(ValueError):
        task_identity('custom', 'unknown')
    robots.register(replace(robot, robot_id='other', capabilities=frozenset()))
    with pytest.raises(ValueError, match='capabilities'):
        task_identity('other', 'reach')


def test_legacy_identity_remains_exact():
    assert task_identity() == LEGACY
