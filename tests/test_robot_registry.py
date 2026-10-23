import pytest
from humanoid_sim.robots.registry import RobotSpec, RobotRegistry
from humanoid_sim.tasks.registry import TaskSpec, TaskRegistry


def test_fixed_base_arm_has_no_implicit_feet_or_joint_count():
    robots = RobotRegistry()
    arm = RobotSpec('arm', 'Arm', 'fixed', ('shoulder',), ('motor',), frozenset({'gripper'}))
    robots.register(arm)
    assert robots.get('arm') == arm
    assert robots.get('arm').base_type == 'fixed'
    assert 'feet' not in arm.capabilities
    with pytest.raises(ValueError):
        robots.register(arm)
    with pytest.raises(ValueError):
        robots.get('../unknown')


def test_task_and_backend_compatibility_checked_before_factory_call():
    arm = RobotSpec('arm', 'Arm', 'fixed', ('joint',), ('motor',), frozenset({'gripper'}))
    tasks = TaskRegistry()
    calls = []
    tasks.register(TaskSpec('reach', frozenset({'gripper'}), frozenset({'cpu'}),
                            lambda robot, config: calls.append((robot, config)) or 'env'))
    tasks.register(TaskSpec('walk', frozenset({'feet'}), frozenset({'cpu'}), lambda *_: None))
    assert [t.task_id for t in tasks.compatible(arm)] == ['reach']
    with pytest.raises(ValueError, match='capabilities'):
        tasks.make('walk', arm, 'cpu', {})
    with pytest.raises(ValueError, match='backend'):
        tasks.make('reach', arm, 'full_gpu', {})
    assert not calls
    assert tasks.make('reach', arm, 'cpu', {'seed': 1}) == 'env'
    assert calls == [(arm, {'seed': 1})]


def test_duplicate_joint_names_rejected():
    with pytest.raises(ValueError):
        RobotSpec('bad', 'Bad', 'floating', ('j', 'j'), ('m',), frozenset())
