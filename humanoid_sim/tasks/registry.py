from dataclasses import dataclass
from typing import Callable, Any
from ..robots.registry import RobotSpec
from .playback import PlaybackSpec
from .config import empty_config


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    required_capabilities: frozenset[str]
    backends: frozenset[str]
    factory: Callable[[RobotSpec, dict], Any]
    identity_factory: Callable[[RobotSpec], dict] | None = None
    training_backends: frozenset[str] = frozenset({'cpu'})
    episode_limit_key: str | None = 'max_episode_steps'
    evaluation_metrics: Callable[[dict, dict, bool], dict] | None = None
    playback: PlaybackSpec = PlaybackSpec()
    config_validator: Callable[[dict], dict] = empty_config
    parameters: tuple = ()
    label: str = ''
    # Lab job kinds (lab_tasks.py) that make sense for this task.
    jobs: frozenset[str] = frozenset({'report', 'diagnostics'})
    # (identity, normalized task_config) -> identity, for settings that change the policy contract
    # (G1: the action limit). None: the contract does not depend on the task config.
    config_identity: Callable[[dict, dict], dict] | None = None
    # Reset options of core.evaluate_stage's episodes (g1_getup: no assist force); None: a plain reset.
    evaluation_options: dict | None = None


class TaskRegistry:
    def __init__(self):
        self._tasks = {}

    def register(self, spec: TaskSpec) -> None:
        if not spec.task_id or spec.task_id in self._tasks:
            raise ValueError(f'Invalid or duplicate task: {spec.task_id}')
        self._tasks[spec.task_id] = spec

    def compatible(self, robot: RobotSpec) -> tuple[TaskSpec, ...]:
        return tuple(t for t in self._tasks.values()
                     if t.required_capabilities <= robot.capabilities)

    def resolve(self, task_id: str, robot: RobotSpec) -> TaskSpec:
        if not isinstance(task_id, str) or task_id not in self._tasks:
            raise ValueError(f'Unknown task: {task_id}')
        task = self._tasks[task_id]
        if not task.required_capabilities <= robot.capabilities:
            raise ValueError(f'Incompatible robot_id {robot.robot_id!r}: missing task capabilities {task.required_capabilities - robot.capabilities}')
        return task

    def make(self, task_id: str, robot: RobotSpec, backend: str, config: dict):
        task = self.resolve(task_id, robot)
        if backend not in task.backends:
            raise ValueError(f'Unsupported task backend: {backend}')
        return task.factory(robot, dict(config))
