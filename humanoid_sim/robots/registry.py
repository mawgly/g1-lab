"""Small explicit registry; registration is code-owned, never browser imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    label: str
    base_type: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    capabilities: frozenset[str]

    def __post_init__(self):
        if not self.robot_id or self.base_type not in ('fixed', 'floating'):
            raise ValueError('Invalid robot identity or base_type')
        for names in (self.joint_names, self.actuator_names):
            if len(set(names)) != len(names) or any(not n for n in names):
                raise ValueError('Joint and actuator names must be unique and nonempty')


class RobotRegistry:
    def __init__(self):
        self._robots = {}

    def register(self, spec: RobotSpec) -> None:
        if spec.robot_id in self._robots:
            raise ValueError(f'Duplicate robot: {spec.robot_id}')
        self._robots[spec.robot_id] = spec

    def get(self, robot_id: str) -> RobotSpec:
        try:
            return self._robots[robot_id]
        except KeyError:
            raise ValueError(f'Unknown robot: {robot_id}') from None

    def all(self) -> tuple[RobotSpec, ...]:
        return tuple(self._robots.values())
