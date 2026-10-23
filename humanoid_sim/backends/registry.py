from functools import lru_cache


class BackendRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, robot_id, task_id, backend, factory):
        key = (robot_id, task_id, backend)
        if key in self._factories or not all(isinstance(x,str) and x for x in key) or not callable(factory):
            raise ValueError('Invalid or duplicate backend adapter')
        self._factories[key] = factory

    def make(self, robot_id, task_id, backend, **config):
        key = (robot_id, task_id, backend)
        if key not in self._factories:
            raise ValueError(f'No backend adapter for {key}')
        return self._factories[key](**config)


def _humanoid_tensor(**config):
    from ..tensor_env import TensorHumanoidEnv
    return TensorHumanoidEnv(**config)


def _g1_tensor(**config):
    from ..g1_tensor_env import TensorG1Env
    return TensorG1Env(**config)


def _g1_legs_tensor(**config):
    return _g1_tensor(legs_only=True, **config)


@lru_cache(maxsize=1)
def backends():
    registry = BackendRegistry()
    registry.register('humanoid', 'humanoid_v5', 'full_gpu', _humanoid_tensor)
    # Implementation registered for explicit testing; public training_backends
    # remains CPU-only until the CUDA correctness/benchmark gates pass.
    registry.register('g1_29dof', 'g1_velocity', 'full_gpu', _g1_tensor)
    registry.register('g1_12dof', 'g1_velocity', 'full_gpu', _g1_legs_tensor)
    return registry
