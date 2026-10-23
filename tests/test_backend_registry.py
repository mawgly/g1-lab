import pytest
from humanoid_sim.backends.registry import BackendRegistry


def test_dispatch_is_explicit_and_rejects_unknown_before_allocation():
    registry=BackendRegistry()
    calls=[]
    factory=lambda **kwargs: calls.append(kwargs) or 'environment'
    registry.register('custom','balance','full_gpu',factory)
    assert registry.make('custom','balance','full_gpu',n_envs=7,seed=42)=='environment'
    assert calls==[dict(n_envs=7,seed=42)]
    with pytest.raises(ValueError):
        registry.make('custom','walk','full_gpu',n_envs=7)
    with pytest.raises(ValueError):
        registry.register('custom','balance','full_gpu',factory)
    assert len(calls)==1
