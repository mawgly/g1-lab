import pytest
from humanoid_sim.evaluation.scenarios import get_suite, get_scenario


def test_splits_are_disjoint_and_protocol_is_immutable():
    suites=[get_suite(s) for s in ('screen','validation','final')]
    seeds=[set(s.seeds) for s in suites]
    assert not (seeds[0]&seeds[1] or seeds[1]&seeds[2] or seeds[0]&seeds[2])
    assert len(suites[1].seeds)==20 and suites[1].seconds==60
    assert len(suites[2].seeds)==30 and suites[2].seconds==60
    with pytest.raises((AttributeError,TypeError)):suites[2].seconds=1


def test_stop_start_schedule_and_unknown_scenario():
    s=get_scenario('stop_start')
    assert s.command_at(0)==(0.,0.,0.)
    assert s.command_at(5)==(1.,0.,0.)
    assert s.command_at(10)==(0.,0.,0.)
    assert s.command_at(15)==(1.,0.,0.)
    with pytest.raises(ValueError):get_scenario('invented')


def test_sprint_scenario_is_outside_the_versioned_suites():
    assert get_scenario('sprint').command_at(7)==(2.,0.,0.)
    for split in ('screen','validation','final'):
        assert get_suite(split).scenarios==('stand','slow','walk','brisk','stop_start')
