import pytest
from humanoid_sim.evaluation.statistics import summarize


def test_zero_falls_does_not_claim_zero_population_risk():
    s=summarize([{'fell':False,'speed':1.},{'fell':False,'speed':3.}])
    assert s['speed']['mean']==2
    assert s['speed']['std']==pytest.approx(2**.5)
    assert s['fell']['wilson95'][0]==pytest.approx(0)
    assert s['fell']['wilson95'][1]>.5


def test_missing_metrics_are_not_imputed_as_zero():
    s=summarize([{'length':None},{'length':.3}])
    assert s['length']['count']==1 and s['length']['mean']==.3
    assert s['length']['std'] is None
    assert summarize([{'length':None}])['length']['mean'] is None
