from copy import deepcopy
import pytest
from humanoid_sim.evaluation.scenarios import get_suite
from humanoid_sim.evolution.selection import select_candidates
from humanoid_sim.checkpoint_schema import LEGACY


RULES=dict(max_fall_rate=.2, max_velocity_error=.6, min_duration_fraction=.8)


def result(error=.2, slip=.1, action=1., falls=0):
    suite=get_suite('screen')
    return dict(protocol=suite.version,split=suite.name,seeds=list(suite.seeds),seconds=suite.seconds,
        policy_contract=dict(LEGACY,robot_id='g1_29dof',task_id='g1_velocity'),
        scenarios={name:dict(robot_id='g1_29dof',task_id='g1_velocity',seconds=suite.seconds,
            episodes=[dict(seed=seed,fell=i<falls,duration_s=suite.seconds if i>=falls else 1.,
                           velocity_error_rmse_m_s=error,contact_slip_speed_m_s=slip,
                           action_rate_rms_s_inv=action) for i,seed in enumerate(suite.seeds)])
                   for name in suite.scenarios})


def test_selection_gates_failures_then_keeps_pareto_tradeoffs():
    candidates={'control':result(), 'worse':result(error=.3,slip=.2,action=2.),
                'accurate':result(error=.1,slip=.2), 'falling':result(error=.01,falls=3)}
    selected=select_candidates(candidates,RULES)
    assert set(selected['selected'])=={'control','accurate'}
    assert selected['decisions']['worse']['state']=='rejected'
    assert 'доминирует' in selected['decisions']['worse']['reason']
    assert 'падений' in selected['decisions']['falling']['reason']


@pytest.mark.parametrize('change', ['final','seed','scenario','nan','missing','duration'])
def test_incomparable_or_incomplete_evidence_is_rejected(change):
    candidate=result()
    if change=='final':candidate['split']='final'
    if change=='seed':candidate['scenarios']['walk']['episodes'][0]['seed']+=1
    if change=='scenario':candidate['scenarios'].pop('stand')
    if change=='nan':candidate['scenarios']['walk']['episodes'][0]['velocity_error_rmse_m_s']=float('nan')
    if change=='missing':candidate['scenarios']['walk']['episodes'][0]['contact_slip_speed_m_s']=None
    if change=='duration':candidate['seconds']=10
    with pytest.raises(ValueError):
        select_candidates({'bad':candidate},RULES)


def test_results_are_not_mutated_and_rules_required():
    candidates={'control':result()};before=deepcopy(candidates)
    select_candidates(candidates,RULES)
    assert candidates==before
    with pytest.raises(ValueError):select_candidates(candidates,{})


def test_different_physical_model_contracts_are_not_ranked_together():
    first=result();second=result()
    second['policy_contract']['model_revision']='changed-robot'
    with pytest.raises(ValueError,match='contract'):
        select_candidates({'a':first,'b':second},RULES)
