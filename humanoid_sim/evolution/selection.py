"""G1 command-suite selection. Never optimize against final holdout episodes."""
import math
from statistics import mean
from ..evaluation.scenarios import get_suite
from ..checkpoint_schema import checkpoint_identity

QUALITY = ('velocity_error_rmse_m_s', 'contact_slip_speed_m_s', 'action_rate_rms_s_inv')


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError('Incomplete or invalid evaluation metric')
    return value


def _evidence(result):
    if result.get('split') not in ('screen','validation'):
        raise ValueError('Selection may not use final holdout results')
    suite=get_suite(result['split'])
    if (result.get('protocol')!=suite.version or result.get('seeds')!=list(suite.seeds)
            or result.get('seconds')!=suite.seconds or set(result.get('scenarios',{}))!=set(suite.scenarios)):
        raise ValueError('Evaluation protocol, seeds, scenarios or duration mismatch')
    scenario_metrics=[]
    for scenario in suite.scenarios:
        data=result['scenarios'][scenario]
        if (data.get('robot_id'),data.get('task_id'),data.get('seconds'))!=('g1_29dof','g1_velocity',suite.seconds):
            raise ValueError('Incompatible evaluation robot/task/duration')
        episodes=data.get('episodes',[])
        if [e.get('seed') for e in episodes]!=list(suite.seeds):
            raise ValueError('Incomplete or duplicated evaluation seeds')
        for episode in episodes:
            if type(episode.get('fell')) is not bool:
                raise ValueError('Missing fall outcome')
            if _number(episode.get('duration_s'))>suite.seconds+.021:
                raise ValueError('Episode exceeds evaluation budget')
            for metric in QUALITY:
                _number(episode.get(metric))
        scenario_metrics.append(dict(fall_rate=mean(e['fell'] for e in episodes),
            duration_fraction=mean(e['duration_s'] for e in episodes)/suite.seconds,
            **{metric:mean(e[metric] for e in episodes) for metric in QUALITY}))
    return dict(fall_rate=max(m['fall_rate'] for m in scenario_metrics),
                duration_fraction=min(m['duration_fraction'] for m in scenario_metrics),
                **{metric:max(m[metric] for m in scenario_metrics) for metric in QUALITY})


def validate_rules(rules: dict) -> dict:
    required={'max_fall_rate','max_velocity_error','min_duration_fraction'}
    if not isinstance(rules,dict) or set(rules)!=required:
        raise ValueError('Explicit selection thresholds must be fixed before a campaign')
    for key,value in rules.items():
        _number(value)
        if key!='max_velocity_error' and value>1:
            raise ValueError('Rate/fraction threshold must be in [0,1]')
    return dict(rules)


def select_candidates(results: dict, rules: dict) -> dict:
    rules=validate_rules(rules)
    if not results:
        raise ValueError('No candidate evaluations')
    if len({(r.get('protocol'),r.get('split')) for r in results.values()})!=1:
        raise ValueError('Candidates must use identical evaluation protocols')
    contracts=[checkpoint_identity(r.get('policy_contract')) for r in results.values()]
    if any(c!=contracts[0] for c in contracts) or (contracts[0]['robot_id'],contracts[0]['task_id'])!=('g1_29dof','g1_velocity'):
        raise ValueError('Candidates have incompatible policy contracts')
    scores={identity:_evidence(result) for identity,result in results.items()}
    decisions={}; eligible=[]
    for identity,metrics in scores.items():
        reasons=[]
        if metrics['fall_rate']>rules['max_fall_rate']:reasons.append('слишком высокая доля падений')
        if metrics['duration_fraction']<rules['min_duration_fraction']:reasons.append('слишком короткие попытки')
        if metrics[QUALITY[0]]>rules['max_velocity_error']:reasons.append('слишком большая ошибка скорости')
        decisions[identity]=dict(state='rejected' if reasons else 'eligible',reason='; '.join(reasons),metrics=metrics)
        if not reasons:eligible.append(identity)
    # Fall rate remains an objective after gating: smoothness cannot dominate
    # a more stable policy unless all objectives are at least as good.
    objectives=('fall_rate',*QUALITY)
    for identity in eligible:
        dominators=[other for other in eligible if other!=identity
            and all(scores[other][key]<=scores[identity][key] for key in objectives)
            and any(scores[other][key]<scores[identity][key] for key in objectives)]
        decisions[identity].update(state='rejected' if dominators else 'selected',
            reason=('Другой вариант доминирует по всем критериям: '+', '.join(sorted(dominators))) if dominators
                   else 'Прошёл пороги; сохранён как недоминируемый вариант')
    return dict(selected=sorted(k for k,v in decisions.items() if v['state']=='selected'),
                decisions=decisions,rules=dict(rules),
                limitation='Selection is provisional; no claim of statistical superiority or human likeness')
