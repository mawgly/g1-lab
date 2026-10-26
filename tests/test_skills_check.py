"""scripts/lab/skills_check.py, the skills check (the user's request of 25.10): a row's verdict and its note, the
reference of a branch, the rows a policy cannot run, a short run on a small stage, the journal's skills card."""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from humanoid_sim import i18n, journal
from test_g1_arm_targets import ARMS, _stage

LAB = Path(__file__).resolve().parents[1] / 'scripts' / 'lab'


@pytest.fixture
def check(monkeypatch):
    """The module, the CUDA variable put back after the test (the probes hide CUDA on import)."""
    monkeypatch.syspath_prepend(str(LAB))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    return importlib.import_module('skills_check')


def test_a_row_passes_on_its_threshold_or_a_reference_that_misses_it_too_and_never_worse(check):
    verdict = check.verdict
    assert verdict(True, True, True, None, None) == (True, 'порог и эталон')
    assert verdict(True, None, None, None, 'н/п: нет команды высоты') == (True, 'порог; эталон н/п')
    assert verdict(False, False, True, None, None) == (True, 'порог не взят, как и у эталона')
    assert verdict(False, True, True, None, None) == (False, 'порог не взят')
    assert verdict(True, True, False, None, None) == (False, 'хуже эталона')
    assert verdict(False, True, False, None, None) == (False, 'порог не взят, хуже эталона')
    assert verdict(None, None, True, None, None) == (True, 'не хуже эталона')
    assert verdict(None, None, None, None, None) == (None, 'не судится')
    assert verdict(False, False, False, 'н/п: нет arm_targets', None) == (None, 'н/п: нет arm_targets')
    compared = check.compared
    lower = ('lower', .25, .01)
    assert compared(lower, dict(v=.04), dict(v=.03)) and not compared(lower, dict(v=.0401), dict(v=.03))
    assert compared(lower, dict(v=.25), dict(v=.2)) and not compared(lower, dict(v=.251), dict(v=.2))
    assert compared(lower, dict(v=float('inf')), dict(v=float('inf'))) and not compared(lower, dict(v=float('inf')),
                                                                                         dict(v=.03))
    assert compared(('higher', 0., .05), dict(v=.85), dict(v=.9)) and not compared(('higher', 0., .05), dict(v=.84),
                                                                                    dict(v=.9))
    assert compared(('falls',), dict(v=3, n=32), dict(v=1, n=32)) and not compared(('falls',), dict(v=4, n=32),
                                                                                    dict(v=1, n=32))
    assert compared(('near', 10., 1.), dict(v=11.4), dict(v=9.6)) and not compared(('near', 10., 1.), dict(v=11.5),
                                                                                    dict(v=9.6))
    assert compared(('rate',), dict(v=.05, lower=.02, upper=.1), dict(v=.01, lower=0., upper=.03))
    assert not compared(('rate',), dict(v=.08, lower=.04, upper=.15), dict(v=.01, lower=0., upper=.03))
    assert compared(lower, dict(v=None), dict(v=.03)) is None and compared(None, dict(v=1.), dict(v=1.)) is None


def test_the_reference_is_the_branch_parent_or_a_warm_start(check, tmp_path):
    lab = tmp_path / 'lab'
    parent = lab / 'exp-parent' / 'stage_000000001000'
    parent.mkdir(parents=True)
    child = lab / 'exp-child' / 'stage_000000002000'
    child.mkdir(parents=True)
    (child.parent / 'experiment.json').write_text(json.dumps(
        {'config': {'source_id': 'exp-parent', 'source_stage': 1000}}), encoding='utf-8')
    assert check.parent_of(child) == parent
    (child.parent / 'experiment.json').write_text(json.dumps(
        {'config': {}, 'warm_start': {'source_id': 'exp-parent', 'source_stage': 1000}}), encoding='utf-8')
    assert check.parent_of(child) == parent
    assert check.reference_path('n8', child) == check.N8
    (child.parent / 'experiment.json').write_text(json.dumps({'config': {}}), encoding='utf-8')
    with pytest.raises(SystemExit):
        check.parent_of(child)


def test_rows_a_policy_cannot_run_are_not_applicable_with_the_reason(check):
    plain = dict(path='x', export=False, height=False, arms=False, delay=0., label='x')
    export = dict(plain, export=True)
    walk = {'side:03': dict(v=.03), 'hoist': dict(v=1.2, da=.01), 'push:stand': dict(v=2, n=32)}
    theirs = {'side:03': dict(v=.02), 'hoist': dict(v=1., da=.01), 'push:stand': dict(v=0, n=32)}
    rows = {r['key']: r for r in check.rows(walk, theirs, plain, plain, False)}
    assert rows['crouch:walk']['ok'] is None and rows['crouch:walk']['note'] == 'н/п: нет команды высоты'
    assert rows['joints:roll0']['note'] == rows['falls:free']['note'] == 'н/п: нет arm_targets'
    assert rows['side:03']['ok'] is True and rows['side:03']['note'] == 'порог и эталон'
    assert rows['side:03']['value'] == '0,030' and rows['side:03']['reference'] == '0,020'
    assert rows['push:stand']['ok'] is False and rows['push:stand']['note'] == 'порог не взят'  # 2 of 32, ≤ 1
    assert rows['push:stand']['value'] == '2 из 32'
    assert rows['speed:1']['ok'] is False and rows['speed:1']['note'] == 'не измерено'
    armed = dict(plain, arms=True, height=True)
    exported = {r['key']: r for r in check.rows(walk, theirs, dict(armed, export=True), armed, True)}
    assert exported['falls:free']['ok'] is None and exported['falls:free']['note'] == 'н/п: проба не берёт выгрузку'
    assert exported['hoist']['ok'] is True and exported['joints:roll0']['note'] == 'не измерено'  # both run an export
    assert export['export']
    rows = {r['key']: r for r in check.rows({}, {}, armed, plain, True)}  # against a stage without them
    assert rows['joints:roll0']['reference'] == 'н/п' and rows['joints:roll0']['note'] == 'не измерено'


def test_unitree_v0_column_compares_its_walking_rows_and_is_cached_per_run(check, tmp_path, monkeypatch):
    """The third column: v0's value where it can run the row (walking, the bridge copy, chatter), 'н/п (у v0 нет)'
    on the arm, crouch, getting-up and hoist rows; better / same / worse by the row's own margin both ways; its runs
    are kept per job (key = sha256 of kind and job) and dropped when the v0 policy file changed."""
    plain = dict(path='x', export=False, height=False, arms=False, delay=0., label='x')
    mine = {'side:03': dict(v=.03), 'side:05': dict(v=.04), 'push:stand': dict(v=0, n=32), 'armp:falls': dict(v=0, n=9)}
    v0 = {'side:03': dict(v=.06), 'side:05': dict(v=.045), 'push:stand': dict(v=5, n=32)}
    rows = {r['key']: r for r in check.rows(mine, mine, plain, plain, False, v0=v0)}
    assert (rows['side:03']['v0'], rows['side:03']['v0_vs']) == ('0,060', 'better')  # 0.03 against 0.06
    assert rows['side:05']['v0_vs'] == 'same' and rows['push:stand']['v0_vs'] == 'better'
    assert rows['armp:falls']['v0'] == rows['hoist']['v0'] == 'н/п (у v0 нет)' and rows['armp:falls']['v0_vs'] is None
    assert rows['speed:1']['v0'] == '—' and rows['speed:1']['v0_vs'] is None  # neither measured
    assert check.versus(('lower', .25, .01), dict(v=.08), dict(v=.03)) == 'worse'
    assert check.versus_counts(list(rows.values())) == {'better': 2, 'same': 1, 'worse': 0}
    assert 'v0' not in check.rows(mine, mine, plain, plain, False)[0]  # without v0: no column
    jobs = check.v0_jobs([0., .02])
    assert all(tag.startswith(check.V0_TAGS + ('chatter:',)) for tag, _, _ in jobs)
    assert not any(tag.startswith(('armp:', 'hoist', 'joints:', 'falls:', 'crouch:')) for tag, _, _ in jobs)
    assert sum(tag.startswith('chatter:') for tag, _, _ in jobs) == 2 * 18
    assert len({check.job_key(kind, job) for _, kind, job in jobs}) == len(jobs)
    monkeypatch.setattr(check, 'V0_CACHE', tmp_path / 'skills_v0.json')
    monkeypatch.setattr(check, 'v0_policy_md5', lambda: 'a')
    check.save_v0({'k': {'fell': None, 'rate': float('nan')}})
    assert check.v0_cache()['k']['fell'] is None
    monkeypatch.setattr(check, 'v0_policy_md5', lambda: 'b')  # another v0 file: the runs are not its
    assert check.v0_cache() == {}
    monkeypatch.setattr(check, 'v0_policy_md5', lambda: None)  # WSL not running: the saved runs stand
    assert 'k' in check.v0_cache()
    root = tmp_path / 'lab'
    journal.set_skills(root, 'exp-1', 1, 'x', dict(branch='exp-0', stage=1, title='y'),
                       [dict(skill='a', label='b', ok=True, v0='0,060', v0_vs='better')])
    assert journal.load(root)['skills']['items'][0]['v0_vs'] == 'better'
    with pytest.raises(ValueError):
        journal.set_skills(root, 'exp-1', 1, 'x', dict(branch='exp-0', stage=1, title='y'),
                           [dict(skill='a', label='b', ok=True, v0='1', v0_vs='лучше')])


def test_goal_check_runs_a_policy_in_a_given_task_config(tmp_path):
    """supervise.check_stage and goal_check.walks with acting and config (how v0 runs goal_check's conditions): the
    stage gives the identity only, no σ."""
    from humanoid_sim.core import load_stage, stage_task_config
    from humanoid_sim.goal_check import walks
    from humanoid_sim.supervise import check_stage
    _, stage = _stage(tmp_path, ARMS)
    policy, norm = load_stage(stage)
    try:
        config = stage_task_config(stage)
        report = check_stage(stage, seconds=1., seeds=1, acting=(policy, norm), config=config)
        assert report['policy_std'] is None and set(report['scenarios']) == {'stand', 'walk', 'stop_start'}
        walk = walks(stage, seeds=1, seconds=1., acting=(policy, norm), turns=((0., .5),), straight=(), config=config)
        assert list(walk['turn']) == [(0., .5)]
    finally:
        norm.close()


def test_measures_of_falls_pushes_and_speeds(check):
    t = dict(path='x', export=False, height=False, arms=True, delay=0., label='x')
    got = {f'falls:free:{k}': dict(fell=k < 2, steps=500 if k < 2 else 1000) for k in range(10)}
    got.update({f'push:stand:a000:1:{k}': dict(fell=None if k else .5) for k in range(4)})
    got.update({f'speed:{c:g}:{s}': dict(fell=None, speed=.95 * c) for c in check.SPEEDS for s in check.SPEED_SEEDS})
    got['speed:2:73001'] = dict(fell=2.2, speed=None)
    measured = check.measures(got, t)
    free = measured['falls:free']
    assert free['falls'] == 2 and free['v'] == pytest.approx(2 / 9) and free['lower'] < free['v'] < free['upper']
    assert measured['push:stand'] == dict(v=1, n=4)
    assert measured['speed:1']['v'] == pytest.approx(.95) and measured['speed:-0.5']['v'] == pytest.approx(.95)
    assert measured['speed:2']['v'] == 0. and measured['speed:2']['falls'] == 1


def test_getting_up_rows_from_getup_probe_and_getup_stance_records(check):
    """A g1_getup stage's rows: ready38's success (up, no cutoff before up + 1 s, the hand-over without a fall) per
    physics and start class, the turn after the hand-over, the loads, the stance; judged against a reference."""
    def record(kind, physics, up=1.2, cut=None, fell=None, turn=4., seconds=4., wrist=0., impact=100.):
        joints = [0.] * 29
        joints[20] = wrist
        return dict(job={'class': kind, 'physics': physics}, up_s=up, success=up is not None and fell is None,
                    cut=cut, seconds=seconds, saturated_free=.01,
                    saturated_joint=joints, impact_n=impact, early_share=.3,
                    hand_over=None if up is None else dict(fell=fell, turn_deg=turn))
    got = {}
    for p in check.GETUP_PHYSICS:
        for k, kind in enumerate(('back', 'front', 'side', 'sit', 'fall')):
            got[f'ready:{p}:{k}'] = record(kind, p)
    got['ready:ours:0'] = record('back', 'ours', cut=[1.5, 3, 30.])  # a cutoff before up + 1 s: not ready
    got['ready:ours:1'] = record('front', 'ours', cut=[2.5, 3, 30.], turn=20.)  # after it: ready
    got['ready:delay20:2'] = record('side', 'delay20', fell='stand')
    got['ready:foot0.3:3'] = record('sit', 'foot0.3', up=None, impact=400.)
    stance = dict(up=1.2, widths={'0.1': .6, '1.0': .30, '2.0': .31, '4.0': .3}, pull=4., roll_error=[.01, -.01],
                  roll_torque=[1., -1.], arm_share=.05, arm_force=3., shoulder_roll=[.05, -.02], flight=.01,
                  touchdown=.8, peak=600., sole_speed=.9, sole_charge=.1)
    for k in range(5):
        got[f'stance:{k}'] = dict(stance, cls='back', seed=k)
    mine = check.getup_measures(got)
    assert mine['ready:ours'] == dict(v=1, n=5, ok=4) and mine['ready:delay20']['ok'] == 4
    assert mine['ready:back'] == dict(v=1, n=3, ok=2) and mine['ready:cut'] == dict(v=1, n=15)
    assert mine['handover:falls'] == dict(v=1, n=14) and mine['handover:turn15'] == dict(v=1, n=5)
    assert mine['handover:turn'] == dict(v=4.) and mine['loads:hits'] == dict(v=1, n=15)
    assert mine['ready:hold'] == dict(v=0, n=5, ok=5)  # getup_probe's own 3 s hold, apart from the hand-over
    assert mine['stance:width1']['v'] == pytest.approx(.30) and mine['stance:flights'] == dict(v=0, n=5)
    theirs = {**mine, 'ready:ours': dict(v=0, n=5, ok=5), 'stance:pull': dict(v=1.)}
    t = dict(path='x', export=False, height=False, arms=False, delay=0., label='x')
    rows = {r['key']: r for r in check.rows(mine, theirs, t, t, False, getup=True)}
    assert {key for key, *_ in check.getup_table()} == set(rows)
    assert rows['ready:ours']['value'] == '4 из 5' and rows['ready:ours']['ok'] is False  # 80 % < 99 %
    assert rows['ready:ours']['note'] == 'порог не взят'  # one more failure than the reference: within + 1
    assert rows['stance:width1']['ok'] is True and rows['stance:width1']['value'] == '0,300 / 0,300'
    assert rows['stance:pull']['ok'] is False and rows['stance:pull']['note'] == 'хуже эталона'  # 4 against 1 N·m
    assert rows['handover:turn15']['value'] == '1 из 5'  # 20 % > 12,5 %, as the reference's own
    assert rows['handover:turn15']['note'] == 'порог не взят, как и у эталона' and rows['handover:turn15']['ok']
    assert rows['ready:back']['note'] == 'не хуже эталона'


def test_a_short_run_on_a_small_stage_and_the_journal_card(check, tmp_path):
    """goal_check's parts, a speed and the crouch on a small arm_targets stage with the height command (its random
    policy falls at once): their rows measured, judged against the stage itself; the rest not measured. The journal
    keeps the table beside the goal check, and the English journal shows its numbers with decimal points."""
    _, stage = _stage(tmp_path, ARMS)
    t = check.traits(stage)
    assert t['height'] and t['arms'] and not t['export']
    jobs = check.jobs_of('stage', t, True)
    kinds = {kind for _, _, kind, _ in jobs}
    assert {'goal', 'speed', 'run', 'push', 'feet', 'arm_probe', 'crouch', 'falls', 'arm_joints', 'chatter',
            'hoist'} <= kinds
    assert sum(tag.startswith('falls:') for _, tag, _, _ in jobs) == 2 * check.FALLS_EPISODES
    threads = torch.get_num_threads()
    got = {}
    try:
        for _, tag, kind, job in jobs:
            if tag.startswith(('goal:', 'speed:0.5:', 'crouch:stand', 'crouch:walk')):
                got[tag] = check.work((0, kind, dict(job, seeds=1) if kind == 'goal' else job))[1]
    finally:
        torch.set_num_threads(threads)
    measured = check.measures(got, t)
    assert {f'goal:{k}' for k in range(10)} <= set(measured) and 'speed:0.5' in measured
    assert {'crouch:stand70', 'crouch:stand64', 'crouch:walk', 'crouch:speed'} <= set(measured)
    rows = {r['key']: r for r in check.rows(measured, measured, t, t, True)}
    assert all(rows[f'goal:{k}']['ok'] is not None for k in (0, 1, 2, 3, 5, 6, 7, 8))  # against itself: never worse
    assert rows['speed:0.5']['note'] in ('порог и эталон', 'порог не взят, как и у эталона')
    assert rows['crouch:walk']['ok'] is True and rows['side:03']['note'] == 'не измерено'
    root = tmp_path / 'lab'
    items = list(rows.values())
    journal.set_skills(root, 'exp-1', 9665937408, 'Ветка · 9665,9 млн',
                       dict(branch='exp-0', stage=9385181184, title='Родитель · 9385,2 млн'), items, seconds=600.,
                       runs=700, at=5)
    saved = journal.load(root)['skills']
    assert saved['against']['stage'] == 9385181184 and saved['runs'] == 700 and len(saved['items']) == len(items)
    assert saved['items'][0]['skill'] == items[0]['skill'] and saved['items'][0]['ok'] == items[0]['ok']
    english = i18n.localize_journal(journal.load(root), 'en', i18n.Content(root))['skills']
    index = next(k for k, r in enumerate(items) if r['key'] == 'speed:0.5')
    assert ',' in items[index]['value'] and english['items'][index]['value'] == items[index]['value'].replace(',', '.')
    assert english['against']['stage'] == 9385181184
    with pytest.raises(ValueError):
        journal.set_skills(root, 'exp-1', 1, 'x', dict(branch='exp-0', stage=1, title='y'), [dict(ok='yes')])
