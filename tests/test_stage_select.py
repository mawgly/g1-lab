"""scripts/lab/probes/stage_select.py, the stage selection rule of iteration 18 with the pushes of iteration 20: its
runs and its verdict, on made-up run results (no simulation)."""
import copy

import pytest

from test_g1_lock_wrists import lab_probe


@pytest.fixture
def ss(monkeypatch):
    return lab_probe('stage_select', monkeypatch)


def stage(tmp_path):
    return tmp_path / 'exp-0000000000000000' / 'stage_000000000100'


def made_up(ss, jobs, rate=.01, stop=.05, turn=4.5):
    """stand_turn.run results with every segment of the case's plan measured (yaw rate `rate`, a stop moving `stop`,
    a stand turning `turn` degrees) and bridge_kick.run results that all stand."""
    runs = [dict(case={k: v for k, v in job.items() if k != 'stage'}, fell_at=None,
                 segs=[dict(name=name, cmd=[vx, vy, wz], yaw_rate=rate, fwd=vx, side=vy, heading_deg=turn,
                            disp=stop if name == 'stop' else 1.)
                       for name, vx, vy, wz, _ in ss.st.PLANS[job['plan']]])
            for kind, job in jobs if kind == 'run']
    pushes = [dict(job={k: v for k, v in job.items() if k != 'path'}, fell=None, fell_before=False)
              for kind, job in jobs if kind == 'push']
    return runs, pushes


def where(runs, **case):
    found = [r for r in runs if all(r['case'].get(k) == v for k, v in case.items())]
    assert found, case
    return found


def segment(run, name, n=0):
    return [g for g in run['segs'] if g['name'] == name][n]


def test_runs_of_the_two_steps(ss, tmp_path):
    """Step 2: stand_turn's bridge group (ours and Unitree +6 ms, 20 each) and side / walk 0.5 / walk 1.0 for the
    other five variants, then the ±1 m/s side check in Unitree +6 ms with the foot capsules (the bridge procedure and
    three reset starts); pushes in the bridge copy with the capsules (judged) and without (printed), 8 directions,
    standing 10 moments x 6 dv, walking 6 phases x 3 dv. Step 3: the critic's fresh check, timing windows step 2
    does not use, no pushes."""
    select = ss.select_jobs(stage(tmp_path))
    every = [job for kind, job in select if kind == 'run']
    assert len(select) == 55 + 4 + 2 * (480 + 144) + 20 and len(every) == 59
    checked = [case for case in every if case['plan'] == 'side1']
    runs = [case for case in every if case['plan'] != 'side1']
    assert len(runs) == 55 and [(c['physics'], c['stale'], c['delay'], c['start'], c.get('phase', 0)) for c in checked] \
        == [('um_caps', True, 3, 'fixair', 0)] + [('um_caps', True, 3, 'reset', 15 * k) for k in range(3)]
    assert [(name, vy) for name, _, vy, _, _ in ss.st.PLANS['side1'] if name != 'stop'] == [
        ('stand', 0.), ('left1', 1.), ('right1', -1.)]
    feet = [job for kind, job in select if kind == 'feet']  # foot_overlap: 4 side steps, 16 sideways pushes
    assert [job['cmd'] for job in feet if job['kind'] == 'side'] == [[0., .5, 0.], [0., -.5, 0.], [0., 1., 0.],
                                                                    [0., -1., 0.]]
    assert sorted((job['dir'], job['mag'], job['offset']) for job in feet if job['kind'] == 'push') == sorted(
        (d, m, o) for d in 'lr' for m in (1., 1.25) for o in (0, 37, 71, 113))
    assert {(job['phys'], job['left']) for job in feet if job['kind'] == 'push'} == {('um', False)}
    count = {name: sum(ss.variant_of(case) == name for case in runs) for name, _, _ in ss.VARIANTS}
    assert count == dict(ours=3, ours_6ms=20, fric=3, fric_6ms=3, nomotor=3, um=3, um_6ms=20)
    for name in ('ours', 'fric', 'fric_6ms', 'nomotor', 'um'):
        assert sorted(case['plan'] for case in runs if ss.variant_of(case) == name) == ['side', 'walk', 'walk1']
    assert [ss.variant_of(case) for case in runs if case['plan'] == 'stand30'] == ['ours_6ms', 'um_6ms']
    every = [job for kind, job in select if kind == 'push']
    assert {(job['phys'], job['left']) for job in every} == {('um_caps', False), ('um', False)}
    assert (ss.PUSH_PHYSICS, ss.FORECAST_PHYSICS) == ('um_caps', 'um') and every[0]['phys'] == 'um_caps'
    assert len(ss.DIRS8) == 8 and ss.kick.DIRS is ss.DIRS8 and set(ss.DIR_LABELS) == set(ss.DIRS8)
    assert ss.DIRS8['a000'] == (1., 0.) and ss.DIRS8['a090'] == pytest.approx((0., 1.))  # forward, left
    for physics in ('um_caps', 'um'):  # the same 624 pushes in both
        pushes = [job for job in every if job['phys'] == physics]
        stand = sorted((job['dir'], job['mag'], job['offset']) for job in pushes if job['plan'] == 'stand')
        assert stand == sorted((d, m, o) for d in ss.DIRS8 for m in (.8, .9, 1., 1.1, 1.25, 1.5)
                               for o in (0, 37, 71, 113, 151, 197, 251, 307, 379, 457))
        walk = sorted((job['dir'], job['mag'], job['offset']) for job in pushes if job['plan'] == 'walk')
        assert walk == sorted((d, m, o) for d in ss.DIRS8 for m in (.75, 1., 1.25) for o in range(0, 300, 50))

    fresh = ss.fresh_jobs(stage(tmp_path))
    cases = [job for kind, job in fresh if kind == 'run']
    assert len(fresh) == len(cases) == 19
    groups = [ss.fresh_group(case) for case in cases]
    assert {name: groups.count(name) for name, _, _ in ss.FRESH} == dict(windows=13, right=1, stopstart=1, delay4=2,
                                                                         delay8=2)
    assert all(case['physics'] == 'um' and case['stale'] for case in cases)
    window = lambda case: (case['plan'], case['which'], case.get('offset', 0))
    seen = {window(case) for case in runs if case.get('timing') == 'real'}
    assert seen and not seen & {window(case) for case in cases if case.get('timing') == 'real'}


def test_a_stage_inside_every_limit_passes_both_steps(ss, tmp_path):
    for jobs, summary in ((ss.select_jobs(stage(tmp_path)), lambda runs, pushes: ss.select_summary(runs, pushes)),
                          (ss.fresh_jobs(stage(tmp_path)), lambda runs, pushes: ss.fresh_summary(runs))):
        got = summary(*made_up(ss, jobs, rate=-.0499, stop=.299))
        assert got['passes'] and got['reasons'] == []
        row = got['rows']['windows' if 'windows' in got['rows'] else 'um_6ms']
        assert row['drift'] == dict(w05=-.0499, w1=-.0499, L=-.0499, R=-.0499) and row['stop'] == .299


def test_step_2_fails_on_a_fall_a_drift_or_a_stop(ss, tmp_path):
    jobs = ss.select_jobs(stage(tmp_path))
    clean = made_up(ss, jobs)

    def verdict(change):
        runs, pushes = copy.deepcopy(clean)
        change(runs, pushes)
        return ss.select_summary(runs, pushes)

    def fall(runs, pushes):
        where(runs, physics='nomotor', plan='side')[0]['fell_at'] = 18.8

    got = verdict(fall)
    assert not got['passes'] and got['reasons'] == ['падений 1 (без моторной модели: side 18.8 с)']
    assert got['rows']['nomotor']['drift']['L'] is None  # a run that fell measures nothing

    def drift(runs, pushes):
        segment(where(runs, physics='um', stale=False, plan='walk1')[0], 'walk')['yaw_rate'] = .05

    assert verdict(drift)['reasons'] == ['увод 1 м/с +0.050, |.| >= 0.05 (Unitree)']

    def stop_after_side_step(runs, pushes):
        segment(where(runs, physics='ours', stale=False, plan='side')[0], 'stop', 1)['disp'] = .3

    assert verdict(stop_after_side_step)['reasons'] == ['сдвиг после стопа за шагом вбок 0.30 м >= 0.3 (наша)']

    def stop_after_walking(runs, pushes):  # stopping after a walk forward is not the side-step stop
        for run in where(runs, plan='walk'):
            segment(run, 'stop')['disp'] = 1.

    assert verdict(stop_after_walking)['passes']


def test_step_2_judges_the_pushes_by_the_critics_rule(ss, tmp_path):
    """Standing up to 1.25 m/s at most 12 of 400, no direction x dv cell with 4 of 10, under half of a direction at
    1.25, 1.5 not judged; walking at most 1 of 144. Judged in the bridge copy with the foot capsules; the same pushes
    without them are counted apart ('pushes_sim2sim') and never fail a stage."""
    clean = made_up(ss, ss.select_jobs(stage(tmp_path)))

    def verdict(*falls, plan='stand', physics='um_caps'):
        """falls: (direction, dv, how many of its moments fall)."""
        runs, pushes = copy.deepcopy(clean)
        for d, mag, n in falls:
            cell = [p for p in pushes if (p['job']['plan'], p['job']['dir'], p['job']['mag'], p['job']['phys'])
                    == (plan, d, mag, physics)]
            for p in cell[:n]:
                p['fell'] = .8
        return ss.select_summary(runs, pushes)

    without = verdict(('a270', 1., 10), ('a090', 1.25, 10), physics='um')  # capsule-free falls: the forecast only
    assert without['passes'] and without['pushes_sim2sim']['stand']['falls']['a270'] == [0, 0, 10, 0, 0, 0]
    assert without['pushes']['stand']['falls']['a270'] == [0] * 6
    assert verdict(('a270', 1., 10))['reasons'] == ['клетки стоя с 4 из 10 и больше: вправо 1.0 — 10']

    got = verdict(('a000', 1.25, 1), ('a090', 1.1, 2), ('a090', 1.5, 1), ('a180', 1., 3), ('a180', 1.1, 2))  # 5968
    assert got['passes'] and got['pushes']['stand']['falls']['a180'] == [0, 0, 3, 2, 0, 0]
    twelve = [(d, mag, 3) for d, mag in (('a000', .8), ('a090', .9), ('a180', 1.), ('a270', 1.1))]
    assert verdict(*twelve)['passes']
    assert verdict(*twelve, ('a045', 1.25, 1))['reasons'] == ['толчки стоя до 1.25 м/с: 13 из 400, можно до 12']
    assert verdict(('a000', 1., 4))['reasons'] == ['клетки стоя с 4 из 10 и больше: вперёд 1.0 — 4']
    assert verdict(('a270', 1.25, 4))['passes']  # at 1.25, the margin's edge, only the half rule judges
    assert verdict(('a270', 1.1, 4))['reasons'] == ['клетки стоя с 4 из 10 и больше: вправо 1.1 — 4']
    assert verdict(('a270', 1.25, 5))['reasons'] == [
        'толчки стоя 1.25 м/с: падает половина и больше (вправо 5 из 10)']
    assert verdict(*[(d, 1.5, 10) for d in ss.DIRS8])['passes']  # 1.5 is printed, not judged
    assert verdict(('a045', 1.25, 1), plan='walk')['passes']
    assert verdict(('a045', 1.25, 1), ('a090', .75, 1), plan='walk')['reasons'] == [
        'толчки на ходу 0.5 м/с: 2 из 144, можно до 1 (вперёд-влево 1, влево 1)']

    runs, pushes = clean
    assert ss.select_summary(runs, [p for p in pushes if p['job']['plan'] == 'stand'])['reasons'] == [
        'толчков на ходу 0, нужно 144 (по 6 в клетке)']
    assert pushes[0]['job']['phys'] == 'um_caps'
    assert ss.select_summary(runs, pushes[1:])['reasons'] == ['толчков стоя 479, нужно 480 (по 10 в клетке)']
    assert ss.select_summary(runs, [p for p in pushes if p['job']['phys'] == 'um_caps'])['passes']  # forecast optional


def test_step_2_prints_the_30_s_turn_and_the_push_tables(ss, tmp_path, capsys):
    runs, pushes = made_up(ss, ss.select_jobs(stage(tmp_path)), turn=-5.4)
    where(runs, physics='ours', plan='stand30')[0]['segs'][0]['heading_deg'] = 30.  # ours +6 ms: not the bridge copy
    for p in [p for p in pushes if p['job']['plan'] == 'stand' and p['job']['dir'] == 'a000'][:6]:
        p['fell'] = 1.
    for p in [p for p in pushes if p['job']['plan'] == 'walk' and p['job']['dir'] == 'a270'][:2]:
        p['fell'] = 1.
    summary = ss.select_summary(runs, pushes)
    assert summary['stand30'] == dict(heading=-5.4, fell=None)
    ss.report(dict(label='этап', select=summary, runs=dict(select=1307), seconds=dict(select=1.)))
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert 'стояние 30 с (Unitree +6 мс): поворот -5.4°' in out
    heads = [k for k, line in enumerate(lines) if line == ['0.8', '0.9', '1.0', '1.1', '1.25', '1.5']]
    assert len(heads) == 2  # with the capsules (judged), then without (the sim2sim forecast)
    for head, first in zip(heads, ('6', '0')):  # the falls were made up in the judged pushes only
        assert [line[0] for line in lines[head + 1:head + 9]] == [ss.DIR_LABELS[d] for d in ss.DIRS8]
        assert lines[head + 1] == ['вперёд', first, '0', '0', '0', '0', '0'] and lines[head + 5][1:] == ['0'] * 6
    assert 'толчки стоя (Unitree +6 мс + капсулы стоп, судится): падений из 10 моментов 0-0.91 с' in out
    assert ('толчки стоя (Unitree +6 мс без капсул, прогноз sim2sim: стопы проходят друг сквозь друга, как в '
            'unitree_mujoco; не судится)') in out
    assert '    до 1.25 м/с: 6 из 400 (можно до 12)\n' in out and '    до 1.25 м/с: 0 из 400 (можно до 12)\n' in out
    assert ('толчки на ходу 0.5 м/с (Unitree +6 мс + капсулы стоп, судится, 6 фаз шага): 2 из 144 (Δv 0.75 / 1.0 / '
            '1.25: 2 / 0 / 0); вправо 2\n') in out
    assert ('шаг вбок ±1 м/с (Unitree +6 мс + капсулы стоп, судится по падениям): прогонов 4, падений 0; скорость '
            'вбок +1.00 / -1.00 м/с, увод +0.010 / +0.010 рад/с, стоп после шага 0.05 м') in out
    assert 'итог: не проходит: клетки стоя с 4 из 10 и больше: вперёд 0.8 — 6; толчки на ходу' in out

    where(runs, physics='um', plan='stand30')[0]['fell_at'] = 12.3
    turn = ss.select_summary(runs, pushes)['stand30']
    assert turn == dict(heading=None, fell=12.3)


def test_step_2_judges_the_side_check_at_1_m_s_on_falls_only(ss, tmp_path):
    """The ±1 m/s side steps (Unitree +6 ms with the foot capsules) fail a stage by a fall; their drift and the stop
    after them are printed, not judged, and do not enter the ±0.3 side step's drift or stop rule."""
    clean = made_up(ss, ss.select_jobs(stage(tmp_path)))

    def verdict(change):
        runs, pushes = copy.deepcopy(clean)
        change([r for r in runs if r['case']['plan'] == 'side1'])
        return ss.select_summary(runs, pushes)

    def drift_and_crawl(checked):
        for r in checked:
            segment(r, 'left1')['yaw_rate'] = .2
            segment(r, 'stop', 0)['disp'] = 1.

    got = verdict(drift_and_crawl)
    assert got['passes'] and got['side1']['drift']['left1'] == .2 and got['side1']['stop'] == 1.
    assert got['rows']['um_6ms']['drift']['L'] == .01 and got['rows']['um_6ms']['stop'] == .05

    def fall(checked):
        checked[0]['fell_at'] = 9.4
        checked[2]['fell_at'] = 21.

    got = verdict(fall)
    assert got['reasons'] == ['шаг вбок ±1 м/с (Unitree +6 мс + капсулы стоп): падений 2 из 4 (side1 9.4 с; '
                              'side1 старт reset 21.0 с)']
    assert got['side1']['speed'] == {'left1': 1., 'right1': -1.}  # the runs that stood
    assert ss.select_summary([r for r in clean[0] if r['case']['plan'] != 'side1'], clean[1])['reasons'] == [
        'шаг вбок ±1 м/с не измерен']


def test_step_2_prints_whether_the_feet_pass_through_each_other_without_judging_it(ss, tmp_path, capsys):
    """foot_overlap's summary: per side command the share of samples overlapping with both soles down, the sideways
    pushes overlapping at all and their share; printed under the pushes, the verdict untouched."""
    jobs = [job for kind, job in ss.select_jobs(stage(tmp_path)) if kind == 'feet']
    results = []
    for job in jobs:
        side = job['kind'] == 'side'
        low = 30 if side and job['cmd'][1] == 1. else 0
        results.append(dict(job={k: v for k, v in job.items() if k != 'path'}, samples=300 if side else 150,
                            overlap=low if side else (3 if job['dir'] == 'l' else 0), low=low if side else 0,
                            dmin=-.019 if low else .04, fell=None))
    feet = ss.feet.summary(results)
    assert feet['side']['влево 1']['share'] == pytest.approx(.1) and feet['side']['вправо 1']['share'] == 0
    assert (feet['push_overlapping'], feet['push_runs'], feet['push_share']) == (8, 16, 0)
    assert feet['pushes']['l']['overlapping'] == 8 and feet['pushes']['r']['overlapping'] == 0
    runs, pushes = made_up(ss, ss.select_jobs(stage(tmp_path)))
    summary = ss.select_summary(runs, pushes)
    ss.report(dict(label='этап', select=summary, feet=feet, runs=dict(select=699), seconds=dict(select=1.)))
    out = capsys.readouterr().out
    assert ('вбок влево 0.5 / вправо 0.5 / влево 1 / вправо 1 — 0.0% / 0.0% / 10.0% / 0.0%; толчки вбок '
            '(Unitree +6 мс, Δv 1.0 / 1.25) — перекрытие в 8 из 16, 0.0% шагов за 3 с') in out
    assert summary['passes'] and 'итог: проходит' in out


def test_step_3_judges_drift_at_the_bridge_delay_only(ss, tmp_path):
    """The 4 / 8 ms runs and stop-start count falls (and the side-step stop), not drift."""
    clean, _ = made_up(ss, ss.fresh_jobs(stage(tmp_path)))

    def verdict(change):
        runs = copy.deepcopy(clean)
        change(runs)
        return ss.fresh_summary(runs)

    def drift_8ms(runs):
        segment(where(runs, delay=4, plan='side')[0], 'left')['yaw_rate'] = -.06
        segment(where(runs, delay=4, plan='walk1')[0], 'walk')['yaw_rate'] = .06
        for g in where(runs, plan='stopstart')[0]['segs']:
            g['yaw_rate'] = .2

    got = verdict(drift_8ms)
    assert got['passes'] and got['rows']['delay8']['drift']['L'] == -.06

    def drift_window(runs):
        segment(where(runs, timing='real', plan='walk1')[1], 'walk')['yaw_rate'] = .06

    assert verdict(drift_window)['reasons'] == ['увод 1 м/с +0.060, |.| >= 0.05 (другие окна времени)']

    def fall_4ms(runs):
        where(runs, delay=2, plan='side')[0]['fell_at'] = -1.  # the critic's mark of an early fall

    assert verdict(fall_4ms)['reasons'] == ['падений 1 (задержка 4 мс: side)']

    def stop_8ms(runs):
        segment(where(runs, delay=4, plan='side')[0], 'stop')['disp'] = .35

    assert verdict(stop_8ms)['reasons'] == ['сдвиг после стопа за шагом вбок 0.35 м >= 0.3 (задержка 8 мс)']
