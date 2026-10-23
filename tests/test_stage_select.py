"""scripts/lab/probes/stage_select.py, the stage selection rule of iteration 18 with the pushes of iteration 20 and the
height checks of iteration 29: its runs and its verdict, on made-up run results (no simulation); stand_turn's height
plans."""
import copy

import pytest

from test_g1_lock_wrists import lab_probe


@pytest.fixture
def ss(monkeypatch):
    return lab_probe('stage_select', monkeypatch)


def stage(tmp_path):
    return tmp_path / 'exp-0000000000000000' / 'stage_000000000100'


def select_jobs(ss, tmp_path, height=False):
    """Step 2's jobs of a made-up stage, without (the rule before) or with task_config height_command."""
    return ss.select_jobs(stage(tmp_path), height)


def made_up(ss, jobs, rate=.01, stop=.05, turn=4.5):
    """stand_turn.run results with every segment of the case's plan measured (yaw rate `rate`, a stop moving `stop`,
    a stand turning `turn` degrees; a height plan's segments with their height and the pelvis 1 cm above it) and
    bridge_kick.run results that all stand."""
    runs = [dict(case={k: v for k, v in job.items() if k != 'stage'}, fell_at=None,
                 segs=[dict(name=name, cmd=[vx, vy, wz], yaw_rate=rate, fwd=vx, side=vy, heading_deg=turn,
                            disp=stop if name == 'stop' else 1.,
                            **(dict(height=height[0], pelvis=height[0] + .01) if height else {}))
                       for name, vx, vy, wz, _, *height in ss.st.PLANS[job['plan']]])
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
    other five variants, every side step of these again at ±0.5 m/s (iteration 27), then the ±1 m/s side check in
    Unitree +6 ms with the foot capsules (the bridge procedure and three reset starts); pushes in the bridge copy with
    the capsules (judged) and without (printed), 8 directions, standing 10 moments x 6 dv, walking 6 phases x 3 dv.
    Step 3: the critic's fresh check, timing windows step 2 does not use, no pushes."""
    select = select_jobs(ss, tmp_path)
    every = [job for kind, job in select if kind == 'run']
    assert len(select) == 55 + 23 + 4 + 2 * (480 + 144) + 20 and len(every) == 82
    checked = [case for case in every if case['plan'] == 'side1']
    runs = [case for case in every if case['plan'] != 'side1']
    assert len(runs) == 78 and [(c['physics'], c['stale'], c['delay'], c['start'], c.get('phase', 0)) for c in checked] \
        == [('um_caps', True, 3, 'fixair', 0)] + [('um_caps', True, 3, 'reset', 15 * k) for k in range(3)]
    assert [(name, vy) for name, _, vy, _, _ in ss.st.PLANS['side1'] if name != 'stop'] == [
        ('stand', 0.), ('left1', 1.), ('right1', -1.)]
    side, side05 = ss.st.PLANS['side'], ss.st.PLANS['side05']  # the ±0.3 side step's segments at ±0.5, named apart
    assert [(name, vy) for name, _, vy, _, _ in side05] == [('stand', 0.), ('left05', .5), ('stop', 0.),
                                                           ('right05', -.5), ('stop', 0.)]
    assert [(vx, wz, secs) for _, vx, _, wz, secs in side05] == [(vx, wz, secs) for _, vx, _, wz, secs in side]
    assert ss.st.RUNS['side05'] == ss.st.RUNS['side']  # its real timing windows are the ±0.3 run's
    rest = lambda case: sorted((k, str(v)) for k, v in case.items() if k != 'plan')  # physics, start, timing, ...
    assert sorted(map(rest, (c for c in runs if c['plan'] == 'side05'))) == sorted(
        map(rest, (c for c in runs if c['plan'] == 'side')))
    feet = [job for kind, job in select if kind == 'feet']  # foot_overlap: 4 side steps, 16 sideways pushes
    assert [job['cmd'] for job in feet if job['kind'] == 'side'] == [[0., .5, 0.], [0., -.5, 0.], [0., 1., 0.],
                                                                    [0., -1., 0.]]
    assert sorted((job['dir'], job['mag'], job['offset']) for job in feet if job['kind'] == 'push') == sorted(
        (d, m, o) for d in 'lr' for m in (1., 1.25) for o in (0, 37, 71, 113))
    assert {(job['phys'], job['left']) for job in feet if job['kind'] == 'push'} == {('um', False)}
    count = {name: sum(ss.variant_of(case) == name for case in runs) for name, _, _ in ss.VARIANTS}
    assert count == dict(ours=4, ours_6ms=29, fric=4, fric_6ms=4, nomotor=4, um=4, um_6ms=29)
    for name in ('ours', 'fric', 'fric_6ms', 'nomotor', 'um'):
        assert sorted(case['plan'] for case in runs if ss.variant_of(case) == name) == [
            'side', 'side05', 'walk', 'walk1']
    for name in ss.BRIDGE:  # regular timing, 5 real windows, 3 reset starts
        assert sum(case['plan'] == 'side05' for case in runs if ss.variant_of(case) == name) == 9
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

    assert not {job['plan'] for kind, job in select if kind == 'run'} & set(ss.st.HEIGHT_PLANS)  # no height command

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
    for jobs, summary in ((select_jobs(ss, tmp_path), lambda runs, pushes: ss.select_summary(runs, pushes)),
                          (ss.fresh_jobs(stage(tmp_path)), lambda runs, pushes: ss.fresh_summary(runs))):
        got = summary(*made_up(ss, jobs, rate=-.0499, stop=.299))
        assert got['passes'] and got['reasons'] == []
        step3 = 'windows' in got['rows']  # step 3 has no ±0.5 m/s side step
        row = got['rows']['windows' if step3 else 'um_6ms']
        side05 = None if step3 else -.0499
        assert row['drift'] == dict(w05=-.0499, w1=-.0499, L=-.0499, R=-.0499, L05=side05, R05=side05)
        assert row['stop'] == .299 and row['stop05'] == (None if step3 else .299)


def test_step_2_fails_on_a_fall_a_drift_or_a_stop(ss, tmp_path):
    jobs = select_jobs(ss, tmp_path)
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


def test_step_2_judges_the_side_step_at_0_5_m_s_as_the_one_at_0_3(ss, tmp_path):
    """Iteration 27: the ±0.5 m/s side step (the stick's limit of the export) by the same limits, |yaw rate| < 0.05 and
    the stop after it < 0.3 m, in keys of its own: it does not enter the ±0.3 step's drift or stop, nor they its."""
    jobs = select_jobs(ss, tmp_path)
    clean = made_up(ss, jobs)

    def verdict(change):
        runs, pushes = copy.deepcopy(clean)
        change(runs)
        return ss.select_summary(runs, pushes)

    def drift(runs):  # 7121240064 in the bridge copy (lab-critic/it27/side05.log)
        segment(where(runs, physics='um', stale=True, plan='side05')[0], 'right05')['yaw_rate'] = .062

    got = verdict(drift)
    assert got['reasons'] == ['увод вправо 0.5 м/с +0.062, |.| >= 0.05 (Unitree +6 мс (связка))']
    assert got['rows']['um_6ms']['drift']['R05'] == .062 and got['rows']['um_6ms']['drift']['R'] == .01

    def drift_at_0_3(runs):
        segment(where(runs, physics='um', stale=True, plan='side')[0], 'left')['yaw_rate'] = -.057

    got = verdict(drift_at_0_3)
    assert got['reasons'] == ['увод влево -0.057, |.| >= 0.05 (Unitree +6 мс (связка))']
    assert got['rows']['um_6ms']['drift']['L05'] == .01

    def stop(runs):
        segment(where(runs, physics='nomotor', plan='side05')[0], 'stop', 1)['disp'] = .33

    got = verdict(stop)
    assert got['reasons'] == ['сдвиг после стопа за шагом вбок 0.5 м/с: 0.33 м >= 0.3 (без моторной модели)']
    assert got['rows']['nomotor']['stop'] == .05 and got['rows']['nomotor']['stop05'] == .33

    def fall(runs):
        where(runs, physics='ours', stale=True, plan='side05', start='reset')[2]['fell_at'] = 30.5

    assert verdict(fall)['reasons'] == ['падений 1 (наша +6 мс: side05 старт reset 30.5 с)']
    without = ss.select_summary([r for r in clean[0] if r['case']['plan'] != 'side05'], clean[1])
    assert without['reasons'] == ['увод влево 0.5 м/с не измерен', 'увод вправо 0.5 м/с не измерен',
                                  'сдвиг после стопа за шагом вбок 0.5 м/с: не измерен']


def test_step_2_judges_the_pushes_by_the_critics_rule(ss, tmp_path):
    """Standing up to 1.25 m/s at most 12 of 400, no direction x dv cell with 4 of 10, under half of a direction at
    1.25, 1.5 not judged; walking at most 1 of 144. Judged in the bridge copy with the foot capsules; the same pushes
    without them are counted apart ('pushes_sim2sim') and never fail a stage."""
    clean = made_up(ss, select_jobs(ss, tmp_path))

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
    runs, pushes = made_up(ss, select_jobs(ss, tmp_path), turn=-5.4)
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
    assert next(line for line in lines if line[:1] == ['прогонов']) == [
        'прогонов', 'падений', '0.5', 'м/с', '1', 'м/с', 'влево', 'вправо', 'стоп,', 'м', 'влево', '0.5', 'вправо',
        '0.5', 'стоп', '0.5', 'v', '0.5', '/', '1']  # the ±0.5 m/s side step in columns of its own
    assert next(line for line in lines if line[:4] == ['Unitree', '+6', 'мс', '(связка)'])[4:] == [
        '29', '0', '+0.010', '+0.010', '+0.010', '+0.010', '0.05', '+0.010', '+0.010', '0.05', '0.50', '/', '1.00']
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
    clean = made_up(ss, select_jobs(ss, tmp_path))

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


def test_stand_turn_plans_with_the_height_command(ss, monkeypatch):
    """Iteration 29: a segment's sixth element is the height command. The crouched walk is 'walk' (stand 5 s, 0.5 m/s
    10 s, stop 5 s) at the heights of g1_ctrl's height patch with the stick down (0.70 walking, 0.78 standing, the
    critic's cwalk.py), the crouched stands 30 s at 0.74 / 0.64; every other plan has none. Group 'crouch' runs the
    three in the bridge copy, none for a stage without the height command, and run() refuses them there before it
    builds an environment."""
    st = ss.st
    assert st.HEIGHT_PLANS == ('crouch_walk05', 'crouch_stand74', 'crouch_stand64')
    assert [segment[:5] for segment in st.PLANS['crouch_walk05']] == st.PLANS['walk']
    assert [segment[5] for segment in st.PLANS['crouch_walk05']] == [.78, .70, .78]
    assert (st.PLANS['crouch_stand74'], st.PLANS['crouch_stand64']) == (
        [('stand', 0., 0., 0., 30., .74)], [('stand', 0., 0., 0., 30., .64)])
    assert all(len(segment) == 5 for key, plan in st.PLANS.items() if key not in st.HEIGHT_PLANS for segment in plan)
    assert not st.has_height('unitree_v0')
    monkeypatch.setattr(st, 'has_height', lambda stage: False)
    assert st.grid('x', 'crouch') == []
    monkeypatch.setattr(st, 'has_height', lambda stage: True)
    assert st.grid('x', 'crouch') == [dict(st.case(plan=plan), stage='x') for plan in st.HEIGHT_PLANS]
    assert st.case() == dict(physics='um', stale=True, delay=3, start='fixair', plan='stand', seed=73000)
    monkeypatch.setattr(st, 'stage_data', lambda stage: (None, None, {'height_command': False}))
    with pytest.raises(ValueError, match='height_command'):
        st.run(st.case(plan='crouch_stand74', stage='x'))


def test_step_2_runs_the_height_checks_for_a_stage_with_the_height_command(ss, tmp_path, monkeypatch):
    """With task_config height_command step 2 adds three runs in the bridge copy (Unitree +6 ms, the bridge
    procedure, as the critic's cwalk.py): the crouched walk and the two crouched stands; everything else is the same.
    Without the flag select_jobs reads it from the stage (stand_turn.has_height)."""
    without, with_height = select_jobs(ss, tmp_path), select_jobs(ss, tmp_path, height=True)
    added = [job for kind, job in with_height if kind == 'run' and job['plan'] in ss.st.HEIGHT_PLANS]
    assert len(with_height) == len(without) + 3 == 55 + 23 + 4 + 3 + 2 * (480 + 144) + 20
    assert [(kind, job) for kind, job in with_height if job not in added] == without
    assert added == [dict(ss.st.case(plan=plan), stage=str(stage(tmp_path))) for plan in (
        'crouch_walk05', 'crouch_stand74', 'crouch_stand64')]
    assert {ss.variant_of(job) for job in added} == {'um_6ms'}
    for height, jobs in ((True, with_height), (False, without)):
        monkeypatch.setattr(ss.st, 'has_height', lambda path: height)
        assert ss.select_jobs(stage(tmp_path)) == jobs


def test_step_2_judges_a_height_stage_on_the_crouched_walk_and_the_30_s_stand(ss, tmp_path):
    """Iteration 29 (the critic, lab-critic/it29c): with the height command step 2 also judges, in the bridge copy,
    the walk at 0.5 m/s at the height 0.70 (|yaw rate| < 0.03) and the turn of the 30 s stand at 0.78 (|.| <= 10
    degrees); the 30 s crouched stands at 0.74 / 0.64 are printed only, falls included. Their reasons come last, the
    crouched runs stay out of the variants' rows. Without the height command the 30 s turn stays printed only."""
    clean = made_up(ss, select_jobs(ss, tmp_path, height=True))

    def verdict(change, height=True):
        runs, pushes = copy.deepcopy(clean)
        change(runs)
        return ss.select_summary(runs, pushes, height)

    def nothing(runs):
        pass

    got = verdict(nothing)
    assert got['passes'] and got['height']['reasons'] == []
    assert got['height']['walk'] == dict(height=.70, runs=1, fell=None, drift=.01, speed=.5, pelvis=pytest.approx(.71))
    assert [(s['plan'], s['height'], s['heading'], s['fell']) for s in got['height']['stands']] == [
        ('crouch_stand74', .74, 4.5, None), ('crouch_stand64', .64, 4.5, None)]
    assert verdict(nothing, height=False)['height'] is None

    def stage_7701(runs):  # the critic's screen.json / cstand.json, Unitree +6 ms
        segment(where(runs, plan='crouch_walk05')[0], 'walk')['yaw_rate'] = -.0513
        where(runs, plan='stand30', physics='um')[0]['segs'][0]['heading_deg'] = -15.79
        where(runs, plan='crouch_stand74')[0]['segs'][0]['heading_deg'] = 25.51
        where(runs, plan='crouch_stand64')[0]['segs'][0]['heading_deg'] = 10.01

    got = verdict(stage_7701)
    assert got['reasons'] == got['height']['reasons'] == [
        'увод на ходу 0.5 м/с на 0.70 -0.051, |.| >= 0.03 (Unitree +6 мс (связка))',
        'поворот стоя 30 с -15.8°, |.| > 10° (Unitree +6 мс (связка))']
    assert got['rows']['um_6ms']['drift']['w05'] == .01 and got['rows']['um_6ms']['runs'] == 29  # not the walk at 0.78
    assert [s['heading'] for s in got['height']['stands']] == [25.51, 10.01]
    assert verdict(stage_7701, height=False)['passes']  # the rule before: the 30 s turn printed only

    def stage_7741(runs):
        segment(where(runs, plan='crouch_walk05')[0], 'walk')['yaw_rate'] = -.0228
        where(runs, plan='stand30', physics='um')[0]['segs'][0]['heading_deg'] = -3.48
        where(runs, plan='crouch_stand64')[0]['segs'][0]['heading_deg'] = -14.77

    assert verdict(stage_7741)['passes']

    def edges(drift, turn):
        def change(runs):
            segment(where(runs, plan='crouch_walk05')[0], 'walk')['yaw_rate'] = drift
            where(runs, plan='stand30', physics='um')[0]['segs'][0]['heading_deg'] = turn
        return verdict(change)['reasons']

    assert edges(-.0299, -10.) == [] and edges(.0299, 10.) == []
    assert edges(.03, 10.01) == ['увод на ходу 0.5 м/с на 0.70 +0.030, |.| >= 0.03 (Unitree +6 мс (связка))',
                                 'поворот стоя 30 с +10.0°, |.| > 10° (Unitree +6 мс (связка))']
    assert edges(.01, -30.)[0].startswith('поворот стоя 30 с -30.0°')
    assert verdict(lambda runs: segment(where(runs, physics='ours', plan='stand30')[0], 'stand').update(
        heading_deg=40.))['passes']  # ours +6 ms: not the bridge copy

    def falls(runs):
        where(runs, plan='crouch_walk05')[0]['fell_at'] = 12.3
        where(runs, plan='crouch_stand64')[0]['fell_at'] = 7.

    got = verdict(falls)
    assert got['reasons'] == ['падение на ходу 0.5 м/с на 0.70 (Unitree +6 мс (связка): crouch_walk05 12.3 с)']
    assert got['height']['walk']['drift'] is None and got['height']['stands'][1] == dict(
        plan='crouch_stand64', height=.64, runs=1, fell='crouch_stand64 7.0 с', heading=None, pelvis=None)

    def stand30_fell(runs):
        where(runs, plan='stand30', physics='um')[0]['fell_at'] = 12.3

    assert verdict(stand30_fell)['reasons'] == ['падений 1 (Unitree +6 мс (связка): stand30 12.3 с)',
                                                'поворот стоя 30 с не измерен']
    runs, pushes = clean
    got = ss.select_summary([r for r in runs if r['case']['plan'] not in ss.st.HEIGHT_PLANS], pushes, True)
    assert got['reasons'] == ['увод на ходу 0.5 м/с на 0.70 не измерен']
    assert [s['runs'] for s in got['height']['stands']] == [0, 0]


def test_step_2_prints_the_height_checks_in_rows_of_their_own(ss, tmp_path, capsys):
    """A stage with the height command: the 30 s turn marked judged, a table of the height checks (a row each, the
    rule last), their reasons after the others; after every stage a table across the stages with the height command,
    step 2 without and with them."""
    runs, pushes = made_up(ss, select_jobs(ss, tmp_path, height=True))
    segment(where(runs, plan='crouch_walk05')[0], 'walk')['yaw_rate'] = -.0513
    where(runs, plan='stand30', physics='um')[0]['segs'][0]['heading_deg'] = -15.79
    where(runs, plan='crouch_stand64')[0]['fell_at'] = 7.
    failing = ss.select_summary(runs, pushes, True)
    where(runs, physics='nomotor', plan='side')[0]['fell_at'] = 18.8
    summary = ss.select_summary(runs, pushes, True)
    ss.report(dict(label='этап', select=summary, runs=dict(select=1353), seconds=dict(select=1.)))
    out = capsys.readouterr().out
    assert 'стояние 30 с (Unitree +6 мс): поворот -15.8° (судится: |.| <= 10°, этап с командой высоты)\n' in out
    lines = [line.split() for line in out.splitlines()]
    k = lines.index(['команда', 'высоты', '(Unitree', '+6', 'мс', '(связка)):'])
    assert lines[k + 1:k + 6] == [
        ['увод,', 'рад/с', 'поворот,', '°', 'v,', 'м/с', 'таз,', 'м', 'правило'],
        ['ходьба', '0.5', 'м/с', 'на', '0.70', '-0.051', '-', '+0.50', '0.710', '|увод|', '<', '0.03'],
        ['стояние', '30', 'с', 'на', '0.78', '-', '-15.8', '-', '-', '|поворот|', '<=', '10°'],
        ['стояние', '30', 'с', 'на', '0.74', '-', '+4.5', '-', '0.750', 'не', 'судится'],
        ['стояние', '30', 'с', 'на', '0.64', '-', '-', '-', '-', 'не', 'судится;', 'упал:', 'crouch_stand64', '7.0',
         'с']]
    assert ('итог: не проходит: падений 1 (без моторной модели: side 18.8 с) | команда высоты: увод на ходу 0.5 м/с '
            'на 0.70 -0.051, |.| >= 0.03 (Unitree +6 мс (связка)); поворот стоя 30 с -15.8°, |.| > 10° (Unitree +6 мс '
            '(связка))\n') in out
    ss.verdict(failing)
    assert capsys.readouterr().out == (
        '  итог: не проходит: команда высоты: увод на ходу 0.5 м/с на 0.70 -0.051, |.| >= 0.03 (Unitree +6 мс '
        '(связка)); поворот стоя 30 с -15.8°, |.| > 10° (Unitree +6 мс (связка))\n')

    clean = ss.select_summary(*made_up(ss, select_jobs(ss, tmp_path, height=True)), True)
    old = ss.select_summary(*made_up(ss, select_jobs(ss, tmp_path)))
    ss.print_height_table([dict(label='этап 7701', select=failing), dict(label='без высоты', select=old),
                           dict(label='этап 7741', select=clean), dict(label='этап', select=summary)])
    lines = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert lines[1][:3] == ['команда', 'высоты', '(Unitree'] and lines[2] == [
        'увод', '0.70', 'стоя', '0.78', 'стоя', '0.74', 'стоя', '0.64', 'шаг', '2', 'без', 'них', 'шаг', '2']
    assert lines[3:] == [
        ['этап', '7701', '-0.051', '-15.8', '+4.5', 'упал', 'проходит', 'не', 'проходит'],
        ['этап', '7741', '+0.010', '+4.5', '+4.5', '+4.5', 'проходит', 'проходит'],
        ['этап', '-0.051', '-15.8', '+4.5', 'упал', 'не', 'проходит', 'не', 'проходит']]
    ss.print_height_table([dict(label='без высоты', select=old)])
    assert capsys.readouterr().out == ''


def test_step_2_prints_whether_the_feet_pass_through_each_other_without_judging_it(ss, tmp_path, capsys):
    """foot_overlap's summary: per side command the share of samples overlapping with both soles down, the sideways
    pushes overlapping at all and their share; printed under the pushes, the verdict untouched."""
    jobs = [job for kind, job in select_jobs(ss, tmp_path) if kind == 'feet']
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
    runs, pushes = made_up(ss, select_jobs(ss, tmp_path))
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
