import json
import threading
import urllib.error
import urllib.request

import pytest

from humanoid_sim import journal
from humanoid_sim.goal_check import foot_summary, items
from humanoid_sim.webapp import DashboardServer

PNG = b'\x89PNG\r\n\x1a\n' + b'0' * 16


def test_journal_keeps_roadmap_check_and_entries_newest_first(tmp_path):
    assert journal.load(tmp_path) == {'roadmap': [], 'check': None, 'entries': [], 'iterations': []}
    journal.set_roadmap(tmp_path, [{'title': 'Ходьба', 'status': 'active', 'detail': 'идёт'},
                                   {'title': 'Бег', 'status': 'next'}])
    journal.set_check(tmp_path, 'exp-1', 674070528, 'Ветка · 674 млн',
                      [{'label': 'Без падений', 'value': '0/5', 'target': '0', 'ok': True}], at=5)
    image = tmp_path / 'collage.png'
    image.write_bytes(PNG)
    journal.add_entry(tmp_path, 'first', 'Пошёл', 'Текст', image=image, at=1,
                      race={'robot': 'g1_29dof', 'task': 'g1_velocity', 'scenario': 'walk',
                            'lanes': [{'experiment': 'exp-1', 'stage': 2}]})
    journal.add_entry(tmp_path, 'second', 'Поворот', 'Ещё текст', at=2)
    journal.add_entry(tmp_path, 'first', 'Пошёл!', 'Исправленный текст', image=image, at=1)  # same id replaces
    data = journal.load(tmp_path)
    assert [s['status'] for s in data['roadmap']] == ['active', 'next'] and data['roadmap'][1]['detail'] == '—'
    assert data['check']['stage'] == 674070528 and data['check']['items'][0]['ok'] is True
    assert [e['id'] for e in data['entries']] == ['second', 'first'] and data['entries'][1]['title'] == 'Пошёл!'
    assert data['entries'][1]['image'] == 'first.png' and data['entries'][1]['race'] is None
    assert journal.image_file(tmp_path, 'first.png').read_bytes() == PNG
    for bad in ('../journal.json', 'journal.json', 'first.txt', 'a/b.png', 'missing.png'):
        assert journal.image_file(tmp_path, bad) is None
    with pytest.raises(ValueError):
        journal.add_entry(tmp_path, 'Bad Id', 'x', 'y')
    with pytest.raises(ValueError):
        journal.set_roadmap(tmp_path, [{'title': 'x', 'status': 'maybe'}])
    with pytest.raises(ValueError):
        journal.add_entry(tmp_path, 'third', 'x', 'y', race={'robot': 'r', 'task': 't', 'lanes': [{'stage': 'latest'}]})


def test_stage_story_and_clips_are_kept_with_the_plan(tmp_path):
    clip = tmp_path / 'c.webp'
    clip.write_bytes(b'RIFF0000WEBP')
    (tmp_path / 'c-poster.jpg').write_bytes(b'\xff\xd8\xff')
    journal.set_roadmap(tmp_path, [{'title': 'Ходьба', 'status': 'active'}, {'title': 'Бег', 'status': 'next'}])
    race = {'robot': 'g1_29dof', 'task': 'g1_velocity', 'scenario': 'walk',
            'lanes': [{'experiment': 'exp-1', 'stage': 2}, {'experiment': 'exp-2', 'stage': 'latest'}]}
    name = journal.add_media(tmp_path, clip, 'stage1-path1.webp')
    journal.set_story(tmp_path, 0, {
        'sections': [{'title': 'Что такое «разъём»', 'text': 'Договорённость о числах.'}],
        'path': [{'title': 'Корень', 'change': 'с нуля', 'result': 'стоит', 'verdict': 'control', 'race': race,
                  'clip': name, 'poster': journal.add_media(tmp_path, tmp_path / 'c-poster.jpg', 'stage1-path1.jpg')}],
        'races': [{'title': 'Весь путь', 'text': 'Все этапы', **race}]})
    plan = journal.load(tmp_path)['roadmap']
    assert plan[1]['story'] is None and plan[0]['story']['path'][0]['clip'] == 'stage1-path1.webp'
    assert plan[0]['story']['races'][0]['lanes'][1]['stage'] == 'latest'
    journal.add_entry(tmp_path, 'entry', 'Заголовок', 'Текст', clip=clip)
    entry = journal.load(tmp_path)['entries'][0]
    assert (entry['clip'], entry['poster']) == ('entry-clip.webp', 'entry-clip.jpg')
    assert journal.image_file(tmp_path, 'entry-clip.webp').read_bytes() == clip.read_bytes()
    with pytest.raises(ValueError):
        journal.set_story(tmp_path, 5, {})
    with pytest.raises(ValueError):
        journal.set_story(tmp_path, 0, {'races': [{'title': 'x', **race, 'clip': '../other.webp'}]})


def test_a_story_section_keeps_its_clip(tmp_path):
    """A section of a stage's story carries a clip and its poster as path steps do; without them it is as before."""
    journal.set_roadmap(tmp_path, [{'title': 'Руки', 'status': 'active'}])
    journal.set_story(tmp_path, 0, {'sections': [
        {'title': 'Провис', 'text': 'Плечо висит.', 'clip': 'arms.webp', 'poster': 'arms.jpg'},
        {'title': 'Без ролика', 'text': 'Текст.'}]})
    sections = journal.load(tmp_path)['roadmap'][0]['story']['sections']
    assert sections[0] == {'title': 'Провис', 'text': 'Плечо висит.', 'clip': 'arms.webp', 'poster': 'arms.jpg'}
    assert sections[1] == {'title': 'Без ролика', 'text': 'Текст.'}
    for bad in ('../arms.webp', 'arms.gif'):
        with pytest.raises(ValueError, match='Ролик'):
            journal.set_story(tmp_path, 0, {'sections': [{'title': 'x', 'text': 'y', 'clip': bad}]})


def test_journal_is_served_read_only_with_its_images(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path / 'original')
    image = tmp_path / 'shot.png'
    image.write_bytes(PNG)
    journal.add_entry(server.lab.root, 'entry', 'Заголовок', 'Текст', image=image)
    journal.add_media(server.lab.root, image, 'clip.webp')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url + '/api/lab/journal') as response:
            assert json.load(response)['entries'][0]['image'] == 'entry.png'
        with urllib.request.urlopen(url + '/api/lab/journal/image?name=entry.png') as response:
            assert response.headers['Content-Type'] == 'image/png' and response.read() == PNG
        with urllib.request.urlopen(url + '/api/lab/journal/image?name=clip.webp') as response:
            assert response.headers['Content-Type'] == 'image/webp'
        for name in ('..%2Fjournal.json', 'journal.json', 'none.png'):
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(url + '/api/lab/journal/image?name=' + name)
            assert exc.value.code == 404
        page = tmp_path / 'figure.html'  # an interactive figure: served as a sandboxed page
        page.write_text('<!doctype html><title>x</title>', encoding='utf-8')
        journal.set_roadmap(server.lab.root, [{'title': 'Этап', 'status': 'active', 'story': {
            'sections': [{'title': 'Разъём', 'text': 'Текст', 'figure': journal.add_figure(server.lab.root, page, 'connector.html')}]}}])
        with urllib.request.urlopen(url + '/api/lab/journal') as response:
            assert json.load(response)['roadmap'][0]['story']['sections'][0]['figure'] == 'connector.html'
        with urllib.request.urlopen(url + '/api/lab/journal/figure?name=connector.html') as response:
            assert response.headers['Content-Type'].startswith('text/html')
            assert response.headers['Content-Security-Policy'] == 'sandbox allow-scripts'
        for name in ('journal.json', 'entry.png', '..%2Fconnector.html'):
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(url + '/api/lab/journal/figure?name=' + name)
            assert exc.value.code == 404
        with pytest.raises(ValueError):
            journal.set_story(server.lab.root, 0, {'sections': [{'title': 'Т', 'text': 'Т', 'figure': 'x.js'}]})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


DRIFT = 'Увод без команды (0,5 и 1 м/с, старт с любой ноги)'
STEP = 'Шаг: подъём стопы и наклон в опоре'


def goal_walk(drift=None, lift=.06, tilt=4., pelvis=.77, left_turn=.34):
    drift = drift or {(1., 'left'): [-.02, .01], (1., 'right'): [.03, -.01], (.5, 'left'): [.01, 0.],
                      (.5, 'right'): [.02, .04]}
    return dict(turn={(1., .5): left_turn, (1., -.5): -.38}, drift=drift, pelvis=pelvis,
                feet=dict(lift_m=lift, stance_tilt_deg=tilt, by_foot={}))


def test_goal_items_tick_each_condition():
    scenario = lambda forward, sideways=0., heading=0., lean=3.: dict(
        falls='0/5', forward_m=forward, sideways_m=sideways, heading_deg=heading, torso_lean_deg=lean, clipped=.2)
    good = {'stand': scenario(-.05, .06), 'walk': scenario(9.), 'stop_start': scenario(4.)}
    rows = items(good, .79, goal_walk())
    assert len(rows) == 10 and all(row['ok'] for row in rows)
    bad = {**good, 'stand': {**scenario(-.44, .06), 'falls': '1/5'}, 'walk': scenario(9., heading=13.)}
    rows = {row['label']: row for row in items(bad, 2.5, goal_walk(pelvis=.65, left_turn=.19))}
    failed = {label for label, row in rows.items() if not row['ok']}
    assert failed == {'Без падений: стоять, 1 м/с, стоп-старт', 'Курс при ходьбе', 'Стоять на месте: уползание за 10 с',
                      'Высота таза на ходу (минимум)', 'Поворот по команде ±0,5 рад/с',
                      'Здоровье: действия у предела, шум σ'}
    assert rows['Стоять на месте: уползание за 10 с']['value'] == '0,44 м'


def test_goal_drift_takes_the_worst_start_and_seed():
    scenario = dict(falls='0/5', forward_m=9., sideways_m=0., heading_deg=0., torso_lean_deg=3., clipped=.2)
    good = {'stand': {**scenario, 'forward_m': .05}, 'walk': scenario, 'stop_start': scenario}
    drift = {(1., 'left'): [.01, .02], (1., 'right'): [.01, -.07], (.5, 'left'): [0., .01], (.5, 'right'): [.02, .03]}
    row = {r['label']: r for r in items(good, .79, goal_walk(drift))}[DRIFT]
    assert not row['ok'] and row['value'] == '0,070 рад/с'  # one seed started on the right foot drifts
    row = {r['label']: r for r in items(good, .79, goal_walk({**drift, (1., 'right'): [.01, -.04]}))}[DRIFT]
    assert row['ok'] and row['value'] == '0,040 рад/с'


def test_goal_step_condition_catches_a_shuffling_foot():
    scenario = dict(falls='0/5', forward_m=9., sideways_m=0., heading_deg=0., torso_lean_deg=3., clipped=.2)
    good = {'stand': {**scenario, 'forward_m': .05}, 'walk': scenario, 'stop_start': scenario}
    step = lambda **kw: {r['label']: r for r in items(good, .79, goal_walk(**kw))}[STEP]
    assert step(lift=.06, tilt=4.)['ok'] and step()['value'] == 'подъём 60 мм, наклон 4°'
    row = step(lift=.004, tilt=50.)  # the heel shuffle seen at 4423.9 M
    assert not row['ok'] and row['value'] == 'подъём 4 мм, наклон 50°'
    assert not step(lift=.02, tilt=4.)['ok'] and not step(lift=.06, tilt=16.)['ok']
    assert not step(lift=None, tilt=None)['ok'] and step(lift=None, tilt=None)['value'] == '—'


def test_foot_summary_takes_the_worst_foot_and_walk():
    # Per step, per foot (left, right): contact, lowest sole point m, tilt deg, ankle-roll origin height m.
    lifting = [[(True, 0., 2., .035), (False, .07, 1., .1)], [(False, .05, 3., .08), (True, 0., 3., .035)]] * 10
    shuffling = [[(True, 0., 2., .035), (True, .001, 50., .06)], [(False, .004, 4., .04), (True, 0., 48., .06)]] * 10
    summary = foot_summary([lifting, shuffling])
    assert summary['lift_m'] == 0. and summary['stance_tilt_deg'] == pytest.approx(49.)
    assert summary['by_foot']['left']['lift_m'] == pytest.approx(.004)
    assert summary['by_foot']['left']['stance_tilt_deg'] == pytest.approx(2.)
    assert summary['by_foot']['right']['lift_m'] == pytest.approx(0.)  # never left the floor in the second walk
    # The reward counts the right foot as swinging on every shuffling contact step (origin ≥ CONTACT_HEIGHT).
    assert summary['by_foot']['right']['missed_stance'] == pytest.approx(20 / 30)
    assert summary['by_foot']['left']['missed_stance'] == 0.
    assert foot_summary([[]])['lift_m'] is None


def test_council_iterations_keep_who_said_what_newest_first(tmp_path):
    root = tmp_path / 'lab'
    journal.add_iteration(root, 'iter-1', 'Стояние', 'Как научить стоять?', [
        {'role': 'lead', 'text': 'Предлагаю запретить шаги стоя.'}, {'role': 'critic', 'text': 'Толчки стоя?'}], at=1)
    journal.add_iteration(root, 'iter-2', 'Увод', 'Почему уводит?', [{'role': 'analyst', 'text': 'Шум велик.'}], at=2)
    journal.add_iteration(root, 'iter-1', 'Стояние', 'Как научить стоять?', [
        {'role': 'lead', 'text': 'Запрет шагов стоя.'}], decision='Ветка с запретом шагов.', at=1)
    data = journal.load(root)
    assert [i['id'] for i in data['iterations']] == ['iter-2', 'iter-1']
    assert data['iterations'][0]['decision'] is None and data['iterations'][1]['decision'] == 'Ветка с запретом шагов.'
    assert data['iterations'][1]['messages'] == [{'role': 'lead', 'text': 'Запрет шагов стоя.'}]
    for messages in ([], [{'role': 'boss', 'text': 'x'}], [{'role': 'lead', 'text': ''}]):
        with pytest.raises(ValueError):
            journal.add_iteration(root, 'iter-3', 'Т', 'В', messages)


def test_an_entry_keeps_several_clips_with_captions(tmp_path):
    from humanoid_sim import i18n
    clip, poster = tmp_path / 'a.webp', tmp_path / 'a.jpg'
    clip.write_bytes(b'RIFF')
    poster.write_bytes(b'\xff\xd8')
    journal.add_entry(tmp_path, 'many', 'Ролики', 'Текст', at=1, clip=None)
    journal.add_entry_clips(tmp_path, 'many', [(clip, poster, 'Первый'), (clip, poster, 'Второй')], keep_main=False)
    entry = journal.load(tmp_path)['entries'][0]
    assert [c['clip'] for c in entry['clips']] == ['many-clip-1.webp', 'many-clip-2.webp']
    assert [c['caption'] for c in entry['clips']] == ['Первый', 'Второй'] and 'clip' not in entry
    assert (journal.folder(tmp_path) / 'many-clip-2.jpg').is_file()
    seen = []
    i18n.map_journal(journal.load(tmp_path), lambda text, kind, field: seen.append((kind, text)) or text)
    assert ('clips', 'Второй') in seen  # captions go to the translator like any lab text
    with pytest.raises(ValueError, match='не найдена'):
        journal.add_entry_clips(tmp_path, 'none', [(clip, poster, 'x')])
