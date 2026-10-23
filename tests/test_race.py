import json
import shutil
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from humanoid_sim.core import save_stage, write_json
from humanoid_sim.webapp import Race, DashboardServer

G1 = dict(robot_id='g1_29dof', task_id='g1_velocity')


@pytest.fixture(scope='module')
def g1_run(tmp_path_factory):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    torch.set_num_threads(2)
    run = tmp_path_factory.mktemp('g1') / 'original'
    write_json(run / 'run_config.json', G1)
    env = VecNormalize(DummyVecEnv([G1VelocityEnv]))
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=3)
    save_stage(run, model, env)
    model.learn(16)
    save_stage(run, model, env)
    env.close()
    return run


def tick(race, n):
    for _ in range(n):
        race.tick()
    return race.frame()


def lanes(*specs):
    return [{'experiment': experiment, 'stage': stage} for experiment, stage in specs]


def test_lanes_run_in_lockstep_with_shared_command(g1_run):
    race = Race({'a': g1_run}, names={'a': 'Ветка A'})
    try:
        frame = race.frame()
        assert [(lane['experiment'], lane['stage']) for lane in frame['lanes']] == [('a', 0), ('a', 16)]
        assert frame['experiments'] == [{'id': 'a', 'name': 'Ветка A', 'stages': [0, 16]}]
        assert frame['supports_command'] and frame['scenario'] == 'walk'
        race.submit({'command': 'lanes', 'lanes': lanes(('a', 16), ('a', 16))})
        frame = tick(race, 20)
        a, b = frame['lanes']
        np.testing.assert_array_equal(a['geoms'], b['geoms'])  # same seed, same policy, same physics
        assert frame['seconds'] == pytest.approx(20 * .02)  # the tick that restarts also steps
        assert all(tuple(lane.env.unwrapped.command) == (1., 0., 0.) for lane in race.lanes)
        from humanoid_sim.tasks.config import G1_REWARD_WEIGHTS
        assert set(a['rewards']) == set(G1_REWARD_WEIGHTS) and a['rewards'] == b['rewards']
        assert set(frame['reward_labels']) >= set(G1_REWARD_WEIGHTS)
        race.submit({'command': 'scenario', 'value': 'stand'})
        frame = tick(race, 3)
        assert frame['command_mps'] == 0 and frame['round'] == 1
        assert all(tuple(lane.env.unwrapped.command) == (0., 0., 0.) for lane in race.lanes)
    finally:
        race.close()


def test_lanes_mix_branches_and_follow_new_checkpoints(g1_run, tmp_path):
    shutil.copytree(g1_run, tmp_path / 'a')
    shutil.copytree(g1_run, tmp_path / 'b')
    race = Race({'a': tmp_path / 'a', 'b': tmp_path / 'b'}, focus='b')
    try:
        frame = race.frame()
        assert [(lane['experiment'], lane['stage'], lane['follow']) for lane in frame['lanes']] == [
            ('b', 0, False), ('b', 16, True)]
        race.submit({'command': 'lanes', 'lanes': lanes(('a', 0), ('b', 'latest'))})
        tick(race, 2)
        shutil.copytree(tmp_path / 'b' / 'stage_000000000016', tmp_path / 'b' / 'stage_000000000032')
        race.poll_at = 0  # pick up new checkpoints now instead of in 2 s
        race.submit({'command': 'restart'})
        frame = tick(race, 2)
        assert [(lane['experiment'], lane['stage'], lane['follow']) for lane in frame['lanes']] == [
            ('a', 0, False), ('b', 32, True)]
        assert {e['id']: e['stages'] for e in frame['experiments']} == {'a': [0, 16], 'b': [0, 16, 32]}
    finally:
        race.close()


def test_bad_controls_are_rejected(g1_run):
    race = Race({'a': g1_run})
    try:
        for body in ({'command': 'lanes', 'lanes': []}, {'command': 'lanes', 'lanes': lanes(*[('a', 0)] * 11)},
                     {'command': 'lanes', 'lanes': lanes(('a', 5))}, {'command': 'lanes', 'lanes': lanes(('a', '0'))},
                     {'command': 'lanes', 'lanes': lanes(('other', 0))},
                     {'command': 'lanes', 'lanes': lanes(('a', 'newest'))},
                     {'command': 'runs', 'runs': {}}, {'command': 'scenario', 'value': 'run'},
                     {'command': 'speed', 'value': 3}, {'command': 'pause', 'value': 1}, {'command': 'delete'}, []):
            with pytest.raises(ValueError):
                race.submit(body)
    finally:
        race.close()


def test_fallen_lanes_stop_measuring_and_race_restarts(g1_run):
    race = Race({'a': g1_run})
    try:
        frame = race.frame()
        for _ in range(2000):
            frame = tick(race, 1)
            if frame['ended']:
                break
        assert frame['ended']
        assert all(lane['fell_at'] is not None or lane['finished'] for lane in frame['lanes'])
        assert all(lane['speed_mps'] is None for lane in frame['lanes'])
        fallen = [lane['distance_m'] for lane in frame['lanes']]
        frame = tick(race, 10)
        assert [lane['distance_m'] for lane in frame['lanes']] == pytest.approx(fallen, abs=.3)
        frame = tick(race, round(Race.HOLD_SECONDS / .02) - 10)
        assert frame['round'] == 2 and frame['seconds'] == 0 and not frame['ended']
    finally:
        race.close()


def test_ten_lanes_and_frame_serialized_once(g1_run):
    race = Race({'a': g1_run})
    try:
        race.submit({'command': 'lanes', 'lanes': lanes(*[('a', 0), ('a', 16)] * 5)})
        frame = tick(race, 4)
        assert len(frame['lanes']) == Race.MAX_LANES == 10
        seq, text = race.frame_json()
        assert race.frame_json()[1] is text and json.loads(text)['seq'] == seq
    finally:
        race.close()


def test_race_needs_saved_stages_of_a_walking_robot(tmp_path):
    write_json(tmp_path / 'g1' / 'run_config.json', G1)
    with pytest.raises(ValueError, match='этапов'):
        Race({'a': tmp_path / 'g1'})
    write_json(tmp_path / 'arm' / 'run_config.json', dict(robot_id='so101', task_id='arm_reach'))
    (tmp_path / 'arm' / 'stage_000000000000').mkdir()
    for name in ('policy.zip', 'normalize.pkl', 'metadata.json'):
        (tmp_path / 'arm' / 'stage_000000000000' / name).write_text('{}')
    with pytest.raises(ValueError, match='шагающих'):
        Race({'a': tmp_path / 'arm'})


def test_initial_landing_is_not_a_step():
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from humanoid_sim.webapp import Cadence
    env = G1VelocityEnv()
    try:
        env.reset(seed=1, options={'command': [0, 0, 0]})
        cadence = Cadence(env.model, ('left_ankle_roll_link', 'right_ankle_roll_link'), env.dt)
        for _ in range(15):  # the robot drops 1–2 cm onto its feet
            env.step(np.zeros(29))
            cadence.update(env.model, env.data)
        assert cadence.count == 0 and all(cadence.landed)
    finally:
        env.close()


def test_race_http_endpoints_by_robot(g1_run, tmp_path):
    original = tmp_path / 'original'
    shutil.copytree(g1_run, original)
    server = DashboardServer(('127.0.0.1', 0), original)
    branch = server.lab.create({'name': 'Ветка', **G1, 'source_id': 'baseline', 'source_stage': 16})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'

    def api(path, body=None):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    try:
        robot = '?robot=g1_29dof'
        assert len(api('/api/race/model' + robot)) == len(api('/api/model'))
        frame = api('/api/race/frame' + robot)
        assert {e['id'] for e in frame['experiments']} == {'baseline', branch['id']}
        api('/api/race/control' + robot, {'command': 'lanes', 'lanes': lanes(('baseline', 0), (branch['id'], 'latest'))})
        with pytest.raises(urllib.error.HTTPError) as exc:
            api('/api/race/control' + robot, {'command': 'lanes', 'lanes': lanes(('baseline', 123))})
        assert exc.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as exc:
            api('/api/race/frame?robot=no_such_robot')
        assert exc.value.code == 400
        # A branch created after the race opened is a valid lane at once, with no race page streaming
        # (the race loop only ticks while one is), as the journal's «Открыть в 3D» needs.
        later = server.lab.create({'name': 'Новая ветка', **G1, 'source_id': 'baseline', 'source_stage': 16})
        api('/api/race/control' + robot, {'command': 'lanes', 'lanes': lanes((later['id'], 'latest'))})
        server.lab.annotate(branch['id'], verdict='best', note='Гипотеза: проверка')
        assert api('/api/lab')['notes'][branch['id']]['verdict'] == 'best'
        with pytest.raises(ValueError):
            server.lab.annotate(branch['id'], verdict='winner')
        with urllib.request.urlopen(url + '/race.html') as response:
            assert 'race.js' in response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
