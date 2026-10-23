import json
import re
import threading
import urllib.request
import urllib.error

import pytest

from humanoid_sim.webapp import WEB, DashboardServer


def test_lab_http_create_isolation_and_baseline_write_rejected(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path/'original')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def api(path, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    try:
        robots = api('/api/lab/robots')
        assert {r['robot_id'] for r in robots} == {'humanoid', 'so101', 'g1_29dof', 'g1_12dof'}
        assert next(r for r in robots if r['robot_id']=='g1_29dof')['tasks'][0]['training_backends'] == ['cpu', 'full_gpu']
        assert next(r for r in robots if r['robot_id']=='so101')['tasks'][0]['training_backends'] == ['cpu']
        for path in ('/api/train/start', '/api/train/legacy-start-disabled', '/api/train/stop'):
            with pytest.raises(urllib.error.HTTPError) as exc:
                api(path, {})
            assert exc.value.code in (400, 404)
        created = api('/api/lab/create', {'name': 'Тест'})
        assert len(api('/api/lab')['experiments']) == 2
        assert api('/api/lab')['server_started'] == server.started  # an open page reloads when it changes
        g1 = api('/api/lab/create', {'name':'Ablation', 'robot_id':'g1_29dof', 'task_id':'g1_velocity',
                                   'task_config':{'reward_weights':{'action_change':0}}})
        assert g1['config']['task_config']['reward_weights']['action_change']==0
        assert g1['config']['task_config']['reward_weights']['velocity']==2
        with pytest.raises(urllib.error.HTTPError) as exc:
            api('/api/lab/create', {'robot_id':'g1_29dof', 'task_id':'g1_velocity',
                                    'task_config':{'reward_weights':{'tilt':-1}}})
        assert exc.value.code==400
        assert api('/api/state?experiment='+created['id'])['stages'] == []
        with pytest.raises(urllib.error.HTTPError):
            api('/api/state?experiment=..%2Fother')
        assert not (tmp_path/'original'/'stop.request').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_lab_page_is_the_read_only_branch_tree(tmp_path):
    """/lab.html is the site's Lab page (labtree.js in site.js's frame), read-only: the old lab.js / lab.css UI with
    its training form is gone and no page loads it, while the API the lab's agents and scripts call stays."""
    server = DashboardServer(('127.0.0.1', 0), tmp_path/'original')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'

    def get(path):
        with urllib.request.urlopen(url + path) as response:
            return response.read().decode('utf-8')
    try:
        page, script = get('/lab.html'), get('/labtree.js')
        assert 'src="labtree.js"' in page and 'href="site.css"' in page
        assert '<meta name="site-mode" content="live"' in page  # humanoid_sim.site_export takes it into the site
        assert "from './site.js'" in script and "frame('lab.html')" in script
        # Every element the script looks up is on its page (also for the race page and its link back).
        for html, js in ((page, script), (get('/race.html'), get('/race.js'))):
            ids = set(re.findall(r"\$\('([\w-]+)'\)", js))
            assert ids and [i for i in ids if f'id="{i}"' not in html] == []
        assert 'href="lab.html"' in get('/race.html')
        # Read-only: nothing on the page posts, starts, stops or queues training.
        for text in (page, script):
            assert "method: 'POST'" not in text and '<form' not in text
            assert not re.search(r'/api/lab/(create|series|enqueue|start|stop|job)', text)
        for name in ('lab.js', 'lab.css'):
            assert not (WEB / name).exists()
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(url + '/' + name)
            assert exc.value.code == 404
        old = re.compile(r'''["'/]lab\.(?:js|css)\b''')
        for file in [*WEB.glob('*.html'), *WEB.glob('*.js'), *WEB.glob('*.css')]:
            assert not old.search(file.read_text(encoding='utf-8')), file.name
        # The training API stays for the agents: a branch is still created and queued over HTTP.
        request = urllib.request.Request(url + '/api/lab/create', data=json.dumps({'name': 'Агент'}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            created = json.load(response)
        request = urllib.request.Request(url + '/api/lab/enqueue', data=json.dumps({'ids': [created['id']]}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['items'][-1] == {'id': created['id'], 'state': 'pending'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_unused_players_are_evicted(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path/'original')
    try:
        for index in range(15):
            experiment = server.lab.create({'name': str(index)})
            player = server.get_player({'experiment': [experiment['id']]})
            assert player.client_count == 0
        assert len(server.players) <= 12
        assert server.player in server.players.values()
    finally:
        server.server_close()
