import json
import threading
import urllib.request
import urllib.error

import numpy as np
import pytest

from humanoid_sim.webapp import Playback, DashboardServer, existing_dashboard
from humanoid_sim.core import save_stage

@pytest.fixture
def stages(tmp_path):
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(2)
    env = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1)
    save_stage(tmp_path, model, env)
    model.learn(16)
    save_stage(tmp_path, model, env)
    env.close()
    return tmp_path


def test_playback_stage_pause_and_reset(stages):
    player = Playback(stages)
    try:
        player.submit({'command': 'stage', 'step': 0})
        player.submit({'command': 'pause', 'value': True})
        player.tick()
        a = player.frame()
        assert a['stage'] == 0 and a['paused'] and not a['follow']
        player.tick()
        np.testing.assert_array_equal(a['geoms'], player.frame()['geoms'])
        player.submit({'command': 'pause', 'value': False})
        player.tick()
        assert player.frame()['episode_seconds'] > 0
        player.submit({'command': 'pause', 'value': True})
        player.submit({'command': 'reset'})
        player.tick()
        assert player.frame()['episode_seconds'] == 0
        player.submit({'command': 'latest'})
        player.tick()
        assert player.frame()['follow']
        assert player.frame()['stage'] == 16
    finally:
        player.close()


def test_bad_controls_leave_playback_unchanged(stages):
    player = Playback(stages)
    try:
        for body in ({'command': 'stage', 'step': '../x'}, {'command': 'stage', 'step': 123},
                     {'command': 'pause', 'value': 'no'}, {'command': 'speed', 'value': 999},
                     {'command': 'delete'}):
            with pytest.raises(ValueError):
                player.submit(body)
    finally:
        player.close()


def test_empty_run_can_open_dashboard(tmp_path):
    player = Playback(tmp_path)
    try:
        player.tick()
        assert player.frame()['stage'] is None
        assert player.catalog()['stages'] == []
        assert len(player.geometry()) == 18
    finally:
        player.close()


def test_tour_advances_and_pause_freezes_timer(stages):
    player = Playback(stages)
    try:
        player.submit({'command': 'tour', 'seconds': 5, 'mode': 'time', 'full_fall': True})
        player.tick()
        assert player.frame()['tour'] and player.frame()['stage'] == 0
        player.submit({'command': 'pause', 'value': True})
        player.tick()
        before = player.stage_elapsed
        player.tick()
        assert player.stage_elapsed == before
        player.stage_elapsed = 6
        player.submit({'command': 'pause', 'value': False})
        player.tick()
        assert player.frame()['stage'] == 16
    finally:
        player.close()


def test_default_ending_freezes_then_resets(stages):
    player = Playback(stages)
    try:
        player.done_until = 1
        player._publish()
        before = player.frame()['geoms']
        for _ in range(10):
            player.tick()
        np.testing.assert_array_equal(before, player.frame()['geoms'])
        for _ in range(40):
            player.tick()
        assert player.episode == 1
    finally:
        player.close()


def test_time_limit_does_not_manufacture_fall(stages):
    player = Playback(stages)
    try:
        player.full_fall = True
        player.time_limit = True
        player.done_until = 1
        player._publish()
        before = player.frame()['geoms']
        player.tick()
        np.testing.assert_array_equal(before, player.frame()['geoms'])
    finally:
        player.close()


def test_http_serves_real_data_and_rejects_foreign_origin(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url + '/') as response:
            assert response.status == 200 and 'text/html' in response.headers['Content-Type']
        with urllib.request.urlopen(url + '/api/state') as response:
            assert json.load(response)['stages'] == []
        assert existing_dashboard(url, tmp_path)
        with pytest.raises(ValueError, match='port'):
            existing_dashboard(url, tmp_path / 'another-run')
        request = urllib.request.Request(url + '/api/control', data=b'{"command":"reset"}',
                                         headers={'Content-Type': 'application/json', 'Origin': 'https://evil.example'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
        request = urllib.request.Request(url + '/api/control', data=b'{"command":"speed","value":999}',
                                         headers={'Content-Type': 'application/json', 'Origin': url})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
