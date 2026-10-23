import numpy as np
import pytest
from humanoid_sim.core import write_json
from humanoid_sim.webapp import Playback


def test_renamed_arm_uses_registered_view_adapter(tmp_path, monkeypatch):
    from dataclasses import replace
    from humanoid_sim.tasks import catalog
    from humanoid_sim.tasks.registry import TaskRegistry
    from humanoid_sim.robots.registry import RobotRegistry
    original_robots, original_tasks = catalog.registries()
    source = original_robots.get('so101')
    task = original_tasks.resolve('arm_reach', source)
    robots, tasks = RobotRegistry(), TaskRegistry()
    robots.register(replace(source, robot_id='custom_arm'))
    tasks.register(replace(task, task_id='custom_reach', identity_factory=lambda r: dict(
        task.identity_factory(r), robot_id='custom_arm', task_id='custom_reach')))
    monkeypatch.setattr(catalog, 'registries', lambda: (robots, tasks))
    write_json(tmp_path/'run_config.json', dict(robot_id='custom_arm', task_id='custom_reach'))
    player = Playback(tmp_path)
    try:
        frame = player.frame()
        assert frame['forward_m'] is None
        assert frame['distance_m'] >= 0
        assert frame['height_m'] is None
        assert frame['view']['metric_key'] == 'distance_m'
    finally:
        player.close()


@pytest.mark.parametrize('robot,task', [('g1_29dof','g1_velocity'), ('so101','arm_reach')])
def test_playback_geometry_and_camera_are_robot_specific(tmp_path, robot, task):
    write_json(tmp_path/'run_config.json',dict(robot_id=robot,task_id=task))
    playback = Playback(tmp_path)
    try:
        assert playback.frame()['robot_id'] == robot
        meshes = [g for g in playback.geometry() if g['type'] == 7]
        assert meshes and len(meshes[0]['vertices']) > 0 and len(meshes[0]['faces']) > 0
        if robot == 'so101':
            assert playback.frame()['forward_m'] is None
            assert playback.frame()['distance_m'] >= 0
        assert np.isfinite(playback.frame()['torso']).all()
        frame = playback.frame()
        assert frame['speed_mps'] is None
        assert (frame['command_mps'] is None) == (robot == 'so101')
    finally:
        playback.close()


def test_speed_is_mean_over_recent_window():
    from humanoid_sim.webapp import SpeedTrack
    track = SpeedTrack()
    for t in np.arange(0, 2.01, .02):
        track.add(t, .8*t, 0.)
    assert track.speed() == pytest.approx(.8)
    for t in np.arange(2.02, 4.01, .02):
        track.add(t, 1.6, .3*(t-2))  # turned 90° and slowed down: only the last 2 s count
    assert track.speed() == pytest.approx(.3, rel=.05)
    track.clear()
    for t in np.arange(0, .5, .02):
        track.add(t, 0., 0.)
    assert track.speed() is None  # less than half a window: not shown
