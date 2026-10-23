from humanoid_sim.viewer import ViewControls


def test_view_controls_switch_pause_reset_and_follow():
    controls = ViewControls()
    stages = [0, 100, 200]
    controls.selection.select(stages)
    controls.command('left', stages)
    assert controls.selection.current == 100
    assert controls.reset_requested and not controls.selection.follow
    controls.reset_requested = False
    controls.command('pause', stages)
    assert controls.paused
    controls.command('latest', stages)
    assert controls.selection.follow and controls.reset_requested
    assert controls.selection.select(stages) == 200
