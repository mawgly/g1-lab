import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

# task_config of exp-7f2494695bab436f (iteration 6, yaw_drift 5), the branch iteration 7 starts from.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, action_limit=10.,
              heading_hold=True, stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
              commands=dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                            straight_probability=.5),
              reward_weights=dict(velocity=2., yaw=2., alive=.15, orientation=2., height=10., torso=1., still=3.,
                                  stand_pose=1., slip=1., feet_flat=2., yaw_drift=5., posture=.3, hips=1.,
                                  action_change=.05, fall=100.))


def test_random_phase_draws_training_starts_and_keeps_evaluation_at_phase_zero():
    env = G1VelocityEnv(real_robot=True, phase_stop=True, history=5, random_phase=True, gait=dict(period=.6))
    try:
        period = round(env.gait['period'] / env.dt)
        assert period == 30
        starts = set()
        for seed in range(40):
            env.reset(seed=seed)
            assert 0 <= env.phase_steps < period
            starts.add(env.phase_steps)
        assert len(starts) > 10  # spread over the cycle, both feet first
        env.reset(seed=3, options={'command': [.5, 0., 0.]})
        assert env.phase_steps == 0  # evaluation scenarios and goal_check keep their start
        env.reset(seed=3, options={'command': [.5, 0., 0.], 'phase_steps': 15})
        assert env.phase_steps == 15
    finally:
        env.close()


def test_random_phase_off_by_default_and_outside_the_checkpoint_identity():
    env = G1VelocityEnv(real_robot=True, phase_stop=True, history=5)
    try:
        for seed in range(5):
            env.reset(seed=seed)
            assert env.phase_steps == 0
    finally:
        env.close()
    saved = task_identity('g1_29dof', 'g1_velocity', RECIPE)
    with_start = {**RECIPE, 'random_phase': True}
    assert task_identity('g1_29dof', 'g1_velocity', with_start) == saved
    validate_resume(saved, task_identity('g1_29dof', 'g1_velocity', with_start))


def test_last_iteration6_stage_resumes_with_random_phase():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    run = Path(__file__).resolve().parents[1] / 'runs' / 'humanoid_lab' / 'exp-7f2494695bab436f'
    if not run.exists() or not list_stages(run):
        pytest.skip('iteration 6 run is not on this machine')
    metadata = read_json(list_stages(run)[-1] / 'metadata.json')
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'random_phase': True})
    validate_resume(checkpoint_identity(metadata), requested)


def test_status_write_failure_does_not_stop_training(tmp_path, monkeypatch, capsys):
    from humanoid_sim import train

    class Model:
        num_timesteps = 7

    callback = train.StageCallback.__new__(train.StageCallback)
    callback.run_dir, callback.start, callback.initial_steps = tmp_path, 0., 0
    callback.target, callback.duration_seconds, callback.timed_out = 10, None, False
    monkeypatch.setattr(type(callback), 'model', Model(), raising=False)

    def locked(path, value):
        raise PermissionError(5, 'Access is denied', str(path))

    monkeypatch.setattr(train, 'write_json', locked)
    callback.status('training')  # a reader holding status.json open must not kill the trainer
    assert 'status.json not updated' in capsys.readouterr().out
    monkeypatch.undo()
    callback_model = Model()
    monkeypatch.setattr(type(callback), 'model', callback_model, raising=False)
    callback.status('training')
    assert (tmp_path / 'status.json').exists() and os.path.getsize(tmp_path / 'status.json') > 0
