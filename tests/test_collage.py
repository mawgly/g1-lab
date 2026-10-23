import numpy as np
import pytest

from humanoid_sim import collage
from humanoid_sim.core import save_stage, write_json

G1 = dict(robot_id='g1_29dof', task_id='g1_velocity', control_dt=.02)


@pytest.fixture(scope='module')
def g1_stage(tmp_path_factory):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    torch.set_num_threads(2)
    run = tmp_path_factory.mktemp('g1')
    write_json(run / 'run_config.json', dict(robot_id=G1['robot_id'], task_id=G1['task_id']))
    env = VecNormalize(DummyVecEnv([G1VelocityEnv]))
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=3)
    stage = save_stage(run, model, env)
    env.close()
    return stage


def test_collage_has_a_row_per_view_and_a_tile_per_time(g1_stage):
    sheet = collage.collage([('g1', g1_stage)], 'walk', times=(.04, .1), views=('side', 'front'))
    assert sheet.size == (collage.HEADER + 3 * collage.TILE[0], 2 * collage.TILE[1])  # two moments + ground path
    tile = np.asarray(sheet)[:collage.TILE[1] - 40, collage.HEADER:collage.HEADER + collage.TILE[0]]
    assert tile.std() > 10  # a rendered scene, not an empty tile


def test_tiles_after_a_fall_repeat_the_fall_frame(monkeypatch):
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    env = G1VelocityEnv()
    env.reset(seed=0)

    def falls_at_third_step(stage, identity, policy, norm, scenario, seed, seconds, on_step):
        for now in (.02, .04, .06):
            on_step(now, env, 1., now > .05)

    monkeypatch.setattr(collage, 'rollout', falls_at_third_step)
    try:
        frames, path, _ = collage.film(None, G1, None, None, None, 0, times=(.04, .1, .2))
    finally:
        env.close()
    assert len(path) == 3
    assert [fallen for _, _, fallen in frames['side']] == [False, True, True]
    assert frames['side'][1][1][0] == 'упал на 0.1 с'
    assert frames['side'][0][1][0] == '0.04 с · корпус +1°'
