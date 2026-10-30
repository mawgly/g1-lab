"""task_config height_scan (30.10, stage 5's height-map teacher): the ground's heights in the observation, a grid ahead
for the actor (minus its mean, clipped, noised in training only) and a grid around the pelvis for the critic (above the
feet's mean ground, exact), both in the pelvis yaw frame. Missing or null is the old task bit for bit (digests of the
code before the setting); the values against terrain_height by hand on a step and a pit; the noise only on the actor's
scan and only in a training episode; insert_terms moves a stage there acting as before; the mirror maps the grids onto
themselves, checked against the physics; the export refuses it."""
import math

import mujoco
import numpy as np
import pytest
import torch
from humanoid_sim.symmetry import Mirror
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import g1_config, g1_config_identity, height_scan_setting
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

from test_g1_terrain_pelvis_support import PROFILE, walking_digest

# md5 of walking_digest before task_config height_scan (37da25d): the parent's pelvis_support, and generator 3's map
OLD_SUPPORT_DIGEST = '9c1507b53e9693454404da457bf152ec'
OLD_GENERATOR_DIGEST = '9e2cb777c5cd90fdbeacf3bdae6be5d9'
SMALL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=False,
             gait={'period': .6, 'stance_fraction': .55})


def test_without_the_setting_the_task_is_the_old_one_bit_for_bit():
    assert g1_config({})['height_scan'] is None and height_scan_setting(None) is None
    assert walking_digest({'pelvis_support': True}) == OLD_SUPPORT_DIGEST
    assert walking_digest({'generator': 3, 'seed': 8, 'pelvis_support': True}) == OLD_GENERATOR_DIGEST
    identity = {'task_id': 'g1_velocity', 'model_revision': 'm', 'action_schema': 'a', 'observation_schema': 'o'}
    base = dict(PROFILE, terrain={})
    old = g1_config_identity(identity, g1_config(base))
    assert old['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-critic4'
    assert old == g1_config_identity(identity, g1_config({**base, 'height_scan': None}))
    new = g1_config_identity(identity, g1_config({**base, 'height_scan': {}}))
    assert new['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hscan165-critic4'
    assert {key: value for key, value in new.items() if key != 'observation_schema'} == {
        key: value for key, value in old.items() if key != 'observation_schema'}
    noisy = g1_config_identity(identity, g1_config({**base, 'height_scan': {'noise': 0.}}))
    assert noisy == new  # the noise is training only
    other = g1_config_identity(identity, g1_config({**base, 'height_scan': {'x': [0., 1.4]}}))
    assert other['observation_schema'].endswith('-hscan165-x0_1.4-y0.5-s0.1-critic4')  # same count, another grid


def test_the_setting_is_checked():
    assert height_scan_setting({}) == {'x': [-.2, 1.2], 'y': .5, 'step': .1, 'noise': .1}
    for bad in ([], {'x': [1., 0.]}, {'x': [0., 1.25]}, {'step': .01}, {'y': -.1}, {'y': .55}, {'noise': .6},
                {'radius': 1.}, {'x': [-2., 3.], 'y': 2., 'step': .05}):
        with pytest.raises(ValueError):
            height_scan_setting(bad)
    for missing in ({'real_robot': False, 'critic_extras': False, 'terrain': {}},
                    {**PROFILE, 'critic_extras': False, 'terrain': {}}, {**PROFILE}):
        with pytest.raises(ValueError, match='height_scan requires'):
            g1_config({**missing, 'height_scan': {}})
    with pytest.raises(ValueError, match='height_scan requires'):
        G1VelocityEnv(height_scan={}, **PROFILE)


HEIGHTS = torch.zeros((61, 61), dtype=torch.float64)  # 3 m × 3 m, cell 0.05, corner (−1, −1)
CORNER, CELL, LIFT = (-1., -1.), .05, .002


def test_a_step_and_a_pit_by_hand():
    """A 0.2 m step at x ≥ 0.5 (node 30) and a 0.15 m pit in it at 0.8 ≤ x ≤ 0.9 (nodes 36-38): the actor's grid
    (x −0.2…1.2 every 0.1, y ±0.5) from a pelvis at the origin facing +x reads heights there minus their mean; turned
    90° the same grid lies along +y, where the floor is flat. The critic's: the ground minus the feet's (lower)
    floor."""
    heights = HEIGHTS.clone()
    heights[:, 30:] = .2
    heights[:, 36:39] = .05  # the pit, 0.15 below the step
    grid = task.scan_grid([-.2, 1.2], .5, .1)
    assert grid.shape == (165, 2) and grid[:11, 0].tolist() == [-.2] * 11
    assert grid[:11, 1].tolist() == pytest.approx([-.5 + .1 * k for k in range(11)])
    position, upright = torch.zeros((1, 3), dtype=torch.float64), torch.eye(3, dtype=torch.float64)[None]
    scan = task.height_scan(position, upright, grid, heights, CORNER, CELL, LIFT)[0].reshape(15, 11)
    x = -.2 + .1 * np.arange(15)
    ground = np.where(x >= .5 - 1e-9, .2, 0.) + LIFT
    ground[(x >= .8 - 1e-9) & (x <= .9 + 1e-9)] = .05 + LIFT
    np.testing.assert_allclose(scan.numpy(), np.repeat((ground - ground.mean())[:, None], 11, axis=1), atol=1e-12)
    assert float(scan[7, 0] - scan[6, 0]) == pytest.approx(.2)  # the step up between x = 0.4 and 0.5
    assert float(scan[10, 0] - scan[9, 0]) == pytest.approx(-.15)  # the pit from x = 0.8
    yaw = math.pi / 2  # facing +y from x = −0.2: the grid's x runs along world y, its y along world −x (−0.7…0.3)
    turned = torch.tensor([[[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]]],
                          dtype=torch.float64)
    behind = torch.tensor([[-.2, 0., .8]], dtype=torch.float64)
    assert task.height_scan(behind, turned, grid, heights, CORNER, CELL, LIFT).abs().max() < 1e-12
    shifted = torch.tensor([[.3, .2, .8]], dtype=torch.float64)  # 0.3 ahead: the pit under x = 0.5…0.6 of the grid
    points = task.scan_points(shifted, upright, grid)
    torch.testing.assert_close(points[0], grid + torch.tensor([.3, .2], dtype=torch.float64))
    feet = torch.tensor([[[.25, .08], [.27, .32]]], dtype=torch.float64)  # both on the lower floor
    critic = task.critic_scan(shifted, upright, feet, task.critic_scan_grid(), heights, CORNER, CELL, LIFT)[0]
    by_hand = task.terrain_height(task.scan_points(shifted, upright, task.critic_scan_grid()), heights, CORNER, CELL,
                                  LIFT)[0] - LIFT
    torch.testing.assert_close(critic, by_hand)
    assert critic.reshape(15, 15)[-1].tolist() == pytest.approx([.2] * 15)  # x = 1.0 in the world: on the step
    assert float(critic.min()) == pytest.approx(0.)


def test_the_scans_on_a_stairs_tile_follow_terrain_height():
    """On the top stairs-up tile, off the platform (the setting's grids at work, a reset with a command: exact): the
    actor's columns are terrain_height under points turned by the pelvis yaw by hand, minus their mean; the critic's
    above the feet's mean ground; neither is in the history; on the plane at the origin both read 0."""
    raw = G1VelocityEnv(terrain={'pelvis_support': True}, height_scan={}, **PROFILE)
    try:
        assert raw.observation_space.shape == (490 + 165 + 4 + 225,) and raw.actor_inputs == 655
        actor, critic = raw.scan_columns
        assert actor == list(range(490, 655)) and critic == list(range(659, 884))
        observation, _ = raw.reset(seed=0, options={'command': [.5, 0., 0.]})
        assert not raw.scan_noised and np.abs(observation[490:655]).max() == 0 and np.abs(observation[659:]).max() == 0
        raw.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': 'stairs_up', 'terrain_level': 9})
        raw.data.qpos[0] += 1.6
        yaw = .4
        raw.data.qpos[3:7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        raw.data.qpos[2] += float(raw._ground(raw.data.qpos[:2])) + .02
        mujoco.mj_forward(raw.model, raw.data)
        observation = raw._observation('fill')
        x, y = raw.data.qpos[:2]
        offsets = np.array([[-.2 + .1 * i, -.5 + .1 * j] for i in range(15) for j in range(11)])
        points = np.stack((x + math.cos(yaw) * offsets[:, 0] - math.sin(yaw) * offsets[:, 1],
                           y + math.sin(yaw) * offsets[:, 0] + math.cos(yaw) * offsets[:, 1]), axis=1)
        ground = raw._ground(points).numpy()
        np.testing.assert_allclose(observation[actor], ground - ground.mean(), atol=1e-6)
        assert np.ptp(ground) > .2  # generator 2's top level: 8 cm steps, three of them under the grid
        offsets = np.array([[-.7 + .1 * i, -.7 + .1 * j] for i in range(15) for j in range(15)])
        points = np.stack((x + math.cos(yaw) * offsets[:, 0] - math.sin(yaw) * offsets[:, 1],
                           y + math.sin(yaw) * offsets[:, 0] + math.cos(yaw) * offsets[:, 1]), axis=1)
        support = raw._ground(raw.data.xpos[raw.feet, :2]).numpy().mean()
        np.testing.assert_allclose(observation[critic], np.clip(raw._ground(points).numpy() - support, -1, 1),
                                   atol=1e-6)
        assert observation[658] == pytest.approx(raw.pelvis_height(), abs=1e-6)  # the critic extras' height, before
        raw.step(np.zeros(29))
        assert raw.frames.shape == (1, 5, raw.frame_size) and raw.frame_size * 5 == 490  # no scan in the history
    finally:
        raw.close()


def test_the_noise_is_the_actors_and_the_training_episodes_only():
    raw = G1VelocityEnv(terrain={'pelvis_support': True}, height_scan={'noise': .1}, **PROFILE)
    try:
        actor, critic = raw.scan_columns
        for seed in (1, 2):
            observation, _ = raw.reset(seed=seed)  # a training reset: on a tile, noised
            assert raw.scan_noised and raw.terrain_type is not None
            exact = task.height_scan(*raw._scan_pose(), raw.scan_grid, *raw._terrain_grid())[0].numpy()
            noise = observation[actor] - exact
            assert np.abs(noise).max() <= .1 + 1e-6 and np.abs(noise).max() > .05 and abs(noise.mean()) < .03
            feet = torch.from_numpy(raw.data.xpos[raw.feet, :2].copy())[None]
            critic_exact = task.critic_scan(*raw._scan_pose(), feet, raw.critic_grid, *raw._terrain_grid())[0]
            np.testing.assert_allclose(observation[critic], critic_exact.numpy(), atol=1e-6)
            for _ in range(3):
                observation = raw.step(np.zeros(29))[0]
            exact = task.height_scan(*raw._scan_pose(), raw.scan_grid, *raw._terrain_grid())[0].numpy()
            assert np.abs(observation[actor] - exact).max() > .05  # every observation, fresh
        raw.set_command([.5, 0., 0.])  # evaluation and scenarios: exact from here on
        observation = raw._observation()
        exact = task.height_scan(*raw._scan_pose(), raw.scan_grid, *raw._terrain_grid())[0].numpy()
        np.testing.assert_allclose(observation[actor], exact, atol=1e-6)
        observation, _ = raw.reset(seed=1, options={'command': [.5, 0., 0.], 'terrain_type': 'rough',
                                                    'terrain_level': 9})
        exact = task.height_scan(*raw._scan_pose(), raw.scan_grid, *raw._terrain_grid())[0].numpy()
        assert np.abs(exact).max() > .01
        np.testing.assert_allclose(observation[actor], exact, atol=1e-6)
    finally:
        raw.close()


def test_fixed_normalization_covers_both_scans():
    raw = G1VelocityEnv(terrain={}, height_scan={}, **PROFILE)
    try:
        positions, mean, variance = raw.fixed_normalization
        actor, critic = raw.scan_columns
        assert positions.tolist() == actor + critic and not mean.any()
        assert variance.tolist() == [task.HEIGHT_SCAN_VARIANCE] * 165 + [task.CRITIC_SCAN_VARIANCE] * 225
    finally:
        raw.close()


def test_the_mirror_maps_each_grid_onto_itself():
    grid = task.scan_grid([-.2, 1.2], .5, .1)
    index = task.scan_mirror(grid)
    torch.testing.assert_close(grid[index], grid * torch.tensor([1., -1.], dtype=torch.float64))
    assert sorted(index.tolist()) == list(range(165)) and (index[index] == np.arange(165)).all()
    raw = G1VelocityEnv(terrain={}, height_scan={}, **PROFILE)
    try:
        mirror = Mirror(raw)
        assert len(mirror.obs_index) == 884
        x = torch.randn(3, 884)
        torch.testing.assert_close(mirror.observation(mirror.observation(x)), x)
        actor, critic = raw.scan_columns
        assert mirror.obs_index[actor].tolist() == [490 + int(k) for k in index]
        assert mirror.obs_index[critic].tolist() == [659 + int(k) for k in task.scan_mirror(raw.critic_grid)]
        assert bool((mirror.obs_sign[actor + critic] == 1).all())
    finally:
        raw.close()


def _ground_field(mirrored):
    """A separable ground a(x) + b(y) on a 6 m × 6 m grid around the origin (cell 0.05): exact on both of MuJoCo's
    triangles, so its y-mirror (rows reversed) is the mirror image of the surface itself; b is lopsided."""
    nodes = -3. + .05 * torch.arange(121, dtype=torch.float64)
    along = .12 * torch.sin(2. * nodes)
    across = .1 * nodes + .06 * torch.sin(3. * nodes) + .04 * (nodes > .2)
    heights = along[None, :] + (across.flip(0) if mirrored else across)[:, None]
    return heights - heights.min()


def test_mirrored_state_and_action_give_the_mirrored_observation_with_the_scans():
    """tests/test_symmetry.py's physics check with height_scan: the robot on the plane at the origin (the physics),
    the scans' ground (terrain_heights, as test_g1_terrain_pelvis_support's stairs) a lopsided field for the first
    robot and its mirror image for the second; a mirrored state stepped with the mirrored action gives the mirrored
    observation, scans and critic extras included."""
    first, second = (G1VelocityEnv(terrain={}, height_scan={}, **SMALL) for _ in range(2))
    try:
        for raw, mirrored in ((first, False), (second, True)):
            raw.terrain_heights, raw.terrain_corner = _ground_field(mirrored), (-3., -3.)
        mirror = Mirror(first)
        permutation, signs = mirror.action_index.numpy(), mirror.action_sign.numpy().astype(float)
        rng = np.random.default_rng(0)
        first.reset(seed=3, options={'command': [.6, .2, .3]})
        first.data.qpos[:2] = [.3, .4]
        first.data.qpos[3:7] = [math.cos(.2), 0., 0., math.sin(.2)]
        mujoco.mj_forward(first.model, first.data)
        for _ in range(7):
            first.step(rng.normal(0, .5, 29))
        second.reset(seed=3, options={'command': [.6, -.2, -.3]})
        qpos, qvel = first.data.qpos.copy(), first.data.qvel.copy()
        second.data.qpos[:7] = qpos[0], -qpos[1], qpos[2], qpos[3], -qpos[4], qpos[5], -qpos[6]
        second.data.qpos[second.qadr] = qpos[first.qadr][permutation] * signs
        second.data.qvel[:6] = qvel[0], -qvel[1], qvel[2], -qvel[3], qvel[4], -qvel[5]
        second.data.qvel[second.vadr] = qvel[first.vadr][permutation] * signs
        second.previous_action[:] = first.previous_action[permutation] * signs
        second.steps = first.steps + round(first.gait['period'] / first.dt) // 2
        mujoco.mj_forward(second.model, second.data)
        action = rng.normal(0, .5, 29)
        observed = first.step(action)[0]
        mirrored = second.step(action[permutation] * signs)[0]
        actor, critic = first.scan_columns
        assert np.abs(observed[actor]).max() > .1 and np.abs(observed[critic]).max() > .1  # the field is seen
        assert np.abs(observed[actor] - observed[actor][task.scan_mirror(first.scan_grid)]).max() > .02  # lopsided
        np.testing.assert_allclose(mirrored, mirror.observation(torch.tensor(observed)).numpy(), rtol=1e-2, atol=2e-3)
    finally:
        first.close()
        second.close()


def test_insert_terms_moves_a_terrain_stage_to_the_height_scan_acting_as_before(tmp_path):
    """A walking terrain stage (the parent's layout: history 5, critic extras) moved by insert_terms: stage 0 has
    the new contract, the steps and training_state, zero weights on the 390 new columns of both first layers, their
    fixed normalization, the old columns' statistics as they were, and acts as the source on training resets with
    noised scans (≤ 1e-6); the export refuses it."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim import export_g1
    from humanoid_sim.checkpoint_schema import checkpoint_identity, validate_resume
    from humanoid_sim.core import load_stage, read_json, save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    from humanoid_sim.tasks.identity import task_identity
    from humanoid_sim.warm_start import check_insert_terms, insert_terms, inserted_difference
    old_task = dict(PROFILE, terrain={'pelvis_support': True})

    def run(name, task_config):
        folder = tmp_path / name
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                      task_config=task_config))
        write_json(folder / 'run_config.json', config)
        return folder, config
    source_run, config = run('old', old_task)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490, 'log_std_init': -1})
        rng = np.random.default_rng(0)  # statistics that matter, as a trained stage's
        env.obs_rms.mean = rng.normal(0., .3, 494)
        env.obs_rms.var = rng.uniform(.2, 2., 494)
        env.obs_rms.count = 8.1e9
        stage = save_stage(source_run, model, env, {'terrain_level': [3], 'terrain_type': [1]})
    finally:
        env.close()
    target, config = run('new', {**old_task, 'height_scan': {}})
    saved = read_json(stage / 'metadata.json')
    check_insert_terms(saved, task_identity('g1_29dof', 'g1_velocity', config['task_config']))
    moved = insert_terms(stage, target)
    metadata = read_json(moved / 'metadata.json')
    validate_resume(checkpoint_identity(metadata), task_identity('g1_29dof', 'g1_velocity', config['task_config']))
    assert metadata['observation_schema'].endswith('-hscan165-critic4') and metadata['steps'] == saved['steps']
    assert metadata['training_state'] == {'terrain_level': [3], 'terrain_type': [1]}
    record = read_json(moved / 'warm_start.json')
    assert record['observation_size'] == [494, 884] and record['inserted'] == ['height_scan', 'critic_scan']
    new_columns = list(range(490, 655)) + list(range(659, 884))
    old_columns = list(range(490)) + list(range(655, 659))
    (old_model, old_norm), (new_model, new_norm) = load_stage(stage), load_stage(moved)
    try:
        assert new_model.policy.actor_inputs == 655
        old_rms, new_rms = old_norm.obs_rms, new_norm.obs_rms
        assert new_rms.count == old_rms.count
        np.testing.assert_array_equal(new_rms.mean[old_columns], old_rms.mean)
        np.testing.assert_array_equal(new_rms.var[old_columns], old_rms.var)
        assert not new_rms.mean[new_columns].any()
        np.testing.assert_allclose(new_rms.var[new_columns], [.01] * 390)
        layers = [value for value in new_model.policy.state_dict().values()
                  if value.dim() == 2 and value.shape[1] == 884]
        assert len(layers) == 2 and all(not layer[:, new_columns].any() for layer in layers)
        old_layers = [value for value in old_model.policy.state_dict().values() if value.dim() == 2
                      and value.shape[1] == 494]
        for layer, old_layer in zip(layers, old_layers):
            torch.testing.assert_close(layer[:, old_columns], old_layer, rtol=0, atol=0)
    finally:
        old_norm.close()
        new_norm.close()
    assert inserted_difference(stage, moved, seeds=(0, 1), steps=40) <= 1e-6
    with pytest.raises(ValueError, match='камер'):
        export_g1.export(moved, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()
