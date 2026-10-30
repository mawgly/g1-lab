"""task_config terrain (stage 5, rough ground blind): the heightfield, the start on a tile, the reward's heights above
the ground under the body, the curriculum's equations, the identity. Without the setting the task is the plane alone
(the walking digests of tests/test_g1_getup_gate.py and tests/test_g1_arm_targets.py keep it bit for bit)."""
import mujoco
import numpy as np
import pytest
import torch
from humanoid_sim.tasks import g1_terrain, g1_tensor_math as task
from humanoid_sim.tasks.config import (TERRAIN_DEFAULTS, TERRAIN_LEVELS, TERRAIN_SEED, TERRAIN_SEED_SUFFIX,
                                       TERRAIN_SUFFIX, g1_config, g1_config_identity, terrain_revision, terrain_seed,
                                       terrain_setting)
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv, load_model, restored_terrain_levels, saved_terrain_level

# Candidate 8's real-robot profile (exp-28691193718e4d0e), the parent of the first terrain branch.
PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, lock_wrists=True,
               knee_floor=.061, foot_collision=True, action_limit=10., heading_hold=True, stand_heading_hold=False,
               side_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
               reward_weights=dict(feet_flat=2., height=10., fall=100.))


@pytest.fixture(scope='module')
def env():
    raw = G1VelocityEnv(terrain={}, **PROFILE)
    yield raw
    raw.close()


def test_without_the_setting_the_model_and_identity_are_the_plane():
    assert g1_config({})['terrain'] is None and g1_config({'terrain': None})['terrain'] is None
    plain, _ = load_model(False, True, False)
    assert plain.nhfield == 0 and plain.ngeom == load_model(False, True, False, terrain=False)[0].ngeom
    raw = G1VelocityEnv(**PROFILE)
    try:
        assert raw.terrain is None and not raw.keeps_training_state and raw.terrain_type is None
        raw.reset(seed=3)
        assert raw.data.qpos[:2].tolist() == [0., 0.] and raw.data.qpos[2] == .8
        info = raw.step(np.zeros(raw.model.nu))[-1]
        assert 'terrain_type' not in info and 'pelvis_height' not in info
    finally:
        raw.close()
    identity = {'task_id': 'g1_velocity', 'model_revision': 'm', 'action_schema': 'a', 'observation_schema': 'o'}
    assert g1_config_identity(identity, g1_config({}))['model_revision'] == 'm'
    assert g1_config_identity(identity, g1_config({'terrain': {}}))['model_revision'] == 'm' + TERRAIN_SUFFIX


def test_settings_are_validated():
    assert terrain_setting({}) == TERRAIN_DEFAULTS
    assert terrain_setting({'up': 1, 'slow_level': 10})['slow_level'] == 10
    for bad in ({'up': 0}, {'down': .8}, {'start_level': 10}, {'slow_level': 1.5}, {'slow_speed': 0.},
                {'levels': 3}, [1], 'flat', {'up': float('nan')}):
        with pytest.raises(ValueError):
            terrain_setting(bad)


def test_the_heights_are_mujocos_surface(env):
    """task.terrain_height against mj_ray on the heightfield: the same triangles, 2000 points over the grid; outside it
    the plane at 0."""
    rng = np.random.default_rng(0)
    data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, data)
    rows, columns = env.terrain_heights.shape
    corner, cell = env.terrain_corner, env.terrain_cell
    points = np.stack((corner[0] + rng.uniform(0, (columns - 1) * cell, 2000),
                       corner[1] + rng.uniform(0, (rows - 1) * cell, 2000)), axis=1)
    found = np.zeros(1, np.int32)
    hits = [2. - mujoco.mj_ray(env.model, data, np.array([*p, 2.]), np.array([0., 0., -1.]), None, 1, -1, found)
            for p in points]
    np.testing.assert_allclose(env._ground(points).numpy(), hits, atol=1e-9, rtol=0)
    assert float(env._ground(np.array([[0., 0.], [corner[0] - 1, corner[1] + 5]])).abs().max()) == 0
    normal = task.terrain_normal(torch.tensor([[0., 0.]], dtype=torch.float64), env.terrain_heights, corner, cell)
    assert normal.tolist() == [[0., 0., 1.]]


def test_each_tile_is_its_own_heightfield_centred_on_it(env):
    """One heightfield geom per tile with its frame at the tile's centre (MJWarp's float32 heightfield contacts went
    metres deep 40 m from one grid-wide heightfield's centre, g1_terrain's docstring); together they are the grid."""
    geoms = g1_terrain.terrain_geoms(env.model)
    centres = g1_terrain.tile_centres()
    assert len(geoms) == env.model.nhfield == centres.shape[0] * centres.shape[1]
    for level in range(TERRAIN_LEVELS):
        for column in range(len(g1_terrain.TERRAIN_TYPES)):
            geom = env.model.geom(g1_terrain.tile_name(level, column))
            np.testing.assert_allclose(geom.pos, [*centres[level, column], g1_terrain.LIFT], atol=1e-12)
            assert env.model.hfield_size[geom.dataid[0]][:2].tolist() == [g1_terrain.TILE / 2] * 2
    np.testing.assert_allclose(env.terrain_heights.numpy(), g1_terrain.terrain_grid(), atol=1e-7, rtol=0)
    assert env.terrain_corner == pytest.approx(g1_terrain.TERRAIN_CORNER) and env.terrain_cell == pytest.approx(
        g1_terrain.CELL)


def test_tiles_meet_without_cliffs_and_reach_their_difficulty():
    grid = g1_terrain.terrain_grid()
    nodes = g1_terrain.NODES
    assert grid[::nodes, :].max() == 0 and grid[:, ::nodes].max() == 0  # every tile border at 0
    top = TERRAIN_LEVELS - 1
    tile = {kind: grid[top * nodes:(top + 1) * nodes + 1, k * nodes:(k + 1) * nodes + 1]
            for k, kind in enumerate(g1_terrain.TERRAIN_TYPES)}
    assert tile['flat'].max() == 0
    assert .05 < tile['rough'].max() <= .06 and .05 < tile['boxes'].max() <= .06
    steps = np.unique(np.round(tile['stairs_down'][nodes // 2, :], 6))
    np.testing.assert_allclose(np.diff(steps), .08, atol=1e-9)  # 8 cm steps at the top level
    slope = tile['slope_down'][nodes // 2, :]
    assert np.degrees(np.arctan(np.abs(np.diff(slope)).max() / g1_terrain.CELL)) == pytest.approx(15., abs=1e-6)


def test_another_seed_is_another_map_of_the_same_tiles():
    """terrain_grid(seed): the same borders, platforms and flat tiles, other random heights, the same height bounds."""
    first, second = g1_terrain.terrain_grid(), g1_terrain.terrain_grid(g1_terrain.TERRAIN_SEED + 1)
    assert first.shape == second.shape and not np.array_equal(first, second)
    nodes = g1_terrain.NODES
    assert second[::nodes, :].max() == 0 and second[:, ::nodes].max() == 0
    centres = slice(nodes // 2 - 10, nodes // 2 + 11)  # inside every tile's spawn platform
    for level in range(TERRAIN_LEVELS):
        for column in range(len(g1_terrain.TERRAIN_TYPES)):
            tile = [grid[level * nodes:(level + 1) * nodes + 1, column * nodes:(column + 1) * nodes + 1]
                    for grid in (first, second)]
            assert np.ptp(tile[1][centres, centres]) == 0
            assert tile[1].max() <= max(tile[0].max() * 1.35, .06)  # stairs: another tread, same step height
    assert np.array_equal(g1_terrain.terrain_grid(), first)  # the task's map is fixed by TERRAIN_SEED


def test_every_start_on_a_tile_is_above_the_ground(env):
    """Each type at levels 0, 4 and 9 from the reset options, and 20 training resets: the robot at the tile's centre,
    every sole sphere above the ground, nothing touching it, the pelvis as high above the platform as above the
    plane."""
    centres = g1_terrain.tile_centres()
    starts = [dict(terrain_type=kind, terrain_level=level, command=[.5, 0., 0.])
              for kind in g1_terrain.TERRAIN_TYPES for level in (0, 4, 9)] + [{}] * 20
    for seed, options in enumerate(starts):
        env.reset(seed=seed, options=options)
        assert env.terrain_type is not None
        np.testing.assert_allclose(env.data.qpos[:2], centres[env.tile_level, env.terrain_type])
        sole = env._sole_height()[0].numpy()
        assert (sole > .01).all(), (options, sole)
        ground = {*g1_terrain.terrain_geoms(env.model), env.model.geom('floor').id}
        assert not any(set(contact.geom) & ground for contact in env.data.contact[:env.data.ncon])
        assert env.pelvis_height() == pytest.approx(.8, abs=1e-9)
    env.reset(seed=1, options={'command': [.5, 0., 0.]})  # a reset with a command alone: the plane at the origin
    assert env.terrain_type is None and env.data.qpos[:3].tolist() == [0., 0., .8]


def test_training_resets_draw_tiles_from_their_own_stream(env):
    """The tile types follow TERRAIN_SHARES; the task's own draws (command, pose) are those of the plane task."""
    plain = G1VelocityEnv(**PROFILE)
    env.set_terrain_level(0)
    try:
        types = []
        for seed in range(400):
            env.reset(seed=seed)
            plain.reset(seed=seed)
            types.append(env.terrain_type)
            if seed < 20:
                np.testing.assert_array_equal(env.command, plain.command)  # level 0: no speed cap
                np.testing.assert_array_equal(env.data.qpos[7:], plain.data.qpos[7:])
        shares = np.bincount(types, minlength=len(g1_terrain.TERRAIN_TYPES)) / len(types)
        np.testing.assert_allclose(shares, g1_terrain.TERRAIN_SHARES, atol=.06)
    finally:
        plain.close()


def tread(env, kind='stairs_down', level=TERRAIN_LEVELS - 1):
    """(x, y, height) of a point on the tile's +x axis with the same height 0.1 m behind and 0.15 m ahead: the middle
    of a stair tread (a ridge's or a pyramid's), at least 0.1 m up."""
    centre = g1_terrain.tile_centres()[level, g1_terrain.TERRAIN_TYPES.index(kind)]
    for x in np.arange(1.2, 3.8, .01):
        span = np.stack((centre[0] + x + np.linspace(-.1, .15, 26), np.full(26, centre[1])), axis=1)
        heights = env._ground(span).numpy()
        if heights.min() > .1 and np.ptp(heights) == 0:
            return centre[0] + x, centre[1], float(heights[0])
    raise AssertionError('no tread found')


def test_a_sole_on_a_stair_tread_is_in_contact(env):
    """The robot standing in home on a tread 0.3-0.6 m up (stairs at the top level): MuJoCo's sole spheres touch the
    heightfield, the reward's sole heights above the tread count both feet as support and the ankle_roll origins stand
    as high above it as above the plane, while their world heights would have missed it."""
    x, y, height = tread(env)
    env.reset(seed=0, options={'command': [0., 0., 0.]})
    env.data.qpos[:2] = x, y
    env.data.qpos[env.qadr] = env.home
    mujoco.mj_kinematics(env.model, env.data)
    env.data.qpos[2] += -.0005 - float(env._sole_height().min())
    mujoco.mj_forward(env.model, env.data)
    for _ in range(10):
        info = env.step(np.zeros(env.model.nu))[-1]
    sole = env._sole_height()[0].numpy()
    assert (sole < task.SOLE_CONTACT_HEIGHT).all() and (sole > -.003).all()
    assert (env.data.geom_xpos[env.soles, 2].min(axis=1) > height).all()  # the world heights miss the support
    terrain = set(g1_terrain.terrain_geoms(env.model))
    touching = {int(g) for c in env.data.contact[:env.data.ncon] for g in c.geom if terrain & set(c.geom)}
    assert set(env.soles.ravel()) & touching
    terms = env._ground_terms()
    np.testing.assert_allclose(terms['foot_height'].numpy(), .034, atol=.004)  # as on the plane (CONTACT_HEIGHT)
    assert not info['fell'] and info['pelvis_height'] == pytest.approx(.78, abs=.02)
    assert info['reward_components']['gait'] == pytest.approx(.5)  # both feet down, as a stand command asks


def test_feet_flat_follows_the_slope(env):
    """On a 15° slope a sole lying on it is flat for feet_flat (the ground's normal), level it is tilted by 15°."""
    centre = g1_terrain.tile_centres()[TERRAIN_LEVELS - 1, g1_terrain.TERRAIN_TYPES.index('slope_down')]
    xy = torch.tensor([[[centre[0] + 2.5, centre[1]], [centre[0] + 2.5, centre[1] + .2]]], dtype=torch.float64)
    normal = task.terrain_normal(xy, env.terrain_heights, env.terrain_corner, env.terrain_cell)
    angle = np.radians(15.)
    on_slope = torch.tensor([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]],
                            dtype=torch.float64)  # pitched down along +x, as the slope descends there
    level = torch.eye(3, dtype=torch.float64)
    zeros = torch.zeros((1, 29), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)[None]
    weights = {**task.G1_REWARD_WEIGHTS, 'feet_flat': 1.}
    for rotation, expected in ((on_slope, 0.), (level, np.sin(angle) ** 2)):
        foot = rotation.expand(1, 2, 3, 3)
        for ground, tilt in ((normal, 2 * expected), (None, 2 * float(rotation[2, :2].square().sum()))):
            components, _ = task.reward_components(
                torch.zeros((1, 6), dtype=torch.float64), eye, torch.tensor([.78], dtype=torch.float64),
                torch.zeros((1, 3), dtype=torch.float64), zeros, zeros, zeros,
                torch.full((1, 2), .034, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64), zeros, zeros[0], eye, weights=weights, foot_rotation=foot,
                floor_normal=ground)
            np.testing.assert_allclose(components['feet_flat'].numpy(), -tilt, atol=1e-6)  # the heightfield's float32


def test_curriculum_equations():
    rotation = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    command = torch.tensor([[1., 0., 0.], [0., -.5, 0.], [0., 0., .5], [1., 0., 0.]], dtype=torch.float64)
    displacement = torch.tensor([[.02, .01], [0., -.01], [.01, 0.], [-.02, 0.]], dtype=torch.float64)
    commanded, walked = task.terrain_progress(displacement, rotation, command, .02)
    np.testing.assert_allclose(commanded.numpy(), [.02, .01, 0., .02])
    np.testing.assert_allclose(walked.numpy(), [.02, .01, 0., 0.])  # along the command, never backwards
    turned = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)[None]  # facing +y
    _, along = task.terrain_progress(torch.tensor([[0., .02]], dtype=torch.float64), turned, command[:1], .02)
    assert along.tolist() == pytest.approx([.02])
    level = torch.tensor([0, 3, 3, 3, 9, 5])
    commanded = torch.tensor([10., 10., 10., 1., 10., 10.])
    walked = torch.tensor([9., 4., 6., 0., 9., 9.])
    fell = torch.tensor([False, False, False, False, False, True])
    top = torch.full((6,), 2)
    new = task.terrain_level_update(level, commanded, walked, fell, top, .8, .5, TERRAIN_LEVELS)
    assert new.tolist() == [1, 2, 3, 3, 2, 4]  # up, down, kept, too short to judge, past the top, fell
    capped = task.terrain_command_cap(torch.tensor([[1.2, .9, .3], [1.2, .9, .3], [.3, 0., .3]]),
                                      torch.tensor([5, 4, 9]), 5, .5)
    np.testing.assert_allclose(capped.numpy(), [[.4, .3, .3], [1.2, .9, .3], [.3, 0., .3]], atol=1e-7)
    types = task.terrain_types(4096, g1_terrain.TERRAIN_SHARES)
    np.testing.assert_allclose(np.bincount(types.numpy()) / 4096, g1_terrain.TERRAIN_SHARES, atol=1 / 4096)


def test_training_state_keeps_the_levels(env):
    env.set_terrain_level(4)
    env.reset(seed=2)
    assert env.keeps_training_state and env.tile_level == 4
    assert env.training_state() == {'terrain_level': [4], 'terrain_type': [env.terrain_type]}
    state = {'terrain_level': [0, 1, 2, 7, 8, 9], 'terrain_type': [0, 0, 0, 3, 3, 3]}
    np.testing.assert_array_equal(restored_terrain_levels(state, [0, 0, 3, 3, 3, 1]), [0, 1, 7, 8, 9, 0])
    assert saved_terrain_level(state) == 4
    env.restore_training_state(state)
    assert env.terrain_level == 4
    for bad in ({'terrain_level': [10]}, {'terrain_level': [1.5]}, {'terrain_level': []}):
        with pytest.raises(ValueError):
            saved_terrain_level(bad)
    with pytest.raises(ValueError):
        env.set_terrain_level(TERRAIN_LEVELS)
    env.set_terrain_level(0)


def test_a_high_level_caps_the_training_speed(env):
    env.set_terrain_level(7)
    try:
        speeds = []
        for seed in range(30):
            env.reset(seed=seed)
            speeds.append(np.linalg.norm(env.command[:2]))
        assert max(speeds) == pytest.approx(.5) and min(speeds) < .5
        env.reset(seed=1, options={'command': [1.5, 0., 0.], 'terrain_type': 'rough', 'terrain_level': 9})
        assert env.command[0] == 1.5  # a set command is never capped
    finally:
        env.set_terrain_level(0)


def test_carry_over_crosses_the_terrain_suffix():
    from humanoid_sim.warm_start import carried_fields
    old = {'model_revision': 'abc-footcol'}
    assert 'model_revision' in carried_fields(old, {'model_revision': 'abc-footcol' + TERRAIN_SUFFIX})
    assert 'model_revision' not in carried_fields(old, {'model_revision': 'abd-footcol' + TERRAIN_SUFFIX})


# md5 of the tiles' hfield_data with terrain {} before task_config terrain had a seed (9ac27b3)
OLD_HFIELD_DIGEST = '8118e9ee42cf3df4b49faa8417e20799'


def test_without_a_seed_the_map_is_the_old_one_bit_for_bit():
    """terrain's seed missing, or TERRAIN_SEED: the old map, the old normalized setting (no seed key) and the old model
    revision; another seed is kept and gets its own suffix; bad seeds are refused."""
    import hashlib
    model, _ = load_model(False, True, False, terrain=True)
    assert hashlib.md5(model.hfield_data.tobytes()).hexdigest() == OLD_HFIELD_DIGEST
    assert 'seed' not in terrain_setting({}) and terrain_setting({'seed': TERRAIN_SEED}) == TERRAIN_DEFAULTS
    assert terrain_setting({'seed': 6}) == {**TERRAIN_DEFAULTS, 'seed': 6}
    identity = {'task_id': 'g1_velocity', 'model_revision': 'm', 'action_schema': 'a', 'observation_schema': 'o'}
    revision = lambda terrain: g1_config_identity(identity, g1_config({'terrain': terrain}))['model_revision']
    assert revision({}) == revision({'seed': TERRAIN_SEED}) == 'm' + TERRAIN_SUFFIX
    assert revision({'seed': 6}) == f'm{TERRAIN_SUFFIX}{TERRAIN_SEED_SUFFIX}6' == 'm-terrain2-seed6'
    assert terrain_revision({'seed': 12}) == '-terrain2-seed12' and terrain_seed(None) == TERRAIN_SEED
    for bad in (-1, 1.5, 6., True, '6', 2 ** 31, None):
        with pytest.raises(ValueError):
            terrain_setting({'seed': bad})


def test_another_seed_is_another_heightfield_of_the_same_tiles():
    """terrain {'seed': 6}: the model's heightfields are terrain_grid(6), not the old map; the same tiles (names,
    centres, sizes; flat tiles, borders and platforms flat as before), and the task starts on the same tiles at the
    new map's platform heights."""
    old, _ = load_model(False, True, False, terrain=True)
    new, _ = load_model(False, True, False, terrain=True, terrain_seed=6)
    heights = g1_terrain.model_heights(new)
    np.testing.assert_allclose(heights[0], g1_terrain.terrain_grid(6), atol=1e-7, rtol=0)
    assert not np.array_equal(old.hfield_data, new.hfield_data)
    assert heights[1:] == g1_terrain.model_heights(old)[1:]
    assert g1_terrain.terrain_geoms(old) == g1_terrain.terrain_geoms(new)
    for field in ('geom_pos', 'geom_size', 'geom_type', 'hfield_size', 'hfield_nrow', 'hfield_ncol', 'hfield_adr'):
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field), err_msg=field)
    nodes = g1_terrain.NODES
    for level in range(TERRAIN_LEVELS):
        tile = heights[0][level * nodes:(level + 1) * nodes + 1, :nodes + 1]  # flat, type 0
        assert tile.max() == 0
    raw = G1VelocityEnv(terrain={'seed': 6}, **PROFILE)
    try:
        assert raw.terrain['seed'] == 6
        np.testing.assert_allclose(raw.terrain_heights.numpy(), g1_terrain.terrain_grid(6), atol=1e-7, rtol=0)
        raw.reset(seed=0, options={'command': [.5, 0., 0.], 'terrain_type': 'stairs_down', 'terrain_level': 9})
        np.testing.assert_allclose(raw.data.qpos[:2], g1_terrain.tile_centres()[9, 4])
        assert raw.pelvis_height() == pytest.approx(.8, abs=1e-9)
        assert (raw._sole_height()[0].numpy() > .01).all()
    finally:
        raw.close()


def test_carried_fields_cross_a_change_of_seed():
    """The seeded terrain suffix is one suffix: a seed may change alone, or terrain come or go with any seed; never
    together with another carried suffix."""
    from humanoid_sim.warm_start import carried_fields, model_suffixes
    assert model_suffixes('abc-footcol-terrain2-seed6') == ('abc', ['-footcol', '-terrain2-seed6'])
    crossed = lambda old, new: 'model_revision' in carried_fields({'model_revision': old}, {'model_revision': new})
    for old, new in (('abc-footcol-terrain2', 'abc-footcol-terrain2-seed6'),
                     ('abc-footcol-terrain2-seed6', 'abc-footcol-terrain2-seed7'),
                     ('abc-footcol', 'abc-footcol-terrain2-seed6')):
        assert crossed(old, new) and crossed(new, old), (old, new)
    for old, new in (('abc', 'abc-footcol-terrain2-seed6'),  # two changes at once
                     ('abc-terrain2', 'abc-footcol-terrain2-seed6'),
                     ('abd-footcol-terrain2', 'abc-footcol-terrain2-seed6'),  # another model
                     ('abc-footcol-terrain2-seed6', 'abc-footcol-terrain2-seed6-terrain2')):
        assert not crossed(old, new), (old, new)


def test_carry_over_moves_a_terrain_stage_to_a_new_seed(tmp_path):
    """terrain_next.py's path: a terrain stage (with the worlds' levels in its training_state) carried to another
    seed keeps its weights, normalization and training_state, gets the seeded revision and loads on the new map;
    carried to its own seed it is refused (an ordinary branch)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import load_stage, read_json, save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    from humanoid_sim.warm_start import carry_over

    def run(name, terrain):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                      task_config={**PROFILE, 'terrain': terrain}))
        (tmp_path / name).mkdir()
        write_json(tmp_path / name / 'run_config.json', config)
        return tmp_path / name, config

    folder, config = run('old', {})
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    state = {'terrain_level': [0, 3, 5, 9], 'terrain_type': [0, 2, 4, 6]}
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        model.learn(16)
        stage = save_stage(folder, model, env, state)
    finally:
        env.close()
    target, _ = run('seed6', {'seed': 6})
    moved = carry_over(stage, target)
    saved, metadata = read_json(stage / 'metadata.json'), read_json(moved / 'metadata.json')
    assert metadata['model_revision'] == saved['model_revision'] + '-seed6'
    assert metadata['training_state'] == state
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        np.testing.assert_allclose(new_norm.venv.envs[0].unwrapped.terrain_heights.numpy(),
                                   g1_terrain.terrain_grid(6), atol=1e-7, rtol=0)
    finally:
        old_norm.close(), new_norm.close()
    again, _ = run('seed7', {'seed': 7})
    assert read_json(carry_over(moved, again) / 'metadata.json')['training_state'] == state
    same, _ = run('same', {'seed': 6})
    with pytest.raises(ValueError):
        carry_over(moved, same)
    assert not any(same.glob('stage_*'))
