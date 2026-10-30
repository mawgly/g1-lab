"""task_config terrain's generator (30.10, the lead after critic it94: a height-map teacher on 15-20 cm stairs, pits and
curbs): missing or 2 is terrain2's map bit for bit (tests/test_g1_terrain.py OLD_HFIELD_DIGEST); 3 is
g1_terrain.GENERATOR_3's grid (2.5 cm cells, stairs 2 -> 20 cm on 0.23-0.31 m treads, pits and curbs), its stairs,
pits, curbs and steep boxes tiles colliding through static boxes (g1_terrain.box_tiles), its own model revision
-terrain3box, which warm_start.carry_over crosses with the tile types mapped by name."""
import hashlib

import mujoco
import numpy as np
import pytest
import torch
from humanoid_sim.tasks import g1_terrain
from humanoid_sim.tasks.config import (TERRAIN_DEFAULTS, TERRAIN_LEVELS, TERRAIN_SUFFIX, g1_config, g1_config_identity,
                                       terrain_generator, terrain_revision, terrain_setting)
from humanoid_sim.tasks.g1_velocity import (G1VelocityEnv, carried_terrain_state, load_model,
                                            restored_terrain_levels)
from test_g1_terrain import OLD_HFIELD_DIGEST, PROFILE

SPEC = g1_terrain.generator(3)
NODES = SPEC.nodes


@pytest.fixture(scope='module')
def env():
    raw = G1VelocityEnv(terrain={'generator': 3}, **PROFILE)
    yield raw
    raw.close()


def tile(grid, kind, level, spec=SPEC):
    nodes = spec.nodes
    column = spec.types.index(kind)
    return grid[level * nodes:(level + 1) * nodes + 1, column * nodes:(column + 1) * nodes + 1]


def test_generator_two_or_missing_is_the_old_map_bit_for_bit():
    assert terrain_setting({'generator': 2}) == terrain_setting({}) == TERRAIN_DEFAULTS
    assert terrain_generator(terrain_setting({})) == terrain_generator(None) == 2
    model, _ = load_model(False, True, False, terrain=True, terrain_generator=2)
    assert hashlib.md5(model.hfield_data.tobytes()).hexdigest() == OLD_HFIELD_DIGEST
    assert np.array_equal(g1_terrain.terrain_grid(5, 2), g1_terrain.terrain_grid())
    assert g1_terrain.generator(2).types == g1_terrain.TERRAIN_TYPES and g1_terrain.generator(2).nodes == 160
    identity = {'task_id': 'g1_velocity', 'model_revision': 'm', 'action_schema': 'a', 'observation_schema': 'o'}
    revision = lambda terrain: g1_config_identity(identity, g1_config({'terrain': terrain}))['model_revision']
    assert revision({'generator': 2}) == 'm' + TERRAIN_SUFFIX
    assert revision({'generator': 3}) == 'm-terrain3box'
    assert revision({'generator': 3, 'seed': 6}) == 'm-terrain3box-seed6'
    assert terrain_revision(terrain_setting({'generator': 3, 'seed': 12})) == '-terrain3box-seed12'
    # generator 2 has no box tiles: every heightfield collides as before
    assert g1_terrain.box_tiles(5, 2) == frozenset() and not g1_terrain.terrain_boxes(model)
    assert g1_terrain.ground_geoms(model) == g1_terrain.terrain_geoms(model)
    assert terrain_setting({'generator': 3}) == {**TERRAIN_DEFAULTS, 'generator': 3}
    for bad in (1, 4, 3., True, '3', None):
        with pytest.raises(ValueError):
            terrain_setting({'generator': bad})


def test_generator_three_tiles_reach_their_difficulty():
    """Per level: stairs of 2 -> 20 cm on treads of 0.23-0.31 m, pits 2 -> 20 cm deep, curbs 2 -> 15 cm high and
    0.15-0.3 m wide; the old rough, boxes and slopes as before; every border 0, every spawn platform flat, nothing
    below 0 or up to the height scale."""
    grid = g1_terrain.terrain_grid(5, 3)
    assert grid.shape == (TERRAIN_LEVELS * NODES + 1, len(SPEC.types) * NODES + 1) and NODES == 320
    assert grid[::NODES, :].max() == 0 and grid[:, ::NODES].max() == 0
    assert grid.min() == 0 and grid.max() < SPEC.elevation == 3.
    centre = slice(NODES // 2 - 36, NODES // 2 + 37)  # ±0.9 m: inside every platform
    for level in range(TERRAIN_LEVELS):
        share = level / (TERRAIN_LEVELS - 1)
        for kind in SPEC.types:
            assert np.ptp(tile(grid, kind, level)[centre, centre]) == 0, (kind, level)
        for kind in ('stairs_up', 'stairs_down'):
            heights = np.unique(np.round(tile(grid, kind, level)[NODES // 2, :], 9))
            np.testing.assert_allclose(np.diff(heights), .02 + .18 * share, atol=1e-9)
            row = tile(grid, kind, level)[NODES // 2, NODES // 2:]  # from the centre out along +x
            edges = np.flatnonzero(np.diff(row) != 0) * SPEC.cell
            # stairs_up: the treads below its ridge (the ridge's top is what is left of the run, any width)
            treads = np.diff(edges)[:3] if kind == 'stairs_up' else np.diff(edges)
            assert ((treads > .23 - SPEC.cell - 1e-9) & (treads < .31 + SPEC.cell + 1e-9)).all(), (kind, treads)
        pits = tile(grid, 'pits', level)
        depth = .02 + .18 * share
        assert pits.max() == pytest.approx(depth, abs=1e-12)
        plateau = np.abs(np.linspace(-4, 4, NODES + 1))
        far = np.maximum(plateau[:, None], plateau[None, :])
        holes = (pits == 0) & (far < 3.5)
        assert holes.any() and (far[holes] > 1.).all()  # pits down to 0, none on the platform or the ramp
        assert np.isin(np.round(pits[far <= 3.5], 12), [0., round(depth, 12)]).all()
        curbs = tile(grid, 'curbs', level)
        assert curbs.max() == pytest.approx(.02 + .13 * share, abs=1e-12)
        row = curbs[NODES // 2, NODES // 2:] > 0
        rises, falls = np.flatnonzero(np.diff(row.astype(int)) == 1), np.flatnonzero(np.diff(row.astype(int)) == -1)
        assert len(rises) >= 2 and len(rises) == len(falls)
        widths = (falls - rises) * SPEC.cell
        assert ((widths > .15 - SPEC.cell - 1e-9) & (widths < .3 + SPEC.cell + 1e-9)).all(), widths
        assert rises[0] * SPEC.cell >= 1. + .5 - SPEC.cell
    top = TERRAIN_LEVELS - 1
    assert .05 < tile(grid, 'rough', top).max() <= .06 and .05 < tile(grid, 'boxes', top).max() <= .06
    slope = tile(grid, 'slope_down', top)[NODES // 2, :]
    assert np.degrees(np.arctan(np.abs(np.diff(slope)).max() / SPEC.cell)) == pytest.approx(15., abs=1e-6)
    assert not np.array_equal(g1_terrain.terrain_grid(6, 3), grid)  # the seed gives another map


def test_generator_three_model_and_heights(env):
    """90 tiles, each its own heightfield centred on its tile; task.terrain_height equals mj_ray on stairs and curbs
    of the top level and anywhere on the grid."""
    geoms = g1_terrain.terrain_geoms(env.model)
    centres = g1_terrain.tile_centres(3)
    assert len(geoms) == env.model.nhfield == TERRAIN_LEVELS * 9 == centres.shape[0] * centres.shape[1]
    assert env.terrain_names == SPEC.types and env.terrain_shares == SPEC.shares
    assert sum(SPEC.shares) == pytest.approx(1.)
    assert env.terrain_corner == pytest.approx(SPEC.corner) == (-36., 40.)
    assert env.terrain_cell == pytest.approx(.025)
    np.testing.assert_allclose(env.terrain_heights.numpy(), g1_terrain.terrain_grid(5, 3), atol=1e-7, rtol=0)
    for level, column in ((9, 8), (0, 0), (9, 4)):
        geom = env.model.geom(g1_terrain.tile_name(level, column))
        np.testing.assert_allclose(geom.pos, [*centres[level, column], g1_terrain.LIFT], atol=1e-12)
    rng = np.random.default_rng(0)
    data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, data)
    points = [centres[9, SPEC.types.index(kind)] + rng.uniform(-4, 4, (300, 2))
              for kind in ('stairs_down', 'curbs', 'pits')]
    rows, columns = env.terrain_heights.shape
    corner, cell = env.terrain_corner, env.terrain_cell
    points.append(np.stack((corner[0] + rng.uniform(0, (columns - 1) * cell, 600),
                            corner[1] + rng.uniform(0, (rows - 1) * cell, 600)), axis=1))
    points = np.concatenate(points)
    found = np.zeros(1, np.int32)
    heightfields = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # group 0: the tiles' heightfields, not the boxes
    hits = [4. - mujoco.mj_ray(env.model, data, np.array([*p, 4.]), np.array([0., 0., -1.]), heightfields, 1, -1,
                               found) for p in points]
    np.testing.assert_allclose(env._ground(points).numpy(), hits, atol=1e-8, rtol=0)


def test_generator_three_box_tiles_collide_through_boxes_of_the_same_ground(env):
    """Stairs, pits and curbs at every level and the boxes tiles whose blocks step more than 5 cm (the top two levels
    here) collide through static boxes, their heightfields with nothing; rough, slopes, flat and the other boxes
    tiles through their heightfield as before. mj_ray on the boxes at cell centres: exactly the reward's ground
    (task.terrain_height) where a cell's four nodes are level, within that cell's lowest and highest node elsewhere
    (the boxes stand at its upper side: within one cell of an edge); every box inside its tile, from LIFT - BASE."""
    model = env.model
    boxed = g1_terrain.box_tiles(5, 3)
    grid = g1_terrain.terrain_grid(5, 3)
    for level in range(TERRAIN_LEVELS):
        for column, kind in enumerate(SPEC.types):
            riser = max(np.abs(np.diff(tile(grid, kind, level), axis=a)).max() for a in (0, 1))
            assert ((level, column) in boxed) == (kind in g1_terrain.BOX_TYPES or riser > g1_terrain.BOX_RISER)
            geom = model.geom(g1_terrain.tile_name(level, column))
            assert (geom.contype[0] == geom.conaffinity[0] == 0) == ((level, column) in boxed)
    assert {SPEC.types[c] for _, c in boxed} == {*g1_terrain.BOX_TYPES, 'boxes'}
    assert sorted(level for level, c in boxed if SPEC.types[c] == 'boxes') == [8, 9]
    boxes = g1_terrain.terrain_boxes(model)
    assert 1000 < len(boxes) < 2000 and g1_terrain.ground_geoms(model) == sorted(
        [i for i in g1_terrain.terrain_geoms(model) if model.geom_contype[i]] + boxes)
    floor = model.geom('floor')
    centres = g1_terrain.tile_centres(3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    owner, tilted = {}, {}
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    for i in boxes:
        level, column, _ = map(int, g1_terrain.BOX_NAME.match(model.geom(i).name).groups())
        owner.setdefault((level, column), []).append(i)
        assert model.geom_contype[i] == floor.contype[0] and model.geom_conaffinity[i] == floor.conaffinity[0]
        assert model.geom_priority[i] == floor.priority[0] and model.geom_group[i] == g1_terrain.BOX_GROUP
        np.testing.assert_allclose(model.geom_friction[i], floor.friction)
        vertices = data.geom_xpos[i] + (corners * model.geom_size[i]) @ data.geom_xmat[i].reshape(3, 3).T
        vertices -= [*centres[level, column], 0.]
        if np.array_equal(model.geom_quat[i], [1., 0., 0., 0.]):  # upright: from LIFT - BASE
            assert vertices[:, 2].min() == pytest.approx(g1_terrain.LIFT - g1_terrain.BASE, abs=1e-12)
        else:  # the pits' ramp: what stands above the floor plane lies in its tile
            tilted[(level, column)] = tilted.get((level, column), 0) + 1
            vertices = vertices[vertices[:, 2] > 0]
        assert (vertices[:, :2] > -4 - 1e-9).all() and (vertices[:, :2] < 4 + 1e-9).all()
    assert set(owner) == boxed
    assert tilted == {(level, SPEC.types.index('pits')): 12 for level in range(TERRAIN_LEVELS)}
    axis = (np.arange(NODES) + .5) * SPEC.cell - 4
    ramp = np.maximum(np.abs(axis)[:, None], np.abs(axis)[None, :]) > 4 - g1_terrain.PIT_RAMP
    for (level, column), ids in owner.items():
        # every cell (but the pits' ramp ring) under exactly one upright box whose top is the cell's height (at the
        # highest of its four nodes; two such tops over one cell would each give a sphere there a contact)
        nodes = tile(g1_terrain.terrain_grid(5, 3), SPEC.types[column], level)
        nodes = (nodes / SPEC.elevation).astype(np.float32).astype(np.float64) * SPEC.elevation
        cells = np.maximum.reduce([nodes[:-1, :-1], nodes[1:, :-1], nodes[:-1, 1:], nodes[1:, 1:]])
        tops = np.zeros(cells.shape, int)
        for i in ids:
            if model.geom_quat[i][0] != 1.:
                continue
            low = model.geom_pos[i] - model.geom_size[i] - [*centres[level, column], 0.]
            high = model.geom_pos[i] + model.geom_size[i] - [*centres[level, column], 0.]
            inside = np.ix_((axis > low[1]) & (axis < high[1]), (axis > low[0]) & (axis < high[0]))
            tops[inside] += np.abs(cells[inside] + g1_terrain.LIFT - high[2]) < 1e-9
            assert (cells[inside] + g1_terrain.LIFT >= high[2] - 1e-9).all()  # never over a lower cell
        open_ = ramp if SPEC.types[column] == 'pits' else np.zeros_like(ramp)
        assert (tops[~open_] == 1).all() and (tops[open_] == 0).all(), (level, column)
    group = np.zeros(6, np.uint8)
    group[g1_terrain.BOX_GROUP] = 1
    found = np.zeros(1, np.int32)
    rng = np.random.default_rng(0)
    axis = (np.arange(NODES) + .5) * SPEC.cell - 4
    heights = env.terrain_heights.numpy()
    edges = 0
    for level, column in sorted(boxed):
        nodes = tile(heights, SPEC.types[column], level)
        corners = np.stack([nodes[:-1, :-1], nodes[1:, :-1], nodes[:-1, 1:], nodes[1:, 1:]])
        rows = np.r_[rng.integers(0, NODES, 300), np.full(NODES, NODES // 2)]
        columns = np.r_[rng.integers(0, NODES, 300), np.arange(NODES)]
        points = np.stack((centres[level, column, 0] + axis[columns], centres[level, column, 1] + axis[rows]), axis=1)
        hits = np.array([5. - mujoco.mj_ray(model, data, np.array([*p, 5.]), np.array([0., 0., -1.]), group, 1, -1,
                                            found) for p in points])
        level_cell = np.ptp(corners[:, rows, columns], axis=0) == 0
        np.testing.assert_allclose(hits[level_cell], env._ground(points[level_cell]).numpy(), atol=1e-9, rtol=0)
        lift = g1_terrain.LIFT
        assert (hits[~level_cell] <= corners.max(axis=0)[rows, columns][~level_cell] + lift + 1e-9).all()
        assert (hits[~level_cell] >= corners.min(axis=0)[rows, columns][~level_cell] + lift - 1e-9).all()
        edges += (~level_cell).sum()
    assert 0 < edges


def test_generator_three_pit_ramp_is_tilted_boxes_on_the_reward_ground(env):
    """The pits' border ramp (PIT_RAMP wide, from the pit depth down to 0 at the border) collides through twelve
    tilted boxes, not a staircase of cells: mj_ray on the boxes at random points of the ring equals the reward's
    ground (task.terrain_height) to 1e-5 m away from the tile's diagonals and within 1 cm on the corners' creases
    (the heightfield's triangles cut a crease cell along the other diagonal); a tile without a ramp is refused."""
    model = env.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(6, np.uint8)
    group[g1_terrain.BOX_GROUP] = 1
    found = np.zeros(1, np.int32)
    rng = np.random.default_rng(1)
    grid = g1_terrain.terrain_grid(5, 3)
    column = SPEC.types.index('pits')
    worst = {}
    for level in range(TERRAIN_LEVELS):
        local = rng.uniform(-4, 4, (6000, 2))
        local = local[np.abs(local).max(axis=1) > 4 - g1_terrain.PIT_RAMP + 1e-6][:600]
        points = g1_terrain.tile_centres(3)[level, column] + local
        hits = np.array([5. - mujoco.mj_ray(model, data, np.array([*p, 5.]), np.array([0., 0., -1.]), group, 1, -1,
                                            found) for p in points])
        error = np.abs(hits - env._ground(points).numpy())
        crease = np.abs(np.abs(local[:, 0]) - np.abs(local[:, 1])) < 2 * SPEC.cell
        assert error[~crease].max() < 1e-5 and error.max() < .01, (level, error[~crease].max(), error.max())
        worst[level] = error.max()
    assert worst[9] > worst[0]  # the creases' cut grows with the slope
    stairs = g1_terrain._tile(grid, 9, SPEC.types.index('stairs_up'), SPEC)
    with pytest.raises(ValueError):
        g1_terrain.tile_boxes(stairs, SPEC.cell, g1_terrain.PIT_RAMP)


def test_generator_three_starts_and_draws(env):
    """Each type at levels 0 and 9: the robot at its tile's centre, every sole above the ground, the pelvis 0.8 m
    above the platform; training resets draw the generator's shares; the info names the type."""
    centres = g1_terrain.tile_centres(3)
    for seed, (kind, level) in enumerate((kind, level) for kind in SPEC.types for level in (0, 9)):
        env.reset(seed=seed, options=dict(terrain_type=kind, terrain_level=level, command=[.5, 0., 0.]))
        np.testing.assert_allclose(env.data.qpos[:2], centres[level, SPEC.types.index(kind)])
        assert (env._sole_height()[0].numpy() > .01).all() and env.pelvis_height() == pytest.approx(.8, abs=1e-9)
        assert env.step(np.zeros(env.model.nu))[-1]['terrain_type'] == kind
    env.set_terrain_level(0)
    types = []
    for seed in range(600):
        env.reset(seed=seed)
        types.append(env.terrain_type)
    np.testing.assert_allclose(np.bincount(types, minlength=9) / len(types), SPEC.shares, atol=.05)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={'terrain_type': 'lava'})


def test_carried_fields_cross_a_change_of_generator():
    """The generator is one change, as the seed: -terrain2-seedN -> -terrain3box-seedN either way, -terrain2 ->
    -terrain3box; never together with the seed or another suffix; terrain1 and 369be7f's -terrain3 are not carried
    suffixes."""
    from humanoid_sim.warm_start import carried_fields, model_suffixes
    assert model_suffixes('abc-footcol-terrain3box-seed6') == ('abc', ['-footcol', '-terrain3box-seed6'])
    assert model_suffixes('abc-footcol-terrain3box') == ('abc', ['-footcol', '-terrain3box'])
    crossed = lambda old, new: 'model_revision' in carried_fields({'model_revision': old}, {'model_revision': new})
    for old, new in (('abc-footcol-terrain2-seed6', 'abc-footcol-terrain3box-seed6'),
                     ('abc-footcol-terrain2', 'abc-footcol-terrain3box'),
                     ('abc-footcol-terrain3box', 'abc-footcol-terrain3box-seed6'),
                     ('abc-footcol', 'abc-footcol-terrain3box-seed6')):
        assert crossed(old, new) and crossed(new, old), (old, new)
    for old, new in (('abc-footcol-terrain2-seed6', 'abc-footcol-terrain3box-seed7'),  # generator and seed at once
                     ('abc-footcol-terrain2', 'abc-footcol-terrain3box-seed6'),
                     ('abc-terrain2', 'abc-footcol-terrain3box'),
                     ('abc-footcol-terrain3box', 'abc-footcol-terrain2-terrain3box'),
                     ('abc-footcol-terrain1', 'abc-footcol-terrain3box'),
                     ('abc-footcol-terrain3', 'abc-footcol-terrain3box')):  # 369be7f's heightfield physics
        assert not crossed(old, new), (old, new)


def test_tile_types_carry_by_name_and_new_ones_start_at_level_zero():
    old, new = g1_terrain.TERRAIN_TYPES, SPEC.types
    state = {'terrain_level': [0, 3, 5, 9, 7], 'terrain_type': [0, 2, 4, 6, 6], 'load_max': [.5]}
    carried = carried_terrain_state(state, old, new)
    assert carried == {**state, 'terrain_new_types': [7, 8]}  # the old types keep their indices
    assert carried_terrain_state(state, old, old) is state
    back = carried_terrain_state({'terrain_level': [1, 2, 3], 'terrain_type': [7, 3, 8]}, new, old)
    assert back == {'terrain_level': [2], 'terrain_type': [3], 'terrain_new_types': []}
    assert carried_terrain_state({'terrain_level': [1], 'terrain_type': [8]}, new, old) == {}
    types = [0, 0, 6, 6, 7, 7, 8, 1, 1]  # pits and curbs at 0; rough had no world either but is old: all levels
    np.testing.assert_array_equal(restored_terrain_levels(carried, types), [0, 0, 9, 7, 0, 0, 0, 0, 5])
    # without the mark a type no world had takes all saved levels, as before
    np.testing.assert_array_equal(restored_terrain_levels(state, [7, 7, 7, 7, 7]), [0, 3, 5, 9, 7])
    with pytest.raises(ValueError):
        carried_terrain_state({'terrain_level': [1], 'terrain_type': [7]}, old, new)


def test_carry_over_moves_a_terrain_stage_to_generator_three(tmp_path):
    """terrain_next.py --generator 3's path: a generator-2 stage on map 6, its worlds' levels in its training_state,
    carried to generator 3 on map 6 keeps its weights and normalization, gets -terrain3box-seed6, its tile types
    mapped by name and pits and curbs marked new; it loads on the new grid."""
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

    folder, config = run('old', {'seed': 6})
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    state = {'terrain_level': [0, 3, 5, 9], 'terrain_type': [0, 2, 4, 6]}
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        model.learn(16)
        stage = save_stage(folder, model, env, state)
    finally:
        env.close()
    target, _ = run('gen3', {'seed': 6, 'generator': 3})
    moved = carry_over(stage, target)
    saved, metadata = read_json(stage / 'metadata.json'), read_json(moved / 'metadata.json')
    assert saved['model_revision'].endswith('-terrain2-seed6')
    assert metadata['model_revision'] == saved['model_revision'][:-len('-terrain2-seed6')] + '-terrain3box-seed6'
    assert metadata['training_state'] == {**state, 'terrain_new_types': [7, 8]}
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        np.testing.assert_allclose(new_norm.venv.envs[0].unwrapped.terrain_heights.numpy(),
                                   g1_terrain.terrain_grid(6, 3), atol=1e-7, rtol=0)
    finally:
        old_norm.close(), new_norm.close()
    both, _ = run('both', {'seed': 7, 'generator': 3})  # the seed and the generator at once: refused
    with pytest.raises(ValueError):
        carry_over(stage, both)
