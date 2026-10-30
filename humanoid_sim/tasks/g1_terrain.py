"""G1 rough terrain (task_config terrain; stage 5 of the roadmap, walking blind on uneven ground).

A grid of TILE m tiles beside the floor plane, one heightfield geom per tile centred on it (terrain_geoms; why not one
heightfield for the grid: below): TERRAIN_LEVELS rows of difficulty (level 0 the mildest)
times the columns of TERRAIN_TYPES, after mjlab's ROUGH_TERRAINS_CFG (docs/lab-knowledge.md «Неровный пол вслепую»):
flat, rough (random heights 2 → 6 cm), boxes (blocks 1 → 6 cm high), stairs (steps 2 → 8 cm, treads 0.30-0.40 m) and
slopes (2 → 15°), the stairs and slopes each as a pyramid down from a platform (`_down`) and as a ridge up from the
centre and down to the border (`_up`). Every tile has a flat spawn platform of 2·PAD m at its centre and height 0 at
its border, so neighbouring tiles meet without a cliff; nothing lies below 0, so the plane stays the floor everywhere
(a mjlab inverted pyramid would need a pit below it). The heightfields are LIFT above the plane, so a foot on a flat
tile touches only them; outside the grid the plane at 0 is the ground, the origin (where resets with a command start)
TERRAIN_CORNER's distance away.

One heightfield per tile, not one for the grid (29.10, terrain2): MJWarp (3.16) computes a heightfield contact in the
heightfield's frame in float32, and its prism witness points (collision_gjk._epa_witness / _inflate on the top
triangle, `_tri_affine_coord`) multiply affine coordinates, which carry float32 cancellation from products of
coordinates, by the vertices' absolute positions. With the grid's centre 40 m from a robot (terrain1) that error is
metres: a 5 mm sole sphere at a cell edge got contacts 1.4-4 m deep, their normal along the line from the
heightfield's centre, and the robots were thrown up within ~10 steps of a reset (candidate 8: ~1100 falls in 300
steps over 256 worlds, flat tiles too; MuJoCo C, float64, gave -5 mm in the same states). Centred on its tile a
heightfield sees its robot within a few metres, and the same states give MuJoCo's contacts.

The tile types and the curriculum follow mjlab (github.com/mujocolab/mjlab, Apache-2.0, src/mjlab/terrains at
8bb2bc9e), but its generator is not vendored: it writes each tile as its own geoms into the MjSpec (the stairs as
hundreds of boxes, the rough tiles through scipy, which our environments do not have) and measures the ground with ray
sensors, while this task's reward needs one exact ground height on both backends (g1_tensor_math.terrain_height over
one heightfield). terrain_grid(seed) gives another random map of the same tiles (task_config terrain's seed).

Generator 3 (task_config terrain's generator, 30.10, the lead after critic it94: a height-map teacher on 15-20 cm
stairs, pits and curbs): the same tiles on a 2.5 cm cell (GENERATOR_3; at 5 cm a 20 cm riser is a 76° slope, and
feet_flat takes the ground's normal under the ankle_roll origin), stairs of 2 -> 20 cm on treads of 0.23-0.31 m (a
13-step pyramid stands 2.6 m, so ELEVATION_3 3 m), and two more types: pits (the tile raised to the level's depth, 2 ->
20 cm, with square holes down to 0 and a ramp down to its border; nothing lies below 0) and curbs (square bands 2 -> 15
cm high, 0.15-0.3 m wide, around the platform, so across any path from the centre). Generator 2 (TERRAIN_GENERATOR,
the module's plain constants) is the map of terrain2 bit for bit.

Generator 3's steps as boxes (30.10, the lead; -terrain3box): MJWarp (3.16) measures a contact with a steep heightfield
cell vertically, so at 15-20 cm risers sole spheres, knees and the torso got contacts down to -19 cm (normal (0, 0, ±1))
in 9.3 % of upright states where MuJoCo C gave at most 1.7 cm, whatever the cell (5 cm too) or ccd_iterations. On
box_tiles (BOX_TYPES: stairs, pits, curbs at every level; any other tile whose largest step between neighbouring
nodes exceeds BOX_RISER: the top levels of `boxes`) the heightfield stays in the model for the reward's heights but
collides with nothing (contype = conaffinity = 0), and static boxes give the same ground (tile_boxes, as mjlab's
primitive terrains build stairs): each cell at the highest of its four nodes, the cells of one height covered by
rectangles from LIFT - BASE up to that height + LIFT (never two of one height over one cell: a sphere there would get
a contact from each), so the physics' ground is the reward's except within one cell of an edge, where it stands at the
upper side. The pits' border ramp, the only gentle slope on these tiles (stairs, curbs, blocks and the pits' plateau
are flat between their edges), is twelve tilted boxes on its four planes instead of a staircase of 1 cm steps
(_ramp_boxes; 30.10, the lead: ~80 boxes a pits tile of the 2229 on map 1000, and MJWarp's NXN broadphase tests every
box against every robot geom). Rough and slope tiles keep their heightfield (cells of a few millimetres' rise, which
MJWarp measures as MuJoCo does).

The geometry is part of the model revision (config.TERRAIN_SUFFIXES): change any constant, a generator or how the
heightfields are laid out, add a generator with its own suffix. A seed other than TERRAIN_SEED adds its own suffix
(config.terrain_revision). The heights the task reads (g1_tensor_math.terrain_height over
the whole grid, model_heights) come from the compiled model's own float32 heightfield data, so they are MuJoCo's
surface, triangle for triangle (neighbouring tiles share their border row, 0 on both).
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import re
import mujoco
import numpy as np
from .config import TERRAIN_GENERATOR, TERRAIN_LEVELS, TERRAIN_SEED

TERRAIN_TYPES = ('flat', 'rough', 'boxes', 'stairs_up', 'stairs_down', 'slope_up', 'slope_down')
# Shares of the worlds per type (the full-GPU task assigns them by world index; the CPU task draws one per training
# reset): flat 25 % as mjlab's plane share, the rest about evenly.
TERRAIN_SHARES = (.25, .15, .15, .1, .1, .125, .125)
TILE = 8.  # m, a tile's side
CELL = .05  # m, the heightfield's spacing (stair and block edges become one cell's slope)
PAD = 1.  # m, half the side of the flat spawn platform at a tile's centre
ROUGH = (.02, .06)  # m, the largest random height at level 0 and at the top level
ROUGH_STEP = .1  # m, spacing of the random heights (linear between them)
ROUGH_RAMP = .5  # m, the random heights fade in over this from the platform and fade out towards the border
BOXES = (.01, .06)  # m, the largest block height
BOX_SIZE = .5  # m
STEPS = (.02, .08)  # m, step height
TREAD = (.30, .40)  # m, tread width, one per tile
SLOPE = (2., 15.)  # degrees
LIFT = .002  # m, the heightfield above the plane
BASE = .1  # m, the heightfield's thickness below its zero
ELEVATION = 1.  # m, the heightfield's height scale (the highest tile is below it)
TERRAIN_CORNER = (-TILE * len(TERRAIN_TYPES) / 2, 40.)  # (x, y) m of the grid's corner of level 0, type 0
# TERRAIN_SEED (config): the map without task_config terrain's seed
TERRAIN_GEOM = 'terrain'  # the prefix of each tile's heightfield and geom name, terrain_<level>_<type index>
TILE_NAME = re.compile(re.escape(TERRAIN_GEOM) + r'_(\d+)_(\d+)$')
NODES = round(TILE / CELL)  # cells per tile side

# Generator 3 (the module's docstring). The old types keep their columns, pits and curbs follow.
TERRAIN_TYPES_3 = (*TERRAIN_TYPES, 'pits', 'curbs')
TERRAIN_SHARES_3 = (.15, .1, .1, .15, .15, .075, .075, .1, .1)
CELL_3 = .025  # m
STEPS_3 = (.02, .20)  # m
TREAD_3 = (.23, .31)  # m
ELEVATION_3 = 3.  # m: stairs_down at the top level, 13 steps of 0.20 m on a 3 m run, stands 2.6 m
PITS = (.02, .20)  # m, the pits' depth (the tile's raised height)
PIT_BLOCK = 1.  # m: the plateau is cut into blocks of this side, each with at most one pit
PIT_SIZE = (.3, .6)  # m, a pit's side (square), one per pit
PIT_SHARE = .6  # the share of the plateau's blocks with a pit
PIT_RIM = .1  # m: a pit keeps at least this from its block's edge (so 0.2 m between neighbouring pits)
PIT_RAMP = .5  # m: the raised plateau goes down to 0 at the tile's border over this
CURBS = (.02, .15)  # m, curb height
CURB_WIDTH = (.15, .3)  # m, one per curb
CURB_GAP = (.5, 1.)  # m, flat ground before each curb (the first one from the platform's edge)
CURB_MARGIN = .2  # m: no curb this close to the tile's border
BOX_TYPES = ('stairs_up', 'stairs_down', 'pits', 'curbs')  # generator 3: tiles that always collide through boxes
BOX_RISER = .05  # m: so does any other tile with a larger step between neighbouring nodes (boxes at its top levels)
BOX_NAME = re.compile(re.escape(TERRAIN_GEOM) + r'_(\d+)_(\d+)_box(\d+)$')  # a box tile's boxes, tile_name + _box<k>
BOX_GROUP = 4  # the boxes' geom group (hidden in MuJoCo's viewers by default; the heightfield draws the same ground)


@dataclass(frozen=True)
class Generator:
    """A tile generator (task_config terrain's generator): its tile types and their shares, the heightfield's cell and
    height scale, the stairs' step heights (level 0, top level) and tread bounds, and the tile types that collide
    through boxes and the riser above which any other tile does (box_tiles; none and None: every tile's heightfield
    collides)."""
    number: int
    types: tuple
    shares: tuple
    cell: float
    elevation: float
    steps: tuple
    tread: tuple
    boxes: tuple = ()
    box_riser: float = None

    @property
    def nodes(self):
        """Cells per tile side."""
        return round(TILE / self.cell)

    @property
    def corner(self):
        """(x, y) m of the grid's corner of level 0, type 0 (the grid centred on x = 0, 40 m from the origin)."""
        return (-TILE * len(self.types) / 2, TERRAIN_CORNER[1])


GENERATORS = {2: Generator(2, TERRAIN_TYPES, TERRAIN_SHARES, CELL, ELEVATION, STEPS, TREAD),
              3: Generator(3, TERRAIN_TYPES_3, TERRAIN_SHARES_3, CELL_3, ELEVATION_3, STEPS_3, TREAD_3, BOX_TYPES,
                           BOX_RISER)}


def generator(number=TERRAIN_GENERATOR):
    """The Generator of task_config terrain's generator number."""
    if number not in GENERATORS:
        raise ValueError('terrain generator must be one of ' + ', '.join(map(str, GENERATORS)))
    return GENERATORS[number]


def difficulty(level):
    """0 at level 0, 1 at the top level."""
    return level / (TERRAIN_LEVELS - 1)


def _lerp(bounds, level):
    return bounds[0] + (bounds[1] - bounds[0]) * difficulty(level)


def tile_heights(kind, level, rng, spec=None):
    """(nodes + 1, nodes + 1) heights (m) of one tile of generator spec (None: generator 2), rows along y, columns along
    x, the tile's centre in the middle: 0 at the border, the spawn platform flat."""
    spec = GENERATORS[TERRAIN_GENERATOR] if spec is None else spec
    nodes = spec.nodes
    axis = np.linspace(-TILE / 2, TILE / 2, nodes + 1)
    x, y = np.meshgrid(axis, axis)
    chebyshev = np.maximum(np.abs(x), np.abs(y))
    edge = TILE / 2
    if kind not in spec.types:
        raise ValueError(f'Unknown terrain type {kind}')
    if kind == 'flat':
        return np.zeros_like(x)
    if kind == 'rough':
        top = _lerp(ROUGH, level)
        coarse = round(TILE / ROUGH_STEP)
        noise = rng.uniform(0., top, (coarse + 1, coarse + 1))
        fine = np.linspace(0, coarse, nodes + 1)  # linear between the random heights, along both axes
        noise = np.array([np.interp(fine, np.arange(coarse + 1), row) for row in noise])
        noise = np.array([np.interp(fine, np.arange(coarse + 1), column) for column in noise.T]).T
        fade = np.clip((chebyshev - PAD) / ROUGH_RAMP, 0., 1.) * np.clip((edge - chebyshev) / ROUGH_RAMP, 0., 1.)
        return noise * fade
    if kind == 'boxes':
        top = _lerp(BOXES, level)
        blocks = round(TILE / BOX_SIZE)
        height = rng.uniform(0., top, (blocks, blocks))
        index = np.clip(np.floor((axis + edge) / BOX_SIZE).astype(int), 0, blocks - 1)
        centre = (np.arange(blocks) + .5) * BOX_SIZE - edge
        far = np.maximum(np.abs(centre)[:, None], np.abs(centre)[None, :])
        height[(far < PAD) | (far > edge - BOX_SIZE)] = 0.  # the platform's blocks and the border ring stay flat
        return height[index[:, None], index[None, :]]
    if kind.startswith(('stairs', 'slope')):
        # Distance walked from the platform's edge (`_up`: up to the ridge halfway to the border, then down) or from
        # the border inwards (`_down`: up to the platform).
        if kind.endswith('_down'):
            run = np.clip(edge - chebyshev, 0., edge - PAD)
        else:
            run = np.clip(np.minimum(chebyshev - PAD, edge - chebyshev), 0., None)
        if kind.startswith('stairs'):
            tread = rng.uniform(*spec.tread)
            return _lerp(spec.steps, level) * np.floor(run / tread)
        return math.tan(math.radians(_lerp(SLOPE, level))) * run
    if kind == 'pits':
        return _pits(x, y, chebyshev, level, rng)
    return _curbs(chebyshev, level, rng)  # 'curbs'


def _pits(x, y, chebyshev, level, rng):
    """Pits: the tile at the level's depth (PITS), down to 0 over PIT_RAMP at the border, and in each PIT_BLOCK block
    of the plateau between the platform and the ramp, with probability PIT_SHARE, a square hole down to 0 of side
    U(PIT_SIZE) placed uniformly with PIT_RIM to its block's edge. The platform stays at the depth, flat."""
    edge = TILE / 2
    depth = _lerp(PITS, level)
    heights = depth * np.clip((edge - chebyshev) / PIT_RAMP, 0., 1.)
    blocks = round(TILE / PIT_BLOCK)
    centre = (np.arange(blocks) + .5) * PIT_BLOCK - edge
    for row in range(blocks):
        for column in range(blocks):
            far = max(abs(centre[row]), abs(centre[column]))
            # the draws of every block, so a block's pit does not depend on whether its neighbours have one
            has, side, (dx, dy) = rng.random() < PIT_SHARE, rng.uniform(*PIT_SIZE), rng.uniform(-1., 1., 2)
            if not has or far - PIT_BLOCK / 2 < PAD or far + PIT_BLOCK / 2 > edge - PIT_RAMP:
                continue
            room = PIT_BLOCK / 2 - PIT_RIM - side / 2
            px, py = centre[column] + dx * room, centre[row] + dy * room
            heights[(np.abs(x - px) <= side / 2) & (np.abs(y - py) <= side / 2)] = 0.
    return heights


def _curbs(chebyshev, level, rng):
    """Curbs: square bands (in Chebyshev distance from the centre, so across any path out of the platform) of the
    level's height (CURBS), each U(CURB_WIDTH) wide after U(CURB_GAP) of flat ground, from the platform's edge until
    one would come within CURB_MARGIN of the border; 0 elsewhere."""
    edge = TILE / 2
    height = _lerp(CURBS, level)
    heights = np.zeros_like(chebyshev)
    start = PAD
    while True:
        start += rng.uniform(*CURB_GAP)
        width = rng.uniform(*CURB_WIDTH)
        if start + width > edge - CURB_MARGIN:
            return heights
        heights[(chebyshev >= start) & (chebyshev < start + width)] = height
        start += width


@lru_cache(maxsize=4)
def terrain_grid(seed=TERRAIN_SEED, number=TERRAIN_GENERATOR):
    """(rows, columns) heights (m) of the whole grid of generator `number`, float64, rows along y (level by level),
    columns along x (type by type); read-only. Each tile has its own random stream (seed, level, type; generator 3
    adds its number), so another seed is another map of the same tiles: the same types, difficulties, platforms and
    borders, other random heights, blocks, treads, pits and curbs. The task's model uses task_config terrain's seed
    and generator (TERRAIN_SEED, TERRAIN_GENERATOR without them; others are their own model revision,
    config.terrain_revision)."""
    spec = generator(number)
    nodes = spec.nodes
    heights = np.zeros((TERRAIN_LEVELS * nodes + 1, len(spec.types) * nodes + 1))
    for level in range(TERRAIN_LEVELS):
        for column, kind in enumerate(spec.types):
            key = (seed, level, column) if number == TERRAIN_GENERATOR else (seed, level, column, number)
            tile = tile_heights(kind, level, np.random.default_rng(key), spec)
            rows, columns = _tile_nodes(level, column, nodes)
            heights[rows, columns] = np.maximum(heights[rows, columns], tile)  # shared borders are 0 on both sides
    if heights.max() >= spec.elevation or heights.min() < 0:
        raise ValueError('terrain heights outside [0, ELEVATION)')
    heights.setflags(write=False)
    return heights


def tile_centres(number=TERRAIN_GENERATOR):
    """(TERRAIN_LEVELS, types, 2) world (x, y) of each tile's centre in generator `number`'s grid."""
    spec = generator(number)
    level, column = np.meshgrid(np.arange(TERRAIN_LEVELS), np.arange(len(spec.types)), indexing='ij')
    return np.stack((spec.corner[0] + (column + .5) * TILE, spec.corner[1] + (level + .5) * TILE), axis=-1)


def tile_name(level, column):
    """The name of the heightfield and the geom of the tile at level (row) and type index (column)."""
    return f'{TERRAIN_GEOM}_{level}_{column}'


def _tile_nodes(level, column, nodes=NODES):
    """The grid's (rows, columns) slices of one tile, its border rows shared with the neighbours."""
    return (slice(level * nodes, (level + 1) * nodes + 1), slice(column * nodes, (column + 1) * nodes + 1))


def _tiles(columns=len(TERRAIN_TYPES)):
    return [(level, column) for level in range(TERRAIN_LEVELS) for column in range(columns)]


def terrain_geoms(model):
    """Ids of the tiles' heightfield geoms in a compiled model (any generator: named tile_name), in model order (empty
    without task_config terrain)."""
    return [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD
            and TILE_NAME.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '')]


def terrain_boxes(model):
    """Ids of the box tiles' boxes (BOX_NAME) in a compiled model, in model order (empty for generator 2)."""
    return [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            and BOX_NAME.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '')]


def ground_geoms(model):
    """Ids of every terrain geom the robot can touch: the colliding tile heightfields and the box tiles' boxes."""
    return sorted([i for i in terrain_geoms(model) if model.geom_contype[i] or model.geom_conaffinity[i]]
                  + terrain_boxes(model))


def _tile(grid, level, column, kind):
    """One tile's (nodes + 1, nodes + 1) heights as the compiled model has them (float32 over the elevation)."""
    data = (grid[_tile_nodes(level, column, kind.nodes)] / kind.elevation).astype(np.float32)
    return data.astype(np.float64) * kind.elevation


@lru_cache(maxsize=4)
def box_tiles(seed=TERRAIN_SEED, number=TERRAIN_GENERATOR):
    """The (level, type index) of the tiles of generator `number`'s map `seed` that collide through boxes (the
    module's docstring): its `boxes` types, and any tile whose largest step between neighbouring nodes exceeds its
    box_riser; empty for generator 2."""
    kind = generator(number)
    grid = terrain_grid(seed, number)
    tiles = set()
    for level, column in _tiles(len(kind.types)):
        if kind.types[column] in kind.boxes:
            tiles.add((level, column))
        elif kind.box_riser is not None:
            tile = grid[_tile_nodes(level, column, kind.nodes)]
            if max(np.abs(np.diff(tile, axis=0)).max(), np.abs(np.diff(tile, axis=1)).max()) > kind.box_riser:
                tiles.add((level, column))
    return frozenset(tiles)


def _runs(cells):
    """Rectangles (row0, row1, column0, column1, height), half-open cell indices, each of one height, that tile
    `cells` exactly: each row's runs of one height, a run continued while the next row has the same run (NaN cells:
    left out)."""
    rectangles, open_ = [], {}
    for r, row in enumerate(cells):
        cut = np.flatnonzero(np.diff(row) != 0) + 1
        now = {(int(s), int(e), float(row[s])) for s, e in zip(np.r_[0, cut], np.r_[cut, len(row)])
               if not math.isnan(row[s])}
        for key in [key for key in open_ if key not in now]:
            rectangles.append((open_.pop(key), r, *key))
        for key in sorted(now):
            open_.setdefault(key, r)
    return rectangles + [(r0, len(cells), *key) for key, r0 in open_.items()]


def _layers(cells):
    """Rectangles as _runs' that may also lie under higher cells (never over lower ones, nor over each other at one
    height, where a sphere would touch both tops): height by height from the lowest, the first cell of that height
    not yet covered grows right, then down, over higher cells and uncovered cells of its height (a pyramid's step
    is one rectangle, the ground at 0 one for the tile); NaN cells are neither covered nor lain under."""
    rectangles = []
    for height in np.unique(cells[np.isfinite(cells)]):
        higher, left = cells > height, cells == height
        while left.any():
            r, c = np.unravel_index(np.argmax(left), left.shape)
            allowed = higher | left
            row = allowed[r, c:]
            c1 = c + (int(np.argmin(row)) if not row.all() else len(row))
            rows = allowed[r:, c:c1].all(axis=1)
            r1 = r + (int(np.argmin(rows)) if not rows.all() else len(rows))
            left[r:r1, c:c1] = False
            rectangles.append((int(r), int(r1), int(c), int(c1), float(height)))
    return rectangles


def _quat(u, v, n):
    """MuJoCo's quaternion of the rotation whose columns are the orthonormal right-handed axes u, v, n."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack((u, v, n), axis=1).ravel())
    return quat


def _slab(origin, u, v, span_u, span_v):
    """A box (centre, half sizes, quaternion) BASE thick under the top face origin + [span_u]·u + [span_v]·v (u, v
    orthonormal; span_* (low, high)): its top face lies in that plane, its normal pointing up."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if np.cross(u, v)[2] < 0:  # right-handed with the normal up
        v, span_v = -v, (-span_v[1], -span_v[0])
    normal = np.cross(u, v)
    centre = np.asarray(origin, float) + sum(span_u) / 2 * u + sum(span_v) / 2 * v - BASE / 2 * normal
    return centre, np.array([(span_u[1] - span_u[0]) / 2, (span_v[1] - span_v[0]) / 2, BASE / 2]), _quat(u, v, normal)


def _ramp_boxes(depth, width, cell):
    """Twelve tilted boxes giving a tile's border ramp exactly: the ring of `width` inside the border going down from
    `depth` at its inner edge to 0 at the border, linearly in the Chebyshev distance (pits' PIT_RAMP), so four planes
    meeting on the tile's diagonals. Per side one box under its strip between the corners (its ends vertical), and at
    each corner one box per side's triangle, its top that side's plane, one face standing on the diagonal crease (a
    box's faces are perpendicular to its top, so this face falls below the other side's plane); the outer ends run on
    until they are LIFT under the floor plane (the neighbour's ground hides them), the inner ends stop at the inner
    edge."""
    edge, inner = TILE / 2, TILE / 2 - width
    slope = depth / width
    c, s = math.cos(math.atan(slope)), math.sin(math.atan(slope))
    down = np.array([c, 0., -s])  # along the +x side's plane, outwards
    on = np.array([inner, 0., depth])
    side = [_slab(on, down, [0., 1., 0.], (0., width / c + 2 * LIFT / s), (-inner, inner))]
    for sign in (1., -1.):  # the +x side's triangles at the +y and -y corners
        start, end = np.array([inner, sign * inner, depth]), np.array([edge, sign * edge, 0.])
        crease = (end - start) / np.linalg.norm(end - start)
        far = np.array([edge, sign * inner, 0.]) - start  # the triangle's third corner
        across = far - far @ crease * crease
        side.append(_slab(start, across / np.linalg.norm(across), crease, (0., np.linalg.norm(across) + cell),
                          (0., np.linalg.norm(end - start) + 2 * LIFT / (-crease[2]))))
    boxes = []
    for turn in range(4):  # the other sides: the +x side's boxes turned about z
        angle = turn * math.pi / 2
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.],
                             [0., 0., 1.]])
        spin = np.array([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        for centre, half, quat in side:
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, spin, quat)
            boxes.append((rotation @ centre, half, turned))
    return boxes


def tile_boxes(heights, cell, ramp=0.):
    """Boxes (centre (x, y, z), half sizes, quaternion) in the tile's frame (m, its centre at 0; z above the
    heightfields' zero) of one tile's (nodes + 1, nodes + 1) heights: each cell at the highest of its four nodes,
    covered by the fewer of _runs' and _layers' rectangles, upright, from -BASE to that height. Where the four nodes
    of a cell are equal the boxes' top is the heightfield's. With a border ramp (`ramp` m wide, the pits'), its ring
    of cells is left to _ramp_boxes' twelve tilted boxes instead (its heights must be the ramp's planes, else
    ValueError)."""
    cells = np.maximum.reduce([heights[:-1, :-1], heights[1:, :-1], heights[:-1, 1:], heights[1:, 1:]])
    edge = TILE / 2
    boxes = []
    if ramp:
        axis = np.linspace(-edge, edge, len(heights))
        chebyshev = np.maximum(np.abs(axis)[:, None], np.abs(axis)[None, :])
        depth = float(heights[len(heights) // 2, round((TILE - ramp) / cell)])
        ring = chebyshev >= edge - ramp - 1e-9
        if depth <= 0 or np.abs(heights[ring] - depth * (edge - chebyshev[ring]) / ramp).max() > 1e-6:
            raise ValueError('the tile has no border ramp of this width')
        middle = (axis[:-1] + axis[1:]) / 2
        cells = np.where(np.maximum(np.abs(middle)[:, None], np.abs(middle)[None, :]) > edge - ramp, np.nan, cells)
        boxes = _ramp_boxes(depth, ramp, cell)
    rectangles = min(_runs(cells), _layers(cells), key=len)
    return [(np.array([((c0 + c1) * cell / 2 - edge), (r0 + r1) * cell / 2 - edge, (top - BASE) / 2]),
             np.array([(c1 - c0) * cell / 2, (r1 - r0) * cell / 2, (top + BASE) / 2]), np.array([1., 0., 0., 0.]))
            for r0, r1, c0, c1, top in rectangles] + boxes


def add_terrain(spec, seed=TERRAIN_SEED, number=TERRAIN_GENERATOR):
    """Adds one heightfield and its geom per tile (tile_name) of generator `number`, each centred on its tile (the
    module's docstring: why), to a G1 scene's MjSpec, with the floor plane's contact filter and contact parameters (so
    the same robot geoms touch it, the sole spheres with the real robot's friction priority); on box_tiles the
    heightfield collides with nothing and tile_boxes' static boxes (BOX_NAME, BOX_GROUP) with the floor's filter and
    contact parameters do (the pits' border ramp as tilted boxes). After compiling, write_terrain sets the
    heightfields' heights."""
    floor = spec.geom('floor')
    kind = generator(number)
    grid = terrain_grid(seed, number)
    centres = tile_centres(number)
    boxed = box_tiles(seed, number)
    nodes = kind.nodes
    contact = dict(condim=floor.condim, friction=list(floor.friction), solref=list(floor.solref),
                   solimp=list(floor.solimp), priority=floor.priority, margin=floor.margin)
    for level, column in _tiles(len(kind.types)):
        name = tile_name(level, column)
        # MuJoCo needs data to compile and rescales it to [0, 1]; write_terrain puts the heights back on the
        # elevation's scale (a flat tile's zeros too).
        spec.add_hfield(name=name, size=[TILE / 2, TILE / 2, kind.elevation, BASE], nrow=nodes + 1, ncol=nodes + 1,
                        userdata=(grid[_tile_nodes(level, column, nodes)] / kind.elevation).astype(np.float32).ravel())
        boxes = (level, column) in boxed
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=[*centres[level, column], LIFT],
            contype=0 if boxes else floor.contype, conaffinity=0 if boxes else floor.conaffinity, **contact,
            rgba=[.45, .42, .38, 1.])
        if not boxes:
            continue
        x, y = centres[level, column]
        ramp = PIT_RAMP if kind.types[column] == 'pits' else 0.
        for k, (centre, half, quat) in enumerate(tile_boxes(_tile(grid, level, column, kind), kind.cell, ramp)):
            spec.worldbody.add_geom(  # the tile's frame LIFT up, as the heightfield
                name=f'{name}_box{k}', type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(),
                pos=[x + centre[0], y + centre[1], LIFT + centre[2]], quat=quat.tolist(), contype=floor.contype,
                conaffinity=floor.conaffinity, **contact, group=BOX_GROUP, rgba=[.45, .42, .38, 1.])


def write_terrain(model, seed=TERRAIN_SEED, number=TERRAIN_GENERATOR):
    """The grid's heights (terrain_grid of seed and generator) into a compiled model's tile heightfields (normalized by
    the generator's elevation, MuJoCo's float32)."""
    kind = generator(number)
    grid = terrain_grid(seed, number)
    for level, column in _tiles(len(kind.types)):
        hfield = model.hfield(tile_name(level, column))
        model.hfield_data[hfield.adr[0]:hfield.adr[0] + hfield.nrow[0] * hfield.ncol[0]] = (
            grid[_tile_nodes(level, column, kind.nodes)] / kind.elevation).ravel()


def model_heights(model):
    """(rows, columns) float64 heights (m) of the whole grid above the heightfields' zero as MuJoCo uses them: each
    tile's float32 data times its height scale; with the grid's corner (x, y), the spacing and the lift for
    terrain_height. The grid's size (levels, types, cells per tile) is read from the model's tiles, so any generator."""
    tiles = [tuple(map(int, TILE_NAME.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)).groups()))
             for i in terrain_geoms(model)]
    levels, columns = max(level for level, _ in tiles) + 1, max(column for _, column in tiles) + 1
    nodes = int(model.hfield(tile_name(0, 0)).nrow[0]) - 1
    heights = np.zeros((levels * nodes + 1, columns * nodes + 1))
    for level, column in tiles:
        hfield = model.hfield(tile_name(level, column))
        nrow, ncol = int(hfield.nrow[0]), int(hfield.ncol[0])
        data = model.hfield_data[hfield.adr[0]:hfield.adr[0] + nrow * ncol].astype(np.float64)
        heights[_tile_nodes(level, column, nodes)] = data.reshape(nrow, ncol) * model.hfield_size[hfield.id][2]
    first = model.hfield(tile_name(0, 0))
    size = model.hfield_size[first.id]
    pos = model.geom_pos[model.geom(tile_name(0, 0)).id]
    corner = (float(pos[0] - size[0]), float(pos[1] - size[1]))
    return heights, corner, 2 * float(size[0]) / (int(first.ncol[0]) - 1), float(pos[2])
