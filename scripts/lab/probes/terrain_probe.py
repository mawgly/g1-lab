"""CPU only. terrain_probe.py <stage dir | experiment[:steps]> [--seed N] [--generator N] [--levels 5,7,9]
[--workers 8]: walking on a rough-ground map the policy has not seen (task_config terrain's seed, tasks/g1_terrain.
terrain_grid; Pavel 29.10).

The stage's own task config with terrain's seed replaced by --seed (default PROBE_SEED, far from the seeds
scripts/lab/terrain_next.py walks through from TERRAIN_SEED; a stage trained without terrain gets the default
curriculum's tiles) and its tile generator by --generator (default the stage's own; 3: 20 cm stairs, pits and curbs,
g1_terrain.GENERATORS), whose every tile type is run. Per tile type and level: 3 deterministic runs (reset seeds SEEDS:
the start pose's noise), a reset
with the fixed command 0.5 m/s straight on the tile (its centre, the platform's height; no delay, motor friction x1, no
pushes or arm perturbation: a reset with a command), the robot turned to face -y before the first step (heading hold
then keeps that heading). -y, not +x: a tile is 8 m, 10 s at 0.5 m/s is ~5 m, and along x the last metre would be the
next tile type, along -y it is the same type one level easier (level 0: the plane in front of the grid).
Prints per type and level: how many runs walked the 10 s, the path along the heading (median, least), and the metre of
each fall (the path at the fall; '*' marks a fall already off its own tile, more than 4 m from its centre)."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
from humanoid_sim.core import list_stages  # noqa: E402

LAB = SIM / 'runs' / 'humanoid_lab'
PROBE_SEED = 1000  # the probe's map: terrain_next.py counts up from TERRAIN_SEED 5, one seed a run
SEEDS = (0, 1, 2)
SPEED = .5  # m/s
SECONDS = 10.
HEADING = -math.pi / 2  # facing -y: towards the easier level of the same type
_cache = {}


def stage_path(spec):
    """A stage folder from a folder or experiment[:steps] (its latest stage without steps)."""
    if Path(spec).is_dir():
        return Path(spec)
    branch, _, steps = spec.partition(':')
    if steps:
        return LAB / branch / f'stage_{int(steps):012d}'
    stages = list_stages(LAB / branch)
    if not stages:
        raise SystemExit(f'у {branch} нет этапов')
    return stages[-1]


def environment(stage, seed, generator):
    """(env, policy, normalizer) of a stage on the probe's map, built once per worker process."""
    key = (str(stage), seed, generator)
    if key not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        from humanoid_sim.tasks.catalog import make_task
        model, norm = load_stage(Path(stage))
        config = stage_task_config(Path(stage))
        terrain = {key: value for key, value in (config.get('terrain') or {}).items()
                   if key not in ('seed', 'generator')}
        config = {**config, 'terrain': {**terrain, 'seed': seed, 'generator': generator}, 'max_episode_steps': 100000}
        _cache[key] = (make_task('g1_velocity', 'g1_29dof', config), model, norm)
    return _cache[key]


def run(job):
    import mujoco
    from humanoid_sim.tasks import g1_terrain
    stage, seed, generator, kind, level, reset_seed = job
    env, model, norm = environment(stage, seed, generator)
    raw = env.unwrapped
    env.reset(seed=reset_seed, options={'command': [SPEED, 0., 0.], 'phase_steps': 0, 'terrain_type': kind,
                                        'terrain_level': level})
    raw.data.qpos[3:7] = [math.cos(HEADING / 2), 0., 0., math.sin(HEADING / 2)]  # turned about z on the platform
    mujoco.mj_forward(raw.model, raw.data)
    raw.heading_target = HEADING
    obs = raw._observation('fill')
    start = raw.data.qpos[:2].copy()
    centre = g1_terrain.tile_centres(generator)[level, raw.terrain_names.index(kind)]
    forward = np.array([math.cos(HEADING), math.sin(HEADING)])
    fell = None
    for _ in range(round(SECONDS / raw.dt)):
        obs, _, terminated, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if terminated:
            fell = True
            break
    path = float((raw.data.qpos[:2] - start) @ forward)
    off_tile = bool(np.abs(raw.data.qpos[:2] - centre).max() > g1_terrain.TILE / 2)
    return job, dict(fell=bool(fell), path=path, off_tile=off_tile)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage')
    parser.add_argument('--seed', type=int, default=PROBE_SEED)
    parser.add_argument('--generator', type=int, default=None)
    parser.add_argument('--levels', default='5,7,9')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.tasks.config import TERRAIN_LEVELS, TERRAIN_SUFFIXES, terrain_generator, terrain_seed, \
        terrain_setting
    from humanoid_sim.tasks.g1_terrain import generator as terrain_spec
    stage = stage_path(args.stage)
    levels = [int(level) for level in args.levels.split(',')]
    if any(not 0 <= level < TERRAIN_LEVELS for level in levels):
        raise SystemExit(f'уровни — целые от 0 до {TERRAIN_LEVELS - 1}')
    own = terrain_setting(stage_task_config(stage).get('terrain'))
    generator = terrain_generator(own) if args.generator is None else args.generator
    if generator not in TERRAIN_SUFFIXES:
        raise SystemExit('генератор — один из ' + ', '.join(map(str, TERRAIN_SUFFIXES)))
    kinds = terrain_spec(generator).types
    sys.stdout.reconfigure(encoding='utf-8')
    if own is not None and terrain_seed(own) == args.seed and terrain_generator(own) == generator:
        print(f'внимание: карта {args.seed} — та, на которой учился этап, не новая')
    jobs = [(str(stage), args.seed, generator, kind, level, seed) for kind in kinds for level in levels
            for seed in SEEDS]
    with Pool(args.workers) as pool:
        results = dict(pool.map(run, jobs, chunksize=1))
    trained = 'ровный пол' if own is None else f'карта {terrain_seed(own)}, генератор {terrain_generator(own)}'
    print(f'{stage.parent.name} @ {int(stage.name[6:]) / 1e6:.1f} млн ({trained}) на карте {args.seed}, генератор '
          f'{generator}: {SPEED:g} м/с прямо {SECONDS:g} с, {len(SEEDS)} сида, детерминированно')
    reached = 0
    for kind in kinds:
        for level in levels:
            rows = [results[(str(stage), args.seed, generator, kind, level, seed)] for seed in SEEDS]
            walked = sum(not row['fell'] for row in rows)
            reached += walked
            paths = [row['path'] for row in rows]
            falls = ', '.join(f"{row['path']:.1f}{'*' if row['off_tile'] else ''}" for row in rows if row['fell'])
            print(f'  {kind:<11} уровень {level}: дошли {walked} из {len(rows)}, путь медиана {np.median(paths):.2f} м, '
                  f'наименьший {min(paths):.2f} м' + (f', падения на метре {falls}' if falls else ''))
    print(f'итого дошли {reached} из {len(jobs)}')


if __name__ == '__main__':
    main()
