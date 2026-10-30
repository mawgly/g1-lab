"""Continue a rough-ground G1 branch on a new map (task_config terrain's seed; tasks/g1_terrain.terrain_grid): the
latest stage of a g1_velocity experiment with terrain (or <parent>:<steps>) as the first stage of a new experiment
whose terrain has another seed, so each run of a few hours walks a floor the policy has not seen (Pavel 29.10: always
different, but realistic floors):
terrain_next.py <parent>[:steps] <name> <note> [--seed N] [--generator N] [--dry-run].

--seed is the new map's seed (missing: the parent's seed + 1, or with another --generator the parent's seed; the
parent's own seed and generator are refused, that is an ordinary branch). The same tile types and levels, other random
heights, blocks and treads: another heightfield, so its own model revision (config.terrain_revision), which
warm_start.carry_over crosses with the weights, the normalization, the optimizer and the stage's training_state (the
worlds' terrain levels, type by type) unchanged. --generator moves the branch to another tile generator
(g1_terrain.GENERATORS; 3: 2.5 cm cells, stairs to 20 cm, pits and curbs) with the parent's seed: carry_over crosses
one change at a time, so not together with --seed; the worlds' tile types are mapped by name and the generator's new
types start at level 0 (g1_velocity.carried_terrain_state). Everything else, the
curriculum and the PPO recipe (worlds, steps, hours, every) are the parent's. As terrain_branch.py, the branch has no
source (the tree shows a root), the manifest's warm_start names the parent stage, and it is queued.
--dry-run checks everything the branch needs (the lab's config schema, the contracts, the environment, the carry-over
into a temporary folder) and prints what it would create, without creating or queueing anything in the lab.

Example: terrain_next.py exp-63520ecfdd8c4205 'G1 реальный · неровный пол вслепую (новая карта)' '…' --dry-run"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.config import terrain_generator, terrain_seed, terrain_setting  # noqa: E402
from humanoid_sim.tasks.g1_terrain import generator as terrain_spec  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402
from humanoid_sim.warm_start import carried_state, carry_over, check_carry_over  # noqa: E402

ARGS = [arg for arg in sys.argv[1:] if arg != '--dry-run']
DRY = len(ARGS) != len(sys.argv) - 1
SEED = GENERATOR = None
for flag in ('--seed', '--generator'):
    if flag in ARGS:
        at = ARGS.index(flag)
        if at + 1 >= len(ARGS):
            raise SystemExit(__doc__)
        if flag == '--seed':
            SEED = int(ARGS[at + 1])
        else:
            GENERATOR = int(ARGS[at + 1])
        del ARGS[at:at + 2]
if len(ARGS) != 3:
    raise SystemExit(__doc__)
PARENT, NAME, NOTE = ARGS
PARENT, _, AT = PARENT.partition(':')
lab = Laboratory(DEFAULT_RUN)
run = lab.resolve(PARENT)
stages = [int(path.name[6:]) for path in list_stages(run)]
if not stages:
    raise SystemExit(f'у {PARENT} пока нет этапов')
stage = int(AT) if AT else stages[-1]
if stage not in stages:
    raise SystemExit(f'у {PARENT} нет этапа {stage}')
parent = read_json(run / 'run_config.json')
if parent['task_id'] != 'g1_velocity':
    raise SystemExit(f'неровный пол — настройка ходьбы g1_velocity, а у {PARENT} задача {parent["task_id"]}')
old_terrain = terrain_setting(parent['task_config'].get('terrain'))
if old_terrain is None:
    raise SystemExit(f'у {PARENT} ровный пол: неровный включает terrain_branch.py')
old_seed, old_generator = terrain_seed(old_terrain), terrain_generator(old_terrain)
generator = old_generator if GENERATOR is None else GENERATOR
seed = (old_seed + 1 if generator == old_generator else old_seed) if SEED is None else SEED
if seed == old_seed and generator == old_generator:
    raise SystemExit(f'у {PARENT} уже карта {seed}: продолжение на той же карте — обычная ветка')
terrain = terrain_setting({**old_terrain, 'seed': seed, 'generator': generator})
task_config = {**parent['task_config'], 'terrain': terrain}
source = run / f'stage_{stage:012d}'
saved = read_json(source / 'metadata.json')
# Refuse before creating anything: a contract carry_over cannot cross, or a setting the environment rejects.
wanted = task_identity(parent['robot_id'], parent['task_id'], task_config)
check_carry_over(saved, wanted)
make_task(parent['task_id'], parent['robot_id'], task_config).close()
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef',
    'range_bounds', 'steps', 'hours')}
config.update(name=f'{NAME} (карта {seed}, с {stage / 1e6:.0f} млн)', preset='custom', task_config=task_config)
state = carried_state(saved, task_config) or {}  # the tile types as the new generator numbers them
levels = state.get('terrain_level')
names = terrain_spec(generator).types
if levels:
    kinds = np.asarray(state.get('terrain_type', [0] * len(levels)))
    per_type = ', '.join(f'{name} {np.mean(np.asarray(levels)[kinds == k]):.1f}'
                         for k, name in enumerate(names) if (kinds == k).any())
    fresh = [names[k] for k in state.get('terrain_new_types', [])]
    carried = (f'уровни миров перенесены (средний {np.mean(levels):.2f}: {per_type})'
               + (f', новые типы с уровня 0: {", ".join(fresh)}' if fresh else ''))
else:
    carried = 'уровней миров у этапа нет: миры начнут с уровня до start_level'
if generator == old_generator:
    change = (f'Новая карта неровного пола: зерно {seed} (у {PARENT[:12]} — {old_seed}), те же типы и уровни плиток, '
              f'другие случайные высоты')
else:
    change = (f'Генератор неровного пола {generator} (у {PARENT[:12]} — {old_generator}), зерно {seed}: '
              f'{", ".join(names)}')
note = (f'{change}; веса перенесены как есть с {PARENT[:12]} @ {stage / 1e6:.0f} млн, {carried}. '
        f'{NOTE} Остальное как у {PARENT[:12]}.')
if DRY:  # what lab.create checks, then the carry-over into a temporary run folder; nothing in the lab
    checked = validate_config(config)
    with tempfile.TemporaryDirectory() as folder:
        write_json(Path(folder) / 'run_config.json', checked)
        moved = carry_over(source, Path(folder))
        metadata = read_json(moved / 'metadata.json')
    print('dry run, nothing created:', PARENT, stage, 'terrain', json.dumps(terrain))
    print('name:', checked['name'])
    print('task:', checked['task_id'], '| backend:', checked['backend'], '| envs:', checked['envs'],
          '| steps:', checked['steps'], '| hours:', checked['hours'])
    for field in ('observation_schema', 'action_schema', 'model_revision'):
        if saved.get(field) != metadata[field]:
            print(f'{field}: {saved.get(field)} -> {metadata[field]}')
    print('training_state kept:', metadata.get('training_state') == saved.get('training_state'),
          '| as the new generator numbers it:', metadata.get('training_state') == (state or None))
    print('note:', note)
    raise SystemExit(0)
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = carry_over(source, path)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage}
write_json(path / 'experiment.json', manifest)
lab.enqueue([identity])
print(identity, moved.name, 'terrain', json.dumps(terrain))
lab.annotate(identity, verdict='active', note=note)
