"""Branch a G1 walking stage onto the rough-ground terrain (task_config terrain, stage 5: walking blind on uneven
ground; tasks/g1_terrain): the latest stage of a g1_velocity experiment (or <parent>:<steps>) with terrain switched on,
weights carried over unchanged (warm_start.carry_over: the heightfield is its own model revision, config.TERRAIN_SUFFIX,
which carry_over crosses as foot_collision's; observations and actions are the same):
terrain_branch.py <parent>[:steps] <name> <note> [--terrain '<JSON>'] [--envs N] [--dry-run].

--terrain sets the curriculum (config.TERRAIN_DEFAULTS: up 0.8 and down 0.5 of the commanded path, start_level 0,
slow_level 5, slow_speed 0.5 m/s); missing, the defaults. The branch is created without a source (as
contract_branch.py), the manifest's warm_start names the parent stage, and it is queued with the parent's PPO
recipe. Its first run starts every world at a level up to start_level (the parent has no terrain levels to carry).
--dry-run checks everything the branch needs (the lab's config schema, the contracts, the environment, the carry-over
into a temporary folder) and prints what it would create, without creating or queueing anything in the lab.

Example (candidate 8): terrain_branch.py exp-28691193718e4d0e:7221510144 'G1 реальный · неровный пол' '…' --dry-run"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.config import terrain_setting  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402
from humanoid_sim.warm_start import carry_over, check_carry_over  # noqa: E402

ARGS = [arg for arg in sys.argv[1:] if arg != '--dry-run']
DRY = len(ARGS) != len(sys.argv) - 1
SETTING = {}
if '--terrain' in ARGS:
    at = ARGS.index('--terrain')
    if at + 1 >= len(ARGS):
        raise SystemExit(__doc__)
    SETTING = json.loads(ARGS[at + 1])
    del ARGS[at:at + 2]
ENVS = None  # --envs N: another world count (terrain2 runs 4096 worlds in the plane's buffers, 10.5 GB, 29.10)
if '--envs' in ARGS:
    at = ARGS.index('--envs')
    if at + 1 >= len(ARGS):
        raise SystemExit(__doc__)
    ENVS = int(ARGS[at + 1])
    del ARGS[at:at + 2]
if len(ARGS) != 3:
    raise SystemExit(__doc__)
PARENT, NAME, NOTE = ARGS
TERRAIN = terrain_setting(SETTING)
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
if parent['task_config'].get('terrain') is not None:
    raise SystemExit(f'у {PARENT} неровный пол уже включён: для другой кривой уровней — task_branch.py')
task_config = {**parent['task_config'], 'terrain': TERRAIN}
source = run / f'stage_{stage:012d}'
saved = read_json(source / 'metadata.json')
# Refuse before creating anything: a contract carry_over cannot cross, or a setting the environment rejects.
wanted = task_identity(parent['robot_id'], parent['task_id'], task_config)
check_carry_over(saved, wanted)
make_task(parent['task_id'], parent['robot_id'], task_config).close()
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
if ENVS is not None:  # the same minibatch count: the batch scales with the worlds
    config.update(envs=ENVS, batch_size=config['batch_size'] * ENVS // config['envs'])
curve = (f'вверх от {TERRAIN["up"]:g}, вниз ниже {TERRAIN["down"]:g} заданного пути, старт до уровня '
         f'{TERRAIN["start_level"]}, с уровня {TERRAIN["slow_level"]} команда до {TERRAIN["slow_speed"]:g} м/с')
note = (f'Задача: неровный пол (плитки 8 м, 10 уровней; {curve}); веса перенесены как есть с {PARENT[:12]} @ '
        f'{stage / 1e6:.0f} млн. {NOTE} Остальное как у {PARENT[:12]}.')
if DRY:  # what lab.create checks, then the carry-over into a temporary run folder; nothing in the lab
    checked = validate_config(config)
    with tempfile.TemporaryDirectory() as folder:
        write_json(Path(folder) / 'run_config.json', checked)
        moved = carry_over(source, Path(folder))
        metadata = read_json(moved / 'metadata.json')
    print('dry run, nothing created:', PARENT, stage, 'terrain', json.dumps(TERRAIN))
    print('name:', checked['name'])
    print('task:', checked['task_id'], '| backend:', checked['backend'], '| envs:', checked['envs'])
    for field in ('observation_schema', 'action_schema', 'model_revision'):
        if saved.get(field) != metadata[field]:
            print(f'{field}: {saved.get(field)} -> {metadata[field]}')
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
print(identity, moved.name, 'terrain', json.dumps(TERRAIN))
lab.annotate(identity, verdict='active', note=note)
