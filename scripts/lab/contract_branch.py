"""Branch from the latest stage of a G1 experiment (or <parent>:<steps>) with one top-level task setting changed that
changes the policy contract but not its layout (lock_wrists, phase_stop, knee_floor; foot_collision changes the model
revision, warm_start.CARRIED_MODEL_SUFFIXES; g1_getup's pd_bound), weights carried over unchanged
(warm_start.carry_over, which also keeps the stage's training_state, g1_getup's assist forces):
contract_branch.py <parent>[:steps] <key> <JSON value> <name> <note> [--dry-run].

An ordinary branch (task_branch.py) is refused there by validate_resume; this one is created without a source and
gets the carried-over stage, the manifest's warm_start records where it came from. Queued like task_branch.py.
--dry-run checks everything the branch needs (the lab's config schema, the contracts, the environment, the carry-over
itself into a temporary folder) and prints what it would create, without creating or queueing anything in the lab.

Examples: contract_branch.py exp-14bae200a11f4856:5461377024 lock_wrists true 'G1 реальный · запястья закреплены' '…'
contract_branch.py exp-4baf085b246b4a35:140378112 pd_bound '[1.0, 0.6]' 'G1 вставание · мягкость' '…' --dry-run"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task, resolve_task  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402
from humanoid_sim.warm_start import carry_over, check_carry_over  # noqa: E402

ARGS = [arg for arg in sys.argv[1:] if arg != '--dry-run']
DRY = len(ARGS) != len(sys.argv) - 1
if len(ARGS) != 5:
    raise SystemExit(__doc__)
PARENT, KEY, VALUE, NAME, NOTE = ARGS[0], ARGS[1], json.loads(ARGS[2]), ARGS[3], ARGS[4]
if KEY in ('reward_weights', 'commands', 'gait'):
    raise SystemExit(f'{KEY}: группа настроек — для неё reward_branch.py / command_branch.py')
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
# The parent's effective value, defaults included, by its own task's schema (g1_velocity or g1_getup).
before = resolve_task(parent['robot_id'], parent['task_id']).config_validator(parent['task_config'])[KEY]
task_config = {**parent['task_config'], KEY: VALUE}
source = run / f'stage_{stage:012d}'
saved = read_json(source / 'metadata.json')
# Refuse before creating anything: the same contract (an ordinary branch, task_branch.py), an incompatible one, or a
# setting the environment rejects. carry_over checks the layout again.
wanted = task_identity(parent['robot_id'], parent['task_id'], task_config)
check_carry_over(saved, wanted)
make_task(parent['task_id'], parent['robot_id'], task_config).close()
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
note = (f'Задача: {KEY} {before} → {VALUE} (новый контракт, веса перенесены как есть с {PARENT[:12]} @ '
        f'{stage / 1e6:.0f} млн). {NOTE} Остальное как у {PARENT[:12]}.')
if DRY:  # what lab.create checks, then the carry-over into a temporary run folder; nothing in the lab
    checked = validate_config(config)
    with tempfile.TemporaryDirectory() as folder:
        write_json(Path(folder) / 'run_config.json', checked)
        moved = carry_over(source, Path(folder))
        metadata = read_json(moved / 'metadata.json')
    print('dry run, nothing created:', PARENT, stage, f'{KEY} {before} -> {VALUE}')
    print('name:', checked['name'])
    for field in ('observation_schema', 'action_schema', 'model_revision'):
        if saved.get(field) != metadata[field]:
            print(f'{field}: {saved.get(field)} -> {metadata[field]}')
    state = metadata.get('training_state') or {}
    print('training_state carried:', ', '.join(f'{key} ({len(values)} worlds, mean {sum(values) / len(values):g})'
                                               for key, values in state.items()) or 'none')
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
print(identity, moved.name, f'{KEY} {before} -> {VALUE}')
lab.annotate(identity, verdict='active', note=note)
