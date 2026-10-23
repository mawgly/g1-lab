"""Branch from the latest stage of a G1 experiment (or <parent>:<steps>) with top-level task settings changed (one
change: settings that only act together, such as arm_perturb with arm_perturb_radians and arm_perturb_contact_mask, go
in one call):
task_branch.py <parent>[:steps] <key> <JSON value> [<key> <JSON value> ...] <name> <note> [--dry-run].

--dry-run checks everything the branch needs (the lab's config schema, the source stage's checkpoint identity, the
environment accepting the settings) and prints what it would create, without creating or queueing anything."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.checkpoint_schema import validate_resume  # noqa: E402
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.config import g1_config  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402

ARGS = [arg for arg in sys.argv[1:] if arg != '--dry-run']
DRY = len(ARGS) != len(sys.argv) - 1
if len(ARGS) < 5 or len(ARGS) % 2 == 0:
    raise SystemExit(__doc__)
PARENT, PAIRS, NAME, NOTE = ARGS[0], ARGS[1:-2], ARGS[-2], ARGS[-1]
CHANGES = {key: json.loads(value) for key, value in zip(PAIRS[::2], PAIRS[1::2])}
if len(CHANGES) != len(PAIRS) // 2:
    raise SystemExit('одна настройка указана дважды')
for key in CHANGES:
    if key in ('reward_weights', 'commands', 'gait'):
        raise SystemExit(f'{key}: группа настроек — для неё reward_branch.py / command_branch.py')
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
before = g1_config(parent['task_config'])  # the parent's effective values, defaults included
task_config = {**parent['task_config'], **CHANGES}
change = ', '.join(f'{key} {before.get(key)} → {value}' for key, value in CHANGES.items())
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (от {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3, every=10_000_000,
              source_id=PARENT, source_stage=stage, task_config=task_config)
note = f'Задача: {change}. {NOTE} Остальное как у {PARENT[:12]}.'
if DRY:  # what lab.create checks, plus the environment itself; nothing is written
    checked = validate_config(config)
    source = run / f'stage_{stage:012d}'
    identity = task_identity(checked['robot_id'], checked['task_id'], checked['task_config'])
    validate_resume(read_json(source / 'metadata.json'), identity)
    make_task(checked['task_id'], checked['robot_id'], checked['task_config']).close()
    print('dry run, nothing created:', PARENT, stage, change.replace('→', '->'))
    print('name:', checked['name'])
    print('observation_schema:', identity['observation_schema'], '| action_schema:', identity['action_schema'])
    print('note:', note)
    raise SystemExit(0)
entry = lab.create(config)
identity = entry['id']
lab.enqueue([identity])
print(identity, stage, change.replace('→', '->'))
lab.annotate(identity, verdict='active', note=note)
