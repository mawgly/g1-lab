"""Branch from the latest stage of a G1 experiment (or <parent>:<steps>) with one reward weight changed, g1_velocity or
g1_getup alike (a g1_getup branch keeps the stage's training_state, its worlds' assist forces and pd_bound bounds, which
the branch's copy of the stage carries and its first run restores):
reward_branch.py <parent>[:steps] <key> <value> <name> <note> [--dry-run].

--dry-run checks everything the branch needs (the lab's config schema with the weight, the source stage's checkpoint
identity, the environment accepting the settings) and prints what it would create, without creating or queueing
anything.

Example: reward_branch.py exp-15f0753be2ea4aa4:320864256 impact 2 'G1 вставание · удары' '…' --dry-run"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.checkpoint_schema import validate_resume  # noqa: E402
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402

ARGS = [arg for arg in sys.argv[1:] if arg != '--dry-run']
DRY = len(ARGS) != len(sys.argv) - 1
if len(ARGS) != 5:
    raise SystemExit(__doc__)
PARENT, KEY, VALUE, NAME, NOTE = ARGS[0], ARGS[1], float(ARGS[2]), ARGS[3], ARGS[4]
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
weights = parent['task_config']['reward_weights']
before = weights.get(KEY, 0.)  # a term added after the parent was created is absent there
task_config = {**parent['task_config'], 'reward_weights': {**weights, KEY: VALUE}}  # history and the rest as the parent
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (от {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3, every=10_000_000,
              source_id=PARENT, source_stage=stage, task_config=task_config)
note = f'{KEY} {before:g} → {VALUE:g}. {NOTE} Остальное как у {PARENT[:12]}.'
if DRY:  # what lab.create checks, plus the environment itself; nothing is written
    checked = validate_config(config)
    source = run / f'stage_{stage:012d}'
    saved = read_json(source / 'metadata.json')
    identity = task_identity(checked['robot_id'], checked['task_id'], checked['task_config'])
    validate_resume(saved, identity)
    make_task(checked['task_id'], checked['robot_id'], checked['task_config']).close()
    print('dry run, nothing created:', PARENT, stage, f'{KEY} {before:g} -> {VALUE:g}')
    print('name:', checked['name'])
    print('task:', checked['task_id'], '| backend:', checked['backend'], '| envs:', checked['envs'])
    print('observation_schema:', identity['observation_schema'], '| action_schema:', identity['action_schema'])
    state = saved.get('training_state') or {}
    print('training_state carried:', ', '.join(
        f'{key} ({len(values)} worlds, {min(values):g} to {max(values):g})' for key, values in state.items()) or 'none')
    print('note:', note)
    raise SystemExit(0)
entry = lab.create(config)
identity = entry['id'] if isinstance(entry, dict) else entry
lab.enqueue([identity])
print(identity, stage, f'{KEY} {before:g} -> {VALUE:g}')
lab.annotate(identity, verdict='active', note=note)
