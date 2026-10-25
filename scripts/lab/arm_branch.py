"""Branch from a blind-armed G1 stage (latest, or <parent>:<steps>) to arms on targets (task_config arm_targets,
iteration 40, stage 3): arm_branch.py <parent>[:steps] <name> <note> [--dry-run] [--lab <dir>] [--no-queue]

The parent's task with arm_blind false, arm_targets true and the reward weight arm_track ARM_TRACK_WEIGHT, the parent's
PPO recipe; the stage moved by warm_start.unblind_arms: the blind constant of the arm inputs folded into the first
layers' biases, the arm columns' weights zero and their normalization fixed (mean 0; angles 0.6², speeds 2², last
actions 1.5²), so the start acts as the parent within ~1e-5 (printed: the largest action difference on the parent's
reference runs) and does not use its arms yet. New observation and action contracts (...-armtgt, ...-armres), so, as
blind_branch.py, it is created without a source (the tree shows it as a root) and the manifest's warm_start names the
parent stage. Refused before anything is created: a parent that is not blind, any other contract change, a setting the
environment rejects. Queued unless --no-queue. --dry-run: everything above into a temporary folder, nothing in the lab.
--lab: the original run dir of another lab (a check copy in .tools/); the parent is always read from the lab under runs/.

Example (iteration 40): arm_branch.py exp-fa2450e3a6af4bf4:8784248832 'G1 этап 3 · руки по целям' '...' --dry-run"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.warm_start import check_unblind_arms, start_difference, unblind_arms  # noqa: E402

ARM_TRACK_WEIGHT = 2.
PPO_KEYS = ('backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
            'gae_lambda', 'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std',
            'mirror_coef', 'range_bounds')

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('parent')
parser.add_argument('name')
parser.add_argument('note')
parser.add_argument('--dry-run', action='store_true', help='check and move into a temporary folder only')
parser.add_argument('--lab', type=Path, default=DEFAULT_RUN, help='original run dir of the lab to create the branch in')
parser.add_argument('--no-queue', action='store_true', help='create and annotate only')
args = parser.parse_args()
PARENT, _, AT = args.parent.partition(':')
parents = Laboratory(DEFAULT_RUN)
run = parents.resolve(PARENT)
stages = [int(path.name[6:]) for path in list_stages(run)]
if not stages:
    raise SystemExit(f'у {PARENT} пока нет этапов')
stage = int(AT) if AT else stages[-1]
if stage not in stages:
    raise SystemExit(f'у {PARENT} нет этапа {stage}')
parent = read_json(run / 'run_config.json')
weights = {**parent['task_config'].get('reward_weights', {}), 'arm_track': ARM_TRACK_WEIGHT}
task_config = {**parent['task_config'], 'arm_blind': False, 'arm_targets': True, 'reward_weights': weights}
config = {key: parent.get(key, DEFAULTS[key]) for key in PPO_KEYS}
source = run / f'stage_{stage:012d}'
saved = read_json(source / 'metadata.json')
# Refuse before creating anything; unblind_arms checks again.
check_unblind_arms(saved, config['robot_id'], config['task_id'], task_config)
make_task(config['task_id'], config['robot_id'], task_config).close()
config.update(name=f'{args.name} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
note = (f'Руки по целям: arm_blind → false, arm_targets → true, вес arm_track {ARM_TRACK_WEIGHT:g} (новые контракты '
        f'наблюдения …-armtgt и действия …-armres; веса с {PARENT[:12]} @ {stage / 1e6:.0f} млн, постоянные входы рук '
        f'свёрнуты в смещения первых слоёв, веса рук с нуля). {args.note} Остальное как у {PARENT[:12]}.')


def report(moved):
    metadata = read_json(moved / 'metadata.json')
    for field in ('observation_schema', 'action_schema', 'model_revision'):
        if saved.get(field) != metadata[field]:
            print(f'{field}: {saved.get(field)} -> {metadata[field]}')
    print(f'start against the parent: largest action difference {start_difference(source, moved):.2g}')


if args.dry_run:  # what lab.create checks, then the move into a temporary run folder; nothing in the lab
    checked = validate_config(config)
    with tempfile.TemporaryDirectory() as folder:
        write_json(Path(folder) / 'run_config.json', checked)
        report(unblind_arms(source, Path(folder)))
    print('dry run, nothing created:', PARENT, stage, '| name:', checked['name'])
    print('note:', note)
    raise SystemExit(0)
lab = Laboratory(args.lab)
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = unblind_arms(source, path)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage, 'method': 'unblind_arms'}
write_json(path / 'experiment.json', manifest)
if not args.no_queue:
    lab.enqueue([identity])
print(identity, moved)
report(moved)
lab.annotate(identity, verdict='active', note=note)
