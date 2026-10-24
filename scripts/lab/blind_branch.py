"""Branch from a G1 stage (latest, or <parent>:<steps>) with blind arms (task_config arm_blind, iteration 31):
blind_branch.py <parent>[:steps] <name> <note> [--lab <dir>] [--no-queue]

The parent's task with arm_blind true and its policy as it is; normalize.pkl with the arm columns (joint angles,
velocities and last actions of joints 15-28, every history frame) at variance 1e12 and mean -z*1e6, z the parent stage's
own mean normalized arm inputs standing, at 0.5 and at 1 m/s (warm_start.blind_reference, reset seed 60000), so the
policy sees those constants whatever its arms do; training keeps them frozen. A new observation contract (...-armblind),
so, as contract_branch.py, it is created without a source (the tree shows it as a root) and the manifest's warm_start
names the parent stage. Refused before anything is created: a parent that is blind already or not the real-robot
profile, a setting the environment rejects, a parent that falls in a reference run. Queued unless --no-queue. --lab:
the original run dir of another lab (a check copy in .tools/, e.g. .tools/it31-gate/humanoid; its experiments go to
..._lab next to it); the parent is always read from the lab under runs/.

Example (iteration 31): blind_branch.py exp-04f74e3838184b82:7741440000 'G1 этап 2 · слепые руки' '...'"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.config import g1_config  # noqa: E402
from humanoid_sim.warm_start import blind_arms, blind_reference, check_blind_arms  # noqa: E402

PPO_KEYS = ('backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
            'gae_lambda', 'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std',
            'mirror_coef', 'range_bounds')

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('parent')
parser.add_argument('name')
parser.add_argument('note')
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
before = g1_config(parent['task_config'])['arm_blind']  # the parent's effective value, defaults included
task_config = {**parent['task_config'], 'arm_blind': True}
config = {key: parent.get(key, DEFAULTS[key]) for key in PPO_KEYS}
source = run / f'stage_{stage:012d}'
# Refuse before creating anything; blind_arms checks again.
check_blind_arms(read_json(source / 'metadata.json'), config['robot_id'], config['task_id'], task_config)
make_task(config['task_id'], config['robot_id'], task_config).close()
z, _ = blind_reference(source)
config.update(name=f'{args.name} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
lab = Laboratory(args.lab)
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = blind_arms(source, path, z)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage, 'method': 'blind_arms'}
write_json(path / 'experiment.json', manifest)
if not args.no_queue:
    lab.enqueue([identity])
print(identity, moved, read_json(moved / 'warm_start.json')['to_schema'], f'arm_blind {before} -> True',
      f'|z| max {abs(z).max():.2f}, mean {abs(z).mean():.2f} over {len(z)} columns')
lab.annotate(identity, verdict='active', note=f'Слепые руки: arm_blind {before} → true (новый контракт наблюдения '
             f'…-armblind; веса как есть с {PARENT[:12]} @ {stage / 1e6:.0f} млн, входы рук — постоянные числа '
             f'этого этапа). {args.note} Остальное как у {PARENT[:12]}.')
