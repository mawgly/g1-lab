"""New G1 root from a stage (latest, or <parent>:<steps>) under an observation contract with terms added
(task_config height_command, upper_body_slot), weights moved by warm_start.insert_terms:
insert_root.py <parent>[:steps] <task_config JSON patch> <PPO JSON patch> <name> <note> [--lab <dir>] [--no-queue]

The task patch replaces top-level settings; its groups (reward_weights, commands, gait) are merged key by key into the
parent's. Refused before anything is created: the same observation contract, any other contract change, or a setting
the environment rejects. Created without a source (the tree shows a root); the manifest's warm_start names the parent
stage. Queued like contract_branch.py unless --no-queue. --lab: the original run dir of another lab (a check copy in
.tools/, e.g. .tools/it28-gate/humanoid; its experiments go to …_lab next to it); the parent is always read from the
lab under runs/.

Example (iteration 28): insert_root.py exp-28691193718e4d0e:7221510144 '{"height_command": true, "upper_body_slot":
true, "commands": {"crouch_share": [0.5, 0.33], "crouch_min": [0.64, 0.70], "crouch_speed": 1.0}, "reward_weights":
{"height_ceiling": 2.0}}' '{}' 'G1 этап 2 · присед' '…' (in PowerShell 5.1 escape the quotes: '{\"height_command\": …}')"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402
from humanoid_sim.warm_start import check_insert_terms, insert_terms  # noqa: E402

GROUPS = ('reward_weights', 'commands', 'gait')
PPO_KEYS = ('backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
            'gae_lambda', 'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std',
            'mirror_coef', 'range_bounds')

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('parent')
parser.add_argument('task', type=json.loads)
parser.add_argument('ppo', type=json.loads)
parser.add_argument('name')
parser.add_argument('note')
parser.add_argument('--lab', type=Path, default=DEFAULT_RUN, help='original run dir of the lab to create the root in')
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
task_config = dict(parent['task_config'])
for key, value in args.task.items():
    task_config[key] = {**task_config.get(key, {}), **value} if key in GROUPS and isinstance(value, dict) else value
config = {key: parent.get(key, DEFAULTS[key]) for key in PPO_KEYS}
unknown = set(args.ppo) - set(config)
if unknown:
    raise SystemExit(f'{unknown}: не параметры PPO')
config.update(args.ppo)
source = run / f'stage_{stage:012d}'
# Refuse before creating anything; insert_terms checks the layout again.
check_insert_terms(read_json(source / 'metadata.json'), task_identity(config['robot_id'], config['task_id'], task_config))
make_task(config['task_id'], config['robot_id'], task_config).close()
config.update(name=f'{args.name} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
lab = Laboratory(args.lab)
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = insert_terms(source, path)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage, 'method': 'insert_terms'}
write_json(path / 'experiment.json', manifest)
if not args.no_queue:
    lab.enqueue([identity])
print(identity, moved, read_json(moved / 'warm_start.json')['to_schema'])
lab.annotate(identity, verdict='active', note=f'Новый корень от {PARENT[:12]} @ {stage / 1e6:.0f} млн: во вход добавлены '
             f'термы, веса старых входов перенесены как есть, новые начинаются с нулей. Задача: '
             f'{json.dumps(args.task, ensure_ascii=False)}; PPO: {json.dumps(args.ppo, ensure_ascii=False)}. {args.note}')
