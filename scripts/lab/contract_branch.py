"""Branch from the latest stage of a G1 experiment (or <parent>:<steps>) with one top-level task setting changed that
changes the policy contract but not its layout (lock_wrists, phase_stop, knee_floor; foot_collision changes the model
revision, warm_start.CARRIED_MODEL_SUFFIXES), weights carried over unchanged (warm_start.carry_over):
contract_branch.py <parent>[:steps] <key> <JSON value> <name> <note>.

An ordinary branch (task_branch.py) is refused there by validate_resume; this one is created without a source and
gets the carried-over stage, the manifest's warm_start records where it came from. Queued like task_branch.py.

Example: contract_branch.py exp-14bae200a11f4856:5461377024 lock_wrists true 'G1 реальный · запястья закреплены' '…'"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.config import g1_config  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402
from humanoid_sim.warm_start import carry_over, check_carry_over  # noqa: E402

PARENT, KEY, VALUE, NAME, NOTE = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]
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
before = g1_config(parent['task_config'])[KEY]  # the parent's effective value, defaults included
task_config = {**parent['task_config'], KEY: VALUE}
source = run / f'stage_{stage:012d}'
# Refuse before creating anything: the same contract (an ordinary branch, task_branch.py), an incompatible one, or a
# setting the environment rejects. carry_over checks the layout again.
check_carry_over(read_json(source / 'metadata.json'), task_identity(parent['robot_id'], parent['task_id'], task_config))
make_task(parent['task_id'], parent['robot_id'], task_config).close()
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config=task_config)
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = carry_over(source, path)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage}
write_json(path / 'experiment.json', manifest)
lab.enqueue([identity])
print(identity, moved.name, f'{KEY} {before} -> {VALUE}')
lab.annotate(identity, verdict='active', note=f'Задача: {KEY} {before} → {VALUE} (новый контракт, веса перенесены как '
             f'есть с {PARENT[:12]} @ {stage / 1e6:.0f} млн). {NOTE} Остальное как у {PARENT[:12]}.')
