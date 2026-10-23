"""Branch from the latest stage of a G1 experiment (or <parent>:<steps>) with one PPO setting changed:
ppo_branch.py <parent>[:steps] <key> <value> <name> <note>."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402

PARENT, KEY, VALUE, NAME, NOTE = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
if KEY not in DEFAULTS or KEY in ('task_config', 'name', 'source_id', 'source_stage'):
    raise SystemExit(f'{KEY}: не параметр PPO')
PARENT, _, AT = PARENT.partition(':')
lab = Laboratory(DEFAULT_RUN)
stages = [int(path.name[6:]) for path in list_stages(lab.resolve(PARENT))]
stage = int(AT) if AT else stages[-1]
if stage not in stages:
    raise SystemExit(f'у {PARENT} нет этапа {stage}')
parent = read_json(lab.root / PARENT / 'run_config.json')
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
    'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
before = config.get(KEY)
config[KEY] = type(DEFAULTS[KEY])(VALUE) if DEFAULTS[KEY] is not None else VALUE
config.update(name=f'{NAME} (от {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, source_id=PARENT, source_stage=stage, task_config=parent['task_config'])
entry = lab.create(config)
identity = entry['id'] if isinstance(entry, dict) else entry
lab.enqueue([identity])
print(identity, stage, f'{KEY} {before} -> {config[KEY]}')
lab.annotate(identity, verdict='active', note=f'{KEY} {before} → {config[KEY]}. {NOTE} Остальное как у {PARENT[:12]}.')
