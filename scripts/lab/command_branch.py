"""Branch from a stage (default: the latest) of a G1 experiment with command settings changed (one change: settings
that only act together, such as side_share with side_stop, go in one call):
command_branch.py <parent>[:steps] <key> <value> [<key> <value> ...] <name> <note>."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402
from humanoid_sim.tasks.config import G1_COMMANDS  # noqa: E402

PARENT, PAIRS, NAME, NOTE = sys.argv[1], sys.argv[2:-2], sys.argv[-2], sys.argv[-1]
if not PAIRS or len(PAIRS) % 2:
    raise SystemExit(__doc__)
CHANGES = {key: float(value) for key, value in zip(PAIRS[::2], PAIRS[1::2])}
for key in CHANGES:
    if key not in G1_COMMANDS:
        raise SystemExit(f'{key}: не настройка команд')
PARENT, _, chosen = PARENT.partition(':')
lab = Laboratory(DEFAULT_RUN)
stage = int(chosen) if chosen else int(list_stages(lab.resolve(PARENT))[-1].name[6:])
parent = read_json(lab.root / PARENT / 'run_config.json')
commands = {**G1_COMMANDS, **parent['task_config'].get('commands', {})}
change = ', '.join(f'{key} {commands[key]:g} → {value:g}' for key, value in CHANGES.items())
task_config = {**parent['task_config'], 'commands': {**commands, **CHANGES}}
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'every', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
    'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
config.update(name=f'{NAME} (от {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3, every=10_000_000,
              source_id=PARENT, source_stage=stage, task_config=task_config)
entry = lab.create(config)
identity = entry['id']
lab.enqueue([identity])
print(identity, stage, change.replace('→', '->'))
lab.annotate(identity, verdict='active', note=f'Команды: {change}. {NOTE} Остальное как у {PARENT[:12]}.')
