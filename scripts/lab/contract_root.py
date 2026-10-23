"""New G1 root under a changed contract with the same layout, weights carried over (warm_start.carry_over):
contract_root.py <parent> <task_config JSON patch> <PPO JSON patch> <name> <note>

Example: contract_root.py exp-… '{"phase_stop": true}' '{"max_std": 0.5}' 'G1 реальный · фаза как у Unitree' '…'"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, write_json  # noqa: E402
from humanoid_sim.lab import DEFAULTS, Laboratory  # noqa: E402
from humanoid_sim.warm_start import carry_over  # noqa: E402

PARENT, TASK, PPO, NAME, NOTE = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]
lab = Laboratory(DEFAULT_RUN)
source = list_stages(lab.resolve(PARENT))[-1]
stage = int(source.name[6:])
parent = read_json(lab.root / PARENT / 'run_config.json')
config = {key: parent.get(key, DEFAULTS[key]) for key in (
    'backend', 'envs', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
    'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef', 'range_bounds')}
unknown = set(PPO) - set(config)
if unknown:
    raise SystemExit(f'{unknown}: не параметры PPO')
config.update(PPO)
config.update(name=f'{NAME} (перенос с {stage / 1e6:.0f} млн)', preset='custom', steps=1_000_000_000, hours=3,
              every=10_000_000, task_config={**parent['task_config'], **TASK})
entry = lab.create(config)
identity = entry['id']
path = lab.root / identity
moved = carry_over(source, path)
manifest = read_json(path / 'experiment.json')
manifest['warm_start'] = {'source_id': PARENT, 'source_stage': stage}
write_json(path / 'experiment.json', manifest)
lab.enqueue([identity])
print(identity, moved.name)
lab.annotate(identity, verdict='active', note=f'Новый корень от {PARENT[:12]} @ {stage / 1e6:.0f} млн, веса перенесены как '
             f'есть. Задача: {json.dumps(TASK, ensure_ascii=False)}; PPO: {json.dumps(PPO, ensure_ascii=False)}. {NOTE}')
