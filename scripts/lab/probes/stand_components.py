"""Mean reward components per step while standing 10 s: deterministic vs stochastic (sigma x1, x0.5). CPU only."""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = Path(sys.argv[1])
command = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0., 0., 0.]
model, norm = load_stage(stage)
config = stage_task_config(stage)
base = model.policy.log_std.data.clone()
table = {}
for label, det, factor in (('det', True, 1.), ('x1', False, 1.), ('x0.5', False, .5)):
    model.policy.log_std.data.copy_(base + math.log(factor))
    sums, n = defaultdict(float), 0
    for seed in range(73000, 73003):
        env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 500})
        obs, _ = env.reset(seed=seed, options={'command': command})
        for _ in range(500):
            obs, r, term, trunc, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=det)[0])
            for k, v in info['reward_components'].items():
                sums[k] += v
            n += 1
            if term:
                break
        env.close()
    table[label] = {k: v / n for k, v in sums.items()}
keys = [k for k in table['det'] if any(abs(table[l][k]) > 1e-4 for l in table)]
print(stage.name, 'command', command)
print(f'{"term":14s} {"det":>8s} {"x1":>8s} {"x0.5":>8s}')
for k in keys:
    print(f'{k:14s} {table["det"][k]:8.3f} {table["x1"][k]:8.3f} {table["x0.5"][k]:8.3f}')
print(f'{"TOTAL":14s} {sum(table["det"].values()):8.3f} {sum(table["x1"].values()):8.3f} {sum(table["x0.5"].values()):8.3f}')
norm.close()
