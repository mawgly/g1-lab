"""Foot (ankle-roll origin) height and tilt while standing, deterministic policy. CPU only."""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = Path(sys.argv[1])
model, norm = load_stage(stage)
config = stage_task_config(stage)
for seed in range(73000, 73002):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 500})
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.]})
    h0 = raw.data.xpos[raw.feet, 2].copy()
    heights, tilts = [], []
    for _ in range(500):
        obs, *_ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        heights.append(raw.data.xpos[raw.feet, 2].copy())
        tilts.append(np.degrees(np.arccos(np.clip(raw.data.xmat[raw.feet][:, 8], -1, 1))))
    env.close()
    h, t = np.array(heights), np.array(tilts)
    print(f'seed {seed}: reset height {h0.round(3)}; height pct 10/50/90 L {np.percentile(h[:,0],[10,50,90]).round(3)} '
          f'R {np.percentile(h[:,1],[10,50,90]).round(3)}; share > 0.05 L {np.mean(h[:,0]>.05):.2f} R {np.mean(h[:,1]>.05):.2f}; '
          f'tilt deg median L {np.median(t[:,0]):.1f} R {np.median(t[:,1]):.1f}, 90% L {np.percentile(t[:,0],90):.1f} R {np.percentile(t[:,1],90):.1f}')
norm.close()
