"""Left/right foot pitch and roll (deg) while standing, deterministic; CPU only."""
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
env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 500})
raw = env.unwrapped
obs, _ = env.reset(seed=73000, options={'command': [0., 0., 0.]})
pitch, roll, yaw0, yaws = [], [], None, []
for _ in range(500):
    obs, *_ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
    m = raw.data.xmat[raw.feet].reshape(-1, 3, 3)
    pitch.append(np.degrees(np.arcsin(-m[:, 2, 0])))  # x axis z-component: toe up/down
    roll.append(np.degrees(np.arcsin(m[:, 2, 1])))
    y = np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0]))
    yaw0 = y if yaw0 is None else yaw0
    yaws.append(y - yaw0)
env.close()
pitch, roll, yaws = map(np.array, (pitch, roll, yaws))
print(stage.name, 'median pitch L/R (+ = toe up)', np.median(pitch, 0).round(1), ' roll L/R', np.median(roll, 0).round(1),
      ' foot yaw change over 10 s L/R', yaws[-1].round(1), ' max |yaw change|', np.abs(yaws).max(0).round(1))
norm.close()
