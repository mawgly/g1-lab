"""CPU only: policy sigma capped (as max_std would do) -> stand reward/step and drift at 0.5 m/s (heading hold off)."""
import math
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
base = model.policy.log_std.data.clone()


def episode(command, hold, seed):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': hold, 'max_episode_steps': 500})
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': command})
    track, rewards = [], []
    for _ in range(500):
        obs, r, term, trunc, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=False)[0])
        rewards.append(float(r))
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if term:
            break
    env.close()
    track = np.unwrap(track)
    return float(np.mean(rewards)), (track[-1] - track[100]) / ((len(track) - 101) * .02)


for cap in (1., .5, .3):
    model.policy.log_std.data.copy_(base.clamp(max=math.log(cap)))
    stand = [episode([0., 0., 0.], True, 73000 + s)[0] for s in range(3)]
    walk = [episode([.5, 0., 0.], False, 73000 + s) for s in range(8)]
    print(f'cap {cap}: stand reward/step {np.mean(stand):+.2f} | 0.5 m/s reward/step {np.mean([w[0] for w in walk]):+.2f} '
          f'yaw rate {np.mean([w[1] for w in walk]):+.4f} sd {np.std([w[1] for w in walk]):.4f}')
norm.close()
