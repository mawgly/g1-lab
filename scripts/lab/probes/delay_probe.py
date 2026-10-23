"""CPU only: delay_probe.py <stage> [seeds]. Walking under a fixed motor-target delay (task_config action_delay, set
per episode through reset options action_delay_steps; 1 step = one 2 ms physics substep): straight 0.5 and 1 m/s for
10 s, heading hold off, start phase 0, deterministic. Prints forward speed (m/s, over 10 s), yaw rate after 2 s
(rad/s, as goal_check) and falls for delays 0 / 10 / 20 ms. Acceptance for iteration 9: at 20 ms, 1 m/s speed
>= 0.9 m/s and |yaw rate| < 0.05."""
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
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
model, norm = load_stage(stage)
# The probe sets the delay itself, so the env allows the largest one asked for (20 ms).
config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': 500, 'action_delay': .02}


def walk(speed, delay, seed):
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': [speed, 0., 0.], 'action_delay_steps': delay})
    start = raw.data.xpos[pelvis][:2].copy()
    track, fell, steps = [], False, 0
    for steps in range(1, 501):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if term:
            fell = True
            break
    forward = float(raw.data.xpos[pelvis][0] - start[0])
    env.close()
    track = np.unwrap(track)
    rate = (track[-1] - track[100]) / ((len(track) - 101) * .02) if len(track) > 101 else float('nan')
    return forward / (steps * .02), rate, fell


for delay in (0, 5, 10):
    parts = []
    for speed in (.5, 1.):
        rows = [walk(speed, delay, 73000 + s) for s in range(seeds)]
        parts.append(f'{speed} m/s: speed {np.mean([r[0] for r in rows]):.2f} yaw {np.mean([r[1] for r in rows]):+.3f} '
                     f'(worst {max(abs(r[1]) for r in rows):.3f}) falls {sum(r[2] for r in rows)}')
    print(f'{stage.parent.name[:12]} {stage.name} delay {delay * 2:2d} ms | ' + ' | '.join(parts), flush=True)
norm.close()
