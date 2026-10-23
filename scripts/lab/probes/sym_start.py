"""CPU only: sym_start.py <stage> [seeds]. Is the drift a choice between two mirrored gaits, or built into the physics?
The mirror-symmetric policy (as sym_probe.py) and the stage's own policy walk at 0.5 m/s with no turn command
(heading hold off), starting the gait phase at 0 (left foot first) or half a cycle later (right foot first).
With a symmetric policy and symmetric physics, the half-cycle start is close to the mirror image of the usual one,
so the drift should flip sign if it is a choice between two mirrored gaits; keeping its sign points at the physics."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.symmetry import Mirror  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = Path(sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
model, norm = load_stage(stage)
config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': 600}
mirror = Mirror(make_task('g1_velocity', 'g1_29dof', config))


def act(obs, symmetric):
    own = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    if not symmetric:
        return own
    flipped = mirror.observation(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    other = model.predict(norm.normalize_obs(flipped), deterministic=True)[0]
    return .5 * (own + mirror.action(torch.as_tensor(other, dtype=torch.float32)).numpy())


def drift(symmetric, offset, seed, speed=.5):
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': [speed, 0., 0.]})
    raw.steps = offset
    track = []
    for _ in range(500):
        obs, _, term, _, _ = env.step(act(obs, symmetric))
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if term:
            break
    env.close()
    track = np.unwrap(track)
    return (track[-1] - track[100]) / ((len(track) - 101) * .02)


probe = make_task('g1_velocity', 'g1_29dof', config).unwrapped
half = int(round(probe.gait['period'] / probe.dt / 2))
for symmetric in (True, False):
    for offset in (0, half):
        rates = [drift(symmetric, offset, 73000 + s) for s in range(seeds)]
        print(f'{stage.name} {"symmetric" if symmetric else "own      "} phase start +{offset:2d} steps: '
              f'yaw rate {np.mean(rates):+.4f}  seeds {" ".join(f"{v:+.3f}" for v in rates)}', flush=True)
norm.close()
