"""CPU only: sym_probe.py <stage> [seeds]. Does a mirror-symmetric policy by construction drift?
Deterministic action a = ½·(π(s) + M_a·π(M_s·s)) (states mirrored before normalization, as the mirror loss does)
against the stage's own π(s): yaw rate without a turn command (rad/s after 2 s, heading hold off, as goal_check)
at 0.5 and 1 m/s, distance at 1 m/s, the turn at ±0.5 rad/s, standing displacement over 10 s, falls."""
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
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
model, norm = load_stage(stage)
config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': 500}
mirror = Mirror(make_task('g1_velocity', 'g1_29dof', config))


def act(obs, symmetric):
    own = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    if not symmetric:
        return own
    flipped = mirror.observation(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    other = model.predict(norm.normalize_obs(flipped), deterministic=True)[0]
    return .5 * (own + mirror.action(torch.as_tensor(other, dtype=torch.float32)).numpy())


def episode(command, symmetric, seed):
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': command})
    start = raw.data.xpos[pelvis][:2].copy()
    track, fell = [], False
    for _ in range(500):
        obs, _, term, _, _ = env.step(act(obs, symmetric))
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if term:
            fell = True
            break
    moved = float(np.linalg.norm(raw.data.xpos[pelvis][:2] - start))
    env.close()
    track = np.unwrap(track)
    rate = (track[-1] - track[100]) / ((len(track) - 101) * .02) if len(track) > 101 else float('nan')
    return rate, moved, fell


for symmetric in (False, True):
    label = 'symmetric' if symmetric else 'own     '
    parts = []
    for name, command, value in (('drift 0.5', [.5, 0., 0.], 0), ('drift 1.0', [1., 0., 0.], 0),
                                 ('dist 1.0', [1., 0., 0.], 1), ('turn +0.5', [.5, 0., .5], 0),
                                 ('turn -0.5', [.5, 0., -.5], 0), ('stand', [0., 0., 0.], 1)):
        rows = [episode(command, symmetric, 73000 + s) for s in range(seeds)]
        values = [r[value] for r in rows]
        falls = sum(r[2] for r in rows)
        parts.append(f'{name} {np.mean(values):+.3f}' + (f' (falls {falls})' if falls else ''))
    print(stage.name, label, ' | '.join(parts), flush=True)
norm.close()
