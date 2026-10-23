"""CPU only: arcs_probe.py <stage> <policy: own|sym> [speed]. Two mirrored arcs, a slow mode, or one biased gait?
Deterministic policy, no turn command, heading hold off, zero reset noise (joints exactly at home), long episodes
(32 s). Phase start 0 (left foot first) or half a cycle later. Optional yaw kick at 12 s (pelvis yaw rate += kick).
Prints the mean yaw rate (rad/s, from heading) in windows 2-10 s, 14-22 s, 22-32 s."""
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
import mujoco  # noqa: E402

stage = Path(sys.argv[1])
symmetric = sys.argv[2] == 'sym'
speed = float(sys.argv[3]) if len(sys.argv) > 3 else .5
model, norm = load_stage(stage)
STEPS = 1600
config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': STEPS + 10}
mirror = Mirror(make_task('g1_velocity', 'g1_29dof', config))


def act(obs):
    own = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    if not symmetric:
        return own
    flipped = mirror.observation(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    other = model.predict(norm.normalize_obs(flipped), deterministic=True)[0]
    return .5 * (own + mirror.action(torch.as_tensor(other, dtype=torch.float32)).numpy())


def run(offset, kick):
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    env.reset(seed=73000, options={'command': [speed, 0., 0.]})
    raw.data.qpos[raw.qadr] = raw.home  # exactly symmetric start
    raw.data.qvel[:] = 0
    mujoco.mj_forward(raw.model, raw.data)
    raw.steps = offset
    obs = raw._observation('fill')
    track = []
    for t in range(STEPS):
        if kick and t == 600:
            raw.data.qvel[5] += kick
        obs, _, term, _, _ = env.step(act(obs))
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if term:
            break
    env.close()
    track = np.unwrap(track)
    def rate(a, b):
        a, b = int(a / .02), min(int(b / .02), len(track) - 1)
        return (track[b] - track[a]) / ((b - a) * .02) if b > a else float('nan')
    return rate(2, 10), rate(14, 22), rate(22, 32), len(track)


probe = make_task('g1_velocity', 'g1_29dof', config).unwrapped
half = int(round(probe.gait['period'] / probe.dt / 2))
for offset, kick in ((0, 0.), (half, 0.), (0, -.78), (0, .78), (0, -.3)):
    a, b, c, n = run(offset, kick)
    print(f'{stage.name} {"sym" if symmetric else "own"} v={speed} phase+{offset:2d} kick {kick:+.2f}: '
          f'2-10s {a:+.4f} | 14-22s {b:+.4f} | 22-32s {c:+.4f} | steps {n}', flush=True)
norm.close()
