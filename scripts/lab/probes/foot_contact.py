"""CPU only: foot_contact.py <stage> [speed] [phase offset steps]. True foot contact vs what the reward counts.
Own policy, deterministic, straight, heading hold off, zero reset noise, steps 100-600. Per foot: share of steps with a
real MuJoCo contact (a foot sphere touching the floor), share counted as contact by the reward (ankle-roll origin
< CONTACT_HEIGHT), true slip = mean horizontal foot speed while really in contact (m/s), share of really-in-contact
steps that the reward misses, sole clearance (lowest sphere bottom) 95th percentile (m), ankle origin height mean."""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
import mujoco  # noqa: E402

stage = Path(sys.argv[1])
speed = float(sys.argv[2]) if len(sys.argv) > 2 else .5
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
model, norm = load_stage(stage)
config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': 700}
env = make_task('g1_velocity', 'g1_29dof', config)
raw = env.unwrapped
spheres = [set(g for g in range(raw.model.ngeom) if raw.model.geom_bodyid[g] == f and raw.model.geom_contype[g])
           for f in raw.feet]
env.reset(seed=73000, options={'command': [speed, 0., 0.]})
raw.data.qpos[raw.qadr] = raw.home
raw.data.qvel[:] = 0
mujoco.mj_forward(raw.model, raw.data)
raw.steps = offset
obs = raw._observation('fill')
real, counted, slip, clear, origin = [[], []], [[], []], [[], []], [[], []], [[], []]
prev = raw.data.xpos[raw.feet, :2].copy()
for t in range(600):
    obs, _, term, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
    now = raw.data.xpos[raw.feet, :2].copy()
    if t >= 100:
        touching = [False, False]
        for c in raw.data.contact[:raw.data.ncon]:
            for k in range(2):
                if (c.geom1 in spheres[k] or c.geom2 in spheres[k]) and c.dist < .002:
                    touching[k] = True
        for k, f in enumerate(raw.feet):
            real[k].append(touching[k])
            counted[k].append(raw.data.xpos[f][2] < task.CONTACT_HEIGHT)
            if touching[k]:
                slip[k].append(np.linalg.norm(now[k] - prev[k]) / raw.dt)
            clear[k].append(min(raw.data.geom_xpos[g][2] - raw.model.geom_size[g][0] for g in spheres[k]))
            origin[k].append(raw.data.xpos[f][2])
    prev = now
    if term:
        break
env.close()
norm.close()
print(f'{stage.name} v={speed} phase+{offset}')
for k, name in enumerate(('left', 'right')):
    r, c = np.array(real[k]), np.array(counted[k])
    missed = float(np.mean(~c[r])) if r.any() else float('nan')
    print(f'  {name:5s}: real contact {r.mean():.2f}, counted {c.mean():.2f}, missed {missed:.2f}, '
          f'true slip {np.mean(slip[k]) if slip[k] else float("nan"):.3f} m/s, sole clearance p95 '
          f'{np.percentile(clear[k], 95):.3f} m max {np.max(clear[k]):.3f}, origin mean {np.mean(origin[k]):.3f} m')
