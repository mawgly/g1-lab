"""How a G1 stage stands: stand_probe.py <branch:steps> [seeds]. Steps (lift-offs), sliding of planted feet,
pelvis drift per second and where the drift happens (first 2 s vs the rest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, load_stage, stage_task_config  # noqa: E402
from humanoid_sim.goal_check import find_stage  # noqa: E402
from humanoid_sim.lab import Laboratory  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = find_stage(Laboratory(DEFAULT_RUN), sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
policy, norm = load_stage(stage)
config = stage_task_config(stage)
rows = []
for seed in range(73000, 73000 + seeds):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 500})
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.]})
    lifts, slide, path, contact_before = 0, 0., [raw.data.qpos[:2].copy()], None
    feet_before = raw.data.xpos[raw.feet, :2].copy()
    for step in range(500):
        obs, *_ = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        contact = raw.data.xpos[raw.feet, 2] < task.CONTACT_HEIGHT
        feet = raw.data.xpos[raw.feet, :2].copy()
        slide += float((np.linalg.norm(feet - feet_before, axis=1) * contact).sum())
        if contact_before is not None:
            lifts += int((contact_before & ~contact).sum())
        contact_before, feet_before = contact, feet
        path.append(raw.data.qpos[:2].copy())
    path = np.array(path)
    early, late = np.linalg.norm(path[100] - path[0]), np.linalg.norm(path[-1] - path[100])
    rows.append((lifts, slide, early, late, *(path[-1] - path[0])))
    env.close()
for seed, (lifts, slide, early, late, dx, dy) in zip(range(73000, 73000 + seeds), rows):
    print(f'seed {seed}: отрывов стоп {lifts}, скольжение стоп на земле {slide:.2f} м, '
          f'таз за 0–2 с {early:.2f} м, за 2–10 с {late:.2f} м, итог x {dx:+.2f} y {dy:+.2f}')
