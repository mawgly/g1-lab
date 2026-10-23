"""CPU-only probe (no GPU): noise_probe.py <stage dir> <mode> [seeds]
mode drift: yaw rate at 0.5 m/s, yaw 0, heading hold off (as goal_check), deterministic vs stochastic policy.
mode stand: stand 10 s; deterministic / stochastic sigma x1 / x0.5; how soon rule A (anchor feet at 1 s,
            violation if foot height > 0.05 m or foot moved > 0.03 m) would end the episode; lift-offs; crawl.
mode stopstart: walk 1 m/s 5 s then stand; rule A anchored 1 s after the stand command.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = Path(sys.argv[1])
mode = sys.argv[2]
seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
model, norm = load_stage(stage)
config = stage_task_config(stage)
base_log_std = model.policy.log_std.data.clone()
DT = 0.02


def set_sigma(factor):
    model.policy.log_std.data.copy_(base_log_std + math.log(factor))


def act(obs, deterministic):
    return model.predict(norm.normalize_obs(obs), deterministic=deterministic)[0]


def drift(deterministic, seed):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 500})
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': [.5, 0., 0.]})
    track = [math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0])]
    fell = False
    rewards = []
    for step in range(500):
        obs, reward, terminated, truncated, _ = env.step(act(obs, deterministic))
        rewards.append(float(reward))
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        if terminated:
            fell = True
            break
    env.close()
    track = np.unwrap(track)
    start = min(100, len(track) - 2)
    REWARDS.append(np.mean(rewards))
    return (track[-1] - track[start]) / ((len(track) - 1 - start) * DT), fell


REWARDS = []


def stand(deterministic, seed, walk_first=False):
    seconds = 15 if walk_first else 10
    steps = int(seconds / DT)
    stand_from = 250 if walk_first else 0
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': steps})
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed, options={'command': [1. if walk_first else 0., 0., 0.]})
    anchor, violation, lifts, prev_contact, rewards, fell = None, None, 0, None, [], False
    max_h, max_d = 0., 0.
    start_xy = None
    for step in range(steps):
        if walk_first and step == stand_from:
            raw.set_command([0., 0., 0.])
            obs = raw._observation()
        obs, reward, terminated, truncated, _ = env.step(act(obs, deterministic))
        if step >= stand_from:
            rewards.append(float(reward))
        feet = raw.data.xpos[raw.feet].copy()
        contact = feet[:, 2] < task.CONTACT_HEIGHT
        if step >= stand_from + 50:
            if anchor is None:
                anchor = feet[:, :2].copy()
                start_xy = raw.data.qpos[:2].copy()
            moved = np.linalg.norm(feet[:, :2] - anchor, axis=1).max()
            max_h, max_d = max(max_h, float(feet[:, 2].max())), max(max_d, float(moved))
            if violation is None and (feet[:, 2].max() > .05 or moved > .03):
                violation = (step - stand_from + 1) * DT
            if prev_contact is not None:
                lifts += int((prev_contact & ~contact).sum())
            prev_contact = contact
        if terminated:
            fell = True
            break
    crawl = float(np.linalg.norm(raw.data.qpos[:2] - start_xy)) if start_xy is not None else float('nan')
    env.close()
    return violation, lifts, crawl, float(np.mean(rewards)), fell, max_h, max_d


t0 = time.time()
print(stage.name, mode)
if mode == 'drift':
    for label, det, factor in (('deterministic', True, 1.), ('stochastic x1', False, 1.), ('stochastic x0.5', False, .5)):
        set_sigma(factor)
        n = 2 if det else seeds
        REWARDS.clear()
        rates = [drift(det, 73000 + s) for s in range(n)]
        values = [r for r, _ in rates]
        print(f'  {label:16s} yaw rate mean {np.mean(values):+.4f}  sd {np.std(values):.4f}  '
              f'per seed {" ".join(f"{v:+.3f}" for v in values)}  falls {sum(f for _, f in rates)}  '
              f'reward/step {np.mean(REWARDS):.2f}')
else:
    walk_first = mode == 'stopstart'
    for label, det, factor in (('deterministic', True, 1.), ('stochastic x1', False, 1.), ('stochastic x0.5', False, .5),
                               ('stochastic x0.25', False, .25)):
        set_sigma(factor)
        rows = [stand(det, 73000 + s, walk_first) for s in range(seeds)]
        viol = [r[0] for r in rows]
        print(f'  {label:16s} rule-A ends at (s after stand cmd) {" ".join("-" if v is None else f"{v:.2f}" for v in viol)} | '
              f'lifts {" ".join(str(r[1]) for r in rows)} | crawl {np.mean([r[2] for r in rows]):.2f} m | '
              f'reward/step {np.mean([r[3] for r in rows]):.2f} | falls {sum(r[4] for r in rows)} | '
              f'max foot h {max(r[5] for r in rows):.3f} max move {max(r[6] for r in rows):.3f}')
print(f'  ({time.time() - t0:.0f} s)')
norm.close()
