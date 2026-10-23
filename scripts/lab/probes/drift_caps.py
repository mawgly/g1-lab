"""CPU only: drift_caps.py <stage> [seeds]. Yaw rate (rad/s, after 2 s, as goal_check) at 0.5 and 1 m/s, yaw 0,
heading hold off: deterministic vs stochastic with sigma capped at 0.5 / 0.3. Plus, heading hold on (as in training),
stochastic at the current sigma: steady heading error and the yaw command the policy sees, and the pelvis
yaw-rate oscillation (rms of per-step yaw rate around its mean) that a per-step |omega| penalty would hit."""
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
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
model, norm = load_stage(stage)
config = stage_task_config(stage)
base = model.policy.log_std.data.clone()
print(stage.name, 'sigma geo-mean', float(base.exp().log().mean().exp()), 'max', float(base.exp().max()))


def episode(speed, hold, det, seed):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': hold, 'max_episode_steps': 500})
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    obs, _ = env.reset(seed=seed, options={'command': [speed, 0., 0.]})
    track, rates, cmds, fell = [], [], [], False
    for _ in range(500):
        obs, r, term, trunc, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=det)[0])
        track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
        rates.append(float(raw.data.qvel[5]))
        cmds.append(float(info['command'][2]) if 'command' in info else float('nan'))
        if term:
            fell = True
            break
    env.close()
    track = np.unwrap(track)
    rate = (track[-1] - track[100]) / ((len(track) - 101) * .02)
    err = float(np.mean(np.unwrap(track[100:]) - track[0]))  # heading relative to the start (target is the start)
    osc = float(np.std(rates[100:]))
    return rate, err, float(np.nanmean(cmds[100:])), osc, fell


for speed in (.5, 1.):
    for label, det, cap in (('det', True, 1.), ('cap0.5', False, .5), ('cap0.3', False, .3)):
        model.policy.log_std.data.copy_(base.clamp(max=math.log(cap)))
        rows = [episode(speed, False, det, 73000 + s) for s in range(2 if det else seeds)]
        rates = [r[0] for r in rows]
        print(f'  {speed} m/s hold off {label:7s} yaw rate {np.mean(rates):+.4f} sd {np.std(rates):.4f} '
              f'seeds {" ".join(f"{v:+.3f}" for v in rates)} | per-step yaw-rate rms {np.mean([r[3] for r in rows]):.3f} '
              f'| falls {sum(r[4] for r in rows)}')
model.policy.log_std.data.copy_(base)
for label, det in (('det', True), ('cur sigma', False)):
    rows = [episode(.5, True, det, 73000 + s) for s in range(2 if det else 5)]
    print(f'  0.5 m/s hold ON {label:9s} mean heading err {np.degrees(np.mean([r[1] for r in rows])):+.1f} deg, '
          f'seen yaw cmd {np.mean([r[2] for r in rows]):+.4f}, yaw rate {np.mean([r[0] for r in rows]):+.4f}')
norm.close()
