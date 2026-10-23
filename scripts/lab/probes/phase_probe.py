"""CPU only, ~30 s: current stage with phase_stop on (phase input (0,0) when standing), before any training.
Stand 10 s and stop-start (1 m/s 5 s -> stand 5 s -> 1 m/s 5 s); deterministic and sigma capped at 0.5."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

stage = Path(sys.argv[1])
model, norm = load_stage(stage)
base = model.policy.log_std.data.clone()


def run(phase_stop, schedule, det, seed):
    steps = int(schedule[-1][0] / .02)
    env = make_task('g1_velocity', 'g1_29dof', {**stage_task_config(stage), 'phase_stop': phase_stop,
                                                 'max_episode_steps': steps})
    raw = env.unwrapped
    obs, _ = env.reset(seed=seed, options={'command': list(schedule[0][1])})
    lifts, prev, fell, start = 0, None, False, None
    for step in range(steps):
        t = step * .02
        cmd = [c for s, c in schedule if s <= t][-1]
        raw.set_command(list(cmd))
        obs = raw._observation()
        obs, r, term, trunc, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=det)[0])
        standing = not any(cmd)
        contact = raw.data.xpos[raw.feet, 2] < task.CONTACT_HEIGHT
        if standing and t >= ([s for s, c in schedule if s <= t][-1] + 1.):
            if start is None:
                start = raw.data.qpos[:2].copy()
            if prev is not None:
                lifts += int((prev & ~contact).sum())
        prev = contact
        if term:
            fell = True
            break
    crawl = float(np.linalg.norm(raw.data.qpos[:2] - start)) if start is not None else float('nan')
    env.close()
    return fell, lifts, crawl


STAND = ((0., (0., 0., 0.)), (10., None))
STOP = ((0., (1., 0., 0.)), (5., (0., 0., 0.)), (10., (1., 0., 0.)), (15., None))
for label, det, cap in (('det', True, 1.), ('cap0.5', False, .5)):
    model.policy.log_std.data.copy_(base.clamp(max=math.log(cap)))
    for ps in (False, True):
        for name, sched in (('stand', STAND), ('stopstart', STOP)):
            sched = tuple((s, c if c is not None else (0., 0., 0.)) for s, c in sched)
            rows = [run(ps, sched, det, 73000 + s) for s in range(3)]
            print(f'{label:7s} phase_stop={ps!s:5s} {name:9s} falls {sum(r[0] for r in rows)}/3 '
                  f'lifts {" ".join(str(r[1]) for r in rows)} crawl {np.nanmean([r[2] for r in rows]):.2f} m')
norm.close()
