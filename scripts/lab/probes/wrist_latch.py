"""CPU only: wrist_latch.py <stage> [--out x.json] [--seeds 5]. The wrist-pitch "latch" (iteration 14; the critic's
latch_modes / latch_modes2 probes in one file): candidate 2 (exp-e4da6963 @ 5300.9 M) always keeps exactly one wrist
pitch bent to the edge of the action box (+40°), the left one after a training reset; moved in by 4-8°, the bend jumps
to the other wrist; both at home, it falls standing (2 of 5) or turns 30-70°. The acceptance check of arm_perturb.

A robot-like entry: stand 3 s (command 0), then a straight command v for 10 s (v = 0: keep standing), heading hold
off, deterministic policy, seeds from 73000. The command reaches the observation one step later, as a stick on the
robot. Modes:
  left      the policy as is;
  right     for the first 1 s the left wrist pitch action is 0 and the right one +action_limit, then free;
  bothhome  both wrist pitch actions 0 throughout.
Per run: which wrist is bent (more than 25° from home) at the walk start and at the end, the fall time (s from the
reset), speed over the walk (m/s, along the heading it started at), yaw rate after its first 2 s (rad/s, as
goal_check) and over the whole walk, heading change (°). Summary per mode and command."""
import argparse
import json
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODES = ('left', 'right', 'bothhome')
SPEEDS = (0., .5, 1.)
STAND_STEPS, WALK_STEPS, FORCE_STEPS, SETTLE_STEPS = 150, 500, 50, 100
BENT = 25.  # degrees from home; a latched wrist sits at +40


def fmt(value, spec):
    return 'n/a' if value is None else format(value, spec)


def bent(angles):
    left, right = (angle > BENT for angle in angles)
    return {(True, False): 'left', (False, True): 'right', (True, True): 'both'}.get((left, right), 'none')


def run(job):
    import torch
    torch.set_num_threads(1)
    from humanoid_sim.core import load_stage, stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    stage, mode, speed, seed = job
    model, norm = load_stage(Path(stage))
    config = {**stage_task_config(Path(stage)), 'heading_hold': False, 'max_episode_steps': 100000}
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    names = [raw.model.joint(j).name for j in raw.joints]
    left, right = names.index('left_wrist_pitch_joint'), names.index('right_wrist_pitch_joint')

    def wrists():
        angle = np.degrees(raw.data.qpos[raw.qadr] - raw.home)
        return [round(float(angle[left]), 1), round(float(angle[right]), 1)]

    obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.]})
    track, start, fell, walk_start = [], None, None, None
    for step in range(STAND_STEPS + WALK_STEPS):
        if step == STAND_STEPS:
            raw.set_command([speed, 0., 0.])
            walk_start = wrists()
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        if mode == 'right' and step < FORCE_STEPS:
            action[left], action[right] = 0., raw.action_limit
        elif mode == 'bothhome':
            action[left] = action[right] = 0.
        obs, _, terminated, _, _ = env.step(action)
        if step >= STAND_STEPS:
            if start is None:
                start = raw.data.qpos[:2].copy()
            track.append(math.atan2(raw.data.xmat[raw.pelvis][3], raw.data.xmat[raw.pelvis][0]))
        if terminated:
            fell = round((step + 1) * raw.dt, 2)
            break
    walk_end = wrists()
    out = dict(mode=mode, v=speed, seed=seed, fell=fell, wrists_walk_start=walk_start, wrists_end=walk_end,
               bent_walk_start=None if walk_start is None else bent(walk_start), bent_end=bent(walk_end))
    if len(track) > SETTLE_STEPS + 10:
        track = np.unwrap(track)
        seconds = (len(track) - 1) * raw.dt
        moved = raw.data.qpos[:2] - start
        out.update(speed=float((moved[0] * math.cos(track[0]) + moved[1] * math.sin(track[0])) / seconds),
                   yaw=float((track[-1] - track[SETTLE_STEPS]) / ((len(track) - 1 - SETTLE_STEPS) * raw.dt)),
                   yaw_walk=float((track[-1] - track[0]) / seconds), heading=float(math.degrees(track[-1] - track[0])))
    env.close()
    norm.close()
    return out


def summary(runs):
    rows = []
    for mode in MODES:
        for speed in SPEEDS:
            group = [r for r in runs if r['mode'] == mode and r['v'] == speed]
            measured = [r for r in group if 'yaw' in r]
            rows.append(dict(mode=mode, v=speed, runs=len(group), falls=sum(r['fell'] is not None for r in group),
                             bent_walk_start=dict(Counter(r['bent_walk_start'] for r in group)),
                             bent_end=dict(Counter(r['bent_end'] for r in group)),
                             speed=float(np.mean([r['speed'] for r in measured])) if measured else None,
                             yaw_mean=float(np.mean([r['yaw'] for r in measured])) if measured else None,
                             yaw_worst=max((r['yaw'] for r in measured), key=abs) if measured else None,
                             heading_mean=float(np.mean([r['heading'] for r in measured])) if measured else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage')
    parser.add_argument('--out')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    stage = str(Path(args.stage).resolve())
    jobs = [(stage, mode, speed, seed) for mode in MODES for speed in SPEEDS for seed in range(73000, 73000 + args.seeds)]
    with Pool(min(12, len(jobs), max(1, (os.cpu_count() or 2) - 2))) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    rows = summary(runs)
    for r in runs:
        print(f"{r['mode']:8s} v{r['v']:.1f} s{r['seed']}: fell {r['fell']} | wrists L,R ° at walk {r['wrists_walk_start']} "
              f"end {r['wrists_end']} | speed {r.get('speed', math.nan):.3f} yaw {r.get('yaw', math.nan):+.3f} "
              f"heading {r.get('heading', math.nan):+.1f}")
    for row in rows:
        print(f"SUMMARY {row['mode']:8s} v{row['v']:.1f}: falls {row['falls']}/{row['runs']} | bent at walk "
              f"{row['bent_walk_start']} end {row['bent_end']} | speed {fmt(row['speed'], '.3f')} | yaw mean "
              f"{fmt(row['yaw_mean'], '+.3f')} worst {fmt(row['yaw_worst'], '+.3f')} | heading "
              f"{fmt(row['heading_mean'], '+.1f')}°")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(stage=stage, runs=runs, summary=rows), indent=1))


if __name__ == '__main__':
    main()
