"""Our physics for the sim2sim scenarios: gamepad.py's stick segments in the CPU G1 task.

    .venv\\Scripts\\python.exe scripts\\sim2sim\\ours.py <stage dir> [--scenario turn|side|stopstart] [--seeds 2]

Deterministic policy, heading hold off (the robot has none), our reset standing (no FixStand or band), then
each segment's command as a step (the remote's smoothing, ~0.2 s, is not modelled). Run k starts its gait
clock 15·k control steps on (the clock at the first walk differs between runs, as in g1_ctrl). Measured as
analyze.py does: after the first 2 s of a segment (1 s under 8 s) the yaw rate and the forward / sideways speed
of the pelvis path in its heading frame; over the whole segment the displacement, heading change and falls
(the task's own fall test). CPU only; prints a table in Russian, --out writes JSON.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # the GPU belongs to training

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import gamepad  # noqa: E402
from humanoid_sim.checkpoint_schema import checkpoint_identity  # noqa: E402
from humanoid_sim.core import load_stage, read_json, stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

SETTLE = 2.0
SEED = 73000
PHASE_STEP = 15  # half a 0.6 s gait cycle at 0.02 s


def motion(track, a, b, dt):
    """track rows (x, y, z, yaw) per control step; steps a..b as analyze.motion."""
    rows = track[a:b + 1]
    d = np.diff(rows[:, :2], axis=0)
    h = rows[:-1, 3]
    seconds = (len(rows) - 1) * dt
    turn = float(rows[-1, 3] - rows[0, 3])
    return dict(seconds=seconds, forward_speed=float(np.sum(d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h))) / seconds,
                sideways_speed=float(np.sum(-d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h))) / seconds,
                yaw_rate=turn / seconds, heading_deg=math.degrees(turn),
                distance_m=float(np.linalg.norm(rows[-1, :2] - rows[0, :2])),
                path_m=float(np.linalg.norm(d, axis=1).sum()), pelvis_z_min=float(rows[:, 2].min()))


def run(stage, plan, run_index, policy, norm):
    identity = checkpoint_identity(read_json(stage / 'metadata.json'))
    dt = identity['control_dt']
    total = sum(int(round(seconds / dt)) for *_, seconds in plan)
    config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': total + 10}
    env = make_task(identity['task_id'], identity['robot_id'], config)
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id

    def pose():
        m = raw.data.xmat[pelvis]
        return [*raw.data.qpos[:3], math.atan2(m[3], m[0])]

    try:
        obs, _ = env.reset(seed=SEED + run_index, options={'command': [0., 0., 0.],
                                                           'phase_steps': PHASE_STEP * run_index})
        track, bounds, fell_at = [pose()], [], None
        for name, vx, vy, wz, seconds in plan:
            raw.set_command([vx, vy, wz])
            obs = raw._observation('replace')
            start = len(track) - 1
            for _ in range(int(round(seconds / dt))):
                obs, _, terminated, _, _ = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
                track.append(pose())
                if terminated:
                    fell_at = (len(track) - 1) * dt
                    break
            bounds.append((name, [vx, vy, wz], start, len(track) - 1))
            if fell_at is not None:
                break
    finally:
        env.close()
    track = np.array(track)
    track[:, 3] = np.unwrap(track[:, 3])
    out = []
    for k, (name, command, a, b) in enumerate(bounds):
        seconds = (b - a) * dt
        settle = SETTLE if seconds >= 8 else 1.0
        fell = fell_at is not None and k == len(bounds) - 1  # the walk stops at a fall
        settled = motion(track, a + int(round(settle / dt)), b, dt) if seconds > settle + dt else None
        out.append(dict(name=name, command=command, start_s=a * dt, seconds=seconds, settle_s=settle,
                        fell_s=fell_at if fell else None, whole=motion(track, a, b, dt), settled=settled))
    return dict(run=run_index, phase_steps=PHASE_STEP * run_index, fell_s=fell_at, segments=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', type=Path)
    parser.add_argument('--scenario', choices=('turn', 'side', 'stopstart'), default='turn')
    parser.add_argument('--seeds', type=int, default=2, help='runs (seed and gait-clock start differ)')
    parser.add_argument('--stand-seconds', type=float, default=5.0)
    parser.add_argument('--walk', type=float, default=0.5)
    parser.add_argument('--yaw', type=float, default=0.5)
    parser.add_argument('--side', type=float, default=0.3)
    parser.add_argument('--after-seconds', type=float, default=5.0)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    plan = gamepad.segments(argparse.Namespace(**vars(args), passive_walking=False))
    import torch
    torch.set_num_threads(2)  # leave the CPU to the trainer
    policy, norm = load_stage(args.stage)
    try:
        results = [run(args.stage, plan, k, policy, norm) for k in range(args.seeds)]
    finally:
        norm.close()
    for r in results:
        fell = 'нет' if r['fell_s'] is None else f"на {r['fell_s']:.1f} с"
        print(f"Наша физика, {args.scenario}, прогон {r['run']} (часы фазы +{r['phase_steps']}): падение {fell}")
        for s in r['segments']:
            vx, vy, wz = s['command']
            st, wh = s['settled'], s['whole']
            measured = ('—' if st is None else
                        f"вперёд {st['forward_speed']:+.2f} м/с, вбок {st['sideways_speed']:+.2f} м/с, "
                        f"рысканье {st['yaw_rate']:+.3f} рад/с, смещение {st['distance_m']:.2f} м")
            print(f"{s['start_s']:5.1f} с {s['name']:<11} ({vx:+.2f}, {vy:+.2f}, {wz:+.2f}) {s['seconds']:.1f} с; "
                  f"после {s['settle_s']:g} с: {measured}; весь отрезок: смещение {wh['distance_m']:.2f} м, "
                  f"курс {wh['heading_deg']:+.1f}°, таз ≥ {wh['pelvis_z_min']:.3f} м")
    if args.out:
        report = dict(stage=str(args.stage), scenario=args.scenario, runs=results)
        args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
