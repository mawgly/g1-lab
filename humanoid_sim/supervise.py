"""One fixed check of research-tree branches, so supervisor decisions compare the same numbers.

    .venv\\Scripts\\python.exe -m humanoid_sim.supervise exp-… [exp-…:228753408 …]

For each branch (latest stage, or the given one): CPU episodes of 10 s on 3 seeds per scenario
(stand, walk 1 m/s, stop-start), recording survival time, forward progress, heading drift and,
for G1, torso lean. Scenario commands are fixed, so training-time pushes and resampling are absent.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import DEFAULT_RUN, list_stages, load_stage, read_json, stage_task_config
from .checkpoint_schema import checkpoint_identity
from .lab import Laboratory

SCENARIOS = ('stand', 'walk', 'stop_start')


def find_stage(lab, item):
    """`exp-…` → its latest stage, `exp-…:steps` → that stage; None when the branch has no stages."""
    identity, _, steps = item.partition(':')
    stages = list_stages(lab.resolve(identity))
    if not stages:
        return None
    return next((s for s in stages if steps and int(s.name[6:]) == int(steps)), stages[-1])


def rollout(stage, identity, policy, norm, scenario, seed, seconds, on_step=None):
    """One deterministic CPU episode under the scenario's fixed commands.

    on_step(seconds, raw_env, torso_lean_deg, fell) runs after every control step (lean is None without a torso)."""
    from .tasks.catalog import make_task
    dt = identity['control_dt']
    limit = math.ceil(seconds / dt)
    env = make_task(identity['task_id'], identity['robot_id'], {**stage_task_config(stage), 'max_episode_steps': limit})
    try:
        raw = env.unwrapped
        commanded = hasattr(raw, 'set_command')
        obs, _ = env.reset(seed=seed, options={'command': scenario.command_at(0.)} if commanded else None)
        torso = raw.model.body('torso_link').id if commanded else None
        root = raw.model.body(0 if not commanded else 'pelvis').id
        leans, fell_at, path_deg, clipped = [], None, 0., []
        start = raw.data.qpos[:2].copy()

        def yaw():
            rotation = raw.data.xmat[root].reshape(3, 3)
            return math.atan2(rotation[1, 0], rotation[0, 0])

        for step in range(limit):
            if commanded:
                raw.set_command(scenario.command_at(step * dt))
                obs = raw._observation()
            action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
            clipped.append(float(np.mean(np.abs(action) >= .99 * env.action_space.high)))  # joints pinned at the clip
            obs, _, terminated, truncated, _ = env.step(action)
            lean = None
            if torso is not None:
                up = raw.data.xmat[torso].reshape(3, 3)[:, 2]
                lean = math.degrees(math.atan2(up[0], up[2]))  # forward lean while heading ~+x
                leans.append(lean)
            # Path deviation: the widest bearing of the position from the start, seen against the +x course,
            # once the robot is 2 m away. Circles reach 90°+ even if the final heading is back near zero;
            # the per-step pelvis twist does not count, unlike a sum of heading changes.
            dx, dy = raw.data.qpos[0] - start[0], raw.data.qpos[1] - start[1]
            if dx * dx + dy * dy > 4.:
                path_deg = max(path_deg, abs(math.degrees(math.atan2(dy, dx))))
            if terminated:
                fell_at = (step + 1) * dt
            if on_step is not None:
                on_step((step + 1) * dt, raw, lean, terminated)
            if terminated or truncated:
                break
        return dict(fell_at=fell_at, forward_m=float(raw.data.qpos[0]), sideways_m=float(raw.data.qpos[1]),
                    heading_deg=float(math.degrees(yaw())), path_deg=float(path_deg),
                    clipped=float(np.mean(clipped)) if clipped else 0.,
                    torso_lean_deg=float(np.mean(leans)) if leans else None)
    finally:
        env.close()


def check_stage(stage, seconds=10., seeds=3, seed_start=73000, scenarios=SCENARIOS, acting=None):
    """acting: (policy, norm) that act in the stage's environment instead of its own (an export folder's
    export_g1.ExportedPolicy as both); σ stays the stage's training noise."""
    from .evaluation.scenarios import get_scenario
    identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
    policy, norm = load_stage(stage)
    actor, normalizer = acting or (policy, norm)
    result = {}
    try:
        for name in scenarios:
            scenario = get_scenario(name)
            rows = [rollout(stage, identity, actor, normalizer, scenario, seed, seconds)
                    for seed in range(seed_start, seed_start + seeds)]
            falls = [r['fell_at'] for r in rows if r['fell_at'] is not None]
            result[name] = dict(
                falls=f'{len(falls)}/{len(rows)}',
                survived_s=float(np.mean([r['fell_at'] or seconds for r in rows])),
                forward_m=float(np.mean([r['forward_m'] for r in rows])),
                sideways_m=float(np.mean([abs(r['sideways_m']) for r in rows])),
                heading_deg=float(np.mean([abs(r['heading_deg']) for r in rows])),
                path_deg=float(np.mean([r['path_deg'] for r in rows])),
                clipped=float(np.mean([r['clipped'] for r in rows])),
                torso_lean_deg=None if rows[0]['torso_lean_deg'] is None else float(np.mean([r['torso_lean_deg'] for r in rows])))
        # Training noise of the policy: with actions clipped to ±1 an exploding std means random bang-bang.
        std = float(np.exp(policy.policy.log_std.detach().cpu().numpy()).mean()) if hasattr(policy.policy, 'log_std') else None
    finally:
        norm.close()
    return dict(stage=int(Path(stage).name[6:]), seconds=seconds, seeds=seeds, policy_std=std, scenarios=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branches', nargs='+', help='experiment id, optionally :steps of a stage')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--scenarios', default=','.join(SCENARIOS),
                        help='comma separated (sprint: constant 2 m/s, for branches trained on it)')
    args = parser.parse_args()
    scenarios = tuple(s.strip() for s in args.scenarios.split(',') if s.strip())
    lab = Laboratory(DEFAULT_RUN)
    report = {}
    for item in args.branches:
        stage = find_stage(lab, item)
        if stage is None:
            report[item] = {'error': 'нет этапов'}
            continue
        report[item] = check_stage(stage, args.seconds, args.seeds, scenarios=scenarios)
        line = ' | '.join(f"{name}: {v['falls']} пад., {v['survived_s']:.1f} с, вперёд {v['forward_m']:.1f} м, "
                          f"курс {v['heading_deg']:.0f}°, путь уходит до {v['path_deg']:.0f}°" + (f", корпус {v['torso_lean_deg']:.0f}°" if v['torso_lean_deg'] is not None else '')
                          for name, v in report[item]['scenarios'].items())
        clipped = max(v['clipped'] for v in report[item]['scenarios'].values())
        std = report[item]['policy_std']
        health = f" || действия у предела: {100 * clipped:.0f}%" + (f", шум σ {std:.2g}" if std is not None else '')
        print(f"{item} @ {report[item]['stage'] / 1e6:.1f} млн — {line}{health}", flush=True)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
