"""CPU only. crouch.py <stage dir | export dir> [out.json] [--seeds 5] [--workers 2] [--parts ABCD]

The height command of iteration 28 (task_config height_command): deterministic policy, heading hold off, fixed
commands (resets with a command: no training draws), the height set by set_height as the stick would.
  A  standing: 2 s at 0.78, 8 s at 0.70, 8 s at 0.64, 4 s at 0.78;
  B  walking 0.5 and 1 m/s at 0.70, 10 s each (measured after 3 s);
  C  walking 0.5 m/s: 3 s at 0.78, 5 s at 0.70, 4 s at 0.78;
  D  standing at 0.70, a pelvis push after 3 s (set_force_push, 0.1 s; forward, back, left, right at Δv 0.5 and 1),
     printed only.
Per segment: pelvis height (mean over its last 2 s), the time to reach the commanded height (within 1 cm), falls (the
environment's termination: below c − 0.23 or tilted), creep (pelvis xy over the segment, standing), torso lean
(forward, deg), sole lift (standing: the highest sole point over the segment; walking: p95 of the sole height off the
floor, the worse foot), the share of steps with an ankle pitch within ANKLE_MARGIN of its stop (−0.873) and the peak
knee torque (|ctrl|, N·m). Walking also: speed and yaw rate without a turn command. A stage without the height
command runs at 0.78 throughout (noted). 5 seeds from 73000, at most `workers` processes (default 2).
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stand_turn as st  # noqa: E402

SEED = 73000
ANKLE_MARGIN = .05  # rad from the ankle pitch's lower stop
REACH = .01  # m: the commanded height is reached within this
STAND = (0., 0., 0.)
PUSHES = tuple((dv * x, dv * y) for dv in (.5, 1.) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1)))
PUSH_AT = 3.


def schedules(parts):
    """{name: [(seconds, command, height), ...]} of the probe's parts."""
    plans = {}
    if 'A' in parts:
        plans['A стоя 0,78→0,70→0,64→0,78'] = [(2., STAND, .78), (8., STAND, .70), (8., STAND, .64), (4., STAND, .78)]
    if 'B' in parts:
        for speed in (.5, 1.):
            plans[f'B ходьба {speed:g} м/с на 0,70'.replace('.', ',')] = [(10., (speed, 0., 0.), .70)]
    if 'C' in parts:
        plans['C ходьба 0,5: 0,78→0,70→0,78'] = [(3., (.5, 0., 0.), .78), (5., (.5, 0., 0.), .70),
                                                (4., (.5, 0., 0.), .78)]
    if 'D' in parts:
        for push in PUSHES:
            plans[f'D толчок стоя на 0,70 ({push[0]:+g}, {push[1]:+g}) м/с'.replace('.', ',')] = [
                (PUSH_AT, STAND, .70), (5., STAND, .70, push)]
    return plans


def run(job):
    import torch
    from humanoid_sim.goal_check import foot_state, sole_geoms
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    torch.set_num_threads(1)
    model, norm, config = st.stage_data(job['stage'])
    plan = job['plan']
    total = sum(segment[0] for segment in plan)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    names = [raw.model.joint(j).name for j in raw.joints]
    ankles = [names.index(f'{side}_ankle_pitch_joint') for side in ('left', 'right')]
    knees = [names.index(f'{side}_knee_joint') for side in ('left', 'right')]
    stop = float(raw.model.jnt_range[raw.joints][ankles[0], 0])
    soles, floor = sole_geoms(raw), raw.model.geom('floor').id
    crouch = bool(getattr(raw, 'height_command', False))
    obs, _ = env.reset(seed=job['seed'], options={'command': list(plan[0][1]), 'phase_steps': 0})
    rows, fell, segment_starts = [], None, []
    step = 0
    for number, segment in enumerate(plan):
        seconds, command, height = segment[:3]
        segment_starts.append(len(rows))
        raw.set_command(np.asarray(command, dtype=float))
        if crouch:
            raw.set_height(height)
        if len(segment) > 3:  # a push at the segment's start
            raw.set_force_push(np.asarray(segment[3], dtype=float), .1)
        obs = raw._observation('replace')
        for _ in range(round(seconds / raw.dt)):
            obs, _, terminated, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            step += 1
            q = raw.data.qpos
            feet = foot_state(raw, soles, floor)
            torso = torch.from_numpy(raw.data.xmat[raw.torso].reshape(1, 3, 3).copy())
            pelvis = torch.from_numpy(raw.data.xmat[raw.pelvis].reshape(1, 3, 3).copy())
            lean = math.degrees(float(task.torso_lean(torso, pelvis)[0][0]))
            joints = q[raw.qadr]
            rows.append(dict(t=step * raw.dt, z=float(q[2]), x=float(q[0]), y=float(q[1]), yaw=st.heading_of(q),
                             lean=lean, contact=[f[0] for f in feet], sole=[f[1] for f in feet],
                             ankle=[float(joints[i]) for i in ankles],
                             knee_torque=[float(abs(raw.data.ctrl[i])) for i in knees]))
            if terminated:
                fell = step * raw.dt
                break
        if fell is not None:
            break
    env.close()
    segments = []
    for number, segment in enumerate(plan):
        seconds, command, height = segment[:3]
        if number >= len(segment_starts):
            break
        start = segment_starts[number]
        end = segment_starts[number + 1] if number + 1 < len(segment_starts) else len(rows)
        part = rows[start:end]
        if not part:
            break
        walking = bool(np.linalg.norm(command) > 0)
        z = np.array([r['z'] for r in part])
        tail = part[-min(len(part), 100):]
        reached = next((r['t'] - part[0]['t'] + raw.dt for r in part if abs(r['z'] - height) <= REACH), None)
        sole = np.array([r['sole'] for r in part])
        contact = np.array([r['contact'] for r in part])
        if walking:
            settled = part[min(len(part) - 1, 150):] if len(part) > 150 else part
            lifts = [float(np.percentile(np.array([r['sole'][k] for r in settled if not r['contact'][k]]), 95))
                     if any(not r['contact'][k] for r in settled) else 0. for k in range(2)]
            lift = min(lifts)
            yaw = np.unwrap([r['yaw'] for r in settled])
            duration = settled[-1]['t'] - settled[0]['t']
            speed = (settled[-1]['x'] - settled[0]['x']) / duration if duration > 0 else None
            yaw_rate = float((yaw[-1] - yaw[0]) / duration) if duration > 0 else None
        else:  # the highest sole point (−0.6 mm resting: soft contact) counts from the floor up
            lift, speed, yaw_rate = max(0., float(sole.max())), None, None
        ankle = np.array([r['ankle'] for r in part])
        segments.append(dict(
            seconds=seconds, command=list(command), height=height, push=list(segment[3]) if len(segment) > 3 else None,
            pelvis=float(np.mean([r['z'] for r in tail])), pelvis_min=float(z.min()), reached_s=reached,
            creep_m=None if walking else float(math.hypot(part[-1]['x'] - part[0]['x'], part[-1]['y'] - part[0]['y'])),
            lean_deg=float(np.mean([r['lean'] for r in part])), lift_m=lift,
            ankle_stop=float((ankle <= stop + ANKLE_MARGIN).any(axis=1).mean()), ankle_min=float(ankle.min()),
            knee_torque=float(np.max([r['knee_torque'] for r in part])), contact=float(contact.mean()),
            speed=speed, yaw_rate=yaw_rate))
    return dict(job={k: v for k, v in job.items() if k != 'stage'}, crouch=crouch, fell=fell, total_s=total,
                segments=segments)


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage')
    parser.add_argument('out', nargs='?')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--parts', default='ABCD')
    args = parser.parse_args()
    plans = schedules(args.parts.upper())
    jobs = [dict(stage=args.stage, name=name, plan=plan, seed=SEED + k)
            for name, plan in plans.items() for k in range(args.seeds)]
    if args.workers <= 1:
        results = [run(job) for job in jobs]
    else:
        with Pool(min(args.workers, 2)) as pool:  # sim2sim and checks share the CPU
            results = pool.map(run, jobs, chunksize=1)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding='utf-8')
    print(args.stage + ('' if results[0]['crouch'] else ' — без команды высоты: всё на 0,78'))
    for name in plans:
        rows = [r for r in results if r['job']['name'] == name]
        falls = sum(r['fell'] is not None for r in rows)
        print(f'{name}: падений {falls} из {len(rows)}')
        for k in range(len(plans[name])):
            got = [r['segments'][k] for r in rows if len(r['segments']) > k]
            if not got:
                continue
            mean = lambda key: (None if any(g[key] is None for g in got) else float(np.mean([g[key] for g in got])))
            worst = lambda key: (None if any(g[key] is None for g in got) else float(np.max([g[key] for g in got])))
            reached = [g['reached_s'] for g in got]
            reach = (f'{number(np.mean([r for r in reached if r is not None]), 1)} с'
                     + (f' (не вышли {sum(r is None for r in reached)})' if None in reached else '')
                     if any(r is not None for r in reached) else 'не вышли')
            line = (f'  {number(got[0]["height"])} м, {number(got[0]["seconds"], 0)} с: таз {number(mean("pelvis"), 3)} '
                    f'(мин. {number(min(g["pelvis_min"] for g in got), 3)}), выход {reach}, наклон корпуса '
                    f'{number(mean("lean_deg"), 1)}°, подошва {number(1000 * worst("lift_m"), 0)} мм, голеностоп в упоре '
                    f'{number(100 * worst("ankle_stop"), 0)}% (мин. {number(min(g["ankle_min"] for g in got), 3)}), '
                    f'колено до {number(worst("knee_torque"), 0)} Н·м')
            if got[0]['creep_m'] is not None:
                line += f', уползание {number(worst("creep_m"))} м'
            if got[0]['speed'] is not None:
                rates = [g['yaw_rate'] for g in got]
                line += (f', скорость {number(mean("speed"))} м/с, увод худший '
                         f'{number(max(rates, key=abs), 3)} рад/с')
            print(line)
    print(len(results), 'прогонов, падений', sum(r['fell'] is not None for r in results))


if __name__ == '__main__':
    main()
