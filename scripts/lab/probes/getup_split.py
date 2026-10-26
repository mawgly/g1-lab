"""CPU only. getup_split.py <stage | experiment[:steps]>... [--workers 4] [--per-class 4] [--starts <file.json>]
    [--out <file.json>]
Whether a g1_getup stage gets up through a split and a jump out of it (iteration 60; the critic's lab-critic/it53,
split59.py: stage 1164's flight is a hop out of a split, the feet 1.01-1.10 m apart before it, 0.31-0.41 m after, with the
pelvis below the 0.5 m where stance_width counted).

Starts and episode as getup_style.py / getup_stance.py: fresh draws of the bank's generators per class (back, front,
side, sit, fall), the stage's policy deterministic, our physics (the files' friction, motor friction ×1, no delay),
pd_bound at the stage's lowest, no assist; up = the head top above 1.15 m with the pelvis above 0.70 m (within 6 s),
the run goes on 1 s after it.
Printed, one column per stage: the largest feet width (|W| across the pelvis heading, g1_tensor_math.stance_width) with
the pelvis 0.25-0.6 m up, median over runs (p90); the largest outward hip roll angle of either leg over the rise
(g1_getup.outward_signs: the left leg's +, the right leg's −; task_config hip_roll_out bounds its target), median / p90
over runs; the width 0.1 s before the last lift-off (the start of the last flight
of 40 ms or more before up, no robot-floor contact at all), median, and how many runs had one; the largest hand-floor
normal force per run (the wrist_yaw links' geoms, Unitree's hand capsules), p90 / max over runs; the share of the rise's
steps (start to up) with any motor torque above 90 % of its peak; up within 6 s.
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from getup_stance import KINDS, fresh_starts, label, resolve  # noqa: E402

UP_WITHIN, AFTER = 6., 1.  # s
UP_HEAD, UP_PELVIS = 1.15, .70  # m
SPLIT_PELVIS = (.25, .6)  # m
BEFORE = .1  # s before the last lift-off
SATURATION = .9  # of the motor peak


def run(job):
    import torch
    torch.set_num_threads(1)
    import mujoco
    import getup_probe as gp
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = gp._policy(job['stage'])
    env = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100000})
    raw = env.unwrapped
    m, d = raw.model, raw.data
    try:
        env.reset(seed=job['seed'], options=dict(assist_force=0., action_delay_steps=0, motor_friction_scale=1.,
                                                 pd_bound='min', start=0))
        gp.apply_friction(raw, None)
        d.qpos[:], d.qvel[:] = job['qpos'], job['qvel']
        mujoco.mj_forward(m, d)
        raw.head = float(raw._head()[0])
        obs = raw._observation('fill')
        floor = m.geom('floor').id
        hands = {m.body(name).id for name in ('left_wrist_yaw_link', 'right_wrist_yaw_link')}
        peak = np.asarray(raw.peak)
        force = np.zeros(6)
        widths, pelvis, airborne, saturated = [], [], [], []
        roll_joints = [raw.qadr[j] for j in raw.hip_roll_joints]
        roll_signs = np.array(raw.hip_roll_signs, dtype=float)
        roll = 0.  # the largest outward hip roll angle of either leg over the rise (rad)
        hand_force, up = 0., None
        for i in range(round((UP_WITHIN + AFTER) / raw.dt)):
            action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            obs, _, terminated, _, info = env.step(action)
            if terminated or info.get('bad_state'):
                break
            widths.append(abs(float(task.stance_width(d.xpos[raw.feet][None],
                                                      d.xmat[raw.pelvis].reshape(1, 3, 3))[0])))
            pelvis.append(float(d.qpos[2]))
            touching = False
            for c in range(d.ncon):
                contact = d.contact[c]
                if floor in (contact.geom1, contact.geom2):
                    touching = True
                    geom = contact.geom2 if contact.geom1 == floor else contact.geom1
                    if m.geom_bodyid[geom] in hands:
                        mujoco.mj_contactForce(m, d, c, force)
                        hand_force = max(hand_force, abs(float(force[0])))
            airborne.append(not touching)
            if up is None:
                saturated.append(bool((np.abs(d.ctrl) > SATURATION * peak).any()))
                roll = max(roll, float((roll_signs * d.qpos[roll_joints]).max()))
                if info['head_height'] > UP_HEAD and d.qpos[2] > UP_PELVIS:
                    up = i
                elif (i + 1) * raw.dt >= UP_WITHIN - 1e-9:
                    break
            elif (i - up) * raw.dt >= AFTER - 1e-9:
                break
        widths, pelvis = np.array(widths), np.array(pelvis)
        split = (pelvis > SPLIT_PELVIS[0]) & (pelvis < SPLIT_PELVIS[1])
        # The last flight of GETUP_FLIGHT_SECONDS or more before up (or before the end) and the width BEFORE earlier.
        flight_steps = max(1, round(task.GETUP_FLIGHT_SECONDS / raw.dt))
        end = len(airborne) if up is None else up + 1
        lift_off, length = None, 0
        for k in range(end):
            length = length + 1 if airborne[k] else 0
            if length == flight_steps:
                lift_off = k - flight_steps + 1
        before = None
        if lift_off is not None:
            before = float(widths[max(0, lift_off - 1 - round(BEFORE / raw.dt))]) if lift_off else None
        return dict(cls=job['cls'], seed=job['seed'], up=None if up is None else (up + 1) * raw.dt,
                    split=float(widths[split].max()) if split.any() else None, before=before, hand=hand_force, roll=roll,
                    saturated=int(sum(saturated)), rise_steps=len(saturated))
    finally:
        env.close()


def summary(results):
    split = [r['split'] for r in results if r['split'] is not None]
    before = [r['before'] for r in results if r['before'] is not None]
    hand = [r['hand'] for r in results]
    steps = sum(r['rise_steps'] for r in results)
    roll = [r['roll'] for r in results]
    return dict(starts=len(results), up=sum(r['up'] is not None for r in results),
                roll=(float(np.median(roll)), float(np.percentile(roll, 90))) if roll else None,
                split=(float(np.median(split)), float(np.percentile(split, 90)), len(split)) if split else None,
                before=(float(np.median(before)), len(before)) if before else None,
                hand=(float(np.percentile(hand, 90)), float(max(hand))) if hand else None,
                saturated=sum(r['saturated'] for r in results) / steps if steps else float('nan'))


def table(heads, columns):
    def f(value, digits=2):
        return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.{digits}f}'
    rows = [('наибольшая ширина стоп при тазе 0,25–0,6 м, м: медиана / p90 (прогонов)',
             [f'{f(c["split"][0])} / {f(c["split"][1])} ({c["split"][2]})' if c['split'] else '—' for c in columns]),
            ('наибольший крен бедра наружу при подъёме, рад: медиана / p90',
             [f'{f(c["roll"][0])} / {f(c["roll"][1])}' if c['roll'] else '—' for c in columns]),
            ('ширина стоп за 0,1 с до последнего отрыва, м: медиана (прогонов с полётом)',
             [f'{f(c["before"][0])} ({c["before"][1]})' if c['before'] else '—' for c in columns]),
            ('сила кисти о пол, Н: p90 / макс',
             [f'{f(c["hand"][0], 0)} / {f(c["hand"][1], 0)}' if c['hand'] else '—' for c in columns]),
            ('доля шагов подъёма с моментом > 90 % пика', [f(c['saturated']) for c in columns]),
            ('встали за 6 с', [f'{c["up"]}/{c["starts"]}' for c in columns])]
    first = max(len(r[0]) for r in rows)
    widths = [max(len(h), *(len(r[1][k]) for r in rows)) for k, h in enumerate(heads)]
    print(f'{"мера":<{first}}  ' + '  '.join(f'{h:>{w}}' for h, w in zip(heads, widths)))
    for name, cells in rows:
        print(f'{name:<{first}}  ' + '  '.join(f'{c:>{w}}' for c, w in zip(cells, widths)))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stages', nargs='+', help='этапы g1_getup: папка этапа или эксперимент[:шаги]')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--per-class', type=int, default=4, help='стартов на класс (спина, живот, бок, сидя, падение)')
    parser.add_argument('--starts', type=Path, help='старты из файла (JSON: список {cls, seed, qpos, qvel})')
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    stages = [resolve(spec) for spec in args.stages]
    for stage in stages:
        task_id = json.loads((stage / 'metadata.json').read_text(encoding='utf-8')).get('task_id')
        if task_id != 'g1_getup':
            raise SystemExit(f'{stage}: не этап вставания (task_id {task_id})')
    if args.starts:
        starts, per = [], {}
        for start in json.loads(args.starts.read_text(encoding='utf-8')):
            per[start['cls']] = per.get(start['cls'], 0) + 1
            if per[start['cls']] <= args.per_class:
                starts.append(start)
    else:
        starts = fresh_starts(args.per_class, args.workers)
    jobs = [dict(stage=str(stage), **start) for stage in stages for start in starts]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    per_stage = [[r for r, job in zip(results, jobs) if job['stage'] == str(stage)] for stage in stages]
    print(f'Стартов: {len(starts)} ({", ".join(f"{k} {sum(s["cls"] == k for s in starts)}" for k in KINDS)}); '
          'без шума, наша физика, граница ПД — нижняя по этапу, без подтяжки.')
    table([label(stage) for stage in stages], [summary(rs) for rs in per_stage])
    if args.out:
        args.out.write_text(json.dumps(dict(stages=[str(s) for s in stages], runs=per_stage), indent=1),
                            encoding='utf-8')


if __name__ == '__main__':
    main()
