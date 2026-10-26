"""CPU only. getup_slip.py <stage | experiment[:steps]>... [--friction 0.3,1.0] [--workers 4] [--per-class 4]
    [--starts <file.json>] [--out <file.json>]
How fast a g1_getup stage's soles slide over the floor while it gets up (iteration 62; the critic's lab-critic/it53,
slip61.py): stage 1164 makes its split by dragging, the soles sliding at 3-4 m/s while below 1 cm (a human's ~0.18
m/s), and with stance_width 4 the legs close by dragging too (up in 108 of 120 at foot friction 0.3).

Starts and episode as getup_split.py: fresh draws of the bank's generators per class (back, front, side, sit, fall),
the stage's policy deterministic, our physics with the sole spheres' friction set to each --friction value
(getup_probe.apply_friction; the rest the files'), motor friction ×1, no delay, pd_bound at the stage's lowest, no
assist; up = the head top above 1.15 m with the pelvis above 0.70 m within 6 s; the run goes on to 9 s.
A foot slides: its sole (g1_tensor_math.sole_height) below SOLE_CONTACT_HEIGHT 1 cm and its ankle_roll origin moving
horizontally (displacement over the control step / dt, the sole_slip term's speed). Printed per stage and friction:
the support soles' speed over the rise (start to up), median / p90 over every foot-step in contact; the largest such
speed per run, median / p90 over the runs; up and stayed up (head above 1.0 m at the end, never back below 0.6 m after
being above 1.0 m), of the runs.
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

UP_WITHIN, LENGTH = 6., 9.  # s
UP_HEAD, UP_PELVIS = 1.15, .70  # m
STAYED_HEAD, FELL_HEAD = 1.0, .6  # m


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
        gp.apply_friction(raw, job['friction'])
        d.qpos[:], d.qvel[:] = job['qpos'], job['qvel']
        mujoco.mj_forward(m, d)
        raw.head = float(raw._head()[0])
        obs = raw._observation('fill')
        before = d.xpos[raw.feet, :2].copy()
        speeds, up, high, falls, head = [], None, False, 0, raw.head
        for i in range(round(LENGTH / raw.dt)):
            action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            obs, _, terminated, _, info = env.step(action)
            if terminated or info.get('bad_state'):
                break
            now = d.xpos[raw.feet, :2].copy()
            speed = np.linalg.norm(now - before, axis=1) / raw.dt
            before = now
            sole = raw._sole_height().numpy()[0]
            head = info['head_height']
            if up is None:
                speeds += speed[sole < task.SOLE_CONTACT_HEIGHT].tolist()
                if head > UP_HEAD and d.qpos[2] > UP_PELVIS:
                    up = (i + 1) * raw.dt
                elif (i + 1) * raw.dt >= UP_WITHIN - 1e-9:
                    break
            if head > STAYED_HEAD:
                high = True
            elif high and head < FELL_HEAD:
                falls += 1
                high = False
        return dict(cls=job['cls'], seed=job['seed'], friction=job['friction'], up=up, falls=falls,
                    ok=up is not None and head > STAYED_HEAD and falls == 0, speeds=speeds,
                    largest=max(speeds) if speeds else 0.)
    finally:
        env.close()


def summary(results):
    speeds = [v for r in results for v in r['speeds']]
    largest = [r['largest'] for r in results]
    return dict(n=len(results), ok=sum(r['ok'] for r in results),
                speed=(float(np.median(speeds)), float(np.percentile(speeds, 90))) if speeds else None,
                largest=(float(np.median(largest)), float(np.percentile(largest, 90))) if largest else None)


def table(heads, columns):
    def f(value):
        return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.2f}'
    rows = [('скольжение опорной подошвы при подъёме, м/с: медиана / p90',
             [f'{f(c["speed"][0])} / {f(c["speed"][1])}' if c['speed'] else '—' for c in columns]),
            ('наибольшее скольжение за подъём, м/с: медиана / p90 по прогонам',
             [f'{f(c["largest"][0])} / {f(c["largest"][1])}' if c['largest'] else '—' for c in columns]),
            ('встал и устоял', [f'{c["ok"]} из {c["n"]}' for c in columns])]
    first = max(len(r[0]) for r in rows)
    widths = [max(len(h), *(len(r[1][k]) for r in rows)) for k, h in enumerate(heads)]
    print(f'{"мера":<{first}}  ' + '  '.join(f'{h:>{w}}' for h, w in zip(heads, widths)))
    for name, cells in rows:
        print(f'{name:<{first}}  ' + '  '.join(f'{c:>{w}}' for c, w in zip(cells, widths)))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stages', nargs='+', help='этапы g1_getup: папка этапа или эксперимент[:шаги]')
    parser.add_argument('--friction', default='0.3,1.0', help='трение подошв, через запятую')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--per-class', type=int, default=4, help='стартов на класс (спина, живот, бок, сидя, падение)')
    parser.add_argument('--starts', type=Path, help='старты из файла (JSON: список {cls, seed, qpos, qvel})')
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    stages = [resolve(spec) for spec in args.stages]
    frictions = [float(v) for v in args.friction.split(',')]
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
    jobs = [dict(stage=str(stage), friction=mu, **start) for stage in stages for mu in frictions for start in starts]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    columns, heads = [], []
    for stage in stages:
        for mu in frictions:
            heads.append(f'{label(stage)}, μ {mu:g}')
            columns.append(summary([r for r, job in zip(results, jobs)
                                    if job['stage'] == str(stage) and job['friction'] == mu]))
    print(f'Стартов: {len(starts)} ({", ".join(f"{k} {sum(s["cls"] == k for s in starts)}" for k in KINDS)}); '
          'без шума, наша физика, трение подошв — по столбцам, граница ПД — нижняя по этапу, без подтяжки.')
    table(heads, columns)
    if args.out:
        args.out.write_text(json.dumps(dict(stages=[str(s) for s in stages], frictions=frictions, runs=[
            {k: v for k, v in r.items() if k != 'speeds'} for r in results]), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
