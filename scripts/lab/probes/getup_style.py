"""CPU only. getup_style.py <stage | experiment[:steps]>... [--workers 4] [--per-class 4] [--starts <file.json>]
    [--library lafan1] [--out <file.json>]
How human-like a g1_getup stage gets up, by the style term's own measure (iteration 53; the critic's lab-critic/it53,
knn53.py): d, the distance of each frame to the nearest human frame of the style library (g1_tensor_math.style_features
/ style_distance; the library of tasks/g1_getup_style, subjects 1-4 of LAFAN1's fallAndGetUp), over the frames with the
head top in STYLE_HEAD_BAND (0.3-1.1 m, the rise), from the start until up + 0.5 s.

Starts and episode as getup_stance.py: fresh draws of the bank's generators per class (back, front, side, sit, fall),
the stage's policy deterministic (no noise), our physics (the files' friction, motor friction ×1, no delay), pd_bound
at the stage's lowest, no assist; up (getup_success_pose) within 6 s. The style weight is switched on in the probe's
task (a reward weight: the episode is the same), so the task itself gives d and the nearest frame.
Printed, one column per stage and the held-out human (subject 5's frames against the same library, the positive
control): d median / p90 (the acceptance row), d of the pose part alone (angles, gravity, pelvis height: knn53's
"pose-only"), the RMS over the band frames of the angle residual to the nearest frame per joint group (legs, waist,
arms; rad), the median norm of the speed features (0.1·joint speeds and the pelvis vertical speed; the human's ~0.77,
823's ~2.8 by the critic), the head top's rise 0.3 → 1.1 m (from the last frame below 0.3 m to the crossing of 1.1 m,
s: median / p10) and its vertical speed in the band ((head − head a step before)/dt, head_rise's speed, iteration 57:
median / p90; the human's from subject 5's clips at 50 Hz), the rise (s, median), the band frames per rise and the
flights ≥ 40 ms (no robot-floor contact at all; GETUP_FLIGHT_SECONDS).
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
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # scripts/lab: getup_style_library
from getup_stance import KINDS, fresh_starts, label, resolve  # noqa: E402

UP_WITHIN, AFTER = 6., .5  # s: up within, then the band frames counted for
GROUPS = dict(legs=range(0, 12), waist=range(12, 15), arms=range(15, 25))  # columns of the free joints' angles


def run(job):
    """One stage and start: per band frame d, the pose-only d and the residuals; the rise and its flights."""
    import torch
    torch.set_num_threads(1)
    import mujoco
    import getup_probe as gp
    from humanoid_sim.tasks import g1_getup_style
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = gp._policy(job['stage'])
    weights = {**config.get('reward_weights', {}), 'style': config.get('reward_weights', {}).get('style') or 1.}
    env = make_task('g1_getup', 'g1_29dof', {**config, 'reward_weights': weights, 'style_library': job['library'],
                                             'max_episode_steps': 100000})
    raw = env.unwrapped
    m, d = raw.model, raw.data
    library = g1_getup_style.load(job['library'])
    pose = slice(0, len(raw.style_joints) + 4)
    pose_rows = np.ascontiguousarray(library['features'][:, pose])
    pose_norms = task.style_norms(pose_rows)
    try:
        env.reset(seed=job['seed'], options=dict(assist_force=0., action_delay_steps=0, motor_friction_scale=1.,
                                                 pd_bound='min', start=0))
        gp.apply_friction(raw, None)
        d.qpos[:], d.qvel[:] = job['qpos'], job['qvel']
        mujoco.mj_forward(m, d)
        raw.head = float(raw._head()[0])  # the start's head (head_rise's speed of the first step)
        obs = raw._observation('fill')
        floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        out = dict(cls=job['cls'], seed=job['seed'], up=None, dist=[], pose=[], residual=[], speed=[],
                   heads=[raw.head])
        airborne = longest = 0
        for i in range(round((UP_WITHIN + AFTER) / raw.dt)):
            t = (i + 1) * raw.dt
            action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            obs, _, terminated, _, info = env.step(action)
            if terminated or info.get('bad_state'):
                out['ended'] = t
                break
            out['heads'].append(info['head_height'])
            if info['style_band']:
                features = raw.style_features()
                out['dist'].append(info['style_dist'])
                out['pose'].append(float(task.style_distance(features[:, pose], pose_rows, pose_norms)[0][0]))
                out['residual'].append((features[0, :pose.stop - 4] - library['features'][info['style_frame'],
                                                                                          :pose.stop - 4]).tolist())
                out['speed'].append(float(np.linalg.norm(features[0, pose.stop:])))
            if out['up'] is None:
                touching = any(int(d.contact.geom1[c]) in floor or int(d.contact.geom2[c]) in floor
                               for c in range(d.ncon))
                airborne = 0 if touching else airborne + 1
                longest = max(longest, airborne)
                sole = raw._sole_height().numpy()
                if bool(task.getup_success_pose(raw._head(), d.qpos[2:3], d.xmat[raw.torso].reshape(1, 3, 3),
                                                sole)[0]):
                    out.update(up=t, flight=longest * raw.dt)
                elif t >= UP_WITHIN - 1e-9:
                    break
            elif t - out['up'] >= AFTER - 1e-9:
                break
        return out
    finally:
        env.close()


def describe(dist, pose, residual, speed):
    """The table's numbers of band frames (lists or arrays)."""
    residual = np.asarray(residual).reshape(-1, 25)
    return dict(n=len(dist), dist=(float(np.median(dist)), float(np.percentile(dist, 90))) if len(dist) else None,
                pose=float(np.median(pose)) if len(pose) else float('nan'),
                rms={key: float(np.sqrt((residual[:, list(columns)] ** 2).mean())) if len(residual) else float('nan')
                     for key, columns in GROUPS.items()},
                speed=float(np.median(speed)) if len(speed) else float('nan'))


def head_motion(heads, dt):
    """The head top's rises and speeds of one head height series (m, every dt s): each rise 0.3 → 1.1 m (from the
    last frame below 0.3 m before a crossing of 1.1 m to that crossing, s) and the vertical speeds (head − head a step
    before)/dt of the frames with the head in STYLE_HEAD_BAND (head_rise's speed)."""
    from humanoid_sim.tasks import g1_tensor_math as task
    heads = np.asarray(heads, dtype=float)
    rises, low = [], None
    for k, h in enumerate(heads):
        if h < task.STYLE_HEAD_BAND[0]:
            low = k
        elif h >= task.STYLE_HEAD_BAND[1] and low is not None:
            rises.append((k - low) * dt)
            low = None
    speed = np.diff(heads) / dt
    return rises, speed[task.style_band(heads[1:])].tolist()


def human_heads(holdout=5):
    """The held-out subject's head top heights per clip at the control step (getup_style_library.py's resampling and
    the getting-up model's kinematics) and the step."""
    import mujoco
    from getup_style_library import DATA, resample
    from humanoid_sim.tasks import g1_getup_bank
    from humanoid_sim.tasks import g1_tensor_math as task
    env = g1_getup_bank.getup_model_env()
    m, d = env.model, env.data
    clips = []
    for file in sorted(DATA.glob(f'fallAndGetUp*_subject{holdout}.csv')):
        heads = []
        for row in resample(np.loadtxt(file, delimiter=','), env.dt)[0]:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            heads.append(float(task.head_height(d.xpos[env.torso][None], d.xmat[env.torso].reshape(1, 3, 3))[0]))
        clips.append(heads)
    return clips, env.dt


def motion(rises, speeds):
    return dict(rises=(float(np.median(rises)), float(np.percentile(rises, 10)), len(rises)) if rises else None,
                head_speed=(float(np.median(speeds)), float(np.percentile(speeds, 90))) if speeds else None)


def human(name):
    """Held-out subject's frames against the library: the positive control column."""
    from humanoid_sim.tasks import g1_getup_style
    from humanoid_sim.tasks import g1_tensor_math as task
    library = g1_getup_style.load(name)
    rows, held = library['features'], library['holdout']
    pose = slice(0, len(library['joints']) + 4)
    dist, index = task.style_distance(held, rows, library['norms'])
    pose_dist, _ = task.style_distance(held[:, pose], rows[:, pose], task.style_norms(rows[:, pose]))
    residual = held[:, :pose.stop - 4] - rows[index, :pose.stop - 4]
    clips, dt = human_heads(library['header'].get('holdout_subject', 5))
    rises, speeds = [], []
    for heads in clips:
        r, s = head_motion(heads, dt)
        rises += r
        speeds += s
    return dict(**describe(dist, pose_dist, residual, np.linalg.norm(held[:, pose.stop:], axis=1)),
                **motion(rises, speeds))


def summary(results):
    from humanoid_sim.tasks import g1_tensor_math as task
    up = [r for r in results if r['up'] is not None]
    flat = {key: [v for r in results for v in r[key]] for key in ('dist', 'pose', 'residual', 'speed')}
    rises, speeds = [], []
    for r in results:
        rise, speed = head_motion(r['heads'], .02)
        rises += rise
        speeds += speed
    return dict(**describe(flat['dist'], flat['pose'], flat['residual'], flat['speed']), **motion(rises, speeds),
                starts=len(results),
                up=len(up), rise=float(np.median([r['up'] for r in up])) if up else float('nan'),
                per_rise=float(np.median([len(r['dist']) for r in results])),
                flights=sum(r['flight'] >= task.GETUP_FLIGHT_SECONDS - 1e-9 for r in up))


def table(heads, columns):
    def f(value, digits=2):
        return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.{digits}f}'
    rows = [('d до ближайшего кадра человека: медиана / p90', 'новая строка',
             [f'{f(c["dist"][0])} / {f(c["dist"][1])}' if c['dist'] else '—' for c in columns]),
            ('  d только позы (углы, гравитация, высота), медиана', None, [f(c['pose']) for c in columns]),
            ('  СКО угла до кадра, рад: ноги / пояс / руки', None,
             [f'{f(c["rms"]["legs"])} / {f(c["rms"]["waist"])} / {f(c["rms"]["arms"])}' for c in columns]),
            ('  скорость (норма признаков скорости), медиана', None, [f(c['speed']) for c in columns]),
            ('кадров в полосе 0,3–1,1 м: всего / на подъём', None,
             [f'{c["n"]} / {f(c.get("per_rise"), 0)}' for c in columns]),
            ('подъём головы 0,3 → 1,1 м, с: медиана / p10 (подъёмов)', None,
             [f'{f(c["rises"][0])} / {f(c["rises"][1])} ({c["rises"][2]})' if c.get('rises') else '—'
              for c in columns]),
            ('скорость головы вверх в полосе, м/с: медиана / p90', None,
             [f'{f(c["head_speed"][0])} / {f(c["head_speed"][1])}' if c.get('head_speed') else '—'
              for c in columns]),
            ('встали за 6 с / подъём, с (медиана)', None,
             [f'{c["up"]}/{c["starts"]} / {f(c["rise"])}' if 'up' in c else '—' for c in columns]),
            ('полёт ≥ 40 мс, подъёмов', '≤ 2 из 20',
             [f'{c["flights"]} из {c["up"]}' if 'up' in c else '—' for c in columns])]
    first = max(len(r[0]) for r in rows)
    widths = [max(len(h), *(len(r[2][k]) for r in rows)) for k, h in enumerate(heads)]
    print(f'{"мера":<{first}}  {"порог":>12}  ' + '  '.join(f'{h:>{w}}' for h, w in zip(heads, widths)))
    for name, threshold, cells in rows:
        print(f'{name:<{first}}  {threshold or "":>12}  ' + '  '.join(f'{c:>{w}}' for c, w in zip(cells, widths)))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stages', nargs='+', help='этапы g1_getup: папка этапа или эксперимент[:шаги]')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--per-class', type=int, default=4, help='стартов на класс (спина, живот, бок, сидя, падение)')
    parser.add_argument('--starts', type=Path, help='старты из файла (JSON: список {cls, seed, qpos, qvel})')
    parser.add_argument('--library', default='lafan1', help='библиотека кадров человека (runs/humanoid_lab/getup_style)')
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    stages = [resolve(spec) for spec in args.stages]
    for stage in stages:
        task_id = json.loads((stage / 'metadata.json').read_text(encoding='utf-8')).get('task_id')
        if task_id != 'g1_getup':
            raise SystemExit(f'{stage}: не этап вставания (task_id {task_id})')
    reference = human(args.library)
    if args.starts:
        starts, per = [], {}
        for start in json.loads(args.starts.read_text(encoding='utf-8')):
            per[start['cls']] = per.get(start['cls'], 0) + 1
            if per[start['cls']] <= args.per_class:
                starts.append(start)
    else:
        starts = fresh_starts(args.per_class, args.workers)
    jobs = [dict(stage=str(stage), library=args.library, **start) for stage in stages for start in starts]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    per_stage = [[r for r, job in zip(results, jobs) if job['stage'] == str(stage)] for stage in stages]
    print(f'Стартов: {len(starts)} ({", ".join(f"{k} {sum(s["cls"] == k for s in starts)}" for k in KINDS)}); '
          f'без шума, наша физика, граница ПД — нижняя по этапу, без подтяжки; библиотека {args.library}.')
    table([*(label(stage) for stage in stages), 'человек 5'], [*(summary(rs) for rs in per_stage), reference])
    if args.out:
        args.out.write_text(json.dumps(dict(stages=[str(s) for s in stages], human=reference, starts=[
            dict(cls=s['cls'], seed=s['seed']) for s in starts], runs=per_stage), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
