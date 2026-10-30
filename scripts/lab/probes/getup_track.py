"""CPU only. getup_track.py <stage | reference> [--clip NAME] [--compare NAME …] [--kinematics] [--episodes 10]
[--clip-index K …] [--frame K] [--physics ours,delay20,foot0.3,motor1.25] [--workers 4] [--out <file.json>]
--frame K (iteration 82, spliced clips: does a policy still rise from where the old clip begins in the new one?):
the clip starts from frame K with its velocities and the same start noise (the reset option track_frame), not
from the first frame with none.
A track_clips stage (iteration 78) runs the first-frame starts of every clip of its list (or of --clip-index K, which
may repeat; the reset option track_clip_index), --episodes per clip and physics, and adds a line per clip (reached the
end, standing, the cut's median time) and how many clips reached the end at least once; its bank starts take their
nearest clip.
The acceptance of g1_getup's motion tracking (task_config track_clip, iteration 64): how far a policy follows its human
clip, without the assist (force 0), deterministic.

reference: no policy, the PD follows the clip itself (each step's targets the clip's joints at the next frame, the
locked wrists at home): does our PD with the motor model carry the clip? (--clip, the clip name; by default the stage's
or LAFAN1 fallAndGetUp1 subject 1, frames 2104-2209.) First a table of the clip's own kinematics, no physics, next to
each --compare clip (clip_kinematics: the feet's width, the hips' roll spread, self-penetration, joints beyond the
soft limits, the hands' floor time and a static estimate of their load, both hands and the more loaded one, with the
time it carries more than ONE_HAND_LOAD); --kinematics prints only that.

Starts: the clip's first frame with mjlab's start noise (G1GetupEnv's evaluation reset: assist force set, seeds 74000 +
k), --episodes per physics, in four physics of getup_probe.PHYSICS (ours, a 20 ms delay, foot friction 0.3, motor
friction ×1.25; the files' friction elsewhere, as ready38); then the start bank's held-out rows lying on the back and on
a side (10 each, physics ours; the reference aligned to each, task.track_align).
Per run: whether the clip clock reached the clip's last frame without a tracking termination (track_done), the time
of a termination; whether the robot stands at the end of the hold (the episode truncated, not ended, and its last step
task.getup_standing: head top above 1.15 m, torso tilt below 15°, pelvis above 0.70 m, both soles on the floor; info
track_standing): track_done alone passes a robot left on one knee (the anchor's 0.25 m height tolerance: the front
tracker exp-12d13a353c4647b5 @ 290 M reaches the end 12 of 12 with its head at ~1.0 m and stands 0 of 12), and the
head height at the end; over the clip: per joint the median |q − q_clip|, the feet's largest and median horizontal
distance, the hips' largest spread in roll (left − right, + outwards), the floor's force on the hands (median of the
steps with a touch, largest); the share of physics substeps with a motor above 90 % of its peak (getup_probe.Monitor), the fastest
joint. Printed per physics and per bank class.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import getup_probe  # noqa: E402

PHYSICS = ('ours', 'delay20', 'foot0.3', 'motor1.25')
BANK_CLASSES = ('back', 'side')
DEFAULT_CLIP = 'lafan1-fallAndGetUp1_subject1-2104-2209'
SEED = 74_000
_cache = {}


def _setup(job):
    """(environment, policy or None, normalizer or None) of a job, made once per process and source."""
    key = (job['policy'], job['clip'])
    if key not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.tasks import g1_getup_track
        from humanoid_sim.tasks.catalog import make_task
        if job['policy'] == 'reference':
            clip = g1_getup_track.load(job['clip'])
            policy = norm = None
            config = {'track_clip': {'file': job['clip'], 'md5': clip['md5']}}
        else:
            policy, norm, config = getup_probe._policy(job['policy'])
        env = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100_000})
        if env.unwrapped.clip is None:
            raise SystemExit(f'{job["policy"]}: этап без track_clip')
        _cache[key] = env, policy, norm
    return _cache[key]


def run(job):
    import mujoco  # noqa: F401  (getup_probe's monitor wraps mujoco.mj_step)
    env, policy, norm = _setup(job)
    raw = env.unwrapped
    getup_probe._install_monitor()
    friction, delay, sole = getup_probe.PHYSICS[job['physics']]
    options = dict(assist_force=0., action_delay_steps=delay, motor_friction_scale=friction)
    if job['row'] is not None:
        options['start'] = job['row']
    elif job.get('clip_index') is not None:  # track_clips: this clip's noisy first frame
        options['track_clip_index'] = job['clip_index']
    if job['row'] is None and job.get('frame') is not None:  # --frame: that clip frame, its velocities, the noise
        options['track_frame'] = job['frame']
    obs, _ = env.reset(seed=job['seed'], options=options)
    clip_index = getattr(raw, 'clip_index', None) if raw.clips is not None else None
    getup_probe.apply_friction(raw, sole)
    getup_probe.MONITOR = monitor = getup_probe.Monitor(raw)
    left, right = raw.hip_roll_joints
    errors, feet, spread, hands, speeds = [], [], [], [], []
    info = {}
    try:
        while True:
            if policy is None:
                frame = min(raw.clip_clock + 1, raw.clip_last)
                action = (raw.clip['qpos'][frame, 7:] - raw.home) / raw.scale
            else:
                action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            if info.get('bad_state'):
                break
            q = raw.data.qpos[raw.qadr]
            if raw.clip_clock <= raw.clip_last:
                errors.append(np.abs(q - raw.clip['qpos'][raw.clip_clock, 7:]))
                feet.append(float(np.linalg.norm(raw.data.xpos[raw.feet[0], :2] - raw.data.xpos[raw.feet[1], :2])))
                spread.append(float(q[left] - q[right]))
                hands.append(info['hand_force'])
            speeds.append(np.abs(raw.data.qvel[raw.vadr]))
            if terminated or truncated:
                break
    finally:
        getup_probe.MONITOR = None
    names = [raw.model.joint(j).name[:-6] for j in raw.joints]
    errors = np.array(errors) if errors else np.zeros((1, raw.model.nu))
    speeds = np.array(speeds) if speeds else np.zeros((1, raw.model.nu))
    touched = [h for h in hands if h > 0]
    substeps = max(monitor.substeps, 1)
    return dict(physics=job['physics'], start=job['start'], row=job['row'], seed=job['seed'], clip_index=clip_index,
                clip=None if clip_index is None else raw.clips[clip_index]['name'],
                done=bool(info.get('track_done')), failed=bool(info.get('track_failed')),
                standing=bool(truncated and not terminated and info.get('track_standing')),
                head_end=info.get('head_height'),
                bad_state=bool(info.get('bad_state')), seconds=round(raw.steps * raw.dt, 2),
                clock=int(raw.clip_clock), joint_error=np.median(errors, axis=0).tolist(), joints=names,
                feet_max=max(feet, default=float('nan')), feet_median=float(np.median(feet)) if feet else float('nan'),
                spread_max=max(spread, default=float('nan')),
                hand_median=float(np.median(touched)) if touched else 0., hand_max=max(hands, default=0.),
                hand_share=len(touched) / max(len(hands), 1), saturated=monitor.saturated / substeps,
                saturated_joint=(monitor.over / substeps).tolist(), speed_max=float(speeds.max()),
                speed_joint=names[int(np.argmax(speeds.max(0)))], getup_held=info.get('getup_held'))


RISE_PELVIS = .3  # m: frames with the pelvis above it are «the rise» for the feet's width
TOUCH = .01  # m: a geom this close to the floor touches it (the kinematic support)
PENETRATION = .01  # m: self-penetration deeper than this is counted
SMOOTH = 5  # frames: the centre of mass is averaged over ±SMOOTH before its acceleration is taken
G = 9.81
ONE_HAND_LOAD = 150.  # N on one hand: above it a clip leans on a Dex3-1 hand too hard (the lead's limit, iteration 66)


def _nonnegative_least_squares(a, b, iterations=3000):
    """f ≥ 0 minimizing |a f − b|² (projected gradient; a few dozen unknowns)."""
    step = 1 / max(np.linalg.norm(a, 2) ** 2, 1e-12)
    f = np.zeros(a.shape[1])
    for _ in range(iterations):
        f = np.maximum(f - step * (a.T @ (a @ f - b)), 0.)
    return f


def clip_kinematics(name):
    """The clip `name`'s own kinematics in the getting-up model, no physics: seconds; the feet's horizontal distance
    (ankle_roll origins; max during the rise, pelvis above RISE_PELVIS); the hips' roll spread (left − right, +
    outwards); self-penetration (the deepest contact between two robot geoms, the frames deeper than PENETRATION and
    the bodies); the joints beyond the task's soft limits (task.soft_limits, the joint_limit term: Σ excess, the share
    of frames, the joints); the hands' floor support: the frames with a wrist link's geom within TOUCH of the floor
    and a static estimate of their load: per frame vertical forces ≥ 0 at the touching points (MuJoCo's contacts with
    the floor's margin TOUCH) carrying M·(g + a_z) of the centre of mass with their centre of pressure at its ZMP
    (the COM averaged over ±SMOOTH frames, its acceleration by central differences, the angular momentum's change left
    out), one force per touching body at its contacts' centre, the least squares with the least norm, so the load is
    spread over every body it can use."""
    import copy
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    clip = g1_getup_track.load(name)
    env = getup_model_env()
    model = copy.deepcopy(env.model)
    data = mujoco.MjData(model)
    floor = model.geom('floor').id
    model.geom_margin[floor] = TOUCH
    robot = np.asarray(env.robot_geoms, bool)
    hand_bodies = {b for b in range(model.nbody) if 'wrist' in model.body(b).name}
    hand_sides = [{b for b in hand_bodies if model.body(b).name.startswith(side)} for side in ('left', 'right')]
    left, right = (model.jnt_qposadr[model.joint(f'{side}_hip_roll_joint').id] for side in ('left', 'right'))
    low, high = model.jnt_range[env.joints].T
    soft_low, soft_high = task.soft_limits(low, high)
    names = [model.joint(j).name[:-6] for j in env.joints]
    mass = float(model.body_subtreemass[env.pelvis])
    qpos = clip['qpos']
    dt = env.dt
    com = []
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com.append(data.subtree_com[env.pelvis].copy())
    com = np.array(com)
    index = np.clip(np.arange(len(com))[:, None] + np.arange(-SMOOTH, SMOOTH + 1)[None], 0, len(com) - 1)
    smooth = com[index].mean(1)
    acceleration = np.zeros_like(com)
    acceleration[1:-1] = (smooth[2:] - 2 * smooth[1:-1] + smooth[:-2]) / dt ** 2
    feet, spread, depth, deep_pairs, excess, beyond, hand_load, hand_touch = [], [], [], {}, [], set(), [], []
    one_hand = []  # per frame the more loaded hand's share of hand_load (N)
    for k, row in enumerate(qpos):
        data.qpos[:] = row
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        q = row[env.qadr]
        feet.append(float(np.linalg.norm(data.xpos[env.feet[0], :2] - data.xpos[env.feet[1], :2])))
        spread.append(float(row[left] - row[right]))
        over = np.maximum(q - soft_high, 0) + np.maximum(soft_low - q, 0)
        excess.append(float(over.sum()))
        beyond.update(names[j] for j in np.flatnonzero(over > 1e-9))
        deepest, points = 0., {}
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if robot[g1] and robot[g2]:
                if contact.dist < deepest:
                    deepest = float(contact.dist)
                if contact.dist < -PENETRATION:
                    pair = tuple(sorted(model.body(model.geom_bodyid[g]).name for g in (g1, g2)))
                    deep_pairs[pair] = deep_pairs.get(pair, 0) + 1
            elif floor in (g1, g2) and contact.dist < TOUCH:
                body = int(model.geom_bodyid[g2 if g1 == floor else g1])
                points.setdefault(body, []).append(contact.pos[:2].copy())
        depth.append(deepest)
        bodies = sorted(points)
        hand = np.array([b in hand_bodies for b in bodies], bool)
        hand_touch.append(bool(hand.any()))
        if hand.any():
            support = G + acceleration[k, 2]
            zmp = com[k, :2] - com[k, 2] * acceleration[k, :2] / max(support, 1e-6)
            centres = np.array([np.mean(points[b], axis=0) for b in bodies])
            a = np.vstack([np.ones(len(centres)), centres.T])
            f = _nonnegative_least_squares(a, mass * max(support, 0.) * np.array([1., *zmp]))
            hand_load.append(float(f[hand].sum()))
            one_hand.append(max(float(f[[b in side for b in bodies]].sum()) for side in hand_sides))
        else:
            hand_load.append(0.)
            one_hand.append(0.)
    pelvis = qpos[:, 2]
    rise = pelvis > RISE_PELVIS
    hand_touch = np.array(hand_touch)
    touched = np.array(hand_load)[hand_touch]
    single = np.array(one_hand)[hand_touch]
    return dict(clip=name, seconds=round((len(qpos) - 1) * dt, 2), source=clip['header'].get('source'),
                dataset=clip['header'].get('dataset', 'lafan1'),
                feet_max=max(feet), feet_rise_max=max(np.array(feet)[rise], default=float('nan')),
                feet_end=feet[-1], spread_max=max(spread), spread_min=min(spread),
                spread_rise_max=max(np.array(spread)[rise], default=float('nan')),
                self_depth=-min(depth), self_frames=int((np.array(depth) < -PENETRATION).sum()),
                self_pairs={' / '.join(p): n for p, n in sorted(deep_pairs.items(), key=lambda x: -x[1])[:4]},
                limit_excess_max=max(excess), limit_frames=float(np.mean(np.array(excess) > 1e-9)),
                limit_joints=sorted(beyond), hand_seconds=round(float(hand_touch.sum()) * dt, 2),
                hand_rise_seconds=round(float((hand_touch & rise).sum()) * dt, 2),
                hand_load_median=float(np.median(touched)) if len(touched) else 0.,
                hand_load_max=float(touched.max()) if len(touched) else 0., weight=mass * G,
                one_hand_median=float(np.median(single)) if len(single) else 0.,
                one_hand_max=float(single.max()) if len(single) else 0.,
                one_hand_over=round(float((single > ONE_HAND_LOAD).sum()) * dt, 2),
                one_hand=[round(v, 1) for v in one_hand])


def print_kinematics(rows):
    """A table of clip_kinematics rows, one column per clip."""
    lines = [
        ('длительность, с', lambda r: f'{r["seconds"]:.2f}'),
        ('стопы врозь: max в подъёме (таз > 0,3 м) / max / в конце, м',
         lambda r: f'{r["feet_rise_max"]:.2f} / {r["feet_max"]:.2f} / {r["feet_end"]:.2f}'),
        ('разведение бёдер по крену (лев. − прав.), рад: max в подъёме / max / min',
         lambda r: f'{r["spread_rise_max"]:.2f} / {r["spread_max"]:.2f} / {r["spread_min"]:.2f}'),
        ('проникание в себя: глубже всего, см / кадров > 1 см',
         lambda r: f'{r["self_depth"] * 100:.1f} / {r["self_frames"]}'),
        ('где (кадров)', lambda r: '; '.join(f'{p} {n}' for p, n in r['self_pairs'].items()) or '—'),
        ('за мягкими пределами: Σ max, рад / доля кадров',
         lambda r: f'{r["limit_excess_max"]:.2f} / {r["limit_frames"]:.0%}'),
        ('какие суставы', lambda r: ', '.join(r['limit_joints']) or '—'),
        ('кисти на полу: всего / в подъёме, с', lambda r: f'{r["hand_seconds"]:.2f} / {r["hand_rise_seconds"]:.2f}'),
        ('нагрузка кистей по кинематике: медиана / max, Н (вес)',
         lambda r: f'{r["hand_load_median"]:.0f} / {r["hand_load_max"]:.0f} ({r["weight"]:.0f})'),
        (f'на одну кисть (больше нагруженную): медиана / max, Н / дольше {ONE_HAND_LOAD:.0f} Н, с',
         lambda r: f'{r["one_hand_median"]:.0f} / {r["one_hand_max"]:.0f} / {r["one_hand_over"]:.2f}'),
    ]
    print('Кинематика клипа (без физики): ' + ' | '.join(f'{r["clip"]} ({r["dataset"]})' for r in rows))
    for title, value in lines:
        print(f'   {title}: ' + ' | '.join(value(r) for r in rows))


def summary(runs, title):
    if not runs:
        return
    from humanoid_sim.tasks import g1_tensor_math as task
    done = sum(r['done'] for r in runs)
    standing = sum(r.get('standing', False) for r in runs)
    failed = [r['seconds'] for r in runs if not r['done']]
    errors = np.median([r['joint_error'] for r in runs], axis=0)
    names = runs[0]['joints']
    worst = np.argsort(-errors)[:6]
    saturated = np.mean([r['saturated_joint'] for r in runs], axis=0)
    heads = [r['head_end'] for r in runs if r.get('head_end') is not None]
    print(f'{title}: дошли до конца клипа {done} из {len(runs)} ({done / len(runs):.0%}); встали и стоят в конце '
          f'удержания {standing} из {len(runs)} ({standing / len(runs):.0%}; макушка > {task.GETUP_STAND_HEAD:.2f} м, '
          f'наклон < {np.degrees(task.GETUP_STAND_TILT):.0f}°, таз > {task.GETUP_STAND_PELVIS:.2f} м, обе подошвы '
          f'на полу)' + (f', макушка в конце медиана {np.median(heads):.2f} м' if heads else '')
          + (f'; обрыв на {np.median(failed):.2f} с (медиана)' if failed else ''))
    print(f'   ошибка суставов (медиана |q − клип|): все {np.median(errors):.3f} рад; больше всего '
          + ', '.join(f'{names[j]} {errors[j]:.2f}' for j in worst))
    print(f'   стопы врозь: max {np.median([r["feet_max"] for r in runs]):.2f} м, медиана '
          f'{np.median([r["feet_median"] for r in runs]):.2f} м; разведение бёдер по крену (лев. − прав.) max '
          f'{np.median([r["spread_max"] for r in runs]):.2f} рад')
    print(f'   сила пола на кисти: медиана касания {np.median([r["hand_median"] for r in runs]):.0f} Н, max '
          f'{max(r["hand_max"] for r in runs):.0f} Н, касание в {np.mean([r["hand_share"] for r in runs]):.0%} шагов')
    joint = int(np.argmax(saturated))
    print(f'   моторы > 90 % пика: {np.mean([r["saturated"] for r in runs]):.1%} времени (больше всего {names[joint]} '
          f'{saturated[joint]:.1%}); скорость суставов max {max(r["speed_max"] for r in runs):.1f} рад/с '
          f'({max(runs, key=lambda r: r["speed_max"])["speed_joint"]})')


def clip_table(runs, names, classes):
    """track_clips: one line per clip of its first-frame runs (every physics): reached the end, standing, the median
    time of a cut."""
    print('по клипам (шумный первый кадр, все физики): клип, класс начала — до конца / стоят / прогонов; обрыв, с')
    for index, name in enumerate(names):
        mine = [r for r in runs if r['start'] == 'clip' and r['clip_index'] == index]
        if not mine:
            continue
        done = sum(r['done'] for r in mine)
        standing = sum(r.get('standing', False) for r in mine)
        cut = [r['seconds'] for r in mine if not r['done']]
        print(f'   {index:2d} {name} ({classes[index]}): {done} / {standing} / {len(mine)}'
              + (f'; обрыв {np.median(cut):.2f}' if cut else ''))
    done = sum(any(r['done'] for r in runs if r['start'] == 'clip' and r['clip_index'] == k) for k in range(len(names)))
    print(f'   клипов, дошедших до конца хотя бы раз: {done} из {len(names)}')


def stage_clips(policy):
    """(names, start classes) of a track_clips stage's clips, None for another stage."""
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    entries = stage_task_config(Path(policy)).get('track_clips')
    if not entries:
        return None
    classes = []
    for entry in entries:
        clip = g1_getup_track.load(entry['file'], entry['md5'])
        rotation = g1_getup_track.hold_frames(clip)['rot'][:1, 0]
        kind = int(task.track_start_class(task.projected_gravity(rotation))[0])
        classes.append(task.TRACK_CLASSES[kind] if kind >= 0 else '—')
    return [entry['file'] for entry in entries], classes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy', help='этап g1_getup с track_clip или reference (ПД по самому клипу, без политики)')
    parser.add_argument('--clip-index', type=int, action='append', default=[],
                        help='этап с track_clips: только этот клип (номер в списке, можно несколько раз; по умолчанию '
                             'все, с итогом по каждому)')
    parser.add_argument('--physics', default=','.join(PHYSICS), help='физики через запятую: ' + ', '.join(PHYSICS))
    parser.add_argument('--clip', default=DEFAULT_CLIP, help='клип для reference (runs/humanoid_lab/getup_track/<имя>.npz)')
    parser.add_argument('--episodes', type=int, default=10, help='стартов с шумного первого кадра на физику')
    parser.add_argument('--frame', type=int, help='старты с этого кадра клипа (его скорости и тот же шум), а не с первого')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--compare', action='append', default=[],
                        help='reference: ещё клип для таблицы кинематики (можно несколько раз)')
    parser.add_argument('--kinematics', action='store_true', help='reference: только кинематика клипов, без прогонов')
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    policy = 'reference' if args.policy == 'reference' else str(Path(args.policy).resolve())
    kinematics = []
    if policy == 'reference':
        kinematics = [clip_kinematics(name) for name in [args.clip, *args.compare]]
        print_kinematics(kinematics)
        if args.kinematics:
            if args.out:
                args.out.write_text(json.dumps(dict(kinematics=kinematics), indent=1), encoding='utf-8')
            return
    physics = [p for p in args.physics.split(',') if p]
    if set(physics) - set(PHYSICS):
        raise SystemExit(f'физики: {", ".join(PHYSICS)}')
    clips = None if policy == 'reference' else stage_clips(policy)
    if args.clip_index and clips is None:
        raise SystemExit('--clip-index — для этапа с track_clips')
    indices = [None] if clips is None else (args.clip_index or list(range(len(clips[0]))))
    if clips is not None and any(not 0 <= k < len(clips[0]) for k in indices):
        raise SystemExit(f'--clip-index: клипы 0…{len(clips[0]) - 1}')
    jobs = [dict(policy=policy, clip=args.clip, physics=p, start='clip', row=None, seed=SEED + k, clip_index=index,
                 frame=args.frame) for index in indices for p in physics for k in range(args.episodes)]
    bank = bank_module.load()
    for kind in BANK_CLASSES:
        rows = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)))
        jobs += [dict(policy=policy, clip=args.clip, physics='ours', start=kind, row=int(row), seed=SEED + 500 + k)
                 for k, row in enumerate(rows)]
    with Pool(args.workers) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    print(f'{policy}: следование клипу ({"ПД по клипу, без политики" if policy == "reference" else "политика"})')
    first = 'шумный первый кадр' if args.frame is None else f'шумный кадр {args.frame} ({args.frame * .02:.2f} с)'
    for p in physics:
        summary([r for r in runs if r['start'] == 'clip' and r['physics'] == p], f'{first}, {p}')
    for kind in BANK_CLASSES:
        summary([r for r in runs if r['start'] == kind], f'банк, {kind} (отложенные строки, ours)')
    if clips is not None:
        clip_table(runs, *clips)
    if args.out:
        args.out.write_text(json.dumps(dict(policy=policy, clip=args.clip, runs=runs, kinematics=kinematics), indent=1),
                            encoding='utf-8')


if __name__ == '__main__':
    main()
