"""CPU only. getup_settle.py <stage | random> [--clips NAME,NAME] [--tracker <stage> …] [--physics ours,…]
[--per-class 10] [--workers 4] [--out <file.json>]
The acceptance of g1_getup's settling (task_config settle_clip, iteration 66): from a fall pose to the nearest of the
clips' first frames, without the assist (force 0), deterministic.

Starts: the start bank's held-out rows of every settling class (g1_tensor_math.SETTLE_CLASSES: back, front, side,
fall, sit; --per-class of each, all by default), in four physics of getup_probe.PHYSICS (ours, a 20 ms delay, foot
friction 0.3, motor friction ×1.25; the files' friction elsewhere, as ready38), seeds 76000 + k. random: no stage, a
random policy (actions N(0, 1) per joint, the first stage's σ, clipped to the box) in a settle task of --clips (default
the back's cmu-140_09-61-571 and the front's cmu-140_01-235-571, which starts lying on the chest with the hands under
the shoulders: its static load stays within 141 N on one hand, the take from 151 reaches 190) with impact 2 and
sole_impact 1, to check that every
number is computed.
Per run, over the task's 7 s: whether it settled (near the nearest target, every free joint slower than 0.5 rad/s for
0.3 s: the switch for g1_ctrl) and when, into which clip's first frame; at the end the gravity angle (deg) and mean
|q − q0| (rad) to the nearest target; the floor's force on the hands (median of the steps with a touch, largest, share
of steps), the fastest approach speed to the floor of a body other than the feet and of a sole (m/s, impact's and
sole_impact's) with the steps above 1.5 / 0.5 m/s; the share of physics substeps with a motor above 90 % of its peak
(getup_probe.Monitor), the fastest joint.
Chain (--tracker, once per clip; a g1_getup stage with track_clip): after a run settles, its state goes to the tracker
of the clip it settled into, as g1_ctrl would switch (the reference aligned to the robot's pelvis, track_align, with
the bank start's grace; a fresh delay queue and history), until the clip's end + 3 s or a tracking termination: the
share that reached the clip's last frame (track_done) and, of those, got up and stood 3 s (getup_held). Printed per
class and physics, then per class over the physics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import getup_probe  # noqa: E402

PHYSICS = ('ours', 'delay20', 'foot0.3', 'motor1.25')
DEFAULT_CLIPS = ('cmu-140_09-61-571', 'cmu-140_01-235-571')
RANDOM_STD = 1.
SEED = 76_000
_cache = {}


class RandomPolicy:
    """A random policy with predict's signature: N(0, RANDOM_STD) per joint, its own stream per run (seed)."""

    def __init__(self, seed, size):
        self.rng, self.size = np.random.default_rng(seed), size

    def predict(self, obs, deterministic=True):
        return self.rng.normal(0., RANDOM_STD, self.size), None


class Identity:
    def normalize_obs(self, obs):
        return obs


def random_config(clips):
    from humanoid_sim.tasks import g1_getup_track
    return {'settle_clip': [{'file': name, 'md5': g1_getup_track.load(name)['md5']} for name in clips],
            'reward_weights': {'impact': 2., 'sole_impact': 1.}}


def _settler(job):
    """(environment, policy or None, normalizer) of the settling stage (or random), made once per process."""
    key = ('settle', job['policy'], tuple(job['clips']))
    if key not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.tasks.catalog import make_task
        if job['policy'] == 'random':
            policy, norm, config = None, Identity(), random_config(job['clips'])
        else:
            policy, norm, config = getup_probe._policy(job['policy'])
        env = make_task('g1_getup', 'g1_29dof', config)
        if env.unwrapped.settle is None:
            raise SystemExit(f'{job["policy"]}: этап без settle_clip')
        _cache[key] = env, policy, norm
    return _cache[key]


def _tracker(path):
    """(environment, policy, normalizer) of a track_clip stage, made once per process."""
    key = ('track', path)
    if key not in _cache:
        from humanoid_sim.tasks.catalog import make_task
        policy, norm, config = getup_probe._policy(path)
        env = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100_000})
        if env.unwrapped.clip is None:
            raise SystemExit(f'{path}: трекер без track_clip')
        _cache[key] = env, policy, norm
    return _cache[key]


def chain(job, state, target):
    """The tracker of clip `target` from the settled state (qpos, qvel): its record (track_done, track_failed, the
    clip clock at the end, getup_held), or None without a tracker of that clip."""
    import mujoco
    from humanoid_sim.tasks import g1_tensor_math as task
    path = job['trackers'].get(target)
    if path is None:
        return None
    env, policy, norm = _tracker(path)
    raw = env.unwrapped
    friction, delay, sole = getup_probe.PHYSICS[job['physics']]
    env.reset(seed=job['seed'] + 1, options=dict(assist_force=0., action_delay_steps=delay,
                                                 motor_friction_scale=friction, track_frame=0, track_noise=False))
    getup_probe.apply_friction(raw, sole)
    raw.data.qpos[:], raw.data.qvel[:] = state
    mujoco.mj_forward(raw.model, raw.data)
    # As a bank start: the reference's first pelvis pose turned and moved onto the robot's, the limbs' grace.
    yaw, offset = task.track_align(raw.data.xpos[raw.pelvis][None], raw.data.xmat[raw.pelvis].reshape(1, 3, 3),
                                   raw.clip_frames['pos'][0, 0], raw.clip_frames['rot'][0, 0])
    raw.clip_yaw, raw.clip_offset, raw.clip_start = float(yaw[0]), offset[0].copy(), 'bank'
    obs = raw._observation('fill')
    info = {}
    while True:
        obs, _, terminated, truncated, info = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if info.get('bad_state') or terminated or truncated:
            break
    return dict(tracker=path, done=bool(info.get('track_done')), failed=bool(info.get('track_failed')),
                clock=int(raw.clip_clock), held=info.get('getup_held') is True, seconds=round(raw.steps * raw.dt, 2))


def run(job):
    import mujoco  # noqa: F401  (getup_probe's monitor wraps mujoco.mj_step)
    env, policy, norm = _settler(job)
    raw = env.unwrapped
    if policy is None:
        policy = RandomPolicy(job['seed'], raw.model.nu)
    getup_probe._install_monitor()
    friction, delay, sole = getup_probe.PHYSICS[job['physics']]
    obs, _ = env.reset(seed=job['seed'], options=dict(assist_force=0., action_delay_steps=delay,
                                                      motor_friction_scale=friction, start=job['row']))
    getup_probe.apply_friction(raw, sole)
    getup_probe.MONITOR = monitor = getup_probe.Monitor(raw)
    hands, body_speed, sole_speed, speeds = [], [], [], []
    info, state = {}, None
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if info.get('bad_state'):
                break
            hands.append(info['hand_force'])
            if raw.impact_substeps is not None:
                body_speed.append(float(raw.impact_substeps.max()))
            if raw.sole_impact_substeps is not None:
                sole_speed.append(float(raw.sole_impact_substeps.max()))
            speeds.append(np.abs(raw.data.qvel[raw.vadr]))
            if state is None and info['settle_done']:
                state = (raw.data.qpos.copy(), raw.data.qvel.copy())
            if terminated or truncated:
                break
    finally:
        getup_probe.MONITOR = None
    names = [raw.model.joint(j).name[:-6] for j in raw.joints]
    speeds = np.array(speeds) if speeds else np.zeros((1, raw.model.nu))
    touched = [h for h in hands if h > 0]
    substeps = max(monitor.substeps, 1)

    def fastest(values, threshold):
        return dict(max=max(values, default=float('nan')), over=int(sum(v > threshold for v in values)))
    record = dict(physics=job['physics'], start=job['start'], row=job['row'], seed=job['seed'],
                  settled=bool(info.get('settle_done')), settle_s=info.get('settle_seconds'),
                  target=info.get('settle_target'), bad_state=bool(info.get('bad_state')),
                  angle=info.get('settle_angle'), deviation=info.get('settle_deviation'),
                  seconds=round(raw.steps * raw.dt, 2), hand_median=float(np.median(touched)) if touched else 0.,
                  hand_max=max(hands, default=0.), hand_share=len(touched) / max(len(hands), 1),
                  body=fastest(body_speed, 1.5), sole=fastest(sole_speed, .5), saturated=monitor.saturated / substeps,
                  speed_max=float(speeds.max()), speed_joint=names[int(np.argmax(speeds.max(0)))])
    if job['trackers'] and state is not None:
        record['chain'] = chain(job, state, record['target'])
    return record


def summary(runs, title):
    if not runs:
        return
    settled = [r for r in runs if r['settled']]
    targets = Counter(r['target'] for r in settled)
    print(f'{title}: улёгся {len(settled)} из {len(runs)} ({len(settled) / len(runs):.0%})'
          + (f' за {np.median([r["settle_s"] for r in settled]):.2f} с (медиана)' if settled else '')
          + ('; в позу ' + ', '.join(f'{name} {n}' for name, n in targets.most_common()) if targets else ''))
    finite = [r for r in runs if r['angle'] is not None]
    if finite:
        print(f'   в конце до ближайшей цели: наклон таза {np.median([r["angle"] for r in finite]):.0f}°, '
              f'|q − q0| {np.median([r["deviation"] for r in finite]):.2f} рад (медианы)')
    print(f'   сила пола на кисти: медиана касания {np.median([r["hand_median"] for r in runs]):.0f} Н, max '
          f'{max(r["hand_max"] for r in runs):.0f} Н, касание в {np.mean([r["hand_share"] for r in runs]):.0%} шагов')
    for key, title_, threshold in (('body', 'тело (не стопы)', 1.5), ('sole', 'подошвы', .5)):
        values = [r[key]['max'] for r in runs if np.isfinite(r[key]['max'])]
        if values:
            print(f'   удары, {title_}: скорость сближения с полом max {np.median(values):.2f} м/с (медиана прогонов), '
                  f'{max(values):.2f} худший; шагов выше {threshold:g} м/с — {np.mean([r[key]["over"] for r in runs]):.1f} '
                  'на прогон')
    print(f'   моторы > 90 % пика: {np.mean([r["saturated"] for r in runs]):.1%} времени; скорость суставов max '
          f'{max(r["speed_max"] for r in runs):.1f} рад/с ({max(runs, key=lambda r: r["speed_max"])["speed_joint"]})')
    chained = [r['chain'] for r in settled if r.get('chain')]
    missing = sum(1 for r in settled if 'chain' in r and r['chain'] is None)
    if chained or missing:
        done = sum(c['done'] for c in chained)
        print(f'   цепочка → трекер: дошли до конца клипа {done} из {len(chained)}'
              + (f', встали и простояли 3 с {sum(c["held"] for c in chained)}' if chained else '')
              + (f'; нет трекера для позы — {missing}' if missing else ''))


def main():
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    from humanoid_sim.tasks import g1_tensor_math as task
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument('policy', help='этап g1_getup с settle_clip или random (случайная политика)')
    parser.add_argument('--clips', default=','.join(DEFAULT_CLIPS), help='random: клипы целей через запятую')
    parser.add_argument('--tracker', action='append', default=[], help='этап-трекер (track_clip) для цепочки')
    parser.add_argument('--physics', default=','.join(PHYSICS))
    parser.add_argument('--per-class', type=int, help='отложенных строк на класс (по умолчанию все)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    policy = 'random' if args.policy == 'random' else str(Path(args.policy).resolve())
    clips = tuple(args.clips.split(','))
    trackers = {}
    for path in args.tracker:
        path = str(Path(path).resolve())
        config = getup_probe._policy(path)[2]
        clip = (config.get('track_clip') or {}).get('file')
        if clip is None:
            raise SystemExit(f'{path}: этап без track_clip, не трекер')
        trackers[clip] = path
    physics = args.physics.split(',')
    if set(physics) - set(getup_probe.PHYSICS):
        raise SystemExit(f'неизвестная физика: {sorted(set(physics) - set(getup_probe.PHYSICS))}')
    bank = bank_module.load()
    jobs = []
    for kind in task.SETTLE_CLASSES:
        if kind not in bank['classes']:
            continue
        rows = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)))
        rows = rows[:args.per_class] if args.per_class else rows
        jobs += [dict(policy=policy, clips=clips, trackers=trackers, physics=p, start=kind, row=int(row),
                      seed=SEED + 100 * i + k) for i, p in enumerate(physics) for k, row in enumerate(rows)]
    with Pool(args.workers) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    print(f'{policy}: укладка в начальную позу клипа ({len(runs)} прогонов, без подтяжки)'
          + (f'; трекеры: {", ".join(f"{c} ← {p}" for c, p in trackers.items())}' if trackers else ''))
    for kind in task.SETTLE_CLASSES:
        for p in physics:
            summary([r for r in runs if r['start'] == kind and r['physics'] == p], f'{kind}, {p}')
    print('по классам, все физики:')
    for kind in task.SETTLE_CLASSES:
        summary([r for r in runs if r['start'] == kind], f'{kind}')
    if args.out:
        args.out.write_text(json.dumps(dict(policy=policy, clips=clips, trackers=trackers, runs=runs), indent=1),
                            encoding='utf-8')


if __name__ == '__main__':
    main()
