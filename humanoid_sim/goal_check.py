"""The G1 walking goal (stage 1 of docs/g1-roadmap.md) as one check with a tick per condition.

    .venv\\Scripts\\python.exe -m humanoid_sim.goal_check exp-…[:steps] [--seeds 5] [--journal]
    .venv\\Scripts\\python.exe -m humanoid_sim.goal_check --export <export_g1 folder> [--seeds 5] [--journal]

Runs the supervise scenarios (stand, walk 1 m/s, stop-start), then fixed-command walks with heading hold off:
turns at 1 m/s (±0.5 rad/s) and straight at 1 and 0.5 m/s, each straight walk started on either foot (gait phase
0 or half a cycle on). The straight walks give the drift without a turn command and the step: how high a foot
rises off the floor and how flat it stands. With --journal the result becomes the «Дневник» tab's current check.
--export judges an export folder's policy.onnx (export_g1.ExportedPolicy: what g1_ctrl runs, stand trim included) in
the CPU environment of the stage it came from; σ is that stage's. Its --journal check goes under that stage's branch,
titled with the export folder, so the journal says which of the two was checked.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint_schema import checkpoint_identity
from .core import DEFAULT_RUN, load_stage, read_json, stage_task_config
from .lab import Laboratory
from .supervise import check_stage, find_stage
from .tasks import g1_tensor_math as task

TURNS = ((1., .5), (1., -.5))  # (forward m/s, yaw rad/s), started at gait phase 0
STRAIGHT = (1., .5)  # forward m/s without a turn command
STARTS = ('left', 'right')  # gait phase 0 (left foot first) or half a cycle on (right foot first)
SETTLE_STEPS = 100  # the first 2 s are not measured
FEET = ('left', 'right')  # task.FEET order


def sole_geoms(raw):
    """Per foot, its collision spheres (the sole's contact points): the environment's soles, the foot geoms that touch
    the floor (task_config foot_collision's capsules collide only with each other)."""
    model = raw.model
    soles = [[int(g) for g in foot] for foot in raw.soles]
    if not all(soles) or any(model.geom_type[g] != 2 for geoms in soles for g in geoms):  # mjGEOM_SPHERE
        raise ValueError('G1 feet are expected to touch the floor with spheres')
    return soles


def foot_state(raw, soles, floor):
    """Per foot: (a sole sphere touches the floor in MuJoCo, height of the lowest sole point m, tilt of the
    ankle-roll link's z axis from vertical deg, ankle-roll origin height m — what the reward reads)."""
    count = raw.data.ncon
    touching = set()
    for a, b in zip(raw.data.contact.geom1[:count], raw.data.contact.geom2[:count]):
        if a == floor or b == floor:
            touching.update((int(a), int(b)))
    rows = []
    for foot, geoms in zip(raw.feet, soles):
        sole = min(raw.data.geom_xpos[g][2] - raw.model.geom_size[g][0] for g in geoms)
        tilt = math.degrees(math.acos(min(1., max(-1., raw.data.xmat[foot][8]))))
        rows.append((bool(touching.intersection(geoms)), float(sole), tilt, float(raw.data.xpos[foot][2])))
    return rows


def foot_summary(runs):
    """runs: per straight walk, per measured step, foot_state rows. Per foot over all walks: lift = the lowest
    over walks of the 95th percentile of sole height on steps without contact (0 for a foot that never leaves
    the floor), stance tilt = the highest over walks of the mean tilt on steps with contact, missed stance = the
    share of contact steps the reward counts as swing (ankle-roll origin ≥ CONTACT_HEIGHT). Worst foot first."""
    by_foot = {}
    for k, name in enumerate(FEET):
        lifts, tilts, contacts, missed = [], [], 0, 0
        for run in runs:
            if not run:
                continue
            rows = np.array([step[k] for step in run], dtype=float)  # contact, sole, tilt, origin
            contact = rows[:, 0] > 0
            lifts.append(float(np.percentile(rows[~contact, 1], 95)) if (~contact).any() else 0.)
            if contact.any():
                tilts.append(float(rows[contact, 2].mean()))
            contacts += int(contact.sum())
            missed += int((rows[contact, 3] >= task.CONTACT_HEIGHT).sum())
        by_foot[name] = dict(lift_m=min(lifts) if lifts else None, stance_tilt_deg=max(tilts) if tilts else None,
                             missed_stance=missed / contacts if contacts else None)
    lifts = [f['lift_m'] for f in by_foot.values() if f['lift_m'] is not None]
    tilts = [f['stance_tilt_deg'] for f in by_foot.values() if f['stance_tilt_deg'] is not None]
    return dict(lift_m=min(lifts) if len(lifts) == len(FEET) else None,
                stance_tilt_deg=max(tilts) if len(tilts) == len(FEET) else None, by_foot=by_foot)


def walks(stage, seeds=2, seconds=10., seed_start=73000, acting=None, turns=TURNS, straight=STRAIGHT, config=None):
    """Deterministic CPU walks at fixed commands, heading hold off, measured after the first 2 s.

    turn: {(speed, yaw): mean yaw rate}; drift: {(speed, start): [yaw rate per seed]} without a turn command;
    pelvis: the lowest pelvis height on the walks started at phase 0; feet: foot_summary of the straight walks.
    acting: (policy, norm) that act instead of the stage's own (export_g1.ExportedPolicy as both). turns, straight:
    the (forward, yaw) turn commands and the straight speeds walked (the goal's by default; scripts/lab/skills_check.py
    walks other turns alone). config: the task_config to walk in instead of the stage's (with acting: Unitree v0)."""
    from .tasks.catalog import make_task
    identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
    dt = identity['control_dt']
    policy, norm = acting or load_stage(stage)
    limit = math.ceil(seconds / dt)
    config = {**(stage_task_config(stage) if config is None else config), 'heading_hold': False,
              'max_episode_steps': limit}
    runs = [(speed, yaw, 'left') for speed, yaw in turns] + [(speed, 0., start) for speed in straight
                                                             for start in STARTS]
    result = dict(turn={}, drift={}, pelvis=math.inf)
    feet = []
    try:
        for speed, yaw, start in runs:
            measured = []
            for seed in range(seed_start, seed_start + seeds):
                env = make_task(identity['task_id'], identity['robot_id'], config)
                raw = env.unwrapped
                pelvis = raw.model.body('pelvis').id
                heading = lambda: math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0])
                soles, floor = sole_geoms(raw), raw.model.geom('floor').id
                # Half a gait cycle in control steps: the other foot leads.
                phase = 0 if start == 'left' else round(raw.gait['period'] / raw.dt / 2)
                steps = []
                try:
                    obs, _ = env.reset(seed=seed, options={'command': [speed, 0., yaw], 'phase_steps': phase})
                    track = [heading()]
                    for step in range(limit):
                        obs, _, terminated, truncated, _ = env.step(
                            policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
                        track.append(heading())
                        if step >= SETTLE_STEPS:
                            if start == 'left':
                                result['pelvis'] = min(result['pelvis'], float(raw.data.qpos[2]))
                            if yaw == 0.:
                                steps.append(foot_state(raw, soles, floor))
                        if terminated or truncated:
                            break
                finally:
                    env.close()
                track = np.unwrap(track)
                first = min(SETTLE_STEPS, len(track) - 2)
                measured.append(float((track[-1] - track[first]) / ((len(track) - 1 - first) * dt)))
                if yaw == 0.:
                    feet.append(steps)
            if yaw == 0.:
                result['drift'][(speed, start)] = measured
            else:
                result['turn'][(speed, yaw)] = float(np.mean(measured))
    finally:
        if acting is None:
            norm.close()
    result['feet'] = foot_summary(feet)
    return result


def items(scenarios, std, walk):
    """Goal conditions as journal check rows: label, measured value, target, ok."""
    stand, walking = scenarios['stand'], scenarios['walk']
    falls = [v['falls'] for v in scenarios.values()]
    crawl = math.hypot(stand['forward_m'], stand['sideways_m'])
    leans = [v['torso_lean_deg'] for v in scenarios.values() if v['torso_lean_deg'] is not None]
    clipped = max(v['clipped'] for v in scenarios.values())
    left, right = walk['turn'][(1., .5)], walk['turn'][(1., -.5)]
    drift = max(abs(rate) for rates in walk['drift'].values() for rate in rates)
    pelvis = walk['pelvis']
    lift, tilt = walk['feet']['lift_m'], walk['feet']['stance_tilt_deg']
    stepping = lift is not None and tilt is not None
    fmt = lambda v, d=2: f'{v:.{d}f}'.replace('.', ',')
    return [
        dict(label='Без падений: стоять, 1 м/с, стоп-старт', value=', '.join(falls), target='0 во всех',
             ok=all(f.startswith('0/') for f in falls)),
        dict(label='Ходьба 1 м/с: метров за 10 с', value=f"{fmt(walking['forward_m'], 1)} м", target='8–12 м',
             ok=8 <= walking['forward_m'] <= 12),
        dict(label='Курс при ходьбе', value=f"{fmt(walking['heading_deg'], 1)}°", target='< 10°',
             ok=walking['heading_deg'] < 10),
        dict(label='Стоять на месте: уползание за 10 с', value=f'{fmt(crawl)} м', target='< 0,3 м', ok=crawl < .3),
        dict(label='Наклон корпуса', value=f'{min(leans):+.0f}…{max(leans):+.0f}°' if leans else '—',
             target='−5…+15°', ok=bool(leans) and -5 <= min(leans) and max(leans) <= 15),
        dict(label='Высота таза на ходу (минимум)', value=f'{fmt(pelvis)} м', target='≥ 0,70 м', ok=pelvis >= .7),
        dict(label='Поворот по команде ±0,5 рад/с', value=f'{left:+.2f} / {right:+.2f} рад/с'.replace('.', ','),
             target='не меньше ±0,25', ok=left >= .25 and right <= -.25),
        dict(label='Увод без команды (0,5 и 1 м/с, старт с любой ноги)', value=f'{fmt(drift, 3)} рад/с',
             target='< 0,05', ok=drift < .05),
        dict(label='Здоровье: действия у предела, шум σ', value=f'{clipped:.0%}, σ {fmt(std or 0)}',
             target='< 30%, σ < 2', ok=clipped < .3 and std is not None and std < 2),
        dict(label='Шаг: подъём стопы и наклон в опоре',
             value=f'подъём {lift * 1000:.0f} мм, наклон {tilt:.0f}°' if stepping else '—',
             target='подъём ≥ 30 мм, наклон < 15°', ok=stepping and lift >= .03 and tilt < 15),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', nargs='?', help='experiment id, optionally :steps of a stage')
    parser.add_argument('--export', type=Path,
                        help='an export_g1 folder instead of a branch: its policy.onnx (what goes to the robot, '
                             'stand trim included) acts in the CPU environment of the stage it came from')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--journal', action='store_true',
                        help='record as the journal tab\'s current check (an export: signed by the export, under its '
                             'branch)')
    args = parser.parse_args()
    if (args.branch is None) == (args.export is None):
        parser.error('нужна ветка или --export <папка выгрузки>, что-то одно')
    lab = Laboratory(DEFAULT_RUN)
    exported = None
    if args.export is not None:
        from .export_g1 import ExportedPolicy
        exported = ExportedPolicy(args.export)
        stage = exported.stage
        branch = stage.parent.name
        if args.journal and stage.parent.parent.resolve() != lab.root.resolve():
            raise SystemExit(f'Этап выгрузки не из лаборатории ({stage}): в «Дневник» не записать')
    else:
        stage = find_stage(lab, args.branch)
        if stage is None:
            raise SystemExit('нет этапов')
        branch = args.branch.partition(':')[0]
    acting = None if exported is None else (exported, exported)
    report = check_stage(stage, seeds=args.seeds, acting=acting)
    walk = walks(stage, seeds=args.seeds, acting=acting)
    rows = items(report['scenarios'], report['policy_std'], walk)
    passed = sum(bool(r['ok']) for r in rows)
    for row in rows:
        print(f"{'✓' if row['ok'] else '✗'} {row['label']}: {row['value']} (цель {row['target']})")
    source = f'{branch} @ {report["stage"] / 1e6:.1f} млн'
    print(f'{source if exported is None else f"выгрузка {exported.folder} ({source})"} — выполнено {passed} из '
          f'{len(rows)}')
    if args.journal:
        from . import journal
        name = read_json(lab.root / branch / 'experiment.json', {}).get('name', branch)
        millions = f'{report["stage"] / 1e6:.1f}'.replace('.', ',')
        title = f'{name} · {millions} млн' + ('' if exported is None else f' · выгрузка {exported.folder.name}')
        journal.set_check(lab.root, branch, report['stage'], title, rows)
    print(json.dumps({'stage': report['stage'], 'export': None if exported is None else str(exported.folder),
                      'scenarios': report['scenarios'], 'policy_std': report['policy_std'],
                      'yaw_rates': {f'{s}:{y}': v for (s, y), v in walk['turn'].items()},
                      'drift_rates': {f'{s}:{start}': v for (s, start), v in walk['drift'].items()},
                      'pelvis_min_m': walk['pelvis'], 'feet': walk['feet'], 'items': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
