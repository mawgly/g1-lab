"""CPU only. getup_probe.py <stage | export_g1 folder> [--out <file.json>] [--workers 4] [--velocity <stage | folder>]
    [--physics ours,...] [--quick] [--pd-bound min|off|<c>]
The critic's acceptance of getting up (g1_getup, iteration 33), without the assist (force 0).
PD error bound (task_config pd_bound, iteration 34): by the stage, at the setting's lowest c (min, the default; none
without the setting, as evaluation and export); off: none; a number c: that bound, on any stage (the motor targets
within c·y1/kp of the observed angles, G1GetupEnv's reset option pd_bound).

Starts (56): the start bank's held-out rows lying on the back, the front and a side (10 each), 16 falls of candidate 8
(g1_getup_bank.falling with this probe's own seeds: pushed over standing or walking, then 2 s of Passive; the first 16
of FALL_SEED, FALL_SEED + 1, ... that fall) and the held-out sitting or kneeling rows (10). --quick: one of each class,
physics ours only.
Physics (each against ours, one change): ours (the files' friction, motor model as measured, no delay), motor friction
×0.25 and ×1.25, a 20 ms delay (10 physics substeps), foot friction 0.3 and 1.2 (the sole spheres; the rest as ours).
Success: within 6 s up (pelvis ≥ 0.70 m, head top ≥ 1.20 m, torso tilt ≤ 15°, both feet in contact:
g1_tensor_math.getup_success_pose), then 3 s standing (the pose without the feet) without the pelvis creeping more
than 0.3 m from where it got up. Printed per start class and physics: the success share, the median time to get up,
the share of time any motor is above 90 % of its peak (the motor model's y1, as um_probe), the hardest head or torso
hit on the floor (contact normal force, ≤ 300 N) and the torques of the first 0.2 s (≤ 60 % of peak).
Hand-over for every success, as on the robot: GetUp -> FixStand (um_probe's copy: from the last target to Unitree's
FixStand pose over 2 s with its gains, 0.5 ms physics steps, for --fixstand-seconds, 2 by default) -> Velocity of candidate 8 (--velocity; history from the
first frame, last action 0, clock 0): 5 s standing, then 5 s walking 0.5 m/s, in the same physics; a fall (the walking
task's: pelvis below 0.55 m or tilted past 60°) anywhere fails it; the pelvis turn over the 5 s standing is recorded.
For scripts/lab/skills_check.py (the critic's ready38 / turn39), a job may also carry max_joint_speed (g1_ctrl GetUp's
29 limits: the first speed cutoff, record['cut']) and snapshot (seconds after up: the hand-over from that moment, while
standing, instead of after the hold); without them the run is the one before.
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

CLASSES = ('back', 'front', 'side', 'fall', 'sit')
HELD_OUT = ('back', 'front', 'side', 'sit')  # from the bank's held-out rows; falls are made here
FALLS = 16
FALL_SEED = 39_000_000  # past the bank's seeds (g1_getup_bank.SEED_BASE + class·10^6 + i, classes 0-5)
# physics: (motor friction multiplier, delay in physics substeps, sole friction or None for the files' values)
PHYSICS = {'ours': (1., 0, None), 'motor0.25': (.25, 0, None), 'motor1.25': (1.25, 0, None), 'delay20': (1., 10, None),
           'foot0.3': (1., 0, .3), 'foot1.2': (1., 0, 1.2)}
UP_WITHIN, HOLD, CREEP = 6., 3., .3  # s, s, m
SATURATION, IMPACT, EARLY_SECONDS, EARLY_SHARE = .9, 300., .2, .6
STAND_SECONDS, WALK_SECONDS, WALK_SPEED = 5., 5., .5
# FixStand before Velocity (s; --fixstand-seconds): um_probe's 2 s, the whole interpolation. On the floor, without the
# band, that copy tips over at 1.34 s even from the walking task's reset pose (um_probe fix_ground; its fix_short is 0.5
# s), so a 2 s hand-over fails in FixStand whatever got the robot up: the operator has to go on to Velocity sooner.
FIXSTAND_SECONDS = 2.
_cache = {}
MONITOR = None  # the running episode's Monitor: every mj_step of this process reports to it


class Monitor:
    """Per physics substep of the getting-up phase: motor torques against their peaks, the largest normal force
    between the floor and a head or torso geom; with limits ((29,) rad/s, g1_ctrl GetUp's max_joint_speed), the first
    substep a joint passed its limit, as g1_ctrl's cutoff to Passive (the critic's ready38: cut = (s, joint, rad/s))."""

    def __init__(self, raw, limits=None):
        import mujoco
        from humanoid_sim.tasks.g1_velocity import motor_parameters
        names = [raw.model.joint(j).name for j in raw.joints]
        self.peak = motor_parameters(names)[2]  # y1: the motor's low-speed peak torque
        model = raw.model
        torso = model.body('torso_link').id
        self.floor = model.geom('floor').id
        self.hitting = {g for g in range(model.ngeom) if model.geom_bodyid[g] == torso
                        and (model.geom_contype[g] or model.geom_conaffinity[g])}
        self.force = np.zeros(6)
        self.free = ~raw.wrist_mask.numpy()  # the joints the policy drives (lock_wrists holds the others at home)
        self.substeps = self.saturated = self.saturated_free = 0
        self.over = np.zeros(raw.model.nu)
        self.early = 0.
        self.impact = 0.
        self.early_substeps = round(EARLY_SECONDS / model.opt.timestep)
        self.mujoco = mujoco
        self.limits = None if limits is None else np.asarray(limits, dtype=float)
        self.vadr = np.asarray(raw.vadr)
        self.cut = None

    def after(self, model, data):
        share = np.abs(data.ctrl) / self.peak
        over = share > SATURATION
        self.over += over
        self.saturated += bool(over.any())
        self.saturated_free += bool(over[self.free].any())
        if self.substeps < self.early_substeps:
            self.early = max(self.early, float(share.max()))
        self.substeps += 1
        if self.limits is not None and self.cut is None:
            speed = np.abs(data.qvel[self.vadr])
            if (speed > self.limits).any():
                joint = int(np.argmax(speed - self.limits))
                self.cut = (self.substeps * model.opt.timestep, joint, float(speed[joint]))
        for k in range(data.ncon):
            first, second = int(data.contact.geom1[k]), int(data.contact.geom2[k])
            if (first == self.floor and second in self.hitting) or (second == self.floor and first in self.hitting):
                self.mujoco.mj_contactForce(model, data, k, self.force)
                self.impact = max(self.impact, float(self.force[0]))


def _install_monitor():
    import mujoco
    if getattr(mujoco.mj_step, 'getup_probe', False):
        return
    original = mujoco.mj_step

    def step(model, data):
        original(model, data)
        if MONITOR is not None:
            MONITOR.after(model, data)
    step.getup_probe = True
    mujoco.mj_step = step


def _policy(path):
    if path not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import stage_task_config
        from humanoid_sim.export_g1 import load_policy
        policy, norm, stage = load_policy(Path(path))
        _cache[path] = (policy, norm, stage_task_config(stage))
    return _cache[path]


def make_fall(seed):
    """(qpos, qvel) of candidate 8 pushed over (g1_getup_bank.falling), or None when it did not fall."""
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    rng = np.random.default_rng(seed)
    raw, _, ok = bank_module.falling(bank_module.FALL_STAGE, rng, seed)
    return (raw.data.qpos.tolist(), raw.data.qvel.tolist()) if ok else None


def apply_friction(raw, sole):
    """The files' friction everywhere (G1GetupEnv.nominal_friction; a walking environment's model is the files'),
    then the sole spheres at `sole` (None: theirs)."""
    if hasattr(raw, 'nominal_friction'):
        raw.model.geom_friction[:, 0] = raw.nominal_friction
    if sole is not None:
        raw.model.geom_friction[raw.soles.ravel(), 0] = sole


def getup_phase(job):
    """The getting-up episode: up within UP_WITHIN s, then HOLD s; returns its record and the environment. Optional
    job keys: max_joint_speed ((29,) rad/s: the record's cut, Monitor), snapshot (s after up: the state and the PD
    target then, record['snapshot'], for a hand-over from that moment, as the critic's ready38 / turn39 at up + 1 s;
    taken unless the robot has fallen by the walking task's rule, also after the standing pose was lost, which still
    fails the success)."""
    global MONITOR
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = _policy(job['policy'])
    friction, delay, sole = PHYSICS[job['physics']]
    env = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100000})
    raw = env.unwrapped
    options = dict(assist_force=0., action_delay_steps=delay, motor_friction_scale=friction,
                   start=job['row'] if job['row'] is not None else 0, pd_bound=job['pd_bound'])
    obs, _ = env.reset(seed=job['seed'], options=options)
    if job['state'] is not None:  # a fall of candidate 8: its state in place of the bank row
        import mujoco
        raw.data.qpos[:], raw.data.qvel[:] = job['state']
        mujoco.mj_forward(raw.model, raw.data)
        obs = raw._observation('fill')
    apply_friction(raw, sole)
    MONITOR = monitor = Monitor(raw, job.get('max_joint_speed'))
    up_at = held = None
    target = None
    start_xy = None
    hold_steps = 0
    snapshot = None
    limit = round((UP_WITHIN + HOLD) / raw.dt)
    for step in range(1, limit + 1):
        action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        target = raw._target(action)
        obs, _, terminated, _, info = env.step(action)
        if info.get('bad_state'):
            held = False
            break
        head = task.head_height(raw.data.xpos[raw.torso][None], raw.data.xmat[raw.torso].reshape(1, 3, 3))
        rotation = raw.data.xmat[raw.torso].reshape(1, 3, 3)
        height = raw.data.qpos[2:3]
        if up_at is None:
            up = bool(task.getup_success_pose(head, height, rotation, raw._sole_height().numpy())[0])
            if up and step * raw.dt <= UP_WITHIN + 1e-9:
                up_at, start_xy = step * raw.dt, raw.data.qpos[:2].copy()
            elif step * raw.dt >= UP_WITHIN - 1e-9:
                held = False
                break
        else:
            standing = bool(task.getup_success_pose(head, height, rotation)[0])
            lost = not standing or np.linalg.norm(raw.data.qpos[:2] - start_xy) > CREEP
            if job.get('snapshot') is None:
                if lost:
                    held = False
                    break
            else:  # ready38: the snapshot whatever the pose, unless fallen by the walking task's rule
                if lost:
                    held = False
                pelvis = raw.data.xmat[raw.pelvis].reshape(3, 3)
                if raw.data.qpos[2] < task.FALL_HEIGHT or pelvis[2, 2] < task.FALL_UPRIGHT:
                    break
                if snapshot is None and step * raw.dt - up_at >= job['snapshot'] - 1e-9:
                    snapshot = dict(qpos=raw.data.qpos.tolist(), qvel=raw.data.qvel.tolist(), target=target.tolist())
                if held is False and snapshot is not None:
                    break
            hold_steps += 1
            if hold_steps >= round(HOLD / raw.dt):
                held = True if held is None else held
                break
        if terminated:  # a joint past 50 rad/s
            held = False
            break
    MONITOR = None
    record = dict(job={k: v for k, v in job.items() if k not in ('state', 'policy', 'velocity', 'max_joint_speed')},
                  success=bool(up_at is not None and held), up_s=up_at,
                  saturated=monitor.saturated / max(monitor.substeps, 1),
                  saturated_free=monitor.saturated_free / max(monitor.substeps, 1),
                  saturated_joint=(monitor.over / max(monitor.substeps, 1)).tolist(),
                  impact_n=monitor.impact, early_share=monitor.early, seconds=monitor.substeps * raw.model.opt.timestep,
                  pd_bound=raw.bound)
    if job.get('max_joint_speed') is not None:
        record['cut'] = None if monitor.cut is None else list(monitor.cut)
    if job.get('snapshot') is not None:
        record['snapshot'] = snapshot
    return record, env, target


def fixstand(raw, q_start, friction, seconds):
    """um_probe's FixStand copy with the motor friction multiplier of the physics: from q_start to Unitree's pose over
    FIX_T s with its gains, 0.5 ms steps, for `seconds`; the walking task's fall rule, checked every 20 ms."""
    import mujoco
    import torch
    from um_probe import FIX_KD, FIX_KP, FIX_QS, FIX_T
    from humanoid_sim.tasks import g1_tensor_math as task
    h_policy, h = raw.model.opt.timestep, .0005
    raw.model.opt.timestep = h
    scale = torch.tensor([friction], dtype=torch.float64)
    fell = None
    try:
        for i in range(round(seconds / h)):
            a = min(i * h / FIX_T, 1.)
            q, v = raw.data.qpos[raw.qadr], raw.data.qvel[raw.vadr]
            torque = FIX_KP * (q_start + (FIX_QS - q_start) * a - q) - FIX_KD * v
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(v.copy()), *raw._motor_tensors,
                                       friction_scale=scale).numpy()
            raw.data.ctrl[:] = np.clip(torque, raw.model.actuator_ctrlrange[:, 0], raw.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(raw.model, raw.data)
            if i % 40 == 39:
                mujoco.mj_forward(raw.model, raw.data)
                rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
                if raw.data.qpos[2] < task.FALL_HEIGHT or rotation[2, 2] < task.FALL_UPRIGHT:
                    fell = (i + 1) * h
                    break
    finally:
        raw.model.opt.timestep = h_policy
    mujoco.mj_forward(raw.model, raw.data)
    return fell


def hand_over(job, getup_env, target, state=None):
    """GetUp -> FixStand -> Velocity of the walking stage, in the same physics, from the getting-up end state (state:
    (qpos, qvel) instead, a snapshot). Also the pelvis turn over the standing seconds (deg, as the critic's turn39:
    the heading's changes summed; turn_deg, None after a fall)."""
    import math
    import mujoco
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = _policy(job['velocity'])
    friction, delay, sole = PHYSICS[job['physics']]
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'hand_collision': True,
                                                 'max_episode_steps': 100000})
    raw = env.unwrapped

    def heading():
        rotation = raw.data.xmat[raw.pelvis].reshape(3, 3)
        return math.atan2(rotation[1, 0], rotation[0, 0])
    try:
        env.reset(seed=job['seed'], options={'command': [0., 0., 0.], 'phase_steps': 0, 'action_delay_steps': delay,
                                             'motor_friction_scale': friction})
        apply_friction(raw, sole)
        source = getup_env.unwrapped.data
        raw.data.qpos[:], raw.data.qvel[:] = (source.qpos, source.qvel) if state is None else state
        mujoco.mj_forward(raw.model, raw.data)
        fell = fixstand(raw, target, friction, job['fixstand'])
        if fell is not None:
            return dict(fell='fixstand', fell_at=fell)
        # g1_ctrl's Velocity enter: deploy gains, history from the first frame, last action 0, clock restarted.
        raw.steps, raw.phase_steps = 0, 0
        raw.previous_action[:] = 0.
        raw.set_command(np.zeros(3))
        obs = raw._observation('fill')
        start = raw.data.qpos[:2].copy()
        moved = turn = None
        previous, total = heading(), 0.
        for step in range(round((STAND_SECONDS + WALK_SECONDS) / raw.dt)):
            if step == round(STAND_SECONDS / raw.dt):
                moved, turn = float(np.linalg.norm(raw.data.qpos[:2] - start)), math.degrees(total)
                raw.set_command(np.array([WALK_SPEED, 0., 0.]))
                obs = raw._observation('replace')
                start = raw.data.qpos[:2].copy()
            obs, _, fell, _, _ = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if moved is None:
                now = heading()
                total += math.remainder(now - previous, 2 * math.pi)
                previous = now
            if fell:
                return dict(fell='stand' if moved is None else 'walk', fell_at=(step + 1) * raw.dt, turn_deg=turn)
        walked = float(np.linalg.norm(raw.data.qpos[:2] - start)) / WALK_SECONDS
        return dict(fell=None, stand_creep_m=moved, walk_speed=walked, turn_deg=turn)
    finally:
        env.close()


def run(job):
    """One start: the getting-up record and, after a success, the hand-over; with job['snapshot'] the hand-over starts
    from the snapshot's state and target whenever there is one (up and standing until then)."""
    import torch
    torch.set_num_threads(1)
    _install_monitor()
    record, env, target = getup_phase(job)
    try:
        snapshot = record.pop('snapshot', None) if job.get('snapshot') is not None else None
        if job.get('snapshot') is not None:
            record['hand_over'] = None if snapshot is None else hand_over(
                job, env, np.array(snapshot['target']), state=(snapshot['qpos'], snapshot['qvel']))
        else:
            record['hand_over'] = hand_over(job, env, target) if record['success'] else None
    finally:
        env.close()
    return record


def starts(quick):
    """(class, bank row or None, fall state or None) of every start."""
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    bank = bank_module.load()
    chosen = []
    for kind in HELD_OUT:
        rows = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)))
        chosen += [(kind, int(row), None) for row in rows[:1 if quick else len(rows)]]
    wanted = 1 if quick else FALLS
    falls, seed = [], FALL_SEED
    with Pool(WORKERS) as pool:
        while len(falls) < wanted:
            batch = list(range(seed, seed + max(WORKERS, wanted - len(falls) + 2)))
            seed = batch[-1] + 1
            falls += [state for state in pool.map(make_fall, batch) if state is not None]
    chosen += [('fall', None, state) for state in falls[:wanted]]
    return chosen


def median(values):
    return float(np.median(values)) if values else float('nan')


def report(results, physics):
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env, joint_names
    names = [name.replace('_joint', '') for name in joint_names(getup_model_env())]
    print('Доля вставших (за 6 с, затем 3 с стоя, уползание до 0,3 м), без подтяжки:')
    print('класс     ' + ''.join(f'{p:>11}' for p in physics))
    for kind in (*CLASSES, 'все'):
        cells = []
        for p in physics:
            runs = [r for r in results if r['job']['physics'] == p and (kind == 'все' or r['job']['class'] == kind)]
            cells.append(f'{sum(r["success"] for r in runs):>4}/{len(runs):<3} ' if runs else f'{"—":>11}')
        print(f'{kind:<10}' + ''.join(f'{c:>11}' for c in cells))
    for p in physics:
        runs = [r for r in results if r['job']['physics'] == p]
        up = [r['up_s'] for r in runs if r['success']]
        saturated = [r['saturated'] for r in runs]
        free = [r['saturated_free'] for r in runs]
        joints = np.max([r['saturated_joint'] for r in runs], axis=0)
        worst = int(np.argmax(joints))
        hits = [r['impact_n'] for r in runs]
        early = [r['early_share'] for r in runs]
        print(f'{p}: медиана вставания {median(up):.2f} с; моменты > 90 % пика — {np.mean(saturated):.1%} времени, '
              f'без закреплённых запястий {np.mean(free):.1%} (больше всего у {names[worst]}: {joints[worst]:.1%}); '
              f'удар головой/торсом о пол '
              f'max {max(hits):.0f} Н, > {IMPACT:.0f} Н в {sum(h > IMPACT for h in hits)} из {len(runs)}; моменты '
              f'первых 0,2 с max {max(early):.0%} пика, > {EARLY_SHARE:.0%} в {sum(e > EARLY_SHARE for e in early)}')
        handed = [r['hand_over'] for r in runs if r['hand_over'] is not None]
        if handed:
            falls = [h for h in handed if h['fell']]
            places = ', '.join(f'{place} {sum(h["fell"] == place for h in falls)}' for place in ('fixstand', 'stand', 'walk'))
            speeds = [h['walk_speed'] for h in handed if not h['fell']]
            creeps = [h['stand_creep_m'] for h in handed if not h['fell']]
            print(f'   передача FixStand -> Velocity: падений {len(falls)} из {len(handed)} ({places}); '
                  f'стоя 5 с сместился (медиана) на {median(creeps):.2f} м, ходьба 0,5 м/с — медиана '
                  f'{median(speeds):.2f} м/с')


def main():
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy', help='этап g1_getup или папка export_g1')
    parser.add_argument('--out', type=Path, help='записи прогонов, JSON')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--velocity', default=None, help='этап или выгрузка ходьбы для передачи (по умолчанию №8)')
    parser.add_argument('--physics', default=','.join(PHYSICS), help='через запятую: ' + ', '.join(PHYSICS))
    parser.add_argument('--fixstand-seconds', type=float, default=FIXSTAND_SECONDS,
                        help='FixStand перед Velocity, с (2 — вся интерполяция; на полу без ленты копия падает на 1,34 с)')
    parser.add_argument('--quick', action='store_true', help='по одному старту каждого класса, физика ours')
    parser.add_argument('--pd-bound', default='min',
                        help='граница ошибки ПД: min — нижняя по настройке этапа (без настройки — нет), off — без '
                             'границы, число c — доля пика момента на любом этапе')
    args = parser.parse_args()
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    WORKERS = args.workers
    physics = ['ours'] if args.quick else args.physics.split(',')
    if set(physics) - set(PHYSICS):
        parser.error(f'--physics: {sorted(set(physics) - set(PHYSICS))} не из {", ".join(PHYSICS)}')
    try:
        bound = {'min': 'min', 'off': None}[args.pd_bound] if args.pd_bound in ('min', 'off') else float(args.pd_bound)
    except ValueError:
        parser.error(f'--pd-bound: min, off или число, а не {args.pd_bound}')
    velocity = str(Path(args.velocity or bank_module.FALL_STAGE).resolve())
    policy = str(Path(args.policy).resolve())
    jobs = [dict(policy=policy, velocity=velocity, physics=p, **{'class': kind}, row=row, state=state, seed=73000 + k,
                 fixstand=args.fixstand_seconds, pd_bound=bound)
            for p in physics for k, (kind, row, state) in enumerate(starts(args.quick))]
    with Pool(WORKERS) as pool:
        results = pool.map(run, jobs, chunksize=1)
    used = {r['pd_bound'] for r in results}
    print('Граница ошибки ПД: ' + (', '.join(f'c = {c:g}' for c in sorted(used - {None})) if used - {None} else 'нет'))
    report(results, physics)
    if args.out:
        args.out.write_text(json.dumps(dict(policy=policy, velocity=velocity, physics=physics, runs=results), indent=1),
                            encoding='utf-8')


WORKERS = 4
if __name__ == '__main__':
    main()
