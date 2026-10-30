"""CPU only. getup_handover_bank.py <first stage> --target <clip> [--frame 18] [--classes back,side]
[--physics ours,delay20,foot0.3,motor1.25] [--runs 4000] [--first-seconds 3] [--pd S [key=value …]] [--workers 8]
[--seed 83000] [--out <npz>] [--force]
The hand-over bank of g1_getup's task_config track_handover_starts (iteration 83, the lead's decision on
scripts/lab/probes/getup_chain.py): the chain «getting-up policy 1164 → PD 1 s → tracker 73» stands up from the back
10/9/10/10 of 10, but the PD lands the robot on its hands at 1-2.4 kN, while handed over directly (0 of 80) the tracker
does not take a robot in motion (9-14 rad/s, 0.9 rad from any clip frame). So the tracker trains from those very
states: this script records them.

The first policy (a plain g1_getup stage, pd_bound at its lowest) runs from the start bank's training rows of --classes
(never the held-out ones getup_chain judges on; --runs over the classes and the --physics of getup_probe.PHYSICS in
turn, the rows drawn without repeats per class), deterministic, with the CUDA task's sensor noise on every observed
frame and an encoder bias per run (chatter.noisy, as getup_chain), no assist, until getup_chain.HandOver fires on the
robot's own sensors against frame --frame of the target clip (the splice's join, A493 v3 frame 18 by default). Then the
simulated state (qpos, qvel) and the first policy's last applied actions (newest first, g1_getup_handover.
HANDOVER_ACTIONS of them: the actions in flight under a delay when the tracker takes over) are kept; nothing else is
needed for an honest start, since the tracker enters as g1_ctrl enters a policy state: its last action 0 and its
history filled with its first frame (the task's reset does both). Runs without a hand-over within --first-seconds, or
ended by a non-finite state or a joint above GETUP_MAX_JOINT_SPEED, are counted and dropped.

The tracker's clip clock starts at --frame (the header's frame), not at the frame nearest the pose: the rule steers the
hand-over towards that frame, the PD variant of getup_chain (10 of 10 from the back) starts the tracker there, and the
nearest frame of a robot in motion jumps (it is kept per row, `nearest`, for reference).

--pd S [ramp=… arms=… legs=… kp=… kd=… kp_to=… kd_to=…] (iteration 86; getup_chain.py --handover pd's form, parsed by
getup_chain.variants and run by getup_chain.pd_transition, the chain's own code): the robot gets the tracker after
g1_ctrl's GetUpBlend, not straight from the first policy (the chain 1164 → pd 3.5 → tracker 730 got 0 of 10 up from the
stomach: at the transition's end the chest lies on the floor, the right arm 0.6 rad from the clip's pose, a state the
tracker never trained from). So after the hand-over the same run goes on with the PD transition to frame --frame (from
the first policy's last target; pd_bound off, the gains scaled by its shape while it runs, as in the chain), and the
bank keeps the state at its end: qpos, qvel, and as actions the transition's last applied actions, newest first. That is
what is in flight honestly: the transition's action each step is its quintic target over the action scale
((goal − home)/scale through applied_action, so _target gives back the very PD target GetUpBlend wrote), and the task's
delay queue holds applied actions, so under an action delay the tracker's first substeps follow the blend's last
targets, as the motors do on the robot after GetUpBlend's last lowcmd (the chain's pd variant hands the same queue to
B). The rest is as g1_ctrl enters a policy state, and the task's track_handover reset does it: last action 0, history
filled with the first frame; the clock at --frame, which the transition ends on. The gains at the switch are the
policy's (with kp, kd ≠ 1 and no kp_to / kd_to they jump, as in the chain and in GetUpBlend's last step). Runs that end
during the transition (a non-finite state, a joint above GETUP_MAX_JOINT_SPEED) are dropped and counted. The header's pd
names the transition (seconds, the full shape); per row PD_ARRAYS of g1_getup_handover (the pose the tracker gets);
the default name ends in -pd<S>. Without --pd the bank is the one before, bit for bit.

Human range (iteration 86, Pavel 29.10): a bank made with --pd or --human-rom has per row human_rom, whether its source
row of the start bank has every joint within HUMAN_ROM (judged on that row, before the run; about half the back and
front rows lie outside, legs folded behind the back), so probes can split the starts by it; the header's human_rom
gives the limits and the counts per class. --human-rom (off by default; Pavel kept the other poses: a robot may fall
that way) also takes only those rows (name suffix -hrom).

The file (g1_getup_handover: arrays only, no pickle) goes to runs/humanoid_lab/getup_bank/<--out or
handover-<steps>-<clip md5[:8]>-f<frame>>.npz; its md5 is printed with the setting to branch with. Per class and physics
the hand-over share, its time, the rules that fired, the pose then, and the share of rows that the tracking
terminations of the anchor (its height and gravity against the reference at the frame) would end on their first step.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / 'scripts' / 'lab' / 'probes'))
import getup_chain  # noqa: E402
import getup_probe  # noqa: E402

PHYSICS = getup_chain.PHYSICS
CLASSES = getup_chain.CLASSES
RULES = ('least', 'good', 'left', 'timeout', 'torso')
SEED = 83_000
RUNS = 4000
# --human-rom (iteration 86, Pavel 29.10: a robot lying with its legs folded behind its back, "impossible for a human",
# rises from there with a jerk, dangerous on the hardware): a human's range of motion, degrees in G1's signs, per joint
# kind (both legs): (low, high). hip_roll is (inward, outward) by magnitude: outward is + on the left leg, − on the right.
HUMAN_ROM = dict(hip_pitch=(-125., 30.),  # − = flexion
                 hip_roll=(35., 50.),  # inward ≤ 35, outward ≤ 50
                 hip_yaw=(-50., 50.), knee=(-5., 145.), ankle_pitch=(-55., 30.), waist_pitch=(-30., 80.))
_cache = {}


def human_rom(qpos, names):
    """(n,) bool: rows of qpos (n, 36) whose joints (names: the 29 joint names in qpos[7:]'s order) lie within
    HUMAN_ROM; joints of other kinds are not judged."""
    joints = np.degrees(np.asarray(qpos)[:, 7:])
    inside = np.ones(len(joints), dtype=bool)
    for k, name in enumerate(names):
        kind = next((key for key in HUMAN_ROM if name.endswith(f'{key}_joint')), None)
        if kind is None:
            continue
        low, high = HUMAN_ROM[kind]
        if kind == 'hip_roll':  # outward + on the left leg, − on the right
            outward = 1. if name.startswith('left') else -1.
            low, high = (-low, high) if outward > 0 else (-high, low)
        inside &= (joints[:, k] >= low) & (joints[:, k] <= high)
    return inside


def human_rom_rows(bank):
    """(n,) bool over the start bank's rows: human_rom of each row's qpos, by the getting-up model's joint names."""
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    env = bank_module.getup_model_env()
    try:
        return human_rom(bank['qpos'], bank_module.joint_names(env))
    finally:
        env.close()


def _target(name, frame):
    """The target clip's goal at `frame` and what a row needs (once per process): (gravity (3,), joints (29,), clip
    gravity, joints and anchor heights of its frames, the free joints, the anchor's height and torso z-axis z at the
    frame)."""
    if (name, frame) not in _cache:
        from humanoid_sim.tasks import g1_getup_track
        from humanoid_sim.tasks import g1_tensor_math as task
        from humanoid_sim.tasks.g1_getup import G1GetupEnv
        clip = g1_getup_track.load(name)
        env = G1GetupEnv(track_clip={'file': name, 'md5': clip['md5']}, track_bank=clip)
        if not 0 <= frame <= env.clip_last:
            raise SystemExit(f'--frame: кадр клипа от 0 до {env.clip_last}')
        anchor = env.track_anchor
        _cache[(name, frame)] = dict(
            md5=clip['md5'], gravity=env.clip_gravity[frame].copy(), joints=env.clip_joints[frame].copy(),
            clip_gravity=env.clip_gravity, clip_joints=env.clip_joints, clip_anchor=env.clip_anchor_height,
            free=env.nearest_joints, anchor_z=float(env.clip_frames['pos'][frame, anchor, 2]),
            anchor_up=float(env.clip_frames['rot'][frame, anchor, 2, 2]), last=env.clip_last,
            torso=task.projected_gravity(env.clip_frames['rot'][frame, anchor][None])[0])
    return _cache[(name, frame)]


def run(job):
    """One start of the first policy to its hand-over (job pd (seconds, shape items): then the PD transition): the
    row's record (None fields without a hand-over)."""
    import torch
    torch.set_num_threads(1)
    import chatter
    from humanoid_sim.g1_tensor_env import ENCODER_BIAS
    from humanoid_sim.tasks import g1_getup_handover
    from humanoid_sim.tasks import g1_tensor_math as task
    goal = _target(job['target'], job['frame'])
    env, policy, norm, exact = getup_chain._setup(job['first'], False)
    raw = env.unwrapped
    friction, delay, sole = getup_probe.PHYSICS[job['physics']]
    rng = np.random.default_rng((job['seed'], chatter.NOISE_STREAM))
    bias = (2 * rng.random(raw.model.nu) - 1) * ENCODER_BIAS
    raw._observation = getup_chain.sensed(raw, exact, rng, bias)
    obs, _ = env.reset(seed=job['seed'], options=dict(assist_force=0., start=job['row'], action_delay_steps=delay,
                                                      motor_friction_scale=friction, pd_bound='min'))
    getup_probe.apply_friction(raw, sole)
    rule = getup_chain.HandOver(raw.dt, torso=job['rule'] == 'torso')
    applied = [np.zeros(raw.model.nu)] * g1_getup_handover.HANDOVER_ACTIONS  # newest first
    record = dict(row=job['row'], start=job['start'], physics=job['physics'], seed=job['seed'], fired=None,
                  ended=None)
    for step in range(1, int(round(job['first_seconds'] / raw.dt)) + 1):
        action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
        target = raw._target(action) if job.get('pd') is not None else None  # --pd: where the transition starts
        obs, _, terminated, _, info = env.step(action)
        applied = [action.copy(), *applied[:-1]]
        if info.get('bad_state') or terminated:
            record['ended'] = 'bad_state' if info.get('bad_state') else 'speed'
            return record
        gravity, joints = getup_chain.reading(raw)
        angle, deviation = getup_chain.errors(raw, gravity, joints, goal['gravity'], goal['joints'])
        gap = getup_chain.torso_gap(raw, gravity, joints, goal['torso'])
        fired = rule(step, angle, deviation, gap)
        if fired:
            nearest = int(task.track_nearest_frame(gravity[None], joints[None], goal['clip_anchor'][:1],
                                                   goal['clip_gravity'], goal['clip_joints'], goal['clip_anchor'],
                                                   goal['free'])[0])
            torso = raw.data.xmat[raw.torso].reshape(3, 3)
            free = ~raw.wrist_mask.numpy()
            record.update(fired=fired, seconds=step * raw.dt, angle=angle, deviation=deviation, nearest=nearest,
                          torso_gap=gap,
                          joint_speed=float(np.abs(raw.data.qvel[raw.vadr])[free].max()),
                          qpos=raw.data.qpos.copy(), qvel=raw.data.qvel.copy(), actions=np.array(applied),
                          anchor_gap=float(raw.data.xpos[raw.torso, 2]) - goal['anchor_z'],
                          up_gap=float(torso[2, 2]) - goal['anchor_up'])
            if job.get('pd') is not None:
                return _transition(env, raw, job['pd'], goal, target, applied, record)
            return record
    return record


def _transition(env, raw, pd, goal, start, applied, record):
    """--pd: the PD transition (getup_chain.pd_transition) from the first policy's last target `start` to the frame,
    then the record's state, actions and PD_ARRAYS at its end (pd_ended 'bad_state' or 'speed' when it ended there;
    the hand-over's own fields stay)."""
    seconds, shape = pd
    out = getup_chain.pd_transition(env, start, goal['joints'], seconds, dict(shape), applied)
    if out['fell'] is not None:
        record['pd_ended'] = out['fell']
        return record
    gravity, joints = getup_chain.true_pose(raw)
    angle, deviation = getup_chain.errors(raw, gravity, joints, goal['gravity'], goal['joints'])
    torso = raw.data.xmat[raw.torso].reshape(3, 3)
    free = ~raw.wrist_mask.numpy()
    record.update(qpos=raw.data.qpos.copy(), qvel=raw.data.qvel.copy(), actions=np.array(out['applied']),
                  pd_angle=angle, pd_deviation=deviation,
                  pd_joint_speed=float(np.abs(raw.data.qvel[raw.vadr])[free].max()),
                  pd_torso=float(raw.data.xpos[raw.torso, 2]),
                  pd_arm_worst=float(np.abs(joints - goal['joints'])[list(raw.press_joints)].max()),
                  pd_anchor_gap=float(raw.data.xpos[raw.torso, 2]) - goal['anchor_z'],
                  pd_up_gap=float(torso[2, 2]) - goal['anchor_up'])
    return record


def jobs_of(first, target, frame, classes, physics, runs, first_seconds, seed, rule='pelvis', pd=None,
            human=False):
    """The runs: per class runs // len(classes) training rows of the start bank drawn without repeats (all of them
    when there are fewer), the physics in turn, a seed each; pd (seconds, shape items) or None: --pd's transition;
    human: only rows within HUMAN_ROM (judged on the row itself, before any run)."""
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    bank = bank_module.load()
    rng = np.random.default_rng(seed)
    jobs = []
    within = human_rom_rows(bank) if human else np.ones(len(bank['qpos']), dtype=bool)
    for kind in classes:
        rows = np.flatnonzero(~bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)) & within)
        rows = rng.permutation(rows)[:runs // len(classes)]
        for k, row in enumerate(rows):
            jobs.append(dict(first=first, target=target, frame=frame, start=kind, row=int(row),
                             physics=physics[k % len(physics)], seed=seed + len(jobs), first_seconds=first_seconds,
                             rule=rule, pd=pd))
    return jobs, bank


def summary(records, classes, physics):
    print('класс | физика | передач / прогонов | время передачи с (медиана, p90) | правила | угол° / откл. рад / сустав '
          'рад/с (медианы) | разрыв торса набл. / истинный, якоря м (медианы) | конец сразу по корпусу')
    for kind in classes:
        for p in physics:
            mine = [r for r in records if r['start'] == kind and r['physics'] == p]
            handed = [r for r in mine if r['fired']]
            if not mine:
                continue
            rules = {name: sum(r['fired'] == name for r in handed) for name in RULES}
            ended = sum(abs(r['anchor_gap']) > .25 or abs(r['up_gap']) > .8 for r in handed)

            def median(key, scale=1.):
                return np.median([r[key] * scale for r in handed]) if handed else float('nan')
            seconds = [r['seconds'] for r in handed]
            print(f'{kind} | {p} | {len(handed)} / {len(mine)} | '
                  f'{median("seconds"):.2f}, {np.percentile(seconds, 90) if seconds else float("nan"):.2f} | '
                  + ', '.join(f'{name} {n}' for name, n in rules.items() if n)
                  + f' | {median("angle", 180 / np.pi):.0f} / {median("deviation"):.2f} / {median("joint_speed"):.1f} '
                  f'| {median("torso_gap"):.2f} / {np.median([abs(r["up_gap"]) for r in handed]) if handed else float("nan"):.2f}, '
                  f'{np.median([abs(r["anchor_gap"]) for r in handed]) if handed else float("nan"):.3f} | {ended}')
    dropped = [r for r in records if not r['fired']]
    print(f'без передачи: {len(dropped)} (оборвались {sum(r["ended"] is not None for r in dropped)})')
    if not any('pd_ended' in r or 'pd_angle' in r for r in records):
        return
    print('после ПД-перехода, на входе в трекер: класс | физика | строк / передач | угол таза° / откл. суставов рад / '
          'высота торса м / худший сустав руки рад / сустав рад/с (медианы) | конец сразу по корпусу')
    for kind in classes:
        for p in physics:
            handed = [r for r in records if r['start'] == kind and r['physics'] == p and r['fired']]
            kept = [r for r in handed if 'pd_angle' in r]
            if not handed:
                continue

            def median(key, scale=1.):
                return np.median([r[key] * scale for r in kept]) if kept else float('nan')
            ended = sum(abs(r['pd_anchor_gap']) > .25 or abs(r['pd_up_gap']) > .8 for r in kept)
            print(f'{kind} | {p} | {len(kept)} / {len(handed)} | {median("pd_angle", 180 / np.pi):.0f} / '
                  f'{median("pd_deviation"):.2f} / {median("pd_torso"):.3f} / {median("pd_arm_worst"):.2f} / '
                  f'{median("pd_joint_speed"):.1f} | {ended}')
    print(f'оборвались на переходе: {sum("pd_ended" in r for r in records)}')


def main():
    from humanoid_sim.tasks import g1_getup_handover
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument('first', help='этап политики вставания (обычный g1_getup)')
    parser.add_argument('--target', required=True, help='клип трекера (имя в runs/humanoid_lab/getup_track)')
    parser.add_argument('--frame', type=int, default=getup_chain.TARGET_FRAME,
                        help=f'кадр клипа: цель правила передачи и начало часов трекера (по умолчанию '
                             f'{getup_chain.TARGET_FRAME})')
    parser.add_argument('--classes', default=','.join(CLASSES), help='классы обучающих строк банка через запятую')
    parser.add_argument('--physics', default=','.join(PHYSICS), help='физики через запятую: '
                                                                    + ', '.join(getup_probe.PHYSICS))
    parser.add_argument('--runs', type=int, default=RUNS, help=f'прогонов всего (по умолчанию {RUNS})')
    parser.add_argument('--first-seconds', type=float, default=getup_chain.FIRST_SECONDS,
                        help=f'с политики до передачи самое большее (по умолчанию {getup_chain.FIRST_SECONDS})')
    parser.add_argument('--rule', default='pelvis', choices=getup_chain.RULES,
                        help='правило передачи getup_chain: pelvis (прежнее) или torso (и торс опущен, как в кадре)')
    parser.add_argument('--pd', nargs='+', metavar='S',
                        help='после передачи ПД-переход к кадру --frame за S с, форма как у getup_chain.py --handover '
                             'pd S [ramp=… arms=… legs=… kp=… kd=… kp_to=… kd_to=…]; в банк — состояние в его конце')
    parser.add_argument('--human-rom', action='store_true',
                        help='только строки банка с суставами в пределах движения человека (HUMAN_ROM, по самой '
                             'строке до прогона)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--out', type=Path, help='файл банка (по умолчанию runs/humanoid_lab/getup_bank/'
                                                 'handover-<шаги>-<md5 клипа>-f<кадр>.npz)')
    parser.add_argument('--force', action='store_true', help='перезаписать существующий файл')
    args = parser.parse_args()
    physics = [p for p in args.physics.split(',') if p]
    if not physics or set(physics) - set(getup_probe.PHYSICS):
        raise SystemExit(f'физики: {", ".join(getup_probe.PHYSICS)}')
    classes = [k for k in args.classes.split(',') if k]
    if args.runs < len(classes):
        raise SystemExit('--runs: не меньше числа классов')
    first = str((SIM / args.first).resolve() if not Path(args.first).is_absolute() else Path(args.first))
    goal = _target(args.target, args.frame)
    steps = int(json.loads((Path(first) / 'metadata.json').read_text(encoding='utf-8'))['steps'])
    pd = None
    if args.pd:
        (_, pd_seconds, pd_shape), = getup_chain.variants([['pd', *args.pd]])
        pd = (pd_seconds, pd_shape)
    suffix = '' if args.rule == 'pelvis' else f'-{args.rule}'
    if pd is not None:
        suffix += '-pd' + f'{pd[0]:g}'.replace('.', 'p')
    if args.human_rom:
        suffix += '-hrom'
    out = args.out or g1_getup_handover.bank_path(f'handover-{steps}-{goal["md5"][:8]}-f{args.frame}{suffix}')
    if out.exists() and not args.force:
        raise SystemExit(f'{out} уже есть (--force, чтобы перезаписать)')
    jobs, bank = jobs_of(first, args.target, args.frame, classes, physics, args.runs, args.first_seconds, args.seed,
                         args.rule, pd, args.human_rom)
    if set(classes) - set(bank['classes']):
        raise SystemExit(f'классы банка: {", ".join(bank["classes"])}')
    human = None
    if pd is not None or args.human_rom:  # rows within a human's range, per class: training rows within / all
        within = human_rom_rows(bank)
        human = {}
        for kind in classes:
            rows = ~bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind))
            human[kind] = [int((rows & within).sum()), int(rows.sum())]
        print('в пределах человека (обучающие строки банка' + (', только они' if args.human_rom else '') + '): '
              + ', '.join(f'{kind} {n} из {total}' for kind, (n, total) in human.items()))
    began = time.time()
    with Pool(min(args.workers, len(jobs))) as pool:
        records = pool.map(run, jobs, chunksize=4)
    seconds = time.time() - began
    handed = [r for r in records if r['fired'] and 'pd_ended' not in r]
    if not handed:
        raise SystemExit('ни одной передачи: банк не записан')
    arrays = dict(qpos=np.array([r['qpos'] for r in handed]), qvel=np.array([r['qvel'] for r in handed]),
                  actions=np.array([r['actions'] for r in handed]),
                  row=np.array([r['row'] for r in handed], dtype=np.int64),
                  start_class=np.array([bank['classes'].index(r['start']) for r in handed], dtype=np.int64),
                  physics=np.array([list(getup_probe.PHYSICS).index(r['physics']) for r in handed], dtype=np.int64),
                  seed=np.array([r['seed'] for r in handed], dtype=np.int64),
                  seconds=np.array([r['seconds'] for r in handed]),
                  rule=np.array([RULES.index(r['fired']) for r in handed], dtype=np.int64),
                  angle=np.array([r['angle'] for r in handed]), deviation=np.array([r['deviation'] for r in handed]),
                  joint_speed=np.array([r['joint_speed'] for r in handed]),
                  nearest=np.array([r['nearest'] for r in handed], dtype=np.int64))
    if pd is not None:
        arrays.update({key: np.array([r[key] for r in handed]) for key in g1_getup_handover.PD_ARRAYS})
    if human is not None:  # per row: its source row within a human's range (probes split by it)
        arrays['human_rom'] = within[arrays['row']]
    header = dict(first=os.path.relpath(first, SIM).replace('\\', '/'), first_steps=steps,
                  clip=dict(file=args.target, md5=goal['md5']), frame=args.frame, classes=list(bank['classes']),
                  physics=list(getup_probe.PHYSICS), rules=list(RULES), rule=args.rule,
                  settings=dict(first_seconds=args.first_seconds, seed=args.seed, runs=len(jobs),
                                classes=classes, physics=physics, sensors='chatter.noisy + encoder bias',
                                rule='getup_chain.HandOver', deterministic=True),
                  counts=dict(runs=len(jobs), handed=len(handed),
                              ended=sum(r['ended'] is not None for r in records)),
                  seconds=round(seconds, 1), created_at=time.time())
    if pd is not None:  # the PD transition after the hand-over (getup_chain.pd_transition), its full shape
        header['pd'] = dict(seconds=pd[0], shape={**getup_chain.PD_DEFAULTS, **dict(pd[1])},
                            code='getup_chain.pd_transition')
        header['counts']['pd_ended'] = sum('pd_ended' in r for r in records)
    if human is not None:  # the limits, whether rows were filtered by them, the training rows within per class
        header['human_rom'] = dict(limits_degrees={key: list(value) for key, value in HUMAN_ROM.items()},
                                   hip_roll='(inward, outward); outward + left, − right', filtered=args.human_rom,
                                   rows=human, banked=int(arrays['human_rom'].sum()))
    md5 =g1_getup_handover.save(out, arrays, header)
    summary(records, classes, physics)
    print(f'банк передачи: {out} — {len(handed)} строк из {len(jobs)} прогонов за {seconds:.0f} с; md5 {md5}')
    setting = {'share': .3, 'file': out.stem, 'md5': md5}
    print('настройка: track_handover_starts ' + json.dumps(setting))


if __name__ == '__main__':
    main()
