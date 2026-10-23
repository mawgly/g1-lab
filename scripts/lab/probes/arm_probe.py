"""CPU only: arm_probe.py <stage|export folder>... [--amplitude 1.0 ...] [--seeds 6] [--workers 2] [--mask] [--out x.json].

Arms held by force, as training's arm_perturb segments (iteration 30; the critic's armprobe.py, lab-critic/it29c): from
t = 2 s two segments of 100 steps (set_arm_perturb at steps 100 and 200, 4 s in all) hold every arm joint at one pose,
U(±A rad) around home per joint (an arm_perturb_draw pose of amplitude A, task_config arm_perturb_radians; the action
box clips the wrist pitch and yaw, which lock_wrists holds at home anyway), one pose per draw (numpy default_rng(draw),
draws 0 .. seeds − 1), plus one run without a segment (A 0) as the reference. Our physics, deterministic policy, heading
hold off, reset seed 73000 at phase 0. Plans: walk 0.5 m/s straight 12 s, walk 1 m/s 12 s, stand 10 s.

Per run: the fall time (s), from 2 s on the yaw rate (rad/s) and heading change (°), the speed along x (m/s) and the
displacement (m; standing: the creep), and over the segments the mean self_contact, arm_press and total reward per step
(the stage's own weights; --mask also sets arm_perturb_contact_mask, a reward change only: the motion is the same).

Acceptance (iteration 30), per amplitude: no falls; standing, |heading change| ≤ 10° and creep ≤ 0.2 m in every draw;
walking 0.5 and 1 m/s, |yaw rate| ≤ 0.05 rad/s in every draw. Speeds are printed, not judged."""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PLANS = {'walk05': ((.5, 0., 0.), 12.), 'walk1': ((1., 0., 0.), 12.), 'stand': ((0., 0., 0.), 10.)}
START, SEGMENT, RESET_SEED = 100, 100, 73000  # control steps: segments from 2 s, each 2 s
TURN, CREEP, RATE = 10., .2, .05  # acceptance: standing heading change (°), creep (m); walking |yaw rate| (rad/s)
_cache = {}


def stage_data(stage):
    """(policy, normalizer, task config) of a stage, or of an export folder's policy.onnx in its stage's task."""
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        from humanoid_sim.export_g1 import ExportedPolicy, is_export
        if is_export(stage):  # policy and normalizer at once (normalization inside the graph)
            exported = ExportedPolicy(stage)
            _cache[stage] = (exported, exported, stage_task_config(exported.stage))
        else:
            model, norm = load_stage(Path(stage))
            _cache[stage] = (model, norm, stage_task_config(Path(stage)))
    return _cache[stage]


def heading(qpos):
    w, x, y, z = qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(job):
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = stage_data(job['stage'])
    config = {**config, 'heading_hold': False, 'max_episode_steps': 100000}
    if job['mask']:
        config['arm_perturb_contact_mask'] = True
    env = make_task('g1_velocity', 'g1_29dof', config)
    raw = env.unwrapped
    command, seconds = PLANS[job['plan']]
    obs, _ = env.reset(seed=RESET_SEED, options={'command': list(command), 'phase_steps': 0})
    held = np.zeros(raw.action_space.shape)
    arms = list(raw.arm_joints)
    held[arms] = np.random.default_rng(job['draw']).uniform(-1, 1, len(arms)) * job['A'] / np.asarray(raw.scale)[arms]
    yaw, xy, terms, fell = [], [], [], None
    steps = round(seconds / raw.dt)
    for k in range(steps):
        if job['A'] > 0 and k in (START, START + SEGMENT):
            raw.set_arm_perturb(SEGMENT, held)
        obs, _, terminated, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        yaw.append(heading(raw.data.qpos))
        xy.append(raw.data.qpos[:2].copy())
        if START <= k < START + 2 * SEGMENT:
            components = info['reward_components']
            terms.append((components.get('self_contact', 0.), components.get('arm_press', 0.),
                          sum(components.values())))
        if terminated:
            fell = round((k + 1) * raw.dt, 2)
            break
    env.close()
    out = dict(stage=job['stage'], plan=job['plan'], A=job['A'], draw=job['draw'], fell=fell,
               terms=np.mean(terms, axis=0).tolist() if terms else None)
    if fell is None:
        yaw = np.unwrap(yaw)
        span = (steps - 1 - START) * raw.dt
        out.update(rate=float((yaw[-1] - yaw[START]) / span), turn_deg=math.degrees(yaw[-1] - yaw[START]),
                   speed=float((xy[-1][0] - xy[START][0]) / span), disp=float(np.linalg.norm(xy[-1] - xy[START])))
    return out


def label(stage):
    return Path(stage).name[-12:]


def verdict(runs, stage, amplitude):
    """The acceptance of one stage at one amplitude, with the numbers it rests on."""
    group = [r for r in runs if r['stage'] == stage and r['A'] == amplitude]
    falls = sum(r['fell'] is not None for r in group)
    stand = [r for r in group if r['plan'] == 'stand' and r['fell'] is None]
    held = sum(abs(r['turn_deg']) <= TURN and r['disp'] <= CREEP for r in stand)
    drift = {plan: max((abs(r['rate']) for r in group if r['plan'] == plan and r['fell'] is None), default=None)
             for plan in ('walk05', 'walk1')}
    standing = sum(r['plan'] == 'stand' for r in group)
    passed = (falls == 0 and held == standing
              and all(value is not None and value <= RATE for value in drift.values()))
    return dict(stage=stage, A=amplitude, passed=passed, falls=falls, runs=len(group), stand_held=held,
                stand_runs=standing, turn_worst=max((r['turn_deg'] for r in stand), key=abs, default=None),
                creep_worst=max((r['disp'] for r in stand), default=None), drift_05=drift['walk05'],
                drift_1=drift['walk1'])


def fmt(value, spec):
    return 'n/a' if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--amplitude', type=float, nargs='+', default=[1.], help='A, rad (several: one table each)')
    parser.add_argument('--seeds', type=int, default=6, help='pose draws per amplitude')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--mask', action='store_true', help='with arm_perturb_contact_mask (the terms only)')
    parser.add_argument('--out')
    args = parser.parse_args()
    if any(not 0 < a <= 2 for a in args.amplitude) or args.seeds < 1 or args.workers < 1:
        raise SystemExit('amplitude in (0, 2] rad, seeds and workers at least 1')
    stages = [str(Path(stage).resolve()) for stage in args.stages]
    amplitudes = [0., *args.amplitude]
    jobs = [dict(stage=stage, plan=plan, A=a, draw=draw, mask=args.mask) for stage in stages for plan in PLANS
            for a in amplitudes for draw in range(1 if a == 0 else args.seeds)]
    started = time.time()
    if args.workers == 1:
        runs = [run(job) for job in jobs]
    else:
        with Pool(min(args.workers, len(jobs))) as pool:
            runs = pool.map(run, jobs, chunksize=1)
    print('runs', len(runs), '%.0f s' % (time.time() - started))
    for stage in stages:
        for plan in PLANS:
            for a in amplitudes:
                group = [r for r in runs if r['stage'] == stage and r['plan'] == plan and r['A'] == a]
                falls = sum(r['fell'] is not None for r in group)
                kept = [r for r in group if r['fell'] is None]
                if plan == 'stand':
                    values = ' '.join('%+.0f' % r['turn_deg'] for r in kept)
                    extra = ' creep ' + ' '.join('%.2f' % r['disp'] for r in kept)
                else:
                    values = ' '.join('%+.3f' % r['rate'] for r in kept)
                    extra = ' v ' + ' '.join('%.2f' % r['speed'] for r in kept)
                terms = [r['terms'] for r in kept if r['terms']]
                terms = (' | self_contact %.3f arm_press %.3f total %.2f' % tuple(np.mean(terms, axis=0))
                         if terms else '')
                print('%s %-6s A %.1f falls %d/%d | %s |%s%s' % (label(stage), plan, a, falls, len(group), values,
                                                                  extra, terms))
    verdicts = [verdict(runs, stage, a) for stage in stages for a in args.amplitude]
    for v in verdicts:
        print(f"VERDICT {label(v['stage'])} A {v['A']:.1f}: {'pass' if v['passed'] else 'FAIL'} | falls "
              f"{v['falls']}/{v['runs']} | standing heading <= {TURN:g} deg and creep <= {CREEP:g} m "
              f"{v['stand_held']}/{v['stand_runs']} (worst {fmt(v['turn_worst'], '+.0f')} deg, "
              f"{fmt(v['creep_worst'], '.2f')} m) | drift 0.5 m/s {fmt(v['drift_05'], '.3f')}, 1 m/s "
              f"{fmt(v['drift_1'], '.3f')} (<= {RATE:g})")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(stages=stages, amplitudes=args.amplitude, mask=args.mask, runs=runs,
                                                  verdicts=verdicts), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
