"""CPU only. getup_stance.py <stage | experiment[:steps]>... [--workers 4] [--per-class 4] [--starts <file.json>]
    [--out <file.json>]
How a g1_getup stage gets up and stands after it, several stages in one table (iteration 44; the critic's
lab-critic/it44g: rise44.py, noise44.py, dither44.py, arms44.py).

Starts: fresh draws of the start bank's generators (g1_getup_bank.make_state) with seeds the bank never used, as the
critic's ready38 / starts280.json: per class (back, front, side, sit, fall: candidate 8 pushed over) seeds
47 000 000 + class·100 000 + i, the first --per-class (4) that give a state; --starts takes such a file instead
(a list of {cls, seed, qpos, qvel}). Every stage runs on the same starts.
Episode: the stage's policy, deterministic (no noise), our physics (the files' friction, motor friction ×1, no delay),
pd_bound at the stage's lowest (none without the setting), no assist; "up" (pelvis ≥ 0.70 m, head top ≥ 1.20 m, tilt
≤ 15°, both soles down: g1_tensor_math.getup_success_pose) within 6 s, then 5 s more.
Measured per start:
- the rise, every control step until up (the state after the step, as rise44.py): the longest stretch without any
  robot-floor contact (a flight from 40 ms, GETUP_FLIGHT_SECONDS), the fastest foot touchdown (vertical speed of the
  ankle_roll origin over the step in which a foot's sole force first exceeds 20 N, m/s down), the largest sole force on
  one foot (normal force summed over its sole spheres, N; the robot weighs 344 N), and the sole_impact term's own
  measure (the largest sole approach speed over the rise's physics substeps and its unweighted Σ max(0, v − 0.5)²);
- the feet's width (left − right across the pelvis heading, m) at up + 0.1 / 1 / 2 / 4 s (noise44.py without noise);
- the hips' roll pull over up + 4..5 s: PD target − angle of each hip roll joint and kp times it (N·m), inwards
  positive (left −, right +; dither44.py);
- the arms on the body over up + 2..4 s: the share of steps whose summed contact force between an arm link (shoulder,
  elbow, wrist, hand) and the torso, pelvis or waist exceeds 10 N, its median over the steps, and the shoulder roll
  angles against home (arms44.py).
Printed: one row per measure, one column per stage (medians over the starts that got up; p90 and max where noted)
and the critic's acceptance threshold (lab-knowledge, «Вставание 823»). Stage 823 with its 20 starts made: 16 s on 4
processes. Stage 823 of branch 39 (exp-f19f12fb1eff4ef3) gives the critic's numbers: widths 0.601 / 0.454 / 0.451 /
0.447 m, hip roll −0.147 / +0.157 rad (−14.6 / +15.5 N·m), arms on the body 100 % of steps, flights ≥ 40 ms in 18 of
20, touchdown 2.33 m/s, sole force up to 904 N.
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

KINDS = ('back', 'front', 'side', 'sit', 'fall')
START_SEED = 47_000_000  # past the bank's seeds (g1_getup_bank.SEED_BASE 33e6 + class·1e6 + i), as ready38
UP_WITHIN, AFTER = 6., 5.  # s: up within, then measured for
MARKS = (.1, 1., 2., 4.)  # s after up: the feet's width
PULL_WINDOW, ARM_WINDOW = (4., 5.), (2., 4.)  # s after up
LOADED = 20.  # N on one foot's soles: that foot is down
ARM_FORCE = 10.  # N
HIP_ROLL = (1, 7)  # left, right (joint order)
SHOULDER_ROLL = (16, 23)
ARM_PARTS, CORE_PARTS = ('shoulder', 'elbow', 'wrist', 'hand'), ('torso', 'pelvis', 'waist')
# The critic's acceptance thresholds (lab-critic/it44g; docs/lab-knowledge.md «Вставание 823»).
THRESHOLDS = dict(width_med='≤ 0,34', width_p90='≤ 0,40', pull='≤ 6', arm_share='≤ 10 %', flights='≤ 2 из 20',
                  touchdown='≤ 1,0', force='≤ 700')


def resolve(spec):
    """A stage folder, or <experiment>[:steps] of the lab under runs/ (its latest stage without steps)."""
    path = Path(spec)
    if (path / 'metadata.json').is_file():
        return path.resolve()
    from humanoid_sim.core import DEFAULT_RUN, list_stages
    from humanoid_sim.lab import Laboratory
    name, _, steps = spec.partition(':')
    run = Laboratory(DEFAULT_RUN).resolve(name)
    stage = run / f'stage_{int(steps):012d}' if steps else list_stages(run)[-1]
    if not (stage / 'metadata.json').is_file():
        raise SystemExit(f'нет этапа {spec}')
    return stage


def make_start(job):
    import torch
    torch.set_num_threads(1)
    from humanoid_sim.tasks import g1_getup_bank as bank
    kind, seed = job
    state = bank.make_state((kind, seed, str(bank.FALL_STAGE)))
    if state is None:
        return None
    return dict(cls=kind, seed=seed, qpos=state[0].tolist(), qvel=state[1].tolist())


def fresh_starts(per_class, workers):
    """The first per_class states per class of the seeds START_SEED + class·100 000 + i (ready38's starts)."""
    chosen = []
    with Pool(workers) as pool:
        for k, kind in enumerate(KINDS):
            got, i = [], 0
            while len(got) < per_class:
                batch = [(kind, START_SEED + k * 100_000 + i + j) for j in range(per_class - len(got) + 2)]
                i += len(batch)
                got += [s for s in pool.map(make_start, batch) if s is not None]
            chosen += sorted(got, key=lambda s: s['seed'])[:per_class]
    return chosen


def heading(rotation):
    return math.atan2(rotation[1, 0], rotation[0, 0])


def width(raw):
    """Left − right foot (ankle_roll origins) across the pelvis heading, m (the critic's turn39.feet)."""
    d = raw.data
    h = heading(d.xmat[raw.pelvis].reshape(3, 3))
    dx, dy = d.xpos[raw.feet[0], :2] - d.xpos[raw.feet[1], :2]
    return float(-math.sin(h) * dx + math.cos(h) * dy)


def run(job):
    """One stage and start: the rise's and the stance's measures (None where the start did not get up)."""
    import torch
    torch.set_num_threads(1)
    import mujoco
    import getup_probe as gp
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    policy, norm, config = gp._policy(job['stage'])
    # sole_impact on (a reward weight: the episode is the same) so the task reads the soles' approach speeds.
    weights = {**config.get('reward_weights', {}), 'sole_impact': config.get('reward_weights', {}).get('sole_impact')
               or 1.}
    env = make_task('g1_getup', 'g1_29dof', {**config, 'reward_weights': weights, 'max_episode_steps': 100000})
    raw = env.unwrapped
    m, d = raw.model, raw.data
    try:
        env.reset(seed=job['seed'], options=dict(assist_force=0., action_delay_steps=0, motor_friction_scale=1.,
                                                 pd_bound='min', start=0))
        gp.apply_friction(raw, None)
        d.qpos[:], d.qvel[:] = job['qpos'], job['qvel']
        mujoco.mj_forward(m, d)
        obs = raw._observation('fill')
        floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        foot_of = {int(g): k for k in range(2) for g in raw.soles[k]}
        names = [m.body(b).name for b in range(m.nbody)]
        arm = {b for b, n in enumerate(names) if any(part in n for part in ARM_PARTS)}
        core = {b for b, n in enumerate(names) if any(part in n for part in CORE_PARTS)}
        force6 = np.zeros(6)
        out = dict(cls=job['cls'], seed=job['seed'], up=None, widths={})
        airborne = longest = 0
        peak = touchdown = sole_speed = sole_charge = 0.
        loaded = [False, False]
        previous_z = d.xpos[raw.feet, 2].copy()
        up = None
        errors, arm_forces, shoulders = [], [], []
        for i in range(round((UP_WITHIN + AFTER) / raw.dt)):
            t = (i + 1) * raw.dt
            action = raw.applied_action(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            obs, _, terminated, _, info = env.step(action)
            if terminated or info.get('bad_state'):
                out['ended'] = t
                break
            if up is None:  # the rise: the state after this step
                foot_force, touching = [0., 0.], False
                for c in range(d.ncon):
                    g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                    if g1 in floor or g2 in floor:
                        touching = True
                        g = g2 if g1 in floor else g1
                        if g in foot_of:
                            mujoco.mj_contactForce(m, d, c, force6)
                            foot_force[foot_of[g]] += abs(force6[0])
                z = d.xpos[raw.feet, 2].copy()
                down = -(z - previous_z) / raw.dt
                previous_z = z
                for k in range(2):
                    now = foot_force[k] > LOADED
                    if now and not loaded[k]:
                        touchdown = max(touchdown, float(down[k]))
                    loaded[k] = now
                airborne = 0 if touching else airborne + 1
                longest = max(longest, airborne)
                peak = max(peak, max(foot_force))
                speed = raw.sole_impact_substeps.max(0)
                sole_speed = max(sole_speed, float(speed.max()))
                sole_charge += float(task.impact_penalty(speed[None], task.GETUP_SOLE_IMPACT_SPEED)[0])
                sole = raw._sole_height().numpy()
                if bool(task.getup_success_pose(raw._head(), d.qpos[2:3], d.xmat[raw.torso].reshape(1, 3, 3),
                                                sole)[0]):
                    up = t
                    out.update(up=t, flight=longest * raw.dt, touchdown=touchdown, peak=peak, sole_speed=sole_speed,
                               sole_charge=sole_charge)
                elif t >= UP_WITHIN - 1e-9:
                    break
                continue
            after = t - up
            for mark in MARKS:
                if str(mark) not in out['widths'] and after >= mark - 1e-9:
                    out['widths'][str(mark)] = width(raw)
            if ARM_WINDOW[0] <= after < ARM_WINDOW[1]:
                total = 0.
                for c in range(d.ncon):
                    b1, b2 = m.geom_bodyid[d.contact.geom1[c]], m.geom_bodyid[d.contact.geom2[c]]
                    if (b1 in arm and b2 in core) or (b2 in arm and b1 in core):
                        mujoco.mj_contactForce(m, d, c, force6)
                        total += float(np.linalg.norm(force6[:3]))
                arm_forces.append(total)
                shoulders.append(d.qpos[raw.qadr][list(SHOULDER_ROLL)].copy())
            if PULL_WINDOW[0] <= after < PULL_WINDOW[1]:
                target = raw._target(np.clip(action, -raw.action_limit, raw.action_limit))
                errors.append(target - d.qpos[raw.qadr])
            if after >= AFTER - 1e-9:
                break
        if up is not None and errors and arm_forces:
            error = np.mean(errors, 0)[list(HIP_ROLL)]
            torque = error * raw.kp[list(HIP_ROLL)]
            shoulder = np.mean(shoulders, 0) - raw.home[list(SHOULDER_ROLL)]
            out.update(pull=float((torque[1] - torque[0]) / 2), roll_error=error.tolist(), roll_torque=torque.tolist(),
                       arm_share=float(np.mean(np.array(arm_forces) > ARM_FORCE)),
                       arm_force=float(np.median(arm_forces)), shoulder_roll=shoulder.tolist())
        return out
    finally:
        env.close()


def summary(results):
    """Medians (and the rest) over the starts that got up and stood the whole window."""
    from humanoid_sim.tasks import g1_tensor_math as task
    up = [r for r in results if r['up'] is not None]
    full = [r for r in up if 'pull' in r]

    def med(values):
        return float(np.median(values)) if len(values) else float('nan')
    widths = {mark: [r['widths'][str(mark)] for r in full if str(mark) in r['widths']] for mark in MARKS}
    return dict(
        n=len(results), up=len(up), stood=len(full), rise=med([r['up'] for r in up]),
        widths={mark: (med(v), float(np.percentile(v, 90)) if v else float('nan')) for mark, v in widths.items()},
        pull=med([r['pull'] for r in full]), roll_torque=[med([r['roll_torque'][k] for r in full]) for k in range(2)],
        roll_error=[med([r['roll_error'][k] for r in full]) for k in range(2)],
        arm_share=med([r['arm_share'] for r in full]), arm_force=med([r['arm_force'] for r in full]),
        shoulder_roll=[med([r['shoulder_roll'][k] for r in full]) for k in range(2)],
        flights=sum(r['flight'] >= task.GETUP_FLIGHT_SECONDS - 1e-9 for r in up), flight=med([r['flight'] for r in up]),
        flight_max=max((r['flight'] for r in up), default=float('nan')), touchdown=med([r['touchdown'] for r in up]),
        touchdown_max=max((r['touchdown'] for r in up), default=float('nan')),
        force=med([r['peak'] for r in up]), force_max=max((r['peak'] for r in up), default=float('nan')),
        sole_speed=med([r['sole_speed'] for r in up]), sole_charge=med([r['sole_charge'] for r in up]))


def label(stage):
    stage = Path(stage)
    return f'{stage.parent.name[4:12]} @ {int(stage.name[6:]) / 1e6:.0f}'


def table(stages, summaries):
    def f(value, digits=2):
        return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.{digits}f}'
    rows = [('встали / стояли 5 с', None, [f'{s["up"]}/{s["n"]} / {s["stood"]}' for s in summaries]),
            ('подъём, с (медиана)', None, [f(s['rise']) for s in summaries])]
    for mark in MARKS:
        rows.append((f'ширина стоп +{mark:g} с, м: медиана / p90',
                     f'{THRESHOLDS["width_med"]} / {THRESHOLDS["width_p90"]}' if mark in (1., 2.) else None,
                     [f'{f(s["widths"][mark][0], 3)} / {f(s["widths"][mark][1], 3)}' for s in summaries]))
    rows += [
        ('тяга крена бёдер внутрь, Н·м (+4..5 с)', THRESHOLDS['pull'], [f(s['pull'], 1) for s in summaries]),
        ('  крен бёдер: цель − угол Л / П, рад', None,
         [f'{s["roll_error"][0]:+.3f} / {s["roll_error"][1]:+.3f}' for s in summaries]),
        ('  крен бёдер: kp·(цель − угол) Л / П, Н·м', None,
         [f'{s["roll_torque"][0]:+.1f} / {s["roll_torque"][1]:+.1f}' for s in summaries]),
        ('рука–корпус > 10 Н, доля шагов (+2..4 с)', THRESHOLDS['arm_share'],
         [f'{s["arm_share"]:.0%}' if not math.isnan(s['arm_share']) else '—' for s in summaries]),
        ('  сила рука–корпус, Н (медиана)', None, [f(s['arm_force'], 0) for s in summaries]),
        ('  крен плеч от home Л / П, рад', '±0,08',
         [f'{s["shoulder_roll"][0]:+.3f} / {s["shoulder_roll"][1]:+.3f}' for s in summaries]),
        ('полёт ≥ 40 мс, подъёмов', THRESHOLDS['flights'], [f'{s["flights"]} из {s["up"]}' for s in summaries]),
        ('  самый долгий полёт, мс: медиана / max', None,
         [f'{f(s["flight"] * 1000, 0)} / {f(s["flight_max"] * 1000, 0)}' for s in summaries]),
        ('касание стопы, м/с вниз: медиана / max', THRESHOLDS['touchdown'],
         [f'{f(s["touchdown"])} / {f(s["touchdown_max"])}' for s in summaries]),
        ('пик силы на стопу, Н: медиана / max', THRESHOLDS['force'],
         [f'{f(s["force"], 0)} / {f(s["force_max"], 0)}' for s in summaries]),
        ('sole_impact: max сближения подошвы, м/с', None, [f(s['sole_speed']) for s in summaries]),
        ('sole_impact: Σ max(0, v − 0,5)² за подъём', None, [f(s['sole_charge'], 3) for s in summaries]),
    ]
    heads = [label(stage) for stage in stages]
    first = max(len(r[0]) for r in rows)
    widths = [max(len(h), *(len(r[2][k]) for r in rows)) for k, h in enumerate(heads)]
    print(f'{"мера":<{first}}  {"порог":>14}  ' + '  '.join(f'{h:>{w}}' for h, w in zip(heads, widths)))
    for name, threshold, cells in rows:
        print(f'{name:<{first}}  {threshold or "":>14}  ' + '  '.join(f'{c:>{w}}' for c, w in zip(cells, widths)))


def main():
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
    summaries = [summary(rs) for rs in per_stage]
    print(f'Стартов: {len(starts)} ({", ".join(f"{k} {sum(s["cls"] == k for s in starts)}" for k in KINDS)}); '
          'без шума, наша физика, граница ПД — нижняя по этапу, без подтяжки.')
    table(stages, summaries)
    if args.out:
        args.out.write_text(json.dumps(dict(stages=[str(s) for s in stages], starts=[
            dict(cls=s['cls'], seed=s['seed']) for s in starts], runs=per_stage), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
