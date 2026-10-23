"""CPU only. speed_accept.py <branch>[:steps] [...] [--last N] [--workers W]: the stage-2 speed acceptance of iteration 22
(docs/lab-knowledge.md) on G1 stages, deterministic, seeds 73000-73004:
- walk 10 s from a standing start at a fixed command 1.5 and 2 m/s: mean forward speed after 3 s, falls, heading change;
- walk 6 s at 2 m/s, then stand 6 s: braking distance (stop command -> |v| < 0.1 m/s) and creep from 2 s after the stop;
- hip pitch at the motor's torque peak at 2 m/s (speed_terms.py, seed 73000).
Passes: speed >= 1.35 at 1.5 and >= 1.7 at 2 (worst seed), 0 falls, heading < 15 deg at 2, braking <= 1.0 m, creep < 0.1 m,
hip at peak <= 5%."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import math
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_terms as S  # noqa: E402

from humanoid_sim.core import list_stages  # noqa: E402

SEEDS = range(73000, 73005)


def stages(spec, last):
    branch, _, steps = spec.partition(':')
    folder = os.path.join(S.LAB, branch)
    if steps:
        return [os.path.join(folder, f'stage_{int(steps):012d}')]
    return [str(path) for path in list_stages(folder)[-last:]]


def episode(stage, command, seed, stop_at=None, steps=500):
    import torch
    torch.set_num_threads(1)
    import stand_turn as st
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config = st.stage_data(stage)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 100000})
    raw, data = env.unwrapped, env.unwrapped.data
    obs, _ = env.reset(seed=seed, options={'command': [command, 0., 0.], 'phase_steps': 0})
    start = st.heading_of(data.qpos)
    fell, speeds, brake, stop_pos, creep_from = None, [], None, None, None
    for t in range(steps):
        if t == stop_at:
            raw.set_command(np.zeros(3))
            obs = raw._observation('replace')
            stop_pos = data.qpos[:2].copy()
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if term:
            fell = (t + 1) * raw.dt
            break
        heading = st.heading_of(data.qpos)
        if t >= 150 and (stop_at is None or t < stop_at):
            speeds.append(data.qvel[0] * math.cos(heading) + data.qvel[1] * math.sin(heading))
        if stop_at is not None and t >= stop_at and brake is None and np.linalg.norm(data.qvel[:2]) < .1:
            brake = float(np.linalg.norm(data.qpos[:2] - stop_pos))
        if stop_at is not None and t == stop_at + 100:
            creep_from = data.qpos[:2].copy()
    turned = math.degrees(math.remainder(st.heading_of(data.qpos) - start, 2 * math.pi))
    creep = float(np.linalg.norm(data.qpos[:2] - creep_from)) if creep_from is not None and fell is None else None
    env.close()
    return dict(fell=fell, speed=float(np.mean(speeds)) if speeds else None, turned=turned, brake=brake, creep=creep)


def job(args):
    stage, kind, command, seed = args
    if kind == 'hip':
        S.POL['x'] = stage
        return args, {'hip': S.run(dict(pol='x', cmd=command, seed=seed))['groups']['hip_pitch']['sat']}
    return args, episode(stage, command, seed, stop_at=300 if kind == 'stop' else None,
                         steps=600 if kind == 'stop' else 500)


def verdict(rows):
    walk = {c: [r for (k, cmd, _), r in rows if k == 'walk' and cmd == c] for c in (1.5, 2.)}
    stop = [r for (k, _, _), r in rows if k == 'stop']
    hip = [r['hip'] for (k, _, _), r in rows if k == 'hip'][0]
    falls = sum(r['fell'] is not None for (k, _, _), r in rows if k != 'hip')
    v15 = min((r['speed'] or 0.) for r in walk[1.5])
    v2 = min((r['speed'] or 0.) for r in walk[2.])
    turn = max(abs(r['turned']) for r in walk[2.])
    brake = max((r['brake'] if r['brake'] is not None else math.inf) for r in stop)
    creep = max((r['creep'] if r['creep'] is not None else math.inf) for r in stop)
    checks = [('скорость при 1,5 (худший seed)', f'{v15:.2f} м/с', v15 >= 1.35, '≥ 1,35'),
              ('скорость при 2 (худший seed)', f'{v2:.2f} м/с', v2 >= 1.7, '≥ 1,7'),
              ('падения (ходьба с места и стоп, 15 прогонов)', f'{falls}', falls == 0, '0'),
              ('курс за 10 с при 2', f'{turn:.1f}°', turn < 15., '< 15°'),
              ('торможение 2 → стоп', f'{brake:.2f} м', brake <= 1., '≤ 1,0 м'),
              ('уползание через 2–6 с после стопа', f'{creep:.3f} м', creep < .1, '< 0,1 м'),
              ('бедро на пике момента при 2', f'{hip:.1%}', hip <= .05, '≤ 5%')]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--last', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    paths = [path for spec in args.stages for path in stages(spec, args.last)]
    jobs = [(path, kind, command, seed) for path in paths
            for kind, command, seeds in (('walk', 1.5, SEEDS), ('walk', 2., SEEDS), ('stop', 2., SEEDS),
                                         ('hip', 2., (73000,)))
            for seed in seeds]
    with Pool(args.workers) as pool:
        results = pool.map(job, jobs, chunksize=1)
    sys.stdout.reconfigure(encoding='utf-8')
    for path in paths:
        rows = [((kind, command, seed), r) for (p, kind, command, seed), r in results if p == path]
        checks = verdict(rows)
        name = f"{os.path.basename(os.path.dirname(path))} @ {int(os.path.basename(path)[6:]) / 1e6:.1f} млн"
        print(f"{name}: приёмка скорости — {sum(ok for _, _, ok, _ in checks)} из {len(checks)}")
        for label, value, ok, target in checks:
            print(f"  {'✓' if ok else '✗'} {label}: {value} (цель {target})")


if __name__ == '__main__':
    main()
