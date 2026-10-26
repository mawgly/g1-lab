"""The drift rows of a walking policy, CPU (the critic's lab-critic/it45r/drift49.py and sym_cf.py, iteration 50):
drift_rows.py <stage | experiment[:steps] | export dir>... [--mirror] [--workers 4] [--json <file>]

Each run: deterministic, a reset with the command (heading hold off, free arms, seed 73000; the walks also from the
gait phase 15), 10 s, measured after 2 s: the yaw rate and the speed along the heading. Commands: turning on the spot
±0.5 rad/s, turning at 1 m/s ±0.5, side steps ±0.3 and ±0.5 m/s, walking 0.5 and 1 m/s (phases 0 and 15), backwards
0.5 m/s. An export_g1 folder runs its policy.onnx (export_g1.ExportedPolicy: what g1_ctrl runs, --symmetric-walk
included) in its stage's task; --mirror also runs each stage symmetrized on the fly (½(π(o) + M_a·π(M_o·o)) always).
Printed per policy, with the acceptance of the symmetrized walk (iteration 50, 47c @ 9886): turning on the spot ≥ 0.58
of the command both ways and its mean |bias| ≤ 0.01 rad/s; side steps ±0.3 |yaw rate| ≤ 0.04, ±0.5 ≤ 0.05; walking
drift ≤ 0.01; turning at 1 m/s ≥ 0.30 rad/s both ways; walking 0.5 m/s ≥ 0.85 of the command; no falls."""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_falls import resolve  # noqa: E402

SCENARIOS = (('place', (0., 0., .5), 0), ('place', (0., 0., -.5), 0), ('turn1', (1., 0., .5), 0),
             ('turn1', (1., 0., -.5), 0), ('side3', (0., .3, 0.), 0), ('side3', (0., -.3, 0.), 0),
             ('side5', (0., .5, 0.), 0), ('side5', (0., -.5, 0.), 0), ('w05', (.5, 0., 0.), 0),
             ('w05', (.5, 0., 0.), 15), ('w1', (1., 0., 0.), 0), ('w1', (1., 0., 0.), 15), ('back', (-.5, 0., 0.), 0))
STEPS, SETTLE, SEED = 500, 100, 73000
# The acceptance of the symmetrized walk (the lead, iteration 50).
PLACE, BIAS, SIDE3, SIDE5, DRIFT, TURN1, SPEED05 = .58, .01, .04, .05, .01, .30, .85
_cache = {}


def setup(path):
    """This worker's policy, normalization, environment (heading hold off) and mirror, once per policy folder."""
    if _cache.get('path') != path:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import stage_task_config
        from humanoid_sim.export_g1 import load_policy
        from humanoid_sim.symmetry import Mirror
        from humanoid_sim.tasks.catalog import make_task
        model, norm, stage = load_policy(Path(path))
        env = make_task('g1_velocity', 'g1_29dof', {**stage_task_config(stage), 'heading_hold': False,
                                                   'max_episode_steps': STEPS})
        _cache.update(path=path, env=env, model=model, norm=norm, mirror=Mirror(env))
    return _cache


def run(job):
    """One command: the yaw rate, the speed along the heading after SETTLE, a fall."""
    import torch
    c = setup(job['path'])
    env, model, norm, mirror = c['env'], c['model'], c['norm'], c['mirror']
    raw = env.unwrapped
    pelvis = raw.model.body('pelvis').id
    heading = lambda: math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0])
    obs, _ = env.reset(seed=SEED, options={'command': list(job['cmd']), 'phase_steps': job['phase']})
    track, xy, fell = [heading()], [raw.data.qpos[:2].copy()], False
    for _ in range(STEPS):
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        if job['mirror']:
            flipped = mirror.observation(torch.from_numpy(np.asarray(obs, dtype=np.float32))).numpy()
            other = model.predict(norm.normalize_obs(flipped), deterministic=True)[0]
            action = .5 * (action + mirror.action(torch.from_numpy(np.asarray(other, dtype=np.float32))).numpy())
        obs, _, fell, truncated, _ = env.step(action)
        track.append(heading())
        xy.append(raw.data.qpos[:2].copy())
        if fell or truncated:
            break
    track = np.unwrap(track)
    span = (len(track) - 1 - SETTLE) * raw.dt
    moved = xy[-1] - xy[SETTLE]
    h = track[SETTLE]
    return dict(job, rate=float((track[-1] - track[SETTLE]) / span) if span > 0 else math.nan,
                speed=float(moved[0] * math.cos(h) + moved[1] * math.sin(h)) / span if span > 0 else math.nan,
                fell=bool(fell))


def summary(runs):
    """The rows of one policy (and how it was run) from its runs, with the acceptance."""
    def rate(scen, cmd):
        return [r['rate'] for r in runs if r['scen'] == scen and list(r['cmd']) == list(cmd)]

    def speed(scen):
        return float(np.mean([r['speed'] for r in runs if r['scen'] == scen]))
    left, right = rate('place', (0., 0., .5))[0], rate('place', (0., 0., -.5))[0]
    out = dict(place=(left / .5, -right / .5), bias=(left + right) / 2,
               turn1=(rate('turn1', (1., 0., .5))[0], -rate('turn1', (1., 0., -.5))[0]),
               side3=(rate('side3', (0., .3, 0.))[0], rate('side3', (0., -.3, 0.))[0]),
               side5=(rate('side5', (0., .5, 0.))[0], rate('side5', (0., -.5, 0.))[0]),
               drift=rate('w05', (.5, 0., 0.)) + rate('w1', (1., 0., 0.)),
               v05=speed('w05') / .5, v1=speed('w1'), back=-speed('back') / .5, falls=sum(r['fell'] for r in runs))
    out['checks'] = {
        'place': min(out['place']) >= PLACE, 'bias': abs(out['bias']) <= BIAS,
        'side3': max(map(abs, out['side3'])) <= SIDE3, 'side5': max(map(abs, out['side5'])) <= SIDE5,
        'drift': max(map(abs, out['drift'])) <= DRIFT, 'turn1': min(out['turn1']) >= TURN1,
        'v05': out['v05'] >= SPEED05, 'falls': out['falls'] == 0}
    out['passes'] = all(out['checks'].values())
    return out


def label(path, mirror):
    path = Path(path)
    name = path.name if (path / 'policy.onnx').is_file() else f'{path.parent.name[4:12]}@{int(path.name[6:]) // 10**6}'
    return name + (' (зеркало на лету)' if mirror else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policies', nargs='+', help='этап, ветка[:шаги] или папка выгрузки')
    parser.add_argument('--mirror', action='store_true', help='ещё и симметризованные на лету этапы')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    paths = [str(resolve(spec).resolve()) for spec in args.policies]
    jobs = [dict(path=path, mirror=mirror, scen=scen, cmd=cmd, phase=phase) for path in paths
            for mirror in ((False, True) if args.mirror and not (Path(path) / 'policy.onnx').is_file() else (False,))
            for scen, cmd, phase in SCENARIOS]
    started = time.time()
    with Pool(args.workers) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    print(f'{len(runs)} прогонов, {time.time() - started:.0f} с')
    print(f'{"политика":<42} | на месте +,- | смещ. | 1 м/с +,-   | вбок 0,3 Л,П  | вбок 0,5 Л,П  | увод   | '
          f'0,5  1    назад | пад. | итог')
    rows = []
    for path in paths:
        for mirror in sorted({job['mirror'] for job in jobs if job['path'] == path}):
            s = summary([r for r in runs if r['path'] == path and r['mirror'] == mirror])
            rows.append(dict(path=path, mirror=mirror, **s))
            failed = ', '.join(k for k, ok in s['checks'].items() if not ok)
            print(f'{label(path, mirror):<42} | {s["place"][0]:.2f},{s["place"][1]:.2f}    | {s["bias"]:+.3f} | '
                  f'{s["turn1"][0]:+.2f},{s["turn1"][1]:+.2f} | {s["side3"][0]:+.3f},{s["side3"][1]:+.3f} | '
                  f'{s["side5"][0]:+.3f},{s["side5"][1]:+.3f} | {max(map(abs, s["drift"])):.3f}  | '
                  f'{s["v05"]:.2f} {s["v1"]:.2f} {s["back"]:.2f}  | {s["falls"]:4d} | '
                  + ('проходит' if s['passes'] else 'мимо: ' + failed))
    print(f'приёмка: на месте ≥ {PLACE:g} обе стороны, |смещение| ≤ {BIAS:g}; вбок ±0,3 ≤ {SIDE3:g}, ±0,5 ≤ {SIDE5:g}; '
          f'увод ≤ {DRIFT:g}; 1 м/с поворот ≥ {TURN1:g} рад/с; 0,5 м/с ≥ {SPEED05:g}; без падений')
    if args.json:
        args.json.write_text(json.dumps(dict(runs=runs, rows=rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
