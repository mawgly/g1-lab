"""CPU only. push_stand.py <stage dir | export dir> <out.json> [workers] [accept|sweep] [raw|sym]

Velocity kicks while standing, falls per kick (iteration 16; port of the critic's sym_kick_sweep.py / sym_push.py).
A run: reset standing (command 0, gait phase 0, heading hold off), at step KICK_AT (2 s) the kick (world vx, vy in
m/s and yaw rate in rad/s) is added to the free joint's velocity, as the training pushes do, then STEPS in all (6 s);
a fall is the environment's termination. The policy acts through env.step (box clip, lock_wrists), so this is the
environment exactly. An export folder's policy.onnx acts through humanoid_sim.export_g1.ExportedPolicy (what g1_ctrl
runs, stand trim included) in the environment of the stage it came from; sym: ½(π(o) + M_a·π(M_o·o)) of the given
policy (stand_turn._Sym).
Groups: accept — forward and backward 0.9 / 1.0 / 1.1 m/s, 4 seeds (24 runs; iteration 16's acceptance: raw
candidate 3 falls 0 of 24, the symmetrised one 7 of 8 forward at 0.9-1.0); sweep — the critic's: ±x and ±y at
0.8 … 1.5 m/s, 4 seeds (112 runs). Prints falls per kick and the mean heading change.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stand_turn as st  # noqa: E402

SEEDS = (73000, 73001, 73002, 73003)
KICK_AT, STEPS = 100, 300


def kicks(group):
    if group == 'accept':
        return [(s * v, 0., 0.) for v in (.9, 1., 1.1) for s in (1, -1)]
    if group == 'sweep':
        return [k for v in (.8, .9, 1., 1.1, 1.2, 1.3, 1.5) for k in ((v, 0., 0.), (-v, 0., 0.), (0., v, 0.),
                                                                      (0., -v, 0.))]
    raise ValueError(group)


def run(job):
    import torch
    from humanoid_sim.tasks.catalog import make_task
    torch.set_num_threads(1)
    model, norm, config = st.stage_data(job['stage'])
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 100000})
    raw = env.unwrapped
    if job.get('sym'):
        from humanoid_sim.symmetry import Mirror
        model = norm = st._Sym(model, norm, Mirror(env))
    env.reset(seed=job['seed'], options={'command': [0., 0., 0.], 'phase_steps': 0})
    raw.set_command(np.zeros(3))
    obs = raw._observation('replace')
    yaw0 = st.heading_of(raw.data.qpos)
    kick, fell = np.asarray(job['kick'], dtype=float), None
    for t in range(STEPS):
        if t == KICK_AT:
            raw.data.qvel[:2] += kick[:2]
            raw.data.qvel[5] += kick[2]
        obs, _, terminated, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if terminated:
            fell = (t + 1) * raw.dt
            break
    yaw = st.heading_of(raw.data.qpos) - yaw0
    env.close()
    return dict(job={k: v for k, v in job.items() if k != 'stage'}, fell=fell,
                yaw_deg=math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))),
                xy=[float(raw.data.qpos[0]), float(raw.data.qpos[1])])


def main():
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    group = sys.argv[4] if len(sys.argv) > 4 else 'accept'
    sym = len(sys.argv) > 5 and sys.argv[5] == 'sym'
    jobs = [dict(stage=stage, seed=seed, kick=list(k), sym=sym) for k in kicks(group) for seed in SEEDS]
    if workers <= 1:
        results = [run(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(run, jobs, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    table = {}
    for r in results:
        table.setdefault(tuple(r['job']['kick']), []).append(r)
    print(f"{stage}{' (sym)' if sym else ''}")
    for kick, rows in table.items():
        falls = sum(r['fell'] is not None for r in rows)
        heading = np.mean([r['yaw_deg'] for r in rows if r['fell'] is None]) if falls < len(rows) else float('nan')
        print('kick vx %+.1f vy %+.1f wz %+.1f: falls %d/%d, heading of the standing ones %+.1f deg'
              % (*kick, falls, len(rows), heading))
    print(len(results), 'runs, falls', sum(r['fell'] is not None for r in results))


if __name__ == '__main__':
    main()
