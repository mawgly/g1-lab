"""CPU only. lean_source.py <stage> <out.json> [workers]
Where the torso lean comes from: torso_link world pitch (as supervise) = pelvis world pitch + waist part;
pelvis pitch vs leg pitch joints (hip_pitch + knee + ankle_pitch) and foot pitch. Stand 10 s det / noise,
walk 1 m/s det, stop_start det. Reward terms torso / orientation / posture / stand_pose per step."""
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
_cache = {}


def _stage(stage):
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        model, norm = load_stage(Path(stage))
        _cache[stage] = (model, norm, stage_task_config(Path(stage)))
    return _cache[stage]


def pitch(raw, body):
    up = raw.data.xmat[body].reshape(3, 3)[:, 2]
    return math.degrees(math.atan2(up[0], up[2]))


def run(job):
    stage, kind, seed, det = job
    import torch
    model, norm, config = _stage(stage)
    from humanoid_sim.tasks.catalog import make_task
    torch.manual_seed(seed)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'max_episode_steps': 600})
    raw = env.unwrapped
    b = {n: raw.model.body(n).id for n in ('pelvis', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link')}
    cmd = lambda t: [0., 0., 0.] if kind == 'stand' or (kind == 'stop_start' and t < 5.) else [1., 0., 0.]
    obs, _ = env.reset(seed=seed, options={'command': cmd(0.)})
    rows, comps = [], {}
    for step in range(500):
        t = step * raw.dt
        raw.set_command(cmd(t))
        obs = raw._observation()
        action = model.predict(norm.normalize_obs(obs), deterministic=det)[0]
        obs, _, term, trunc, info = env.step(action)
        q = raw.data.qpos[raw.qadr] - raw.home
        rows.append(dict(t=(step + 1) * raw.dt, torso=pitch(raw, b['torso_link']), pelvis=pitch(raw, b['pelvis']),
                         footL=pitch(raw, b['left_ankle_roll_link']), footR=pitch(raw, b['right_ankle_roll_link']),
                         wy=math.degrees(q[12]), wr=math.degrees(q[13]), wp=math.degrees(q[14]),
                         legL=math.degrees(q[0] + q[3] + q[4]), legR=math.degrees(q[6] + q[9] + q[10]),
                         hipL=math.degrees(q[0]), hipR=math.degrees(q[6]), kneeL=math.degrees(q[3]),
                         kneeR=math.degrees(q[9]), ankL=math.degrees(q[4]), ankR=math.degrees(q[10]),
                         shL=math.degrees(q[15]), shR=math.degrees(q[22]), elL=math.degrees(q[18]),
                         elR=math.degrees(q[25]), z=float(raw.data.qpos[2])))
        for k in ('torso', 'orientation', 'posture', 'stand_pose', 'height', 'still'):
            comps.setdefault(k, []).append(float(info['reward_components'][k]))
        comps.setdefault('total', []).append(float(sum(info['reward_components'].values())))
        if term:
            break
    env.close()
    return dict(kind=kind, seed=seed, det=det, rows=rows, comps={k: float(np.mean(v)) for k, v in comps.items()},
                fell=bool(term))


def jobs(stage):
    j = []
    for s in range(73000, 73003):
        j += [(stage, 'stand', s, True), (stage, 'stand', s, False), (stage, 'walk', s, True),
              (stage, 'stop_start', s, True)]
    return j


KEYS = ('torso', 'pelvis', 'wp', 'wr', 'wy', 'legL', 'legR', 'footL', 'footR', 'hipL', 'hipR', 'kneeL', 'kneeR',
        'ankL', 'ankR', 'shL', 'shR', 'elL', 'elR', 'z')

if __name__ == '__main__':
    stage, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    with Pool(workers) as pool:
        results = pool.map(run, jobs(stage), chunksize=1)
    Path(out).write_text(json.dumps([{k: v for k, v in r.items() if k != 'rows'} | {
        'mean': {key: float(np.mean([x[key] for x in r['rows']])) for key in KEYS}} for r in results], indent=1))
    print(Path(stage).name)
    for kind, det in (('stand', True), ('stand', False), ('walk', True), ('stop_start', True)):
        rs = [r for r in results if r['kind'] == kind and r['det'] == det]
        for window in ((0., 2.), (2., 10.), (0., 10.)) if kind != 'stop_start' else ((0., 5.), (5., 10.), (0., 10.)):
            m = {key: np.mean([np.mean([x[key] for x in r['rows'] if window[0] < x['t'] <= window[1]]) for r in rs])
                 for key in KEYS}
            print(f"{kind:10s} {'det' if det else 'sig'} {window[0]:>4.0f}-{window[1]:<4.0f}s torso {m['torso']:+5.1f} "
                  f"pelvis {m['pelvis']:+5.1f} waist p/r/y {m['wp']:+5.1f}/{m['wr']:+4.1f}/{m['wy']:+4.1f} "
                  f"legsum L/R {m['legL']:+5.1f}/{m['legR']:+5.1f} foot L/R {m['footL']:+5.1f}/{m['footR']:+5.1f} "
                  f"hip {m['hipL']:+5.1f}/{m['hipR']:+5.1f} knee {m['kneeL']:+5.1f}/{m['kneeR']:+5.1f} "
                  f"ank {m['ankL']:+5.1f}/{m['ankR']:+5.1f} sh {m['shL']:+5.1f}/{m['shR']:+5.1f} "
                  f"el {m['elL']:+5.1f}/{m['elR']:+5.1f} z {m['z']:.3f}")
        c = {k: np.mean([r['comps'][k] for r in rs]) for k in rs[0]['comps']}
        print('   per seed torso:', ' '.join(f"{np.mean([x['torso'] for x in r['rows']]):+.1f}" for r in rs),
              '| reward/step', ' '.join(f'{k} {v:+.4f}' for k, v in c.items()), '| fell', sum(r['fell'] for r in rs))
