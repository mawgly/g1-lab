"""CPU only. bridge_kick.py <out.json> <workers> <policy...>
Copy of the bridge (stand_turn.run: unitree_mujoco physics 'um', PD on a 2 ms older state, 6 ms delay, FixStand on the
band, band off at 1.4 s) plus the push patch of unitree_mujoco: horizontal force M*dv/0.1 s on the pelvis for 50
physics ticks, direction in the pelvis heading at the push. Standing plan; push PUSH_S s after band off + offset.
Deterministic per case, so the push tick offset varies instead of the seed. Policy: alias=path, unitree_v0;
suffix '@L' = FixStand left wrist pitch 0.7 (candidate 2's bridge runs, --fixstand-wrist left).
Env: KICK_MAGS, KICK_DIRS, KICK_PHYS (um / ours), KICK_PLAN (stand / walk: 0.5 m/s walk, push 3 s into it).
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIM = r'C:\Users\Admin\Documents\Codex\Sim'
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, 'scripts', 'lab', 'probes'))
import stand_turn as st  # noqa: E402
import um_probe as um  # noqa: E402

MAGS = tuple(float(x) for x in os.environ.get('KICK_MAGS', '.5,.75,1').split(','))
DIRS = {d: v for d, v in {'f': (1., 0.), 'b': (-1., 0.), 'l': (0., 1.), 'r': (0., -1.)}.items()
        if d in os.environ.get('KICK_DIRS', 'fblr')}
PHYS = os.environ.get('KICK_PHYS', 'um').split(',')
PLAN = os.environ.get('KICK_PLAN', 'stand')
OFFSETS = (0, 37, 71, 113)
st.PLANS['kstand'] = [('stand', 0., 0., 0., 8.)]
st.PLANS['kwalk'] = [('stand', 0., 0., 0., 3.), ('walk', .5, 0., 0., 8.)]
PUSH = {}


class PushStepper(um.Stepper):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.banded, self.after, self.d, self.vmax, self.zmin = False, 0, None, 0., 9.
        import mujoco
        self.mass = float(mujoco.mj_getTotalmass(self.raw.model))

    def substep(self, target):
        raw = self.raw
        if self.band is not None:
            self.banded = True
        counting = (self.band is None) and (self.banded or PUSH['start'] == 'reset')
        t0 = PUSH['tick']
        if counting:
            if self.after == t0:
                h = st.heading_of(raw.data.qpos)
                lx, ly = PUSH['dir']
                self.d = np.array([lx * math.cos(h) - ly * math.sin(h), lx * math.sin(h) + ly * math.cos(h)])
            if self.d is not None and t0 <= self.after < t0 + 50:
                raw.data.xfrc_applied[raw.pelvis, :2] = self.mass * PUSH['mag'] / .1 * self.d
            else:
                raw.data.xfrc_applied[raw.pelvis, :] = 0.
        super().substep(target)
        if counting:
            if self.d is not None and self.after < t0 + 500:
                self.vmax = max(self.vmax, float(raw.data.qvel[:2] @ self.d))
                self.zmin = min(self.zmin, float(raw.data.qpos[2]))
            self.after += 1
        PUSH['stepper'] = self


st.um.Stepper = PushStepper


def run(job):
    fix_qs = um.FIX_QS.copy()
    if job.get('left'):
        um.FIX_QS[20] = 0.699999988
    base = 1500 if PLAN == 'stand' else 3000
    PUSH.clear()
    PUSH.update(tick=base + job['offset'], dir=DIRS[job['dir']], mag=job['mag'], start='fixair')
    case = st.case(physics=job['phys'], plan='kstand' if PLAN == 'stand' else 'kwalk', stage=job['path'])
    try:
        r = st.run(case)
    finally:
        um.FIX_QS[:] = fix_qs
    s = PUSH['stepper']
    fell = r['fell_at']
    # fell_at is counted from the Velocity entry (ticks of the run loop): push at release 700 + tick
    t_push = (700 + PUSH['tick']) * st.TICK
    return dict(job={k: v for k, v in job.items() if k != 'path'}, fell=None if fell is None else fell - t_push,
                fell_before=fell is not None and fell < t_push, fix=r['fix'], vmax=s.vmax, zmin=s.zmin, segs=r['segs'])


def main():
    out, workers = sys.argv[1], int(sys.argv[2])
    pols = []
    for p in sys.argv[3:]:
        left = p.endswith('@L')
        p = p[:-2] if left else p
        n, path = p.split('=', 1) if '=' in p else (p, p)
        pols.append((n + ('@L' if left else ''), path, left))
    jobs = [dict(name=n, path=path, left=left, phys=ph, dir=dr, mag=g, offset=o)
            for n, path, left in pols for ph in PHYS for dr in DIRS for g in MAGS for o in OFFSETS]
    with Pool(workers, maxtasksperchild=16) as pool:
        results = pool.map(run, jobs, chunksize=1)
    Path(out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    tab = {}
    for r in results:
        j = r['job']
        tab.setdefault((j['name'], j['phys'], j['dir']), {}).setdefault(j['mag'], []).append(r)
    print('plan', PLAN, 'falls of', len(OFFSETS), 'per magnitude', MAGS, '; peak pelvis speed along the push')
    for (n, ph, dr), by in tab.items():
        cells = ' '.join('%d' % sum(r['fell'] is not None for r in by[g]) for g in MAGS)
        pre = sum(r['fell_before'] for g in MAGS for r in by[g])
        vm = ' '.join('%.2f' % np.mean([r['vmax'] for r in by[g]]) for g in MAGS)
        print(f'{n:10s} {ph:5s} {dr}: {cells}   vmax {vm}' + (f'  FELL BEFORE PUSH {pre}' if pre else ''))


if __name__ == '__main__':
    main()
