"""robust_table.py <rp.json>: summary of robust_probe results."""
import json
import sys

import numpy as np

R = json.loads(open(sys.argv[1]).read())


def walkseg(r):
    return r['segs'][-1] if r['segs'] else None


def line(label, rs):
    falls = sum(r['fell_at'] is not None for r in rs)
    segs = [walkseg(r) for r in rs if walkseg(r)]
    if not segs:
        print(f'{label}: falls {falls}/{len(rs)} (no measured segment)')
        return
    y = [s['yaw_rate'] for s in segs]
    print(f"{label}: falls {falls}/{len(rs)} yaw {np.mean(y):+.3f} [{min(y):+.3f},{max(y):+.3f}] "
          f"speed {np.mean([s['speed'] for s in segs]):.2f} side {np.mean([s['side'] for s in segs]):+.2f} "
          f"lift {1000 * min(s['lift'] for s in segs):.0f}mm tilt {max(s['tilt'] for s in segs):.0f} "
          f"lean {np.mean([s['lean'] for s in segs]):+.1f}")


sel = lambda **kw: [r for r in R if all(r[k] == v for k, v in kw.items())]
print('== reset start: clock b+k (b=0 left, b=15 right), delay 0')
for v in (.5, 1.):
    for b in (0, 15):
        for k in (-3, -2, -1, 0, 1, 2, 3):
            line(f'v={v} b={b:2d} k={k:+d}', sel(kind='reset', v=v, phase=b + k, delay=0, k=0))
print('== jump: clock jumps k steps at 5 s (measured 7..15 s)')
for v in (.5, 1.):
    for b in (0, 15):
        for k in (-3, -2, -1, 0, 1, 2, 3):
            line(f'v={v} b={b:2d} k={k:+d}', sel(kind='jump', v=v, phase=b, k=k, delay=0))
print('== standwalk: stand 60 steps, walk at clock p (delay 0)')
for v in (.5, 1.):
    rs = sel(kind='standwalk', v=v, delay=0)
    y = {r['phase']: walkseg(r)['yaw_rate'] if walkseg(r) else float('nan') for r in rs}
    line(f'v={v} all phases', rs)
    print('   per phase:', ' '.join(f'{p}:{y[p]:+.3f}' for p in sorted(y)))
print('== jitter')
for v in (.5, 1.):
    for J in (1, 2):
        line(f'v={v} J=+-{J}', sel(kind='jitter', v=v, k=J))
print('== delay (substeps of 2 ms)')
for d in (0, 5, 10, 20):
    for v in (.5, 1.):
        for b in (0, 15):
            line(f'd={d:2d} v={v} b={b:2d}', sel(kind='reset', v=v, phase=b, delay=d, k=0))
    rs = sel(kind='stand', delay=d)
    segs = [r['segs'][0] for r in rs if r['segs']]
    print(f"d={d:2d} stand: falls {sum(r['fell_at'] is not None for r in rs)}/{len(rs)} crawl "
          f"{' '.join(f'{s['crawl']:.2f}' for s in segs)} liftoffs {' '.join(str(s['liftoffs']) for s in segs)} "
          f"lean {np.mean([s['lean'] for s in segs]):+.1f}")
    rs = sel(kind='stopstart', delay=d)
    for r in rs:
        print(f"d={d:2d} stopstart seed {r['seed']} fell {r['fell_at']} segs " + ' | '.join(
            (f"walk{s['cmd'][0]} v {s['speed']:.2f} yaw {s['yaw_rate']:+.3f} lift {1000 * s['lift']:.0f}"
             if s['cmd'][0] else f"stand crawl {s['crawl']:.2f} lo {s['liftoffs']}") for s in r['segs']))
for d in (5, 10):
    for v in (.5, 1.):
        rs = sel(kind='standwalk', v=v, delay=d)
        line(f'standwalk d={d} v={v} phases {sorted(r["phase"] for r in rs)}', rs)
