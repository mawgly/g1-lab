"""Side steps over a series of stages, iteration 25's check (from the critic's lab-critic/it25/side_series.py).

The bridge copy (stand_turn physics 'um', its default start and delay) runs two plans:
  'side'   ±0.3 m/s;
  'side1'  ±1 m/s.

Per stage it prints:
  - the yaw drift (rad/s) of each side step;
  - the sideways speed at ±1;
  - how far the robot moves in the stop after each ±1 step (m);
  - falls.

Then a summary of the series:
  - the worst |drift ±1| per stage, mean and median;
  - how many stages keep both ±1 drifts below 0.05;
  - how many keep both ±0.3 drifts below 0.05.

side_series.py <branch> [--last N | --steps S ...] [--workers 4] [--physics um|ours] [--out file.json]
CPU only; keep --workers at 4 or less while a trainer runs (at 8 it loses ~15% steps/s)."""
import argparse
import json
import os
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import stand_turn as st  # noqa: E402

LIMIT = 0.05


def job(args):
    stage, plan, physics = args
    result = st.run(st.case(stage=stage, plan=plan, physics=physics))
    segs = [dict(name=s['name'], yaw=s['yaw_rate'], side=s['side'], disp=s['disp'])
            for s in result['segs'] if not s.get('missing')]
    return dict(stage=stage, plan=plan, fell=result['fell_at'], segs=segs)


def after(segs, name, key):
    """The value of `key` of segment `name` and of the stop right after it (None when missing)."""
    for i, seg in enumerate(segs):
        if seg['name'] == name:
            stop = segs[i + 1] if i + 1 < len(segs) and segs[i + 1]['name'] == 'stop' else None
            return seg[key], (stop['disp'] if stop else None)
    return None, None


def stages_of(branch, last, steps):
    from humanoid_sim.core import DEFAULT_RUN, list_stages
    from humanoid_sim.lab import Laboratory
    found = list_stages(Laboratory(DEFAULT_RUN).resolve(branch))
    if steps:
        chosen = [path for path in found if int(path.name[6:]) in set(steps)]
        missing = set(steps) - {int(path.name[6:]) for path in chosen}
        if missing:
            raise SystemExit(f'у {branch} нет этапов {sorted(missing)}')
        return chosen
    return found[-last:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('branch')
    parser.add_argument('--last', type=int, default=5)
    parser.add_argument('--steps', type=int, nargs='*')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--physics', default='um', help="stand_turn physics: 'um' (bridge copy, default) or 'ours' "
                        '(our CPU task with a zero turn command and no heading hold: the critic\'s "ours without '
                        'hold")')
    parser.add_argument('--out')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    stages = stages_of(args.branch, args.last, args.steps)
    jobs = [(str(stage), plan, args.physics) for stage in stages for plan in ('side', 'side1')]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(job, jobs, chunksize=1)
    by = {(r['stage'], r['plan']): r for r in results}
    rows = []
    for stage in stages:
        small, big = by[(str(stage), 'side')], by[(str(stage), 'side1')]
        yaw_l, _ = after(small['segs'], 'left', 'yaw')
        yaw_r, _ = after(small['segs'], 'right', 'yaw')
        yaw_l1, stop_l1 = after(big['segs'], 'left1', 'yaw')
        yaw_r1, stop_r1 = after(big['segs'], 'right1', 'yaw')
        speed_l1, _ = after(big['segs'], 'left1', 'side')
        speed_r1, _ = after(big['segs'], 'right1', 'side')
        rows.append(dict(steps=int(stage.name[6:]), yaw03=(yaw_l, yaw_r), yaw1=(yaw_l1, yaw_r1),
                         speed1=(speed_l1, None if speed_r1 is None else -speed_r1), stop1=(stop_l1, stop_r1),
                         fell=[r['fell'] for r in (small, big) if r['fell'] is not None]))

    def show(pair, digits=3, sign=True):
        form = f'{{:+.{digits}f}}' if sign else f'{{:.{digits}f}}'
        return '/'.join('—' if value is None else form.format(value) for value in pair)

    print('этап, млн | увод ±0,3 влево/вправо | увод ±1 влево/вправо | скорость ±1 | стоп после ±1, м | падения')
    for row in rows:
        print(f"{row['steps'] / 1e6:.1f} | {show(row['yaw03'])} | {show(row['yaw1'])} | "
              f"{show(row['speed1'], 2, False)} | {show(row['stop1'], 2, False)} | {row['fell'] or 0}")
    worst = [max(abs(v) for v in row['yaw1'] if v is not None) for row in rows if any(row['yaw1'])]
    both1 = sum(all(v is not None and abs(v) < LIMIT for v in row['yaw1']) for row in rows)
    both03 = sum(all(v is not None and abs(v) < LIMIT for v in row['yaw03']) for row in rows)
    speeds = [[row['speed1'][k] for row in rows if row['speed1'][k] is not None] for k in (0, 1)]
    stops = [v for row in rows for v in row['stop1'] if v is not None]
    falls = sum(bool(row['fell']) for row in rows)
    summary = dict(stages=len(rows), worst_mean=statistics.mean(worst) if worst else None,
                   worst_median=statistics.median(worst) if worst else None, both1=both1, both03=both03,
                   speed_median=[statistics.median(s) if s else None for s in speeds],
                   stop_max=max(stops) if stops else None, falls=falls)
    print(f"серия {len(rows)}: худший |увод ±1| среднее {summary['worst_mean']:.3f}, медиана "
          f"{summary['worst_median']:.3f}; оба ±1 < {LIMIT:g}: {both1} из {len(rows)}; оба ±0,3 < {LIMIT:g}: "
          f"{both03} из {len(rows)}; скорость ±1 медиана {show(summary['speed_median'], 2, False)}; стоп после ±1 "
          f"до {summary['stop_max']:.2f} м; этапов с падением {falls}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(rows=rows, summary=summary, results=results), indent=1),
                                  encoding='utf-8')


if __name__ == '__main__':
    main()
