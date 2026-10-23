"""CPU only. speed_probe.py <branch>[:steps] [<branch>[:steps] ...] [--last N] [--workers W]: G1 stages at commands
1.5 and 2 m/s (speed_terms.py, the critic's iteration-22 measurement: 10 s, metrics after 3 s, seed 73000, deterministic).
Prints the forward speed, the fall time, the reward per step with feet_flat and slip, the mean swing-foot tilt, the share
of hip-pitch sub-steps at the motor's peak torque and the pelvis vertical speed. A branch without :steps takes its last
N stages (default 1). Iteration 22 at +50 M: at 2 m/s, speed <= 1.5, hip at peak >= 4% and swing tilt <= 10 deg ->
branch with feet_flat_stance; speed >= 1.6 -> go on (docs/lab-knowledge.md, iteration 22)."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_terms as S  # noqa: E402

from humanoid_sim.core import list_stages  # noqa: E402


def stages(spec, last):
    branch, _, steps = spec.partition(':')
    folder = os.path.join(S.LAB, branch)
    if steps:
        return [os.path.join(folder, f'stage_{int(steps):012d}')]
    return [str(path) for path in list_stages(folder)[-last:]]


def job(args):
    path, command = args
    S.POL['x'] = path
    return path, command, S.run(dict(pol='x', cmd=command, seed=73000))


def verdict(result):
    speed, hip, tilt = result['vx'], result['groups']['hip_pitch']['sat'], result['tilt_swing'] or 0.
    if result['fell'] is None and speed >= 1.6:
        return 'дальше'
    if speed <= 1.5 and hip >= .04 and tilt <= 10.:
        return 'триггер feet_flat_stance'
    return 'смотреть'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stages', nargs='+')
    parser.add_argument('--last', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    paths = [path for spec in args.stages for path in stages(spec, args.last)]
    with Pool(min(args.workers, 2 * len(paths))) as pool:
        results = pool.map(job, [(path, command) for path in paths for command in (1.5, 2.)], chunksize=1)
    sys.stdout.reconfigure(encoding='utf-8')
    for path, command, r in results:
        c = r['comps']
        name = f"{os.path.basename(os.path.dirname(path))[:12]} {int(os.path.basename(path)[6:]) / 1e6:7.1f}"
        line = (f"{name} команда {command:.1f}: vx {r['vx']:+.2f} падение {r['fell']} награда {r['total']:+.2f}"
                f" feet_flat {c.get('feet_flat', 0):+.3f} slip {c.get('slip', 0):+.3f}"
                f" наклон в переносе {r['tilt_swing'] or 0:.1f}° бедро на пике {r['groups']['hip_pitch']['sat']:.1%}"
                f" vz {r['vz_rms'] or 0:.2f}")
        print(line + (f" → {verdict(r)}" if command == 2. else ''))


if __name__ == '__main__':
    main()
