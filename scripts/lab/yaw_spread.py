"""Per-seed yaw rate without a turn command, from either foot: yaw_spread.py <branch:steps> [seeds]."""
import sys
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN  # noqa: E402
from humanoid_sim.goal_check import find_stage, walks  # noqa: E402
from humanoid_sim.lab import Laboratory  # noqa: E402

stage = find_stage(Laboratory(DEFAULT_RUN), sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for key, values in walks(stage, seeds=seeds)['drift'].items():  # (speed, start foot): one rate per seed
    print(sys.argv[1], key, 'mean', round(statistics.mean(values), 3), 'sd', round(statistics.stdev(values), 3),
          'per seed', [round(v, 3) for v in values])
