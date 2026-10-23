# Full CUDA training

User-approved scope: preserve the hybrid trainer; run MuJoCo Warp physics,
observations/rewards, normalization, rollout storage and PPO on CUDA; keep browser
checkpoints and reports compatible. Measure 256/512/1024 worlds before selecting
a default. CPU remains responsible for orchestration, scalar diagnostics, files,
evaluation and browser playback.

Implementation:

- `tensor_env.py`: zero-copy Warp/Torch state, masked CUDA resets, Newton/RK4.
- `tensor_ppo.py`: CUDA running statistics, GAE and PPO; SB3 policy architecture.
- `full_gpu_train.py`: isolated runtime, existing locks, stop/timer and stage export.
- `full_gpu_benchmark.py`: equal 8192-transition rollouts, careful preset, seed42,
  same mini-batch/epochs for hybrid and full GPU. Separate setup/warmup from measured
  training; no checkpoint evaluation inside the measurement. CUDA reset RNG differs
  from hybrid NumPy, so timings are comparable workloads, not identical trajectories.
- Browser: explicit experimental Full GPU option, separate end-to-end benchmark.

Validation gates:

1. Running statistics against SB3; GAE reset/time-limit examples.
2. PPO updates and CPU archive replay/optimizer resume equivalence.
3. Real CUDA physics parity, resets, custom stream and absence of per-step host reads.
4. End-to-end benchmark on the actual RTX4080SUPER.
5. Bounded training with intermediate exports and CPU evaluation, checking browser API.

Do not claim a speedup until measured. Keep hybrid as default if full CUDA is slower.
The GPU physics variant uses Newton instead of Gym's default PGS and float32;
training and evaluation trajectories are not expected to be bit-identical.
