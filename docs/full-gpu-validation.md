# Full GPU verification — 2026-10-19

RTX 4080 SUPER, isolated `.venv-fullgpu`, PyTorch2.14.1+cu130.
59 tests passed in the CUDA runtime, including real physics, zero-copy state,
selective reset/stream ordering, PPO math, CPU replay/export and stop lifecycle.
Browser API smoke checks passed. Visual browser verification was unavailable
because the browser tool rejected access; no visual verification is claimed.

## Short throughput comparison

Eight measured rollouts after one warmup; 8192 transitions per rollout, careful
preset, seed42, batch256, epochs5, KLtarget0.02. Hybrid runs in the existing
`.venv-gpu` CPU-PyTorch runtime. Includes physics and PPO; excludes evaluation
and checkpoint export. Different reset RNGs mean trajectories are not identical.

| Worlds | Full GPU steps/s | Hybrid steps/s | Ratio |
|---|---:|---:|---:|
| 256 | 6498 | 4065 | 1.60 |
| 512 | 7360 | 4875 | 1.51 |
| 1024 | 7810 | 3556 | 2.20 |

Raw results: `runs/humanoid_lab/full_gpu_benchmark.json` (also `lab/`).
These replace the preliminary comparison that ran hybrid CPU kernels inside
the CUDA PyTorch environment.

## Complete learning run

Experiment `exp-0183f7e4077145e6`, fresh careful preset, seed43, 256worlds,
32steps/world. Completed2007040 steps in317.281seconds including stage work.
Five checkpoints saved and reported; sleep protection released after completion.

| Steps | Evaluation return | Mean upright seconds |
|---:|---:|---:|
| 0 | 181 | 0.549 |
| 507904 | 457 | 1.284 |
| 1015808 | 493 | 1.35 |
| 1523712 | 5736 | 15.0 |
| 2007040 | 6133 | 15.0 |

Final evaluation: seeds60000–60004, no falls in5/5episodes, mean forward16.75m.
Each episode is capped at15seconds; this does not establish indefinite walking.

The earlier hybrid careful seed43 run completed in~214.9seconds, with final
return5920 and15seconds upright. Thus the short throughput improvement did not
translate into a faster complete run in this comparison. One run per backend
does not establish statistical superiority; keep hybrid as browser default and
offer fullGPU explicitly. Both archives remain available.

Automatic approval review rejected deletion of temporary wheel download parts
with `blocked by policy`; those files remain. Runtime installation is complete.
