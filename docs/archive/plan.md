# SO-ARM101 Implementation Plan

Goal: deliver and verify the complete locally runnable reaching experiment described in design.md.
Architecture: upstream MJCF + Gymnasium environment + SB3 PPO/SAC training + evaluation/viewer CLI.
Tech stack: Python 3.12, MuJoCo, Gymnasium, Stable-Baselines3, PyTorch CPU.
Execution: inline in the user-selected empty Sim workspace; no existing repository or checkout to isolate. User explicitly requested all installation and implementation; proceed without further approval handoffs.

## Tasks
- [x] Install standalone Python, create .venv, install dependencies, vendor official model and preserve license/revision.
- [x] Write environment behavior tests (seed reproducibility, bounded finite physics, action effects, attainable target, invalid input, horizon). Observe missing implementation failure, implement arm_sim/env.py and pass suite.
- [x] Implement train/evaluate/view CLI with validation and model metadata. Test model save/load and deterministic evaluation. Train and evaluate 100 held-out episodes against random controls.
- [x] Add Windows launchers, Russian README, render preview and verify native viewer. Review whole project and close any defects.

## Review focus
- Paths containing spaces and execution from another working directory: launchers cd to own directory.
- Missing model: clear actionable error; manual viewer works without trained checkpoint.
- Invalid/nonfinite actions: reject before mutating physics.
- Time-limit vs success termination: distinguish truncation from termination.
- Unreachable or colliding targets: filter candidate configurations, retain goal configuration for tests only through environment state.

## Evidence ledger
- Initial folder empty; Python command was only Windows Store alias. Installing Python 3.12.10 per-user, official signed installer verified.
- Upstream model: TheRobotStudio/SO-ARM100 Simulation/SO101, new calibration.
- Python MSI installer failed with Windows Installer error 2203 / code 3. Installed independent uv-managed Python 3.12.15 under .tools/python instead; .venv contains 29 pinned packages. setup.ps1 rerun passed.
- Environment tests initially failed for missing implementation; six behavioral tests passed after implementation. CLI tests initially failed for missing implementation, then full 10-test suite passed.
- First 1M-step training completed. Native manual viewer and trained viewer both launched and exited cleanly; initial held-out evaluation 100/100 vs 0/100 random.
- Fresh reviewer found stale site_xpos after mj_step and mutable checkpoint pathname in report. Two regression tests reproduced both defects. Added mj_forward after integration and evaluate from an in-memory checkpoint snapshot with SHA-256 and timesteps. Full suite: 12 passed.
- Final training rerun from scratch on corrected physics; initial report is preliminary until overwritten by final evaluation after training completes.
- Final training exited 0. Final held-out evaluation after completion: 100/100 trained vs 0/100 random; 0.18 cm vs 26.66 cm mean final error. Final pytest: 12 passed. pip check: all 29 packages compatible. Resume CLI smoke passed, saving a new last checkpoint. Both manual.cmd and demo.cmd ran from outside the project and exited 0; native viewer rendered and reported successful goals. Fresh review of corrections found no remaining important defects.
