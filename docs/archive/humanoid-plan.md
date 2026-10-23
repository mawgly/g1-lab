# Humanoid-v5 experiment, 2026-10-18

User asked to try Humanoid-v5 and watch intermediate stages. Existing permission covers installation and local execution. Extend the local experiment with a separate humanoid module and launchers; preserve the arm project. Implement inline.

Design: standard Gymnasium Humanoid-v5, PPO on CPU with 8 parallel environments, VecNormalize observation/reward normalization. Save policy and matching normalization statistics together at rollout boundaries around every 100k steps, including the untrained baseline. Publish a checkpoint directory atomically only after its files are complete; viewer only reads published directories. Keep every stage, never overwrite old stages. Resume latest by default and refuse simultaneous training for one run via OS file lock. Initial run: 2 million additional steps, learning progress is experimental, no promised walking threshold.

Watching: separate MuJoCo window, live/latest mode, left/right stage selection, L follow latest, R reset, Space pause. HUD shows stage number, training state, current episode time and evaluation (mean return, time before termination, forward distance on fixed seeds). Latest reload at episode boundary or manual key switch, never load mismatched normalization. Closing viewer does not stop trainer. Stop via separate launcher requesting graceful save; continue from latest. Progress/status JSON for live stage display. Finite native viewer smoke option for verification.

Tasks:
- [x] Tests first: checkpoint roundtrip preserves deterministic actions and normalized observations, incomplete snapshots ignored, stage selection clamps and live-follow updates, evaluation reproducible, run lock rejects concurrent writer.
- [x] Implement checkpoint helpers, training/status/evaluation CLI and Windows controls. Smoke train/save/resume.
- [x] Implement viewer, verify render and native window/controls; add Russian instructions.
- [x] Run complete tests and fresh review. Launch a real experiment and live viewer; report actual stage and state, without claiming learned walking before evidence.

Evidence: gymnasium[mujoco] needed imageio; installed and lockfile updated. RED missing module, then 4 checkpoint/eval/selection/lock tests passed. Viewer control test RED then GREEN. Whole suite now 19 passed, including unchanged arm tests. Checkpoint smoke run 0/1024/2048 and resume3072 worked. Native viewer ran with screenshot; inspected scene and HUD. Launcher works from outside project. Fresh reviewer identified navigation origin during live updates and finalization status on I/O failure; regression tests failed before fixes and passed after. Reviewer verified fixes with no remaining important findings. Main trainer stopped gracefully at288392, resumed toward2M; second trainer correctly rejected by OS lock. Separate main viewer running and user can close it without stopping training. Learning is intentionally ongoing for user observation, not claimed complete or solved.
