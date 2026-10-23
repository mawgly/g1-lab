# SO-ARM101 reaching experiment

User authorized full installation and implementation on 2026-10-15. Use the actual SO-101 follower model from TheRobotStudio, not a two-link substitute. Run locally on Windows without camera or physical hardware.

Deliverables: isolated Python environment, upstream MJCF and meshes with license and revision, Gymnasium environment, Stable-Baselines3 training with saved model, held-out evaluation against random actions, native MuJoCo viewer, Windows launchers, Russian instructions, tests and a measured trained result.

The policy commands incremental position targets for five arm joints. The gripper stays open for reaching. Observations include joint positions and velocities, end-effector position and target displacement. Reward uses distance, progress, small action cost and sustained success. Targets are sampled using forward kinematics of valid collision-free configurations, with height and workspace bounds. Physics retains gravity, actuator force limits and collisions. No camera or external AI API is required.

Episodes start near a fixed home pose and last at most 150 control steps. Success means the gripper reference point remains within 2 cm of the target for 5 consecutive steps. Evaluation uses disjoint fixed seeds and reports success fraction and final distance; random actions use the same episodes. Acceptance: native model loads, finite stable physics, reproducible reset, meaningful action effects, successful save/reload, viewer renders, and trained policy beats random with at least 80% success on 100 held-out goals. Training length may be extended based on measured results.

This is a simulation experiment. Hardware control, motor calibration, grasping objects and camera vision are subsequent tasks. The upstream model is an approximation of hardware dynamics.
