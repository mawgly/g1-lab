# Third-party material

The project's own code is under the Apache License 2.0 (`LICENSE`). The files below come from other projects
and keep their own licenses, which are stored next to them.

## Included in this repository

| Path | Source | License |
|---|---|---|
| `assets/robots/g1/g1_29dof.xml`, `scene_29dof.xml`, `meshes/` | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`; SHA-256 of every file in `assets/robots/g1/provenance.json` | BSD 3-Clause, Unitree Robotics: `assets/robots/g1/LICENSE` |
| `assets/robots/g1/controller.json` (PD gains, action scale, default pose) | `deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml` of [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab), commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` (URL in the file's `source` field) | Apache 2.0: `assets/robots/g1/CONTROLLER_LICENSE` |
| `models/so101/` | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101), `Simulation/SO101`, revision in `models/so101/UPSTREAM_REVISION.txt`, unchanged | Apache 2.0: `models/so101/LICENSE` |
| `humanoid_sim/web/vendor/three.core.js`, `three.module.js`, `OrbitControls.js` | [three.js](https://github.com/mrdoob/three.js) r180 | MIT: `humanoid_sim/web/vendor/LICENSE-three.txt` |
| `humanoid_sim/figures/anatomy/` (`kit/iso-kit.mjs`, `scripts/inline-kit.mjs`) | [wheresryan22/anatomy](https://github.com/wheresryan22/anatomy), commit `1482e8469b0a9e0d8cc18ea2740a01423d29ae80`, unchanged | MIT: `humanoid_sim/figures/anatomy/LICENSE` |
| `scripts/sim2sim/patches/unitree_mujoco-push.patch` | Changes to `simulate/src/main.cc` of unitree_mujoco (commit above); the diff context lines are Unitree's code | BSD 3-Clause, Unitree Robotics |
| `scripts/sim2sim/patches/g1_ctrl-*.patch` | Changes to `deploy/` (g1_ctrl) of unitree_rl_mjlab (commit above); the diff context lines are Unitree's code | Apache 2.0 |

## Derived from other projects

- Constants and formulas re-implemented from Unitree's training code, each cited in a code comment: the motor
  torque–speed curves and friction (`unitree_rl_lab` `unitree_actuators.py`, Apache 2.0), the foot collision
  capsules, the foot clearance and self-collision terms and the deploy observation layout (`unitree_rl_mjlab`,
  Apache 2.0).
- Methods taken from papers and open-source RL code (legged_gym, Isaac Lab, mjlab, rsl_rl, rl_games) are named in
  code comments where they are used. No code from these projects is included.

## Fetched by the setup scripts, not included

- Python packages from `requirements-lock.txt` and `requirements-gpu.txt` (MuJoCo, MuJoCo Warp, Gymnasium,
  Stable-Baselines3, PyTorch and others), installed by `setup.cmd` / `setup_fullgpu.cmd` under their own
  licenses. The Gymnasium Humanoid-v5 model is loaded from the installed Gymnasium package.
- `scripts/sim2sim/setup.sh` clones unitree_sdk2, unitree_mujoco and unitree_rl_mjlab at pinned commits and
  downloads the MuJoCo 3.3.6 release into WSL; nothing of them is stored here.

## Data used locally, not included

- CMU Graphics Lab Motion Capture Database ([mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/)), subject 140 (getting up
  from the ground), read from `.tools/datasets/cmu_mocap_140` (untracked) by `scripts/lab/cmu_mocap.py` and
  retargeted to the G1 by `scripts/lab/getup_track_clip.py --source cmu` into a getting-up tracking clip under
  `runs/` (untracked). The database's terms allow its use in commercially sold products but not reselling the data,
  even in converted form, so neither the recordings nor the clip are published. The data used in this project was
  obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.
- BONES-SEED ([bones.studio/datasets/seed](https://bones.studio/datasets/seed)), Bones Studio's motion capture in
  NVIDIA's Unitree G1 retarget: getting-up takes read from `.tools/datasets/bones_seed` (untracked) by
  `scripts/lab/getup_track_clip.py --source bones` into getting-up tracking clips under `runs/` (untracked). Its
  license (the BONES-SEED Dataset License Agreement) lets a qualifying startup train and commercialize models on it,
  but not redistribute or publish the data, derived clips or a competing dataset, so neither is published. Training
  data includes Motion Data by Bones Studio (https://bones.studio/). Use of the underlying dataset is subject to the
  BONES Motion Capture Dataset License Agreement.
