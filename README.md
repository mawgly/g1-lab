# G1 Lab: teaching a Unitree G1 humanoid to walk, from scratch

A one-person home lab that trains walking policies for the Unitree G1 (29 DoF) in MuJoCo with reinforcement
learning. AI agents (Claude, by Anthropic) write the code and run the experiments: a lead agent plans each
iteration, and critic, scout, analyst, engineer and chronicler agents (`.claude/agents/`) review the idea, look up
how others solve it, measure, implement and write the journal. The owner, Pavel Maltsev, watches and corrects.

Training is from scratch: random initialisation, PPO, no pretrained models and no motion-capture data.

The lab journal, the branch tree and the comparisons are on **https://www.g1lab.ru** (English, Chinese, Russian).

> [!WARNING]
> **Not tested on a real robot.** Everything here has run only in simulation: our MuJoCo setup and Unitree's own
> simulator (`unitree_mujoco` with Unitree's `g1_ctrl` deploy controller). No policy from this repository has
> been run on a physical G1. A humanoid robot is heavy, and its motors can injure people and damage the robot.
> If you run any of this on hardware, you do so entirely at your own risk: use a hoist, keep people clear, keep
> the emergency stop in hand and follow Unitree's safety instructions. The software is provided "as is", without
> warranty of any kind (see `LICENSE`).

## Results: candidate 7

The current candidate is stage `exp-c362df8e578148d0` @ 6,368,231,424 steps (the step count runs along the whole
branch lineage). It comes from iteration 24, which trained with Unitree's foot collision capsules and their
friction, so in training the feet can no longer pass through each other. It is exported for `g1_ctrl` with the commands
narrowed to forward −0.5…1 m/s, sideways ±0.5 m/s and turning ±1 rad/s.

### In Unitree's simulator (sim2sim)

These results come from `unitree_mujoco` plus `g1_ctrl` (unitree_rl_mjlab deploy) in WSL2, driven by a virtual
gamepad. The set has 22 runs, detailed in `docs/sim2sim.md` («Этап 6368 (итерация 24)»).

| Test | Result |
|---|---|
| Falls | 0 in 22 runs |
| Pushes on the pelvis (a force for 0.1 s) | 0 falls in 52: Δv 0.5 m/s 0 of 12; Δv 1 standing 0 of 8, walking 0 of 4; Δv 1.25 standing 0 of 20; Δv 1.5 standing 0 of 8. Largest lean in standing pushes ≤ 11° |
| Walk at 0.5 m/s | 0.51 m/s, yaw drift −0.004…+0.001 rad/s, ≤ 5 cm sideways in 10 s |
| Walk at 1 m/s | 0.97 m/s, yaw drift +0.028 / +0.031 rad/s (heading +12…+13° in 10 s) |
| Side step ±0.3 m/s | 0.26–0.27 m/s, \|drift\| ≤ 0.005 rad/s |
| Turn ±0.5 rad/s | in place +0.505 / −0.453, walking +0.500 / −0.461 rad/s |
| Stand for 30 s | heading −5.8…−6.3°, 2–3 cm shift, no step |
| Stop from 0.5 m/s | 0.23–0.26 m within 3 s |
| Latency, observation → motor command | median 6 ms, 95th percentile 10 ms |

The checks in our own physics agreed with these: the speed acceptance passed 7 of 7, and `goal_check` passed 10 of
10 with a drift of 0.026 rad/s.

### Against Unitree's stock policy (v0)

Unitree ships a walking policy (v0) with the robot. We ran it in the same harness as ours: our CPU physics with
Unitree's motor curves and the same starts. v0 was not run in Unitree's simulator. Where v0 is better, the table
says so.

| Test | Candidate 7 | Unitree v0 | Better |
|---|---|---|---|
| Speed at a command of 1 / 1.5 / 2 m/s | 1.02 / 1.54 / 1.94 m/s | 0.94 / 1.38 / 1.78 m/s | candidate 7 |
| Creep 2–6 s after stopping from 1.5 / 2 m/s | 0.7–0.9 / 2.9–5.8 cm | 0.2 / 0.2–0.4 cm | **v0** |
| Braking from 2 m/s to < 0.1 m/s | 0.90 m (from 1.93 m/s) | 0.86 m (from 1.78 m/s) | about even |
| Stance foot tilt at 1 / 2 m/s | 1.7° / 5.0° | 4.9° / 12.7° | candidate 7 |

### Known weaknesses

These are listed in `docs/sim2sim.md`:

- **Stopping is softer.** Braking is the same, but afterwards the pelvis keeps moving 7–9 cm forward over the feet.
  After 1 m/s there was one step back in 17 stops.
- **Drift at 1 m/s** is +0.028…+0.031 rad/s.
- **Steps and turns fall short of the command.** The side step reaches 88% of it, and turning right 91–92%.
- **The knees are asymmetric when standing.** The robot slowly turns right while standing, −6° in 30 s.
- **The feet can still touch in side pushes.** With Δv ≥ 1 standing, Unitree's foot capsules would touch in 12 of 18
  pushes, judged from the logged poses. `unitree_mujoco` has no foot capsules, so its simulator cannot show this.
- **Running at 2 m/s was not tested in Unitree's simulator.** It needs a patch for the gamepad stick scale.

## How it works

- **Robot.** The G1 29-DoF model is taken unchanged from `unitree_mujoco`; the SHA-256 of every file is in
  `assets/robots/g1/provenance.json`. The PD gains, action scale and default pose come from unitree_rl_mjlab's
  `deploy.yaml`.
- **Task.** `g1_velocity` follows velocity commands. In its `real_robot` profile, the observations are exactly those
  of Unitree's `deploy.yaml`, with a 5-frame history and no base linear velocity. The profile also has Unitree's
  motor torque–speed curves and friction, sensor noise, randomisation and pushes. The critic gets extra inputs
  (asymmetric actor-critic).
- **One set of equations.** The observation, reward and command equations live in one place,
  `humanoid_sim/tasks/g1_tensor_math.py`, which both backends use: the CPU environment (MuJoCo) and the CUDA one
  (MuJoCo Warp).
- **Training.** Physics and PPO both run on the GPU (`full_gpu_train.py`, `tensor_ppo.py`). Stages are saved in
  Stable-Baselines3 format, so evaluation and playback always run on the CPU MuJoCo.
- **Research loop.** Every experiment is a branch of a tree that changes one thing. Each branch is checked with
  `supervise` and `goal_check`, and each decision is written to the journal. The rules are in
  `docs/research-loop.md`.
- **Export and sim2sim.** `export_g1` writes `policy.onnx` and `deploy.yaml` for Unitree's `g1_ctrl`.
  `scripts/sim2sim/` builds Unitree's simulator and controller in WSL2 and runs scenarios with a virtual gamepad.

Most documents are in Russian, the lab's working language. The site has English and Chinese versions.

## Repository layout

| Path | Contents |
|---|---|
| `humanoid_sim/` | The lab: G1 tasks, CPU and GPU training, evaluation, export, the web server and site (`web/`), the journal |
| `arm_sim/` | The first experiment: the SO-101 arm reaching targets (see `README.ru.md`) |
| `assets/robots/g1/`, `models/so101/` | Robot models (third-party, see `THIRD_PARTY.md`) |
| `scripts/lab/` | Branching helpers (`task_branch.py`, `reward_branch.py`, `command_branch.py`, `ppo_branch.py`, `contract_branch.py`) and probes |
| `scripts/sim2sim/` | Unitree simulator setup, runs, patches for `unitree_mujoco` and `g1_ctrl`, log analysis |
| `scripts/robot/` | The package and protocol for the first run on a real robot (not done yet, see `docs/robot-first-run.md`) |
| `docs/` | Progress, knowledge base, roadmap, sim2sim results |
| `tests/` | pytest suite |
| `.claude/agents/` | The lab's agent roles |

## Setup (Windows only)

The project runs on Windows 10/11 x64. It uses Windows-only file locks and `.cmd` launchers. GPU training, as used
for the G1, needs an NVIDIA GPU with a driver that supports CUDA 13.0. Evaluation, playback and export run on the
CPU. The first setup needs internet access.

- `setup.cmd` downloads uv into `.tools/`, installs Python 3.12 into `.tools/python` and creates `.venv` from
  `requirements-lock.txt`. Then it runs the test suite. Nothing is added to the system PATH.
- `setup_fullgpu.cmd` creates `.venv-fullgpu` with PyTorch 2.14.1+cu130 and MuJoCo Warp (`requirements-gpu.txt`)
  and checks that CUDA works.

`runs/` (experiments and checkpoints), `.venv*` and `.tools/` are not part of the repository.

## Usage

Start the lab server (localhost only):

```powershell
.\.venv\Scripts\python.exe -m humanoid_sim.webapp --open    # http://127.0.0.1:8766
```

The pages show the journal, the branch tree and live 3D races of saved stages. There are no training controls in the
browser.

**Training.** Experiments live in `runs/humanoid_lab/` and are created and queued through
`humanoid_sim.lab.Laboratory`, or the server's POST endpoints `/api/lab/create`, `/api/lab/enqueue` and
`/api/lab/start`. `lab.validate_config` is the config schema. A queue worker runs each job as
`python -m humanoid_sim.cli train --config run_config.json` in the environment the backend needs. To branch from an
existing stage with one setting changed:

```powershell
.\.venv\Scripts\python.exe scripts\lab\task_branch.py <experiment>[:steps] <key> <JSON value> <name> <note>
```

**Evaluation**, on the CPU:

```powershell
.\.venv\Scripts\python.exe -m humanoid_sim.supervise <experiment>[:steps]
.\.venv\Scripts\python.exe -m humanoid_sim.goal_check <experiment>[:steps]
.\.venv\Scripts\python.exe -m humanoid_sim.goal_check --export <export folder>
```

**Export** for Unitree's `g1_ctrl`:

```powershell
.\.venv\Scripts\python.exe -m humanoid_sim.export_g1 <stage or experiment> --out <folder> --command-ranges -0.5 1 0.5 1
```

This writes `policy.onnx` (`obs` [1, 490] → `actions` [1, 29]: normalisation, the actor and the action clip) and
`deploy.yaml`. It then compares onnxruntime with the original policy on 200 real observations.

**Sim2sim.** `scripts/sim2sim/setup.sh` and `run.sh` run in WSL2 Ubuntu (see `docs/sim2sim.md`). Set `ROOT` and
`OWNER` at the top of the scripts for your own WSL user.

**Tests:**

```powershell
.\.venv\Scripts\python.exe -m pytest -q                # CPU tests
.\.venv-fullgpu\Scripts\python.exe -m pytest -q        # adds the GPU tests
```

## Trained policy

The exported candidate 7 (`policy.onnx`, `deploy.yaml`, the run config of its last branch and the same safety
warning) is attached to the GitHub release as `g1-candidate7-policy.zip`. The checkpoints themselves (`runs/`) are
not published.

## License

The project's own code is under the Apache License 2.0 (`LICENSE`). The robot models, three.js, the anatomy kit
and the Unitree-derived files keep their own licenses, which are listed in `THIRD_PARTY.md`.

This is an independent project. It is not affiliated with or endorsed by Unitree Robotics. Unitree and G1 are
trademarks of their owners.
