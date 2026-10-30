---
name: remote-training
description: Train a G1 lab branch on a rented Linux GPU server (immers.cloud or similar) alongside the home GPU, and bring its stages back without mixing models. Use when Pavel rents or mentions a GPU server, when a second branch should train in parallel with the home one, or when choosing which rented GPU to take.
---

# Remote training on a rented GPU server

The home lab (`runs/humanoid_lab`, its branch tree) is the only source of truth. A server is a worker: a branch is
created at home, sent, trained there, and its stages come back into the same local branch folder. Tool:
`scripts/remote/remote.py` (commands below), environment `scripts/remote/setup_linux.sh`. First test 27.10.2026:
immers.cloud, RTX 2080 Ti 11 GB, Ubuntu 26.04 CUDA image, user `ubuntu`, code in `~/sim`.

## A new server

1. Pavel creates it and gives the IP and the key pair. A provider-generated pair comes as a private `.pem` in
   Downloads (a PEM file whose first line names an RSA private key); a pasted `ssh-rsa …` line is only the public half and does not log
   in. Use the `.pem` once to add our own key, then log in only with that key:
   `ssh -i <pem> ubuntu@<ip> "echo '<~/.ssh/immers_gpu_ed25519.pub>' >> ~/.ssh/authorized_keys"`. Never type passwords.
2. Point the `Host gpu1` entry in `~/.ssh/config` at the new IP (key `~/.ssh/immers_gpu_ed25519`, user ubuntu).
3. Check it: `ssh gpu1 'nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv'`.
   - torch cu130 (default) needs Turing (sm_75) or newer and driver ≥ 580;
   - Pascal or Volta need `--cuda cu128`.

## Running a branch there

1. Create the branch locally **without queueing it** (`lab.create(config)` + `annotate`; `task_branch.py` queues by
   itself, so copy its config logic into a scratch script as in the 27.10 session). Set `envs` / `batch_size` for
   the card's memory:
   - the tracking task at 4096 worlds needs ~11.5 GB;
   - on 11 GB use 2048 worlds with batch 12288, which keeps 4 minibatches.
2. Commit: `up` sends HEAD by `git archive`, so uncommitted edits stay home.
3. `.venv\Scripts\python.exe scripts\remote\remote.py up <exp> [--host gpu1] [--steps N] [--cuda cu128]`. It:
   - sends the code;
   - builds `.venv-fullgpu` once;
   - sends the branch's newest stage and the data its task_config reads (pose bank, track / settle clips, human
     starts, style libraries);
   - restores the Warp kernel cache for that GPU architecture from `.tools/remote/warp-cache-sm<NN>.tar`, if one exists;
   - starts `cli train` under nohup;
   - marks the local branch (`remote.json`: host, commit, stage; a `status.json`), so the home queue refuses it.

   The first start on a new architecture compiles Warp kernels for several minutes, and status stays "starting"
   meanwhile.
4. `remote.py status <exp>` gives the GPU, the trainer process, state / steps / steps_per_second and the newest
   stages. `remote.py stop <exp>` asks the trainer to save a stage and exit.
5. `remote.py fetch <exp>` brings back the stages the local branch lacks, plus status, training.log and SB3 logs.
   The first fetch from an architecture also keeps the server's Warp cache. Evaluate fetched stages here on the CPU,
   as with every stage.
6. `remote.py clean [--all]` after the work. It removes the licensed data from the server (BONES-SEED, CMU and
   LAFAN1 clips and libraries may not be redistributed). Do this before Pavel deletes the server; `--all` also
   removes the runs.

## Not mixing models

- **One branch trains in one place.** `lab.enqueue` refuses a branch that has a status.json ("уже выполнялся"). To go
  on elsewhere, create a continuation branch from a fetched stage (`source_id`, `source_stage`; the tree shows parent
  → continuation). 27.10: exp-961de6d6d7f246a5 trained on the server, exp-bfe9469a36fe43b9 continues at home.
- **Checkpoint identity guards loads.** Each stage's metadata carries robot, task, model revision, observation /
  action schemas and control_dt, and `validate_resume` refuses a mismatch on every load and branch. Code changes
  outside the identity (reward formulas) are covered only by the commit in `remote.json`: keep the server on the same
  commit as the home branch it is compared with.
- **Same versions on both machines.** `requirements-gpu.txt` is pinned and both run Warp float32 GPU physics;
  evaluation runs only at home on CPU, so stages from both machines are judged by one measure.
- **Division of labour.** The home GPU keeps the main line. The server takes parallel single-change branches (another
  clip, an A/B variant). The hourly check fetches its stages and compares them in the same tree.

## Choosing a GPU (immers.cloud prices, 27.10.2026)

- Measured: RTX 2080 Ti at 2048 worlds gives 13.7 k steps/s using 5 GB (25 ₽/h, ≈ 0.5 ₽ per million steps). Home
  RTX 4080 Super at 4096 worlds gives 28–30 k.
- Recommended: **RTX 4090 `rtx4090-1.8.16.80`**, 83 ₽/h. 24 GB takes the full 4096 worlds. Estimated (not yet
  measured) ~1.3× home and ~3× the 2080 Ti, at about the same cost per step: a 1 B-step branch ≈ 7 h ≈ 580 ₽.
  8 vCPU and 16 GB RAM are enough (the test used 3 GB).
- Worse picks:
  - RTX 3090: slower for nearly the same price per hour;
  - RTX 3080: 10 GB is too small for 4096 worlds;
  - H100: 4.4× the price, barely faster for this physics;
  - T4 / A2: slow.
- Billing is per second: hourly use beats a monthly rate unless the server runs all the time. Prepaying 1–6 months
  gives 10–25 % off. Measure steps/s at the first start and update this table.

## Pitfalls seen

- Windows-only code: `core.RunLock` falls back to `fcntl` and `night.SleepGuard` does nothing off Windows. The lab
  worker, web server and probes stay Windows-only, so run `cli train` directly (what `up` does).
- `ssh` in a background loop dies with exit 255 when the server goes away; the result already printed still counts.
- Licences: training on our own rented server is not publishing, but nothing licensed stays there after the run
  (`clean`). Policies trained on LAFAN1 data still never go into client packages.
