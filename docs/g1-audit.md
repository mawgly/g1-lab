# G1 integration audit — 2026-10-19 (in progress)

## Verified baseline

The existing workspace is not a Git repository. A trainer, queue worker and HTTP server were confirmed as live processes. They were not stopped. CPU baseline: 22 tests passed across test_humanoid, test_lab, test_lab_http and test_reports. After adding compatibility checks: 28 passed including test_checkpoint_schema.

## Architecture and integration points

- Laboratory.create validates configuration, optionally copies a complete checkpoint and writes experiment/run configuration. The worker serializes jobs using a compute lock and invokes a backend-specific Python environment.
- StageCallback saves snapshots, performs CPU evaluation and refreshes reports. Status uses absolute transitions, while throughput uses added transitions since the start of the session.
- core.save_stage publishes a directory only after policy.zip, normalize.pkl and metadata.json are written. list_stages excludes incomplete stages.
- core.load_stage, evaluate_stage, diagnostics and full-GPU export instantiate Humanoid-v5 directly. They need shared task dispatch before G1 is exposed to users.
- GPUHumanoidVecEnv and TensorHumanoidEnv assume 17 actions, Gym Humanoid observations, reward and reset rules. Adding a G1 XML alone cannot work.
- Full-GPU make_learner restores SB3 optimizer and normalizer, overrides LR, then resets the random seed and starts new simulator states. This is resumed optimization, not bit-exact recovery of the original trajectory.
- Existing rollout/backend differences include Newton on GPU versus PGS in legacy CPU evaluation. G1 evaluation must declare its own solver and task configuration rather than inheriting these silently.

## Implemented protection

checkpoint_schema.py recognizes only the explicitly named legacy Humanoid-v5 format. Unknown environments and unsupported explicit schema versions fail closed. A versioned identity compares robot, task, model revision, observation/action/normalization contracts and control period. Both load_stage and Laboratory.create now check the legacy target before reading weights or creating a branch. G1 dispatch will replace that fixed target when its environment exists.

Tests prove unknown source rejection without creating an experiment, wrong robot rejection before reading weights, incompatible schemas, and legacy roundtrip compatibility. Current schema identifiers are contract names; model-content hashes and ordered joint signatures remain to be implemented with the registry.

## Outstanding audit evidence

- Pinned official G1 assets, license/mesh review, actuator settings and Windows/Warp compatibility.
- Original motion traces for early/best/degraded checkpoints and biomechanical diagnostics.
- Full restore semantics across CPU/hybrid path and TensorPPO, including rollout state and scheduling.
- Browser model geometry assumptions and task switching.
- GPU verification deferred while the user training process owns the compute slot; CPU tests do not establish GPU correctness.

Audit is not yet complete. No G1 training or long experiment was launched.
