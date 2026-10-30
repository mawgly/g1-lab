"""collect.py <stage | experiment:steps | arm_sim run> --out <dir> [--device cuda|cpu] [--worlds 4096] [--steps 1000]
    [--seed 0] [--chunk 50] [--action-noise 0.1 | --sampled | --deterministic]
Transitions of a stage's policy in its own world for the world model (SuperMimic stage 1, humanoid_sim/world_model.py),
for any task with an adapter (humanoid_sim/world_model_adapters, chosen from the stage's robot and task):
- arm_reach (ladder step 0): the SO-101 arm, a lab so101 stage or an arm_sim run folder (runs/so101:
  training_config.json and best_model.zip, raw observations), CPU only; our own simulation, nothing licensed;
- g1_velocity (step 1): G1 standing and walking, the real-robot profile (candidate 8 = exp-28691193718e4d0e:7221510144),
  on cuda the training's CUDA task with its pushes and randomisation; our own simulation;
- g1_getup track_clip (step 2): getting up by tracking one clip (world_model.check_track_config); the clip may be
  BONES-SEED, never published.
--device defaults to cuda where the adapter has a CUDA collector, else cpu. On cuda the CUDA task (the training's
starts, randomization, sensor noise and encoder bias, resets as in training); on cpu one CPU task per world (slow for
the G1; tests and small checks: no sensor noise, no encoder bias, no COM offset, no velocity kicks there). Its actions
(the mode goes into meta.json, action_mode / action_noise):
- --action-noise S (default, S 0.1): the policy's mean (deterministic) plus ε ~ N(0, S²) per joint, clipped to the
  action box: the paper's collection (table A.5, §3.1: a deterministic policy, ε ~ N(0, 0.1²));
- --sampled: sampled as in PPO training with the stage's own σ (stage 570: median 0.44 over the free joints, ~4.4 times
  the paper's ε; the data's one-step changes, and the error scales made of them, grow with it);
- --deterministic: the mean alone.
The logged action is the applied one either way (the G1: box clip, arm perturbation, locked wrists; the arm: ctrl − q).
The stage must be named exactly (a folder or <experiment>:<steps>); a task_config the adapter has no rules for is
refused. Per world and step it keeps:
- state: the raw state before the step (the adapter's layout's RAW: qpos, qvel and, for the G1, the previous applied
  action);
- action: the applied action (what went to the motors; the G1's next state's previous action);
- context: the adapter's ξ of the episode; bias: the encoder bias (observation only; the G1);
- reward (the task's), reward_contact (the adapter's excluded terms, which a state does not give: getting up its
  self_contact, impact and sole_impact, walking its effort and self_contact), terminated, truncated, bad (a non-finite
  state after the step);
- the adapter's fields (getting up: clock, yaw, offset, kind, steps; walking: the observed and requested command, the
  heading target, the yaw-drift EMA before the step, the gait clock and steps after it; the arm: the target, the hold
  counter before, the clipped policy action);
- inputs (walking): the root qvel kick before the physics and the push's Δv over the step;
- contact / contact_next: per contact group (the G1's 7; the arm none) a bit whether a geom of the group touches the
  floor (contact distance ≤ 0) before / after the step; force / force_next: the summed normal force (N).
The state after the step is the next row's state unless the episode ended (then the world was reset): those rows'
following states go into boundary_next (flat index t·worlds + w in boundary_index), the chunk's last into last_next.
Chunks of --chunk steps, chunk_00000.npz … (float32, uncompressed), meta.json last. The data stay in runs/ (e.g.
runs/humanoid_lab/world_model/ladder-<n>-…/data).

4096 G1 worlds × 1000 steps: about 3 GB. Default --out none: give a folder of your own.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from humanoid_sim.world_model_adapters import adapter_for  # noqa: E402
from humanoid_sim.world_model_adapters.contacts import (CONTACT_GROUPS, GROUP_PARTS, bits, contact_groups,  # noqa
                                                        cpu_contacts)
from humanoid_sim.world_model_adapters.getup import GetupTrack  # noqa: E402

HEAD = ('state', 'action', 'context', 'bias', 'reward', 'reward_contact', 'terminated', 'truncated', 'bad')
TAIL = ('contact', 'contact_next', 'force', 'force_next')
FIELDS = HEAD + tuple(field.name for field in GetupTrack.fields) + TAIL  # getting up's, as before the adapters
ACTION_NOISE = .1  # the paper's ε (table A.5)
ACTION_MODES = ('noise', 'sampled', 'deterministic')


def fields_of(adapter):
    """The chunk files' keys of an adapter's data, in order: HEAD, the adapter's fields, inputs (when it has any),
    TAIL; for getting up FIELDS."""
    return HEAD + tuple(field.name for field in adapter.fields) + (('inputs',) if adapter.inputs else ()) + TAIL


@dataclass
class Source:
    """A policy to collect with: its stage (or arm_sim run) folder, robot, task, task_config, metadata and
    policy(mode, noise, seed) → observations → actions (numpy, CPU)."""
    path: Path
    robot_id: str
    task_id: str
    config: dict
    metadata: dict
    policy: object
    steps: int | None = None


def resolve(spec):
    """The Source of a stage folder, <experiment>:<steps> of the lab under runs/ (an experiment without steps is
    refused: its newest stage is often not the one meant, exp-84b0540b's is 652, the world model's stage 570), or an
    arm_sim run folder (training_config.json + best_model.zip; its SO-101 model revision must be the repository's)."""
    path = Path(spec)
    if (path / 'training_config.json').is_file() and (path / 'best_model.zip').is_file():
        from humanoid_sim.world_model_adapters.arm import check_revision
        config = json.loads((path / 'training_config.json').read_text(encoding='utf-8'))
        try:
            check_revision(config)
        except ValueError as error:
            raise SystemExit(str(error))

        def policy(mode, noise, seed):
            from stable_baselines3 import PPO
            return cpu_actor(PPO.load(path / 'best_model.zip', device='cpu'), None, mode, noise, seed)
        return Source(path.resolve(), 'so101', 'arm_reach', {}, config, policy)
    if not (path / 'metadata.json').is_file():
        from humanoid_sim.core import DEFAULT_RUN, list_stages
        from humanoid_sim.lab import Laboratory
        name, _, steps = spec.partition(':')
        run = Laboratory(DEFAULT_RUN).resolve(name)
        if not steps:
            newest = list_stages(run)
            hint = f' (последний: {name}:{int(newest[-1].name[6:])})' if newest else ''
            raise SystemExit(f'укажите этап явно: папка этапа или {name}:<шаги>{hint}')
        path = run / f'stage_{int(steps):012d}'
        if not (path / 'metadata.json').is_file():
            raise SystemExit(f'нет этапа {spec}')
    from humanoid_sim.core import read_json, stage_task_config
    stage = path.resolve()
    metadata = read_json(stage / 'metadata.json')

    def policy(mode, noise, seed):
        return cpu_policy(stage, mode, noise, seed)[0]
    return Source(stage, metadata['robot_id'], metadata['task_id'], stage_task_config(stage), metadata, policy,
                  int(metadata['steps']))


def noisy(mean, noise, low, high, rng):
    """The paper's collection action: mean (worlds, nu) + N(0, noise²), clipped to [low, high] (numpy)."""
    return np.clip(mean + noise * rng.standard_normal(mean.shape), low, high)


def cpu_context(env):
    """world_model.CONTEXT of the CPU getting-up task's episode (GetupTrack.cpu_context)."""
    return GetupTrack().cpu_context(env)


def cpu_raw(env):
    return np.concatenate([env.data.qpos, env.data.qvel, env.previous_action])


class Writer:
    """Rows (one dict of per-world arrays per step) into chunk files of `chunk` steps under `out`, with the following
    states of the rows whose episode ended (boundary) and of the chunk's last row."""

    def __init__(self, out, chunk, fields=FIELDS):
        self.out, self.chunk, self.fields = Path(out), chunk, tuple(fields)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows, self.boundary_index, self.boundary_next = [], [], []
        self.files, self.steps = 0, 0

    def add(self, row, following, boundary):
        """row: fields → (worlds, …) arrays before/at the step; following (worlds, RAW): the state after the step;
        boundary (worlds,) bool: the episode ended (the next row starts another)."""
        t, worlds = len(self.rows), len(following)
        for w in np.flatnonzero(boundary):
            self.boundary_index.append(t * worlds + int(w))
            self.boundary_next.append(following[w])
        self.rows.append(row)
        self.last = following
        self.steps += 1
        if len(self.rows) == self.chunk:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.fields}
        size = self.last.shape[1]
        arrays.update(boundary_index=np.array(self.boundary_index, dtype=np.int64),
                      boundary_next=np.array(self.boundary_next, dtype=np.float32).reshape(-1, size),
                      last_next=self.last.astype(np.float32))
        np.savez(self.out / f'chunk_{self.files:05d}.npz', **arrays)
        self.files += 1
        self.rows, self.boundary_index, self.boundary_next = [], [], []


def _row(adapter, values):
    """One step's row in the chunks' dtypes: HEAD and TAIL as before the adapters, the adapter's fields in theirs."""
    f32 = np.float32
    row = dict(state=values['state'].astype(f32), action=values['action'].astype(f32),
               context=values['context'].astype(f32), bias=values['bias'].astype(f32),
               reward=values['reward'].astype(f32), reward_contact=values['reward_contact'].astype(f32),
               terminated=values['terminated'].astype(bool), truncated=values['truncated'].astype(bool),
               bad=values['bad'].astype(bool))
    for field in adapter.fields:
        row[field.name] = np.asarray(values[field.name]).astype(np.dtype(field.dtype))
    if adapter.inputs:
        row['inputs'] = values['inputs'].astype(f32)
    row.update(contact=values['contact'].astype(np.uint8), contact_next=values['contact_next'].astype(np.uint8),
               force=values['force'].astype(f32), force_next=values['force_next'].astype(f32))
    return row


# --- CPU: one CPU task per world -----------------------------------------------------------------------------------


def _record_exogenous(env):
    """The task's _before_physics hook keeps the step's root qvel and pushes in env.exogenous (Adapter.cpu_inputs)."""
    def hook(pushes):
        env.exogenous = dict(qvel=env.data.qvel[:6].copy(), pushes=None if pushes is None else np.array(pushes))
    env._before_physics = hook


def collect_cpu(envs, policy, steps, out, seed=0, chunk=50, meta=None, adapter=None):
    """Runs `policy` (observations (worlds, obs) → actions (worlds, nu), numpy) in the CPU tasks `envs` (one per world)
    for `steps` control steps, written to `out` (Writer); returns meta (meta.json). adapter: None, getting up (the
    tasks given directly: TrackTerms checks them)."""
    adapter = GetupTrack() if adapter is None else adapter
    started = time.time()
    groups = adapter.cpu_groups(envs[0])
    writer = Writer(out, chunk, fields_of(adapter))
    if adapter.inputs:
        for env in envs:
            _record_exogenous(env)
    observations = np.stack([env.reset(seed=seed + w)[0] for w, env in enumerate(envs)])
    after = [field.name for field in adapter.fields if field.when == 'after']
    before_names = [field.name for field in adapter.fields if field.when == 'before']
    for _ in range(steps):
        pre = [adapter.cpu_contacts(env, groups) for env in envs]
        state = np.stack([adapter.cpu_raw(env) for env in envs])
        context = np.stack([adapter.cpu_context(env) for env in envs])
        bias = np.stack([adapter.cpu_bias(env) for env in envs])
        before = [adapter.cpu_before(env) for env in envs]
        actions = np.asarray(policy(observations), dtype=np.float64)
        row = {key: [] for key in ('reward', 'reward_contact', 'terminated', 'truncated', 'bad', 'touching', 'force',
                                   'action', 'inputs', *after)}
        following = []
        for w, env in enumerate(envs):
            env.exogenous = None
            observation, reward, terminated, truncated, info = env.step(actions[w])
            fields, excluded, applied = adapter.cpu_after(env, info, state[w], actions[w])
            touching, force = adapter.cpu_contacts(env, groups)
            following.append(adapter.cpu_raw(env))
            for key, value in (('reward', reward), ('terminated', terminated), ('truncated', truncated),
                               ('bad', bool(info.get('bad_state', False))), ('touching', touching), ('force', force),
                               ('action', applied), ('reward_contact', excluded), *fields.items()):
                row[key].append(value)
            if adapter.inputs:
                row['inputs'].append(adapter.cpu_inputs(env, state[w]))
            if terminated or truncated:
                observation, _ = env.reset()
            observations[w] = observation
        boundary = np.array(row['terminated']) | np.array(row['truncated'])
        values = {key: np.array(row[key]) for key in ('reward', 'reward_contact', 'terminated', 'truncated', 'bad',
                                                      'action', *after)}
        values.update(state=state, context=context, bias=bias, contact=bits(np.array([p[0] for p in pre])),
                      contact_next=bits(np.array(row['touching'])), force=np.array([p[1] for p in pre]),
                      force_next=np.array(row['force']))
        for name in before_names:
            values[name] = np.array([fields_before[name] for fields_before in before])
        if adapter.inputs:
            values['inputs'] = np.array(row['inputs'])
        writer.add(_row(adapter, values), np.stack(following), boundary)
    writer.flush()
    return _finish(out, writer, len(envs), steps, envs[0], 'cpu', seed, started, meta, adapter)


def _finish(out, writer, worlds, steps, reference, backend, seed, started, meta, adapter):
    from humanoid_sim import world_model as wm
    meta = dict(meta or {})
    numbers = adapter.describe(reference)
    meta.update(version=wm.VERSION, backend=backend, worlds=worlds, steps=steps, seed=seed, chunks=writer.files,
                chunk=writer.chunk, nu=numbers['nu'], raw=adapter.layout().raw, context=list(adapter.context),
                contact_groups=list(adapter.contact_groups), frame_skip=numbers['frame_skip'], dt=numbers['dt'],
                home=numbers['home'], seconds=round(time.time() - started, 1), licence=adapter.licence,
                adapter=adapter.name, inputs=list(adapter.inputs), ladder_step=adapter.ladder.step,
                fields=[dict(name=f.name, dtype=f.dtype, shape=list(f.shape), when=f.when) for f in adapter.fields])
    (Path(out) / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return meta


def cpu_actor(model, norm=None, mode='noise', noise=ACTION_NOISE, seed=0):
    """observations (worlds, obs) → actions (worlds, nu), numpy, of an SB3 policy `model` on its VecNormalize `norm`
    (None: raw observations, an arm_sim run) in an ACTION_MODES mode (noise: the mean plus N(0, noise²) from
    default_rng(seed), clipped to the action space)."""
    if mode not in ACTION_MODES:
        raise ValueError(f'неизвестный режим действий {mode}')
    rng = np.random.default_rng(seed)
    low, high = model.action_space.low, model.action_space.high

    def act(observations):
        seen = observations if norm is None else norm.normalize_obs(observations)
        action = model.predict(seen, deterministic=mode != 'sampled')[0]
        return noisy(action, noise, low, high, rng) if mode == 'noise' else action
    return act


def cpu_policy(stage, mode='noise', noise=ACTION_NOISE, seed=0):
    """(policy, task_config): the stage's SB3 policy on its VecNormalize, on the CPU (cpu_actor)."""
    from humanoid_sim.core import load_stage, stage_task_config
    model, norm = load_stage(stage)
    return cpu_actor(model, norm, mode, noise, seed), stage_task_config(stage)


# --- CUDA: the training's CUDA task ----------------------------------------------------------------------------------


def collecting_env(adapter=None, **settings):
    """The adapter's CUDA task (cuda_base; None: getting up, TensorG1GetupEnv) that, when armed (post = {}), keeps each
    step's state after it before the ended worlds reset: _masked_reset runs at the end of every step with the step's
    done mask, so its first call while armed sees every world after the physics, the reward and the clock; and the
    _before_physics hook (walking) keeps the step's root qvel after the kicks and its pushes in `exogenous`."""
    import torch
    adapter = GetupTrack() if adapter is None else adapter
    from humanoid_sim.world_model_adapters.contacts import contact_groups as groups_of

    class CollectingEnv(adapter.cuda_base):
        post = None
        exogenous = None

        def _masked_reset(self, mask, *args, **kwargs):
            if self.post is not None and not self.post:
                self.post.update(self.snapshot())
                self.post['raw'] = torch.cat((self.qpos, self.qvel, self.previous_action), 1).clone()
                self.post['contact_part'] = self._contact_part()
            return super()._masked_reset(mask, *args, **kwargs)

        def _before_physics(self, pushes):
            if self.post is not None:
                self.exogenous = dict(qvel=self.qvel[:, :6].clone(), pushes=None if pushes is None else pushes.clone())

        def _contact_part(self):
            """(n,) the step's terms a state does not give (the adapter's excluded ones, weighted, negative)."""
            return self.adapter.cuda_excluded(self)

        def snapshot(self):
            """Every world now: contacts per group (touching, force) and the adapter's 'after' fields."""
            touching, force = self._groups()
            return dict(touching=touching, force=force, **self.adapter.cuda_after(self))

        def _groups(self):
            """(touching (n, G) bool, force (n, G) N): the floor contacts of each world's contact groups after the last
            forward (its contacts and constraint forces, decoded again)."""
            groups, n = len(self.adapter.contact_groups), self.n_envs
            if not hasattr(self, '_group_of_geom'):
                ref = self.reference
                self._group_of_geom = torch.as_tensor(groups_of(ref.model, ref.robot_geoms, self.adapter.contact_groups,
                                                                self.adapter.group_parts), device=self.device)
                self._floor_geom = self.adapter.cuda_floor(self)
                self._dist = self.wp.to_torch(self.data.contact.dist)
                self._geoms = self.wp.to_torch(self.data.contact.geom)
                self._slots = torch.arange(self.data.naconmax, device=self.device)
            self.wp.capture_launch(self.contact_force_graph)
            geom = self._geoms.long()
            live = (self._slots < self.nacon) & (geom >= 0).all(dim=1)
            geom = geom.clamp(min=0)
            first, second = geom[:, 0] == self._floor_geom, geom[:, 1] == self._floor_geom
            group = self._group_of_geom[torch.where(first, geom[:, 1], geom[:, 0])]
            counted = live & (first ^ second) & (group >= 0)
            world = self.contact_world.long().clamp(0, n - 1)
            index = torch.where(counted, world * groups + group.clamp(min=0), n * groups)
            force = torch.zeros(n * groups + 1, device=self.device).scatter_add_(
                0, index, torch.where(counted, self.contact_force[:, 0].abs(), 0.))[:-1].view(n, groups)
            touched = torch.where(counted & (self._dist <= 0), index, n * groups)
            count = torch.zeros(n * groups + 1, device=self.device).scatter_add_(
                0, touched, torch.ones_like(self._dist))[:-1].view(n, groups)
            return count > 0, force

        def context(self):
            """(n, context) each world's episode context now."""
            return self.adapter.cuda_context(self)

    CollectingEnv.adapter = adapter
    return CollectingEnv(**settings)


def cuda_actor(policy, norm, mode='noise', noise=ACTION_NOISE, seed=0, device='cuda'):
    """observations (worlds, obs) → actions (worlds, nu), CUDA tensors, of an SB3 policy on a TensorNormalize in an
    ACTION_MODES mode (noise: the mean plus N(0, noise²) from a generator seeded `seed`; run_cuda clips to the box)."""
    import torch
    if mode not in ACTION_MODES:
        raise ValueError(f'неизвестный режим действий {mode}')
    generator = torch.Generator(device=device).manual_seed(seed)

    def act(observations):
        action = policy(norm.observation(observations, update=False), deterministic=mode != 'sampled')[0]
        if mode == 'noise':
            action = action + noise * torch.randn(action.shape, device=action.device, dtype=action.dtype,
                                                  generator=generator)
        return action
    return act


def collect_cuda(stage, worlds, steps, out, seed=0, chunk=50, mode='noise', noise=ACTION_NOISE, meta=None,
                 adapter=None):
    """Runs the stage's policy in `worlds` worlds of the adapter's CUDA task for `steps` control steps (collect_cpu's
    rows)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import read_json, stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.tensor_ppo import TensorNormalize
    config = stage_task_config(stage)
    metadata = read_json(Path(stage) / 'metadata.json')
    adapter = adapter or adapter_for(metadata['robot_id'], metadata['task_id'], config)
    adapter.check('cuda')
    env = collecting_env(adapter, n_envs=worlds, seed=seed, **config)
    saved = metadata.get('training_state')
    if saved:
        env.restore_training_state(saved)
    base = DummyVecEnv([lambda: make_task(metadata['task_id'], metadata['robot_id'], config)])
    export = VecNormalize.load(Path(stage) / 'normalize.pkl', base)
    norm = TensorNormalize(env.observation_space.shape, worlds, env.device, fixed=env.fixed_normalization)
    norm.import_normalizer(export)
    policy = PPO.load(Path(stage) / 'policy.zip', device='cuda').policy
    policy.set_training_mode(False)
    act = cuda_actor(policy, norm, mode, noise, seed + 1, env.device)
    try:
        return run_cuda(env, act, steps, out, seed, chunk, meta)
    finally:
        base.close()


def run_cuda(env, act, steps, out, seed=0, chunk=50, meta=None):
    """Runs `act` (observations (worlds, obs) → actions (worlds, nu), CUDA tensors) in the CUDA task `env`
    (collecting_env, its adapter env.adapter) from its reset for `steps` control steps, written to `out` (Writer);
    returns meta."""
    import torch
    adapter = env.adapter
    started = time.time()
    low = torch.as_tensor(env.action_space.low, device=env.device)
    high = torch.as_tensor(env.action_space.high, device=env.device)
    shifts = torch.arange(len(adapter.contact_groups), device=env.device)
    writer = Writer(out, chunk, fields_of(adapter))
    before_names = [field.name for field in adapter.fields if field.when == 'before']
    after_names = [field.name for field in adapter.fields if field.when == 'after']

    def packed(touching):
        return (touching.long() << shifts).sum(1).to(torch.uint8)
    with torch.no_grad():
        observations = env.reset()
        stream = env.wp.stream_from_torch(torch.cuda.current_stream(env.device))
        for _ in range(steps):
            with env.wp.ScopedStream(stream):  # the contact decode on torch's stream, as the task's own launches
                pre = env.snapshot()
            state = torch.cat((env.qpos, env.qvel, env.previous_action), 1).clone()
            context, bias = env.context(), env.encoder_bias.clone()
            before = adapter.cuda_before(env)
            actions = act(observations)
            env.post, env.exogenous = {}, None
            observations, reward, terminated, truncated, _ = env.step(actions.clamp(low, high))
            post, env.post = env.post, None
            following = post['raw']
            action = following[:, env.qpos.shape[1] + env.qvel.shape[1]:]
            bad = ~torch.isfinite(following).all(1)
            row = dict(state=state, action=action, context=context, bias=bias, reward=reward,
                       reward_contact=post['contact_part'], terminated=terminated, truncated=truncated, bad=bad,
                       contact=packed(pre['touching']), contact_next=packed(post['touching']), force=pre['force'],
                       force_next=post['force'], **{name: before[name] for name in before_names},
                       **{name: post[name] for name in after_names})
            if adapter.inputs:
                row['inputs'] = adapter.cuda_inputs(env, state)
            row = {key: value.cpu().numpy() for key, value in row.items()}
            writer.add(_row(adapter, row), following.cpu().numpy(), row['terminated'] | row['truncated'])
    writer.flush()
    meta = dict(meta or {})
    try:  # Warp buffer overflows or a non-finite state: kept in meta (bad rows are marked), not lost with the data
        env.check_health()
        meta['health'] = 'ok'
    except RuntimeError as error:
        meta['health'] = str(error)
    return _finish(out, writer, env.n_envs, steps, env.reference, 'cuda', seed, started, meta, adapter)


def main():
    # Redirected on Windows, stdout is cp1251, which has no × or →.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Сбор переходов политики этапа для модели мира.')
    parser.add_argument('stage', help='папка этапа, <эксперимент>:<шаги> или папка прогона arm_sim (runs/so101)')
    parser.add_argument('--out', required=True, help='папка данных (в runs/, не в git)')
    parser.add_argument('--worlds', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=1000, help='шагов управления на мир')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default=None,
                        help='по умолчанию cuda, если у задачи есть сборщик на CUDA, иначе cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=50, help='шагов в одном файле')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--action-noise', type=float, default=None, metavar='S',
                       help=f'среднее политики + N(0, S²), как в статье (по умолчанию {ACTION_NOISE})')
    modes.add_argument('--sampled', action='store_true', help='сэмплировать с σ политики, как в обучении PPO')
    modes.add_argument('--deterministic', action='store_true', help='среднее политики без шума')
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1 or args.chunk < 1:
        raise SystemExit('--worlds, --steps и --chunk должны быть положительными')
    if args.action_noise is not None and not args.action_noise > 0:
        raise SystemExit('--action-noise должен быть положительным (без шума — --deterministic)')
    mode = 'sampled' if args.sampled else 'deterministic' if args.deterministic else 'noise'
    noise = (ACTION_NOISE if args.action_noise is None else args.action_noise) if mode == 'noise' else None
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'папка {out} не пуста')
    source = resolve(args.stage)
    try:
        adapter = adapter_for(source.robot_id, source.task_id, source.config)
        device = args.device or ('cuda' if 'cuda' in adapter.backends else 'cpu')
        adapter.check(device)
    except ValueError as error:
        raise SystemExit(str(error))
    print(f'этап {source.path}' + (f' ({source.steps} шагов)' if source.steps is not None else '')
          + f', адаптер {adapter.name} (ступень {adapter.ladder.step}), действия: {mode}'
          + (f' {noise}' if noise else ''))
    if adapter.licence:
        print(adapter.licence)
    meta = dict(stage=str(source.path), stage_steps=source.steps, robot_id=source.robot_id, task_id=source.task_id,
                model_revision=source.metadata.get('model_revision'), action_mode=mode, action_noise=noise,
                deterministic=mode == 'deterministic', task_config=source.config)
    if device == 'cuda':
        meta = collect_cuda(source.path, args.worlds, args.steps, out, args.seed, args.chunk, mode, noise, meta,
                            adapter)
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        policy = source.policy(mode, noise, args.seed + 1)
        envs = [adapter.cpu_env(source.config) for _ in range(args.worlds)]
        meta = collect_cpu(envs, policy, args.steps, out, args.seed, args.chunk, meta, adapter)
    print(f'{meta["worlds"]} миров × {meta["steps"]} шагов за {meta["seconds"]} с → {out}')
    if meta.get('health') not in (None, 'ok'):
        print(f'внимание: сбор данных не здоров: {meta["health"]}')


if __name__ == '__main__':
    main()
