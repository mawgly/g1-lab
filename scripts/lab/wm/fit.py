"""fit.py <data dir> --out <dir> [--minutes M] [--max-steps N] [--device cuda|cpu] [--hidden 2048,2048,2048]
    [--batch N|auto] [--speed auto|plain|graph|compile] [--horizons 8,32] [--holdout 0.1] [--eval-windows 50000]
    [--eval-windows-32 12500] [--eval-stride 4] [--evaluate-only] [--resume] [--checkpoint-minutes 10]
    [--curve-windows 8192]
Fits the world model (humanoid_sim/world_model.py) to collect.py's transitions and judges it on held-out worlds
(SuperMimic stage 1), for any task with an adapter (humanoid_sim/world_model_adapters; the data's meta.json names it,
data without a name are getting up). --minutes, --hidden and --batch default to the adapter's for the device (getting up
60 min, 2048³, 2048; walking on cuda 60 min, 2048³, auto, on cpu 10 min, 512³, 512; the arm 10 min, 512³, 512).

Training: windows of world_model.WINDOW (8) steps from the training worlds (all but the last --holdout share), started
at any step; window_loss (8-step autoregressive rollout on the logged actions and per-step inputs, Σ of the steps' mean
squared state error in error_scale units plus BCE of the termination; steps after an episode boundary masked); Adam
2e-4, --batch windows a step (BATCHES_PER_UPDATE of them make the paper's update), gradient norm 1. The termination head
reads the adapter's goal features of each predicted state (getting up: TrackTerms' margins against the reference at
its logged clock; walking: the fall's margins; the arm: the tip's distance and the hold counter). --speed (humanoid_sim/
world_model_speed.py): on cuda the whole step is one CUDA graph (auto, graph; compile adds torch.compile, Linux only),
on the CPU the eager loop (plain), which is the bit-for-bit reference; the windows are drawn the same way in every mode.
--batch auto picks 8192, 4096 or 2048 windows by the GPU's free memory (cuda only).

Checkpoints: every --checkpoint-minutes of training (evaluations not counted) a quick evaluation on --curve-windows
held-out windows (one row of report.json's `curve`, also curve.json) and out/model.pt + out/train_state.pt (Adam, the
window sampler, steps, seconds, the curve, the batch). --resume goes on from them, with the saved batch (another
explicit --batch is refused, auto is not chosen again); gpu_busy is this session's replay time over its seconds;
--minutes and --max-steps count the whole run, the steps before the interruption included.

Report (report.json and the table printed), on windows of the held-out worlds every --eval-stride steps, each number
beside the same for standing still (persistence: ŝ_j = s_0, the window's first state, at every step):
- the 8-step state error: the mean over the steps j = 1..8 and the coordinates of the squared error divided by a
  scale. Three scales, the same errors: «разброс» (the headline): each coordinate's spread (std) in the data (pelvis
  height, joints, velocities), for xy and the rotation, which have no spread of their own, their change over 8 steps;
  «8 шагов»: every coordinate's change over 8 steps; «шаг»: its one-step change, the loss's units (error_scale). The
  paper (Fig. 3c: 0.007–0.011) gives "the state only, normalized by a predefined nominal error, measured online on
  policy rollouts", without the nominal error, so no scale here is theirs and their numbers are not comparable. skill
  = 1 − MSE_model / MSE_persist in the same units (1 perfect, 0 no better than standing still).
- the same over --horizons beyond 8 (32: the horizon SuperMimic trains on; report['horizon_32']) on a grid of its own
  (--eval-windows-32 windows), «разброс» and the bodies' scales of the 8-step pass, so the horizons compare, «N шагов»
  its own N-step change; reported, not judged.
- by group: root (position, rotation; a fixed base has none), joints, velocities, bodies (the adapter's tracked bodies
  by Kinematics: positions, orientations, linear and angular velocities, each over its 8-step change), and in physical
  units (m, rad, m/s, rad/s; RMS at j = 8 and over all j).
- per the adapter's phases (getting up: the rise, the clip clock ≤ the clip's last frame, and the hold after it;
  walking: by the requested command, standing, walking straight, walking along an arc, turning in place, side steps;
  the arm: none) and over all. Coordinates of joints the task holds fixed (Adapter.held: the arm's gripper) are left
  out of every error number (their spread is the floor, so noise there would decide it); each set also gives the
  deepest step j a window reaches and the windows alive to the last step, and a horizon no window lives to is n/a
  (report['horizon_<k>']['reached'] false: the arm's episodes end on success after ~10 steps).
- the same on the windows with a contact change (a contact group's floor force crossing --contact-force, 10 N, between
  a step's state and the next: collect.py's force / force_next) and per family, on the calm windows, and the adapter's
  own sets (walking: pushed_windows, with a kick or a push in it).
- the reward by the task's formulas from states (the adapter's Terms: all but the excluded terms), without the terms
  that agree by construction (getting up action_rate, walking action_change, the arm its action term: the logged action
  is the same in prediction and truth), of the predicted states against the same of the true ones: Pearson r, MAE,
  RMSE, beside standing still's; skill = 1 − MSE_model / MSE_persist and the anomaly correlation r(r_model − r_persist,
  r_true − r_persist), over every step, the contact-change steps, per j; and the check that the formula (the agreed
  terms included) of the true states equals the simulator's reward minus its excluded terms (max, mean, p99, the share
  above 1e-3).
- the termination head: BCE beside a constant at the base rate, AUROC, AP, recall and precision at 0.5; and the task's
  termination rule on the true following states against the logged labels (the data's own check) and on the predicted
  ones (the analytic rule as a predictor).
- collect.py's meta health: anything but 'ok' (a Warp buffer overflow, a non-finite state) is warned about and kept.
Thresholds (ours, not the paper's), report['ladder'], one gate per ladder step (Adapter.ladder):
- step 0, the arm: 8-step «разброс» ≤ 0.02 over all windows without the gripper, skill ≥ 0.9, reward anomaly r ≥ 0.9
  on all steps; AUROC shown, not judged (the head reads the logged hold counter, which decides the termination);
- step 1, walking: «разброс» ≤ 0.02 over all windows, skill ≥ 0.8, reward anomaly r ≥ 0.9 on contact-change steps,
  AUROC ≥ 0.9 with at least 20 falls (else not judged);
- step 2, getting up (stage 1's threshold, Pavel 27.10): the rise's «разброс» ≤ 0.02 and the reward's anomaly
  correlation on the rise's contact-change steps ≥ 0.9; skill and AUROC shown.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from humanoid_sim.world_model_adapters import adapter_of_meta  # noqa: E402
from humanoid_sim.world_model_adapters.getup import FAMILIES, GetupTrack  # noqa: E402,F401

NORMALIZATION_ROWS = 200_000
SCALE_WINDOWS = 20_000  # training windows the report's scales come from
# N: a contact group (collect.py's contact groups) is on the floor while its summed normal force is above this. The
# geometric bits (contact distance ≤ 0) flicker: the 5 mm sole spheres touch and leave within a step, so in a first
# check of stage 570 72 % of the steps had a "change"; SELF_CONTACT_FORCE's 10 N, as the task counts a contact.
CONTACT_FORCE = 10.
THRESHOLD_ERROR, THRESHOLD_REWARD = .02, .9  # getting up's (GetupTrack.ladder)
CHECKPOINT_MINUTES = 10.
CURVE_WINDOWS = 8192
HORIZONS = (8, 32)
LONG_WINDOWS = 12_500  # windows of a horizon beyond 8
COORDINATES_TOP = 8  # the error coordinates the report names first
PHASES = ('rise', 'hold', 'all')  # getting up's
WINDOW_SETS = ('all', 'contact_windows', 'contact_windows_hands', 'contact_windows_knees', 'contact_windows_feet',
               'calm_windows')  # getting up's
SCALES = ('spread', 'motion', 'loss')
MEASURE_SEED = 12345  # --batch auto's memory probe draws its windows from a generator of its own


def load(data_dir):
    """(meta, arrays) of collect.py's folder: every chunk's fields concatenated over time (T, worlds, …), next (T,
    worlds, RAW) the state after each step (the next row's, or boundary_next / last_next where it is not)."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / 'meta.json').read_text(encoding='utf-8'))
    files = sorted(data_dir.glob('chunk_*.npz'))
    if not files:
        raise SystemExit(f'нет данных в {data_dir}')
    chunks = [np.load(path) for path in files]
    keys = [key for key in chunks[0].files if key not in ('boundary_index', 'boundary_next', 'last_next')]
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    state = arrays['state']
    total, worlds = state.shape[:2]
    following = np.empty_like(state)
    following[:-1] = state[1:]
    following[-1] = chunks[-1]['last_next']
    flat = following.reshape(total * worlds, -1)
    first = 0
    for chunk in chunks:
        if len(chunk['boundary_index']):
            flat[first * worlds + chunk['boundary_index']] = chunk['boundary_next']
        first += len(chunk['state'])
    arrays['next'] = following
    return meta, arrays


def reference_task(meta):
    """The CPU task of the data's task_config (the adapter's formulas' constants), None without a task_config; a
    task_config the adapter does not know is refused."""
    return adapter_of_meta(meta).reference(meta)


def health_warning(meta):
    """None when collect.py's meta health is 'ok' or absent (the CPU collector has none), else the warning."""
    health = meta.get('health')
    if health is None or health == 'ok':
        return None
    return f'внимание: сбор данных не здоров ({health}); отчёт помечен'


class Data:
    """The arrays as torch tensors on `device` (raw states, actions, contexts, the step's labels and the adapter's
    fields and inputs) and the windows' gathering."""

    BASE = ('state', 'next', 'action', 'context', 'terminated', 'truncated', 'bad', 'reward', 'reward_contact', 'force',
            'force_next')
    TENSORS = BASE + tuple(field.name for field in GetupTrack.fields)  # getting up's

    def __init__(self, arrays, device, holdout, adapter=None):
        import torch
        adapter = GetupTrack() if adapter is None else adapter
        self.torch = torch
        self.device = device
        self.values = {}
        keys = self.BASE + tuple(field.name for field in adapter.fields) + (('inputs',) if adapter.inputs else ())
        for key in keys:
            value = arrays[key]
            dtype = {np.bool_: torch.bool, np.uint8: torch.uint8}.get(value.dtype.type)
            tensor = torch.from_numpy(value.astype(np.int64) if value.dtype.kind in 'iu' and dtype is None else value)
            self.values[key] = tensor.to(device)
        self.total, self.worlds = arrays['state'].shape[:2]
        held = max(1, int(round(self.worlds * holdout)))
        if held >= self.worlds:
            raise SystemExit('нужно больше миров: отложенная часть съела все')
        self.train_worlds = torch.arange(self.worlds - held, device=device)
        self.held_worlds = torch.arange(self.worlds - held, self.worlds, device=device)
        self.boundary = self.values['terminated'] | self.values['truncated'] | self.values['bad']
        self.history = 1  # the model's (fit sets it): window adds the history − 1 states before each start

    def window(self, starts, worlds, k, device, keys=None):
        """A batch of k-step windows at (starts (n,), worlds (n,)): dict of tensors on `device`: state and context at
        the start, the step fields (n, k, …), all of them or only `keys` (and bad, which alive needs), and alive."""
        torch = self.torch
        t = starts[:, None] + torch.arange(k, device=starts.device)
        w = worlds[:, None].expand(-1, k)
        v = self.values
        steps = [key for key in v if key not in ('state', 'context', 'truncated')
                 and (keys is None or key in keys or key == 'bad')]
        batch = dict(state=v['state'][starts, worlds], context=v['context'][starts, worlds],
                     **{key: v[key][t, w] for key in steps})
        boundary = self.boundary[t, w]
        # alive_j: transition j is in the window's first episode and its following state is finite.
        before = torch.cat((torch.zeros_like(boundary[:, :1]), boundary[:, :-1]), 1).cumsum(1) > 0
        batch['alive'] = ~before & ~batch['bad']
        if self.history > 1:
            batch['past'] = self.past(starts, worlds)
        return {key: value.to(device) for key, value in batch.items()}

    def past(self, starts, worlds):
        """(n, history − 1, RAW) the raw states before each window's start, oldest first; before its episode's first
        state (or the data's) the earliest one of the episode repeats, as a controller fills its history at a reset."""
        torch = self.torch
        current, valid, picked = starts, torch.ones_like(starts, dtype=torch.bool), []
        for j in range(1, self.history):
            candidate = (starts - j).clamp(min=0)
            valid = valid & (starts - j >= 0) & ~self.boundary[candidate, worlds]
            current = torch.where(valid, candidate, current)
            picked.append(current)
        index = torch.stack(picked[::-1], 1)
        return self.values['state'][index, worlds[:, None]]

    def sample(self, n, k, generator):
        torch = self.torch
        worlds = self.train_worlds[torch.randint(len(self.train_worlds), (n,), generator=generator,
                                                 device='cpu').to(self.device)]
        starts = torch.randint(self.total - k + 1, (n,), generator=generator, device='cpu').to(self.device)
        return starts, worlds


def past_of(layout, batch):
    """The canonical earlier states of a window batch (Data.past), None for a model without history."""
    return layout.from_raw(batch['past']) if 'past' in batch else None


def goal_function(terms, batch):
    """Step j's goal features of a predicted state (the adapter's Terms.goal: getting up TrackTerms.margins at the
    logged clock), None without terms. No gradient flows through them: the head learns to read them, the dynamics
    learn from the state error."""
    if terms is None:
        return None
    return terms.goal(batch)


def normalize(model, data, terms, layout, device, generator, adapter=None):
    """fit_normalization on up to NORMALIZATION_ROWS training transitions inside episodes."""
    from humanoid_sim import world_model as wm
    adapter = GetupTrack() if adapter is None else adapter
    starts, worlds = data.sample(min(NORMALIZATION_ROWS, data.total * len(data.train_worlds)), 1, generator)
    batch = data.window(starts, worlds, 1, device)
    keep = batch['alive'][:, 0] & ~(batch['terminated'][:, 0])
    state = layout.from_raw(batch['state'][keep])
    following = layout.from_raw(batch['next'][keep, 0])
    goal = None
    if terms is not None:
        goal = terms.goal_rows(following, {key: batch[key][keep, 0] for key in adapter.goal_keys})
    inputs = batch['inputs'][keep, 0] if 'inputs' in batch else None
    past = past_of(layout, batch)
    wm.fit_normalization(model, state, batch['action'][keep, 0], batch['context'][keep], following, goal,
                         inputs=inputs, past=None if past is None else past[keep])
    return int(keep.sum())


def training_keys(adapter):
    """The window fields a training step reads: the targets, actions, labels, the goal's fields and the inputs."""
    return ('next', 'action', 'terminated', 'bad', *adapter.goal_keys, *(('inputs',) if adapter.inputs else ()))


def train(model, data, terms, device, minutes, max_steps, batch_size, learning_rate, seed, log, resume=None,
          checkpoint=None, checkpoint_seconds=None, adapter=None, speed='plain', info=None, window=None):
    """Adam steps on sampled windows until `minutes` (None: no limit) or `max_steps` (None: no limit) of the whole run;
    resume: train_state.pt's dict (Adam, the sampler's generator, steps and seconds so far) or None; checkpoint(steps,
    seconds, optimizer, generator) is called every checkpoint_seconds of training (its own time not counted). speed:
    world_model_speed.speed_mode's request; info (a dict or None) gets the mode, its reason and the timing. Returns
    (steps, seconds, the last logged losses, optimizer, generator). window: the training windows' steps (None:
    world_model.WINDOW)."""
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_speed import GraphStep, IndexFeed, WARMUP_STEPS, speed_mode
    adapter = GetupTrack() if adapter is None else adapter
    info = {} if info is None else info
    k = window or wm.WINDOW
    mode, reason = speed_mode(device, speed)
    generator = torch.Generator().manual_seed(seed)
    if mode == 'plain':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, capturable=True, foreach=True)
    steps, seconds, last = 0, 0., {}
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        generator.set_state(resume['generator'])
        steps, seconds = int(resume['steps']), float(resume['seconds'])
    resumed = seconds  # this session's training seconds are seconds − resumed (gpu_busy: its replays alone)
    # A state saved in the other mode (load_state_dict takes its groups' settings): this mode's capturable flag and the
    # step counters where it keeps them (the device for a capturable Adam, the CPU otherwise; a no-op within a mode).
    for group in optimizer.param_groups:
        group['capturable'] = mode != 'plain'
        group['lr'] = learning_rate  # a resumed run may anneal: --lr below the saved one (the default is the same)
    for parameter, saved in optimizer.state.items():
        if torch.is_tensor(saved.get('step')):
            saved['step'] = saved['step'].to(dtype=torch.float32, device=parameter.device if mode != 'plain' else 'cpu')
    layout = model.layout
    keys = training_keys(adapter)
    every = max(1, wm.BATCHES_PER_UPDATE * 10)
    next_checkpoint = None if checkpoint is None or checkpoint_seconds is None else seconds + checkpoint_seconds
    started = time.time()
    runner = None
    if mode != 'plain':
        feed = IndexFeed(data, batch_size, k, generator, device)

        def loss():
            batch = data.window(feed.starts, feed.worlds, k, device, keys)
            state, targets = layout.from_raw(batch['state']), layout.from_raw(batch['next'])
            return wm.window_loss(model, state, batch['action'], batch['context'], targets, batch['terminated'],
                                  batch['alive'], goal_function(terms, batch), inputs=batch.get('inputs'),
                                  past=past_of(layout, batch))[:2]
        runner = GraphStep(model, loss, optimizer, feed, wm.GRAD_CLIP, compile=mode == 'compile')

    def elapsed():
        return seconds + time.time() - started
    while (max_steps is None or steps < max_steps) and (minutes is None or elapsed() < 60 * minutes):
        if runner is None:
            batch = data.window(*data.sample(batch_size, k, generator), k, device, keys)
            state = layout.from_raw(batch['state'])
            targets = layout.from_raw(batch['next'])
            loss, parts, _, _ = wm.window_loss(model, state, batch['action'], batch['context'], targets,
                                               batch['terminated'], batch['alive'], goal_function(terms, batch),
                                               inputs=batch.get('inputs'), past=past_of(layout, batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), wm.GRAD_CLIP)
            optimizer.step()
        else:
            runner.step()
            if runner.compile_error is not None and 'compile_error' not in info:
                info['compile_error'] = runner.compile_error
                log(f'torch.compile недоступен: {runner.compile_error}; считаю графом')
            if runner.fallback is not None and 'fallback' not in info:
                info['fallback'] = runner.fallback
                log(f'ускорение недоступно: {runner.fallback}; считаю без него')
        steps += 1
        if steps % every == 0 or steps == 1:
            last = ({key: float(value) for key, value in parts.items()} if runner is None else runner.losses())
            log(f'шаг {steps}: mse {last["mse"]:.4f}, bce {last["bce"]:.4f}, {elapsed():.0f} с')
        if next_checkpoint is not None and elapsed() >= next_checkpoint:
            if runner is not None:
                torch.cuda.synchronize()
            seconds = elapsed()
            checkpoint(steps, seconds, optimizer, generator)
            started = time.time()  # the checkpoint's evaluation is not training time
            next_checkpoint = seconds + checkpoint_seconds
    if runner is not None:
        torch.cuda.synchronize()
    seconds = elapsed()
    session = seconds - resumed
    # speed: the mode the steps ran in (a failed compile ran the graph, a failed capture eager steps), speed_requested
    # what speed_mode gave.
    effective = mode
    if runner is not None and runner.fallback is not None:
        effective = 'plain'
    elif runner is not None and mode == 'compile' and not runner.compiled:
        effective = 'graph'
    info.update(speed=effective, speed_requested=mode, speed_reason=reason,
                steps_per_second=steps / seconds if seconds else None, session_seconds=session,
                warmup_steps=0 if runner is None else min(WARMUP_STEPS, runner.done))
    if runner is not None:
        replay = runner.timing()
        info.update(replay_ms=replay, gpu_busy=replay / 1000 / session if session > 0 else None)
        if runner.compile_error is not None:
            info['compile_error'] = runner.compile_error
    return steps, seconds, last, optimizer, generator


def _pearson(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _reward_stats(predicted, true, persist):
    """The predicted rewards (without the agreed terms) against the true ones, beside standing still's."""
    if not len(true):
        return dict(count=0)
    error, still = predicted - true, persist - true
    mse, mse_persist = float((error ** 2).mean()), float((still ** 2).mean())
    return dict(count=int(len(true)), r=_pearson(predicted, true), mae=float(np.abs(error).mean()),
                rmse=math.sqrt(mse), std_true=float(true.std()), mean_true=float(true.mean()),
                r_persist=_pearson(persist, true), mae_persist=float(np.abs(still).mean()),
                rmse_persist=math.sqrt(mse_persist), skill=1 - mse / mse_persist if mse_persist else None,
                anomaly_r=_pearson(predicted - persist, true - persist))


def _auroc(score, label):
    """The area under the ROC curve of scores (m,) for bool labels (m,), ties ranked by their mean; None without both
    classes."""
    positives = int(label.sum())
    negatives = len(label) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind='mergesort')
    _, inverse, counts = np.unique(score[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = np.empty(len(score))
    ranks[order] = ((ends - counts + 1 + ends) / 2)[inverse]
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(score, label):
    """The average precision of scores (m,) for bool labels (m,): the mean precision at each positive's rank."""
    positives = int(label.sum())
    if not positives:
        return None
    ordered = label[np.argsort(-score, kind='mergesort')]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def _rule_stats(said, label):
    """A bool predictor (m,) against bool labels (m,): agreement, misses, false ends, recall, precision."""
    return dict(agree=float((said == label).mean()) if len(label) else None, missed=int((label & ~said).sum()),
                extra=int((said & ~label).sum()),
                recall=float((said & label).sum() / label.sum()) if label.any() else None,
                precision=float((said & label).sum() / said.sum()) if said.any() else None)


def _flat(batch, n, k, adapter):
    """The window's fields flattened over its steps (n·k, …) for the adapter's Terms: its fields, the action, the
    context (repeated per step) and the inputs."""
    flat = {field.name: batch[field.name].reshape(n * k, *batch[field.name].shape[2:]) for field in adapter.fields}
    flat['action'] = batch['action'].reshape(n * k, -1)
    flat['context'] = batch['context'][:, None].expand(n, k, -1).reshape(n * k, -1)
    if 'inputs' in batch:
        flat['inputs'] = batch['inputs'].reshape(n * k, -1)
    return flat


def evaluate(model, data, terms, device, windows, stride, batch_size=4096, contact_force=CONTACT_FORCE, adapter=None,
             k=None, base_scales=None):
    """The report's numbers on the held-out worlds' windows of k steps (WINDOW by default; see the module docstring),
    a dict. base_scales: the 8-step pass's 'spread' and 'body' scales for a longer horizon (its own 'motion')."""
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.tasks import g1_tensor_math as task
    adapter = GetupTrack() if adapter is None else adapter
    layout, k = model.layout, k or wm.WINDOW
    starts = torch.arange(0, data.total - k + 1, stride, device=data.device)
    grid_t, grid_w = torch.meshgrid(starts, data.held_worlds, indexing='ij')
    grid_t, grid_w = grid_t.flatten(), grid_w.flatten()
    if len(grid_t) > windows:
        pick = torch.randperm(len(grid_t), generator=torch.Generator().manual_seed(0))[:windows].to(data.device)
        grid_t, grid_w = grid_t[pick], grid_w[pick]
    phase_keys = [key for key, _ in adapter.phases]
    extra_sets = []
    collected = {key: [] for key in ('alive', 'error', 'standing', 'body', 'body_persist', 'reward_pred',
                                     'reward_true', 'reward_persist', 'reward_full', 'reward_sim', 'term_prob',
                                     'term_label', 'rule_true', 'rule_pred', 'change', 'families',
                                     *(f'phase_{key}' for key in phase_keys))}

    def body_errors(robot, truth):
        return torch.stack(((robot[0] - truth[0]).norm(dim=-1), task.rotation_angle(truth[1], robot[1]),
                            (robot[2] - truth[2]).norm(dim=-1), (robot[3] - truth[3]).norm(dim=-1)), -1)

    def tracking(components):  # the reward without the terms that agree by construction
        return sum(value for key, value in components.items() if key not in adapter.agreed)
    model.eval()
    with torch.no_grad():
        # The scales from training windows' true k-step changes.
        count = min(SCALE_WINDOWS, (data.total - k + 1) * len(data.train_worlds))
        sample = data.window(*data.sample(count, k, torch.Generator().manual_seed(7)), k, device)
        scales = _scales(layout, layout.from_raw(sample['state']), torch.nan_to_num(layout.from_raw(sample['next'])),
                         sample['alive'], terms)
        for first in range(0, len(grid_t), batch_size):
            part = slice(first, first + batch_size)
            batch = data.window(grid_t[part], grid_w[part], k, device)
            state = layout.from_raw(batch['state'])
            targets = torch.nan_to_num(layout.from_raw(batch['next']))
            alive = batch['alive']
            predicted, logits = model.rollout(state, batch['action'], batch['context'], goal_function(terms, batch),
                                              inputs=batch.get('inputs'), past=past_of(layout, batch))
            standing = state[:, None].expand_as(targets)
            collected['alive'].append(alive.cpu())
            masks = terms.phase_masks(batch) if terms is not None else {}
            for index, key in enumerate(phase_keys):  # without terms the first phase is every step (as before)
                mask = masks.get(key, torch.full_like(alive, index == 0))
                collected[f'phase_{key}'].append(mask.cpu())
            collected['error'].append(layout.error(predicted, targets).cpu())
            collected['standing'].append(layout.error(standing, targets).cpu())
            n = len(state)
            flat_pred, flat_true = predicted.reshape(n * k, -1), targets.reshape(n * k, -1)
            flat_persist = standing.reshape(n * k, -1)
            if terms is not None:
                robot_true = terms.bodies(flat_true)
                if robot_true is not None:
                    body = body_errors(terms.bodies(flat_pred), robot_true)  # (n·k, B, 4)
                    still = body_errors(terms.bodies(flat_persist), robot_true)
                    collected['body'].append(body.reshape(n, k, *body.shape[1:]).cpu())
                    collected['body_persist'].append(still.reshape(n, k, *still.shape[1:]).cpu())
                previous_pred = torch.cat((state[:, None], predicted[:, :-1]), 1).reshape(n * k, -1)
                previous_true = torch.cat((state[:, None], targets[:, :-1]), 1).reshape(n * k, -1)
                flat = _flat(batch, n, k, adapter)
                reward_true, true_parts = terms.reward(flat_true, previous_true, flat)
                _, pred_parts = terms.reward(flat_pred, previous_pred, flat)
                _, persist_parts = terms.reward(flat_persist, flat_persist, flat)
                collected['reward_pred'].append(tracking(pred_parts).reshape(n, k).cpu())
                collected['reward_true'].append(tracking(true_parts).reshape(n, k).cpu())
                collected['reward_persist'].append(tracking(persist_parts).reshape(n, k).cpu())
                collected['reward_full'].append(reward_true.reshape(n, k).cpu())
                collected['reward_sim'].append((batch['reward'] - batch['reward_contact']).cpu())
                said = terms.terminated(flat_true, flat)
                if said is not None:
                    collected['rule_true'].append(said.reshape(n, k).cpu())
                    collected['rule_pred'].append(terms.terminated(flat_pred, flat).reshape(n, k).cpu())
                for name, mask in terms.window_sets(batch).items():
                    if name not in extra_sets:
                        extra_sets.append(name)
                        collected[f'set_{name}'] = []
                    collected[f'set_{name}'].append(mask.cpu())
            collected['term_prob'].append(torch.sigmoid(logits).cpu())
            collected['term_label'].append(batch['terminated'].cpu())
            # A group's contact changes when its floor force crosses contact_force between the step's two states.
            changed = (batch['force'] > contact_force) != (batch['force_next'] > contact_force)  # (n, k, groups)
            if adapter.families:
                families = torch.stack([changed[..., list(groups)].any(-1) for _, groups in adapter.families], -1)
            else:
                families = torch.zeros((*changed.shape[:2], 0), dtype=torch.bool, device=changed.device)
            collected['families'].append(families.cpu())
            collected['change'].append(families.any(-1).cpu())
    model.train()
    stacked = {key: torch.cat(value).numpy() for key, value in collected.items() if value}
    numbers = {key: value.cpu().numpy() for key, value in scales.items()}
    if base_scales is not None:  # a longer horizon: the 8-step pass's spread and bodies, its own motion
        for key in ('spread', 'body'):
            if key in base_scales:
                numbers[key] = np.asarray(base_scales[key], dtype=np.float32)
    result = _summary(stacked, layout, numbers, model.error_scale.cpu().numpy(), terms is not None, adapter,
                      extra_sets, horizon=base_scales is not None)
    if base_scales is not None:
        # A horizon no held-out window lives to (the arm's episodes end on success after ~10 steps) is n/a: its mean
        # covers the steps up to the deepest one only.
        deepest = result['errors']['all']['all'].get('deepest', 0)
        result.update(windows=int(len(grid_t)), deepest=deepest, reached=deepest == k)
    return result


def _scales(layout, state, targets, alive, terms):
    """Error scales from true windows: 'motion', every error coordinate's RMS change over the window's steps (alive
    windows); 'spread', the coordinate's spread in the data (the layout's spread: pelvis height, joints, velocities;
    for xy and the rotation their change over the window); 'body', the tracked bodies' RMS change over the window of
    position, orientation, velocities (only with the adapter's bodies)."""
    import torch
    from humanoid_sim.tasks import g1_tensor_math as task
    whole = alive.all(1)
    last = targets[whole, -1]
    first = state[whole]
    change = layout.error(last, first)
    motion = change.square().mean(0).sqrt()
    motion = torch.where(motion > 0, motion, torch.ones_like(motion))
    rows = targets[alive]
    spread = layout.spread(rows, motion)
    spread = torch.maximum(spread, 1e-3 * spread.median())
    scales = dict(motion=motion, spread=spread)
    robot_first = None if terms is None else terms.bodies(first)
    if robot_first is not None:
        robot_last = terms.bodies(last)
        scales['body'] = torch.stack((
            (robot_last[0] - robot_first[0]).norm(dim=-1).square().mean().sqrt(),
            task.rotation_angle(robot_first[1], robot_last[1]).square().mean().sqrt(),
            (robot_last[2] - robot_first[2]).norm(dim=-1).square().mean().sqrt(),
            (robot_last[3] - robot_first[3]).norm(dim=-1).square().mean().sqrt()))
    return scales


def _skill(model, persist):
    return 1 - model / persist if persist else None


def judged_columns(layout, adapter):
    """None when the report judges every error coordinate, else (columns, groups, slices): the error coordinates kept
    without the adapter's held joints' angle and speed (Adapter.held), and the layout's error groups and physical
    slices as index arrays into the kept columns."""
    if not adapter.held:
        return None
    held = {layout.error_slices[name].start + joint for joint in adapter.held for name in ('joints', 'speeds')}
    columns = np.array([c for c in range(layout.errors) if c not in held])
    position = {int(c): i for i, c in enumerate(columns)}

    def kept(parts):
        return np.array([position[c] for part in parts for c in range(part.start, part.stop) if c in position],
                        dtype=np.int64)
    return (columns, {group: kept(parts) for group, parts in layout.error_groups.items()},
            {name: kept((part,)) for name, part in layout.error_slices.items()})


def _errors(stacked, layout, scales, loss_scale, mask, judged=None):
    """The error numbers over the (window, j) entries `mask` (n, k): per scale the mean squared scaled error of the
    model and of standing still, by group, skill; physical RMS; the deepest step j any window reaches and the windows
    alive to the last. judged: judged_columns' (columns, groups, slices), None for every coordinate."""
    alive = stacked['alive'] & mask
    if not alive.any():
        return dict(count=0)
    k = alive.shape[1]
    reached = np.flatnonzero(alive.any(0))
    result = dict(windows=int(mask.any(1).sum()), steps=int(alive.sum()), deepest=int(reached[-1]) + 1,
                  full_windows=int(alive[:, -1].sum()))
    full_error, full_standing = stacked['error'], stacked['standing']
    if judged is None:
        groups = {group: np.concatenate([np.arange(part.start, part.stop) for part in parts])
                  for group, parts in layout.error_groups.items()}
        slices = layout.error_slices
    else:
        columns, groups, slices = judged
        full_error, full_standing = full_error[..., columns], full_standing[..., columns]
        scales = {key: value[columns] if key in ('motion', 'spread') else value for key, value in scales.items()}
        loss_scale = loss_scale[columns]
    error, standing = full_error[alive], full_standing[alive]
    for name in SCALES:
        scale = loss_scale if name == 'loss' else scales[name]
        model, still = (error / scale) ** 2, (standing / scale) ** 2
        per_step = []
        for j in range(k):
            entries = alive[:, j]
            per_step.append(float(((full_error[:, j][entries] / scale) ** 2).mean()) if entries.any()
                            else float('nan'))
        result[name] = dict(mean=float(model.mean()), persist=float(still.mean()),
                            skill=_skill(float(model.mean()), float(still.mean())), per_step=per_step,
                            groups={group: float(model[:, columns].mean()) for group, columns in groups.items()},
                            groups_persist={group: float(still[:, columns].mean()) for group, columns in groups.items()})
    if 'body' in stacked:  # the tracked bodies' errors over their change, the mean of the four quantities
        model = float(((stacked['body'][alive] / scales['body']) ** 2).mean())
        still = float(((stacked['body_persist'][alive] / scales['body']) ** 2).mean())
        result['spread']['groups']['bodies'], result['spread']['groups_persist']['bodies'] = model, still
    result['skill'] = result['spread']['skill']
    physical = full_error
    last = alive[:, -1]
    at = f'rms_step{k}'
    result['physical'] = {}
    for name, part in slices.items():
        if judged is not None and not len(part):
            continue
        values = physical[..., part]
        result['physical'][name] = {'rms': float(np.sqrt((values[alive] ** 2).mean())),
                                    at: float(np.sqrt((values[:, -1][last] ** 2).mean())) if last.any() else None}
    for index, name in enumerate(('body_position', 'body_rotation', 'body_linear', 'body_angular')):
        if 'body' not in stacked:
            break
        values = stacked['body'][..., index]
        result['physical'][name] = {'rms': float(np.sqrt((values[alive] ** 2).mean())),
                                    at: float(np.sqrt((values[:, -1][last] ** 2).mean())) if last.any() else None}
    return result


def window_sets(adapter, extra=()):
    """The report's window sets in order: all, the contact-change windows, per family, the calm ones, the adapter's."""
    return ('all', 'contact_windows', *(f'contact_windows_{name}' for name, _ in adapter.families), 'calm_windows',
            *extra)


def phase_titles(adapter):
    return (*adapter.phases, ('all', 'всё'))


def _summary(stacked, layout, scales, loss_scale, track, adapter=None, extra=(), horizon=False):
    adapter = GetupTrack() if adapter is None else adapter
    alive = stacked['alive']
    every = np.ones(alive.shape, dtype=bool)
    phases = {key: stacked[f'phase_{key}'] for key, _ in adapter.phases}
    phases['all'] = every
    window_change = stacked['change'].any(1)[:, None] & every
    sets = {'all': every, 'contact_windows': window_change, 'calm_windows': ~window_change}
    for index, (name, _) in enumerate(adapter.families):
        sets[f'contact_windows_{name}'] = stacked['families'][..., index].any(1)[:, None] & sets['all']
    for name in extra:
        sets[name] = stacked[f'set_{name}'][:, None] & every
    names = window_sets(adapter, extra)
    judged = judged_columns(layout, adapter)
    result = dict(errors={phase: {name: _errors(stacked, layout, scales, loss_scale, phase_mask & sets[name], judged)
                                  for name in names} for phase, phase_mask in phases.items()})
    if track:
        predicted, true, persist = stacked['reward_pred'], stacked['reward_true'], stacked['reward_persist']
        reward = {}
        for phase, phase_mask in phases.items():
            entries = alive & phase_mask
            change = stacked['change'] & entries
            windows = window_change & entries
            reward[phase] = dict(
                all=_reward_stats(predicted[entries], true[entries], persist[entries]),
                contact_steps=_reward_stats(predicted[change], true[change], persist[change]))
            if not horizon:
                reward[phase].update(
                    contact_windows=_reward_stats(predicted[windows], true[windows], persist[windows]),
                    per_step=[_reward_stats(predicted[:, j][entries[:, j]], true[:, j][entries[:, j]],
                                            persist[:, j][entries[:, j]]) for j in range(alive.shape[1])])
        if not horizon:
            difference = np.abs(stacked['reward_full'] - stacked['reward_sim'])[alive]
            reward['formula_check'] = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                                           p99_abs=float(np.quantile(difference, .99)),
                                           share_above_1e3=float((difference > 1e-3).mean()))
        result['reward'] = reward
    probability, label = stacked['term_prob'][alive].astype(np.float64), stacked['term_label'][alive]
    eps = 1e-7
    rate = float(label.mean()) if len(label) else 0.
    target = label.astype(np.float64)

    def bce(p):
        p = np.clip(p, eps, 1 - eps)
        return float(-(target * np.log(p) + (1 - target) * np.log(1 - p)).mean())
    said = probability > .5
    term = dict(positives=int(label.sum()), steps=int(len(label)), base_rate=rate, bce=bce(probability),
                bce_constant=bce(np.full_like(probability, rate)), auroc=_auroc(probability, label),
                ap=_average_precision(probability, label),
                recall=float((said & label).sum() / label.sum()) if label.any() else None,
                precision=float((said & label).sum() / said.sum()) if said.any() else None)
    if 'rule_true' in stacked and not horizon:  # the task's rule on the true following states and the predicted ones
        term['rule_true'] = _rule_stats(stacked['rule_true'][alive], label)
        term['rule_predicted'] = _rule_stats(stacked['rule_pred'][alive], label)
    if horizon:
        term = {key: term[key] for key in ('positives', 'steps', 'bce', 'bce_constant', 'auroc')}
    result['termination'] = term
    result['scales'] = {key: value.tolist() for key, value in scales.items()}
    if not horizon:  # where the error sits: each error coordinate's mean squared «разброс» error over every alive step
        entries = stacked['error'][alive] / scales['spread']
        share = (entries ** 2).mean(0)
        names = [f'{name}[{i}]' for name, part in layout.error_slices.items() for i in range(part.stop - part.start)]
        order = np.argsort(-share)
        result['coordinates'] = dict(names=names, spread=share.tolist(),
                                     top=[dict(name=names[i], spread=float(share[i]), share=float(share[i] / share.sum()))
                                          for i in order[:COORDINATES_TOP]])
    if judged is not None:
        result['held_joints'] = list(adapter.held)
        result['every_coordinate'] = {phase: _errors(stacked, layout, scales, loss_scale, phase_mask, None)
                                      .get('spread', {}).get('mean')
                                      for phase, phase_mask in phases.items()}  # the held joints' angle and speed in
    return result


def _number(value, digits=4):
    return '—' if value is None or (isinstance(value, float) and math.isnan(value)) else f'{value:.{digits}f}'


def _finite(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def ladder(report, adapter=None):
    """report['ladder']: the adapter's ladder step's criteria (Thresholds) on the 8-step pass: the «разброс» error and
    skill over all windows of its phase, the reward's anomaly correlation on its reward set, the termination head's
    AUROC (not judged below `positives` terminations); passed when every gating criterion that is judged passes."""
    adapter = GetupTrack() if adapter is None else adapter
    gate = adapter.ladder
    spread = report['errors'][gate.phase]['all'].get('spread', {})
    error, skill = spread.get('mean'), spread.get('skill')
    reward = (report.get('reward') or {}).get(gate.phase, {}).get(gate.reward_set, {}).get('anomaly_r')
    term = report['termination']
    measurable = term['positives'] >= gate.positives and term['steps'] > term['positives']
    auroc = term['auroc'] if measurable else None
    criteria = []
    notes = dict(gate.notes)

    def criterion(name, value, threshold, passed, reason=None):
        gating = name in gate.gating
        if reason is None and not gating:
            reason = notes.get(name)
        criteria.append(dict(name=name, value=value, threshold=threshold, gating=gating, passed=passed,
                             reason=reason))
    criterion('error', error, gate.error, _finite(error) and error <= gate.error)
    criterion('skill', skill, gate.skill, None if gate.skill is None else _finite(skill) and skill >= gate.skill,
              None if gate.skill is not None else 'показан, не судим')
    criterion('reward', reward, gate.reward_r, _finite(reward) and reward >= gate.reward_r)
    if gate.auroc is None:
        criterion('auroc', auroc, None, None, 'показан, не судим')
    elif not measurable:
        criterion('auroc', term['auroc'], gate.auroc, None, 'мало обрывов, не судим')
    else:
        criterion('auroc', auroc, gate.auroc, _finite(auroc) and auroc >= gate.auroc)
    passed = all(c['passed'] for c in criteria if c['gating'] and c['passed'] is not None)
    return dict(adapter=adapter.name, step=gate.step, phase=gate.phase, reward_set=gate.reward_set,
                criteria=criteria, passed=bool(passed))


def verdict(report, adapter=None):
    """(passed, error, reward anomaly r): the ladder step's gate (for getting up stage 1's threshold on the rise)."""
    adapter = GetupTrack() if adapter is None else adapter
    result = report.get('ladder') or ladder(report, adapter)
    values = {c['name']: c['value'] for c in result['criteria']}
    return result['passed'], values['error'], values['reward']


def print_report(report, log=print, adapter=None):
    adapter = GetupTrack() if adapter is None else adapter
    gate = adapter.ladder
    titles = dict(phase_titles(adapter))
    if report.get('health_warning'):
        log(report['health_warning'])
    log('8-шаговая ошибка состояния на отложенных мирах: среднее квадрата ошибки, делённой на разброс координаты в '
        'данных (для xy и поворота — на её изменение за 8 шагов); в скобках «стоять на месте» и skill = 1 − '
        'MSE модели / MSE «стоять»; рядом те же в единицах «8 шагов» и «шаг»:')
    if report.get('coordinates'):
        log(' больше всего ошибки (все координаты, доля): ' + ', '.join(
            f'{row["name"]} {row["share"]:.2f}' for row in report['coordinates']['top'][:5]))
    if report.get('held_joints'):
        log(f' без суставов, которые задача держит неподвижными (угол и скорость): {report["held_joints"]}')
        if report.get('every_coordinate'):
            log(' с ними «разброс»: ' + ', '.join(f'{title} {_number(report["every_coordinate"].get(phase))}'
                                                  for phase, title in phase_titles(adapter)))
    for phase, title in phase_titles(adapter):
        log(f' {title}:')
        for name, part in report['errors'][phase].items():
            if not part.get('steps'):
                log(f'  {name}: нет окон')
                continue
            spread, groups, still = part['spread'], part['spread']['groups'], part['spread']['groups_persist']
            full = part.get('full_windows', part['windows'])
            reach = '' if full >= part['windows'] else f', до 8-го шага доживают {full}'
            log(f'  {name} ({part["windows"]} окон{reach}): разброс {_number(spread["mean"])} (стоять '
                f'{_number(spread["persist"])}, skill {_number(spread["skill"], 3)}), 8 шагов '
                f'{_number(part["motion"]["mean"])} (стоять {_number(part["motion"]["persist"])}), шаг '
                f'{_number(part["loss"]["mean"])} (стоять {_number(part["loss"]["persist"])}); '
                + ', '.join(f'{label} {_number(groups.get(key))}/{_number(still.get(key))}'
                            for key, label in (('root', 'корень'), ('joints', 'суставы'), ('velocities', 'скорости'),
                                               ('bodies', 'тела')) if key in groups))
    longer = sorted((int(key[8:]), value) for key, value in report.items() if key.startswith('horizon_'))
    if longer:
        log(' разброс по горизонтам, все окна (в скобках «стоять на месте»): 8 | '
            + ' | '.join(str(k) for k, _ in longer))
        for phase, title in phase_titles(adapter):
            part = report['errors'][phase]['all'].get('spread', {})
            cells = [f'{_number(part.get("mean"))} ({_number(part.get("persist"))})']
            for k, value in longer:
                if 'skipped' in value:
                    cells.append(value['skipped'])
                    continue
                other_part = value['errors'][phase]['all']
                other = other_part.get('spread', {})
                cell = f'{_number(other.get("mean"))} ({_number(other.get("persist"))})'
                deepest = other_part.get('deepest')
                if deepest is not None and deepest < k:
                    cell = f'н/д: окна живут до j = {deepest}, по ним {cell}'
                cells.append(cell)
            log(f'  {title}: ' + ' | '.join(cells))
    physical = report['errors'][gate.phase]['all'].get('physical', {})
    if physical:
        log(f'  RMS на 8-м шаге, {titles[gate.phase]}: ' + ', '.join(f'{name} {_number(value["rms_step8"])}'
                                                                   for name, value in physical.items()))
    reward = report.get('reward')
    if reward:
        log(f' награда без {", ".join(adapter.agreed) or "—"} по модели (в скобках «стоять на месте»), skill и '
            'корреляция отклонений от «стоять» (r_модели − r_стоять против r_правды − r_стоять):')
        for phase, title in phase_titles(adapter):
            for name in ('all', 'contact_steps', 'contact_windows'):
                part = reward[phase][name]
                if part.get('count'):
                    log(f'  {title}, {name} ({part["count"]}): r {_number(part["r"], 3)} ({_number(part["r_persist"], 3)}),'
                        f' MAE {_number(part["mae"])} ({_number(part["mae_persist"])}), skill '
                        f'{_number(part["skill"], 3)}, корреляция отклонений {_number(part["anomaly_r"], 3)} '
                        f'(σ настоящей {_number(part["std_true"])})')
        log(f'  {titles[gate.phase]}, корреляция отклонений по шагам j: '
            + ' '.join(_number(part.get('anomaly_r'), 3) for part in reward[gate.phase]['per_step']))
        check = reward['formula_check']
        log(f'  формула по настоящим состояниям против награды симулятора без {", ".join(adapter.excluded) or "—"}: '
            f'макс |Δ| {_number(check["max_abs"], 6)}, среднее {_number(check["mean_abs"], 6)}, p99 '
            f'{_number(check["p99_abs"], 6)}, доля выше 1e-3 {_number(check["share_above_1e3"], 4)}')
    term = report['termination']
    log(f' обрыв: {term["positives"]} из {term["steps"]} (частота {_number(term["base_rate"], 5)}), BCE '
        f'{_number(term["bce"], 5)} (константа {_number(term["bce_constant"], 5)}), AUROC {_number(term["auroc"], 3)}, '
        f'AP {_number(term["ap"], 3)}, полнота {_number(term["recall"], 3)}, точность {_number(term["precision"], 3)}')
    for key, title in (('rule_true', 'правило задачи на настоящих состояниях'),
                       ('rule_predicted', 'правило задачи на предсказанных')):
        if key in term:
            rule = term[key]
            log(f'  {title}: совпадает {_number(rule["agree"], 5)}, пропущено {rule["missed"]}, лишних '
                f'{rule["extra"]}, полнота {_number(rule["recall"], 3)}, точность {_number(rule["precision"], 3)}')
    result = report.get('ladder') or ladder(report, adapter)
    words = {True: 'да', False: 'нет', None: 'не судим'}
    log(f'Порог ступени {result["step"]} лестницы ({result["adapter"]}, этап 1 SuperMimic; {titles[gate.phase]}): '
        + '; '.join(f'{c["name"]} {_number(c["value"], 4)}'
                    + (f' (порог {c["threshold"]})' if c['threshold'] is not None else '')
                    + f' — {c["reason"] or words[c["passed"]]}'
                    + ('' if c['gating'] or c['passed'] is None or c['reason'] else ', не решает')
                    for c in result['criteria'])
        + f': {"пройден" if result["passed"] else "не пройден"}'
        + (' (сбор данных не здоров)' if report.get('health_warning') else ''))
    return result['passed']


def curve_row(report, steps, seconds, adapter=None):
    """One row of the training curve: the ladder phase's numbers of a quick evaluation."""
    adapter = GetupTrack() if adapter is None else adapter
    gate = adapter.ladder
    rise = report['errors'][gate.phase]['all']
    reward = (report.get('reward') or {}).get(gate.phase, {}).get(gate.reward_set, {})
    term = report['termination']
    return dict(steps=steps, seconds=round(seconds, 1), spread=rise.get('spread', {}).get('mean'),
                persist=rise.get('spread', {}).get('persist'), skill=rise.get('skill'),
                reward_anomaly_r=reward.get('anomaly_r'), reward_skill=reward.get('skill'), bce=term['bce'],
                bce_constant=term['bce_constant'], auroc=term['auroc'])


def _save(path, save):
    """Writes through a temporary file and os.replace, so an interruption never leaves half a file."""
    temporary = path.with_name(path.name + '.tmp')
    save(temporary)
    os.replace(temporary, path)


def fit(data_dir, out, minutes=None, max_steps=None, device=None, hidden=None, batch=None, holdout=.1,
        learning_rate=None, eval_windows=50_000, eval_stride=4, seed=0, reference=None, evaluate_only=False,
        contact_force=CONTACT_FORCE, resume=False, checkpoint_minutes=CHECKPOINT_MINUTES, curve_windows=CURVE_WINDOWS,
        log=print, horizons=HORIZONS, speed='auto', long_windows=LONG_WINDOWS, loss_units='step', window=None,
        history=1, ensemble=None):
    """Fits (or with evaluate_only loads out/model.pt; with resume goes on from out/model.pt and out/train_state.pt)
    and evaluates; writes out/model.pt, out/train_state.pt, out/curve.json and out/report.json and returns the report.
    reference: the CPU task of the data's task (tests); None: made from meta's task_config. hidden, batch: None, the
    adapter's for the device; batch 'auto' (cuda: world_model_speed.choose_batch). horizons: the report's horizons, 8
    first (the training's window), each longer one in report['horizon_<k>']."""
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_speed import choose_batch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = str(device).startswith('cuda')
    horizons = tuple(sorted(set(int(k) for k in horizons)))
    if wm.WINDOW not in horizons:
        raise SystemExit(f'--horizons: нужен горизонт обучения {wm.WINDOW}')
    torch.backends.cuda.matmul.allow_tf32 = True  # the paper's TF32
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = time.time()
    meta, arrays = load(data_dir)
    adapter = adapter_of_meta(meta)
    defaults = adapter.defaults['cuda' if cuda else 'cpu']
    warning = health_warning(meta)
    if warning:
        log(warning)
    data = Data(arrays, device, holdout, adapter)
    del arrays
    reference = reference if reference is not None else adapter.reference(meta)
    terms = None if reference is None else adapter.bind(reference, device)
    log(f'данные: {data.total} шагов × {data.worlds} миров ({len(data.held_worlds)} отложено), '
        f'{time.time() - loaded:.0f} с; адаптер {adapter.name} (ступень {adapter.ladder.step}); действия: '
        f'{meta.get("action_mode", "—")}' + (f' {meta["action_noise"]}' if meta.get('action_noise') else ''))
    goal = adapter.goal if terms is not None else 0
    requested = batch
    batch = defaults['batch'] if batch is None else batch
    batch_mode = 'auto' if batch == 'auto' else 'fixed'
    learning_rate = learning_rate or wm.LEARNING_RATE
    curve, rows, steps, seconds, last, info = [], None, None, None, {}, {}
    if evaluate_only and ensemble:  # members' fit folders on the same data: judged as their mean (wm.Ensemble)
        model = wm.Ensemble.load([Path(folder) / 'model.pt' for folder in ensemble], device)
        data.history = model.history
        info['ensemble'] = [str(folder) for folder in ensemble]
        log(f'ансамбль из {len(ensemble)} моделей: {", ".join(str(folder) for folder in ensemble)}')
    elif evaluate_only:
        model, _ = wm.WorldModel.load(out / 'model.pt', device)
        data.history = model.history
    else:
        state = None
        if resume:
            if not (out / 'model.pt').is_file() or not (out / 'train_state.pt').is_file():
                raise SystemExit(f'нечего продолжать: нет {out / "model.pt"} или {out / "train_state.pt"}')
            model, _ = wm.WorldModel.load(out / 'model.pt', device)
            data.history = model.history
            state = torch.load(out / 'train_state.pt', map_location='cpu', weights_only=False)
            curve, rows = list(state.get('curve', [])), state.get('normalization_rows')
            log(f'продолжаю с шага {state["steps"]} ({state["seconds"]:.0f} с обучения)')
            saved = state.get('batch')
            if saved is not None:  # the run's batch, not a new auto choice on whatever memory is free now
                if requested is not None and requested != 'auto' and requested != saved:
                    raise SystemExit(f'--batch {requested}: прогон обучался по {saved} окон; продолжение только с '
                                     f'ними (или без --batch)')
                batch, batch_mode = saved, state.get('batch_mode', batch_mode)
                info['batch_note'] = f'{batch} окон, как до перерыва'
                log(f'окон в шаге: {info["batch_note"]}')
        else:
            model = wm.WorldModel(meta['nu'], len(meta['context']), goal, hidden=tuple(hidden or defaults['hidden']),
                                  history=history, **adapter.model_settings()).to(device)
            data.history = model.history
            model.home.copy_(torch.as_tensor(meta['home'], dtype=torch.float32))
            rows = normalize(model, data, terms, model.layout, device, torch.Generator().manual_seed(seed + 1), adapter)
            log(f'нормировка по {rows} переходам')
            if loss_units == 'spread':  # train in the report's units: each coordinate's spread (xy, rotation: k-step change)
                k = HORIZONS[0]
                count = min(SCALE_WINDOWS, (data.total - k + 1) * len(data.train_worlds))
                sample = data.window(*data.sample(count, k, torch.Generator().manual_seed(seed + 2)), k, device)
                with torch.no_grad():
                    spread = _scales(model.layout, model.layout.from_raw(sample['state']),
                                     torch.nan_to_num(model.layout.from_raw(sample['next'])), sample['alive'],
                                     terms)['spread']
                for part in model.layout.error_slices.values():  # a still coordinate (a held gripper: spread 0)
                    floor = wm.ERROR_FLOOR * spread[part].median()  # would weigh a thousand times the rest
                    spread[part] = torch.maximum(spread[part], floor.clamp(min=wm.STD_FLOOR))
                model.error_scale.copy_(spread.to(model.error_scale))
                info['loss_units'] = 'spread'
                log('единицы потерь: разброс координат (как в проверке)')
        if batch == 'auto':
            if not cuda:
                batch = wm.BATCH_WINDOWS
                info['batch_note'] = f'auto только на CUDA: {batch} окон'
            else:
                keys = training_keys(adapter)

                def measure(n):
                    sample = data.window(*data.sample(n, window or wm.WINDOW,
                                                      torch.Generator().manual_seed(MEASURE_SEED)),
                                         window or wm.WINDOW, device, keys)
                    layout = model.layout
                    return wm.window_loss(model, layout.from_raw(sample['state']), sample['action'],
                                          sample['context'], layout.from_raw(sample['next']), sample['terminated'],
                                          sample['alive'], goal_function(terms, sample),
                                          inputs=sample.get('inputs'), past=past_of(layout, sample))[:2]
                batch, info['batch_note'] = choose_batch(model, measure, device)
            log(f'окон в шаге: {info["batch_note"]}')

        def checkpoint(steps, seconds, optimizer, generator):
            report = evaluate(model, data, terms, device, curve_windows, eval_stride, contact_force=contact_force,
                              adapter=adapter)
            row = curve_row(report, steps, seconds, adapter)
            curve.append(row)
            log(f'проверка, шаг {steps}, {seconds:.0f} с: {dict(phase_titles(adapter))[adapter.ladder.phase]} разброс '
                f'{_number(row["spread"])} (стоять {_number(row["persist"])}, skill {_number(row["skill"], 3)}), '
                f'корреляция отклонений награды {_number(row["reward_anomaly_r"], 3)}, BCE {_number(row["bce"], 5)} '
                f'(константа {_number(row["bce_constant"], 5)}), AUROC {_number(row["auroc"], 3)}')
            _save(out / 'model.pt', lambda path: model.save(path, data=str(data_dir), steps=steps, seconds=seconds))
            _save(out / 'train_state.pt', lambda path: torch.save(dict(
                optimizer=optimizer.state_dict(), generator=generator.get_state(), steps=steps, seconds=seconds,
                curve=curve, normalization_rows=rows, batch=batch, batch_mode=batch_mode, learning_rate=learning_rate,
                seed=seed), path))
            (out / 'curve.json').write_text(json.dumps(curve, ensure_ascii=False, indent=1), encoding='utf-8')
        steps, seconds, last, optimizer, generator = train(
            model, data, terms, device, minutes, max_steps, batch, learning_rate, seed, log, resume=state,
            checkpoint=checkpoint, checkpoint_seconds=None if checkpoint_minutes is None else 60 * checkpoint_minutes,
            adapter=adapter, speed=speed, info=info, window=window)
        info['window'] = window or wm.WINDOW
        if not curve or curve[-1]['steps'] != steps:  # the end's checkpoint, unless one was just made
            checkpoint(steps, seconds, optimizer, generator)
    started = time.time()
    report = evaluate(model, data, terms, device, eval_windows, eval_stride, contact_force=contact_force,
                      adapter=adapter)
    for k in horizons:
        if k == wm.WINDOW:
            continue
        if data.total < k:
            report[f'horizon_{k}'] = {'skipped': f'данных меньше {k} шагов'}
            continue
        report[f'horizon_{k}'] = evaluate(model, data, terms, device, long_windows, eval_stride,
                                          batch_size=max(1, 4096 * wm.WINDOW // k), contact_force=contact_force,
                                          adapter=adapter, k=k, base_scales=report['scales'])
    report.update(data=str(data_dir), meta={key: meta.get(key) for key in (
                      'stage', 'stage_steps', 'worlds', 'steps', 'backend', 'seed', 'action_mode', 'action_noise',
                      'health', 'adapter', 'ladder_step')},
                  health=meta.get('health'), health_warning=warning, curve=curve,
                  training=dict(steps=steps, seconds=seconds, last=last, normalization_rows=rows,
                                windows_per_second=steps * batch / seconds if seconds else None,
                                device=str(device), hidden=model.settings['hidden'],
                                batch=None if evaluate_only else batch, batch_mode=batch_mode, **info),
                  evaluation_seconds=round(time.time() - started, 1), adapter=adapter.name)
    report['ladder'] = ladder(report, adapter)
    report['passed'] = print_report(report, log, adapter)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    return report


def main():
    import torch
    # Redirected on Windows, stdout is cp1251, which has no × or σ.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Обучение модели мира на переходах collect.py и отчёт.')
    parser.add_argument('data')
    parser.add_argument('--out', required=True)
    parser.add_argument('--minutes', type=float, default=None,
                        help='минут обучения всего, с продолжением вместе (по умолчанию адаптера)')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default=None)
    parser.add_argument('--hidden', default=None, help='ширины слоёв через запятую (по умолчанию адаптера)')
    parser.add_argument('--batch', default=None, help='окон в шаге Adam или auto (по умолчанию адаптера)')
    parser.add_argument('--window', type=int, default=None, help='шагов в окне обучения (по умолчанию 8)')
    parser.add_argument('--ensemble', default=None,
                        help='с --evaluate-only: папки fit.py через запятую, судить их среднее (отчёт в --out)')
    parser.add_argument('--history', type=int, default=1,
                        help='сколько состояний видит модель: текущее и history − 1 предыдущих (новая модель)')
    parser.add_argument('--loss-units', choices=('step', 'spread'), default='step',
                        help='единицы ошибки в потерях: изменение за шаг (как раньше) или разброс, как в проверке')
    parser.add_argument('--speed', choices=('auto', 'plain', 'graph', 'compile'), default='auto',
                        help='auto: граф CUDA на cuda, обычный цикл на процессоре; compile только на Linux')
    parser.add_argument('--horizons', default=','.join(str(k) for k in HORIZONS),
                        help='горизонты отчёта через запятую, 8 обязателен')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--holdout', type=float, default=.1, help='доля отложенных миров')
    parser.add_argument('--eval-windows', type=int, default=50_000)
    parser.add_argument('--eval-windows-32', type=int, default=LONG_WINDOWS, help='окон горизонтов длиннее 8')
    parser.add_argument('--eval-stride', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate-only', action='store_true', help='только отчёт по готовой out/model.pt')
    parser.add_argument('--resume', action='store_true', help='продолжить с out/model.pt и out/train_state.pt')
    parser.add_argument('--checkpoint-minutes', type=float, default=CHECKPOINT_MINUTES,
                        help='минут обучения между проверками на отложенных данных и сохранениями')
    parser.add_argument('--curve-windows', type=int, default=CURVE_WINDOWS, help='окон в проверке для кривой')
    parser.add_argument('--contact-force', type=float, default=CONTACT_FORCE,
                        help='Н: группа касается пола, пока её нормальная сила выше')
    args = parser.parse_args()
    if args.resume and args.evaluate_only:
        raise SystemExit('--resume и --evaluate-only вместе не бывают')
    meta_path = Path(args.data) / 'meta.json'
    if not meta_path.is_file():
        raise SystemExit(f'нет данных в {args.data}')
    try:
        adapter = adapter_of_meta(json.loads(meta_path.read_text(encoding='utf-8')))
    except ValueError as error:
        raise SystemExit(str(error))
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    minutes = adapter.defaults[device]['minutes'] if args.minutes is None else args.minutes
    hidden = None if args.hidden is None else tuple(int(v) for v in args.hidden.split(','))
    batch = None if args.batch is None else 'auto' if args.batch == 'auto' else int(args.batch)
    try:
        horizons = tuple(int(v) for v in args.horizons.split(','))
    except ValueError:
        raise SystemExit('--horizons: целые числа через запятую')
    fit(args.data, args.out, minutes, args.max_steps, device, hidden, batch, args.holdout, args.lr,
        args.eval_windows, args.eval_stride, args.seed, evaluate_only=args.evaluate_only,
        contact_force=args.contact_force, resume=args.resume, checkpoint_minutes=args.checkpoint_minutes,
        curve_windows=args.curve_windows, horizons=horizons, speed=args.speed, long_windows=args.eval_windows_32,
        loss_units=args.loss_units, window=args.window, history=args.history,
        ensemble=args.ensemble.split(',') if args.ensemble else None)


if __name__ == '__main__':
    main()
