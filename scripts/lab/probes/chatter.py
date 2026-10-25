"""Torque chattering, CPU (iteration 42, the user's request of 25.10):
chatter.py <stage | experiment[:steps] | export dir | unitree_v0> [--workers 4] [--json <file>] [--seconds 10]
[--seeds 3] [--delay <s>] [--randomize] [--kicks]

On the robot, sensor noise passes through the policy into a 50 Hz jitter of the PD targets and from them into the motor
torques, which heats the windings; the training's action_change penalty (-w Σ(Δa)²) shapes it, but nothing measured it
for stage selection. Each run: the deterministic policy (as on the robot; an export_g1 folder through
export_g1.ExportedPolicy, its policy.onnx in its stage's CPU environment; 'unitree_v0' Unitree's shipped velocity policy,
stand_turn.stage_data), heading hold off, a reset with the scenario's command at phase 0, and the realism the CUDA task
adds at training_realism 1 that a robot has too, as train_falls.py carries it over: sensor noise on every observed frame
(g1_tensor_env.NOISE, uniform: angular velocity, gravity, joint angles and speeds) and an encoder bias per run
(ENCODER_BIAS) on the observed joint angles. The action delay as in training: k ~ U{0..round(delay / 2 ms)} physics
substeps per run (reset option action_delay_steps), delay the stage's task_config action_delay (--delay overrides; the
reference runs with the judged stage's delay and the same draws). Nominal otherwise: motor friction x1, the foot
friction and pelvis COM of the model, no pushes; --randomize draws the foot friction U(FOOT_FRICTION_RANGE) and the COM
offset (COM_OFFSET) per run, --kicks adds the velocity kicks REAL_PUSH every REAL_PUSH_INTERVAL s on average.
Scenarios (SCENARIOS, --seconds each, measured from WARMUP s): standing, walking 0.5 and 1 m/s, stop-start at 0.5 m/s
(stand, walk, stand, walk, stand, a fifth of the run each), side steps at +0.3 and -0.3 m/s; for an arm_targets stage
also 'carry': standing with 1 kg per hand (set_hand_load) in the carry pose nearest shoulder pitch -1.1 / elbow 1.1 of
the arm pose library (arm_counterfactual's carry-1.1), a 1 s quintic from 0.5 s, measured from 2.5 s. Seeds SEED + k:
the noise, the bias and the delay draw (numpy default_rng((seed, NOISE_STREAM))) and the reset.
Raw series per joint (29): the PD target of every control step (50 Hz; G1VelocityEnv._target of the applied action,
home moved by the arm targets under arm_targets) and the motor's torque at every physics substep (500 Hz): the PD torque
clipped to the speed-dependent peak, g1_tensor_math.motor_torque without the joint friction it then subtracts (the
gearbox's friction, a tanh of the joint speed that flips with every sign change of v; the current, and so the heat,
makes the clipped torque). Per run and joint (band(): linear detrend, Hann-window periodogram, the DC bin left out):
  action_hf_share       the share of the PD target's variance above ACTION_CUTOFF 5 Hz (NaN for a target that never
                        moves: the locked wrists), and action_hf_rms_rad, the RMS of that part (rad);
  torque_hf_share       the same for the motor torque above TORQUE_CUTOFF 10 Hz, torque_hf_rms_nm the RMS of that part
                        (N m) and torque_hf_peak_share that over the motor's low-speed peak (motor_parameters y1);
  torque_rms_peak_share the RMS of the whole torque over the peak: the thermal I²t proxy, with no minutes (there is no
                        model of the windings' heating; none is made up);
  mean_abs_delta_action the mean |a_t - a_t-1| of the applied action per control step.
Runs combine per scenario over the seeds and then over the scenarios with equal weights (RMS values as the root of the
mean square, the rest as means; a run that fell is left out and counted); the six base scenarios only, so the reference
(Unitree v0, which has no arm targets) is measured on the same; 'carry' is printed and kept apart. The worst joint of a
key is the largest value among the joints the policy drives (its PD target moves; not the locked wrists).
JSON (--json): {stage, path, settings, summary, scenarios, reference_scenarios, carry, joints, reference_joints, runs,
reference_runs, reasons}. summary (read by the passport generator): action_hf_share, torque_hf_share, torque_hf_rms_nm,
torque_rms_peak_share, mean_abs_delta_action, action_hf_rms_rad, torque_hf_peak_share (each the worst joint's value),
worst_joint (the joint name of the worst torque_hf_rms_nm, the first gated key), worst_joints (key -> joint name), runs,
falls (scenario, seed and control step of each run that fell), reference (the same keys for Unitree v0 under the same
scenarios, seeds and delay), ok (bool) and gate (the rule as text). scenarios / reference_scenarios: name -> the same
keys for that scenario alone; carry: the same for the carry scenario (None without arm_targets); joints /
reference_joints: key -> 29 values in actuator order, names under 'names'; runs: per run the raw per-joint numbers (NaN
written as null); reasons: why ok is false.
Gate (GATED, GATE_FACTOR): on the worst joint, torque_hf_rms_nm and action_hf_rms_rad at most 1.5 x v0's, and no fall;
the data and the reasons at GATE_FACTOR. Candidate 8 against v0 (25.10): 4.57 / 4.08 N m, 0.135 / 0.112 rad, ok;
without the sensor noise 4.28 / 3.07 N m and 0.113 / 0.086 rad: the noise adds 14 % of candidate 8's high-frequency
torque power and 77 % of v0's (v0 sees one frame, ours five). About 20 s for a stage and the reference on 4 processes
(36 runs of ~2.3 s)."""
import argparse
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stand_turn as st  # noqa: E402

REFERENCE = 'unitree_v0'
SECONDS, WARMUP, SEEDS, SEED, NOISE_STREAM = 10., 1., 3, 74000, 4242
ACTION_CUTOFF, TORQUE_CUTOFF = 5., 10.  # Hz: the PD target at 50 Hz, the motor torque at 500 Hz
STILL = 1e-9  # a series whose detrended std is at or below this does not move: its shares are NaN
CARRY_LOAD, CARRY_MOVE_AT, CARRY_FROM, CARRY_PITCH, CARRY_ELBOW = 1., .5, 2.5, -1.1, 1.1
SCENARIOS = {  # name: (label, [(share of the run, command)])
    'stand': ('стоя', [(1., (0., 0., 0.))]),
    'walk05': ('ходьба 0.5 м/с', [(1., (.5, 0., 0.))]),
    'walk1': ('ходьба 1 м/с', [(1., (1., 0., 0.))]),
    'stopstart': ('стоп-старт 0.5 м/с', [(.2, (0., 0., 0.)), (.2, (.5, 0., 0.)), (.2, (0., 0., 0.)),
                                         (.2, (.5, 0., 0.)), (.2, (0., 0., 0.))]),
    'side_left': ('вбок влево 0.3 м/с', [(1., (0., .3, 0.))]),
    'side_right': ('вбок вправо 0.3 м/с', [(1., (0., -.3, 0.))]),
    'carry': ('несу 1 кг, стоя', [(1., (0., 0., 0.))]),
}
BASE = ('stand', 'walk05', 'walk1', 'stopstart', 'side_left', 'side_right')
# Per-run fields, each a value per joint; the RMS ones combine as the root of the mean square.
FIELDS = ('action_hf_share', 'action_hf_rms_rad', 'torque_hf_share', 'torque_hf_rms_nm', 'torque_rms_nm',
          'mean_abs_delta_action')
RMS_FIELDS = ('action_hf_rms_rad', 'torque_hf_rms_nm', 'torque_rms_nm')
KEYS = ('action_hf_share', 'torque_hf_share', 'torque_hf_rms_nm', 'torque_rms_peak_share', 'mean_abs_delta_action',
        'action_hf_rms_rad', 'torque_hf_peak_share')
LABELS = dict(action_hf_share='цель > 5 Гц, доля', torque_hf_share='момент > 10 Гц, доля',
              torque_hf_rms_nm='ВЧ момента, Н·м', torque_rms_peak_share='СКЗ момента / пик',
              mean_abs_delta_action='|Δa| за шаг', action_hf_rms_rad='ВЧ цели, рад',
              torque_hf_peak_share='ВЧ момента / пик')
# The gate, relative to Unitree v0 (run on the real G1) in the same runs: the worst joint's torque_hf_rms_nm (the
# chatter's torque, what heats) and action_hf_rms_rad (the PD target's jitter itself) at most GATE_FACTOR x v0's, and no
# fall. The data (25.10, the six scenarios x 3 seeds, sensor noise, the candidates' training delay 0): candidate 8
# (exp-28691193718e4d0e @ 7221510144) 4.57 N m / 0.135 rad against v0's 4.08 / 0.112 (x1.12 / x1.21), its export the
# same, candidate 10 (exp-fa2450e3a6af4bf4 @ 8784248832) 4.59 / 0.135 (x1.13 / x1.21), branch 41 @ 9111601152 4.89 /
# 0.143 (x1.20 / x1.27); candidate 8's own stochastic policy (its training sigma), a policy that does chatter, 10.7 /
# 0.219 (x2.62 / x1.96). A seed moves a summary by 3-8 %, a delay of 6 ms by under 3 %. 1.5 in RMS is 2.25 in power: a
# policy whose chatter heats more than twice v0's fails, the candidates pass with a margin of 1.2-1.3. Not
# action_hf_share, the lead's first idea: v0's worst is 0.69, so 1.5 x it is above 1 and could never fail, and a share
# is high on joints that hardly move (standing, the arms' targets 0.7-0.85 for both).
GATED = ('torque_hf_rms_nm', 'action_hf_rms_rad')
GATE_FACTOR = 1.5


def resolve(spec):
    """'unitree_v0', a stage or export_g1 folder, or <experiment>[:steps] of the lab under runs/ (its latest stage)."""
    if spec == REFERENCE:
        return REFERENCE
    from train_falls import resolve as lab_stage
    return str(Path(lab_stage(spec)).resolve())


def task_config_of(stage):
    """The task_config a run of `stage` uses (an export folder: its stage's; unitree_v0: stand_turn's), read without the
    policy."""
    if stage == REFERENCE:
        return st.stage_data(REFERENCE)[2]
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.export_g1 import ExportedPolicy, is_export
    return stage_task_config(ExportedPolicy(stage).stage if is_export(stage) else Path(stage))


def has_arm_targets(stage):
    """Whether the stage (an export folder: its stage) has task_config arm_targets: then the carry scenario runs too."""
    return stage != REFERENCE and bool(task_config_of(stage).get('arm_targets'))


def stage_delay(stage):
    """The stage's training action delay (s, task_config action_delay; 0 without it)."""
    return float(task_config_of(stage).get('action_delay') or 0.)


def label(stage):
    if stage == REFERENCE:
        return 'Unitree v0'
    path = Path(stage)
    if (path / 'policy.onnx').is_file():
        return path.name + ' (выгрузка)'
    return '%s @ %s' % (path.parent.name, path.name[6:].lstrip('0'))


def band(x, rate, cutoff):
    """Per column of x ((n,) or (n, k), sampled at `rate` Hz): (share, rms) of the part above `cutoff` Hz. The column
    less its least-squares line, times a Hann window, gives the periodogram P (the DC bin left out; the window's leakage
    stays within ±2 bins of rate / n); share = Σ P(f > cutoff) / Σ P(f > 0), rms = std(detrended)·√share. A column that
    does not move (std <= STILL) gives NaN for both."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2
    centred = x - x.mean(axis=0)
    detrended = centred - np.outer(t, t @ centred / (t @ t))
    power = np.abs(np.fft.rfft(detrended * np.hanning(n)[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(n, 1. / rate)
    total, high = power[1:].sum(axis=0), power[freq > cutoff].sum(axis=0)
    std = detrended.std(axis=0)
    moving = (std > STILL) & (total > 0)
    share = np.full(x.shape[1], np.nan)
    share[moving] = high[moving] / total[moving]
    return share, std * np.sqrt(share)


def noisy(raw, exact, rng, bias):
    """raw._observation with the CUDA task's sensor noise on the newest frame (every history frame after a fill) and the
    encoder bias on the joint angles, the critic extras exact, as train_falls.realism."""
    import torch
    from humanoid_sim.g1_tensor_env import NOISE
    from humanoid_sim.tasks import g1_tensor_math as task
    nu, extra = raw.model.nu, raw.extra_sizes
    spans = task.deploy_terms(nu, extra)
    terms = ((spans[0], NOISE['angular_velocity']), (spans[1], NOISE['gravity']), (spans[4], NOISE['joint_position']),
             (spans[5], NOISE['joint_velocity']))

    def frame_noise(frame):
        for (start, end), scale in terms:
            frame[start:end] += torch.from_numpy((2 * rng.random(end - start) - 1) * scale).to(frame.dtype)
        frame[spans[4][0]:spans[4][1]] += torch.from_numpy(bias).to(frame.dtype)
        return frame

    def observation(history='replace'):
        obs = exact(history)
        if raw.frames is None:  # one frame: the observation itself
            obs[:raw.frame_size] = frame_noise(torch.from_numpy(obs[:raw.frame_size].astype(np.float64))).numpy()
            return obs
        frame = frame_noise(raw.frames[0, -1].clone())
        if history == 'fill':
            raw.frames[0] = frame
        else:
            raw.frames[0, -1] = frame
        obs[:raw.actor_inputs] = task.history_observation(raw.frames, nu, extra)[0].numpy()
        return obs
    return observation


def carry_pose(raw):
    """(nu,) the carry pose: the arm pose library's carry row nearest shoulder pitch CARRY_PITCH with the elbow near
    CARRY_ELBOW (arm_counterfactual's carry-1.1), home elsewhere."""
    from humanoid_sim.tasks.g1_velocity import arm_pose_library
    names, _, carry, _ = arm_pose_library()
    row = carry[int(np.argmin(np.abs(carry[:, 0] - CARRY_PITCH) + .2 * np.abs(carry[:, 3] - CARRY_ELBOW)))]
    joints = [raw.model.joint(j).name for j in raw.joints]
    pose = raw.home.copy()
    for name, value in zip(names, row):
        pose[joints.index(name)] = value
    return pose


def schedule(scenario, steps):
    """[(first control step, command)] of a scenario over `steps` control steps."""
    out, at = [], 0.
    for share, command in SCENARIOS[scenario][1]:
        out.append((int(round(at * steps)), command))
        at += share
    return out


def finite(value):
    """A number for JSON: NaN and inf as None."""
    return None if value is None or not math.isfinite(value) else float(value)


def run(job):
    """One run: per joint the six FIELDS over the measured window, the motor peaks, the delay drawn; fell = the
    control step of a fall (None)."""
    import torch
    torch.set_num_threads(1)
    from humanoid_sim.g1_tensor_env import COM_OFFSET, ENCODER_BIAS, FOOT_FRICTION_RANGE, REAL_PUSH, REAL_PUSH_INTERVAL
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.tasks.g1_velocity import motor_parameters
    stage, scenario, seed = job['stage'], job['scenario'], job['seed']
    model, norm, config = st.stage_data(stage)
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 10 ** 6})
    raw = env.unwrapped
    if raw.motors is None:
        raise ValueError(f'{stage}: the probe needs the real-robot motor model (task_config real_robot)')
    if scenario == 'carry' and not raw.arm_targets:
        raise ValueError(f'{stage}: the carry scenario needs task_config arm_targets')
    names = [raw.model.joint(j).name for j in raw.joints]
    peak = motor_parameters(names)[2]
    rng = np.random.default_rng((seed, NOISE_STREAM))
    delay = float(config.get('action_delay') or 0.) if job.get('delay') is None else float(job['delay'])
    delay_steps = int(rng.integers(round(delay / raw.model.opt.timestep) + 1))
    bias = (2 * rng.random(raw.model.nu) - 1) * ENCODER_BIAS
    if job.get('randomize'):
        raw.model.geom_friction[raw.foot_geoms, 0] = rng.uniform(*FOOT_FRICTION_RANGE)
        raw.model.body_ipos[raw.pelvis] += (2 * rng.random(3) - 1) * np.asarray(COM_OFFSET)
    raw._observation = noisy(raw, raw._observation, rng, bias)
    steps = int(round(job['seconds'] / raw.dt))
    plan = schedule(scenario, steps)
    start = int(round((CARRY_FROM if scenario == 'carry' else WARMUP) / raw.dt))
    torques, original = [], task.motor_torque

    def recorded(torque, joint_velocity, x1, x2, y1, y2, static, viscous, *args, **kwargs):
        # the motor's own torque: the PD torque clipped to its peak, before the joint friction is subtracted
        torques.append(original(torque, joint_velocity, x1, x2, y1, y2, static * 0, viscous * 0).numpy().copy())
        return original(torque, joint_velocity, x1, x2, y1, y2, static, viscous, *args, **kwargs)
    task.motor_torque = recorded
    actions, targets, fell = [], [], None
    try:
        obs, _ = env.reset(seed=seed, options={'command': list(plan[0][1]), 'phase_steps': 0,
                                               'action_delay_steps': delay_steps})
        if scenario == 'carry':
            raw.set_hand_load(CARRY_LOAD)
            pose = carry_pose(raw)
        changes = dict(plan[1:])
        for k in range(steps):
            if k in changes:
                raw.set_command(np.asarray(changes[k], dtype=float))
                obs = raw._observation('replace')
            if scenario == 'carry' and k == int(round(CARRY_MOVE_AT / raw.dt)):
                raw.set_arm_targets(pose, 1, seconds=1.)
                obs = raw._observation('replace')
            if job.get('kicks') and rng.random() < raw.dt / REAL_PUSH_INTERVAL:
                raw.data.qvel[:6] += (2 * rng.random(6) - 1) * np.asarray(REAL_PUSH)
            action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
            obs, _, terminated, _, _ = env.step(action)
            actions.append(raw.previous_action.copy())  # the applied action (box clip, lock_wrists)
            targets.append(raw._target(raw.previous_action, raw.previous_offset) if raw.previous_flag
                           else raw._target(raw.previous_action))
            if terminated:
                fell = k
                break
    finally:
        task.motor_torque = original
        env.close()
    out = dict(stage=label(stage), scenario=scenario, seed=seed, delay=delay, delay_steps=delay_steps, fell=fell,
               names=names, peak=peak.tolist())
    if fell is not None or len(torques) != steps * raw.frame_skip:
        return out
    actions, targets, torques = np.array(actions), np.array(targets), np.array(torques)
    action_share, action_rms = band(targets[start:], 1. / raw.dt, ACTION_CUTOFF)
    torque_share, torque_rms = band(torques[start * raw.frame_skip:], 1. / raw.model.opt.timestep, TORQUE_CUTOFF)
    values = dict(action_hf_share=action_share, action_hf_rms_rad=action_rms, torque_hf_share=torque_share,
                  torque_hf_rms_nm=torque_rms,
                  torque_rms_nm=np.sqrt(np.mean(torques[start * raw.frame_skip:] ** 2, axis=0)),
                  mean_abs_delta_action=np.abs(np.diff(actions[start - 1:], axis=0)).mean(axis=0))
    out.update({key: [finite(v) for v in values[key]] for key in FIELDS})
    return out


def combine(values, field):
    """(joints,) the per-joint value of `field` over the rows of values (rows, joints), NaN for none: the root of the
    mean square for RMS_FIELDS, else the mean; NaN where no row has a number."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # a joint with no number in any row: NaN, as wanted
        return np.sqrt(np.nanmean(values ** 2, axis=0)) if field in RMS_FIELDS else np.nanmean(values, axis=0)


def joint_values(runs, scenarios):
    """key -> (joints,) over the scenarios present in `runs` (those that did not fall): per scenario over its seeds,
    then over the scenarios with equal weights; the peaks divide the torque RMS into torque_hf_peak_share and
    torque_rms_peak_share. None if no run stood."""
    stood = [r for r in runs if r['scenario'] in scenarios and r['fell'] is None and 'torque_rms_nm' in r]
    if not stood:
        return None

    def rows(field, name):
        return np.array([[np.nan if v is None else v for v in r[field]] for r in stood if r['scenario'] == name],
                        dtype=np.float64)
    present = [name for name in scenarios if any(r['scenario'] == name for r in stood)]
    out = {field: combine(np.array([combine(rows(field, name), field) for name in present]), field)
           for field in FIELDS}
    peak = np.asarray(stood[0]['peak'])
    out['torque_hf_peak_share'] = out['torque_hf_rms_nm'] / peak
    out['torque_rms_peak_share'] = out.pop('torque_rms_nm') / peak
    return out


def worst(values, names):
    """The summary of per-joint values: each key's worst (largest) joint value, worst_joints (key -> joint name) and
    worst_joint (that of GATED[0]). Only the joints the policy drives count, those whose PD target moves (a finite
    action_hf_share): a locked wrist's target is home throughout, its torque the physics' answer to the body's motion
    (torque_hf_share 0.97-0.99 at 0.17 N m, the same with v0's free wrists), not the policy's."""
    out, which = {}, {}
    driven = np.isfinite(values['action_hf_share'])
    for key in KEYS:
        column = np.where(driven, values[key], np.nan)
        if np.isnan(column).all():
            out[key], which[key] = None, None
            continue
        k = int(np.nanargmax(column))
        out[key], which[key] = float(column[k]), names[k]
    out['worst_joint'], out['worst_joints'] = which[GATED[0]], which
    return out


def summary_of(runs, scenarios=BASE):
    """The summary keys of runs over `scenarios` (None if none of them ran), with runs and falls."""
    present = [r for r in runs if r['scenario'] in scenarios]
    if not present:
        return None
    values = joint_values(present, scenarios)
    out = worst(values, present[0]['names']) if values else {key: None for key in KEYS + ('worst_joint',)}
    out.update(runs=len(present), falls=[dict(scenario=r['scenario'], seed=r['seed'], step=r['fell'])
                                         for r in present if r['fell'] is not None])
    return out


def gate_text():
    return 'худший сустав: %s <= %g x Unitree v0, без падений' % (' и '.join(GATED), GATE_FACTOR)


def judge(summary, reference):
    """(ok, reasons) of a summary against the reference's: a fall, or a GATED key above GATE_FACTOR x the reference."""
    reasons = []
    if summary['falls']:
        reasons.append('падений %d (%s)' % (len(summary['falls']), ', '.join(
            '%s зерно %d' % (SCENARIOS[f['scenario']][0], f['seed']) for f in summary['falls'])))
    for key in GATED:
        mine, theirs = summary.get(key), (reference or {}).get(key)
        if mine is None or theirs is None:
            reasons.append('%s не измерен' % key)
        elif mine > GATE_FACTOR * theirs:
            reasons.append('%s %.3g (%s) > %g x v0 %.3g (%s)' % (
                key, mine, summary['worst_joints'][key], GATE_FACTOR, theirs, reference['worst_joints'][key]))
    return not reasons, reasons


def summarize(runs, reference_runs):
    """{summary, scenarios, carry, joints, reference_joints, reasons} of a stage's runs against the reference's."""
    summary = summary_of(runs)
    reference = summary_of(reference_runs)
    ok, reasons = judge(summary, reference)
    summary.update(reference=reference, ok=ok, gate=gate_text())
    scenarios = {name: summary_of(runs, (name,)) for name in BASE}
    carry = summary_of(runs, ('carry',))
    names = runs[0]['names']

    def listed(values):
        return None if values is None else dict(names=names, **{key: [finite(v) for v in values[key]] for key in KEYS})
    return dict(summary=summary, scenarios=scenarios, carry=carry,
                reference_scenarios={name: summary_of(reference_runs, (name,)) for name in BASE},
                joints=listed(joint_values(runs, BASE)), reference_joints=listed(joint_values(reference_runs, BASE)),
                reasons=reasons)


def jobs(stage, arms=False, seeds=SEEDS, seconds=SECONDS, delay=None, randomize=False, kicks=False, seed=SEED):
    """run() jobs of one policy: the base scenarios, and the carry scenario with arms (an arm_targets stage), seeds each.
    delay None: the stage's task_config action_delay, resolved in run()."""
    names = BASE + (('carry',) if arms else ())
    return [dict(stage=stage, scenario=name, seed=seed + k, seconds=seconds, delay=delay, randomize=randomize,
                 kicks=kicks) for name in names for k in range(seeds)]


def reference_jobs(delay, **kwargs):
    """The reference's run() jobs at the judged stage's delay (s)."""
    return jobs(REFERENCE, arms=False, delay=float(delay), **kwargs)


def number(value, fmt):
    return '-' if value is None else fmt % value


def short(name):
    return '-' if name is None else name.removesuffix('_joint')


TABLE = ('torque_hf_rms_nm', 'torque_hf_peak_share', 'torque_hf_share', 'action_hf_share', 'action_hf_rms_rad',
         'torque_rms_peak_share', 'mean_abs_delta_action')
FORMATS = dict(torque_hf_rms_nm='%.2f', torque_hf_peak_share='%.3f', torque_hf_share='%.3f', action_hf_share='%.3f',
               action_hf_rms_rad='%.4f', torque_rms_peak_share='%.3f', mean_abs_delta_action='%.3f')


def cell(row, key):
    return '%s %s' % (number(row.get(key), FORMATS[key]), short(row['worst_joints'][key])) if row and row.get(
        'worst_joints') else '-'


def print_table(title, rows):
    """rows: [(label, summary)], the worst joint's value and its name per key."""
    print(title)
    print('  %-22s' % '' + ''.join(' %-28s' % LABELS[key] for key in TABLE))
    for text, row in rows:
        falls = ' (упал %d)' % len(row['falls']) if row and row.get('falls') else ''
        print('  %-22s' % (text + falls) + ''.join(' %-28s' % cell(row, key) for key in TABLE))


def line(result):
    """One line for stage_select: the gated keys against v0, the verdict."""
    summary, reference = result['summary'], result['summary']['reference'] or {}
    parts = []
    for key in GATED + tuple(key for key in KEYS[:5] if key not in GATED):
        mine, theirs = summary.get(key), reference.get(key)
        ratio = ' x%.2f' % (mine / theirs) if mine is not None and theirs else ''
        parts.append('%s %s (v0 %s%s)' % (LABELS[key], number(mine, FORMATS[key]), number(theirs, FORMATS[key]),
                                          ratio))
    return 'дрожание моментов (CPU, шум датчиков; судится: %s): %s; худший сустав %s — %s' % (
        summary['gate'], ', '.join(parts), short(summary['worst_joint']), 'ok' if summary['ok'] else 'не ok')


def report(name, result):
    """The tables of one stage against the reference and its verdict."""
    summary = result['summary']
    print('\n== %s' % name)
    print_table('  по сценариям (худший сустав):', [(SCENARIOS[s][0], result['scenarios'][s]) for s in BASE]
                + ([(SCENARIOS['carry'][0], result['carry'])] if result['carry'] else []))
    print_table('  итог по 6 сценариям:', [('этап', summary), ('Unitree v0', summary['reference'])])
    if result['summary']['reference'] is not None:
        print_table('  Unitree v0 по сценариям:', [(SCENARIOS[s][0], result['reference_scenarios'][s]) for s in BASE])
    print('  ' + line(result))
    if result['reasons']:
        print('  причины: ' + '; '.join(result['reasons']))


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Дрожание моментов: ВЧ доля цели PD и момента мотора против Unitree v0')
    parser.add_argument('stage', help='папка этапа или выгрузки export_g1, <эксперимент>[:шаги] или unitree_v0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--seeds', type=int, default=SEEDS)
    parser.add_argument('--delay', type=float, help='задержка действия, с (по умолчанию action_delay этапа)')
    parser.add_argument('--randomize', action='store_true', help='трение стоп и центр масс таза, как в обучении')
    parser.add_argument('--kicks', action='store_true', help='толчки скоростью, как в обучении')
    args = parser.parse_args(argv)
    stage = resolve(args.stage)
    delay = stage_delay(stage) if args.delay is None else args.delay
    common = dict(seeds=args.seeds, seconds=args.seconds, randomize=args.randomize, kicks=args.kicks)
    mine = jobs(stage, has_arm_targets(stage), delay=delay, **common)
    theirs = reference_jobs(delay, **common) if stage != REFERENCE else []
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run, mine + theirs, chunksize=1)
    runs, reference_runs = results[:len(mine)], results[len(mine):] or results[:len(mine)]
    result = summarize(runs, reference_runs)
    print('%s: %d прогонов по %g с, зёрен %d, задержка до %g мс%s%s, %.0f с' % (
        label(stage), len(results), args.seconds, args.seeds, 1000 * delay, ', трение и ЦМ' if args.randomize else '',
        ', толчки' if args.kicks else '', time.time() - started))
    report(label(stage), result)
    if args.json:
        settings = dict(common, delay=delay, scenarios=list(BASE) + (['carry'] if result['carry'] else []),
                        warmup=WARMUP, action_cutoff=ACTION_CUTOFF, torque_cutoff=TORQUE_CUTOFF, seed=SEED,
                        gate=dict(keys=GATED, factor=GATE_FACTOR))
        payload = dict(stage=label(stage), path=stage, settings=settings, runs=runs, reference_runs=reference_runs,
                       **{key: value for key, value in result.items() if key != 'reasons'}, reasons=result['reasons'])
        args.json.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
