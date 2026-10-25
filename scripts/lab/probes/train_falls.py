"""Falls in training conditions, CPU (the critic's lab-critic/it40b/train_falls.py, iteration 40b; the gravity oracle
from lab-critic/it42/tf_ff.py, iteration 42):
train_falls.py <stage | experiment[:steps]> [--episodes 400] [--workers 4] [--conditions free,home:0,target:0.5,...]
[--ff-beta 1] [--seed 90000] [--json <file>]

Episodes as the full-GPU trainer gives them: training resets (no command: commands resampled, random phase, crouch,
force pushes, motor friction, arm perturbation as the stage's task_config), the stage's stochastic policy (its σ, the
training's action noise), 1000 control steps, with what the CUDA task adds at training_realism 1 (g1_tensor_env):
sensor noise on every frame (NOISE, uniform; angular velocity, gravity, joint angles and speeds), an encoder bias per
episode (ENCODER_BIAS), the feet's friction U(FOOT_FRICTION_RANGE) and a pelvis COM offset (COM_OFFSET) per episode,
velocity kicks REAL_PUSH every REAL_PUSH_INTERVAL s on average. Conditions: free (flag 0, free arms), home:<kg> (flag 1,
targets home), target:<kg> (flag 1, the training's targets), the load per hand fixed (set_hand_load); by default free,
home 0 / 0.5 / 1 kg and target 0 / 0.5 / 1 kg for an arm_targets stage, free alone for another. The mode is forced at
the training reset's own draw (task.arm_mode_draw), the same seeds in every condition. home_ff:<kg> / target_ff:<kg>
are the same episodes with an oracle on the arms: each step adds beta qfrc_bias / (kp scale) to the ten tracked
joints' actions (gravity_action, beta --ff-beta), the static torque the PD lacks with the loads included, so the
policy's sag goes (no integrator: no windup when the targets move; the action noise and the moving targets stay;
stage 9111 of branch 41 at 1 kg: arm |error| 0.158 -> 0.118, shoulder pitch +0.175 -> +0.084). The legs are the
policy's own, trained on its sag: when its falls come down to the oracle's, the legs have learned to carry tracked
arms. Prints per condition the episodes, falls and falls per 1000-step episode (falls / (steps / 1000)) with
the exact 95 % Poisson interval, and with flag 1 the arms' mean |error| and the shoulder pitch's mean q - target (rad,
both arms; + is a sag)."""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EPISODE_STEPS = 1000
DEFAULT_ARMS = ('free', 'home:0', 'home:0.5', 'home:1', 'target:0', 'target:0.5', 'target:1')
_cache = {}


def resolve(spec):
    """A stage folder, or <experiment>[:steps] of the lab under runs/ (its latest stage)."""
    path = Path(spec)
    if path.is_dir():
        return path
    from humanoid_sim.core import DEFAULT_RUN, list_stages
    from humanoid_sim.lab import Laboratory
    name, _, steps = spec.partition(':')
    run = Laboratory(DEFAULT_RUN).resolve(name)
    return run / f'stage_{int(steps):012d}' if steps else list_stages(run)[-1]


def condition(text):
    """(mode, kg, oracle) of 'free', 'home:<kg>' / 'target:<kg>' or, with the gravity oracle, 'home_ff:<kg>' /
    'target_ff:<kg>'."""
    name, _, load = text.partition(':')
    mode = name.removesuffix('_ff')
    if not (text == 'free' or (mode in ('home', 'target') and load)):
        raise ValueError(f'condition: free, home[_ff]:<kg> or target[_ff]:<kg>, not {text!r}')
    return mode, float(load or 0.), name != mode


def gravity_action(raw, beta=1.):
    """(tracked joints,) the oracle's addition to their actions: beta qfrc_bias / (kp scale) of this state, so the PD
    target moves by the static torque (gravity, the loads included) over kp."""
    tracked = list(raw.track_joints)
    gain = beta / (np.asarray(raw.kp)[tracked] * raw.scale[tracked])
    return gain * raw.data.qfrc_bias[np.asarray(raw.vadr)[tracked]]


def setup(stage):
    """This worker's policy, normalization, environment, its observation as built and its nominal pelvis COM, once per
    stage (a worker given another stage builds that one's; scripts/lab/skills_check.py runs two in one pool); the arm
    mode draw wrapped once, so that a job's condition fixes it (_cache['mode'], None: the draw's own)."""
    if _cache.get('stage') != Path(stage):
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.core import load_stage, stage_task_config
        from humanoid_sim.tasks import g1_tensor_math as task
        from humanoid_sim.tasks.catalog import make_task
        if 'env' in _cache:
            _cache['env'].close()
        model, norm = load_stage(stage)
        env = make_task('g1_velocity', 'g1_29dof', {**stage_task_config(stage), 'max_episode_steps': EPISODE_STEPS})
        raw = env.unwrapped
        if not _cache.get('wrapped'):
            original = task.arm_mode_draw

            def draw(uniform, n):  # the training reset's own draw, then the forced mode
                out = original(uniform, n)
                return out if _cache['mode'] is None else torch.full_like(out, _cache['mode'])
            task.arm_mode_draw = draw
        _cache.update(stage=Path(stage), env=env, model=model, norm=norm, mode=None, observation=raw._observation,
                      com=raw.model.body_ipos[raw.pelvis].copy(), wrapped=True)
    return _cache


def realism(raw, rng, bias):
    """raw._observation with the CUDA task's sensor noise on the newest frame (every history frame after a fill) and the
    encoder bias on the joint angles; the critic extras exact, as there."""
    import torch
    from humanoid_sim.g1_tensor_env import NOISE
    from humanoid_sim.tasks import g1_tensor_math as task
    clean = _cache['observation']
    nu, extra = raw.model.nu, raw.extra_sizes
    spans = task.deploy_terms(nu, extra)
    terms = ((spans[0], NOISE['angular_velocity']), (spans[1], NOISE['gravity']), (spans[4], NOISE['joint_position']),
             (spans[5], NOISE['joint_velocity']))

    def noisy(frame):
        for (start, end), scale in terms:
            frame[start:end] += torch.from_numpy((2 * rng.random(end - start) - 1) * scale).to(frame.dtype)
        frame[spans[4][0]:spans[4][1]] += torch.from_numpy(bias).to(frame.dtype)
        return frame

    def observation(history='replace'):
        obs = clean(history)
        if raw.frames is None:  # one frame: the observation itself
            obs[:raw.frame_size] = noisy(torch.from_numpy(obs[:raw.frame_size].astype(np.float64))).numpy()
            return obs
        frame = noisy(raw.frames[0, -1].clone())
        if history == 'fill':
            raw.frames[0] = frame
        else:
            raw.frames[0, -1] = frame
        obs[:raw.actor_inputs] = task.history_observation(raw.frames, nu, extra)[0].numpy()
        return obs
    return observation


def run(job):
    import torch
    from humanoid_sim.g1_tensor_env import COM_OFFSET, ENCODER_BIAS, FOOT_FRICTION_RANGE, REAL_PUSH, REAL_PUSH_INTERVAL
    from humanoid_sim.tasks import g1_tensor_math as task
    cache = setup(Path(job['stage']))
    env, model, norm = cache['env'], cache['model'], cache['norm']
    raw = env.unwrapped
    mode, load, oracle = condition(job['condition'])
    arms = raw.arm_targets
    cache['mode'] = {'free': task.ARM_FREE, 'home': task.ARM_HOME, 'target': task.ARM_TARGET}[mode] if arms else None
    seed = job['seed']
    rng = np.random.default_rng((seed, 404))
    torch.manual_seed(seed)  # the policy's action noise
    m = raw.model
    m.geom_friction[raw.foot_geoms, 0] = rng.uniform(*FOOT_FRICTION_RANGE)
    m.body_ipos[raw.pelvis] = cache['com'] + (2 * rng.random(3) - 1) * np.asarray(COM_OFFSET)
    raw._observation = realism(raw, rng, (2 * rng.random(m.nu) - 1) * ENCODER_BIAS)
    if arms:
        raw.load_max = 0.
    obs, _ = env.reset(seed=seed)
    if arms and load and raw.arm_flag:
        raw.set_hand_load(load)
    tracked = list(raw.track_joints)
    names = [raw.model.joint(j).name for j in raw.joints]
    pitch = [tracked.index(names.index(f'{side}_shoulder_pitch_joint')) for side in ('left', 'right')] if arms else []
    beta = job.get('beta', 1.) if oracle and arms and raw.arm_flag else 0.
    steps, fell, errors, shoulder = 0, False, [], []
    for _ in range(EPISODE_STEPS):
        if rng.random() < raw.dt / REAL_PUSH_INTERVAL:
            raw.data.qvel[:6] += (2 * rng.random(6) - 1) * np.asarray(REAL_PUSH)
        action = model.predict(norm.normalize_obs(obs), deterministic=False)[0]
        if beta:
            action = action.copy()
            action[tracked] += gravity_action(raw, beta)
        obs, _, terminated, truncated, info = env.step(action)
        if info.get('arm_error') is not None:
            errors.append(info['arm_error'])
            error = raw.data.qpos[raw.qadr][tracked] - (raw.home[tracked] + raw.arm_offset[tracked])
            shoulder.append(float(np.mean(error[pitch])))
        steps += 1
        if terminated:
            fell = True
            break
    return dict(condition=job['condition'], seed=seed, steps=steps, fell=fell, beta=beta,
                mode=int(getattr(raw, 'arm_mode', 0)), hand_load=[float(v) for v in raw.hand_load],
                error=float(np.mean(errors)) if errors else None, shoulder=float(np.mean(shoulder)) if shoulder else None)


def poisson_interval(falls, exposure):
    """Exact 95 % interval of a Poisson rate: falls over `exposure` episodes of EPISODE_STEPS."""
    def cdf(k, rate):
        return sum(math.exp(-rate + i * math.log(rate) - math.lgamma(i + 1)) for i in range(k + 1)) if rate > 0 else 1.

    def bisect(holds, low=1e-9, high=1e4):
        for _ in range(100):
            middle = (low + high) / 2
            low, high = (middle, high) if holds(middle) else (low, middle)
        return low
    lower = bisect(lambda rate: 1 - cdf(falls - 1, rate) < .025) / exposure if falls else 0.
    upper = bisect(lambda rate: cdf(falls, rate) > .025) / exposure
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage')
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--conditions', help='comma-separated: free, home[_ff]:<kg>, target[_ff]:<kg>')
    parser.add_argument('--ff-beta', type=float, default=1., help='the gravity oracle\'s share of qfrc_bias (_ff)')
    parser.add_argument('--seed', type=int, default=90000)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    stage = resolve(args.stage)
    from humanoid_sim.core import stage_task_config
    arms = stage_task_config(stage).get('arm_targets')
    conditions = args.conditions.split(',') if args.conditions else list(DEFAULT_ARMS if arms else ('free',))
    for text in conditions:
        mode, _, _ = condition(text)
        if mode != 'free' and not arms:
            raise SystemExit(f'{stage}: not an arm_targets stage, only the free condition')
    jobs = [dict(stage=str(stage), condition=text, seed=args.seed + k, beta=args.ff_beta) for text in conditions
            for k in range(args.episodes)]
    started = time.time()
    with Pool(args.workers) as pool:
        runs = pool.map(run, jobs, chunksize=1)
    print(f'{stage}: {args.episodes} episodes per condition, {time.time() - started:.0f} s')
    print('condition    episodes falls  per 1000-step episode [95 %]   arm |e|  shoulder pitch e')
    for text in conditions:
        rows = [r for r in runs if r['condition'] == text]
        falls, exposure = sum(r['fell'] for r in rows), sum(r['steps'] for r in rows) / EPISODE_STEPS
        lower, upper = poisson_interval(falls, exposure)
        line = f'{text:<12} {len(rows):8d} {falls:5d}  {100 * falls / exposure:6.2f}% [{100 * lower:.2f}-{100 * upper:.2f}]'
        errors = [r['error'] for r in rows if r['error'] is not None]
        if errors:
            line += (f'{"":>{max(0, 51 - len(line))}}{np.mean(errors):7.3f}  '
                     f'{np.mean([r["shoulder"] for r in rows if r["shoulder"] is not None]):+.3f}')
        print(line)
    if args.json:
        args.json.write_text(json.dumps(runs, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
