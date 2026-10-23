import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from arm_sim.env import ArmReachEnv, ROOT

DEFAULT_RUN = ROOT / 'runs' / 'so101'


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise ValueError('Must be a positive integer')
    return result


def load_policy(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Run train.cmd first.')
    return PPO.load(path, device='cpu')


def evaluate_policy(policy, episodes=100, seed=10000, max_steps=150):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    env = ArmReachEnv(max_steps=max_steps)
    records = []
    try:
        for episode in range(episodes):
            obs, initial = env.reset(seed=seed + episode)
            rng = np.random.default_rng(seed + episode)
            total = 0.0
            for step in range(max_steps):
                action = rng.uniform(-1, 1, 5) if policy is None else policy.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            records.append({'seed': seed + episode, 'success': info['is_success'],
                            'initial_distance_m': initial['distance'], 'final_distance_m': info['distance'],
                            'steps': step + 1, 'return': total})
    finally:
        env.close()
    return {'success_rate': float(np.mean([r['success'] for r in records])),
            'mean_final_distance_m': float(np.mean([r['final_distance_m'] for r in records])),
            'episodes': records}


def train(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = make_vec_env(ArmReachEnv, n_envs=args.envs, seed=args.seed,
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else None,
                       monitor_dir=str(args.run_dir / 'monitor'))
    evaluation = make_vec_env(ArmReachEnv, n_envs=1, seed=5000,
                              vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    metadata = {'algorithm': 'PPO', 'seed': args.seed, 'steps_requested': args.steps,
                'envs': args.envs, 'success_tolerance_m': 0.02, 'success_hold_steps': 5,
                'model_revision': (ROOT / 'models/so101/UPSTREAM_REVISION.txt').read_text().strip(),
                'observation': 'sin(q),cos(q),qvel*0.1,target*3,tip*3,(target-tip)*3',
                'action': 'five joint position increments, max 0.08 rad per 40 ms',
                'resume': str(args.resume) if args.resume else None}
    (args.run_dir / 'training_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    model = (PPO.load(args.resume, env=env, device='cpu') if args.resume else
             PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=512, batch_size=256,
                 n_epochs=10, gamma=0.98, gae_lambda=0.95, ent_coef=0.0,
                 policy_kwargs={'net_arch': [128, 128]}, device='cpu', seed=args.seed, verbose=1))
    model.set_logger(configure(str(args.run_dir), ['stdout', 'csv']))
    callbacks = [EvalCallback(evaluation, best_model_save_path=str(args.run_dir),
                              log_path=str(args.run_dir), eval_freq=max(1, 50000 // args.envs),
                              n_eval_episodes=25, deterministic=True),
                 CheckpointCallback(save_freq=max(1, 100000 // args.envs), save_path=str(args.run_dir / 'checkpoints'))]
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=10,
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('Interrupted: saving current model.', flush=True)
    finally:
        model.save(args.run_dir / 'last_model')
        env.close()
        evaluation.close()
    print(f'Saved: {args.run_dir / "last_model.zip"}', flush=True)


def evaluate(args):
    torch.set_num_threads(2)
    if not args.model.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.model}. Run train.cmd first.')
    checkpoint = args.model.read_bytes()
    policy = PPO.load(io.BytesIO(checkpoint), device='cpu')
    result = {'model': str(args.model), 'seed': args.seed,
              'checkpoint_sha256': hashlib.sha256(checkpoint).hexdigest(),
              'training_timesteps': policy.num_timesteps,
              'trained': evaluate_policy(policy, args.episodes, args.seed),
              'random': evaluate_policy(None, args.episodes, args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for label in ('trained', 'random'):
        r = result[label]
        print(f'{label}: success={r["success_rate"]:.1%}, mean final distance={r["mean_final_distance_m"] * 100:.2f} cm')
    print(f'Report: {args.output}')


def view(args):
    import mujoco.viewer
    from PIL import Image
    torch.set_num_threads(2)
    env = ArmReachEnv()
    policy = None if args.manual else load_policy(args.model)
    obs, _ = env.reset(seed=args.seed)
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(args.snapshot)
        print(f'Snapshot: {args.snapshot}')
        env.close()
        return
    start = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = env.camera.lookat
            viewer.cam.distance = env.camera.distance
            viewer.cam.azimuth = env.camera.azimuth
            viewer.cam.elevation = env.camera.elevation
            viewer.sync()
            print('Viewer ready. Close the window to exit.' + (' Use Control sliders.' if args.manual else ''), flush=True)
            episode = 0
            while viewer.is_running() and (not args.seconds or time.monotonic() - start < args.seconds):
                tick = time.monotonic()
                reset_pending = False
                with viewer.lock():
                    if args.manual:
                        mujoco.mj_step(env.model, env.data, nstep=20)
                    else:
                        action = policy.predict(obs, deterministic=True)[0]
                        obs, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            print(f'Target {episode + 1}: success={info["is_success"]}, distance={info["distance"] * 100:.1f} cm', flush=True)
                            episode += 1
                            reset_pending = True
                viewer.sync()
                if reset_pending:
                    time.sleep(0.6)
                    with viewer.lock():
                        obs, _ = env.reset(seed=args.seed + episode)
                time.sleep(max(0, 0.04 - (time.monotonic() - tick)))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='SO-ARM101: MuJoCo reaching experiment')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--steps', type=positive_int, default=1000000)
    p.add_argument('--envs', type=positive_int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    p.add_argument('--resume', type=Path)
    p.set_defaults(func=train)
    p = sub.add_parser('evaluate')
    p.add_argument('--model', type=Path, default=DEFAULT_RUN / 'best_model.zip')
    p.add_argument('--episodes', type=positive_int, default=100)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--output', type=Path, default=DEFAULT_RUN / 'evaluation.json')
    p.set_defaults(func=evaluate)
    p = sub.add_parser('view')
    p.add_argument('--model', type=Path, default=DEFAULT_RUN / 'best_model.zip')
    p.add_argument('--manual', action='store_true')
    p.add_argument('--seed', type=int, default=20000)
    p.add_argument('--seconds', type=positive_int)
    p.add_argument('--snapshot', type=Path)
    p.set_defaults(func=view)
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
