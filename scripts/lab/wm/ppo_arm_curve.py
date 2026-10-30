"""PPO's curve on the arm in simulator steps, for dyna_arm.py: arm_sim.cli's PPO (the same hyperparameters, 8
environments), judged every --every steps on runs/so101's 100 seeds with the policy's mean (arm_sim.cli.evaluate_policy).

    .venv\Scripts\python.exe scripts\lab\wm\ppo_arm_curve.py --out runs\humanoid_lab\world_model\ppo-arm
        [--seeds 42,43,44] [--steps 200000] [--every 10000]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from arm_sim.cli import evaluate_policy
    from arm_sim.env import ArmReachEnv
    parser = argparse.ArgumentParser(description='кривая PPO руки по шагам симулятора')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--every', type=int, default=10_000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    runs = {}
    for seed in (int(s) for s in args.seeds.split(',')):
        curve, started = [], time.time()

        class Judge(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next = args.every

            def _on_step(self):
                if self.num_timesteps >= self.next:
                    self.next += args.every
                    result = evaluate_policy(self.model, 100, 10000)
                    steps = [e['steps'] for e in result['episodes'] if e['success']]
                    curve.append(dict(simulator_steps=self.num_timesteps, success_rate=result['success_rate'],
                                      mean_steps=sum(steps) / len(steps) if steps else None,
                                      final_distance_m=result['mean_final_distance_m'],
                                      seconds=round(time.time() - started)))
                    print(f'PPO seed {seed}: {self.num_timesteps} шагов, успех {result["success_rate"]:.2f}',
                          flush=True)
                return True
        env = make_vec_env(ArmReachEnv, n_envs=8, seed=seed, vec_env_cls=SubprocVecEnv)
        model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10, gamma=0.98,
                    gae_lambda=0.95, ent_coef=0.0, policy_kwargs={'net_arch': [128, 128]}, device='cpu', seed=seed)
        try:
            model.learn(total_timesteps=args.steps, callback=Judge())
        finally:
            env.close()
        runs[seed] = curve
        (out / 'curve.json').write_text(json.dumps(dict(runs=runs, args=vars(args)), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
