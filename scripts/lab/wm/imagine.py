"""SuperMimic's stage 2 on the arm: a policy learned only inside a fitted world model (no simulator step in training),
then judged in the real task. The first check that the world model is good enough to learn from, before the G1.

    .venv-fullgpu\\Scripts\\python.exe scripts\\lab\\wm\\imagine.py runs\\humanoid_lab\\world_model\\loss-units-arm\\spread
        --out runs\\humanoid_lab\\world_model\\imagine-arm [--minutes 10] [--horizon 32] [--device cuda]

Starts: ArmReachEnv.reset states (qpos, qvel, target), drawn once on the CPU. Policy: MLP 256×256 ELU → tanh of a mean
plus a learned σ (reparameterised noise in training, the mean when judged), the observation ArmReachEnv._observation
made from the state (the tip by Kinematics.points). Imagination: the applied action u = clip(q + 0.08·a, ctrlrange) − q
of the five axes and home − q of the gripper (ArmReachEnv.step), the world model's step, ArmTerms' reward (the task's
own formula); the hold counter is carried without a gradient and an episode's rewards after its success are masked.
Loss: −mean discounted return over the horizon, back-propagated through the model (no critic: the task ends in ~10
steps, inside the horizon). Judged every --eval-seconds in ArmReachEnv on 100 seeds from 10000, as
runs/so101/evaluation.json (PPO, 950 k env steps: success 1.0)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EVAL_SEEDS = range(10000, 10100)  # runs/so101/evaluation.json's seeds


def make_env():
    from humanoid_sim.tasks.catalog import make_task
    return make_task('arm_reach', 'so101', {}).unwrapped


def draw_starts(env, count, seed):
    qpos, qvel, target = [], [], []
    for index in range(count):
        env.reset(seed=seed + index)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        target.append(env.target.copy())
    return np.array(qpos), np.array(qvel), np.array(target)


class Policy:
    def __init__(self, device):
        import torch
        self.net = torch.nn.Sequential(torch.nn.Linear(24, 256), torch.nn.ELU(), torch.nn.Linear(256, 256),
                                       torch.nn.ELU(), torch.nn.Linear(256, 5)).to(device)
        self.log_std = torch.nn.Parameter(torch.full((5,), -1., device=device))

    def parameters(self):
        return [*self.net.parameters(), self.log_std]

    def act(self, observation, noise=None):
        import torch
        mean = self.net(observation)
        return torch.tanh(mean if noise is None else mean + self.log_std.exp() * noise)


class Imagination:
    """The arm task on world-model states (module docstring)."""

    def __init__(self, model, env, device):
        import torch
        from humanoid_sim.world_model_adapters.arm import ArmTerms
        self.model, self.device = model, device
        self.terms = ArmTerms(env, device)
        limits = env.model.actuator_ctrlrange[:5]
        self.low = torch.as_tensor(limits[:, 0], dtype=torch.float32, device=device)
        self.high = torch.as_tensor(limits[:, 1], dtype=torch.float32, device=device)
        self.gripper_home = float(env.home[5])

    def tip(self, state):
        terms = self.terms
        return terms.kinematics.points(state, terms.layout, terms.tip_body, terms.tip_local)[:, 0]

    def observation(self, state, target):
        import torch
        q, v, tip = state[:, :5], state[:, 6:11], self.tip(state)
        return torch.cat((q.sin(), q.cos(), .1 * v, 3 * target, 3 * tip, 3 * (target - tip)), -1)

    def applied(self, state, action):
        import torch
        q = state[:, :6]
        axes = torch.minimum(torch.maximum(q[:, :5] + .08 * action, self.low), self.high) - q[:, :5]
        return torch.cat((axes, self.gripper_home - q[:, 5:6]), -1)

    def returns(self, policy, state, target, horizon, gamma, noise=True):
        """(the mean discounted return, the share that succeeded, the final distance) of imagined episodes."""
        import torch
        n = len(state)
        context = torch.zeros((n, self.model.settings['context']), device=self.device)
        hold = torch.zeros(n, dtype=torch.long, device=self.device)
        alive = torch.ones(n, device=self.device)
        total = torch.zeros(n, device=self.device)
        for t in range(horizon):
            action = policy.act(self.observation(state, target),
                                torch.randn((n, 5), device=self.device) if noise else None)
            following = self.model.step(state, self.applied(state, action), context)
            reward, _ = self.terms.reward(following, state, dict(target=target, hold_before=hold,
                                                                  policy_action=action))
            total = total + gamma ** t * alive * reward
            with torch.no_grad():
                distance = self.terms.distance(following, target)
                hold = (hold + 1) * (distance < .02)
                alive = alive * (hold < 5)
            state = following
        return total.mean(), float(1 - alive.mean()), float(self.terms.distance(state, target).detach().mean())


def judge(policy, env, device):
    """ArmReachEnv on EVAL_SEEDS with the policy's mean: success rate, mean steps of the successes, final distance."""
    import torch
    successes, steps, distances = 0, [], []
    for seed in EVAL_SEEDS:
        observation, info = env.reset(seed=seed)
        for step in range(env.max_steps):
            with torch.no_grad():
                action = policy.act(torch.as_tensor(observation, device=device)[None])[0].cpu().numpy()
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += bool(info['is_success'])
        if info['is_success']:
            steps.append(step + 1)
        distances.append(info['distance'])
    return dict(success_rate=successes / len(EVAL_SEEDS), mean_steps=float(np.mean(steps)) if steps else None,
                final_distance_m=float(np.mean(distances)))


def main():
    import torch
    from humanoid_sim import world_model as wm
    parser = argparse.ArgumentParser(description='политика руки, обученная только внутри модели мира')
    parser.add_argument('fit', help='папка fit.py с model.pt (модель мира руки)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--minutes', type=float, default=10.)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--starts', type=int, default=8192)
    parser.add_argument('--eval-seconds', type=float, default=60.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    model, _ = wm.WorldModel.load(Path(args.fit) / 'model.pt', device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    env = make_env()
    started = time.time()
    qpos, qvel, target = draw_starts(env, args.starts, 1_000_000 + args.seed)  # far from the judged seeds
    print(f'стартов {args.starts} за {time.time() - started:.0f} с', flush=True)
    states = torch.as_tensor(np.concatenate((qpos, qvel), 1), dtype=torch.float32, device=device)
    targets = torch.as_tensor(target, dtype=torch.float32, device=device)
    world = Imagination(model, env, device)
    policy = Policy(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    curve, best, steps, started, judged = [], None, 0, time.time(), 0.
    while True:
        elapsed = time.time() - started
        if elapsed - judged >= args.eval_seconds or elapsed >= 60 * args.minutes:
            judged = elapsed
            result = judge(policy, env, device)
            with torch.no_grad():
                pick = torch.randint(0, len(states), (args.batch,), generator=generator).to(device)
                _, imagined, distance = world.returns(policy, states[pick], targets[pick], args.horizon, args.gamma,
                                                      noise=False)
            result.update(updates=steps, seconds=round(elapsed), imagined_success=imagined,
                          imagined_distance_m=distance, std=float(policy.log_std.exp().mean()))
            curve.append(result)
            print(f'обновлений {steps}, {elapsed:.0f} с: в симуляторе успех {result["success_rate"]:.2f}, шагов '
                  f'{result["mean_steps"]}, конечное расстояние {1000 * result["final_distance_m"]:.1f} мм; в модели '
                  f'успех {imagined:.2f}, {1000 * distance:.1f} мм; σ {result["std"]:.3f}', flush=True)
            if best is None or (result['success_rate'], -result['final_distance_m']) > (
                    best['success_rate'], -best['final_distance_m']):
                best = result
                torch.save(dict(net=policy.net.state_dict(), log_std=policy.log_std.detach().cpu()), out / 'policy.pt')
            (out / 'curve.json').write_text(json.dumps(dict(curve=curve, best=best, args=vars(args)),
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
            if elapsed >= 60 * args.minutes:
                break
        pick = torch.randint(0, len(states), (args.batch,), generator=generator).to(device)
        value, _, _ = world.returns(policy, states[pick], targets[pick], args.horizon, args.gamma)
        optimizer.zero_grad()
        (-value / args.horizon).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        steps += 1
    print(f'лучшее: {json.dumps(best, ensure_ascii=False)}')


if __name__ == '__main__':
    main()
