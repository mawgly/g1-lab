"""SuperMimic's stage 2 on G1 walking: a walking policy learned only inside a fitted world model (ladder step 1's), then
judged in the real CPU task against candidate 8. Short-horizon BPTT through the model with a critic (SHAC's actor and
TD(λ) critic), no simulator step in training.

    .venv-fullgpu\\Scripts\\python.exe scripts\\lab\\wm\\imagine_walk.py
        runs\\humanoid_lab\\world_model\\ladder-1-walk\\fit-2x-spread --data runs\\humanoid_lab\\world_model\\ladder-1-walk\\data
        --out runs\\humanoid_lab\\world_model\\imagine-walk [--minutes 60] [--ensemble a,b]

The world: the model's canonical G1 states (wm.Layout), its context ξ and the walking task's fields from the data's
starts (candidate 8's transitions: state, ξ, requested command, heading target, yaw-drift EMA, gait clock), then each
imagined step: the observed command by heading hold (task.effective_command, as G1VelocityEnv's observation), the
policy's action clipped to the box with the locked wrists zero (the applied action the model takes), the model's next
state, WalkTerms' reward (the task's own formula without effort and self_contact) and fall, the EMA and the clock on.
Episodes run --episode steps (several horizons, the end of one horizon starts the next), then restart from the data.

The policy's observation is privileged and compact (not the 490-input deploy contract): gravity in the pelvis frame,
the pelvis' linear and angular velocity in its frame, joints − home, joint speeds, the previous applied action, the
observed command, the gait phase (sin, cos; zero under a stand command, as phase_stop). judge() builds the same from the
CPU task's state, runs fixed commands for --judge-seconds and reports falls and the speed reached, beside candidate 8's
own policy in the same runs."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import fit  # noqa: E402

CANDIDATE8 = 'exp-28691193718e4d0e:7221510144'
OBSERVATION, HIDDEN = 101, (512, 256, 128)  # the policy's input size and the actor's / critic's layers
JUDGE_COMMANDS = ((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.), (0., .3, 0.), (0., -.3, 0.), (.5, 0., .5))


class WalkWorld:
    """Imagined walking on a world model (module docstring)."""

    def __init__(self, model, data, reference, device):
        import torch
        from humanoid_sim.world_model_adapters.walk import WalkTerms
        self.model, self.data, self.reference, self.device = model, data, reference, device
        self.terms = WalkTerms(reference, device)
        self.layout = model.layout
        self.limit = float(reference.action_limit)
        locked = list(reference.wrist_joints) if reference.lock_wrists else []
        self.locked = torch.as_tensor(locked, dtype=torch.long, device=device)
        self.home = torch.as_tensor(reference.home, dtype=torch.float32, device=device)
        self.dt, self.period = float(reference.dt), float(reference.gait['period'])
        self.hold = bool(reference.heading_hold)
        self.stand_hold, self.side_hold = bool(reference.stand_heading_hold), bool(reference.side_heading_hold)

    def starts(self, n, generator):
        """n start rows from the data's training worlds: canonical states, ξ, and the task's fields."""
        import torch
        data = self.data
        while True:
            t, w = data.sample(4 * n, 1, generator)
            batch = data.window(t, w, 1, 'cpu')
            keep = batch['alive'][:, 0] & ~batch['terminated'][:, 0]
            if int(keep.sum()) >= n:
                break
        index = torch.nonzero(keep)[:n, 0]
        pick = lambda value: value[index].to(self.device)  # noqa: E731
        state = self.layout.from_raw(pick(batch['state']).float())
        return dict(state=state, context=pick(batch['context']).float(), command=pick(batch['command'][:, 0]).float(),
                    heading=pick(batch['heading_target'][:, 0]).float(),
                    drift=pick(batch['yaw_drift_before'][:, 0]).float(),
                    phase=pick(batch['phase_steps'][:, 0]).long())

    def observed_command(self, state, command, heading):
        from humanoid_sim.tasks import g1_tensor_math as task
        if not self.hold:
            return command
        return task.effective_command(command, self.layout.rotation(state), heading, self.stand_hold, self.side_hold)

    def observation(self, state, observed, phase):
        """(n, 101) the policy's input (module docstring)."""
        import torch
        from humanoid_sim.tasks import g1_tensor_math as task
        layout = self.layout
        rotation = layout.rotation(state)
        angle = 2 * math.pi * task.gait_phase(phase, self.dt, self.period).float()
        moving = task.command_moving(observed).float()[:, None]
        return torch.cat((-rotation[..., 2, :], layout.part(state, 'linear'), layout.part(state, 'angular'),
                          layout.part(state, 'joints') - self.home, layout.part(state, 'speeds'),
                          layout.part(state, 'previous'), observed,
                          moving * torch.stack((angle.sin(), angle.cos()), -1)), -1)

    def applied(self, action):
        import torch
        applied = action.clamp(-self.limit, self.limit)
        if len(self.locked):
            applied = applied.index_fill(1, self.locked, 0.)
        return applied

    def step(self, world, action):
        """(following state, reward, fell) of one imagined step; world's fields move on in place (no gradient)."""
        import torch
        from humanoid_sim.tasks import g1_tensor_math as task
        state = world['state']
        observed = self.observed_command(state, world['command'], world['heading'])
        following = self.model.step(state, self.applied(action), world['context'])
        flat = dict(context=world['context'], observed_command=observed, command=world['command'],
                    heading_target=world['heading'], yaw_drift_before=world['drift'], phase_steps=world['phase'])
        reward, _ = self.terms.reward(following, state, flat)
        fell = self.terms.terminated(following, flat)
        with torch.no_grad():
            velocity = self.terms.velocity(following, world['context'][:, 3:6])
            world['drift'] = task.yaw_drift_update(world['drift'], velocity, observed, self.dt)
            world['phase'] = world['phase'] + 1
        return following, reward, fell


def mlp(inputs, hidden, outputs, device):
    import torch
    layers, width = [], inputs
    for size in hidden:
        layers += [torch.nn.Linear(width, size), torch.nn.ELU()]
        width = size
    layers.append(torch.nn.Linear(width, outputs))
    return torch.nn.Sequential(*layers).to(device)


class Normalizer:
    def __init__(self, sample):
        self.mean, self.std = sample.mean(0), sample.std(0).clamp(min=1e-3)

    def __call__(self, x):
        return ((x - self.mean) / self.std).clamp(-10, 10)


class Learner:
    """SHAC on a WalkWorld: the actor, the critic and its slow target, their Adams, the imagined worlds and their
    ages; update() is one horizon (module docstring). The world (its model and starts) may be swapped between updates,
    as dyna_walk.py's rounds do."""

    def __init__(self, world, normalize, args, device, generator, inputs=OBSERVATION):
        import torch
        self.world, self.normalize, self.args, self.device, self.generator = world, normalize, args, device, generator
        self.inputs = inputs  # the observation's size (imagine_getup.py's differs)
        self.actor = mlp(inputs, HIDDEN, 29, device)
        self.critic = mlp(inputs, HIDDEN, 1, device)
        with torch.no_grad():
            self.actor[-1].weight.mul_(.01)
            self.actor[-1].bias.zero_()
        self.target = mlp(inputs, HIDDEN, 1, device)
        self.target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.updates = 0
        self.restart_all()

    def restart_all(self):
        import torch
        self.worlds = self.world.starts(self.args.worlds, self.generator)
        self.age = torch.zeros(self.args.worlds, dtype=torch.long, device=self.device)

    def observe(self, state):
        world, worlds = self.world, self.worlds
        observed = world.observed_command(state, worlds['command'], worlds['heading'])
        return self.normalize(world.observation(state, observed, worlds['phase']))

    def update(self):
        """One actor step on an imagined horizon from the current worlds, the critic on its TD(λ) targets, the worlds
        on (fallen, non-finite or old ones restart from the data); returns a log line."""
        import torch
        args, world, worlds, device = self.args, self.world, self.worlds, self.device
        n = args.worlds
        state = worlds['state']
        rewards, alive_steps, observations = [], [], []
        alive = torch.ones(n, device=device)
        total = torch.zeros(n, device=device)
        for t in range(args.horizon):
            x = self.observe(state)
            observations.append(x.detach())
            action = self.actor(x) + args.std * torch.randn((n, 29), device=device)
            worlds['state'] = state
            following, reward, fell = world.step(worlds, action)
            reward = torch.nan_to_num(reward, nan=0., posinf=0., neginf=0.)
            total = total + args.gamma ** t * alive * reward
            rewards.append(reward.detach())
            alive_steps.append(alive.detach())
            alive = alive * (~fell).float()
            state = following
        x_last = self.observe(state)
        total = total + args.gamma ** args.horizon * alive * self.target(x_last)[:, 0]
        actor_loss = -(total / args.horizon).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        self.actor_optimizer.step()
        with torch.no_grad():
            values = [self.target(x)[:, 0] for x in observations] + [self.target(x_last.detach())[:, 0]]
            ended = alive.detach()
            returns, upcoming = [], values[-1] * ended
            for t in reversed(range(args.horizon)):
                live_next = alive_steps[t + 1] if t + 1 < args.horizon else ended
                upcoming = rewards[t] + args.gamma * live_next * ((1 - args.lam) * values[t + 1] + args.lam * upcoming)
                returns.append(upcoming)
            returns = torch.stack(returns[::-1])
            inputs = torch.stack(observations)
            weight = torch.stack(alive_steps)
        for _ in range(args.critic_iterations):
            predicted = self.critic(inputs.reshape(-1, self.inputs))[:, 0].reshape(args.horizon, -1)
            critic_loss = ((predicted - returns) ** 2 * weight).sum() / weight.sum().clamp(min=1)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
            self.critic_optimizer.step()
        with torch.no_grad():
            for p, q in zip(self.target.parameters(), self.critic.parameters()):
                p.mul_(.995).add_(.005 * q)
            worlds['state'] = state.detach()
            self.age = self.age + args.horizon
            restart = (alive < .5) | (self.age >= args.episode) | ~torch.isfinite(state).all(1)
            if bool(restart.any()):
                fresh = world.starts(int(restart.sum()), self.generator)
                for key in worlds:
                    worlds[key] = worlds[key].clone()
                    worlds[key][restart] = fresh[key]
                self.age[restart] = 0
        self.updates += 1
        return (f'обновление {self.updates}: награда за шаг {float(torch.stack(rewards).mean()):.3f}, упало в '
                f'воображении {float(1 - alive.mean()):.3f}, ценность {float(values[0].mean()):.2f}, критик '
                f'{float(critic_loss.detach()):.3f}')

    def cpu_policy(self):
        """(actor, normalizer) copies on the CPU, for judge()."""
        import torch
        policy = mlp(self.inputs, HIDDEN, 29, 'cpu')
        policy.load_state_dict({k: v.cpu() for k, v in self.actor.state_dict().items()})
        normalize = Normalizer(torch.zeros(2, self.inputs))
        normalize.mean, normalize.std = self.normalize.mean.cpu(), self.normalize.std.cpu()
        return policy, normalize

    def save(self, path):
        import torch
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        target=self.target.state_dict(), mean=self.normalize.mean, std=self.normalize.std,
                        actor_optimizer=self.actor_optimizer.state_dict(),
                        critic_optimizer=self.critic_optimizer.state_dict(), updates=self.updates), path)

    def load(self, path):
        import torch
        saved = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(saved['actor'])
        self.critic.load_state_dict(saved['critic'])
        self.target.load_state_dict(saved.get('target', saved['critic']))
        self.normalize.mean, self.normalize.std = saved['mean'].to(self.device), saved['std'].to(self.device)
        if 'actor_optimizer' in saved:
            self.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            self.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.updates = int(saved.get('updates', 0))


def judge(policy, normalize, reference_config, seconds, device, candidate=None, seeds=(0,)):
    """The CPU task (candidate 8's task_config, evaluation resets with fixed commands): per command and seed whether
    it fell, the pelvis' mean forward / sideways speed and yaw rate over the last two thirds, for `policy` (our
    observation) and for candidate 8's own policy (SB3 on its observation) when `candidate` is given."""
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_velocity', 'g1_29dof', reference_config).unwrapped
    layout = wm.Layout(env.model.nu)
    home = torch.as_tensor(env.home, dtype=torch.float32)
    rows = []
    for name, act in (('мир', 'ours'), ('кандидат 8', 'candidate')):
        if act == 'candidate' and candidate is None:
            continue
        for command in JUDGE_COMMANDS:
            for seed in seeds:
                observation, _ = env.reset(seed=seed, options=dict(command=np.array(command)))
                steps, fell, speeds = int(seconds / env.dt), False, []
                for step in range(steps):
                    if act == 'ours':
                        raw = torch.as_tensor(np.concatenate([env.data.qpos, env.data.qvel, env.previous_action]),
                                              dtype=torch.float32)[None]
                        state = layout.from_raw(raw)
                        rotation = layout.rotation(state)
                        phase = torch.tensor([env.steps + 1 + env.phase_steps])
                        angle = 2 * math.pi * task.gait_phase(phase, env.dt, env.gait['period']).float()
                        observed = torch.as_tensor(env.observed_command, dtype=torch.float32)[None]
                        moving = task.command_moving(observed).float()[:, None]
                        x = torch.cat((-rotation[..., 2, :], layout.part(state, 'linear'),
                                       layout.part(state, 'angular'), layout.part(state, 'joints') - home,
                                       layout.part(state, 'speeds'), layout.part(state, 'previous'), observed,
                                       moving * torch.stack((angle.sin(), angle.cos()), -1)), -1).to(device)
                        with torch.no_grad():
                            action = policy(normalize(x))[0].clamp(-env.action_limit, env.action_limit).cpu().numpy()
                    else:
                        action = candidate(observation)
                    observation, _, terminated, truncated, info = env.step(action)
                    if step >= steps // 3:
                        rotation = env.data.xmat[env.pelvis].reshape(3, 3)
                        heading = math.atan2(rotation[1, 0], rotation[0, 0])
                        v = env.data.qvel[:3]
                        speeds.append((math.cos(heading) * v[0] + math.sin(heading) * v[1],
                                       -math.sin(heading) * v[0] + math.cos(heading) * v[1], env.data.qvel[5]))
                    if terminated:
                        fell = True
                        break
                mean = np.mean(speeds, 0) if speeds else np.full(3, np.nan)
                rows.append(dict(policy=name, command=list(command), seed=seed, fell=fell, seconds=(step + 1) * env.dt,
                                 vx=float(mean[0]), vy=float(mean[1]), wz=float(mean[2])))
    env.close()
    return rows


def candidate_policy():
    from humanoid_sim.core import load_stage
    sys.path.insert(0, str(HERE))
    import collect
    source = collect.resolve(CANDIDATE8)
    model, norm = load_stage(source.path)

    def act(observation):
        seen = norm.normalize_obs(observation[None])
        return model.predict(seen, deterministic=True)[0][0]
    return act, source.config


def summary(rows):
    lines = []
    for row in rows:
        c = row['command']
        lines.append(f"{row['policy']:10s} команда ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}) зерно {row['seed']}: "
                     + ('УПАЛ на ' + f"{row['seconds']:.1f} с" if row['fell'] else 'стоит/идёт')
                     + f", vx {row['vx']:+.2f}, vy {row['vy']:+.2f}, ωz {row['wz']:+.2f}")
    return lines


def main():
    import torch
    from humanoid_sim import world_model as wm
    parser = argparse.ArgumentParser(description='политика ходьбы G1, обученная только внутри модели мира')
    parser.add_argument('fit', help='папка fit.py с model.pt (или первая модель ансамбля)')
    parser.add_argument('--data', required=True, help='данные collect.py для стартов (ходьба кандидата 8)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--ensemble', default=None, help='ещё папки fit.py через запятую: шаг — их среднее')
    parser.add_argument('--minutes', type=float, default=60.)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--episode', type=int, default=128, help='шагов воображаемого эпизода до нового старта')
    parser.add_argument('--worlds', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--lam', type=float, default=.95)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--critic-iterations', type=int, default=8)
    parser.add_argument('--std', type=float, default=.3, help='шум действий в обучении (единицы действия)')
    parser.add_argument('--judge-every', type=float, default=10., help='минут между проверками в симуляторе')
    parser.add_argument('--judge-seconds', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out, device = Path(args.out), args.device
    out.mkdir(parents=True, exist_ok=True)
    folders = [Path(args.fit)] + ([Path(f) for f in args.ensemble.split(',')] if args.ensemble else [])
    if len(folders) > 1:
        model = wm.Ensemble.load([f / 'model.pt' for f in folders], device)
    else:
        model, _ = wm.WorldModel.load(folders[0] / 'model.pt', device)
    if model.history > 1:
        raise SystemExit('модели с историей пока не поддержаны в воображении ходьбы')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta, arrays = fit.load(args.data)
    adapter = fit.adapter_of_meta(meta)
    data = fit.Data(arrays, 'cpu', .1, adapter)
    del arrays
    reference = adapter.reference(meta)
    world = WalkWorld(model, data, reference, device)
    generator = torch.Generator().manual_seed(args.seed)
    first = world.starts(8192, generator)
    observed = world.observed_command(first['state'], first['command'], first['heading'])
    normalize = Normalizer(world.observation(first['state'], observed, first['phase']))
    learner = Learner(world, normalize, args, device, generator)
    candidate, config = candidate_policy()
    curve, started, judged = [], time.time(), -1e9
    log = open(out / 'train.log', 'a', encoding='utf-8')

    def say(text):
        print(text, flush=True)
        log.write(text + '\n')
        log.flush()
    say(f'модель: {", ".join(map(str, folders))}; старты из {args.data}; миров {args.worlds}, горизонт {args.horizon}')
    while True:
        elapsed = time.time() - started
        if elapsed - judged >= 60 * args.judge_every or elapsed >= 60 * args.minutes:
            judged = elapsed
            policy_cpu, norm_cpu = learner.cpu_policy()
            rows = judge(policy_cpu, norm_cpu, config, args.judge_seconds, 'cpu', candidate if not curve else None)
            if not curve:
                say('\n'.join(line for line in summary(rows) if line.startswith('кандидат')))
            ours = [row for row in rows if row['policy'] == 'мир']
            say(f'обновлений {learner.updates}, {elapsed / 60:.1f} мин: в симуляторе упал '
                f'{sum(row["fell"] for row in ours)} из {len(ours)}')
            say('\n'.join(summary(ours)))
            curve.append(dict(updates=learner.updates, minutes=elapsed / 60, rows=rows))
            learner.save(out / 'policy.pt')
            (out / 'curve.json').write_text(json.dumps(dict(curve=curve, args=vars(args)), ensure_ascii=False,
                                                       indent=1), encoding='utf-8')
            if elapsed >= 60 * args.minutes:
                break
        line = learner.update()
        if learner.updates % 50 == 0:
            say(line + f', {(time.time() - started) / 60:.1f} мин')


if __name__ == '__main__':
    main()
