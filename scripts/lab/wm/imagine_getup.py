"""SuperMimic's stage 2 on getting up by tracking one clip: a tracker learned only inside the getting-up world model
(stage 570 of branch 73's data, A493 v3, fit-spread), judged in the CPU task beside the PPO tracker it was collected
with. imagine_walk.Learner (BPTT over --horizon steps and a TD(λ) critic) on GetupWorld.

    .venv-fullgpu\\Scripts\\python.exe scripts\\lab\\wm\\imagine_getup.py
        runs\\humanoid_lab\\world_model\\exp-84b0540b0092467d-570\\fit-spread
        --data runs\\humanoid_lab\\world_model\\exp-84b0540b0092467d-570\\data --out <dir> [--minutes 60]

The world: the model's canonical states and ξ, the clip's clock, yaw and offset of the data's starts (the PPO
tracker's transitions), then each imagined step: the policy's action clipped to the box with the locked wrists zero
(the applied action the model takes), the model's next state, world_model.TrackTerms' reward (the task's tracking
terms, action rate and joint limits; contact terms 0) and termination at the next clock, the clock on. The policy's
observation is privileged: the state's invariant features (Layout.network_features), the reference of the coming
clock re-anchored under the robot (task.track_relative) as each tracked body's offset from the robot's in the
anchor's heading frame, the reference anchor's height above the robot's and its gravity, and the clip's progress."""
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
import imagine_walk as iw  # noqa: E402

PPO_STAGE = 'exp-84b0540b0092467d:570359808'


class GetupWorld:
    def __init__(self, model, data, reference, device):
        import torch
        from humanoid_sim.world_model_adapters.getup import GetupTerms
        self.model, self.data, self.device = model, data, device
        self.terms = GetupTerms(reference, device)
        self.track = self.terms.track
        self.layout = model.layout
        self.home = torch.as_tensor(reference.home, dtype=torch.float32, device=device)
        self.limit = float(reference.action_limit)
        locked = list(reference.wrist_joints) if reference.lock_wrists else []
        self.locked = torch.as_tensor(locked, dtype=torch.long, device=device)
        self.last = float(self.track.clip_last)
        self.inputs = None

    def starts(self, n, generator):
        import torch
        data = self.data
        while True:
            t, w = data.sample(4 * n, 1, generator)
            batch = data.window(t, w, 1, 'cpu')
            keep = batch['alive'][:, 0] & ~batch['terminated'][:, 0] & (batch['clock'][:, 0] <= self.track.clip_last)
            if int(keep.sum()) >= n:
                break
        index = torch.nonzero(keep)[:n, 0]
        pick = lambda value: value[index].to(self.device)  # noqa: E731
        return dict(state=self.layout.from_raw(pick(batch['state']).float()), context=pick(batch['context']).float(),
                    clock=pick(batch['clock'][:, 0]).long(), yaw=pick(batch['yaw'][:, 0]).float(),
                    offset=pick(batch['offset'][:, 0]).float(), kind=pick(batch['kind'][:, 0]).long(),
                    steps=pick(batch['steps'][:, 0]).long())

    def observation(self, state, worlds):
        import torch
        from humanoid_sim.tasks import g1_tensor_math as task
        track = self.track
        position, rotation, _, _ = track.robot(state)
        ref_position, ref_rotation, _, _ = track.reference(worlds['clock'], worlds['yaw'], worlds['offset'])
        relative, _ = task.track_relative(position, rotation, ref_position, ref_rotation, track.anchor)
        anchor = rotation[:, track.anchor]
        heading = torch.atan2(anchor[:, 1, 0], anchor[:, 0, 0])
        cos, sin = heading.cos(), heading.sin()
        offset = relative - position
        local = torch.stack((cos[:, None] * offset[..., 0] + sin[:, None] * offset[..., 1],
                             -sin[:, None] * offset[..., 0] + cos[:, None] * offset[..., 1], offset[..., 2]), -1)
        height = ref_position[:, track.anchor, 2] - position[:, track.anchor, 2]
        gravity = ref_rotation[:, track.anchor, 2, :]
        progress = (worlds['clock'].float() / self.last).clamp(max=1.5)
        return torch.cat((self.layout.network_features(state, self.home), local.flatten(1), height[:, None], gravity,
                          progress[:, None]), -1)

    def applied(self, action):
        applied = action.clamp(-self.limit, self.limit)
        return applied.index_fill(1, self.locked, 0.) if len(self.locked) else applied

    def step(self, worlds, action):
        import torch
        state = worlds['state']
        following = self.model.step(state, self.applied(action), worlds['context'])
        flat = dict(clock=worlds['clock'], yaw=worlds['yaw'], offset=worlds['offset'], kind=worlds['kind'],
                    steps=worlds['steps'] + 1)
        reward, _ = self.terms.reward(following, state, flat)
        fell = self.terms.terminated(following, flat)
        with torch.no_grad():
            worlds['clock'] = worlds['clock'] + 1
            worlds['steps'] = worlds['steps'] + 1
        return following, reward, fell


class GetupLearner(iw.Learner):
    def observe(self, state):
        return self.normalize(self.world.observation(state, self.worlds))


def judge(learner, config, episodes, ppo=None):
    """The CPU task's evaluation reset (the noisy first frame, no assist) for `episodes` seeds: the share that reached
    the clip's end, standing at the hold's end, for the learner's policy and (ppo) the PPO tracker."""
    import torch
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim import world_model as wm
    env = make_task('g1_getup', 'g1_29dof', config).unwrapped
    layout = wm.Layout(env.model.nu)
    policy, normalize = learner.cpu_policy()
    world_cpu = learner.world_cpu
    rows = []
    for name in ('мир', 'PPO'):
        if name == 'PPO' and ppo is None:
            continue
        for seed in range(episodes):
            observation, _ = env.reset(seed=seed, options=dict(assist_force=0.))
            done = standing = False
            for step in range(int(env.max_episode_steps)):
                if name == 'мир':
                    raw = torch.as_tensor(np.concatenate([env.data.qpos, env.data.qvel, env.previous_action]),
                                          dtype=torch.float32)[None]
                    state = layout.from_raw(raw)
                    worlds = dict(clock=torch.tensor([env.clip_clock + 1]), yaw=torch.tensor([float(env.clip_yaw)]),
                                  offset=torch.as_tensor(np.array(env.clip_offset), dtype=torch.float32)[None])
                    with torch.no_grad():
                        action = policy(normalize(world_cpu.observation(state, worlds)))[0].numpy()
                else:
                    action = ppo(observation)
                observation, _, terminated, truncated, info = env.step(action)
                done = done or bool(info.get('track_done'))
                if terminated or truncated:
                    break
            standing = bool(info.get('track_standing', False))  # the task's «up and standing» at the end
            rows.append(dict(policy=name, seed=seed, done=done, standing=standing, fell=bool(terminated),
                             seconds=round((step + 1) * env.dt, 2)))
    env.close()
    return rows


def main():
    import torch
    from humanoid_sim import world_model as wm
    parser = argparse.ArgumentParser(description='трекер вставания, выученный только внутри модели мира')
    parser.add_argument('fit')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--minutes', type=float, default=60.)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--episode', type=int, default=128)
    parser.add_argument('--worlds', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--lam', type=float, default=.95)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--critic-iterations', type=int, default=8)
    parser.add_argument('--std', type=float, default=.3)
    parser.add_argument('--judge-every', type=float, default=15.)
    parser.add_argument('--judge-episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out, device = Path(args.out), args.device
    out.mkdir(parents=True, exist_ok=True)
    model, _ = wm.WorldModel.load(Path(args.fit) / 'model.pt', device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta, arrays = fit.load(args.data)
    adapter = fit.adapter_of_meta(meta)
    data = fit.Data(arrays, 'cpu', .1, adapter)
    del arrays
    reference = adapter.reference(meta)
    world = GetupWorld(model, data, reference, device)
    generator = torch.Generator().manual_seed(args.seed)
    first = world.starts(8192, generator)
    sample = world.observation(first['state'], first)
    normalize = iw.Normalizer(sample)
    learner = GetupLearner(world, normalize, args, device, generator, inputs=sample.shape[1])
    learner.world_cpu = GetupWorld(wm.WorldModel.load(Path(args.fit) / 'model.pt', 'cpu')[0], data, reference, 'cpu')
    import collect
    from humanoid_sim.core import load_stage
    source = collect.resolve(PPO_STAGE)
    ppo_model, ppo_norm = load_stage(source.path)

    def ppo(observation):
        return ppo_model.predict(ppo_norm.normalize_obs(observation[None]), deterministic=True)[0][0]
    config = meta['task_config']
    log = open(out / 'train.log', 'a', encoding='utf-8')

    def say(text):
        print(text, flush=True)
        log.write(text + '\n')
        log.flush()
    say(f'модель {args.fit}; входов политики {sample.shape[1]}; миров {args.worlds}, горизонт {args.horizon}')
    curve, started, judged = [], time.time(), -1e9
    while True:
        elapsed = time.time() - started
        if elapsed - judged >= 60 * args.judge_every or elapsed >= 60 * args.minutes:
            judged = elapsed
            rows = judge(learner, config, args.judge_episodes, ppo if not curve else None)
            for name in ('PPO', 'мир'):
                mine = [r for r in rows if r['policy'] == name]
                if mine:
                    say(f'{name}: {elapsed / 60:.1f} мин, обновлений {learner.updates}: дошёл до конца клипа '
                        f'{sum(r["done"] for r in mine)} из {len(mine)}, стоит в конце {sum(r["standing"] for r in mine)}'
                        f', оборвался {sum(r["fell"] for r in mine)}; секунд ' + ' '.join(str(r['seconds']) for r in mine))
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
