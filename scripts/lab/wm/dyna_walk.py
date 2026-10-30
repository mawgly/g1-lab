"""Rounds of collect → model → policy on G1 walking (SuperMimic's loop, dyna_arm.py's on the G1): the policy learned
inside the world model (imagine_walk.Learner) walks in the training's CUDA task (candidate 8's task_config: its
pushes, noise and randomisation, its command resampling, from training resets), those transitions join the data, the
world model goes on training on all of it (fit.py, resumed), the policy goes on learning inside the updated model, and
is judged in the CPU task. Each round's new data are what the policy does now, so the model learns the states where
it was wrong (imagine_walk alone: no fall in imagination, falls within a second in the simulator).

    .venv-fullgpu\\Scripts\\python.exe scripts\\lab\\wm\\dyna_walk.py
        --fit runs\\humanoid_lab\\world_model\\ladder-1-walk\\fit-2x-spread
        --base-data runs\\humanoid_lab\\world_model\\ladder-1-walk\\data
        --out runs\\humanoid_lab\\world_model\\dyna-walk [--policy <imagine_walk out>\\policy.pt] [--rounds 6]

Simulator steps counted: each round's --collect-worlds × --collect-steps (the base data are candidate 8's, not counted
here; the judging episodes are not training)."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import collect  # noqa: E402
import dyna_arm  # noqa: E402
import fit  # noqa: E402
import imagine_walk as iw  # noqa: E402


def cuda_act(env, learner, std, device):
    """The policy on the CUDA task's own state: imagine_walk's observation from qpos, qvel, the previous applied
    action, the observed command and the gait clock of the coming step (lengths + phase offset + 1), plus noise."""
    import torch
    from humanoid_sim.tasks import g1_tensor_math as task
    world = learner.world
    layout = world.layout

    def act(observations):
        raw = torch.cat((env.qpos, env.qvel, env.previous_action), 1).float()
        state = layout.from_raw(raw)
        observed = env.observed_command.float()
        phase = (env.lengths + env.phase_offset + 1).long()
        x = learner.normalize(world.observation(state, observed, phase))
        with torch.no_grad():
            action = learner.actor(x)
            return action + std * torch.randn_like(action)
    return act


def load_world(fit_dir, data_dir, reference, device):
    import torch
    from humanoid_sim import world_model as wm
    model, _ = wm.WorldModel.load(Path(fit_dir) / 'model.pt', device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta, arrays = fit.load(data_dir)
    data = fit.Data(arrays, 'cpu', .1, fit.adapter_of_meta(meta))
    del arrays
    torch.cuda.empty_cache()
    return iw.WalkWorld(model, data, reference, device), meta


def main():
    import torch
    parser = argparse.ArgumentParser(description='ходьба G1: сбор → модель мира → политика по раундам')
    parser.add_argument('--fit', required=True, help='папка fit.py: модель мира, с которой начать (копируется)')
    parser.add_argument('--base-data', required=True, help='данные collect.py, с которых начать (ходьба кандидата 8)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy', default=None, help='policy.pt imagine_walk.py, с которой начать')
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--collect-worlds', type=int, default=4096)
    parser.add_argument('--collect-steps', type=int, default=250)
    parser.add_argument('--collect-std', type=float, default=.3)
    parser.add_argument('--fit-minutes', type=float, default=8.)
    parser.add_argument('--policy-minutes', type=float, default=15.)
    parser.add_argument('--worlds', type=int, default=2048)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--episode', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--lam', type=float, default=.95)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--critic-iterations', type=int, default=8)
    parser.add_argument('--std', type=float, default=.3)
    parser.add_argument('--judge-seconds', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    device = 'cuda'
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'dyna.log', 'a', encoding='utf-8')

    def say(text):
        print(text, flush=True)
        log.write(text + '\n')
        log.flush()
    fitted = out / 'fit'
    if not fitted.exists():
        fitted.mkdir()
        for name in ('model.pt', 'train_state.pt'):
            shutil.copyfile(Path(args.fit) / name, fitted / name)
    candidate, config = iw.candidate_policy()
    meta = json.loads((Path(args.base_data) / 'meta.json').read_text(encoding='utf-8'))
    adapter = fit.adapter_of_meta(meta)
    reference = adapter.reference(meta)
    world, _ = load_world(fitted, args.base_data, reference, device)
    generator = torch.Generator().manual_seed(args.seed)
    first = world.starts(8192, generator)
    observed = world.observed_command(first['state'], first['command'], first['heading'])
    learner = iw.Learner(world, iw.Normalizer(world.observation(first['state'], observed, first['phase'])), args,
                         device, generator)
    if args.policy:
        learner.load(args.policy)
        say(f'политика из {args.policy} ({learner.updates} обновлений)')
    rounds, curve, simulator, started = [], [], 0, time.time()
    collect_meta = {key: meta.get(key) for key in ('robot_id', 'task_id', 'model_revision')}
    for number in range(args.rounds):
        # 1. The policy walks in the training's CUDA task.
        folder = out / f'round_{number:02d}'
        if folder.exists():
            shutil.rmtree(folder)
        env = collect.collecting_env(adapter, n_envs=args.collect_worlds, seed=10_000 + number, **config)
        collect.run_cuda(env, cuda_act(env, learner, args.collect_std, device), args.collect_steps, folder,
                         seed=10_000 + number, chunk=50,
                         meta=dict(collect_meta, stage=None, action_mode='dyna', action_noise=args.collect_std,
                                   deterministic=False, task_config=config))
        del env
        torch.cuda.empty_cache()
        simulator += args.collect_worlds * args.collect_steps
        rounds.append(folder)
        # 2. The world model goes on training on the base data and every round.
        dyna_arm.merge([Path(args.base_data), *rounds], out / 'data')
        state = torch.load(fitted / 'train_state.pt', map_location='cpu', weights_only=False)
        report = fit.fit(out / 'data', fitted, minutes=state['seconds'] / 60 + args.fit_minutes, device=device,
                         resume=True, horizons=(8,), eval_windows=8192, curve_windows=4096, checkpoint_minutes=None,
                         log=lambda _: None)
        error = next(c['value'] for c in report['ladder']['criteria'] if c['name'] == 'error')
        torch.cuda.empty_cache()
        # 3. The policy goes on learning inside the updated model.
        learner.world, _ = load_world(fitted, out / 'data', reference, device)
        learner.restart_all()
        until = time.time() + 60 * args.policy_minutes
        while time.time() < until:
            line = learner.update()
        # 4. Judged in the CPU task.
        policy_cpu, norm_cpu = learner.cpu_policy()
        rows = iw.judge(policy_cpu, norm_cpu, config, args.judge_seconds, 'cpu', seeds=(0, 1))
        falls = sum(row['fell'] for row in rows)
        say(f'раунд {number}: шагов симулятора {simulator}, ошибка модели {error:.4f}, {line}; в симуляторе упал '
            f'{falls} из {len(rows)}, {(time.time() - started) / 60:.0f} мин')
        say('\n'.join(iw.summary(rows)))
        curve.append(dict(round=number, simulator_steps=simulator, world_error=error, falls=falls, rows=rows,
                          minutes=(time.time() - started) / 60))
        learner.save(out / 'policy.pt')
        (out / 'curve.json').write_text(json.dumps(dict(curve=curve, args=vars(args)), ensure_ascii=False, indent=1),
                                        encoding='utf-8')


if __name__ == '__main__':
    main()
