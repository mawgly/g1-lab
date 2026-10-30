"""The honest comparison on the arm: a world model from scratch, then rounds of collect → model → policy (SuperMimic's
loop), against PPO, counted in simulator control steps (only the collected steps count: the judging episodes and the
resets that draw imagination's starts do not, as PPO's evaluation does not).

    .venv-fullgpu\\Scripts\\python.exe scripts\\lab\\wm\\dyna_arm.py --out runs\\humanoid_lab\\world_model\\dyna-arm
        [--worlds 64] [--steps 200] [--rounds 12] [--fit-minutes 3] [--updates 300] [--seed 0]

Round 0 collects with uniform random actions (ArmReachEnv's box), every later round with the current policy's own
noise (imagine.Policy, σ learned); the rounds' data are merged into one collect.py folder (each round's last step
made an episode boundary, so no window crosses a seam), a fresh world model is fitted on all of it (fit.py, «разброс»
units) and the same policy goes on learning inside it (imagine.py) for --updates updates, then is judged in
ArmReachEnv on runs/so101's seeds. ppo_arm_curve.py gives PPO's curve on the same seeds."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import collect  # noqa: E402
import fit  # noqa: E402
import imagine  # noqa: E402


def merge(rounds, out):
    """The rounds' collect.py folders as one folder `out`: their chunks renamed in order, each round's last chunk with
    its last row an episode boundary (truncated, its following state from last_next) and meta's steps summed."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index, steps = 0, 0
    for number, folder in enumerate(rounds):
        files = sorted(folder.glob('chunk_*.npz'))
        for position, path in enumerate(files):
            target = out / f'chunk_{index:05d}.npz'
            index += 1
            if position < len(files) - 1 or number == len(rounds) - 1:
                try:  # a hard link where it can (the walking base data are gigabytes), else a copy
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
                continue
            with np.load(path) as chunk:
                arrays = {key: chunk[key] for key in chunk.files}
            worlds = arrays['state'].shape[1]
            last = len(arrays['state']) - 1
            arrays['truncated'] = arrays['truncated'].copy()
            arrays['truncated'][last] = True
            known = set(int(i) for i in arrays['boundary_index'])
            extra = [w for w in range(worlds) if last * worlds + w not in known]
            arrays['boundary_index'] = np.concatenate((arrays['boundary_index'],
                                                       np.array([last * worlds + w for w in extra], dtype=np.int64)))
            arrays['boundary_next'] = np.concatenate((arrays['boundary_next'].reshape(-1, arrays['last_next'].shape[1]),
                                                      arrays['last_next'][extra]))
            np.savez(target, **arrays)
        steps += json.loads((folder / 'meta.json').read_text(encoding='utf-8'))['steps']
    meta = json.loads((rounds[0] / 'meta.json').read_text(encoding='utf-8'))
    meta.update(steps=steps, chunks=index, rounds=len(rounds), stage=None, action_mode='dyna')
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_adapters.arm import ArmReach
    parser = argparse.ArgumentParser(description='модель мира руки с нуля: сбор → модель → политика, против PPO')
    parser.add_argument('--out', required=True)
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--steps', type=int, default=200, help='шагов управления на мир в раунде')
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--fit-minutes', type=float, default=3.)
    parser.add_argument('--updates', type=int, default=300, help='обновлений политики в модели за раунд')
    parser.add_argument('--target', type=float, default=1., help='успех, на котором остановиться')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out, device = Path(args.out), args.device
    out.mkdir(parents=True, exist_ok=True)
    adapter = ArmReach({})
    env = imagine.make_env()
    started = time.time()
    qpos, qvel, target = imagine.draw_starts(env, 8192, 1_000_000 + args.seed)
    states = torch.as_tensor(np.concatenate((qpos, qvel), 1), dtype=torch.float32, device=device)
    targets = torch.as_tensor(target, dtype=torch.float32, device=device)
    policy = imagine.Policy(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rounds, curve, simulator = [], [], 0
    meta = dict(robot_id='so101', task_id='arm_reach', task_config={}, stage=None,
                model_revision=(HERE.parents[2] / 'models/so101/UPSTREAM_REVISION.txt').read_text().strip())
    for number in range(args.rounds):
        folder = out / f'round_{number:02d}'
        if folder.exists():
            shutil.rmtree(folder)
        if number == 0:
            def act(observations):
                return rng.uniform(-1, 1, (len(observations), 5))
            mode = 'random'
        else:
            def act(observations):
                with torch.no_grad():
                    seen = torch.as_tensor(observations, dtype=torch.float32, device=device)
                    return policy.act(seen, torch.randn((len(seen), 5), device=device)).cpu().numpy()
            mode = 'policy'
        envs = [adapter.cpu_env({}) for _ in range(args.worlds)]
        collect.collect_cpu(envs, act, args.steps, folder, seed=args.seed * 100_000 + number * 1000, chunk=50,
                            meta=dict(meta, action_mode=mode), adapter=adapter)
        simulator += args.worlds * args.steps
        rounds.append(folder)
        merge(rounds, out / 'data')
        fitted = out / 'fit'
        if fitted.exists():
            shutil.rmtree(fitted)
        report = fit.fit(out / 'data', fitted, minutes=args.fit_minutes, device=device, loss_units='spread',
                         checkpoint_minutes=None, horizons=(wm.WINDOW,), log=lambda _: None)
        model, _ = wm.WorldModel.load(fitted / 'model.pt', device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        world = imagine.Imagination(model, env, device)
        for _ in range(args.updates):
            pick = torch.randint(0, len(states), (2048,), generator=generator).to(device)
            value, _, _ = world.returns(policy, states[pick], targets[pick], 32, .99)
            optimizer.zero_grad()
            (-value / 32).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
        judged = imagine.judge(policy, env, device)
        error = next((c['value'] for c in report['ladder']['criteria'] if c['name'] == 'error'), None)
        row = dict(round=number, simulator_steps=simulator, mode=mode, world_error=error,
                   seconds=round(time.time() - started), std=float(policy.log_std.detach().exp().mean()), **judged)
        curve.append(row)
        print(f'раунд {number}: шагов симулятора {simulator}, ошибка модели {error:.4f}, успех '
              f'{judged["success_rate"]:.2f}, шагов {judged["mean_steps"]}, расстояние '
              f'{1000 * judged["final_distance_m"]:.1f} мм, {row["seconds"]} с', flush=True)
        (out / 'curve.json').write_text(json.dumps(dict(curve=curve, args=vars(args)), ensure_ascii=False, indent=1),
                                        encoding='utf-8')
        torch.save(dict(net=policy.net.state_dict(), log_std=policy.log_std.detach().cpu()), out / 'policy.pt')
        if judged['success_rate'] >= args.target:
            break


if __name__ == '__main__':
    main()
