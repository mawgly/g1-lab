"""The reach skill for the real SO-101, learned only inside the world model fitted on its recordings (real_to_data.py
+ fit.py), and run on the arm.

    train: .venv\\Scripts\\python.exe scripts\\so101\\reach_real.py train --fit <fit dir> --data <data dir>
               --targets runs\\so101_real\\targets-1cm.json --out runs\\so101_real\\reach [--minutes 10]
    run:   .venv\\Scripts\\python.exe scripts\\so101\\reach_real.py run COM3 --policy runs\\so101_real\\reach
               [--count 10]  (the arm moves: someone beside it)

Kept inside what the model knows: starts are the recordings' states; targets are the gripper tip (our model's
gripperframe) of the safe poses safe_targets.py drew within the recorded box, at least TARGET_HEIGHT above the base's
plane; the action moves the goal by at most STEP rad per 40 ms tick (the recordings' 40 °/s, not ArmReachEnv's 0.08),
inside the recorded box. imagine.Imagination's reward and success (2 cm held 5 ticks) and its BPTT without a critic,
over --horizon ticks. On the arm the observation's tip comes from the measured angles through the same kinematics (no
camera), every goal passes safe_targets' check before it is sent, and the stall and heat guards of record_random.py
stop it; it ends holding its last pose."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'wm'))
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
STEP = .1  # rad: the goal at most this far from the measured angle (5.7°; the recordings' lag reached 7.7°, a
# goal only 0.028 ahead moved the lagging arm 0.5° a tick, too slow for a 64-tick horizon)
TARGET_HEIGHT, TIP_FLOOR = .03, .01  # m above the base's plane: targets, and every goal pose's tip
RATE = .028  # rad: the goal moves at most this much per tick (the recordings' 40 °/s)
CHANGE, STILL_COST = 2., 1.  # reward: −CHANGE·Σ(Δa)², −STILL_COST·Σq̇² within NEAR of the target
SPEED_NOISE = .1  # rad/s: the arm's speeds are finite differences of 25 Hz angles; training sees them as noisy
SMOOTH = .5  # the action the goal follows: SMOOTH·previous + (1 − SMOOTH)·policy's, in training and on the arm
# (28.10: the wrist goal flipped every tick near the target, a period-2 chatter from the noisy speed input)
NEAR, STILL = .03, .15  # m; rad/s: a hit is within 2 cm with every joint slower than STILL, 5 ticks running
SWAY_TICKS, SWAY_FLIPS, SWAY_STEP = 25, 8, .003  # a goal reversing more than 8 times in 1 s (steps > 0.17°) sways
BOX = np.radians([[-60., 60.], [-60., 60.], [-60., 60.], [-60., 88.], [-90., 90.]])


def tips_of(env, poses_degrees):
    import mujoco
    names = [env.model.joint(i).name for i in range(env.model.njnt)]
    qadr = env.model.jnt_qposadr[[names.index(n) for n in JOINTS]]
    data, out = env.probe, []
    for angles in poses_degrees:
        data.qpos[:] = env.home
        data.qpos[qadr] = np.radians(angles)
        mujoco.mj_kinematics(env.model, data)
        out.append(data.site_xpos[env.tip_id].copy())
    return np.array(out)


def limited(goal, previous, low, high, rate):
    """The goal moved from the previous one by at most rate per joint, inside the box (numpy or torch)."""
    import torch
    if torch.is_tensor(goal):
        step = torch.clamp(goal - previous, -rate, rate)
        return torch.minimum(torch.maximum(previous + step, low), high)
    return np.clip(previous + np.clip(goal - previous, -rate, rate), low, high)


def smooth_returns(world, policy, state, target, horizon, gamma, step, low, high, noise=True):
    """imagine.Imagination.returns with the goal rate-limited (RATE a tick, as on the arm), a cost on action change
    and on speed near the target, and a hit only when the arm holds still within 2 cm: (the mean discounted return,
    the share that hit, the final distance)."""
    import torch
    n, device = len(state), state.device
    context = torch.zeros((n, world.model.settings['context']), device=device)
    hold = torch.zeros(n, dtype=torch.long, device=device)
    alive = torch.ones(n, device=device)
    total = torch.zeros(n, device=device)
    goal = state[:, :5]
    previous_action = torch.zeros((n, 5), device=device)
    smoothed = torch.zeros((n, 5), device=device)
    for t in range(horizon):
        seen = state.clone()
        seen[:, 6:11] = seen[:, 6:11] + SPEED_NOISE * torch.randn((n, 5), device=device)  # as the arm's differences
        action = policy.act(world.observation(seen, target), torch.randn((n, 5), device=device) if noise else None)
        smoothed = SMOOTH * smoothed + (1 - SMOOTH) * action
        q = state[:, :6]
        goal = limited(q[:, :5] + step * smoothed, goal, low, high, RATE)
        applied = torch.cat((goal - q[:, :5], world.gripper_home - q[:, 5:6]), -1)
        following = world.model.step(state, applied, context)
        # the task's reward without its hit bonus (a hold counter of 0 never reaches 5): the still hit replaces it
        reward, _ = world.terms.reward(following, state, dict(target=target, hold_before=torch.zeros_like(hold),
                                                              policy_action=action))
        distance = world.terms.distance(following, target)
        speed = following[:, 6:11]
        reward = reward - CHANGE * ((action - previous_action) ** 2).sum(-1)
        reward = reward - STILL_COST * (distance < NEAR).float() * (speed ** 2).sum(-1)
        with torch.no_grad():
            still = (distance < .02) & (speed.abs().max(-1).values < STILL)
            hold = (hold + 1) * still
            hit = hold >= 5
        reward = reward + 10 * hit.float()
        total = total + gamma ** t * alive * reward
        with torch.no_grad():
            alive = alive * (~hit).float()
        previous_action, state = action, following
    return total.mean(), float(1 - alive.mean()), float(world.terms.distance(state, target).detach().mean())


def train(args):
    import torch
    import fit
    import imagine
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_adapters.arm import ArmReach
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    model, _ = wm.WorldModel.load(Path(args.fit) / 'model.pt', device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    env = imagine.make_env()
    poses = json.loads(Path(args.targets).read_text(encoding='utf-8'))['targets']
    tips = tips_of(env, poses)
    tips = tips[tips[:, 2] > TARGET_HEIGHT]
    held = len(tips) // 5  # the last fifth judges
    train_tips, judge_tips = tips[:-held], tips[-held:]
    meta, arrays = fit.load(args.data)
    data = fit.Data(arrays, 'cpu', .2, ArmReach({}))
    states = torch.as_tensor(arrays['state'][:, data.train_worlds.numpy()].reshape(-1, arrays['state'].shape[-1]),
                             dtype=torch.float32, device=device)
    judge_states = torch.as_tensor(arrays['state'][:, data.held_worlds.numpy()].reshape(-1, arrays['state'].shape[-1]),
                                   dtype=torch.float32, device=device)
    world = imagine.Imagination(model, env, device)
    low = torch.as_tensor(BOX[:, 0], dtype=torch.float32, device=device)
    high = torch.as_tensor(BOX[:, 1], dtype=torch.float32, device=device)

    def applied(state, action):  # the goal moves STEP at most, inside the recorded box; the gripper held
        q = state[:, :6]
        goal = torch.minimum(torch.maximum(q[:, :5] + args.step * action, low), high)
        return torch.cat((goal - q[:, :5], world.gripper_home - q[:, 5:6]), -1)
    world.applied = applied
    policy = imagine.Policy(device)
    if args.resume and (out / 'policy.pt').is_file():  # go on from the best one saved
        saved = torch.load(out / 'policy.pt', map_location=device, weights_only=False)
        policy.net.load_state_dict(saved['net'])
        print(f"продолжаю с лучшей: {saved['updates']} обновлений, успех {saved['success']:.2f}", flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    targets = torch.as_tensor(train_tips, dtype=torch.float32, device=device)
    judge_targets = torch.as_tensor(judge_tips, dtype=torch.float32, device=device)
    pick_judge = torch.randint(0, len(judge_states), (1024,), generator=generator)
    pick_target = torch.randint(0, len(judge_targets), (1024,), generator=generator)
    started, updates, curve, best = time.time(), 0, [], None
    while time.time() - started < 60 * args.minutes:
        rows = torch.randint(0, len(states), (args.batch,), generator=generator)
        goals = torch.randint(0, len(targets), (args.batch,), generator=generator)
        value, _, _ = smooth_returns(world, policy, states[rows], targets[goals], args.horizon, .99, args.step, low,
                                     high)
        optimizer.zero_grad()
        (-value / args.horizon).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        updates += 1
        if updates % 25 == 0:
            with torch.no_grad():
                _, success, distance = smooth_returns(world, policy, judge_states[pick_judge],
                                                      judge_targets[pick_target], args.horizon, .99, args.step, low,
                                                      high, noise=False)
            curve.append(dict(updates=updates, seconds=round(time.time() - started), success=success,
                              distance_m=distance))
            print(f'обновлений {updates}, {time.time() - started:.0f} с: в модели (отложенные старты и цели) успех '
                  f'{success:.2f}, конечное расстояние {distance * 100:.1f} см', flush=True)
            if best is None or (success, -distance) > best:  # the best one is kept: BPTT can collapse late
                best = (success, -distance)
                torch.save(dict(net={k: v.cpu() for k, v in policy.net.state_dict().items()}, smooth=True,
                                log_std=policy.log_std.detach().cpu(), step=args.step, judge_tips=judge_tips.tolist(),
                                curve=curve, fit=str(args.fit), updates=updates, success=success, distance_m=distance),
                           out / 'policy.pt')


class Bridge:
    """arm_bridge.py in .venv-lerobot as a child process: JSON lines in and out."""

    def __init__(self, port):
        import subprocess
        python = ROOT / '.venv-lerobot' / 'Scripts' / 'python.exe'
        self.process = subprocess.Popen([str(python), str(ROOT / 'scripts' / 'so101' / 'arm_bridge.py'), port],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                        cwd=str(ROOT / 'scripts' / 'so101'))

    def call(self, **request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('мост к руке закрылся')
            if line.startswith('{'):
                reply = json.loads(line)
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                return reply

    def close(self):
        try:
            self.call(op='close')
        finally:
            self.process.wait(timeout=10)


def run(args):
    """On the real arm: --count held-out targets, one after another from where the arm is."""
    import mujoco
    import torch
    import imagine
    saved = torch.load(Path(args.policy) / 'policy.pt', map_location='cpu', weights_only=False)
    policy = imagine.Policy('cpu')
    policy.net.load_state_dict(saved['net'])
    env = imagine.make_env()
    names = [env.model.joint(i).name for i in range(env.model.njnt)]
    qadr = env.model.jnt_qposadr[[names.index(n) for n in JOINTS]]
    probe = env.probe

    def tip_and_safe(angles):
        probe.qpos[:] = env.home
        probe.qpos[qadr] = angles
        mujoco.mj_forward(env.model, probe)
        return probe.site_xpos[env.tip_id].copy(), probe.ncon == 0 and probe.site_xpos[env.tip_id][2] > TIP_FLOOR
    rng = np.random.default_rng(args.seed)
    targets = np.array(saved['judge_tips'])[rng.choice(len(saved['judge_tips']), args.count, replace=False)]
    arm = Bridge(args.port)
    start = arm.call(op='enable')['present']
    goal = dict(start)
    results, log = [], []
    hot_reads, read_at = 0, time.time()
    try:
        previous, last = np.radians([start[n] for n in JOINTS]), time.time()
        sent = previous.copy()  # the last goal sent (rad): the next one moves at most RATE from it
        smoothed = np.zeros(5)
        moves = []  # the goal's per-joint step signs of the last SWAY_TICKS ticks: a sway stops the run
        for number, target in enumerate(targets):
            hold, reached, ticks, reason, distance = 0, False, 0, 'время вышло', float('nan')
            while ticks < args.ticks:
                present = arm.call(op='read')['present']
                q = np.radians([present[n] for n in JOINTS])
                now = time.time()
                speed = (q - previous) / max(now - last, 1e-3)
                previous, last = q, now
                tip, _ = tip_and_safe(q)
                distance = float(np.linalg.norm(tip - target))
                hold = hold + 1 if distance < .02 and np.abs(speed).max() < STILL else 0
                if hold >= 5:
                    reached, reason = True, 'дотянулась'
                    break
                if now - read_at >= 1.:
                    read_at = now
                    temperature = arm.call(op='temperature')['temperature']
                    hot_reads = hot_reads + 1 if max(temperature.values()) > 50 else 0
                    if hot_reads >= 2:
                        reason = 'перегрев: два отсчёта выше 50 °C'
                        break
                observation = np.concatenate((np.sin(q), np.cos(q), .1 * speed, 3 * target, 3 * tip,
                                              3 * (target - tip))).astype(np.float32)
                with torch.no_grad():
                    action = policy.act(torch.as_tensor(observation)[None])[0].numpy()
                smoothed = SMOOTH * smoothed + (1 - SMOOTH) * action if saved.get('smooth') else action
                new = limited(q + saved['step'] * smoothed, sent, BOX[:, 0], BOX[:, 1], RATE)
                moves = (moves + [np.where(np.abs(new - sent) > SWAY_STEP, np.sign(new - sent), 0.)])[-SWAY_TICKS:]
                if len(moves) == SWAY_TICKS:
                    signs = np.array(moves)
                    flips = [(np.diff(column[column != 0]) != 0).sum() for column in signs.T]
                    if max(flips) > SWAY_FLIPS:
                        reason = f'раскачка: цель {max(flips)} раз сменила направление за {SWAY_TICKS} тактов'
                        break
                sent = new
                if not tip_and_safe(new)[1]:
                    reason = 'цель не прошла проверку безопасности'
                    break
                lag = max(abs(goal[n] - present[n]) for n in JOINTS)
                if lag > 15:
                    reason = f'застревание: отставание {lag:.1f}°'
                    break
                goal.update({n: float(np.degrees(v)) for n, v in zip(JOINTS, new)})
                arm.call(op='goal', goal=goal)
                log.append(dict(target=number, t=round(now, 3), present=present, goal=dict(goal), distance=distance))
                ticks += 1
                time.sleep(max(0., .04 - (time.time() - now)))
            results.append(dict(target=target.tolist(), reached=reached, ticks=ticks, distance_m=distance,
                                reason=reason))
            print(f'цель {number + 1}: {reason}, тактов {ticks}, расстояние {distance * 100:.1f} см', flush=True)
            if not reached and reason != 'время вышло':
                break
    finally:
        out = ROOT / 'runs' / 'so101_real' / time.strftime('reach-%Y%m%d-%H%M%S.json')
        out.write_text(json.dumps(dict(results=results, log=log), ensure_ascii=False), encoding='utf-8')  # first
        try:
            arm.call(op='goal', goal=arm.call(op='read')['present'])
        except RuntimeError as error:
            print(f'рука не ответила при остановке: {error}')
        finally:
            try:
                arm.close()
            except Exception as error:  # noqa: BLE001
                print(f'мост закрылся с ошибкой: {error}')
        print(f'дотянулась {sum(r["reached"] for r in results)} из {len(results)}; запись {out}; моторы держат позу')


def main():
    import torch
    parser = argparse.ArgumentParser(description='навык «дотянуться» для настоящей SO-101 из модели мира')
    sub = parser.add_subparsers(dest='mode', required=True)
    t = sub.add_parser('train')
    t.add_argument('--fit', required=True)
    t.add_argument('--data', required=True)
    t.add_argument('--targets', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--minutes', type=float, default=10.)
    t.add_argument('--horizon', type=int, default=64)
    t.add_argument('--batch', type=int, default=1024)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--step', type=float, default=STEP)
    t.add_argument('--resume', action='store_true')
    t.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    r = sub.add_parser('run')
    r.add_argument('port')
    r.add_argument('--policy', required=True)
    r.add_argument('--count', type=int, default=10)
    r.add_argument('--ticks', type=int, default=150)
    r.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    (train if args.mode == 'train' else run)(args)


if __name__ == '__main__':
    main()
