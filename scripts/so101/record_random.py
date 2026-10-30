"""Recording for the real SO-101's world model: smooth random motions in a small box around the pose the arm holds,
at the simulator's control step (40 ms, ArmReachEnv), with each tick's goal, present position, velocity, load and
(every second) temperature logged. A new target every 0.5–1.5 s, reached by a cosine blend at most MAX_SPEED °/s.
Stops at once (goal = present, the motors hold) when a joint lags its goal by more than STALL degrees or a motor
passes HOT °C; at the end the arm comes back to the start pose and keeps holding it (motion_check.py --release).

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\record_random.py COM3 [--minutes 5] [--seed 0]"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from motion_check import JOINTS, ROOT, make_bus

TICK = .04  # ArmReachEnv's control step
BOX = dict(shoulder_pan=25., shoulder_lift=10., elbow_flex=10., wrist_flex=10., wrist_roll=40.)  # ± degrees
LIMIT = 88.  # degrees: no goal beyond it
MAX_SPEED = 40.  # °/s
STALL, HOT = 15., 50.


def main():
    parser = argparse.ArgumentParser(description='запись плавных случайных движений SO-101 для модели мира')
    parser.add_argument('port')
    parser.add_argument('--minutes', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--targets', default=None, help='цепочка целей safe_targets.py вместо коробки вокруг позы')
    parser.add_argument('--gripper', default=None, help='хват по цепочке: «от,до» из 0–100 (иначе держится)')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    bus = make_bus(args.port)
    bus.connect()
    start = bus.sync_read('Present_Position')
    box = {name: (max(-LIMIT, start[name] - half), min(LIMIT, start[name] + half)) for name, half in BOX.items()}
    box['gripper'] = (start['gripper'], start['gripper'])  # held: it stalled closing (28.10), and the task holds it
    goal = dict(start)
    bus.sync_write('Goal_Position', goal)
    bus.enable_torque()
    log, reason, began = [], 'время вышло', time.time()
    read_at, hot_reads = -1e9, 0

    chain = None if args.targets is None else json.loads(Path(args.targets).read_text(encoding='utf-8'))
    grip = None if args.gripper is None else tuple(float(v) for v in args.gripper.split(','))
    stop = ROOT / 'runs' / 'so101_real' / 'stop'  # touch it to end the recording with the log kept
    stop.unlink(missing_ok=True)
    order = iter(chain['targets']) if chain else None

    def plan(origin):
        if order is not None:  # safe_targets.py's chain: the arm's five joints, the gripper held
            target = dict(zip(chain['joints'], next(order)), gripper=start['gripper'] if grip is None else rng.uniform(*grip))
        else:
            target = {name: rng.uniform(*box[name]) for name in JOINTS}
        largest = max(abs(target[name] - origin[name]) for name in JOINTS)
        seconds = max(rng.uniform(.5, 1.5), largest / MAX_SPEED)
        return target, seconds

    try:
        origin, (target, seconds), segment = dict(goal), plan(goal), time.time()
        next_tick = time.time()
        while time.time() - began < 60 * args.minutes:
            if stop.exists():
                reason = 'остановлено по файлу stop'
                break
            now = time.time()
            u = min(1., (now - segment) / seconds)
            blend = .5 - .5 * math.cos(math.pi * u)
            goal = {name: origin[name] + (target[name] - origin[name]) * blend for name in JOINTS}
            bus.sync_write('Goal_Position', goal)
            present = bus.sync_read('Present_Position')
            velocity = bus.sync_read('Present_Velocity', normalize=False)
            load = bus.sync_read('Present_Load', normalize=False)
            row = dict(t=round(now - began, 4), goal=goal, present=present, velocity=velocity, load=load)
            if now - read_at >= 1.:  # once a second; one read above HOT can be a bus glitch, two in a row stop
                read_at = now
                row['temperature'] = bus.sync_read('Present_Temperature', normalize=False)
                hot_reads = hot_reads + 1 if max(row['temperature'].values()) > HOT else 0
            log.append(row)
            lag = max(abs(goal[name] - present[name]) for name in JOINTS)
            if lag > STALL or hot_reads >= 2:
                reason = f'остановка: отставание {lag:.1f}° или два отсчёта температуры выше {HOT} °C'
                bus.sync_write('Goal_Position', present)
                break
            if u >= 1.:
                origin, (target, seconds), segment = dict(goal), plan(goal), now
            next_tick += TICK
            time.sleep(max(0., next_tick - time.time()))
        else:
            present = bus.sync_read('Present_Position')
            for k in range(1, 76):  # 3 s back to the start pose
                bus.sync_write('Goal_Position', {n: present[n] + (start[n] - present[n]) * k / 75 for n in JOINTS})
                time.sleep(TICK)
    finally:
        out = ROOT / 'runs' / 'so101_real'
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime('random-%Y%m%d-%H%M%S.json')
        path.write_text(json.dumps(dict(tick=TICK, box=box, start=start, seed=args.seed, reason=reason, log=log,
                                        targets=args.targets),
                                   ensure_ascii=False), encoding='utf-8')
        bus.disconnect(disable_torque=False)
        ticks = len(log)
        lags = [max(abs(r['goal'][n] - r['present'][n]) for n in JOINTS) for r in log]
        print(f'{reason}; тактов {ticks} за {log[-1]["t"] if log else 0:.0f} с '
              f'(в среднем {ticks / max(log[-1]["t"], 1e-9) if log else 0:.1f} Гц), наибольшее отставание '
              f'{max(lags, default=0):.1f}°; запись {path}; моторы держат позу')


if __name__ == '__main__':
    main()
