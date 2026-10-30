"""A record_random.py log of the real SO-101 as collect.py data for the arm's world model (fit.py, the arm_reach
adapter): the log cut into --worlds contiguous segments (fit.py holds the last ones out), the state FixedLayout's
(angles | speeds; speeds by central differences of the logged angles), the gripper held at the model's home, the
action the applied u = goal − angle (ArmReach's convention, 0 on the gripper), no context. The task's fields (target,
hold counter, policy action) mean nothing here: zeros, and the policy action (goal − angle)/0.08 clipped to ±1.

    .venv\\Scripts\\python.exe scripts\\so101\\real_to_data.py <runs\\so101_real\\random-….json> --out <data dir>
        [--worlds 10]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'wm'))

import collect  # noqa: E402

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def main():
    from humanoid_sim.world_model_adapters.arm import ArmReach
    from humanoid_sim.world_model_adapters.contacts import bits
    parser = argparse.ArgumentParser(description='запись настоящей SO-101 как данные модели мира')
    parser.add_argument('log')
    parser.add_argument('--out', required=True)
    parser.add_argument('--worlds', type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'папка {out} не пуста')
    record = json.loads(Path(args.log).read_text(encoding='utf-8'))
    log = record['log']
    adapter = ArmReach({})
    env = adapter.cpu_env({})
    q = np.radians([[row['present'][n] for n in JOINTS] for row in log])
    goal = np.radians([[row['goal'][n] for n in JOINTS] for row in log])
    t = np.array([row['t'] for row in log])
    speed = np.gradient(q, t, axis=0)
    n, worlds = len(q), args.worlds
    length = (n - 1) // worlds
    gripper = float(env.home[5])
    raw = np.concatenate([q, np.full((n, 1), gripper), speed, np.zeros((n, 1))], 1)
    action = np.concatenate([goal - q, np.zeros((n, 1))], 1)
    policy = np.clip((goal - q) / .08, -1, 1)
    writer = collect.Writer(out, 50, collect.fields_of(adapter))
    started = time.time()
    empty = np.zeros((worlds, 0))
    for step in range(length):
        index = np.arange(worlds) * length + step
        boundary = np.full(worlds, step == length - 1)
        values = dict(state=raw[index], action=action[index], context=empty, bias=np.zeros((worlds, adapter.bias)),
                      reward=np.zeros(worlds), reward_contact=np.zeros(worlds), terminated=np.zeros(worlds, bool),
                      truncated=boundary, bad=np.zeros(worlds, bool), contact=bits(np.zeros((worlds, 0), bool)),
                      contact_next=bits(np.zeros((worlds, 0), bool)), force=empty, force_next=empty,
                      target=np.zeros((worlds, 3)), hold_before=np.zeros(worlds, np.int64),
                      policy_action=policy[index])
        writer.add(collect._row(adapter, values), raw[index + 1], boundary)
    writer.flush()
    meta = dict(stage=f'real:{Path(args.log).name}', stage_steps=None, robot_id='so101', task_id='arm_reach',
                model_revision=(ROOT / 'models/so101/UPSTREAM_REVISION.txt').read_text().strip(),
                action_mode='real', action_noise=None, deterministic=False, task_config={})
    meta = collect._finish(out, writer, worlds, length, env, 'real', 0, started, meta, adapter)
    print(f'{worlds} отрезков × {length} шагов → {out}')


if __name__ == '__main__':
    main()
