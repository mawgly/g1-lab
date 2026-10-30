"""First powered check of a SO-101 arm (LeRobot's driver and its calibration file): the motors hold the pose they are
in, then, from the base to the gripper, each joint moves STEP degrees (the gripper STEP of its 0–100 range) in one
second by small goal steps, holds half a second and comes back the same way; every 20 ms the present positions are
logged. The motors stay powered at the end, holding the pose (released, the shoulder would fall): run with --release,
holding the arm, to switch the torque off.

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\motion_check.py COM3 [--step 5] [--release]

Goals never leave the calibrated range (the joint's limits minus a margin). The log goes to runs/so101_real/
(angles in LeRobot's degrees, which the MuJoCo model so101_new_calib shares: zero is the middle of the range)."""
import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = Path.home() / '.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json'
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
TICK = .02


def make_bus(port):
    calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    motors = {name: Motor(calibration[name]['id'], 'sts3215',
                          MotorNormMode.RANGE_0_100 if name == 'gripper' else MotorNormMode.DEGREES)
              for name in JOINTS}
    return FeetechMotorsBus(port, motors=motors,
                            calibration={name: MotorCalibration(**calibration[name]) for name in JOINTS})


def main():
    parser = argparse.ArgumentParser(description='первая проверка SO-101 с включёнными моторами')
    parser.add_argument('port')
    parser.add_argument('--step', type=float, default=5.)
    parser.add_argument('--release', action='store_true', help='только выключить моторы (рука в ваших руках)')
    args = parser.parse_args()
    bus = make_bus(args.port)
    bus.connect()
    if args.release:
        bus.disable_torque()
        bus.disconnect(disable_torque=False)
        print('моторы выключены')
        return
    log, started = [], time.time()

    def record(phase, goal):
        present = bus.sync_read('Present_Position')
        log.append(dict(t=round(time.time() - started, 3), phase=phase, goal=dict(goal), present=present))
        return present

    try:
        bus.disable_torque()
        bus.configure_motors()
        for name in JOINTS:  # LeRobot's SO-101 follower configure(): position mode, its gains, the gripper's limits
            bus.write('Operating_Mode', name, 0)
            bus.write('P_Coefficient', name, 16)
            bus.write('I_Coefficient', name, 0)
            bus.write('D_Coefficient', name, 32)
        bus.write('Max_Torque_Limit', 'gripper', 500)
        bus.write('Protection_Current', 'gripper', 250)
        bus.write('Overload_Torque', 'gripper', 25)
        start = bus.sync_read('Present_Position')
        goal = dict(start)
        bus.sync_write('Goal_Position', goal)
        bus.enable_torque()
        print('держу позу:', {k: round(v, 1) for k, v in start.items()})
        for _ in range(50):
            record('hold', goal)
            time.sleep(TICK)
        for name in JOINTS:
            base = start[name]
            low, high = (2., 98.) if name == 'gripper' else (-90., 90.)
            sign = 1. if base + args.step <= high else -1.
            target = min(max(base + sign * args.step, low), high)
            for leg, (a, b) in (('out', (base, target)), ('back', (target, base))):
                for k in range(1, 51):
                    goal[name] = a + (b - a) * k / 50
                    bus.sync_write('Goal_Position', goal)
                    record(f'{name}:{leg}', goal)
                    time.sleep(TICK)
                for _ in range(25):
                    record(f'{name}:{leg}-hold', goal)
                    time.sleep(TICK)
            reached = [row['present'][name] for row in log if row['phase'] == f'{name}:out-hold']
            back = [row['present'][name] for row in log if row['phase'] == f'{name}:back-hold']
            print(f'{name}: цель {target:+.1f}, дошёл до {reached[-1]:+.1f} (ошибка {reached[-1] - target:+.2f}), '
                  f'вернулся к {back[-1]:+.1f} (было {base:+.1f})', flush=True)
    finally:
        out = ROOT / 'runs' / 'so101_real'
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime('motion_check-%Y%m%d-%H%M%S.json')
        path.write_text(json.dumps(dict(step=args.step, tick=TICK, calibration=str(CALIBRATION), log=log),
                                   ensure_ascii=False), encoding='utf-8')
        bus.disconnect(disable_torque=False)  # the motors keep holding the pose
        print(f'запись: {path}; моторы держат позу (выключить: --release, придерживая руку)')


if __name__ == '__main__':
    main()
