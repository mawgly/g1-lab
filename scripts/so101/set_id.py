"""Give ONE SO-101 servo its joint's ID, in any order (LeRobot's lerobot-setup-motors always goes from the gripper to
the base). Only the servo to set may be on the bus: unplug the others first, or leave only servos that already have
other IDs. LeRobot's own setup_motor does the write (torque off, ID, then 1 000 000 baud).

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\set_id.py COM3 shoulder_lift

Joints and IDs (LeRobot's so101): shoulder_pan 1, shoulder_lift 2, elbow_flex 3, wrist_flex 4, wrist_roll 5, gripper 6.
Check afterwards with scripts\\so101\\scan_bus.py."""
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

IDS = {'shoulder_pan': 1, 'shoulder_lift': 2, 'elbow_flex': 3, 'wrist_flex': 4, 'wrist_roll': 5, 'gripper': 6}


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in IDS:
        raise SystemExit(__doc__)
    port, joint = sys.argv[1], sys.argv[2]
    bus = FeetechMotorsBus(port, motors={joint: Motor(IDS[joint], 'sts3215', MotorNormMode.RANGE_M100_100)})
    try:
        bus.setup_motor(joint)
    except RuntimeError as error:
        raise SystemExit(f'Не вышло: {error}\nНа шине должен быть ровно один ненастроенный мотор (номер 1).') from error
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    print(f'{joint}: номер {IDS[joint]}, скорость 1 000 000. Проверка: scripts\\so101\\scan_bus.py {port}')


if __name__ == '__main__':
    main()
