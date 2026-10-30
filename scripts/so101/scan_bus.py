"""Read-only check of a SO-101 arm on its servo bus (Feetech STS3215, LeRobot's driver): which servo IDs answer at
which baud rate, and for each the position, supply voltage and temperature. Nothing is written to the servos and the
torque is not switched on, so it is safe to run with the arm wired any way.

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\scan_bus.py COM5

The port comes from `lerobot-find-port` (unplug and plug the board's USB-C when it asks). A SO-101 configured by
`lerobot-setup-motors` answers with IDs 1..6 at 1 000 000 baud: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex,
4 wrist_flex, 5 wrist_roll, 6 gripper (LeRobot's so101 joint order). New servos all answer as ID 1 at that rate,
so more than one of them on the bus at once collide until each has its own ID."""
import sys

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as TABLE

JOINTS = {1: 'shoulder_pan', 2: 'shoulder_lift', 3: 'elbow_flex', 4: 'wrist_flex', 5: 'wrist_roll', 6: 'gripper'}


def register(bus, motor_id, name):
    """One control table register of one servo, read raw over the bus (the driver's packet handler)."""
    address, size = TABLE[name]
    reader = {1: bus.packet_handler.read1ByteTxRx, 2: bus.packet_handler.read2ByteTxRx}[size]
    value, result, error = reader(bus.port_handler, motor_id, address)
    return None if result != 0 else value


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    port = sys.argv[1]
    found = FeetechMotorsBus.scan_port(port)
    if not any(found.values()):
        raise SystemExit(f'{port}: ни один сервопривод не ответил. Проверьте питание платы, перемычку (режим USB) и '
                         f'кабели шины.')
    for baudrate, ids in found.items():
        if not ids:
            continue
        print(f'скорость {baudrate}: ответили ID {sorted(ids)}')
    ids = found.get(1_000_000) or next(ids for ids in found.values() if ids)
    baudrate = 1_000_000 if found.get(1_000_000) else next(rate for rate, ids in found.items() if ids)
    bus = FeetechMotorsBus(port, motors={})
    bus.connect(handshake=False)
    bus.set_baudrate(baudrate)
    try:
        for motor_id in sorted(ids):
            position = register(bus, motor_id, 'Present_Position')
            voltage = register(bus, motor_id, 'Present_Voltage')
            temperature = register(bus, motor_id, 'Present_Temperature')
            print(f'  ID {motor_id} ({JOINTS.get(motor_id, "не из SO-101")}): положение {position} из 4095, '
                  f'питание {None if voltage is None else voltage / 10:.1f} В, температура {temperature} °C')
    finally:
        bus.disconnect(disable_torque=False)
    missing = sorted(set(JOINTS) - set(ids))
    if missing:
        print('не ответили:', ', '.join(f'{i} {JOINTS[i]}' for i in missing))
    if set(ids) == {1} and len(ids) == 1:
        print('Ответил один ID 1: либо подключён один сервопривод, либо всем ещё не заданы ID '
              '(lerobot-setup-motors задаёт их по одному).')


if __name__ == '__main__':
    main()
