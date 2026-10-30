"""Teach by hand: the motors switch off (hold the arm: the shoulder falls without them) and for --seconds the joint
angles are logged at 20 Hz while you move the arm, e.g. down to the floor or the surface it should reach. Read only
after the release: nothing is commanded. The log goes to runs/so101_real/teach-….json; floor_from_teach.py turns it
into the floor's height under our model's base.

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\teach.py COM3 [--seconds 60]"""
import argparse
import json
import time

from motion_check import JOINTS, ROOT, make_bus


def main():
    parser = argparse.ArgumentParser(description='запись позы, которую задаёт рука человека (моторы выключены)')
    parser.add_argument('port')
    parser.add_argument('--seconds', type=float, default=60.)
    args = parser.parse_args()
    bus = make_bus(args.port)
    bus.connect()
    bus.disable_torque()
    print(f'моторы выключены: двигайте руку, пишу {args.seconds:.0f} с', flush=True)
    log, began = [], time.time()
    try:
        while time.time() - began < args.seconds:
            log.append(dict(t=round(time.time() - began, 3), present=bus.sync_read('Present_Position')))
            time.sleep(.05)
    finally:
        out = ROOT / 'runs' / 'so101_real'
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime('teach-%Y%m%d-%H%M%S.json')
        path.write_text(json.dumps(dict(joints=JOINTS, log=log), ensure_ascii=False), encoding='utf-8')
        bus.disconnect(disable_torque=False)
        print(f'записано {len(log)} поз: {path}')


if __name__ == '__main__':
    main()
