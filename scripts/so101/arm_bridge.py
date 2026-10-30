"""The SO-101's bus for a process in another environment (reach_real.py run, in .venv, which has MuJoCo but not
LeRobot): one JSON request per stdin line, one JSON reply per stdout line. Requests: {"op": "read"} → {"present":
{joint: degrees}}; {"op": "goal", "goal": {…}} → {"ok": true}; {"op": "enable"} (goal = present, torque on);
{"op": "temperature"}; {"op": "close"} (the motors keep holding). Any error replies {"error": "…"}.

    .venv-lerobot\\Scripts\\python.exe scripts\\so101\\arm_bridge.py COM3"""
import json
import sys

from motion_check import make_bus


def main():
    bus = make_bus(sys.argv[1])
    bus.connect()
    out = sys.stdout
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request['op']
                if op == 'read':
                    reply = dict(present=bus.sync_read('Present_Position'))
                elif op == 'goal':
                    bus.sync_write('Goal_Position', request['goal'])
                    reply = dict(ok=True)
                elif op == 'enable':
                    present = bus.sync_read('Present_Position')
                    bus.sync_write('Goal_Position', present)
                    bus.enable_torque()
                    reply = dict(present=present)
                elif op == 'temperature':
                    reply = dict(temperature=bus.sync_read('Present_Temperature', normalize=False))
                elif op == 'close':
                    out.write(json.dumps(dict(ok=True)) + '\n')
                    out.flush()
                    break
                else:
                    reply = dict(error=f'unknown op {op}')
            except Exception as error:  # noqa: BLE001 — reported to the caller, which decides
                reply = dict(error=repr(error))
            out.write(json.dumps(reply) + '\n')
            out.flush()
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == '__main__':
    main()
