"""Virtual gamepad and scenario for sim2sim (unitree_mujoco + g1_ctrl).

g1_ctrl reads the remote only from rt/lowstate.wireless_remote, which unitree_mujoco fills from
its gamepad (`use_joystick: 1`) and also republishes as rt/wirelesscontroller. The simulator opens
`joystick_device` with a plain non-blocking open(), so a FIFO works as /dev/input/js0: this script
writes Linux joystick events (struct js_event) into it, with the Xbox layout of
unitree_mujoco/simulate/src/physics_joystick.h. The elastic band is driven with key presses sent to
the simulator window (keys from unitree_mujoco's user_key_cb: 9 on/off, 8 lower, 7 raise), see Keys.

Usage: python gamepad.py <fifo> <window id> <events.csv> [--scenario <name>] [...]
Every scenario starts the same way (Passive -> FixStand on the band -> Velocity -> band off -> stand).
  walk       forward --walk m/s for --walk-seconds, then stand (the first sim2sim check)
  turn       turn stick +-yaw standing (5 s each) and while walking --walk m/s (10 s each)
  side       sideways +-side m/s, 10 s each
  sidestop   sideways --side m/s (negative: right) for 10 s, then stand --after-seconds s (the stop after a
             side step alone; in `side` the step the other way starts 3 s into it)
  stopstart  (walk --walk m/s 3 s, stop 3 s) x 3
  passive    LT + B from Velocity (standing, or walking with --passive-walking), then back on the band,
             FixStand and Velocity again
  stand      only stand, --stand-seconds s with no command (a long stand: --stand-seconds 30)
  push       pushes on the pelvis forward, back, left, right (--push-order), --push-pause s apart: dv 0.5 m/s,
             then 1.0 m/s (--push-strength), or one series per --push-dv value (0.5-2.25 m/s in steps of 0.25);
             standing, or walking --walk m/s with --push-walking (needs the unitree_mujoco push patch, PUSH_KEYS)
  crouch     the height command (iteration 28; g1_ctrl with patches/g1_ctrl-height.patch and a height export):
             standing 0.70 m 8 s, 0.64 m 8 s, back to 0.78 m 4 s, then walking 0.5 and 1 m/s at 0.70 m, 10 s each,
             and stand at 0.78 m (the right stick down: c = 0.78 + 0.14 min(0, ry), at least 0.70 while walking)
--ry holds the right stick's vertical axis (-1..1, down negative: a crouch) through the commanded part of the other
scenarios (walk: while walking); segments with the stick down are named <name>@<height> (the standing height; the
patch keeps a walk at 0.70 m or above).
Every action is logged with time.monotonic_ns() (CLOCK_MONOTONIC, same clock as sim2sim_probe); a
`seg` event (value name:vx:vy:wz) starts each commanded segment, which lasts until the next event; a
`push` event (value direction:dv) follows each push key.
"""

import argparse
import os
import struct
import time

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
AXIS_MAX = 32767
AXIS_SCALE = 32768  # physics_joystick.h: value / (1 << (joystick_bits - 1)), joystick_bits 16

# Xbox layout, physics_joystick.h: buttons A 0, B 1, X 2, Y 3, LB 4, RB 5; axes lx 0, ly 1 (inverted),
# LT 2 (pressed when > 0), rx 3, ry 4, RT 5, DX 6, DY 7 (up when < 0).
BUTTON_A = 0
BUTTON_B = 1
AXIS_LX = 0
AXIS_LY = 1
AXIS_LT = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_RT = 5
AXIS_DY = 7
# Height command of g1_ctrl-height.patch: c = NOMINAL_HEIGHT + CROUCH_DEPTH * min(0, ry), the export's height_params
# for crouch_min 0.64 (deploy.yaml observations.height_command.params).
NOMINAL_HEIGHT = 0.78
CROUCH_DEPTH = 0.14

PAUSE = 3.0  # s standing between the commanded segments of a scenario

# Window keys of the unitree_mujoco push patch (scripts/sim2sim/patches/unitree_mujoco-push.patch): a 0.1 s
# horizontal force M dv / T on the pelvis in its heading frame, dv 0.5 m/s plus what each modifier of the key
# adds (PUSH_MODIFIERS: Shift 1.0 m/s, Ctrl+Shift 1.25, Alt 1.5). Numpad keys: the X keymap follows the Windows
# input language (22.10: Russian, no Latin letter keysyms), the numpad does not.
PUSH_KEYS = (('forward', 'KP_8'), ('back', 'KP_2'), ('left', 'KP_4'), ('right', 'KP_6'))
PUSH_DV = {'weak': 0.5, 'strong': 1.0}
PUSH_BASE = 0.5  # m/s, a push key without modifiers
PUSH_MODIFIERS = (('Ctrl', 0.25), ('Shift', 0.5), ('Alt', 1.0))  # m/s each adds (the patch's user_key_cb)
PUSH_LEAD = 3.0  # s of walking before the first push of `push --push-walking`


def push_modifiers(dv):
    """Modifiers of a push key for dv m/s (PUSH_MODIFIERS), None if no combination gives it."""
    for mask in range(1 << len(PUSH_MODIFIERS)):
        chosen = [(name, add) for k, (name, add) in enumerate(PUSH_MODIFIERS) if mask >> k & 1]
        if abs(PUSH_BASE + sum(add for _, add in chosen) - dv) < 1e-6:
            return tuple(name for name, _ in chosen)
    return None


def push_dvs(text):
    """--push-dv: comma-separated dv of each series, m/s."""
    return [float(value) for value in text.split(',')]


def stick_for(height):
    """The right stick's vertical axis for a pelvis height command (m) of g1_ctrl-height.patch, -1..0."""
    return max(-1.0, min(0.0, (height - NOMINAL_HEIGHT) / CROUCH_DEPTH))


def height_of(ry):
    """The height command (m) the patch makes of a right stick ry, standing."""
    return NOMINAL_HEIGHT + CROUCH_DEPTH * min(0.0, ry)


class Gamepad:
    def __init__(self, fifo):
        # Blocks until the simulator has the FIFO open for reading.
        self.fd = os.open(fifo, os.O_WRONLY)
        self.t0 = time.monotonic()

    def _event(self, kind, number, value):
        stamp = int((time.monotonic() - self.t0) * 1000) & 0xFFFFFFFF
        os.write(self.fd, struct.pack('<IhBB', stamp, value, kind, number))

    def button(self, number, pressed):
        self._event(JS_EVENT_BUTTON, number, 1 if pressed else 0)

    def axis(self, number, value):
        self._event(JS_EVENT_AXIS, number, max(-AXIS_MAX, min(AXIS_MAX, int(round(value)))))

    def close(self):
        os.close(self.fd)


class Keys:
    """Key presses for the simulator window: XSendEvent stamped with the X server's current time.

    `xdotool key --window` sends with XTEST (server time) while the window has the focus and with XSendEvent at
    time 0 otherwise; GLFW 3.4 with an input method keeps the last press time per key and drops a press that
    is not later, so after the window lost the focus every further press of a key it had got with the focus
    was lost (22.10: band never back on in one passive run). Stamping every press with the current server
    time works either way."""

    def __init__(self, window):
        from Xlib import X, display

        self.X = X
        self.display = display.Display()
        self.window = self.display.create_resource_object('window', int(window))
        self.root = self.display.screen().root
        # An unmapped window of our own: changing its property yields a PropertyNotify carrying the server time.
        self.clock = self.root.create_window(0, 0, 1, 1, 0, X.CopyFromParent, event_mask=X.PropertyChangeMask)
        self.atom = self.display.intern_atom('SIM2SIM_CLOCK')

    def now(self):
        from Xlib import Xatom

        self.clock.change_property(self.atom, Xatom.STRING, 8, b'')
        self.display.flush()
        while True:
            event = self.display.next_event()
            if event.type == self.X.PropertyNotify:
                return event.time

    def press(self, name, modifiers=()):
        from Xlib import XK
        from Xlib.protocol import event

        code = self.display.keysym_to_keycode(XK.string_to_keysym(name))
        if not code:  # e.g. a Latin letter under a Russian keymap: the press would go nowhere
            raise SystemExit(f'gamepad.py: key {name!r} is not in the X keymap')
        stamp = self.now()
        focus = self.display.get_input_focus().focus
        focused = getattr(focus, 'id', focus) == self.window.id
        # GLFW takes the modifiers from the event's state (Mod1 is Alt)
        masks = dict(Shift=self.X.ShiftMask, Ctrl=self.X.ControlMask, Alt=self.X.Mod1Mask)
        state = 0
        for modifier in modifiers:
            state |= masks[modifier]
        for kind, mask in ((event.KeyPress, self.X.KeyPressMask), (event.KeyRelease, self.X.KeyReleaseMask)):
            self.window.send_event(kind(time=stamp, root=self.root, window=self.window, same_screen=1,
                                        child=self.X.NONE, root_x=0, root_y=0, event_x=0, event_y=0, state=state,
                                        detail=code), propagate=True, event_mask=mask)
        self.display.flush()
        return stamp, focused


def segments(args):
    """Commanded part of a scenario after the band is off: (name, vx, vy, wz, seconds[, ry]); ry (the right stick's
    vertical axis) is --ry where not given."""
    stand = [('stand', 0., 0., 0., args.stand_seconds)]
    v, w, s = args.walk, args.yaw, args.side
    if args.scenario == 'crouch':
        low, lowest = stick_for(0.70), stick_for(0.64)
        return stand + [('crouch', 0., 0., 0., 8., low), ('crouch', 0., 0., 0., 8., lowest),
                        ('rise', 0., 0., 0., 4., 0.), ('walk', 0.5, 0., 0., 10., low), ('walk', 1.0, 0., 0., 10., low),
                        ('stop', 0., 0., 0., args.after_seconds, 0.)]
    if args.scenario == 'turn':
        return stand + [('turn_stand', 0., 0., w, 5.), ('stop', 0., 0., 0., PAUSE),
                        ('turn_stand', 0., 0., -w, 5.), ('stop', 0., 0., 0., PAUSE),
                        ('turn_walk', v, 0., w, 10.), ('stop', 0., 0., 0., PAUSE),
                        ('turn_walk', v, 0., -w, 10.), ('stop', 0., 0., 0., args.after_seconds)]
    if args.scenario == 'side':
        return stand + [('side', 0., s, 0., 10.), ('stop', 0., 0., 0., PAUSE),
                        ('side', 0., -s, 0., 10.), ('stop', 0., 0., 0., args.after_seconds)]
    if args.scenario == 'sidestop':
        return stand + [('side', 0., s, 0., 10.), ('stop', 0., 0., 0., args.after_seconds)]
    if args.scenario == 'stopstart':
        return stand + [('walk', v, 0., 0., 3.), ('stop', 0., 0., 0., 3.)] * 3
    if args.scenario == 'passive':
        return stand + ([('walk', v, 0., 0., 3.)] if args.passive_walking else [])
    if args.scenario == 'stand':
        return stand
    if args.scenario == 'push':  # the pushes follow (Scenario.pushes)
        return stand + ([('walk', v, 0., 0., PUSH_LEAD)] if args.push_walking else [])
    raise ValueError(args.scenario)


class Scenario:
    def __init__(self, pad, window, log_path):
        self.pad = pad
        self.keys = Keys(window)
        self.log = open(log_path, 'w', encoding='utf-8')
        self.log.write('t_ns,event,value\n')

    def mark(self, event, value=''):
        self.log.write(f'{time.monotonic_ns()},{event},{value}\n')
        self.log.flush()
        print(f'{time.strftime("%H:%M:%S")} {event} {value}', flush=True)

    def key(self, key, modifiers=()):
        stamp, focused = self.keys.press(key, modifiers)
        name = '+'.join((*modifiers, key))
        print(f'{time.strftime("%H:%M:%S")} key {name} (X time {stamp}, window focused: {focused})', flush=True)

    def combo(self, trigger_axis, press, name):
        """Hold a trigger (L2/R2), then press a button or d-pad direction, as on the remote."""
        self.pad.axis(trigger_axis, AXIS_MAX)
        time.sleep(0.4)  # the trigger is an Axis smoothed twice (sim and g1_ctrl), > 0.5 after ~50 ms
        press(True)
        self.mark(name)
        time.sleep(0.3)
        press(False)
        time.sleep(0.1)
        self.pad.axis(trigger_axis, 0)

    def stick(self, vx, vy, wz, ry=0.):
        """velocity_commands = (ly, -lx, -rx) in g1_ctrl; ly = -axis 1, lx = axis 0, rx = axis 3 (/ 32768); the
        height command (g1_ctrl-height.patch) reads ry = -axis 4."""
        self.pad.axis(AXIS_LY, -vx * AXIS_SCALE)
        self.pad.axis(AXIS_LX, -vy * AXIS_SCALE)
        self.pad.axis(AXIS_RX, -wz * AXIS_SCALE)
        self.pad.axis(AXIS_RY, -ry * AXIS_SCALE)

    def fixstand(self, name):
        # Passive -> FixStand: LT + up.on_pressed (deploy/robots/g1/config/config.yaml)
        self.combo(AXIS_LT, lambda on: self.pad.axis(AXIS_DY, -AXIS_MAX if on else 0), name)

    def velocity(self, name):
        # FixStand -> Velocity: RT + A.on_pressed
        self.combo(AXIS_RT, lambda on: self.pad.button(BUTTON_A, on), name)

    def run(self, args):
        self.mark('start', args.scenario)
        time.sleep(args.passive_seconds)
        self.fixstand('fixstand')
        time.sleep(3.0)  # FixStand moves to the home pose over 2 s (ts: [0, 2])
        # Lower the robot onto the floor: each '8' lengthens the band by 0.1 m (anchor 0,0,3, 200 N/m on
        # the pelvis distance): with 0.6 m the band still carries ~95% of the weight; FixStand alone does
        # not hold the robot on the floor (docs/lab-knowledge.md, um_probe fix_air).
        for _ in range(args.band_presses):
            self.key('8')
            time.sleep(0.5)
        self.mark('band_lowered', f'{args.band_presses * 0.1:.1f}')
        time.sleep(1.5)
        self.velocity('velocity')
        time.sleep(args.band_seconds)
        self.key('9')  # band off
        self.mark('band_off')
        if args.scenario == 'walk':
            time.sleep(args.stand_seconds)
            # Forward: ly = -axis/32768, velocity_commands obs[0] = ly (no scaling, clamped to lin_vel_x)
            self.pad.axis(AXIS_LY, -args.walk * AXIS_SCALE)
            self.pad.axis(AXIS_RY, -args.ry * AXIS_SCALE)  # --ry: walk at the height command
            self.mark('walk', f'{args.walk}')
            time.sleep(args.walk_seconds)
            self.pad.axis(AXIS_LY, 0)
            self.pad.axis(AXIS_RY, 0)
            self.mark('stop')
            time.sleep(args.after_seconds)
            self.mark('end')
            return
        for name, vx, vy, wz, seconds, *rest in segments(args):
            ry = rest[0] if rest else args.ry
            self.stick(vx, vy, wz, ry)
            if ry < 0:  # the height command standing (the patch keeps a walk at 0.70 m or above)
                name = f'{name}@{height_of(ry):.2f}'
            self.mark('seg', f'{name}:{vx:g}:{vy:g}:{wz:g}')
            time.sleep(seconds)
        if args.scenario == 'passive':
            self.passive(args)
        if args.scenario == 'push':
            self.pushes(args)
        self.stick(0., 0., 0.)
        self.mark('end')

    def pushes(self, args):
        """Pushes forward, back, left, right (--push-order), each followed by --push-pause s, dv 0.5 m/s first,
        then 1.0 m/s (--push-strength), or a series for each --push-dv value. The stick stays as it was (walking
        with --push-walking: then stop and stand)."""
        levels = ('weak', 'strong') if args.push_strength == 'both' else (args.push_strength,)
        dvs = push_dvs(args.push_dv) if args.push_dv else [PUSH_DV[level] for level in levels]
        keys = dict(PUSH_KEYS)
        for dv in dvs:
            for direction in args.push_order.split(','):
                self.key(keys[direction], push_modifiers(dv))
                self.mark('push', f'{direction}:{dv:g}')
                time.sleep(args.push_pause)
        if args.push_walking:
            self.stick(0., 0., 0.)
            self.mark('seg', 'stop:0:0:0')
            time.sleep(args.after_seconds)

    def passive(self, args):
        """Velocity -> Passive (LT + B.on_pressed: kp 0, kd 3), then the way back: band on, FixStand, Velocity."""
        self.combo(AXIS_LT, lambda on: self.pad.button(BUTTON_B, on), 'passive')
        self.stick(0., 0., 0.)
        time.sleep(args.passive_hold)
        self.key('9')  # band on again (same length), as the hoist on the robot
        self.mark('band_on')
        time.sleep(3.0)
        self.fixstand('fixstand_again')
        time.sleep(3.0)
        self.velocity('velocity_again')
        time.sleep(args.band_seconds)
        self.key('9')
        self.mark('band_off_again')
        self.mark('seg', 'stand_again:0:0:0')
        time.sleep(args.stand_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('fifo')
    parser.add_argument('window')
    parser.add_argument('events')
    parser.add_argument('--scenario', choices=('walk', 'turn', 'side', 'sidestop', 'stopstart', 'passive', 'stand',
                                               'push', 'crouch'), default='walk')
    parser.add_argument('--ry', type=float, default=0.,
                        help='right stick vertical axis through the commanded segments, -1..1 (down negative: the '
                             'height command of g1_ctrl-height.patch, c = 0.78 + 0.14 min(0, ry))')
    parser.add_argument('--passive-seconds', type=float, default=2.0)
    parser.add_argument('--band-presses', type=int, default=6)
    parser.add_argument('--band-seconds', type=float, default=1.0)
    parser.add_argument('--stand-seconds', type=float, default=5.0)
    parser.add_argument('--walk', type=float, default=0.5)
    parser.add_argument('--walk-seconds', type=float, default=10.0)
    parser.add_argument('--after-seconds', type=float, default=5.0)
    parser.add_argument('--yaw', type=float, default=0.5, help='turn: stick yaw rate, rad/s')
    parser.add_argument('--side', type=float, default=0.3,
                        help='side, sidestop: sideways speed, m/s (sidestop: negative steps right)')
    parser.add_argument('--passive-walking', action='store_true', help='passive: press LT + B while walking')
    parser.add_argument('--passive-hold', type=float, default=4.0, help='passive: seconds in Passive')
    parser.add_argument('--push-walking', action='store_true', help='push: push while walking --walk m/s')
    parser.add_argument('--push-strength', choices=('weak', 'strong', 'both'), default='both',
                        help='push: dv 0.5 m/s (weak), 1.0 m/s (strong) or both series, weak first')
    parser.add_argument('--push-dv', help='push: comma-separated dv of each series instead of --push-strength, m/s '
                                          '(0.5-2.25 in steps of 0.25), e.g. 1.25,1.5')
    parser.add_argument('--push-pause', type=float, default=4.0, help='push: seconds after each push')
    parser.add_argument('--push-order', default=','.join(d for d, _ in PUSH_KEYS),
                        help='push: directions of each series, comma-separated (a fall ends what the series shows)')
    args = parser.parse_args()
    unknown = set(args.push_order.split(',')) - {d for d, _ in PUSH_KEYS}
    if unknown:
        parser.error(f'--push-order: unknown {sorted(unknown)}')
    if not -1. <= args.ry <= 1.:
        parser.error(f'--ry: -1..1, not {args.ry:g}')
    if args.push_dv:
        try:
            bad = [dv for dv in push_dvs(args.push_dv) if push_modifiers(dv) is None]
        except ValueError:
            parser.error(f'--push-dv: numbers, not {args.push_dv!r}')
        if bad:
            parser.error(f'--push-dv: {bad} not in 0.5-2.25 m/s in steps of 0.25')

    pad = Gamepad(args.fifo)
    scenario = Scenario(pad, args.window, args.events)
    try:
        scenario.run(args)
    finally:
        pad.close()


if __name__ == '__main__':
    main()
