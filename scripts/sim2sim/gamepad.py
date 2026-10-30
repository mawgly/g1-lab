"""Virtual gamepad and scenario for sim2sim (unitree_mujoco + g1_ctrl).

g1_ctrl reads the remote only from rt/lowstate.wireless_remote, which unitree_mujoco fills from
its gamepad (`use_joystick: 1`) and also republishes as rt/wirelesscontroller. The simulator opens
`joystick_device` with a plain non-blocking open(), so a FIFO works as /dev/input/js0: this script
writes Linux joystick events (struct js_event) into it, with the Xbox layout of
unitree_mujoco/simulate/src/physics_joystick.h. The elastic band is driven with key presses sent to
the simulator window (keys from unitree_mujoco's user_key_cb: 9 on/off, 8 lower, 7 raise), see Keys.

Usage: python gamepad.py <fifo> <window id> <events.csv> [--scenario <name>] [...]
Every scenario but getup starts the same way (Passive -> FixStand on the band -> Velocity -> band off -> stand).
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
  arms       arm poses (iteration 40; g1_ctrl with patches/g1_ctrl-arms.patch and an arm_targets export, whose
             deploy.yaml run.sh hands over as --deploy): standing, each named pose (--poses, default all in deploy.yaml
             order) by its keys (LB + a button) held --pose-seconds, the release keys and PAUSE after each; then with
             --arm-walk <pose> that pose, walking --walk m/s for --walk-seconds, stop, release, --after-seconds
  getup      getting up (iteration 33; g1_ctrl with patches/g1_ctrl-getup.patch and run.sh --getup <export>): the
             robot falls (--fall), lies --lie-seconds, then LT + down: GetUp for --getup-seconds (after its 10 s
             timeout it goes back to Passive by itself); with --getup-velocity then RT + A: Velocity (the walking
             export), standing --stand-seconds, walking --walk m/s for --walk-seconds and standing --after-seconds
             (iteration 35); with --getup-fixstand LT + up: FixStand instead.
             --fall band (the old way): in Passive the band off, the limp robot drops ~0.5 m; --fall-push
             <direction>[:dv] pushes the pelvis right after the band is off (24.10: it landed on its back whatever
             the push, once on its front without one). --fall stand: the usual start, then standing in Velocity
             --fall-push pushes it (needed here; 24.10: forward from 1.5 m/s on its front, back or sideways as
             pushed) and --fall-passive s after the push LT + B (Passive, limp), unless Velocity's tilt check went
             there first; without --fall-passive only a fall gets it to Passive. With --chain-log (run.sh
             --getup-track, the getting-up chain of g1_ctrl-getupchain.patch) GetUp hands over by itself; the
             scenario follows g1_ctrl's log (events handover, track, chain_left) and presses RT + A --track-seconds
             after GetUpTrack's entry, or at --getup-seconds if it never comes
--ry holds the right stick's vertical axis (-1..1, down negative: a crouch) through the commanded part of the other
scenarios (walk: while walking); segments with the stick down are named <name>@<height> (the standing height; the
patch keeps a walk at 0.70 m or above).
Every action is logged with time.monotonic_ns() (CLOCK_MONOTONIC, same clock as sim2sim_probe); a
`seg` event (value name:vx:vy:wz) starts each commanded segment, which lasts until the next event; a
`push` event (value direction:dv) follows each push key; an `arm` event (value the pose's name or release) each arm
pose's keys.
"""

import argparse
import os
import re
import struct
import time
from pathlib import Path

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
AXIS_MAX = 32767
AXIS_SCALE = 32768  # physics_joystick.h: value / (1 << (joystick_bits - 1)), joystick_bits 16

# Xbox layout, physics_joystick.h: buttons A 0, B 1, X 2, Y 3, LB 4, RB 5; axes lx 0, ly 1 (inverted),
# LT 2 (pressed when > 0), rx 3, ry 4, RT 5, DX 6, DY 7 (up when < 0).
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
AXIS_LX = 0
AXIS_LY = 1
AXIS_LT = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_RT = 5
AXIS_DX = 6
AXIS_DY = 7
# The arm poses' keys in deploy.yaml (unitree_joystick_dsl names, g1_ctrl-arms.patch): '<modifier> + <key>', a
# shoulder button held and a face button or d-pad direction pressed. D-pad: (axis, sign) of physics_joystick.h.
ARM_MODIFIERS = {'LB': BUTTON_LB, 'RB': BUTTON_RB}
ARM_BUTTONS = {'A': BUTTON_A, 'B': BUTTON_B, 'X': BUTTON_X, 'Y': BUTTON_Y}
ARM_DPAD = {'up': (AXIS_DY, -1), 'down': (AXIS_DY, 1), 'left': (AXIS_DX, -1), 'right': (AXIS_DX, 1)}
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
# The getting-up chain (g1_ctrl-getupchain.patch): its states and g1_ctrl's lines for its changes; the log is read
# every CHAIN_POLL s.
CHAIN_STATES = ('GetUp', 'GetUpBlend', 'GetUpTrack')
CHAIN_HANDOVER = 'FSM: Change state from GetUp to GetUpBlend'
CHAIN_TRACK = 'FSM: Change state from GetUpBlend to GetUpTrack'
CHAIN_CHANGE = re.compile(r'FSM: Change state from (\w+) to (\w+)')
CHAIN_POLL = 0.05


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


def fall_push(text):
    """--fall-push: (direction, dv m/s) of '<direction>[:dv]', dv PUSH_BASE without it."""
    direction, _, dv = text.partition(':')
    return direction, float(dv) if dv else PUSH_BASE


def arm_keys(expression):
    """(modifier button, (kind, number, sign)) of an arm pose's keys '<LB|RB> + <A|B|X|Y|up|down|left|right>':
    kind 'button' (sign 1) or 'axis' (a d-pad direction)."""
    parts = [part.strip() for part in expression.split('+')]
    if len(parts) != 2 or parts[0] not in ARM_MODIFIERS or (parts[1] not in ARM_BUTTONS and parts[1] not in ARM_DPAD):
        raise ValueError(f'arm keys: <LB|RB> + <A|B|X|Y|up|down|left|right>, not {expression!r}')
    if parts[1] in ARM_BUTTONS:
        return ARM_MODIFIERS[parts[0]], ('button', ARM_BUTTONS[parts[1]], 1)
    axis, sign = ARM_DPAD[parts[1]]
    return ARM_MODIFIERS[parts[0]], ('axis', axis, sign)


def arm_poses(path):
    """(poses [(name, keys)], release keys) of a deploy.yaml's arm_command term (export_g1.arm_command_params)."""
    import yaml
    deploy = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    term = (deploy.get('observations') or {}).get('arm_command')
    if term is None:
        raise SystemExit(f'gamepad.py: {path} has no arm_command term (not an arm_targets export)')
    params = term['params']
    return [(pose['name'], pose['keys']) for pose in params['poses']], params['release']


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

    def start_on_floor(self, args):
        """Every scenario's start: Passive -> FixStand on the band, lowered onto the floor, Velocity, band off."""
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

    def arm(self, expression, name):
        """An arm pose's keys (arm_keys): the shoulder button held, the face button or d-pad direction pressed 0.3 s
        (g1_ctrl-arms.patch takes the rising edge at the policy's 50 Hz), both let go; an `arm` event."""
        modifier, (kind, number, sign) = arm_keys(expression)
        press = ((lambda on: self.pad.button(number, on)) if kind == 'button'
                 else (lambda on: self.pad.axis(number, sign * AXIS_MAX if on else 0)))
        self.pad.button(modifier, True)
        time.sleep(0.1)
        press(True)
        self.mark('arm', name)
        time.sleep(0.3)
        press(False)
        time.sleep(0.1)
        self.pad.button(modifier, False)

    def arms(self, args):
        """Standing, each pose of --poses by its keys, held --pose-seconds, then the release and PAUSE; with
        --arm-walk that pose, walking --walk m/s for --walk-seconds, stop, release, --after-seconds."""
        poses, release = arm_poses(args.deploy)
        keys = dict(poses)
        wanted = args.poses.split(',') if args.poses else [name for name, _ in poses]
        unknown = [name for name in wanted + ([args.arm_walk] if args.arm_walk else []) if name not in keys]
        if unknown:
            raise SystemExit(f'gamepad.py: poses {unknown} not in {args.deploy} ({", ".join(keys)})')
        self.mark('seg', 'stand:0:0:0')
        time.sleep(args.stand_seconds)
        for name in wanted:
            self.arm(keys[name], name)
            self.mark('seg', f'arm_{name}:0:0:0')
            time.sleep(args.pose_seconds)
            self.arm(release, 'release')
            self.mark('seg', 'arm_release:0:0:0')
            time.sleep(PAUSE)
        if args.arm_walk:
            self.arm(keys[args.arm_walk], args.arm_walk)
            self.mark('seg', f'arm_{args.arm_walk}:0:0:0')
            time.sleep(PAUSE)
            self.stick(args.walk, 0., 0.)
            self.mark('seg', f'walk_{args.arm_walk}:{args.walk:g}:0:0')
            time.sleep(args.walk_seconds)
            self.stick(0., 0., 0.)
            self.mark('seg', f'stop_{args.arm_walk}:0:0:0')
            time.sleep(PAUSE)
            self.arm(release, 'release')
            self.mark('seg', 'arm_release:0:0:0')
            time.sleep(args.after_seconds)

    def run(self, args):
        self.mark('start', args.scenario)
        if args.scenario == 'getup':
            self.getup(args)
            return
        self.start_on_floor(args)
        if args.scenario == 'arms':
            self.arms(args)
            self.mark('end')
            return
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

    def push(self, direction, dv):
        """One push key of the unitree_mujoco push patch (PUSH_KEYS, modifiers for dv) and its `push` event."""
        self.key(dict(PUSH_KEYS)[direction], push_modifiers(dv))
        self.mark('push', f'{direction}:{dv:g}')

    def getup(self, args):
        """The robot falls (--fall) and lies (Passive: kp 0, kd 3); then Passive -> GetUp with LT + down.on_pressed
        (patches/g1_ctrl-getup.patch's config.yaml) and GetUp for --getup-seconds; with --getup-velocity then GetUp ->
        Velocity (RT + A.on_pressed, as from FixStand) and the stand / walk / stop segments, with --getup-fixstand
        GetUp -> FixStand (LT + up.on_pressed)."""
        if args.fall == 'stand':  # pushed over standing in Velocity
            self.start_on_floor(args)
            self.mark('seg', 'stand:0:0:0')
            time.sleep(args.fall_stand_seconds)
            self.push(*fall_push(args.fall_push))
            if args.fall_passive is not None:
                time.sleep(args.fall_passive)
                # Velocity -> Passive (limp); nothing if its tilt check already went there (Passive has no LT + B)
                self.combo(AXIS_LT, lambda on: self.pad.button(BUTTON_B, on), 'passive')
        else:  # dropped from the band in Passive
            time.sleep(args.passive_seconds)
            self.key('9')  # band off: Passive lets the robot fall
            self.mark('band_off')
            if args.fall_push:
                self.push(*fall_push(args.fall_push))
        self.mark('lying')
        time.sleep(args.lie_seconds)
        self.combo(AXIS_LT, lambda on: self.pad.axis(AXIS_DY, AXIS_MAX if on else 0), 'getup')
        if args.chain_log:  # the getting-up chain: RT + A --track-seconds after GetUpTrack's entry
            if self.wait_chain(Path(args.chain_log), time.monotonic() + args.getup_seconds):
                time.sleep(args.track_seconds)
        else:
            time.sleep(args.getup_seconds)
        if args.getup_velocity:
            self.velocity('velocity_after_getup')
            for name, vx, seconds in (('stand', 0., args.stand_seconds), ('walk', args.walk, args.walk_seconds),
                                      ('stop', 0., args.after_seconds)):
                self.stick(vx, 0., 0.)
                self.mark('seg', f'{name}:{vx:g}:0:0')
                time.sleep(seconds)
            self.stick(0., 0., 0.)
        elif args.getup_fixstand:
            self.fixstand('fixstand')
            time.sleep(3.0)
        self.mark('end')

    def wait_chain(self, log, deadline):
        """getup with --chain-log (run.sh --getup-track): follows g1_ctrl's log until GetUpTrack is entered or the
        deadline (time.monotonic) passes; marks `handover` (GetUp -> GetUpBlend), `track` (GetUpBlend -> GetUpTrack) or
        `chain_left` (a chain state to another, e.g. Passive), each once. True once GetUpTrack is entered."""
        marks = ((CHAIN_HANDOVER, 'handover'), (CHAIN_TRACK, 'track'))
        seen = set()
        while time.monotonic() < deadline:
            text = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
            for line, name in marks:
                if line in text and name not in seen:
                    seen.add(name)
                    self.mark(name)
            if 'track' in seen:
                return True
            left = [m for m in CHAIN_CHANGE.finditer(text) if m.group(1) in CHAIN_STATES
                    and m.group(2) not in CHAIN_STATES]
            if left:
                self.mark('chain_left', left[0].group(2))
                return False
            time.sleep(CHAIN_POLL)
        return False

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
                                               'push', 'crouch', 'getup', 'arms'), default='walk')
    parser.add_argument('--deploy', help='arms: the export\'s deploy.yaml (its arm_command poses; run.sh gives it)')
    parser.add_argument('--poses', help='arms: comma-separated pose names, in this order (default all)')
    parser.add_argument('--pose-seconds', type=float, default=6.0, help='arms: seconds each pose is held')
    parser.add_argument('--arm-walk', help='arms: then walk --walk m/s holding this pose')
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
    parser.add_argument('--lie-seconds', type=float, default=5.0, help='getup: seconds lying in Passive before GetUp')
    parser.add_argument('--getup-seconds', type=float, default=12.0,
                        help='getup: seconds after LT + down (before RT + A or LT + up)')
    parser.add_argument('--chain-log', help='getup, the getting-up chain (run.sh --getup-track gives it): g1_ctrl\'s '
                                            'log; RT + A (or LT + up) --track-seconds after GetUpTrack\'s entry, or '
                                            'at --getup-seconds after LT + down without it')
    parser.add_argument('--track-seconds', type=float, default=6.6,
                        help='getup with --chain-log: seconds in GetUpTrack before RT + A (run.sh: the clip\'s rest '
                             'from the hand-over frame plus 1.5)')
    parser.add_argument('--getup-fixstand', action='store_true', help='getup: then LT + up, GetUp -> FixStand')
    parser.add_argument('--getup-velocity', action='store_true',
                        help='getup: then RT + A, GetUp -> Velocity: stand --stand-seconds, walk --walk m/s '
                             '--walk-seconds, stand --after-seconds')
    parser.add_argument('--fall', choices=('band', 'stand'), default='band',
                        help='getup: dropped limp from the band in Passive, or pushed over standing in Velocity')
    parser.add_argument('--fall-push', help='getup: <direction>[:dv] push (PUSH_KEYS directions, dv 0.5-2.25 m/s in '
                                            'steps of 0.25, 0.5 without it); band: right after the band is off, '
                                            'stand: after --fall-stand-seconds')
    parser.add_argument('--fall-stand-seconds', type=float, default=2.0,
                        help='getup --fall stand: seconds standing after the band before the push')
    parser.add_argument('--fall-passive', type=float, default=None,
                        help='getup --fall stand: LT + B (Passive) this many seconds after the push')
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
    if args.fall_push:
        try:
            direction, dv = fall_push(args.fall_push)
        except ValueError:
            parser.error(f'--fall-push: <direction>[:dv], not {args.fall_push!r}')
        if direction not in dict(PUSH_KEYS) or push_modifiers(dv) is None:
            parser.error(f'--fall-push: {sorted(dict(PUSH_KEYS))}, dv 0.5-2.25 m/s in steps of 0.25, '
                         f'not {args.fall_push!r}')
    if args.fall == 'stand' and not args.fall_push:
        parser.error('--fall stand needs --fall-push')
    if args.getup_velocity and args.getup_fixstand:
        parser.error('--getup-velocity or --getup-fixstand, not both')
    if args.scenario == 'arms':
        if not args.deploy:
            parser.error('--scenario arms needs --deploy <deploy.yaml> (run.sh gives the export\'s)')
        try:
            for _, keys in arm_poses(args.deploy)[0] + [('release', arm_poses(args.deploy)[1])]:
                arm_keys(keys)
        except ValueError as error:
            parser.error(str(error))

    pad = Gamepad(args.fifo)
    scenario = Scenario(pad, args.window, args.events)
    try:
        scenario.run(args)
    finally:
        pad.close()


if __name__ == '__main__':
    main()
