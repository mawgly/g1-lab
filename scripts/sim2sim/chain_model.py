"""The getting-up chain of scripts/sim2sim/patches/g1_ctrl-getupchain.patch in Python, number for number as its C++
(deploy/include/FSM/getup_chain.h and State_GetUpBlend.h of the patched unitree_rl_mjlab), float32 like it.

The chain (iteration 82, scripts/lab/probes/getup_chain.py; the lead's choice of 27.10): GetUp (the getting-up policy)
runs until the hand-over rule holds (Rule: getup_chain.HandOver's pelvis rule, toward the tracker's hand-over frame,
params/handover.yaml of export_g1 --handover-frame), then GetUpBlend takes the PD from GetUp's last target to that
frame (Blend: getup_chain.pd_schedule's quintic targets and gains, the targets clipped as the tracker's deploy.yaml
clips its actions), then GetUpTrack (the tracker) runs as any policy state. Iteration 89 (each key of GetUp's handover
block missing: as before): `left: reset` (Rule) starts the rule afresh when the pelvis leaves face down instead of
handing over; with retry_angle / abort_angle GetUpBlend goes back to GetUp when the pelvis IMU's gravity lies too far
from the frame's (Retry: back_reason, the returns counted against max_retries). Iteration 91 (both keys missing: as
before): with legs_home_hip / legs_home_seconds a GetUp entry that is not a return runs the legs-home phase first when
both hip pitch angles lie above legs_home_hip (LegsHome: the PD alone, the legs by the quintic to default_joint_pos, the
waist and arms held, getup_chain.legs_home_phase).

tests/test_getup_chain_patch.py checks this against getup_chain.py (the rule, the schedule) and the tracker stage's
environment (the targets); scripts/sim2sim/getup_report.py replays Blend over a sim2sim run's lowcmd (so the C++ is
checked against it) and reads the patch's log lines. Needs numpy and pyyaml only (run.sh's WSL venv).
"""

import math
from pathlib import Path

import numpy as np
import yaml

F = np.float32
# The rule's settings in config.yaml (GetUp's handover block) and their defaults, getup_chain::Rule's.
RULE_DEFAULTS = dict(first_seconds=3.0, face_angle=0.7853982, angle_scale=0.4363323, pose_scale=0.3, pass_margin=0.3,
                     good_score=2.0, face_seconds=1.0)
STATES = ('GetUp', 'GetUpBlend', 'GetUpTrack')  # config.yaml's names of the chain's states, in order
LEFT = ('handover', 'reset')  # the handover block's `left`: hand over (missing, as before) or start the rule afresh
# GetUpBlend's way back (iteration 89): the values the patch's config.yaml gives the handover block's keys; a missing
# angle is off (both missing: no way back), max_retries missing with an angle given is 2.
RETRY_DEFAULTS = dict(retry_angle=1.5707963, abort_angle=2.6179939, max_retries=2)  # 90 deg, 150 deg
# GetUp's legs-home phase (iteration 91, getup_chain::LegsHome): the handover block's keys and their defaults (either
# key turns it on, the other at its default; both missing: off), the share of the phase each leg's quintic takes, the
# legs' and the hip pitch SDK motors.
LEGS_HOME_DEFAULTS = dict(legs_home_hip=2.0, legs_home_seconds=1.0)
LEGS_HOME_RAMP = 0.7
LEGS = tuple(range(12))
HIPS = (0, 6)


def smoothstep(x):
    """getup_chain::smoothstep in float32: the quintic of x clipped to [0, 1]."""
    x = F(min(max(F(x), F(0)), F(1)))
    return F(x * x * x * (F(10) - F(15) * x + F(6) * x * x))


def angle_between(a, b):
    """getup_chain::angle_between in float32: atan2(|a x b|, a . b), rad."""
    a, b = np.asarray(a, F), np.asarray(b, F)
    cross = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], F)
    return F(math.atan2(F(math.sqrt(F((cross * cross).sum()))), F((a * b).sum())))


def imu_gravity(quat):
    """getup_chain::imu_gravity in float32: the pelvis IMU's projected gravity R^T (0, 0, -1) of (w, x, y, z)."""
    w, x, y, z = (F(v) for v in quat)
    return np.array([F(2) * (w * y - x * z), F(-2) * (w * x + y * z), -(w * w - x * x - y * y + z * z)], F)


def back_reason(angle, done, retry_angle, abort_angle):
    """getup_chain::back_reason: 'abort_angle' above abort_angle (any moment), 'retry_angle' with the steps done and
    above retry_angle, else ''; a limit of None or <= 0 is off."""
    angle = F(angle)
    if abort_angle is not None and F(abort_angle) > 0 and angle > F(abort_angle):
        return 'abort_angle'
    if done and retry_angle is not None and F(retry_angle) > 0 and angle > F(retry_angle):
        return 'retry_angle'
    return ''


class Retry:
    """GetUpBlend's way back and GetUp's count of returns (getup_chain::returns, State_RLBase::enter): enter(returned)
    on each GetUp entry gives whether it may hand over; blend(angle, done) once an FSM tick of GetUpBlend gives the
    reason it goes back ('' none), counting the return. handover: config.yaml's GetUp handover block."""

    def __init__(self, handover):
        self.retry_angle = handover.get('retry_angle')
        self.abort_angle = handover.get('abort_angle')
        self.on = self.retry_angle is not None or self.abort_angle is not None
        self.max_retries = int(handover.get('max_retries', RETRY_DEFAULTS['max_retries'])) if self.on else -1
        self.count = 0
        self.pending = False

    def enter(self):
        """A GetUp entry: True if it may hand over (a return's entry keeps the count, any other starts it)."""
        if self.pending:
            self.pending = False
        else:
            self.count = 0
        return not (self.max_retries >= 0 and self.count > self.max_retries)

    def blend(self, angle, done):
        why = back_reason(angle, done, self.retry_angle, self.abort_angle) if self.on else ''
        if why:
            self.count += 1
            self.pending = True
        return why


def deviation(joints, frame):
    """The mean |q - q0| over handover.yaml's free motors, float32 as State_RLBase::handover_due."""
    q, q0 = np.asarray(joints, F), np.asarray(frame['joints'], F)
    total = F(0)
    for motor in frame['free']:
        total = F(total + F(abs(F(q[motor] - q0[motor]))))
    return F(total / F(len(frame['free'])))


class Rule:
    """getup_chain::Rule: step(step, angle, deviation) once a policy step; returns the rule that fired or ''. left
    'reset': leaving face down starts the rule afresh instead of firing 'left'."""

    def __init__(self, dt=0.02, left='handover', **settings):
        if left not in LEFT:
            raise ValueError(f'left: {" or ".join(LEFT)}, not {left}')
        self.left_reset = left == 'reset'
        values = {**RULE_DEFAULTS, **settings}
        self.dt = F(dt)
        for key in ('face_angle', 'angle_scale', 'pose_scale', 'pass_margin', 'good_score', 'face_seconds'):
            setattr(self, key, F(values[key]))
        self.steps = int(round(values['first_seconds'] / dt))  # handover_steps: std::lround(first_seconds / step_dt)
        self.reset()

    def reset(self):
        self.faced_at = -1
        self.least = F(np.inf)

    def step(self, step, angle, deviation):
        angle, deviation = F(angle), F(deviation)
        value = F(F(angle / self.angle_scale) + F(deviation / self.pose_scale))
        if self.faced_at < 0:
            if angle > self.face_angle:
                return ''
            self.faced_at = step
        if value <= self.good_score:
            return 'good'
        if angle > self.face_angle:
            if self.left_reset:
                self.reset()
                return ''
            return 'left'
        if value >= F(self.least + self.pass_margin):
            return 'least'
        if F(F(step - self.faced_at) * self.dt) >= F(self.face_seconds - F(1e-6)):
            return 'timeout'
        self.least = min(self.least, value)
        return ''


class Blend:
    """State_GetUpBlend: the PD transition from `start` (the targets lowcmd holds on entry, SDK order) to the frame's
    joints (the locked motors at default_joint_pos) over `steps` policy steps; at(k) gives step k's (targets, kp, kd),
    k = 1 .. steps, the targets clipped to the tracker deploy.yaml's action clip. cfg: config.yaml's GetUpBlend block,
    deploy: the tracker's deploy.yaml, frame: its handover.yaml."""

    def __init__(self, cfg, deploy, frame):
        ids = [int(v) for v in deploy['joint_ids_map']]
        self.motors = n = len(ids)
        self.step_dt = F(deploy['step_dt'])
        clip = np.asarray(deploy['actions']['JointPositionAction']['clip'], F)
        home = np.asarray(deploy['default_joint_pos'], F)
        self.low, self.high, self.home = np.zeros(n, F), np.zeros(n, F), np.zeros(n, F)
        self.low[ids], self.high[ids], self.home[ids] = clip[:, 0], clip[:, 1], home
        self.stiffness = np.asarray(deploy['stiffness'], F)
        self.damping = np.asarray(deploy['damping'], F)
        self.locked = np.zeros(n, bool)
        self.locked[list(frame.get('locked') or [])] = True
        self.goal = np.asarray(frame['joints'], F).copy()
        self.goal[self.locked] = self.home[self.locked]
        seconds = F(cfg['seconds'])
        self.steps = max(1, int(np.round(F(seconds / self.step_dt))))  # std::lround: half away from zero
        ramp = F(cfg.get('ramp', 0.7))
        arms = F(cfg['arms']) if cfg.get('arms') is not None else ramp
        legs = F(cfg.get('legs', 0.0))
        self.share = np.full(n, ramp, F)
        self.late = np.full(n, F(legs / self.step_dt), F)
        arm_joints = list(cfg.get('arm_joints') or [])
        self.share[arm_joints], self.late[arm_joints] = arms, F(0)
        self.kp, self.kd = F(cfg.get('kp', 1.0)), F(cfg.get('kd', 1.0))
        self.kp_to = F(cfg['kp_to']) if cfg.get('kp_to') is not None else self.kp
        self.kd_to = F(cfg['kd_to']) if cfg.get('kd_to') is not None else self.kd
        self.start = None

    def enter(self, start):
        self.start = np.asarray(start, F).copy()

    def progress(self, k):
        span = np.maximum(F(self.share * F(self.steps) - self.late), F(1))
        return np.clip(F(F(k) - self.late) / span, F(0), F(1)).astype(F)

    def gains(self, k):
        s = smoothstep(F(F(k) / F(self.steps)))
        return F(self.kp + F(self.kp_to - self.kp) * s), F(self.kd + F(self.kd_to - self.kd) * s)

    def at(self, k):
        progress = self.progress(k)
        step = np.array([smoothstep(p) for p in progress], F)
        q = (self.start + step * (self.goal - self.start)).astype(F)
        q = np.where(self.locked, self.home, np.clip(q, self.low, self.high)).astype(F)
        kp, kd = self.gains(k)
        return q, (self.stiffness * kp).astype(F), (self.damping * kd).astype(F)

    def step_at(self, seconds):
        """The policy step State_GetUpBlend::run applies `seconds` after entry: min(steps, 1 + floor(t / step_dt))."""
        return min(self.steps, 1 + int(math.floor(F(seconds) / self.step_dt)))


class LegsHome:
    """State_RLBase's legs-home phase (getup_chain::LegsHome, iteration 91): due(joints, returned) on a GetUp entry
    from lowstate's angles (SDK order), enter(joints) from them, at(k) the targets of policy step k (1 .. steps): the
    legs by the quintic over LEGS_HOME_RAMP of the steps to default_joint_pos, every other motor held at its angle on
    entry, the locked motors at default_joint_pos, all clipped to GetUp's deploy.yaml action clip; the gains are
    GetUp's own (stiffness, damping). handover: config.yaml's GetUp handover block, deploy: GetUp's deploy.yaml, frame:
    the tracker's handover.yaml (its locked motors)."""

    def __init__(self, handover, deploy, frame):
        self.on = any(key in handover for key in LEGS_HOME_DEFAULTS)
        values = {**LEGS_HOME_DEFAULTS, **{key: handover[key] for key in LEGS_HOME_DEFAULTS if key in handover}}
        self.hip, self.seconds = F(values['legs_home_hip']), F(values['legs_home_seconds'])
        ids = [int(v) for v in deploy['joint_ids_map']]
        self.motors = n = len(ids)
        self.step_dt = F(deploy['step_dt'])
        clip = np.asarray(deploy['actions']['JointPositionAction']['clip'], F)
        home = np.asarray(deploy['default_joint_pos'], F)
        self.low, self.high, self.home = np.zeros(n, F), np.zeros(n, F), np.zeros(n, F)
        self.low[ids], self.high[ids], self.home[ids] = clip[:, 0], clip[:, 1], home
        self.stiffness = np.asarray(deploy['stiffness'], F)
        self.damping = np.asarray(deploy['damping'], F)
        self.locked = np.zeros(n, bool)
        self.locked[list(frame.get('locked') or [])] = True
        self.steps = max(1, int(np.round(F(self.seconds / self.step_dt))))  # std::lround: half away from zero
        self.start = self.goal = None

    def due(self, joints, returned=False):
        q = np.asarray(joints, F)
        return bool(self.on and not returned and q[HIPS[0]] > self.hip and q[HIPS[1]] > self.hip)

    def enter(self, joints):
        self.start = np.asarray(joints, F).copy()
        self.goal = self.start.copy()
        self.goal[list(LEGS)] = self.home[list(LEGS)]

    def progress(self, k):
        """Schedule::progress with share LEGS_HOME_RAMP and late 0, the same for every motor."""
        span = max(F(F(LEGS_HOME_RAMP) * F(self.steps)), F(1))
        return F(min(max(F(F(k) / span), F(0)), F(1)))

    def at(self, k):
        s = smoothstep(self.progress(k))
        q = (self.start + s * (self.goal - self.start)).astype(F)
        q = np.where(self.locked, self.home, np.clip(q, self.low, self.high)).astype(F)
        return q, self.stiffness.copy(), self.damping.copy()

    def step_at(self, seconds):
        """The policy step State_RLBase::legs_run applies `seconds` after the phase's start, None once it is over."""
        if F(seconds) >= F(F(self.steps) * self.step_dt):
            return None
        return min(self.steps, 1 + int(math.floor(F(seconds) / self.step_dt)))


def legs_home(config, deploy, frame):
    """LegsHome of a g1_ctrl config.yaml (the dict), GetUp's deploy.yaml and the tracker's handover.yaml (dicts)."""
    return LegsHome(config['FSM']['GetUp']['handover'], deploy, frame)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def chain(config, track_dir):
    """(rule, blend, frame, states) of a g1_ctrl config.yaml (the dict) and the tracker's policy folder (its
    params/deploy.yaml and params/handover.yaml; or an export folder with deploy.yaml and handover.yaml)."""
    track_dir = Path(track_dir)
    params = track_dir / 'params' if (track_dir / 'params').is_dir() else track_dir
    deploy, frame = load_yaml(params / 'deploy.yaml'), load_yaml(params / 'handover.yaml')
    fsm = config['FSM']
    handover = fsm['GetUp']['handover']
    rule = Rule(float(deploy['step_dt']), left=handover.get('left', 'handover'),
                **{k: v for k, v in handover.items() if k in RULE_DEFAULTS})
    return rule, Blend(fsm['GetUpBlend'], deploy, frame), frame, (fsm['GetUp'], fsm['GetUpBlend'], fsm['GetUpTrack'])


def retry(config):
    """Retry of a g1_ctrl config.yaml (the dict): GetUp's handover block's way back (off without its keys)."""
    return Retry(config['FSM']['GetUp']['handover'])
