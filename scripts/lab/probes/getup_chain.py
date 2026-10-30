"""CPU only. getup_chain.py [--first <stage>] [--second <stage>] [--handover direct | --handover pd S …]
[--physics ours,delay20,foot0.3,motor1.25] [--classes back,side] [--first-seconds 3] [--second-seconds S]
[--left handover|reset] [--retry-angle DEG [--abort-angle 150] [--max-retries 2]]
[--legs-home RAD [--legs-home-seconds 1.0]] [--workers 4] [--json <file>]
A chain of two ready policies getting up from the back (iteration 82, the lead's decision): branch 81 learns to track
the spliced roll clip roll-1164902400-r6-phys100-a493v3f18 and breaks at 0.38 s on its leg swing, while the splice was
made in physics from the same three pieces: the plain getting-up policy 1164 from the bank row to face down, a PD
transition, A493 v3. So the chain «policy A → hand-over → policy B» is run here as the robot would run it, two states
of g1_ctrl's FSM.

Policy A (--first, by default exp-207eb498dc824027 @ 1164902400, a plain g1_getup stage, pd_bound at its lowest) runs
from a bank row until the hand-over rule holds (HandOver, by the robot's own sensors: the newest observed frame's pelvis
projected gravity and joint angles, sensor noise and encoder bias included; nothing a robot does not measure):
  - face down: the observed pelvis gravity within FACE_ANGLE of the target frame's (A493 v3 frame TARGET_FRAME 18,
    the splice's join);
  - then the score angle/25° + mean |q − q0|/0.3 rad (getup_roll_clip's CUT_SCORE against the same frame) is followed
    and the hand-over comes at the first step where it has risen PASS_MARGIN above its least since face down (the
    least passed: 1164 never lies still on the stomach, it crosses it with the knees tucked at 10-15 rad/s), or where
    it is at most GOOD_SCORE (within the settling rule's corner), or where the pelvis leaves face down, or
    FACE_SECONDS after face down;
  - none within --first-seconds: no hand-over, A goes on alone to the run's end (recorded, not a chain).
The joint speeds are not in the rule: the sensor noise on them (±1.5 rad/s) is above the settling rule's 0.5 rad/s.
Then, --handover:
  - direct: policy B (--second, by default the tracker of branch 73, exp-84b0540b0092467d @ 570359808, track_clip A493
    v3) at once; its clip clock at the frame nearest the observed pose (task.track_nearest_frame on the observed
    gravity and joints among the clip's lying frames: the anchor within TRACK_NEAREST_ANCHOR of frame 0's height, the
    robot measures no height), the reference aligned under the robot as a bank start's (task.track_align, the bank
    start's grace). A B trained with task_config track_handover_starts (iteration 83) is handed over as it trained:
    through its reset option track_handover (A's state and last applied actions, no noise), the clock and the rule's
    target frame at its hand-over bank's frame (scripts/lab/getup_handover_bank.py), its own longer grace;
  - pd S: the PD alone for S s, its targets from A's last target to the joints of frame TARGET_FRAME by the quintic
    smoothstep over BLEND_RAMP of S, then held (getup_roll_clip's physics blend: pd_bound off, the policy gains), then B
    with its clock at frame TARGET_FRAME.
Iteration 89, as g1_ctrl-getupchain.patch's config.yaml handover keys: --left reset, the pelvis leaving face down
starts the rule afresh instead of handing over; --retry-angle D (pd only), after each PD step the observed pelvis
angle to the target frame's: above --abort-angle at any step, or above D at the last one, back to A (entered again as
g1_ctrl enters GetUp: pd_bound lowest, last action 0, history filled, rule afresh), and after more than --max-retries
returns A runs alone ALONE_SECONDS from its last entry. Each return is recorded (returns).
Iteration 91, as the config.yaml handover keys legs_home_hip / legs_home_seconds: --legs-home H, at the run's start
(A's first entry, not a return) with both observed hip pitch angles above H rad, the PD alone for --legs-home-seconds
first (legs_home_phase: the legs by the quintic from their observed angle to home, the waist and arms held at theirs,
the policy gains), then A as g1_ctrl enters GetUp; the times of the run count from its start, the phase included.
B starts as g1_ctrl enters a policy state: last action 0, the history filled with the first frame; the actions in flight
under a delay are A's (or the PD's). The tracker's actor does not see its clip clock (the clock and the reference are
the critic's extras): on the robot B is a policy that runs for its clip's rest and hold; here the clock only gives the
run's length and the tracking terminations, which are recorded (the first one) but do not end the run, since a robot
has no reference. A run ends at B's clip end plus TRACK_HOLD_SECONDS (or --second-seconds), on a non-finite state or
a joint above GETUP_MAX_JOINT_SPEED.

Starts: the bank's held-out rows of --classes (back, side: 10 each), every --physics (getup_probe.PHYSICS: ours, a 20
ms delay, foot friction 0.3, motor friction ×1.25; the files' friction elsewhere), no assist, deterministic; the CUDA
task's sensor noise on every observed frame and an encoder bias per run (chatter.noisy, as train_falls / chatter),
numpy default_rng((seed, chatter.NOISE_STREAM)).
Per run: the hand-over (reached, its time, the rule that fired, the pose then: angle to the target frame's gravity,
mean |q − q0|, fastest free joint, pelvis speed, torso height; direct: the frame chosen); up (getup_success_pose with
both soles, the time from the start); standing at the end (task.getup_standing: head top above 1.15 m, torso tilt below
15°, pelvis above 0.70 m, both soles down; the same rule as getup_track's «встали и стоят»); the seams: the fastest
free joint over SEAM_BEFORE s before to SEAM_AFTER s after each switch (physics substeps), the PD target's jump at the
switch (largest free joint); the floor's force on both hands (physics substeps; per phase A / PD / B its largest and the
seconds above HAND_HIGH); the feet apart at the end (ankle_roll origins, horizontal); B's first tracking termination (s
after the switch) and how far its clock got. The summary per variant, class and physics, and the lead's goal: standing
by the chain at least GOAL_STANDING of 10 from the back in each of GOAL_PHYSICS.
The first data (27.10, 10 back + 10 side rows × 4 physics): direct 0 of 80 stand (B from A's state in flight, 9-14
rad/s and 0.9 rad from any clip frame, flails); pd 1.0 from the back 10 / 9 / 10 / 10, side 8 / 8 / 9 / 7, up at
5.6 s, the PD's landing on the hands 0.6-2.4 kN for 0.02-0.1 s; pd 2.0 from the back 10 of 10 in each, up at 6.7 s,
hands ≤ 1.3 kN, side 5 / 7 / 8 / 7.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

SIM = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import getup_probe  # noqa: E402

FIRST = 'runs/humanoid_lab/exp-207eb498dc824027/stage_001164902400'
SECOND = 'runs/humanoid_lab/exp-84b0540b0092467d/stage_000570359808'
PHYSICS = ('ours', 'delay20', 'foot0.3', 'motor1.25')
CLASSES = ('back', 'side')
GOAL_PHYSICS = ('ours', 'delay20', 'foot0.3')  # the lead's goal: standing ≥ GOAL_STANDING of 10 from the back in each
GOAL_STANDING = 8
TARGET_FRAME = 18  # the splice's join with A493 v3 (its right hand lies in the torso over frames 0-18)
FACE_ANGLE = np.radians(45.)  # observed pelvis gravity within this of the target frame's: face down
PASS_MARGIN = .3  # the score this far above its least since face down: the least has passed (noise ~0.2)
GOOD_SCORE = 2.  # angle 25° and deviation 0.3 together: the settling rule's corner
FACE_SECONDS = 1.  # s after face down at most
FIRST_SECONDS = 3.  # s of A at most before a hand-over
ALONE_SECONDS = 10.  # s of a run without a hand-over (A alone, the getting-up episode's length)
SEAM_BEFORE, SEAM_AFTER = .1, .5  # s around a switch for its fastest joint
HAND_HIGH = 500.  # N on both hands together: the control steps above it are counted per phase
SEED = 82_500
# --rule torso (iteration 83): the torso's projected gravity z, as the robot knows it (the pelvis IMU's gravity turned by
# the waist joints' angles, both observed: g1_ctrl reads the pelvis IMU), within this of the target frame's torso, the
# quantity the tracker's anchor gravity rule compares (its threshold g1_tensor_math.TRACK_GRAVITY_Z 0.8). The pelvis
# rule handed 1164's robot over with the chest raised: a gap of 0.71 on the tracker's first step (branches 83, 84).
TORSO_GAP = .3
RULES = ('pelvis', 'torso')
# Iteration 89 (the critic's it89: in the Unitree bridge 6 of 9 chain failures end the PD transition on the back, the
# pelvis 176-179° from the target frame's, every success ≤ 11° at the end and ≤ 125° inside it), g1_ctrl-getupchain's
# config.yaml handover keys: --left reset (the pelvis leaving face down starts the rule afresh, LEFTS), --retry-angle
# (GetUpBlend back to GetUp, A again, when the observed pelvis lies more than this from the frame's at the
# transition's end, or more than --abort-angle at any step of it; at most MAX_RETRIES such returns hand over again).
LEFTS = ('handover', 'reset')
ABORT_ANGLE = np.radians(150.)
MAX_RETRIES = 2
# The PD transition's shape (--handover pd S key=value …): ramp, the share of S over which each joint's quintic reaches
# the target frame (then held; getup_roll_clip.BLEND_RAMP); arms, the arm joints' own share (None: ramp); legs, s the
# legs' and waist's quintic starts late (reaching the frame at the same share); kp, kd, the PD gains' multipliers
# while it runs; kp_to, kd_to, the multipliers they reach at the transition's end by the quintic (None: kp, kd
# throughout; B starts at the policy's own gains without a jump only with both at 1). The defaults are the first
# version's transition, bit for bit.
PD_DEFAULTS = dict(ramp=.7, arms=None, legs=0., kp=1., kd=1., kp_to=None, kd_to=None)
# Iteration 91 (the critic's it91: in the Unitree bridge 6 of the 7 runs where 1164 got up alone instead of handing over
# started on the back after a forward push, both hip pitch angles +2.64…+2.88 rad, the stop 2.88, a pose its start bank
# never has, 0 of 40 060 rows; with this phase first 80 of 80 such starts chain in our physics against 39), g1_ctrl-
# getupchain's config.yaml handover legs_home_hip / legs_home_seconds: --legs-home, --legs-home-seconds.
LEGS_HOME_HIP = 2.
LEGS_HOME_SECONDS = 1.
_cache = {}


def torso_gravity(raw, gravity, joints):
    """(3,) the torso's projected gravity from the pelvis's (3,) and the joint angles (29,), both as observed: the
    pelvis-to-torso rotation of the waist joints by mj_kinematics in a data of its own, g_torso = R_ptᵀ·g_pelvis."""
    import mujoco
    data = _cache.setdefault(('torso', id(raw.model)), mujoco.MjData(raw.model))
    data.qpos[:] = 0
    data.qpos[3] = 1.
    data.qpos[raw.qadr] = joints
    mujoco.mj_kinematics(raw.model, data)
    turn = data.xmat[raw.torso].reshape(3, 3).T @ data.xmat[raw.pelvis].reshape(3, 3)
    return turn @ gravity


def torso_gap(raw, gravity, joints, goal_torso):
    """|g_z of the observed torso − g_z of the target frame's torso| (goal_torso its projected gravity, (3,))."""
    return abs(float(torso_gravity(raw, gravity, joints)[2]) - float(goal_torso[2]))


def pd_shares(nu, arm_joints, shape, dt):
    """(share, late) per joint (nu,) of a PD transition's shape (PD_DEFAULTS' keys): share the part of the transition
    over which each joint's quintic reaches the target frame (ramp; the arm joints' arms), late the policy steps the
    legs and waist start later (0 for the arms)."""
    share = np.full(nu, shape['ramp'])
    if shape['arms'] is not None:
        share[list(arm_joints)] = shape['arms']
    late = np.full(nu, shape['legs'] / dt)
    late[list(arm_joints)] = 0.
    return share, late


def pd_schedule(k, blend, share, late, shape):
    """Step k (1 … blend) of a PD transition of `blend` policy steps: (progress per joint before the smoothstep, kp
    multiplier, kd multiplier). The gains go from kp (kd) to kp_to (kd_to) by the smoothstep of k / blend (None: kp,
    kd throughout); the first version's progress k / (ramp·blend) without legs or arms. g1_ctrl-getupchain.patch's
    GetUpBlend runs the same (getup_chain::Schedule; scripts/sim2sim/chain_model.py in Python)."""
    import getup_roll_clip
    kp, kd = shape['kp'], shape['kd']
    if shape['kp_to'] is not None:  # the stiffness back to kp_to by the end
        kp = shape['kp'] + (shape['kp_to'] - shape['kp']) * getup_roll_clip.smoothstep(k / blend)
    if shape['kd_to'] is not None:  # the damping back to kd_to by the end
        kd = shape['kd'] + (shape['kd_to'] - shape['kd']) * getup_roll_clip.smoothstep(k / blend)
    if shape['legs'] or shape['arms'] is not None:
        progress = np.clip((k - late) / np.maximum(share * blend - late, 1.), 0., 1.)
    else:  # the first version's progress, bit for bit
        progress = k / (shape['ramp'] * blend)
    return progress, kp, kd


def pd_transition(env, start, goal_joints, seconds, shape, applied, each=None, stop=None):
    """The PD transition of --handover pd S in env (a g1_getup CPU task right after A's hand-over): pd_bound off, then
    `seconds` of policy steps whose action is the quintic target (start, A's last target, to goal_joints, by
    pd_schedule of shape: PD_DEFAULTS' keys over their defaults) over the action scale, the PD gains scaled while it
    runs and given back after. each(bad) after every step (bad: a non-finite state). It stops on a non-finite state or
    a joint above GETUP_MAX_JOINT_SPEED, or when stop(k, blend) (after each(), k = 1 … blend) is true. Returns
    dict(obs, target: the last PD target, fell: None, 'bad_state' or 'speed', applied: `applied` (the last applied
    actions, newest first) with the transition's applied actions pushed in front, its length kept, stopped: the step
    stop() ended it after, or None)."""
    import getup_roll_clip
    raw = env.unwrapped
    if raw.bound is not None:
        raw.set_pd_bound(None)
    blend = int(round(seconds / raw.dt))
    # The transition's shape (PD_DEFAULTS, --handover pd S key=value …): each joint's quintic from A's last target to
    # the target frame over its share of S (ramp; the arm joints' arms), the legs and waist starting `legs` s late; the
    # PD gains scaled by kp / kd while it runs.
    shape = {**PD_DEFAULTS, **shape}
    share, late = pd_shares(raw.model.nu, raw.arm_joints, shape, raw.dt)
    gains = raw.kp, raw.kd
    raw.kp, raw.kd = gains[0] * shape['kp'], gains[1] * shape['kd']
    obs = target = fell = stopped = None
    applied = list(applied)
    try:
        for k in range(1, blend + 1):
            progress, kp, kd = pd_schedule(k, blend, share, late, shape)
            raw.kp, raw.kd = gains[0] * kp, gains[1] * kd
            goal = start + getup_roll_clip.smoothstep(progress) * (goal_joints - start)
            action = raw.applied_action((goal - raw.home) / raw.scale)
            target = raw._target(action)
            obs, _, terminated, _, info = env.step((goal - raw.home) / raw.scale)
            applied = [action.copy(), *applied[:-1]]
            if each is not None:
                each(bool(info.get('bad_state')))
            if info.get('bad_state'):
                fell = 'bad_state'
                break
            if terminated:
                fell = 'speed'
                break
            if stop is not None and stop(k, blend):
                stopped = k
                break
    finally:
        raw.kp, raw.kd = gains
    return dict(obs=obs, target=target, fell=fell, applied=applied, stopped=stopped)


def leg_joints(raw):
    """(legs, hips): the indices of the leg joints (hip, knee, ankle) and of the two hip pitch joints (left, right)
    among the actuated joints (g1_ctrl's SDK order: 0-11 and 0, 6)."""
    names = [raw.model.joint(j).name for j in raw.joints]
    legs = [k for k, name in enumerate(names) if any(part in name for part in ('hip', 'knee', 'ankle'))]
    return legs, [k for k, name in enumerate(names) if 'hip_pitch' in name]


def legs_home_phase(env, hip, seconds, applied, each=None):
    """The legs-home phase (--legs-home, iteration 91) in env (a g1_getup CPU task right after A's reset): with both
    observed hip pitch angles (the newest frame, noise and bias included, as g1_ctrl reads lowstate) above hip (rad),
    pd_transition for `seconds` from the observed angles to them with the legs at home (the default shape: ramp 0.7,
    the policy gains, the locked wrists at home); then A's entry as g1_ctrl's GetUp policy starts: pd_bound at its
    lowest again if it was on, last action 0, the history filled. None when not due, else dict(hips: the observed ones
    on entry, transition: pd_transition's dict, obs: the observation A starts from)."""
    raw = env.unwrapped
    _, joints = reading(raw)
    legs, hips = leg_joints(raw)
    if not all(joints[k] > hip for k in hips):
        return None
    goal = joints.copy()
    goal[legs] = raw.home[legs]
    bound = raw.bound
    transition = pd_transition(env, joints, goal, seconds, {}, applied, each)
    if bound is not None:
        raw.set_pd_bound('min')
    raw.previous_action[:] = 0
    obs = raw._observation('fill')
    return dict(hips=[float(joints[k]) for k in hips], transition=transition, obs=obs)


def score(angle, deviation):
    """getup_roll_clip's CUT_SCORE: angle / 25° + deviation / 0.3 rad."""
    from humanoid_sim.tasks import g1_tensor_math as task
    return angle / task.SETTLE_DONE_ANGLE + deviation / task.SETTLE_DONE_POSE


class HandOver:
    """The hand-over rule, one call per control step with the observed angle (rad) and deviation (rad) to the target
    frame; returns the rule that fired ('least', 'good', 'left', 'timeout') or None. With torso (--rule torso,
    iteration 83 after branches 83 and 84): the observed torso gravity too (torso_gap, torso_gravity_gap), and the
    hand-over comes at the first step face down (the pelvis within FACE_ANGLE) with the torso within TORSO_GAP of the
    target frame's ('torso'); none else fires, so a run whose chest never comes down within --first-seconds hands
    nothing over. left 'reset' (--left reset, iteration 89): the pelvis leaving face down fires nothing, the rule
    starts afresh (not face down, no least)."""

    def __init__(self, dt, torso=False, left='handover'):
        self.dt = dt
        self.torso = torso
        self.left = left
        self.faced_at = None
        self.least = np.inf

    def __call__(self, step, angle, deviation, torso_gap=None):
        if self.torso:
            if angle > FACE_ANGLE:
                return None
            if self.faced_at is None:
                self.faced_at = step
            return 'torso' if torso_gap <= TORSO_GAP else None
        value = score(angle, deviation)
        if self.faced_at is None:
            if angle > FACE_ANGLE:
                return None
            self.faced_at = step
        if value <= GOOD_SCORE:
            return 'good'
        if angle > FACE_ANGLE:
            if self.left == 'reset':
                self.faced_at, self.least = None, np.inf
                return None
            return 'left'
        if value >= self.least + PASS_MARGIN:
            return 'least'
        if (step - self.faced_at) * self.dt >= FACE_SECONDS - 1e-9:
            return 'timeout'
        self.least = min(self.least, value)
        return None


class Retry:
    """GetUpBlend's way back to GetUp (iteration 89, --retry-angle): once a step of the PD transition, the observed
    pelvis angle (rad) to the target frame's gravity; above abort_angle at any step or above retry_angle at the last
    one, A runs again (counted). allowed(): whether this entry of A may hand over, false after more than max_retries
    returns (then A alone to its run's end). A limit of None is off."""

    def __init__(self, retry_angle=None, abort_angle=ABORT_ANGLE, max_retries=MAX_RETRIES):
        self.retry_angle, self.abort_angle, self.max_retries = retry_angle, abort_angle, max_retries
        self.count = 0

    def allowed(self):
        return self.count <= self.max_retries

    def __call__(self, angle, done):
        why = None
        if self.abort_angle is not None and angle > self.abort_angle:
            why = 'abort_angle'
        elif done and self.retry_angle is not None and angle > self.retry_angle:
            why = 'retry_angle'
        if why:
            self.count += 1
        return why


class Monitor:
    """Per physics substep (getup_probe's mj_step hook): the fastest free joint and the floor's force on the hands
    (Σ of the contact-frame force norms of the hand geoms' floor contacts, G1GetupEnv._hand_force's number), each kept
    as the largest since take()."""

    def __init__(self, raw):
        import mujoco
        from humanoid_sim.tasks import g1_tensor_math as task
        names = [raw.model.body(b).name for b in range(raw.model.nbody)]
        self.hands = raw.robot_geoms & np.array([task.TRACK_HAND_PART in names[b] for b in raw.model.geom_bodyid])
        self.floor = raw.model.geom('floor').id
        self.free = np.asarray(raw.vadr)[~raw.wrist_mask.numpy()]
        # Each hand apart (the left's, the right's geoms by body name): the per-hand load the Dex3-1 bounds.
        self.left = self.hands & np.array(['left' in names[b] for b in raw.model.geom_bodyid])
        self.force = np.zeros(6)
        self.mujoco, self.task = mujoco, task
        self.speed = self.hand = 0.
        self.each = np.zeros(2)

    def after(self, model, data):
        self.speed = max(self.speed, float(np.abs(data.qvel[self.free]).max()))
        if data.ncon:
            selected, robot, _ = self.task.impact_contacts(data.contact.geom, self.floor, self.hands)
            each = np.zeros(2)
            for k in np.flatnonzero(selected):
                self.mujoco.mj_contactForce(model, data, int(k), self.force)
                each[0 if self.left[robot[k]] else 1] += float(np.linalg.norm(self.force[:3]))
            self.hand = max(self.hand, float(each.sum()))
            self.each = np.maximum(self.each, each)

    def take(self):
        out = self.speed, self.hand
        self.speed = self.hand = 0.
        return out

    def take_each(self):
        """(left, right) the largest floor force on each hand since the last call (N)."""
        out = self.each
        self.each = np.zeros(2)
        return out


def _setup(stage, track):
    """(environment, policy, normalizer, clean observation method) of a stage, once per process."""
    if stage not in _cache:
        import torch
        torch.set_num_threads(1)
        from humanoid_sim.tasks.catalog import make_task
        policy, norm, config = getup_probe._policy(stage)
        if bool(config.get('track_clip')) != track or config.get('track_clips') or config.get('settle_clip'):
            raise SystemExit(f'{stage}: нужен этап {"трекера с track_clip" if track else "обычного вставания g1_getup"}')
        env = make_task('g1_getup', 'g1_29dof', {**config, 'max_episode_steps': 100_000})
        _cache[stage] = env, policy, norm, env.unwrapped._observation
    return _cache[stage]


def sensed(raw, exact, rng, bias):
    """raw._observation with chatter.noisy's sensor noise and encoder bias; pd_bound's observed angles are the noisy
    ones (on the robot the ONNX graph bounds around its raw input)."""
    import chatter
    noisy = chatter.noisy(raw, exact, rng, bias)

    def observation(history='replace'):
        obs = noisy(history)
        raw.observed_position = obs[raw.position_columns].astype(np.float64)
        return obs
    return observation


def reading(raw):
    """(gravity (3,), joints (29,)) of the newest observed frame (noise and bias included), what g1_ctrl reads."""
    from humanoid_sim.tasks import g1_tensor_math as task
    frame = raw.frames[0, -1].numpy()
    spans = task.deploy_terms(raw.model.nu, raw.extra_sizes)
    return frame[spans[1][0]:spans[1][1]].copy(), frame[spans[4][0]:spans[4][1]] + raw.home


def errors(raw, gravity, joints, goal_gravity, goal_joints):
    """(angle rad, mean |q − q0| over the free joints) against a frame's pelvis gravity and joints."""
    from humanoid_sim.tasks import g1_tensor_math as task
    free = np.flatnonzero(~raw.wrist_mask.numpy())
    angle, _, deviation = task.settle_errors(gravity[None], joints[None], goal_gravity[None], goal_joints[None], free)
    return float(angle[0, 0]), float(deviation[0, 0])


def true_pose(raw):
    """(gravity, joints) of the simulated state, without noise."""
    from humanoid_sim.tasks import g1_tensor_math as task
    return task.projected_gravity(raw.data.xmat[raw.pelvis].reshape(1, 3, 3))[0], raw.data.qpos[raw.qadr].copy()


def posture(raw):
    """(up: getup_success_pose with both soles, standing: task.getup_standing) of the current state."""
    from humanoid_sim.tasks import g1_tensor_math as task
    head = raw._head()
    height, rotation = raw.data.qpos[2:3], raw.data.xmat[raw.torso].reshape(1, 3, 3)
    soles = raw._sole_height().numpy()
    return (bool(task.getup_success_pose(head, height, rotation, soles)[0]),
            bool(task.getup_standing(head, height, rotation, soles)[0]))


def run(job):
    """One start: A to the hand-over, the PD (pd), B; see the module's docstring."""
    import mujoco
    import torch
    torch.set_num_threads(1)
    from humanoid_sim.g1_tensor_env import ENCODER_BIAS
    from humanoid_sim.tasks import g1_tensor_math as task
    import chatter
    getup_probe._install_monitor()
    env_a, policy_a, norm_a, exact_a = _setup(job['first'], False)
    env_b, policy_b, norm_b, exact_b = _setup(job['second'], True)
    raw_a, raw_b = env_a.unwrapped, env_b.unwrapped
    friction, delay, sole = getup_probe.PHYSICS[job['physics']]
    rng = np.random.default_rng((job['seed'], chatter.NOISE_STREAM))
    bias = (2 * rng.random(raw_a.model.nu) - 1) * ENCODER_BIAS
    raw_a._observation = sensed(raw_a, exact_a, rng, bias)
    raw_b._observation = sensed(raw_b, exact_b, rng, bias)
    options = dict(assist_force=0., start=job['row'], action_delay_steps=delay, motor_friction_scale=friction)
    obs, _ = env_a.reset(seed=job['seed'], options={**options, 'pd_bound': 'min'})
    getup_probe.apply_friction(raw_a, sole)
    if job.get('state') is not None:
        # A from a given state instead of the bank row's (job['state'] = (qpos, qvel); scripts/lab/crowd: a robot
        # that fell walking): A enters as g1_ctrl's GetUp, history filled from it, last action 0.
        import mujoco
        raw_a.data.qpos[:], raw_a.data.qvel[:] = job['state']
        mujoco.mj_forward(raw_a.model, raw_a.data)
        raw_a.previous_action[:] = 0
        obs = raw_a._observation('fill')
    dt = raw_a.dt
    # A tracker trained from hand-over states (task_config track_handover_starts, iteration 83) takes the direct
    # hand-over as its training starts: through its own reset option track_handover, the clock (and the rule's target)
    # at its bank's frame.
    trained = job['handover'] == 'direct' and raw_b.track_handover is not None
    # A pd hand-over goes to the frame a B trained from hand-over states learned to take (its bank's; iteration 90: 21).
    banked = raw_b.track_handover is not None
    frame_target = int(raw_b.handover['header']['frame']) if banked else job.get('target_frame', TARGET_FRAME)
    goal_gravity, goal_joints = raw_b.clip_gravity[frame_target], raw_b.clip_joints[frame_target]
    free = ~raw_a.wrist_mask.numpy()
    # The rule: --rule (job['rule']), for a trained B its bank's (header rule; missing: pelvis).
    which = raw_b.handover['header'].get('rule', 'pelvis') if trained else job.get('rule', 'pelvis')
    goal_torso = task.projected_gravity(raw_b.clip_frames['rot'][frame_target, raw_b.track_anchor][None])[0]
    left = job.get('left', 'handover')
    rule = HandOver(dt, torso=which == 'torso', left=left)
    # GetUpBlend's way back (--retry-angle, iteration 89): None off, the run as before
    retry = None if job.get('retry') is None else Retry(**job['retry'])
    record = dict(physics=job['physics'], start=job['start'], row=job['row'], seed=job['seed'],
                  handover=job['handover'], pd_seconds=job.get('pd_seconds'),
                  pd_shape=sorted(job.get('pd', {}).items()))
    if left != 'handover':
        record['left'] = left
    if retry is not None:
        # per return to A: its time, the transition's step and steps, the observed angle, why, the hand-over before it
        record['retry'] = dict(job['retry'])
        returns = record['returns'] = []
    speeds, hands, phases = [], [], []  # per control step over the whole run: fastest free joint, hand force, A/pd/B
    each = []  # per control step the largest floor force on the left and on the right hand (N)
    up_at, standing, fell = None, False, None
    seams = []  # control steps after which a switch came
    monitor = getup_probe.MONITOR = Monitor(raw_a)
    steps = 0

    def after_step(raw, phase='A'):
        nonlocal up_at, standing
        speed, hand = monitor.take()
        speeds.append(speed)
        hands.append(hand)
        each.append(monitor.take_each())
        phases.append(phase)
        up, standing = posture(raw)
        if up and up_at is None:
            up_at = steps * dt

    try:
        fired = target = None
        applied = [np.zeros(raw_a.model.nu)] * 3  # A's last applied actions, newest first (a trained B's start)
        entry = 0  # the control step A was last entered at (again after a return, --retry-angle)
        if job.get('legs_home') is not None:
            # --legs-home (iteration 91): the PD phase before A's first entry, when the observed hips call for it
            def each_legs(bad):
                nonlocal steps
                steps += 1
                if not bad:
                    after_step(raw_a, 'legs')
            legs = legs_home_phase(env_a, job['legs_home']['hip'], job['legs_home']['seconds'], applied, each_legs)
            record['legs_home'] = dict(job['legs_home'], ran=legs is not None)
            if legs is not None:
                record['legs_home'].update(hips=[round(v, 3) for v in legs['hips']], fell=legs['transition']['fell'],
                                           end_s=round(steps * dt, 3))
                fell, obs = legs['transition']['fell'], legs['obs']
                applied, target = legs['transition']['applied'], legs['transition']['target']
                entry = steps
                if fell is not None:  # the run ends in the phase: no A, no hand-over
                    record.update(rule=None, handover_s=None, faced_s=None, hand_rule=which)
        while fell is None:
            if retry is None or retry.allowed():  # past max_retries returns: no hand-over, A alone
                for _ in range(int(round(job.get('first_seconds', FIRST_SECONDS) / dt))):
                    action = raw_a.applied_action(policy_a.predict(norm_a.normalize_obs(obs), deterministic=True)[0])
                    target = raw_a._target(action)
                    obs, _, terminated, _, info = env_a.step(action)
                    applied = [action.copy(), *applied[:-1]]
                    steps += 1
                    if info.get('bad_state'):
                        fell = 'bad_state'
                        break
                    after_step(raw_a)
                    if terminated:
                        fell = 'speed'
                        break
                    gravity, joints = reading(raw_a)
                    fired = rule(steps, *errors(raw_a, gravity, joints, goal_gravity, goal_joints),
                                 torso_gap(raw_a, gravity, joints, goal_torso) if rule.torso else None)
                    if fired:
                        break
            record.update(rule=fired, handover_s=None if not fired else round(steps * dt, 3),
                          faced_s=None if rule.faced_at is None else round(rule.faced_at * dt, 3), hand_rule=which)
            if not fired:  # no hand-over: A alone to the run's end (ALONE_SECONDS from its last entry)
                while fell is None and steps - entry < round(ALONE_SECONDS / dt):
                    action = raw_a.applied_action(policy_a.predict(norm_a.normalize_obs(obs), deterministic=True)[0])
                    obs, _, terminated, _, info = env_a.step(action)
                    steps += 1
                    if info.get('bad_state'):
                        fell = 'bad_state'
                        break
                    after_step(raw_a)
                    if terminated:
                        fell = 'speed'
                break
            back = None  # a return to A (Retry), set by the transition's stop()
            gravity, joints = true_pose(raw_a)
            angle, deviation = errors(raw_a, gravity, joints, goal_gravity, goal_joints)
            anchor = raw_b.track_bodies[raw_b.track_anchor]
            record['pose'] = dict(angle_deg=round(float(np.degrees(angle)), 1), deviation=round(deviation, 3),
                                  joint_speed=round(float(np.abs(raw_a.data.qvel[raw_a.vadr])[free].max()), 2),
                                  pelvis_speed=round(float(np.linalg.norm(raw_a.data.qvel[:3])), 3),
                                  anchor=round(float(raw_a.data.xpos[anchor, 2]), 3),
                                  torso_gap=round(torso_gap(raw_a, gravity, joints, goal_torso), 3))
            seams.append(steps)
            if job['handover'] == 'pd':
                def each_step(bad):
                    nonlocal steps
                    steps += 1
                    if not bad:
                        after_step(raw_a, 'pd')
                def stop(k, blend):
                    # GetUpBlend's way back on the observed pelvis after the transition's step k (Retry)
                    nonlocal back
                    gravity, joints = reading(raw_a)
                    angle = errors(raw_a, gravity, joints, goal_gravity, goal_joints)[0]
                    why = retry(angle, k == blend)
                    if why:
                        back = dict(at_s=round(steps * dt, 3), pd_step=k, pd_steps=blend,
                                    angle_deg=round(float(np.degrees(angle)), 1), why=why,
                                    handover_s=record['handover_s'], rule=fired)
                    return bool(why)
                start = raw_a.data.qpos[raw_a.qadr].copy() if target is None else target
                transition = pd_transition(env_a, start, goal_joints, job['pd_seconds'], job.get('pd', {}), applied,
                                           each_step, None if retry is None else stop)
                target, fell = transition['target'], transition['fell']
                if job.get('pd_state'):  # the state at the transition's end (tests: getup_handover_bank --pd)
                    record['pd_state'] = dict(qpos=raw_a.data.qpos.tolist(), qvel=raw_a.data.qvel.tolist(),
                                              actions=np.array(transition['applied']).tolist(), fell=fell)
                seams.append(steps)
            if back is not None and fell is None:
                # back to A as g1_ctrl enters GetUp again: pd_bound at its lowest, last action 0, the history filled
                # with this frame, the actions in flight kept; the rule afresh
                returns.append(back)
                raw_a.set_pd_bound('min')
                raw_a.previous_action[:] = 0
                obs = raw_a._observation('fill')
                entry = steps
                rule = HandOver(dt, torso=which == 'torso', left=left)
                fired = None
                continue
            if fell is None:
                last_target = target
                if trained:
                    # B trained from hand-over states: its reset option track_handover with A's state and actions in
                    # flight, as its training starts (the clock at the bank's frame, the hand-over grace).
                    state = dict(qpos=raw_a.data.qpos.copy(), qvel=raw_a.data.qvel.copy(), actions=np.array(applied),
                                 frame=frame_target)
                    obs, _ = env_b.reset(seed=job['seed'], options={
                        **{key: value for key, value in options.items() if key != 'start'}, 'track_handover': state})
                    getup_probe.apply_friction(raw_b, sole)
                    frame = frame_target
                else:
                    # B enters as a g1_ctrl policy state: the bank-start reset's bookkeeping, then A's state.
                    env_b.reset(seed=job['seed'], options=options)
                    getup_probe.apply_friction(raw_b, sole)
                    raw_b.data.qpos[:], raw_b.data.qvel[:] = raw_a.data.qpos, raw_a.data.qvel
                    mujoco.mj_forward(raw_b.model, raw_b.data)
                    raw_b.delay_actions = [a.copy() for a in raw_a.delay_actions]
                    if job['handover'] == 'direct':
                        gravity, joints = reading(raw_a)
                        frame = int(task.track_nearest_frame(
                            gravity[None], joints[None], raw_b.clip_anchor_height[:1], raw_b.clip_gravity,
                            raw_b.clip_joints, raw_b.clip_anchor_height, raw_b.nearest_joints)[0])
                    else:
                        frame = frame_target
                    yaw, offset = task.track_align(raw_b.data.xpos[raw_b.pelvis][None],
                                                   raw_b.data.xmat[raw_b.pelvis].reshape(1, 3, 3),
                                                   raw_b.clip_frames['pos'][frame, 0],
                                                   raw_b.clip_frames['rot'][frame, 0])
                    raw_b.clip_yaw, raw_b.clip_offset = float(yaw[0]), offset[0].copy()
                    raw_b.clip_clock = raw_b.clip_frame = frame
                    raw_b.clip_start, raw_b.clip_reached = 'bank', frame >= raw_b.clip_last
                    raw_b.steps = 0
                    raw_b.previous_action[:] = 0
                    raw_b.head = float(raw_b._head()[0])
                    raw_b.head_window.clear()
                    raw_b.getup_at, raw_b.getup_held, raw_b.hold_steps = None, None, 0
                    rotation = raw_b.data.xmat[raw_b.pelvis].reshape(3, 3)
                    raw_b.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                    obs = raw_b._observation('fill')
                record['trained_start'] = trained
                gravity, joints = true_pose(raw_b)
                record['frame'] = frame
                record['frame_errors'] = [round(float(np.degrees(v)), 1) if k == 0 else round(v, 3)
                                          for k, v in enumerate(errors(raw_b, gravity, joints, raw_b.clip_gravity[frame],
                                                                       raw_b.clip_joints[frame]))]
                monitor = getup_probe.MONITOR = Monitor(raw_b)
                limit = raw_b.clip_end - frame
                if job.get('second_seconds') is not None:
                    limit = min(limit, int(round(job['second_seconds'] / dt)))
                failed = None
                jump = None
                for k in range(limit):
                    action = policy_b.predict(norm_b.normalize_obs(obs), deterministic=True)[0]
                    if jump is None:
                        new = raw_b._target(raw_b.applied_action(action))
                        jump = float(np.abs(new - last_target)[free].max())
                    obs, _, terminated, truncated, info = env_b.step(action)
                    steps += 1
                    if info.get('bad_state'):
                        fell = 'bad_state'
                        break
                    after_step(raw_b, 'B')
                    if info.get('track_failed') and failed is None:
                        failed = round((k + 1) * dt, 2)
                    if terminated and not info.get('track_failed'):
                        fell = 'speed'
                        break
                    if truncated:
                        break
                record.update(target_jump=None if jump is None else round(jump, 3), track_failed_s=failed,
                              clock=int(raw_b.clip_clock), clip_last=int(raw_b.clip_last))
            break
    finally:
        getup_probe.MONITOR = None
    raw = raw_b if record.get('frame') is not None else raw_a
    speeds = np.array(speeds) if speeds else np.zeros(1)
    seam_speed = []
    for at in seams:
        lo, hi = max(at - int(round(SEAM_BEFORE / dt)), 0), at + int(round(SEAM_AFTER / dt))
        seam_speed.append(round(float(speeds[lo:hi].max()), 2) if lo < len(speeds) else None)
    feet = float(np.linalg.norm(raw.data.xpos[raw.feet[0], :2] - raw.data.xpos[raw.feet[1], :2]))
    record.update(seconds=round(steps * dt, 2), fell=fell, up_s=None if up_at is None else round(up_at, 2),
                  standing=bool(standing and fell is None), seam_speed=seam_speed,
                  speed_max=round(float(speeds.max()), 2), hand_max=round(float(max(hands, default=0.)), 1),
                  hand_phase={phase: round(float(max(h for h, p in zip(hands, phases) if p == phase)), 1)
                              for phase in dict.fromkeys(phases)},
                  hand_high_s={phase: round(sum(h > HAND_HIGH for h, p in zip(hands, phases) if p == phase) * dt, 2)
                               for phase in dict.fromkeys(phases)},
                  feet_end=round(feet, 3), head_end=round(float(raw._head()[0]), 3),
                  chained=record.get('frame') is not None,
                  hand_each={phase: [[round(float(v), 1) for v in e] for e, p in zip(each, phases) if p == phase]
                             for phase in dict.fromkeys(phases)})
    return record


def variant_key(r):
    """(handover, pd seconds, pd shape) of a record, its variant."""
    return r['handover'], r['pd_seconds'], tuple(tuple(item) for item in r.get('pd_shape', ()))


def hand_loads(records, phases=('A', 'pd', 'B')):
    """{'left'|'right': (p99, max)} of the per-control-step largest floor force on each hand over the steps of
    `phases` with that hand touching (N), pooled over records; (nan, 0) without a touch."""
    out = {}
    for side, column in (('left', 0), ('right', 1)):
        forces = np.array([e[column] for r in records for p in phases for e in r['hand_each'].get(p, ())])
        forces = forces[forces > 0]
        out[side] = (float(np.percentile(forces, 99)) if len(forces) else float('nan'),
                     float(forces.max()) if len(forces) else 0.)
    return out


def grid(results, kinds, physics):
    """One line per variant: chained-standing counts per class and physics, the per-hand loads (p99 / max over every
    phase, then the largest per phase), the median rise and the seams' fastest joint."""
    print('вариант | спина: цепочкой стоят по физикам | бок | кисти на ПД: лев. p99/max, прав. p99/max, Н | на B: то же | '
          'на A: max лев./прав. | подъём с (медиана) | стык рад/с (медиана / max)')
    for key in dict.fromkeys(variant_key(r) for r in results):
        mine = [r for r in results if variant_key(r) == key]
        cells = []
        for kind in kinds:
            cells.append('/'.join(str(sum(r['standing'] and r['chained'] for r in mine if r['start'] == kind
                                          and r['physics'] == p)) for p in physics))
        def pair(phases):
            loads = hand_loads(mine, phases)
            return (f"{loads['left'][0]:.0f}/{loads['left'][1]:.0f}, {loads['right'][0]:.0f}/{loads['right'][1]:.0f}")
        first = hand_loads(mine, ('A',))
        up = [r['up_s'] for r in mine if r['up_s'] is not None and r['chained']]
        seam = [max(v for v in r['seam_speed'] if v is not None) for r in mine
                if any(v is not None for v in r['seam_speed'])]
        print(' | '.join((name(*key), *cells, pair(('pd',)), pair(('B',)),
                          f"{first['left'][1]:.0f}/{first['right'][1]:.0f}",
                          fmt(np.median(up) if up else None),
                          f'{fmt(np.median(seam) if seam else None, ".1f")} / {fmt(max(seam) if seam else None, ".1f")}')))


def variants(text):
    """[(handover, pd seconds or None, pd shape as a sorted tuple of (key, value))] of --handover's values: direct,
    or pd S [key=value …] with PD_DEFAULTS' keys."""
    out = []
    for value in text:
        if value[0] == 'direct' and len(value) == 1:
            out.append(('direct', None, ()))
        elif value[0] == 'pd' and len(value) >= 2:
            seconds = float(value[1])
            if not .04 <= seconds <= 5.:
                raise SystemExit('--handover pd S: S от 0,04 до 5 с')
            shape = {}
            for item in value[2:]:
                key, _, number = item.partition('=')
                if key not in PD_DEFAULTS or not number:
                    raise SystemExit(f'--handover pd S: ключи {", ".join(PD_DEFAULTS)} вида key=value, а не {item}')
                shape[key] = float(number)
            if not 0 < shape.get('ramp', 1.) <= 1 or not 0 < shape.get('arms', 1.) <= 1 \
                    or shape.get('legs', 0.) < 0 or shape.get('kp', 1.) <= 0 or shape.get('kd', 1.) <= 0 \
                    or shape.get('kp_to', 1.) <= 0 or shape.get('kd_to', 1.) <= 0:
                raise SystemExit('--handover pd S: ramp и arms в (0, 1], legs ≥ 0, kp, kd, kp_to и kd_to > 0')
            if shape.get('legs', 0.) >= seconds * shape.get('ramp', PD_DEFAULTS['ramp']):
                raise SystemExit('--handover pd S: legs меньше ramp·S')
            out.append(('pd', seconds, tuple(sorted(shape.items()))))
        else:
            raise SystemExit(f'--handover: direct или pd S [key=value …], а не {" ".join(value)}')
    return out


def name(handover, seconds, shape=()):
    if handover == 'direct':
        return 'direct'
    return ' '.join([f'pd {seconds:g}', *(f'{key}={value:g}' for key, value in shape)])


def fmt(value, spec='.2f'):
    return '—' if value is None else format(value, spec)


def print_rows(results):
    print('вариант | физика | класс | строка | передача с (правило) | угол° / откл. рад / сустав рад/с / таз м/с / торс м '
          f'| кадр B (угол° / откл.) | встал с | стоит в конце | стык max рад/с | скачок цели рад | кисти max Н (с > '
          f'{HAND_HIGH:.0f} Н) A/ПД/B '
          '| стопы в конце м | обрыв трекера с | часы B')
    for r in results:
        pose = r.get('pose')
        pose = '—' if pose is None else (f'{pose["angle_deg"]:.0f} / {pose["deviation"]:.2f} / {pose["joint_speed"]:.1f}'
                                          f' / {pose["pelvis_speed"]:.2f} / {pose["anchor"]:.2f}')
        frame = '—' if r.get('frame') is None else f'{r["frame"]} ({r["frame_errors"][0]:.0f} / {r["frame_errors"][1]:.2f})'
        handover = '—' if r['rule'] is None else f'{r["handover_s"]:.2f} ({r["rule"]})'
        if r.get('returns'):  # --retry-angle: GetUpBlend's returns to A before the hand-over shown
            handover += '; возвраты в A: ' + ', '.join(f'{b["at_s"]:.2f} с {b["angle_deg"]:.0f}° ({b["why"]})'
                                                       for b in r['returns'])
        print(' | '.join(str(x) for x in (
            name(*variant_key(r)), r['physics'], r['start'], r['row'], handover, pose, frame,
            fmt(r['up_s']), ('да' if r['standing'] else 'нет') + (f' ({r["fell"]})' if r['fell'] else '')
            + ('' if r['chained'] else ' (только A)'),
            ' / '.join(fmt(v, '.1f') for v in r['seam_speed']) or '—', fmt(r.get('target_jump')),
            '/'.join(f'{r["hand_phase"][p]:.0f} ({r["hand_high_s"][p]:.2f})' for p in ('A', 'pd', 'B')
                     if p in r['hand_phase']),
            f'{r["feet_end"]:.2f}', fmt(r.get('track_failed_s')),
            '—' if r.get('clock') is None else f'{r["clock"]}/{r["clip_last"]}')))


def summary(results, kinds, physics):
    print('сводка: передач / встали и стоят в конце (из них цепочкой) / прогонов; медиана передачи и подъёма, с; '
          'стык max рад/с (медиана / max); кисти max Н; стопы в конце, м (медиана)')
    verdicts = {}
    for key in dict.fromkeys(variant_key(r) for r in results):
        for kind in kinds:
            for p in physics:
                mine = [r for r in results if variant_key(r) == key and r['start'] == kind
                        and r['physics'] == p]
                if not mine:
                    continue
                handed = [r for r in mine if r['rule']]
                standing = [r for r in mine if r['standing']]
                chained = [r for r in standing if r['chained']]
                seam = [max(v for v in r['seam_speed'] if v is not None) for r in mine
                        if any(v is not None for v in r['seam_speed'])]
                up = [r['up_s'] for r in mine if r['up_s'] is not None]
                print(f'   {name(*key):<7} {kind:<5} {p:<10} передач {len(handed)}, стоят {len(standing)} '
                      f'(цепочкой {len(chained)}) из {len(mine)}; передача '
                      f'{fmt(np.median([r["handover_s"] for r in handed]) if handed else None)}, подъём '
                      f'{fmt(np.median(up) if up else None)}; стык {fmt(np.median(seam) if seam else None, ".1f")} / '
                      f'{fmt(max(seam) if seam else None, ".1f")}; кисти {max(r["hand_max"] for r in mine):.0f}; стопы '
                      f'{np.median([r["feet_end"] for r in mine]):.2f}'
                      + ('' if not any('legs_home' in r for r in mine) else
                         f'; фаза ног {sum(r["legs_home"]["ran"] for r in mine)}'))
                if kind == 'back' and p in GOAL_PHYSICS:
                    verdicts.setdefault(name(*key), []).append((p, len(chained), len(mine)))
    for key, cells in verdicts.items():
        ok = all(n >= GOAL_STANDING * total / 10 for _, n, total in cells) and len(cells) == len(GOAL_PHYSICS)
        print(f'цель (на спине встали и стоят цепочкой ≥ {GOAL_STANDING} из 10 в {", ".join(GOAL_PHYSICS)}), {key}: '
              + ('да' if ok else 'нет') + ' — ' + ', '.join(f'{p} {n}/{total}' for p, n, total in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument('--first', default=FIRST, help=f'этап A, обычное вставание (по умолчанию {FIRST})')
    parser.add_argument('--second', default=SECOND, help=f'этап B, трекер с track_clip (по умолчанию {SECOND})')
    parser.add_argument('--handover', nargs='+', action='append',
                        help='direct или pd S [ramp=… arms=… legs=… kp=… kd=…] (можно несколько раз; по умолчанию '
                             'оба: direct и pd 1.0)')
    parser.add_argument('--physics', default=','.join(PHYSICS), help='физики через запятую: ' + ', '.join(PHYSICS))
    parser.add_argument('--classes', default=','.join(CLASSES), help='классы отложенных строк банка через запятую')
    parser.add_argument('--rows', type=int, help='первые столько строк каждого класса (по умолчанию все 10)')
    parser.add_argument('--rule', default='pelvis', choices=RULES,
                        help='правило передачи: pelvis (таз и суставы, прежнее) или torso (и торс опущен, как в кадре '
                             'цели); у B, обученного из банка передачи, — правило банка')
    parser.add_argument('--left', default='handover', choices=LEFTS,
                        help='таз ушёл из положения лицом вниз: handover — передать (прежнее), reset — начать '
                             'правило заново (config.yaml handover left)')
    parser.add_argument('--retry-angle', type=float,
                        help='градусы: ПД-переход вернёт в A, если в его конце таз дальше этого от кадра цели '
                             '(config.yaml handover retry_angle; по умолчанию без возвратов)')
    parser.add_argument('--abort-angle', type=float, default=float(np.degrees(ABORT_ANGLE)),
                        help='градусы, с --retry-angle: вернуть в A сразу, как только таз дальше этого (0 — без '
                             f'этой проверки; по умолчанию {np.degrees(ABORT_ANGLE):g})')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES,
                        help=f'с --retry-angle: после стольких возвратов A не передаёт (по умолчанию {MAX_RETRIES})')
    parser.add_argument('--legs-home', type=float,
                        help='рад: если в начале прогона оба тангажа бедра выше этого, сначала ПД-фаза (ноги квинтикой '
                             'в home, пояс и руки держат угол), потом A (config.yaml handover legs_home_hip; по '
                             f'умолчанию без фазы, в сборке {LEGS_HOME_HIP:g})')
    parser.add_argument('--legs-home-seconds', type=float, default=LEGS_HOME_SECONDS,
                        help=f'с --legs-home: длина ПД-фазы, с (legs_home_seconds; по умолчанию {LEGS_HOME_SECONDS:g})')
    parser.add_argument('--first-seconds', type=float, default=FIRST_SECONDS,
                        help=f'с политики A до передачи самое большее (по умолчанию {FIRST_SECONDS})')
    parser.add_argument('--second-seconds', type=float, help='с политики B самое большее (по умолчанию до конца клипа '
                                                             'и удержания)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid-only', action='store_true', help='без построчной таблицы прогонов')
    parser.add_argument('--json', type=Path, help='записи прогонов, JSON')
    args = parser.parse_args()
    from humanoid_sim.tasks import g1_getup_bank as bank_module
    chosen = variants(args.handover or [['direct'], ['pd', '1.0']])
    physics = [p for p in args.physics.split(',') if p]
    if set(physics) - set(getup_probe.PHYSICS):
        raise SystemExit(f'физики: {", ".join(getup_probe.PHYSICS)}')
    kinds = [k for k in args.classes.split(',') if k]
    bank = bank_module.load()
    if set(kinds) - set(bank['classes']):
        raise SystemExit(f'классы банка: {", ".join(bank["classes"])}')
    first, second = (str((SIM / p).resolve() if not Path(p).is_absolute() else Path(p)) for p in (args.first,
                                                                                                    args.second))
    retry = None
    if args.retry_angle is not None:
        if not 0 < args.retry_angle <= 180 or not 0 <= args.abort_angle <= 180 or args.max_retries < 0:
            raise SystemExit('--retry-angle от 0 до 180°, --abort-angle от 0 до 180°, --max-retries ≥ 0')
        retry = dict(retry_angle=float(np.radians(args.retry_angle)),
                     abort_angle=float(np.radians(args.abort_angle)) if args.abort_angle else None,
                     max_retries=args.max_retries)
    legs_home = None
    if args.legs_home is not None:
        if not 0 < args.legs_home < np.pi or not .02 <= args.legs_home_seconds <= 5:
            raise SystemExit('--legs-home от 0 до π рад, --legs-home-seconds от 0,02 до 5 с')
        legs_home = dict(hip=args.legs_home, seconds=args.legs_home_seconds)
    jobs = []
    for kind in kinds:
        rows = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)))[:args.rows]
        for handover, seconds, shape in chosen:
            for p in physics:
                jobs += [dict(first=first, second=second, handover=handover, pd_seconds=seconds, pd=dict(shape),
                              physics=p,
                              start=kind, row=int(row), seed=SEED + k, first_seconds=args.first_seconds, rule=args.rule,
                              second_seconds=args.second_seconds, left=args.left, retry=retry, legs_home=legs_home)
                         for k, row in enumerate(rows)]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(run, jobs, chunksize=1)
    print(f'Цепочка вставания: A {first} → передача → B {second} (кадр цели {TARGET_FRAME}); датчики с шумом и '
          'смещением энкодеров, без подтяжки')
    if not args.grid_only:
        print_rows(results)
    summary(results, kinds, physics)
    grid(results, kinds, physics)
    if args.json:
        args.json.write_text(json.dumps(dict(first=first, second=second, runs=results), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
