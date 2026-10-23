"""Batched G1 task equations (g1_velocity obs-v2); shared by the CPU and CUDA environments.

Device and dtype follow the inputs; no host copies or physics calls.
velocity is MuJoCo object velocity at the pelvis COM in local coordinates:
angular xyz then linear xyz. Free-joint qvel is not a substitute for it.
rotation maps pelvis local coordinates to world coordinates.
"""
import math
import torch
from .config import G1_REWARD_WEIGHTS, backward_share

# Gait clock: each leg is in stance for the first 55% of its 0.8 s cycle, legs half a cycle apart.
# Defaults of the task's `gait` setting; a stance share below 0.5 adds flight phases (running).
GAIT_PERIOD = .8
STANCE_FRACTION = .55
FOOT_OFFSETS = (0., .5)
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
# Ankle-roll body origin height: about 0.034 m when the sole rests on flat ground.
CONTACT_HEIGHT = .05
# task_config sole_contact: contact judged by the lowest point of the sole instead. Each foot touches the floor
# with four 5 mm spheres (heel x −0.05, toe x 0.12, 0.03 below the ankle_roll origin); the lowest point is
# min(sphere centre z − radius): −0.6 mm (soft-contact penetration) standing flat in the home pose, below 0 whenever
# MuJoCo has a sphere-floor contact, whatever the foot's tilt. A foot lifted less than 1 cm still counts as support.
SOLE_CONTACT_HEIGHT = .01
SWING_HEIGHT = .11
# Target sole height of the clearance term (unitree_rl_mjlab velocity_env_cfg foot_clearance target_height 0.10).
# Theirs is measured at the G1 foot sites, which sit on the sole bottom (pos z −0.035 in ankle_roll_link: capsule
# axes at −0.025, radius 0.01), so it is a sole height, as sole_height() here (same −0.035 level in the home pose).
CLEARANCE_HEIGHT = .10
FALL_HEIGHT = .55
FALL_UPRIGHT = .5
# Below this commanded speed the task asks the robot to stand on both feet.
STAND_SPEED = .1
# Hip roll/yaw, waist and arms (29-DoF actuator order): joints that should stay near the default pose
# while walking. Environments pass their own list (legs-only G1 has hip roll/yaw only).
POSTURE_JOINTS = (1, 2, 7, 8, *range(12, 29))
# Hip roll/yaw only (unitree_rl_gym hip_pos): feet under the body without pinning the arms.
HIP_JOINTS = (1, 2, 7, 8)
LEGS = 12  # the first 12 actuated joints are the legs (left, then right), on both G1 models
# Command sampling for training: forward speed only (first curriculum step), some standing.
# Defaults of the task's `commands` setting; a range can be narrowed or raised per experiment.
MIN_FORWARD_SPEED = 0.
MAX_FORWARD_SPEED = 1.5
STAND_PROBABILITY = .2
COMMAND_RESAMPLE_SECONDS = 5.
# Pelvis height while standing in the default pose (unitree_rl_gym G1 base_height_target).
BASE_HEIGHT = .78
# "Faster is better" (Humanoid-v5 style): reward grows with ground speed along the episode's target
# heading (world +x unless the start was turned), not along the pelvis axis, so running in circles
# earns nothing. Clipped so a lunge cannot dominate the return; moving backwards is penalised up to 1 m/s.
FORWARD_SPEED_CAP = 4.


def course_speed(velocity, rotation, heading_target):
    """Horizontal speed (m/s) of the pelvis COM along the target heading.

    velocity is in the pelvis inertial frame; for G1 it coincides with the body frame within 0.03 deg,
    so the body rotation turns it into world coordinates.
    """
    world = torch.bmm(rotation, velocity[:, 3:6].unsqueeze(-1)).squeeze(-1)
    target = heading_target.to(velocity.dtype)
    return world[:, 0] * torch.cos(target) + world[:, 1] * torch.sin(target)
# Training-only perturbation (as in unitree_rl_gym): kick the pelvis horizontally on average
# every 5 s so that standing still is not enough and recovery steps are learned.
# Applied by the CUDA training environment only; CPU evaluation and playback are unperturbed.
PUSH_INTERVAL_SECONDS = 5.
PUSH_VELOCITY = 1.
# Heading hold (legged_gym "heading command"): a zero requested yaw rate becomes a steering command
# back to the episode's initial heading. Observations keep the same layout: the command slot carries it.
HEADING_GAIN = .5
MAX_YAW_COMMAND = 1.
# Upper body above the waist: slight forward lean while walking, upright when standing.
TORSO_BODY = 'torso_link'
TORSO_LEAN = .1  # rad, about 6 degrees


def heading(rotation):
    """Yaw of a body's x axis in the world (rad)."""
    return torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])


def effective_command(command, rotation, heading_target, standing=True):
    """Command the policy sees and is rewarded for when heading hold is on.

    standing=False keeps a stand command exactly zero, as Isaac Lab's UniformVelocityCommand zeroes standing
    envs after its heading control. With standing=True (the original task) a robot told to stand but turned
    more than STAND_SPEED / HEADING_GAIN (~11°) off the episode's heading got a turn-in-place command instead,
    so the reward treated it as moving: the standing terms switched off and the gait asked for steps."""
    error = torch.remainder(heading_target - heading(rotation) + math.pi, 2 * math.pi) - math.pi
    hold = torch.clamp(HEADING_GAIN * error, -MAX_YAW_COMMAND, MAX_YAW_COMMAND)
    steer = command[:, 2] == 0
    if not standing:
        steer = steer & (command[:, :2].abs().sum(dim=1) > 0)
    result = command.clone()
    result[:, 2] = torch.where(steer, hold.to(command.dtype), command[:, 2])
    return result


def torso_lean(torso_rotation, pelvis_rotation):
    """Forward and sideways lean (rad) of the torso's up axis, in the pelvis heading frame."""
    up = torso_rotation[:, :, 2]
    yaw = heading(pelvis_rotation)
    forward = up[:, 0] * torch.cos(yaw) + up[:, 1] * torch.sin(yaw)
    side = up[:, 1] * torch.cos(yaw) - up[:, 0] * torch.sin(yaw)
    return torch.atan2(forward, up[:, 2]), torch.atan2(side, up[:, 2])


def body_velocity(cvel, body_com, subtree_com, inertial_rotation):
    """Match mj_objectVelocity(BODY, local=1), including COM translation.

    BODY uses the inertial frame (ximat), unlike projected gravity, which
    deliberately uses the body frame (xmat).
    """
    angular = cvel[:, :3]
    linear = cvel[:, 3:] + torch.linalg.cross(angular, body_com-subtree_com, dim=1)
    local = inertial_rotation.transpose(1, 2)
    return torch.cat((torch.bmm(local, angular.unsqueeze(-1)).squeeze(-1),
                      torch.bmm(local, linear.unsqueeze(-1)).squeeze(-1)), dim=1)


def gait_phase(steps, dt, period=GAIT_PERIOD):
    return torch.remainder(steps.to(torch.float64) * dt / period, 1.)


def observation(velocity, rotation, command, position, joint_velocity, home, previous_action, phase):
    gravity = -rotation[:, 2, :]
    angle = 2 * math.pi * phase.to(velocity.dtype)
    return torch.cat((velocity, gravity, command, position-home, joint_velocity, previous_action,
                      torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


def pd_target(action, home, scale, joint_low, joint_high, limit=1.):
    """Motor (PD) target of an action: home + scale·clip(action, ±limit), within [joint_low, joint_high]: the joint
    ranges, with task_config knee_floor the knees' lower bound raised to it (target_low). The set point of pd_torque
    on the CUDA task and of the CPU task's controller, and the target the arm_press term compares with."""
    return torch.clamp(home + scale*action.clamp(-limit, limit), min=joint_low, max=joint_high)


# task_config knee_floor (iteration 18): the knee PD target never goes below this angle (rad); knees picked by name,
# on both G1 models (left/right_knee_joint, 3 and 9). Standing, every stage held its knees in the mechanical stop
# (−0.091 rad at the lower limit −0.087, home +0.30), the weight hanging on the stops (8–10 N·m); walking, a knee hit
# the stop 4–5 times a second. The training noise (σ 0.5) keeps it there: the stop cancels the knee's noise. Unitree's
# joint_pos_limits penalty (soft limits at 0.9 of the range: 0.061 for the knee) does not take here, and the rule
# applied at export only made 5 of 64 stack-replica runs fall, so it is trained. g1_ctrl clamps the knee target to
# deploy.yaml's clip [knee_floor, upper limit] exactly so, while its last_action is the ONNX output before that clamp
# (action_manager.h: action() is what process_action got), so the observed previous action stays the policy's.
KNEE_JOINT_PART = 'knee'


def target_low(joint_low, knee_mask, knee_floor=None):
    """Lower bound of each joint's PD target, pd_target's joint_low: the joint range's (joint_low, (joints,)), on the
    knee joints (knee_mask, (joints,) bool) raised to knee_floor (rad; None: off, joint_low itself). So a knee target
    is max(target, knee_floor) for a floor inside the range, whatever the action."""
    if knee_floor is None:
        return joint_low
    return torch.where(knee_mask, joint_low.clamp(min=knee_floor), joint_low)


def pd_torque(action, position, velocity, home, scale, kp, kd,
              joint_low, joint_high, torque_low, torque_high, limit=1.):
    target = pd_target(action, home, scale, joint_low, joint_high, limit)
    return torch.clamp(kp*(target-position)-kd*velocity, min=torque_low, max=torque_high)


def action_lags(delay_steps, frame_skip):
    """task_config action_delay: each motor target reaches the motors delay_steps physics substeps after the policy
    chose it ((n,) long, drawn per episode), as legged_gym's / mjlab's action delay. Returns (n, frame_skip) long:
    for substep j of a control step, how many control steps old the action is whose target the motors follow
    (0: this step's), ceil((k − j) / frame_skip) for j < k, else 0. With k ≤ frame_skip the first k substeps keep
    the previous target and the rest get the new one; at an episode start the older actions are zero (home pose).
    The observed previous action stays the policy output, as g1_ctrl's last_action."""
    substep = torch.arange(frame_skip, device=delay_steps.device)
    return torch.div(delay_steps[:, None] - substep + frame_skip - 1, frame_skip, rounding_mode='floor').clamp(min=0)


def sample_commands(n, uniform, min_speed=MIN_FORWARD_SPEED, max_speed=MAX_FORWARD_SPEED,
                    stand_probability=STAND_PROBABILITY, lateral_speed=0., yaw_rate=0., straight_probability=0.,
                    side_share=0., side_stop=0., previous=None):
    """uniform(n) -> values in [0, 1); returns (n, 3) commands (vx, vy, yaw_rate).

    Sideways and turning commands are drawn only when their ranges are non-zero, so forward-only
    configurations keep their exact random stream. straight_probability: share of walking commands with
    no turn (yaw rate 0; under heading hold they steer to the heading held from that moment, see
    straight_walking), like Isaac Lab's heading-command envs: with turns drawn from ±yaw_rate alone, a
    near-zero yaw command is rare and the policy never learns to walk straight without one.

    side_share (iteration 19): share of walking commands that go purely sideways (pure_sideways: vx 0, no turn,
    |vy| uniform in [STAND_SPEED, lateral_speed] with the drawn sign, so each is a real step), taken from the
    backward ones (vx < 0): a backward command becomes one with probability side_share / backward share, so the
    forward commands keep their share and speeds and the backward ones their speed range. side_stop: when
    previous (the commands being replaced at a resample; None at a reset) is purely sideways, a fresh command that
    is not forward (vx ≤ 0: stand, backward, sideways) becomes "stand" with this probability; forward draws are
    left alone, so the share of forward commands stays exactly the old one. Each setting draws only when non-zero,
    after the old draws, so without them the stream is bit for bit the old one. Why: all 29 falls of iteration 18
    stages in the Unitree bridge copy came 1.5–2.6 s after a stop that followed a sideways step (a stop after
    walking forward never fell), while training drew "sideways → stand" in about 0.7% of the command changes (the
    critic's count) against about 9% for "walk → stand".
    A long run as the CUDA training env draws (4096 worlds, 10 episodes of 1000 steps, resampling every 5 s on
    average: 1.6·10^5 changes; the iteration 18 recipe: vx −0.5…1, vy ±0.5, yaw ±1, stand 0.1, straight 0.5):
    side_share 0.2 with side_stop 1 makes "pure sideways → stand" 6.4% of the changes (side_stop 0: 1.8%; 0.15 with
    1: 5.0%; neither: 0, and "|vx| < 0.1 sideways, no turn → stand" 0.5%), while any walk → stand goes 9.2% → 13.5%;
    shares of time forward 0.60 (unchanged), backward 0.30 → 0.11, sideways 0 → 0.16, stand 0.10 → 0.14
    (tests/test_g1_side_stop.py checks these on a shorter run)."""
    speed, stand = uniform(n), uniform(n)
    command = torch.zeros((n, 3), dtype=speed.dtype, device=speed.device)
    standing = stand < stand_probability
    command[:, 0] = torch.where(standing, 0., min_speed + speed * (max_speed - min_speed))
    if lateral_speed:
        command[:, 1] = torch.where(standing, 0., (2 * uniform(n) - 1) * lateral_speed)
    if yaw_rate:
        command[:, 2] = torch.where(standing, 0., (2 * uniform(n) - 1) * yaw_rate)
        if straight_probability:
            command[:, 2] = torch.where(uniform(n) < straight_probability, 0., command[:, 2])
    if side_share:  # config.g1_commands keeps it within the backward share and lateral_speed above STAND_SPEED
        side = (command[:, 0] < 0) & (uniform(n) < side_share / backward_share(min_speed, max_speed))
        lateral = command[:, 1]
        step = torch.copysign(STAND_SPEED + lateral.abs() * (1 - STAND_SPEED / lateral_speed), lateral)
        command[:, 0] = torch.where(side, 0., command[:, 0])
        command[:, 1] = torch.where(side, step, lateral)
        command[:, 2] = torch.where(side, 0., command[:, 2])
    if side_stop and previous is not None:
        stop = pure_sideways(previous) & (command[:, 0] <= 0) & (uniform(n) < side_stop)
        command = torch.where(stop[:, None], 0., command)
    return command


def pure_sideways(command):
    """Purely sideways commands (vx 0, vy non-zero, no turn): (n,) bool; sample_commands' side_share draws them."""
    return (command[:, 0] == 0) & (command[:, 1] != 0) & (command[:, 2] == 0)


def straight_walking(command):
    """Walking commands with no turn: (n,) bool. Heading hold holds the heading these start at."""
    return (command[:, 2] == 0) & (command[:, :2].abs().sum(dim=1) > 0)


# yaw_drift reward weight: averaging time (s) of the pelvis yaw-rate error. Under heading hold an arc is corrected
# by steering, so on a straight command ω_z − c (c: the steered command) averages to the arc's offset, while the
# exp yaw term is flat near zero (an arc of 0.073 rad/s cost about 0.04 per step) and the error of a single step
# is dominated by the gait's own sway (0.1–0.14 rad/s without noise, 0.3–0.5 with the training noise).
YAW_DRIFT_SECONDS = 1.2


def yaw_drift_update(error, velocity, command, dt, seconds=YAW_DRIFT_SECONDS):
    """One step of the exponential mean e <- e + (dt/seconds)·(ω_z − c − e), (n,). velocity as in
    reward_components (ω_z = velocity[:, 2], the pelvis yaw rate the yaw term tracks), command the rewarded one
    (after heading hold), dt the control step (s). The environments zero e at reset and whenever the command
    changes (resampling, set_command)."""
    return error + (dt / seconds) * (velocity[:, 2] - command[:, 2].to(velocity.dtype) - error)


# self_contact reward weight (unitree_rl_mjlab G1 env_cfgs.py self_collisions, weight −1, threshold 10 N;
# mdp/rewards.py self_collision_cost): a physics substep counts when a contact between two robot geoms (both bodies
# in the pelvis subtree, never the floor) carries a force above SELF_CONTACT_FORCE, the norm of its contact-frame
# force (normal and friction, the "force" of MuJoCo's contact sensor; mj_contactForce, mjw.contact_force). Theirs
# returns the number of such substeps among the 4 of a 20 ms step (history_length = decimation 4); the term here is
# their share of the step's substeps (10 of 2 ms), so −w·share at w equals their −1·count at w = 4. Their sensor
# keeps one contact slot (num_slots 1, reduce "none": whichever matched contact fills it); here any contact counts.
SELF_CONTACT_FORCE = 10.


def self_contact_share(force):
    """(n, substeps) largest robot-robot contact force (N) at each physics substep of a control step (0 without
    such a contact) -> (n,) share of those substeps above SELF_CONTACT_FORCE, for the self_contact term."""
    return (force > SELF_CONTACT_FORCE).to(force.dtype).mean(dim=1)


def contact_step(force):
    """(n, substeps) largest contact force (N) at each physics substep of a control step -> (n,) whether any substep is
    above SELF_CONTACT_FORCE: with foot-foot forces, the steps rollout/foot_contact counts (task_config
    foot_collision; logged, not a reward term)."""
    return (force > SELF_CONTACT_FORCE).any(dim=1)


# Arm joints of the arm_press term (29-DoF actuator order: shoulders, elbows, wrists, indices 15-28); environments
# pick them by joint name, the legs-only G1 has none (the term is then 0).
ARM_JOINT_PARTS = ('shoulder', 'elbow', 'wrist')


def arm_press(target, position, arm_joints):
    """(n,) Σ over the arm joints of (PD target − joint angle)², rad², for the arm_press term. target: (n, joints)
    pd_target of the policy's action this step, within the bounds the PD uses (with action_delay too: the policy's own
    target, not the delayed one the motors follow); position: (n, joints) joint angles after the step. An arm the PD drives into the body keeps
    a large error (the body stops it at the angle while the target stays out at the limit), and the error falls
    smoothly as the press eases; the self_contact share stays 1 until the force drops below 10 N."""
    return (target - position)[:, arm_joints].square().sum(dim=1)


# task_config arm_perturb (domain randomisation, training only): in each world, segments of ARM_PERTURB_SECONDS during
# which one held action replaces the policy's on the arm joints (ARM_JOINT_PARTS), so training sees arm poses the
# policy would not choose. Candidate 2 (exp-e4da6963 @ 5300.9 M) keeps a bit in its wrists: exactly one wrist pitch is
# bent to the box edge (+40°), the left one 0.5 s after every training reset; moved in by 4-8°, the bend jumps to the
# other wrist; both at home, it falls standing or turns 30-70°, and in sim2sim the side came up at random out of
# FixStand (right in 2 of 9 runs, exactly the two drifting the other way). A segment holds the home pose (action 0)
# with probability ARM_PERTURB_HOME, otherwise an action uniform within ±ARM_PERTURB_RADIANS/scale per joint.
ARM_PERTURB_SECONDS = (.5, 2.)
ARM_PERTURB_HOME = .5
ARM_PERTURB_RADIANS = .4


def arm_perturb_lengths(dt):
    """Shortest and longest perturbation segment in control steps (25 and 100 at dt 0.02), both drawn."""
    return round(ARM_PERTURB_SECONDS[0] / dt), round(ARM_PERTURB_SECONDS[1] / dt)


def arm_perturb_onset(share, dt):
    """Probability p, per control step outside a segment, that a segment starts at that step, so that a long run
    spends the share f of its steps under perturbation. Outside a segment every step starts one with probability p
    (the starting step is perturbed, the step after a segment may start the next), so the gaps between segments are
    geometric on {0, 1, ...} with mean (1 − p)/p steps; segments last L̄ = (shortest + longest)/2 steps on average.
    The share is L̄ / (L̄ + (1 − p)/p) = f, hence p = f / (f + (1 − f)·L̄): 0.00177 at f = 0.1 (L̄ = 62.5 steps), a
    1.25 s segment after an 11.25 s gap on average. Episodes start outside a segment, so a 1000-step episode at
    f = 0.1 averages 0.098 (the deficit tends to f(1 − f)·E[L(L+1)]/(2L̄) ≈ 3.2 perturbed steps per episode)."""
    shortest, longest = arm_perturb_lengths(dt)
    mean = (shortest + longest) / 2
    return share / (share + (1 - share) * mean) if share else 0.


def arm_perturb_draw(steps, held, uniform, share, dt, scale, limit=1.):
    """One control step of the arm perturbation process, before the action is applied. steps: (n,) long, control
    steps left in each world's segment, this one included (0: none); held: (n, joints) the segment's action;
    uniform(k) -> k values in [0, 1) (CPU np_random, CUDA generator); scale: (joints,) action scale. Returns the new
    (steps, held): a world outside a segment starts one with probability arm_perturb_onset(share, dt), lasting a
    uniform whole number of steps in arm_perturb_lengths(dt) and holding the home pose (action 0) with probability
    ARM_PERTURB_HOME, else U(±ARM_PERTURB_RADIANS/scale) per joint clipped to the action box ±limit (pd_target then
    clips the target to the joint range: ±0.4 rad stays inside every G1 arm range). A world is perturbed this step
    while steps > 0 (arm_perturb_apply); the environment counts the step off afterwards."""
    n, joints = held.shape
    shortest, longest = arm_perturb_lengths(dt)
    start = (steps == 0) & (uniform(n) < arm_perturb_onset(share, dt))
    length = shortest + (uniform(n) * (longest - shortest + 1)).long().clamp(max=longest - shortest)
    home = uniform(n) < ARM_PERTURB_HOME
    bound = ARM_PERTURB_RADIANS / scale.to(held.dtype)
    random = ((2 * uniform(n * joints).reshape(n, joints).to(held.dtype) - 1) * bound).clamp(-limit, limit)
    action = torch.where(home[:, None], torch.zeros_like(random), random)
    return torch.where(start, length, steps), torch.where(start[:, None], action, held)


def arm_perturb_apply(action, steps, held, arm_mask):
    """The applied action: the policy's (n, joints) with the arm joints (arm_mask, (joints,) bool; none on the
    legs-only G1) replaced by the held action where a segment runs (steps > 0). It replaces the policy's action in
    everything after it: PD targets, the action_delay queue, the reward (action_change, arm_press) and the observed
    previous action, history included; only PPO keeps the policy's own (its log-probability). Otherwise the bit
    the perturbation is to erase would stay readable in the observed previous action."""
    return torch.where((steps > 0)[:, None] & arm_mask, held.to(action.dtype), action)


# task_config lock_wrists: joints (picked by name; left/right_wrist_pitch/yaw_joint, 20, 21, 27, 28 on the 29-DoF G1,
# none on g1_12dof) whose action is always 0, so their motors hold the home pose. They are the only G1 joints whose
# action box lies inside the joint range (±0.70 rad at action_limit 10 and scale 0.07, range ±1.61), so a target at
# the box edge is a pose short of the stop the policy can hold as a memory cell: candidate 2 (exp-e4da6963
# @ 5300.9 M) keeps its mode bit in exactly one wrist pitch bent to the box edge (+10, +40°), and neither
# action_change on the raw action, the posture weight nor arm_perturb removes it (critic's probes, the bit is worth
# ~0.3 per step). Unitree's 23-DoF G1 has no such joints.
LOCKED_WRIST_PARTS = ('wrist_pitch', 'wrist_yaw')


def lock_joints(action, locked):
    """The applied action (n, joints) with the locked joints (locked: (joints,) bool) at 0, the home pose. Applied
    after arm_perturb_apply, so like the perturbation it replaces the policy's action in everything after it: PD
    targets, the action_delay queue, the reward (action_change, arm_press) and the observed previous action, history
    included; only PPO keeps the policy's own (its log-probability). The export puts the same zeros after the
    action clip, so g1_ctrl holds those joints at default_joint_pos."""
    return action.masked_fill(locked, 0.)


# task_config force_push (domain randomisation, training only; iteration 20): pushes as a force on the pelvis, the way
# the sim2sim push patch of unitree_mujoco gives them (scripts/sim2sim/patches/unitree_mujoco-push.patch): the
# horizontal force M·Δv/T at the pelvis COM (M the robot's total mass, 35.112 kg on both G1 models, as in the patch),
# held for T seconds on every physics substep in xfrc_applied, its world direction fixed at the push start. Each world
# starts a push every FORCE_PUSH_INTERVAL seconds (uniform, start to start; the first one that long after a reset) with
# Δv uniform in [0, force_push], the direction uniform relative to the pelvis heading (so uniform in the world too) and
# T uniform in FORCE_PUSH_SECONDS in whole substeps, so the impulse is exactly M·Δv. Why: the velocity pushes of the
# CUDA training (g1_tensor_env.REAL_PUSH, up to ±0.5 m/s added to the whole body) at their largest give 2 steps of
# 7-15 cm and a lean of at most 10°, while the bridge's force on the pelvis drives it to 1.5-2.2 m/s and leans the
# body 15-31°: training never saw those states, and none of 28 stages held 1.25 m/s backwards standing, which
# Unitree v0 holds (critic, iteration 19). Those velocity pushes stay; these come on top.
FORCE_PUSH_BODY = 'pelvis'
FORCE_PUSH_INTERVAL = (2., 4.)
FORCE_PUSH_SECONDS = (.05, .2)


def _whole(uniform, n, shortest, longest):
    """(n,) long, uniform whole numbers in [shortest, longest]."""
    return shortest + (uniform(n) * (longest - shortest + 1)).long().clamp(max=longest - shortest)


def force_push_interval(uniform, n, dt):
    """(n,) long control steps from one push start to the next (and from a reset to the first push): uniform whole
    numbers in FORCE_PUSH_INTERVAL / dt, 100-200 at dt 0.02."""
    return _whole(uniform, n, *(round(seconds / dt) for seconds in FORCE_PUSH_INTERVAL))


def force_push_lengths(substep_dt):
    """Shortest and longest push in physics substeps (25 and 100 at 2 ms), both drawn."""
    return round(FORCE_PUSH_SECONDS[0] / substep_dt), round(FORCE_PUSH_SECONDS[1] / substep_dt)


def force_push_force(velocity, heading, mass, substeps, substep_dt):
    """(n, 3) world force (N) of a push, z 0: velocity (n, 2) its Δv (m/s, forward and left in the pelvis heading frame,
    as the patch's keys), heading (n,) the pelvis yaw (rad) at the push start, mass the robot's (kg), substeps (n,) its
    length T in physics substeps of substep_dt seconds: M·Δv/T turned into the world."""
    cos, sin = torch.cos(heading), torch.sin(heading)
    size = mass / (substeps.to(velocity.dtype) * substep_dt)
    x = size * (velocity[:, 0] * cos - velocity[:, 1] * sin)
    y = size * (velocity[:, 0] * sin + velocity[:, 1] * cos)
    return torch.stack((x, y, torch.zeros_like(x)), dim=1)


def force_push_draw(timer, left, force, heading, uniform, max_dv, mass, dt, substep_dt):
    """One control step of the push process, before its physics. timer: (n,) long, control steps until each world's
    next push starts (0: this step); left: (n,) long, substeps of the running push still to come (0: none); force:
    (n, 3) its world force; heading: (n,) pelvis yaw now; uniform(k) -> k values in [0, 1) (CPU np_random, CUDA
    generator), four draws of n per call. Returns the new (timer, left, force): where timer is 0 a push starts, with
    Δv = max_dv·U, its angle U(0, 2π) from the heading and its length uniform in force_push_lengths(substep_dt), and the
    timer gets a fresh force_push_interval; every timer counts this step off. force_push_substeps then spreads the
    force over the substeps."""
    n = len(timer)
    start = timer == 0
    interval = force_push_interval(uniform, n, dt)
    dv, angle = max_dv * uniform(n), 2 * math.pi * uniform(n)
    length = _whole(uniform, n, *force_push_lengths(substep_dt))
    velocity = torch.stack((dv * torch.cos(angle), dv * torch.sin(angle)), dim=1).to(force.dtype)
    push = force_push_force(velocity, heading.to(force.dtype), mass, length, substep_dt)
    return (torch.where(start, interval, timer) - 1, torch.where(start, length, left),
            torch.where(start[:, None], push, force))


def force_push_substeps(force, left, frame_skip):
    """The pelvis force at each physics substep of this control step, (n, frame_skip, 3): the push's while it lasts
    (left > substep), else 0, so a push may end within a step; and the substeps left after the step, (n,)."""
    on = left[:, None] > torch.arange(frame_skip, device=left.device)
    return force[:, None, :] * on[..., None].to(force.dtype), (left - frame_skip).clamp(min=0)


def motor_torque(torque, joint_velocity, x1, x2, y1, y2, static_friction, viscous_friction, activation=.01):
    """Unitree motor model (unitree_rl_lab UnitreeActuator): the PD torque is clipped to a peak that is
    y1 when torque and velocity point the same way and y2 otherwise, falling linearly to zero between
    speeds x1 and x2; joint friction static·tanh(v/activation) + viscous·v is then subtracted."""
    peak = torch.where(joint_velocity * torque > 0, y1, y2)
    speed = joint_velocity.abs()
    peak = torch.where(speed < x1, peak, (peak - peak / (x2 - x1) * (speed - x1)).clamp(min=0.))
    clipped = torch.maximum(torch.minimum(torque, peak), -peak)
    return clipped - (static_friction * torch.tanh(joint_velocity / activation) + viscous_friction * joint_velocity)


def critic_extras(velocity, height):
    """Exact simulator facts appended for the PPO critic only: base linear velocity (pelvis frame), pelvis height."""
    return torch.cat((velocity[:, 3:6], height[:, None].to(velocity.dtype)), dim=1)


# Unitree zeroes the gait phase input while the command is (nearly) zero, in training and in g1_ctrl on the
# robot (unitree_rl_mjlab: mdp/observations.py phase, deploy/include/isaaclab/envs/mdp/observations/observations.h
# gait_phase, `cmd_norm < 0.1f`); the phase clock itself keeps running.
PHASE_STOP_NORM = .1


def deploy_observation(velocity, rotation, command, position, joint_velocity, home, previous_action, phase,
                       phase_stop=False):
    """Observation exactly as Unitree's G1 controller builds it (deploy.yaml, velocity v0): base angular
    velocity, projected gravity, command, gait phase (sin, cos), joint positions relative to the default
    pose, joint velocities, last action. No base linear velocity: the robot does not measure it.
    phase_stop: the phase reads (0, 0) while the command's norm is below PHASE_STOP_NORM, as on the robot
    (task_config phase_stop; without it the phase ticks even when told to stand, which g1_ctrl never shows)."""
    gravity = -rotation[:, 2, :]
    angle = 2 * math.pi * phase.to(velocity.dtype)
    sin, cos = torch.sin(angle), torch.cos(angle)
    if phase_stop:
        walking = (command.norm(dim=1) >= PHASE_STOP_NORM).to(velocity.dtype)
        sin, cos = sin * walking, cos * walking
    return torch.cat((velocity[:, :3], gravity, command, sin[:, None], cos[:, None],
                      position-home, joint_velocity, previous_action), dim=1)


def deploy_terms(joints):
    """(start, end) of each term in a deploy_observation frame, in deploy.yaml order."""
    sizes = (3, 3, 3, 2, joints, joints, joints)
    starts = [sum(sizes[:k]) for k in range(len(sizes))]
    return tuple((start, start + size) for start, size in zip(starts, sizes))


def history_observation(frames, joints):
    """Observation with history, as Unitree's controller builds it with history_length H per term
    (deploy/include/isaaclab/manager: each term keeps its own buffer, oldest frame first, newest last;
    terms are concatenated in deploy.yaml order). frames: (n, H, frame size), oldest first."""
    return torch.cat([frames[:, :, start:end].reshape(len(frames), -1) for start, end in deploy_terms(joints)],
                     dim=1)


def history_index(joints, history):
    """index[h, i]: position of frame h's (0 = oldest) entry i in history_observation's output."""
    index = []
    for h in range(history):
        index.append([history * start + h * (end - start) + i - start
                      for start, end in deploy_terms(joints) for i in range(start, end)])
    return index


def sole_height(sphere_height, radius):
    """Height (m) of each foot's lowest sole point, (n, 2). sphere_height: (n, 2, k) world z of the centres of
    each foot's k collision spheres; radius: (2, k)."""
    return (sphere_height - radius.to(sphere_height.dtype)).amin(dim=-1)


def gait_targets(phase, command, stance_fraction=STANCE_FRACTION):
    """Expected stance per foot (n, 2) and whether the command asks for motion (n,)."""
    offsets = torch.as_tensor(FOOT_OFFSETS, dtype=phase.dtype, device=phase.device)
    stance = torch.remainder(phase[:, None] + offsets, 1.) < stance_fraction
    moving = (command[:, :2].norm(dim=1) > STAND_SPEED) | (command[:, 2].abs() > STAND_SPEED)
    return stance | ~moving[:, None], moving


def reward_components(velocity, rotation, height, command, torque, action, previous_action,
                      foot_height, phase, position, home, torso_rotation, weights=None, posture_joints=POSTURE_JOINTS,
                      heading_target=None, stance_fraction=STANCE_FRACTION, hip_joints=HIP_JOINTS, foot_speed=None,
                      foot_rotation=None, sole_height=None, sole_contact=None, yaw_drift_error=None,
                      requested_command=None, self_contact=None, arm_press=None, feet_flat_stance=False):
    """foot_speed: (n, 2) horizontal speed of each foot over the control step (m/s), for the slip and clearance
    terms. foot_rotation: (n, 2, 3, 3) world <- body rotations of the FEET bodies, for the feet_flat term.
    sole_height: (n, 2) lowest sole point of each foot (m, see sole_height()), for the clearance term and for
    sole_contact. sole_contact (task_config sole_contact; None: whenever sole_height is given): a foot is in
    contact when its sole is below SOLE_CONTACT_HEIGHT, otherwise when its ankle_roll origin (foot_height) is
    below CONTACT_HEIGHT. The swing term still measures foot_height.
    feet_flat_stance (task_config feet_flat_stance): the feet_flat term counts only the feet in contact (the same
    contact as gait and slip), so a foot in swing may roll heel to toe; False: both feet, every step.
    yaw_drift_error: (n,) the yaw-rate error mean already updated for this step (yaw_drift_update), for the
    yaw_drift term; requested_command: (n, 3) the command before heading hold (whose zero yaw rate it steers), which
    decides what is a straight command, required with yaw_drift_error.
    self_contact: (n,) share of this control step's physics substeps with a robot-robot contact above
    SELF_CONTACT_FORCE (self_contact_share), for the self_contact term; None: the term is zero.
    arm_press: (n,) Σ over the arm joints of (PD target − joint angle)² (arm_press()), for the arm_press term; None:
    the term is zero."""
    weights = G1_REWARD_WEIGHTS if weights is None else weights
    dtype = velocity.dtype
    if yaw_drift_error is not None and requested_command is None:
        raise ValueError('yaw_drift needs the requested command (before heading hold)')
    if heading_target is None:
        heading_target = torch.zeros(len(velocity), dtype=dtype, device=velocity.device)
    if sole_contact is None:
        sole_contact = sole_height is not None
    upright = rotation[:, 2, 2]
    fell = (height < FALL_HEIGHT) | (upright < FALL_UPRIGHT)
    contact = sole_height < SOLE_CONTACT_HEIGHT if sole_contact else foot_height < CONTACT_HEIGHT
    stance, moving = gait_targets(phase, command, stance_fraction)
    swinging = ~contact & moving[:, None]
    lean_forward, lean_side = torso_lean(torso_rotation, rotation)
    lean_target = TORSO_LEAN * moving.to(dtype)
    # sin^2 of each sole's tilt, (n, 2, 2), for feet_flat. feet_flat_stance keeps the feet in contact: Unitree v0 at a
    # command of 2 m/s rolls its swing foot heel to toe (34° on average), feet_flat cost it 0.79 per step, 0.23 of it
    # on the stance foot (by the sole).
    tilt = None if foot_rotation is None else foot_rotation[..., 2, :2].to(dtype).square()
    if tilt is not None and feet_flat_stance:
        tilt = tilt*contact[..., None].to(dtype)
    components = {
        'velocity': weights['velocity']*torch.exp(-(velocity[:, 3:5]-command[:, :2]).square().sum(dim=1)/.25),
        'forward': weights['forward']*course_speed(velocity, rotation, heading_target).clamp(-1., FORWARD_SPEED_CAP),
        # Crab-walking: sideways speed in the pelvis frame (the velocity term covered it only when enabled).
        'sideways': -weights['sideways']*velocity[:, 4].square(),
        # Body turned away from the course while the path stays on it (diagonal "strafe"), in rad^2.
        'heading': -weights['heading']*(torch.remainder(heading(rotation)-heading_target.to(dtype)+math.pi,
                                                        2*math.pi)-math.pi).square(),
        'yaw': weights['yaw']*torch.exp(-(velocity[:, 2]-command[:, 2]).square()/.25),
        # Systematic turn on a straight walking command: linear in |e|, the mean of ω_z − c over about
        # YAW_DRIFT_SECONDS (yaw_drift_update), so it sees the arc's offset that the yaw term above hardly notices.
        'yaw_drift': -weights['yaw_drift']*(torch.zeros_like(height) if yaw_drift_error is None
                                            else yaw_drift_error.to(dtype).abs()
                                            * straight_walking(requested_command).to(dtype)),
        'alive': weights['alive']*(~fell).to(dtype),
        'gait': weights['gait']*(contact == stance).to(dtype).mean(dim=1),
        'swing': -weights['swing']*((foot_height-SWING_HEIGHT).square()*swinging.to(dtype)).sum(dim=1),
        # Swing clearance (unitree_rl_mjlab feet_clearance, weight −1 there): |sole − CLEARANCE_HEIGHT| times the
        # foot's horizontal speed, summed over both feet, while the command asks for motion. Weighted by speed, it
        # acts on a foot travelling (in swing) and not on a planted one; linear in the height, it pulls as hard at
        # 1.5–3 cm as at 0, where under sole_contact (step at 1 cm) only the quadratic swing term pulled the lift.
        'clearance': -weights['clearance']*(torch.zeros_like(height) if sole_height is None or foot_speed is None
                                            else ((sole_height.to(dtype)-CLEARANCE_HEIGHT).abs()
                                                  * foot_speed.to(dtype)).sum(dim=1)*moving.to(dtype)),
        'tilt': -weights['tilt']*(1-upright),
        # legged_gym "orientation": squared projected gravity in the pelvis plane, sin^2 of the tilt.
        'orientation': -weights['orientation']*rotation[:, 2, :2].square().sum(dim=1),
        'height': -weights['height']*(height-BASE_HEIGHT).square(),
        'torso': -weights['torso']*((lean_forward-lean_target).square() + lean_side.square()),
        'base_motion': -weights['base_motion']*(velocity[:, 5].square() + .05*velocity[:, :2].square().sum(dim=1)),
        # Creeping while told to stand: linear in the speed and yaw-rate error, because the exp tracking terms
        # above hardly notice a few cm/s (standing still and creeping 0.3 m in 10 s scored the same).
        'still': -weights['still']*(~moving).to(dtype)*((velocity[:, 3:5]-command[:, :2]).norm(dim=1)
                                                         + (velocity[:, 2]-command[:, 2]).abs()),
        # Legs at the default pose while standing (mjlab's variable_posture with a tight standing std): a robot
        # standing 0.14 rad off it with jittering ankles slid its feet 0.4 m in 10 s without lifting them.
        'stand_pose': -weights['stand_pose']*(~moving).to(dtype)*(position-home)[:, :LEGS].square().sum(dim=1),
        # Feet sliding on the ground (unitree_rl_lab G1 feet_slide, -0.2): standing still the robot shuffled
        # its feet 0.4 m in 10 s without lifting them. Linear, so slow sliding still costs.
        'slip': -weights['slip']*(torch.zeros_like(height) if foot_speed is None
                                  else (foot_speed.to(dtype)*contact.to(dtype)).sum(dim=1)),
        # Feet not flat: sin^2 of each sole's tilt (as 'orientation' for the pelvis), both feet, every step. The
        # ankle_roll frame is level when the sole rests flat (home pose on the floor: 0.0°). A foot on its heel
        # with the toe up > 20° lifts the frame origin above CONTACT_HEIGHT, so the contact terms above (gait,
        # swing, slip) missed the support: the robot shuffled on one heel (toe up 46–52°), soles lifted ≤ 8 mm.
        # Under feet_flat_stance only the feet in contact count; a heel stance is a contact, so it is still charged.
        'feet_flat': -weights['feet_flat']*(torch.zeros_like(height) if tilt is None else tilt.sum(dim=(1, 2))),
        # Body pressed against itself (unitree_rl_mjlab self_collisions): an arm leaning on the torso or the hip
        # holds the robot up on a contact the motors pay for (the candidate stood on 300–425 N of self-contact,
        # its left shoulder at peak torque), invisible to posture (the joint angle stays near home).
        'self_contact': -weights['self_contact']*(torch.zeros_like(height) if self_contact is None
                                                  else self_contact.to(dtype)),
        # Arms pushing against something: squared PD error of the arm joints. The self_contact share is binary
        # (1.00 until the arm torque falls by ~75%), τ² saturates at the motor peak and the angle stays near home
        # while the body holds the arm; this error fell 13.3 → 7.7 → 3.4 → 0.7 rad² as a standing press eased.
        'arm_press': -weights['arm_press']*(torch.zeros_like(height) if arm_press is None
                                            else arm_press.to(dtype)),
        'posture': -weights['posture']*(position-home)[:, posture_joints].square().sum(dim=1),
        'hips': -weights['hips']*(position-home)[:, hip_joints].square().sum(dim=1),
        'effort': -weights['effort']*torque.square().sum(dim=1),
        'action_change': -weights['action_change']*(action-previous_action).square().sum(dim=1),
        'fall': -weights['fall']*fell.to(dtype),
    }
    return components, fell
