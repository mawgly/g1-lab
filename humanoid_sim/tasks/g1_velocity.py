"""G1 command tracking, numerical observations and torque-limited PD control.

Controller values originate from the pinned Unitree deployment configuration.
This is a new training task, not a loader for Unitree's pretrained policies.
Observation, reward and command equations live in g1_tensor_math and are shared
with the CUDA environment, so both backends train the same task.
"""
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
from .config import (ARM_PERTURB_RADIANS, ARM_PERTURB_RADIANS_RANGE, CRITIC_EXTRAS, CROUCH_HEIGHTS, FORCE_PUSH_RANGE,
                     MOTOR_FRICTION_RANGE, g1_commands, g1_gait, g1_reward_weights)
from . import g1_tensor_math as task

ASSETS = Path(__file__).resolve().parents[2] / 'assets' / 'robots' / 'g1'
LEG_JOINTS = tuple(f'{side}_{joint}_joint' for side in ('left', 'right')
                   for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
# The legs-only robot (as unitree_rl_gym's g1_12dof) is derived from the 29-DoF files at load time:
# waist and arm joints are removed, so those links are welded at zero joint angles (arms straight down).
# Part of its checkpoint identity; change it whenever the derivation changes.
LEGS_ONLY_REVISION = 'legs-only-v1: waist and arm joints welded at zero angles'
# foot_collision (task_config): Unitree's training model (unitree_rl_mjlab g1.xml, class foot_capsule) gives each foot 7
# capsules of radius 0.01, colliding with everything, foot against foot included; ours touch the world only with the
# four 5 mm sole spheres and the foot mesh has no collision, so the two feet pass through each other. With the setting
# load_model adds Unitree's capsules (fromto in the ankle_roll frame, which is the same in both models, exactly as in
# g1.xml: the left foot1 ends at y −0.027, the right one at −0.026) that collide only with each other: their contype
# and conaffinity are FOOT_COLLISION_BIT, which no other geom has (the floor and the robot use bit 1), so the floor
# contact stays the sole spheres. Their contact is Unitree's (unitree_rl_mjlab g1_constants.py FULL_COLLISION for
# `(left|right)_foot[1-7]_collision`): condim 3, sliding friction 0.6 (the torsional and rolling ones, which condim 3
# does not use, MuJoCo's defaults, as g1.xml sets none), priority 1. Frictionless (condim 1, the first version) the
# parent stages hardly ever fell (0 of 24 training-like episodes; the feet slid past each other at 400-800 N), with
# friction a graze is a trip, as on the robot: 40-80 % of sideways pushes fall (critic, lab-critic/it24). Group 3
# (hidden in MuJoCo's renderer), no mass (the ankle_roll links have explicit inertia). A foot-foot contact above 10 N is
# charged by the self_contact term (robot geoms on both sides). Part of the checkpoint identity
# (config.FOOT_COLLISION_SUFFIX on the model revision).
FOOT_CAPSULE_RADIUS = .01
FOOT_CAPSULE_CONTACT = dict(condim=3, friction=(.6, .005, .0001), priority=1)
FOOT_CAPSULES = {
    'left_ankle_roll_link': (
        (.1, -.026, -.025, .05, -.027, -.025), (-.044, -.018, -.025, .123, -.018, -.025),
        (-.052, -.01, -.025, .13, -.01, -.025), (-.054, 0., -.025, .132, 0., -.025),
        (-.052, .01, -.025, .13, .01, -.025), (-.044, .018, -.025, .123, .018, -.025),
        (.1, .026, -.025, .05, .026, -.025)),
    'right_ankle_roll_link': (
        (.1, -.026, -.025, .05, -.026, -.025), (-.044, -.018, -.025, .123, -.018, -.025),
        (-.052, -.01, -.025, .13, -.01, -.025), (-.054, 0., -.025, .132, 0., -.025),
        (-.052, .01, -.025, .13, .01, -.025), (-.044, .018, -.025, .123, .018, -.025),
        (.1, .026, -.025, .05, .026, -.025)),
}
FOOT_COLLISION_BIT = 2

# Real-robot motors (docs/g1-real-robot-plan.md). Joint -> Unitree motor as in mjlab g1_constants.py;
# torque-speed curve (x1, x2 rad/s; y1, y2 N·m) and friction (static N·m, viscous N·m·s/rad) measured by
# Unitree (unitree_rl_lab unitree_actuators.py). Ankles and waist roll/pitch are two 5020 motors on a
# linkage (nominal 1:1), so their torques and friction double, as mjlab doubles their armature and gains.
MOTORS = {'7520-14': (22.63, 35.52, 71., 83.3, 1.6, .16), '7520-22': (14.5, 22.7, 111., 131., 2.4, .24),
          '5020': (30.86, 40.13, 24.8, 31.9, .6, .06), '2x5020': (30.86, 40.13, 49.6, 63.8, 1.2, .12),
          '4010': (15.3, 24.76, 4.8, 8.6, .6, .06)}
MOTOR_OF = (('hip_pitch', '7520-14'), ('hip_yaw', '7520-14'), ('waist_yaw', '7520-14'), ('hip_roll', '7520-22'),
            ('knee', '7520-22'), ('ankle', '2x5020'), ('waist_roll', '2x5020'), ('waist_pitch', '2x5020'),
            ('wrist_pitch', '4010'), ('wrist_yaw', '4010'), ('shoulder', '5020'), ('elbow', '5020'),
            ('wrist_roll', '5020'))
# mjlab: stiffness = armature · ω², ω = 2π·10 Hz, so the reflected rotor inertia follows from controller.json.
NATURAL_FREQUENCY = 2 * np.pi * 10.
FOOT_FRICTION = .6  # mjlab foot geoms (priority over the floor)


def motor_parameters(joint_names):
    """Per-joint (x1, x2, y1, y2, static friction, viscous friction) arrays for the real-robot motors."""
    rows = []
    for name in joint_names:
        motor = next((kind for part, kind in MOTOR_OF if part in name), None)
        if motor is None:
            raise ValueError(f'No Unitree motor known for joint {name}')
        rows.append(MOTORS[motor])
    return np.array(rows).T


def floor_contact_geoms(model, bodies):
    """Geoms of `bodies` whose contact filter (contype / conaffinity) passes with the floor's, in geom order: the sole
    spheres of the feet (foot_collision's capsules collide only with each other)."""
    floor = model.geom('floor').id
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] in bodies and (
        model.geom_contype[i] & model.geom_conaffinity[floor] or model.geom_contype[floor] & model.geom_conaffinity[i])]


def apply_real_robot_physics(model, joints, kp, feet):
    """Rotor armature from the PD gains, no ad-hoc joint friction or damping (the motor model has them),
    foot contact friction 0.6 taking priority over the floor."""
    dofs = model.jnt_dofadr[joints]
    model.dof_armature[dofs] = kp / NATURAL_FREQUENCY ** 2
    model.dof_damping[dofs] = 0.
    model.dof_frictionloss[dofs] = 0.
    foot_geoms = floor_contact_geoms(model, feet)
    model.geom_friction[foot_geoms, 0] = FOOT_FRICTION
    model.geom_priority[foot_geoms] = 1
    return foot_geoms


def _batch(value):
    return torch.from_numpy(np.asarray(value, dtype=np.float64))[None]


def load_model(legs_only=False, foot_collision=False):
    """The G1 scene and the indices of its actuators in the 29-DoF controller.json order; foot_collision adds Unitree's
    foot capsules (FOOT_CAPSULES), which collide only foot with foot. Without either, the files as they are."""
    path = str(ASSETS / 'scene_29dof.xml')
    if not legs_only and not foot_collision:
        model = mujoco.MjModel.from_xml_path(path)
        return model, list(range(model.nu))
    spec = mujoco.MjSpec.from_file(path)
    kept = list(range(len(spec.actuators)))
    if legs_only:
        kept = [i for i, actuator in enumerate(spec.actuators) if actuator.target in LEG_JOINTS]
        for sensor in list(spec.sensors):
            if sensor.objtype == mujoco.mjtObj.mjOBJ_JOINT and sensor.objname not in LEG_JOINTS:
                spec.delete(sensor)
        for actuator in list(spec.actuators):
            if actuator.target not in LEG_JOINTS:
                spec.delete(actuator)
        for joint in list(spec.joints):
            if joint.type != mujoco.mjtJoint.mjJNT_FREE and joint.name not in LEG_JOINTS:
                spec.delete(joint)
    if foot_collision:
        for body, capsules in FOOT_CAPSULES.items():
            side = body.split('_')[0]
            for number, fromto in enumerate(capsules, 1):  # Unitree's names: left_foot1_collision ...
                spec.body(body).add_geom(
                    name=f'{side}_foot{number}_collision', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[FOOT_CAPSULE_RADIUS, 0., 0.], fromto=fromto, contype=FOOT_COLLISION_BIT,
                    conaffinity=FOOT_COLLISION_BIT, condim=FOOT_CAPSULE_CONTACT['condim'],
                    friction=list(FOOT_CAPSULE_CONTACT['friction']), priority=FOOT_CAPSULE_CONTACT['priority'],
                    group=3, density=0., rgba=[.2, .2, .2, 1.])
    return spec.compile(), kept


class G1VelocityEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, max_episode_steps=1000, render_mode=None, reward_weights=None, heading_hold=False,
                 commands=None, legs_only=False, gait=None, action_limit=1., real_robot=False, critic_extras=False,
                 training_realism=1., history=1, stand_heading_hold=True, phase_stop=False, sole_contact=False,
                 random_phase=False, action_delay=0., arm_perturb=0., lock_wrists=False, knee_floor=None,
                 force_push=0., feet_flat_stance=False, foot_collision=False, side_heading_hold=True,
                 motor_friction=0., height_command=False, upper_body_slot=False,
                 arm_perturb_radians=ARM_PERTURB_RADIANS, arm_perturb_contact_mask=False):
        self.real_robot = bool(real_robot)
        # height_command: the pelvis height command c (self.height), an observation term after last_action; drawn at
        # training resets and command changes (task.height_draw), BASE_HEIGHT at a reset with a command, set by
        # set_height or options['height'] (set_command keeps it); the reward's height, fall and stand pose follow it.
        # upper_body_slot: a reserved term of zeros after it. Both are terms of the deploy layout (real robot only).
        self.height_command = bool(height_command)
        self.upper_body_slot = bool(upper_body_slot)
        if (self.height_command or self.upper_body_slot) and not self.real_robot:
            raise ValueError('height_command and upper_body_slot require real_robot')
        self.extra_terms = task.extra_terms(self.height_command, self.upper_body_slot)
        self.extra_sizes = tuple(size for _, size in self.extra_terms)
        self.height = task.BASE_HEIGHT
        # Largest multiplier of the motor model's joint friction; a reset without a set command draws the episode's
        # m ~ U(0, motor_friction) (task.motor_friction_draw), as the CUDA task does per world. 0: the original task.
        self.motor_friction = float(motor_friction)
        if not MOTOR_FRICTION_RANGE[0] <= self.motor_friction <= MOTOR_FRICTION_RANGE[1]:
            raise ValueError(f'motor_friction must be in [{MOTOR_FRICTION_RANGE[0]:g}, {MOTOR_FRICTION_RANGE[1]:g}]')
        if self.motor_friction and not self.real_robot:
            raise ValueError('motor_friction requires real_robot')
        # Heading hold also steers a purely sideways command (task.effective_command); False: its turn stays zero.
        self.side_heading_hold = bool(side_heading_hold)
        # The feet collide with each other through Unitree's foot capsules (load_model), as the CUDA task's model.
        self.foot_collision = bool(foot_collision)
        # The feet_flat reward term counts only the feet in contact (task.reward_components), as the CUDA task does.
        self.feet_flat_stance = bool(feet_flat_stance)
        # Largest Δv (m/s) of the force pushes on the pelvis; resets without a set command push every 2-4 s
        # (task.force_push_draw), as the CUDA task does. 0: the original task.
        self.force_push = float(force_push)
        if not FORCE_PUSH_RANGE[0] <= self.force_push <= FORCE_PUSH_RANGE[1]:
            raise ValueError(f'force_push must be in [{FORCE_PUSH_RANGE[0]:g}, {FORCE_PUSH_RANGE[1]:g}] m/s')
        # Lowest knee PD target (rad; None: the joint range), as the CUDA task and g1_ctrl's deploy.yaml clip.
        self.knee_floor = None if knee_floor is None else float(knee_floor)
        # The wrist pitch and yaw joints always get action 0 (task.lock_joints), as the CUDA task does.
        self.lock_wrists = bool(lock_wrists)
        # Share of training time the arm joints follow a held perturbation action instead of the policy's; resets
        # without a set command run the segments (task.arm_perturb_draw), as the CUDA task does. 0: the original task.
        self.arm_perturb = float(arm_perturb)
        if not 0. <= self.arm_perturb <= .5:
            raise ValueError('arm_perturb must be in [0, 0.5]')
        # Amplitude (±rad per arm joint) of a segment's random pose (task.arm_perturb_draw); ARM_PERTURB_RADIANS: the
        # original draw. And whether contacts touching an arm link leave the self_contact share while a segment runs
        # (task.self_contact_pairs; also a set_arm_perturb segment), as the CUDA task does. False: the original term.
        self.arm_perturb_radians = float(arm_perturb_radians)
        low, high = ARM_PERTURB_RADIANS_RANGE
        if not low <= self.arm_perturb_radians <= high:
            raise ValueError(f'arm_perturb_radians must be in [{low:g}, {high:g}] rad')
        self.arm_perturb_contact_mask = bool(arm_perturb_contact_mask)
        # Largest delay (s) from policy output to motor target; a reset without a set command draws the episode's
        # delay in physics substeps (task.action_lags), as the CUDA task does. 0: the original task.
        self.action_delay = float(action_delay)
        if not 0. <= self.action_delay <= .05:
            raise ValueError('action_delay must be in [0, 0.05] s')
        # A reset without a set command starts the gait clock at a random step of its cycle (either foot first).
        self.random_phase = bool(random_phase)
        # Reward contact by the lowest sole point (task.SOLE_CONTACT_HEIGHT) instead of the ankle_roll origin height.
        self.sole_contact = bool(sole_contact)
        # Gait phase reads (0, 0) while told to stand, as Unitree's controller builds it (real robot only).
        self.phase_stop = bool(phase_stop)
        if self.phase_stop and not self.real_robot:
            raise ValueError('phase_stop requires real_robot')
        # Heading hold also while standing (turn in place back to the heading); False: a stand command is zero.
        self.stand_heading_hold = bool(stand_heading_hold)
        # Frames of observation history per term, as Unitree's controller (history_length); real robot only.
        self.history = int(history)
        if self.history < 1 or (self.history > 1 and not self.real_robot):
            raise ValueError('history > 1 requires real_robot')
        self.training_realism = float(training_realism)  # read by the CUDA training env; evaluation is exact
        self.critic_extras = bool(critic_extras)
        if self.critic_extras and not self.real_robot:
            raise ValueError('critic_extras requires real_robot')
        self.reward_weights = g1_reward_weights(reward_weights)
        self.heading_hold = bool(heading_hold)
        self.commands = g1_commands(commands)
        self.gait = g1_gait(gait)
        self.action_limit = float(action_limit)
        self.legs_only = bool(legs_only)
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported render mode')
        self.render_mode = render_mode
        self.model, kept = load_model(self.legs_only, self.foot_collision)
        self.model.opt.timestep = .002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.data = mujoco.MjData(self.model)
        cfg = json.loads((ASSETS / 'controller.json').read_text())
        self.dt = cfg['step_dt']
        self.frame_skip = round(self.dt / self.model.opt.timestep)
        self.max_delay_steps = round(self.action_delay / self.model.opt.timestep)
        # This episode's delay k (substeps), the age in control steps of the action each substep's target comes
        # from (task.action_lags; all 0 without delay) and the older actions still in flight, newest first.
        self.delay_steps = 0
        self.delay_lags = np.zeros(self.frame_skip, dtype=np.int64)
        self.delay_actions = []
        self.kp, self.kd = np.array(cfg['stiffness'])[kept], np.array(cfg['damping'])[kept]
        self.home, self.scale = np.array(cfg['default_joint_pos'])[kept], np.array(cfg['action_scale'])[kept]
        self.joints = self.model.actuator_trnid[:, 0].copy()
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        if self.model.nu != len(self.home):
            raise ValueError('Controller/model actuator mismatch')
        # Joints held near the default pose by the posture term: everything but hip pitch, knees and ankles.
        names = [self.model.joint(j).name for j in self.joints]
        self.posture_joints = tuple(i for i, name in enumerate(names)
                                    if not any(part in name for part in ('hip_pitch', 'knee', 'ankle')))
        self.hip_joints = tuple(i for i, name in enumerate(names) if 'hip_roll' in name or 'hip_yaw' in name)
        # arm_press reward: shoulders, elbows and wrists (15-28 on the 29-DoF G1; none on the legs-only one).
        self.arm_joints = tuple(i for i, name in enumerate(names) if any(part in name for part in task.ARM_JOINT_PARTS))
        # arm_perturb: the joints a perturbation segment holds (the same arms), whether this episode runs segments
        # (training resets only), the control steps left in the current one (0: none) and its held action.
        self.arm_mask = torch.zeros(self.model.nu, dtype=torch.bool)
        self.arm_mask[torch.as_tensor(self.arm_joints, dtype=torch.long)] = True
        self.perturb_live = False
        self.perturb_steps = 0
        self.perturb_action = np.zeros(self.model.nu)
        # lock_wrists: the wrist pitch and yaw joints (20, 21, 27, 28 on the 29-DoF G1). The legs-only G1 has none, so
        # the setting would change the action contract and nothing else: rejected there.
        self.wrist_joints = tuple(i for i, name in enumerate(names)
                                  if any(part in name for part in task.LOCKED_WRIST_PARTS))
        self.wrist_mask = torch.zeros(self.model.nu, dtype=torch.bool)
        self.wrist_mask[torch.as_tensor(self.wrist_joints, dtype=torch.long)] = True
        if self.lock_wrists and not self.wrist_joints:
            raise ValueError('lock_wrists needs the wrist pitch and yaw joints (29-DoF G1)')
        # knee_floor: the knees (3 and 9 on both G1 models) and the PD target's lower bound per joint, the joint range
        # with the knees' raised to the floor (task.target_low; without it the range itself, bit for bit).
        self.knee_joints = tuple(i for i, name in enumerate(names) if task.KNEE_JOINT_PART in name)
        self.knee_mask = torch.zeros(self.model.nu, dtype=torch.bool)
        self.knee_mask[torch.as_tensor(self.knee_joints, dtype=torch.long)] = True
        low, high = self.model.jnt_range[self.joints, 0], self.model.jnt_range[self.joints, 1]
        if self.knee_floor is not None and (not self.knee_joints or not np.isfinite(self.knee_floor) or not all(
                low[k] <= self.knee_floor < high[k] for k in self.knee_joints)):
            raise ValueError('knee_floor must lie within the knee joint range')
        self.target_low = task.target_low(torch.from_numpy(low), self.knee_mask, self.knee_floor).numpy()
        self.target_high = high
        # height_command: per leg its hip pitch, knee and ankle pitch, the joints stand_target bends (both G1 models).
        self.crouch_joints = tuple(tuple(names.index(f'{side}_{part}_joint') for part in task.CROUCH_JOINT_PARTS)
                                   for side in ('left', 'right'))
        self.pelvis = self.model.body('pelvis').id
        self.torso = self.model.body(task.TORSO_BODY).id
        self.feet = [self.model.body(name).id for name in task.FEET]
        # Per foot, its collision spheres (the sole's contact points): (2, k) geom indices, for task.sole_height.
        soles = [floor_contact_geoms(self.model, [foot]) for foot in self.feet]
        if (not soles[0] or len(soles[0]) != len(soles[1])
                or any(self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for foot in soles for g in foot)):
            raise ValueError('G1 feet are expected to touch the floor with the same number of spheres')
        self.soles = np.array(soles)
        self.sole_radius = torch.from_numpy(self.model.geom_size[self.soles, 0].copy())
        # foot_collision: per foot, its capsules ((2, 7) geom indices in FOOT_CAPSULES order; (2, 0) without it).
        self.foot_capsules = np.array([[self.model.geom(f'{name.split("_")[0]}_foot{k}_collision').id
                                        for k in range(1, len(FOOT_CAPSULES[name]) + 1)] if self.foot_collision else []
                                       for name in task.FEET], dtype=np.int64)
        # self_contact reward: per geom, whether it is on the robot (pelvis subtree; the floor is on the world body).
        # foot_collision's capsules are on the ankle_roll links, so a foot-foot contact counts.
        self.robot_geoms = self.model.body_rootid[self.model.geom_bodyid] == self.model.body_rootid[self.pelvis]
        self._contact_force = np.zeros(6)
        # arm_perturb_contact_mask: per geom, whether it is on an arm link, the bodies in the subtrees of the bodies the
        # arm joints move (shoulders, elbows, wrists, hands; none on the legs-only G1, whose arm links are welded to the
        # torso), and whether this control step's contacts with them are left out of self_contact (set by step).
        roots = {int(self.model.jnt_bodyid[self.joints[i]]) for i in self.arm_joints}
        arm_bodies = np.zeros(self.model.nbody, dtype=bool)
        for body in range(1, self.model.nbody):  # a parent always comes before its children
            arm_bodies[body] = body in roots or arm_bodies[self.model.body_parentid[body]]
        self.arm_geoms = arm_bodies[self.model.geom_bodyid]
        self.arm_contacts_masked = False
        # force_push: the pushed body, the robot's total mass (35.112 kg on both G1 models), whether this episode
        # pushes by itself (training resets only), the control steps to its next push, the substeps of the running
        # push still to come (0: none) and its world force (task.force_push_draw / force_push_substeps).
        self.push_body = self.model.body(task.FORCE_PUSH_BODY).id
        self.mass = float(mujoco.mj_getTotalmass(self.model))
        self.push_live = False
        self.push_timer = 0
        self.push_left = 0
        self.push_force = np.zeros(3)
        self.motors = None
        if self.real_robot:
            if self.legs_only:
                raise ValueError('Real-robot profile is defined for the 29-DoF G1')
            self.foot_geoms = apply_real_robot_physics(self.model, self.joints, self.kp, self.feet)
            self.motors = motor_parameters(names)
            self._motor_tensors = tuple(torch.from_numpy(row.copy()) for row in self.motors)
        # motor_friction: this episode's multiplier m of the motor model's joint friction (1: as measured) and the
        # (1,) tensor motor_torque gets (None while m is 1, the original expression).
        self.friction_scale = 1.
        self._friction_scale = None
        self.action_space = spaces.Box(-self.action_limit, self.action_limit, (self.model.nu,), dtype=np.float32)
        # Simulation profile: body angular+linear velocity, gravity, command, joint offsets/velocities,
        # previous action, gait clock. Real robot: the controller's layout without linear velocity, then the extra
        # terms (height_command, upper_body_slot).
        self.frame_size = (11 if self.real_robot else 14) + 3*self.model.nu + sum(self.extra_sizes)
        size = self.frame_size * self.history
        self.actor_inputs = size  # what the robot measures; critic extras follow
        self.frames = None  # (1, history, frame size) torch buffer, oldest first; filled at reset
        size += CRITIC_EXTRAS if self.critic_extras else 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        self.command = np.zeros(3)
        # With heading hold, the observed/rewarded command replaces a zero yaw rate with steering back
        # to heading_target; it is refreshed whenever an observation is built.
        self.observed_command = np.zeros(3)
        self.heading_target = 0.
        # yaw_drift reward: mean pelvis yaw-rate error under the current command (task.yaw_drift_update), (1,);
        # kept only while that weight is non-zero, zeroed whenever the command is set or resampled.
        self.yaw_drift_error = torch.zeros(1, dtype=torch.float64)
        # Commands given by reset options or set_command (evaluation, scenarios) are never resampled.
        self.fixed_command = False
        self.previous_action = np.zeros(self.model.nu)
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.renderer = None

    def set_command(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Expected finite vx, vy, yaw_rate')
        if np.any(np.abs(command) > [2., 1., 2.]):
            raise ValueError('Command exceeds initial walking task limits')
        self.command[:] = command
        self.fixed_command = True
        self.yaw_drift_error = torch.zeros(1, dtype=torch.float64)
        self.perturb_live, self.perturb_steps = False, 0  # evaluation and scenarios: the policy drives the arms
        self.push_live, self.push_left = False, 0  # and nothing pushes the pelvis

    def set_height(self, height):
        """Pelvis height command c (m, within the crouch table's range, CROUCH_HEIGHTS) from the next observation on
        (tests, probes, scenarios; re-read the observation as after set_command), until the next reset or, in a
        training episode, the next command change; as the CUDA task's set_height. set_command keeps it."""
        if not self.height_command:
            raise ValueError('a height command needs task_config height_command')
        low, high = CROUCH_HEIGHTS
        if isinstance(height, bool) or not isinstance(height, (int, float, np.floating, np.integer)) \
                or not np.isfinite(height) or not low <= height <= high:
            raise ValueError(f'height must be a number in [{low:g}, {high:g}] m')
        self.height = float(height)

    def _draw_height(self):
        """A training draw of the height command for the current command (task.height_draw: two draws, always)."""
        uniform = lambda n: torch.from_numpy(self.np_random.random(n))
        crouch = {key: self.commands[key] for key in ('crouch_share', 'crouch_min', 'crouch_speed')}
        self.height = float(task.height_draw(uniform, _batch(self.command.copy()), **crouch)[0])

    @property
    def fixed_normalization(self):
        """(positions, mean, variance) of the observation columns whose normalization is set, not estimated: every
        frame's entries of the extra terms (task.TERM_NORMALIZATION); None without them. warm_start.insert_terms writes
        these into a stage's statistics, TensorNormalize keeps them frozen."""
        if not self.extra_terms:
            return None
        positions, mean, variance = [], [], []
        for name, _ in self.extra_terms:
            columns = task.term_positions(self.model.nu, self.history, self.extra_terms, name)
            positions += columns
            mean += [task.TERM_NORMALIZATION[name][0]] * len(columns)
            variance += [task.TERM_NORMALIZATION[name][1]] * len(columns)
        return np.array(positions, dtype=np.int64), np.array(mean), np.array(variance)

    def set_force_push(self, velocity, seconds=.1):
        """Push the pelvis from the next control step on (tests, probes), as the sim2sim push patch does: the
        horizontal force M·Δv/T for T = `seconds` in whole physics substeps, Δv = `velocity` (forward, left; m/s) in
        the pelvis heading frame now, its direction fixed in the world while it lasts; as the CUDA task's
        set_force_push. Set it after set_command, which ends a running push; force_push's own pushes keep their timer."""
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError('Expected a finite (forward, left) push velocity')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) \
                or round(seconds / self.model.opt.timestep) < 1:
            raise ValueError('A push lasts at least one physics substep')
        substeps = round(seconds / self.model.opt.timestep)
        heading = task.heading(_batch(self.data.xmat[self.pelvis].reshape(3, 3)))
        self.push_left = substeps
        self.push_force = task.force_push_force(_batch(velocity), heading, self.mass, torch.tensor([substeps]),
                                                self.model.opt.timestep)[0].numpy()

    def set_motor_friction(self, scale):
        """Scale the motor model's joint friction (static and viscous) by `scale` from the next physics substep until
        the next reset (tests, probes), as the reset option motor_friction_scale and the CUDA task's
        set_motor_friction; real robot only (the simulation profile has no motor model)."""
        if self.motors is None:
            raise ValueError('motor friction needs the real-robot motor model')
        if isinstance(scale, bool) or not isinstance(scale, (int, float, np.floating, np.integer)) \
                or not np.isfinite(scale) or scale < 0:
            raise ValueError('motor_friction_scale must be a finite non-negative number')
        self.friction_scale = float(scale)
        self._friction_scale = (None if self.friction_scale == 1.
                                else torch.tensor([self.friction_scale], dtype=torch.float64))

    def set_arm_perturb(self, steps, action):
        """Start a perturbation segment now (tests, probes): for the next `steps` control steps the arm joints follow
        `action` (its arm entries, clipped to the action box), as the CUDA task's set_arm_perturb."""
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid perturbation action')
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError('Perturbation steps must be a non-negative integer')
        self.perturb_steps = int(steps)
        self.perturb_action = np.clip(action, -self.action_limit, self.action_limit)

    def _sample_command(self, resample=False):
        """A fresh training command; resample: it replaces the current one mid-episode (side_stop reads that one)."""
        uniform = lambda n: torch.from_numpy(self.np_random.random(n))
        command = task.sample_commands(1, uniform, previous=_batch(self.command.copy()) if resample else None,
                                       **self.commands)
        self.command[:] = command[0].numpy()
        self.yaw_drift_error = torch.zeros(1, dtype=torch.float64)
        if self.commands.get('straight_probability') and bool(task.straight_walking(command)[0]):
            rotation = self.data.xmat[self.pelvis].reshape(3, 3)  # walk straight on from here
            self.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1)
        return velocity

    def _phase(self):
        return task.gait_phase(torch.tensor([self.steps + self.phase_steps]), self.dt, self.gait['period'])

    def _sole_height(self):
        """(1, 2) lowest sole point of each foot, m."""
        return task.sole_height(_batch(self.data.geom_xpos[self.soles, 2]), self.sole_radius)

    def _self_contact_force(self):
        """Largest contact force (N, norm of the contact-frame force) between two robot geoms at this physics step,
        0 without such a contact; for task.self_contact_share. With arm_perturb_contact_mask, in a step whose arms follow
        a perturbation segment (arm_contacts_masked), contacts touching an arm link are left out
        (task.self_contact_pairs)."""
        largest = 0.
        if self.data.ncon:
            masked = (self.arm_geoms, True) if self.arm_contacts_masked else ()
            for k in np.flatnonzero(task.self_contact_pairs(self.data.contact.geom, self.robot_geoms, *masked)):
                mujoco.mj_contactForce(self.model, self.data, int(k), self._contact_force)
                largest = max(largest, float(np.linalg.norm(self._contact_force[:3])))
        return largest

    def _observation(self, history='replace'):
        """history: 'push' adds this frame (after a step), 'fill' fills the buffer with it (reset), 'replace'
        rewrites the newest frame, so re-reading after set_command does not advance the history."""
        rotation = _batch(self.data.xmat[self.pelvis].reshape(3, 3))
        command = _batch(self.command)
        if self.heading_hold:
            command = task.effective_command(command, rotation, _batch(self.heading_target), self.stand_heading_hold,
                                             self.side_heading_hold)
        self.observed_command = command[0].numpy().copy()
        velocity = _batch(self._velocity())
        parts = (velocity, rotation, command, _batch(self.data.qpos[self.qadr]), _batch(self.data.qvel[self.vadr]),
                 _batch(self.home), _batch(self.previous_action), self._phase())
        observed = (task.deploy_observation(*parts, phase_stop=self.phase_stop) if self.real_robot
                    else task.observation(*parts))
        if self.extra_terms:  # height_command [c − 0.78], upper_body_slot zeros
            height = torch.tensor([self.height], dtype=torch.float64) if self.height_command else None
            observed = torch.cat((observed, task.command_terms(height, self.upper_body_slot, observed)), dim=1)
        if self.history > 1:
            if history == 'fill' or self.frames is None:
                self.frames = observed[:, None].repeat(1, self.history, 1)
            elif history == 'push':
                self.frames = torch.cat((self.frames[:, 1:], observed[:, None]), dim=1)
            else:
                self.frames[:, -1] = observed
            observed = task.history_observation(self.frames, self.model.nu, self.extra_sizes)
        if self.critic_extras:
            observed = torch.cat((observed, task.critic_extras(velocity, _batch(self.data.qpos[2]))), dim=1)
        return observed[0].numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # options['phase_steps']: the gait clock starts that many control steps on (evaluation: half a cycle
        # starts on the other foot). Counted apart from self.steps, so the episode keeps its length. Without it,
        # random_phase draws the start for training resets (no command given), as the CUDA task does; resets with
        # a command (evaluation) keep phase 0.
        options = options or {}
        if 'phase_steps' in options:
            phase_steps = options['phase_steps']
        elif self.random_phase and options.get('command') is None:
            phase_steps = int(self.np_random.integers(round(self.gait['period'] / self.dt)))
        else:
            phase_steps = 0
        if isinstance(phase_steps, bool) or not isinstance(phase_steps, (int, np.integer)) or phase_steps < 0:
            raise ValueError('phase_steps must be a non-negative integer')
        self.phase_steps = int(phase_steps)
        # options['action_delay_steps']: the motors get each target that many physics substeps late (probes,
        # evaluation under delay). Without it, action_delay draws k for training resets; resets with a command
        # (evaluation) have none. No draw without action_delay, so the random stream stays the original one.
        if 'action_delay_steps' in options:
            delay_steps = options['action_delay_steps']
        elif self.max_delay_steps and options.get('command') is None:
            delay_steps = int(self.np_random.integers(self.max_delay_steps + 1))
        else:
            delay_steps = 0
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, (int, np.integer)) or delay_steps < 0:
            raise ValueError('action_delay_steps must be a non-negative integer')
        self.delay_steps = int(delay_steps)
        self.delay_lags = task.action_lags(torch.tensor([self.delay_steps]), self.frame_skip)[0].numpy()
        self.delay_actions = [np.zeros(self.model.nu)] * int(self.delay_lags[0])  # zero actions: the home pose
        # arm_perturb: training resets (no command given) run perturbation segments; every reset ends the current one.
        self.perturb_live = bool(self.arm_perturb) and options.get('command') is None
        self.perturb_steps = 0
        self.perturb_action = np.zeros(self.model.nu)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0., 0., .8]
        self.data.qpos[3:7] = [1., 0., 0., 0.]
        self.data.qpos[self.qadr] = self.home + self.np_random.uniform(-.01, .01, self.model.nu)
        self.data.qvel[:] = 0
        self.steps = 0
        self.previous_action[:] = 0
        if (options or {}).get('command') is not None:
            self.set_command(options['command'])
        else:
            self.fixed_command = False
            self._sample_command()
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        self.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        # force_push: training resets (no command given) push the pelvis, the first push 2-4 s on; every reset ends a
        # running one (mj_resetData zeroed xfrc_applied). Drawn last, so the pose and command draws are the old ones,
        # and not at all without force_push.
        self.push_live = bool(self.force_push) and options.get('command') is None
        self.push_left, self.push_force = 0, np.zeros(3)
        self.push_timer = (int(task.force_push_interval(lambda n: torch.from_numpy(self.np_random.random(n)), 1,
                                                        self.dt)[0]) if self.push_live else 0)
        # motor_friction: training resets (no command given) draw the episode's friction multiplier m, after every other
        # draw, so those stay the old ones, and not at all without motor_friction; resets with a command keep m = 1.
        # options['motor_friction_scale'] sets m (probes, evaluation in another friction; real robot only).
        if 'motor_friction_scale' in options:
            scale = options['motor_friction_scale']
        elif self.motor_friction and options.get('command') is None:
            scale = float(task.motor_friction_draw(lambda n: torch.from_numpy(self.np_random.random(n)), 1,
                                                   self.motor_friction)[0])
        else:
            scale = 1.
        if 'motor_friction_scale' in options or self.motors is not None:
            self.set_motor_friction(scale)
        # height_command: training resets (no command given) draw c for the drawn command, after every other draw, so
        # those stay the old ones, and not at all without the setting; a reset with a command starts at BASE_HEIGHT.
        # options['height'] sets c (probes, scenarios).
        self.height = task.BASE_HEIGHT
        if 'height' in options:
            self.set_height(options['height'])
        elif self.height_command and options.get('command') is None:
            self._draw_height()
        return self._observation('fill'), {'x_position': float(self.data.qpos[0])}

    def _target(self, action):
        """PD target of a clipped action, within the joint ranges, the knees at knee_floor or above (task.pd_target
        within target_low / target_high, as the CUDA task's PD)."""
        return task.pd_target(torch.from_numpy(action), torch.from_numpy(self.home), torch.from_numpy(self.scale),
                              torch.from_numpy(self.target_low), torch.from_numpy(self.target_high),
                              self.action_limit).numpy()

    def applied_action(self, action):
        """The action this control step applies for a policy output: clipped to the action box, the arm joints
        following the running perturbation segment's held action (task.arm_perturb_apply; none: untouched), then with
        lock_wrists the wrist pitch and yaw entries at 0, the home pose (task.lock_joints), also in a segment. It
        replaces the policy's in everything after it (PD, delay queue, reward, observed previous action). Changes no
        state: step draws and counts off the segments around it. Probes running their own PD apply what step does
        through it, and bound its target by target_low / target_high: knee_floor acts there (_target), not on the
        action, so the observed previous action keeps a knee action below the floor, as g1_ctrl's last_action."""
        action = np.clip(np.asarray(action, dtype=float), -self.action_limit, self.action_limit)
        if self.perturb_steps:
            action = task.arm_perturb_apply(_batch(action), torch.tensor([self.perturb_steps]),
                                            _batch(self.perturb_action), self.arm_mask)[0].numpy()
        if self.lock_wrists:
            action = task.lock_joints(_batch(action), self.wrist_mask)[0].numpy()
        return action

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid action shape or non-finite action')
        # arm_perturb: a segment may start at this step (the starting step is perturbed) before the action is built.
        if self.perturb_live:
            steps, held = task.arm_perturb_draw(
                torch.tensor([self.perturb_steps]), _batch(self.perturb_action),
                lambda n: torch.from_numpy(self.np_random.random(n)), self.arm_perturb, self.dt,
                torch.from_numpy(self.scale), self.action_limit, self.arm_perturb_radians)
            self.perturb_steps, self.perturb_action = int(steps[0]), held[0].numpy()
        # The applied action (box clip, arm_perturb segment, lock_wrists) replaces the policy's in all below.
        action = self.applied_action(action)
        # arm_perturb_contact_mask: this step's arm contacts leave self_contact while its arms follow a segment.
        self.arm_contacts_masked = self.arm_perturb_contact_mask and self.perturb_steps > 0
        if self.perturb_steps:
            self.perturb_steps -= 1
        target = self._target(action)
        # action_delay: substep j follows the target of the action delay_lags[j] steps old (all 0 without delay).
        targets = [target, *(self._target(past) for past in self.delay_actions)]
        # force_push: a push may start at this step (task.force_push_draw); its force acts on the pelvis at each
        # substep it lasts, written into xfrc_applied before mj_step as the sim2sim push patch does, zeroed after.
        pushes = None
        if self.push_live:
            timer, left, force = task.force_push_draw(
                torch.tensor([self.push_timer]), torch.tensor([self.push_left]), _batch(self.push_force),
                task.heading(_batch(self.data.xmat[self.pelvis].reshape(3, 3))),
                lambda n: torch.from_numpy(self.np_random.random(n)), self.force_push, self.mass, self.dt,
                self.model.opt.timestep)
            self.push_timer, self.push_left, self.push_force = int(timer[0]), int(left[0]), force[0].numpy()
        if self.push_left:
            pushes, left = task.force_push_substeps(_batch(self.push_force), torch.tensor([self.push_left]),
                                                    self.frame_skip)
            pushes, self.push_left = pushes[0].numpy(), int(left[0])
        feet_before = self.data.xpos[self.feet, :2].copy()
        # self_contact: the largest robot-robot contact force of each substep, read after it (its contacts and
        # constraint forces), only while that weight is non-zero.
        self_force = np.zeros(self.frame_skip) if self.reward_weights['self_contact'] else None
        for substep, lag in enumerate(self.delay_lags):
            joint_velocity = self.data.qvel[self.vadr]
            torque = self.kp*(targets[lag]-self.data.qpos[self.qadr])-self.kd*joint_velocity
            if self.motors is not None:  # motor_friction: m only while it is not 1, so the call stays the old one
                friction = {} if self._friction_scale is None else dict(friction_scale=self._friction_scale)
                torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                           *self._motor_tensors, **friction).numpy()
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0],
                                       self.model.actuator_ctrlrange[:, 1])
            if pushes is not None:
                self.data.xfrc_applied[self.push_body, :3] = pushes[substep]
            mujoco.mj_step(self.model, self.data)
            if self_force is not None:
                self_force[substep] = self._self_contact_force()
        if pushes is not None:
            self.data.xfrc_applied[self.push_body, :3] = 0.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Non-finite G1 simulation state')
        self.steps += 1
        velocity = self._velocity()
        yaw_drift = None
        if self.reward_weights['yaw_drift']:  # off: no state, so the reward is the one without the term
            self.yaw_drift_error = task.yaw_drift_update(self.yaw_drift_error, _batch(velocity),
                                                         _batch(self.observed_command), self.dt)
            yaw_drift = self.yaw_drift_error
        # arm_press: the arms' squared error between this step's policy target (also under action_delay, where the
        # motors may still follow an older one) and the angle after the step, only while that weight is non-zero.
        arm_press = None
        if self.reward_weights['arm_press']:
            arm_press = task.arm_press(_batch(target), _batch(self.data.qpos[self.qadr]), self.arm_joints)
        # height_command: the height, fall and stand pose of the height command the policy observed (c changes only
        # at a command change, below, or by set_height between steps).
        height_command = stand_home = None
        if self.height_command:
            height_command = torch.tensor([self.height], dtype=torch.float64)
            stand_home = task.stand_target(_batch(self.home), height_command, self.crouch_joints)
        # Rewarded for the command the policy observed when it chose this action.
        components, fell = task.reward_components(
            _batch(velocity), _batch(self.data.xmat[self.pelvis].reshape(3, 3)), _batch(self.data.qpos[2]),
            _batch(self.observed_command), _batch(self.data.ctrl), _batch(action), _batch(self.previous_action),
            _batch(self.data.xpos[self.feet, 2]), self._phase(), _batch(self.data.qpos[self.qadr]),
            _batch(self.home), _batch(self.data.xmat[self.torso].reshape(3, 3)), weights=self.reward_weights,
            posture_joints=self.posture_joints, heading_target=_batch(self.heading_target),
            stance_fraction=self.gait['stance_fraction'], hip_joints=self.hip_joints,
            foot_speed=_batch(np.linalg.norm(self.data.xpos[self.feet, :2] - feet_before, axis=1) / self.dt),
            foot_rotation=_batch(self.data.xmat[self.feet].reshape(len(self.feet), 3, 3)),
            sole_height=(self._sole_height() if self.sole_contact or self.reward_weights['clearance']
                         else None), sole_contact=self.sole_contact,
            yaw_drift_error=yaw_drift, requested_command=_batch(self.command.copy()),
            self_contact=None if self_force is None else task.self_contact_share(_batch(self_force)),
            arm_press=arm_press, feet_flat_stance=self.feet_flat_stance, height_command=height_command,
            stand_home=stand_home)
        components = {key: float(value[0]) for key, value in components.items()}
        fell = bool(fell[0])
        info = dict(x_position=float(self.data.qpos[0]), fell=fell,
                    command=self.observed_command.copy(), velocity=velocity[3:5].copy(), reward_components=components)
        if self.height_command:
            info['height_command'] = self.height
        self.previous_action[:] = action
        if self.delay_actions:
            self.delay_actions = [action, *self.delay_actions[:-1]]
        # The reward above belongs to the command the action was taken under.
        if not self.fixed_command and self.np_random.random() < self.dt / task.COMMAND_RESAMPLE_SECONDS:
            self._sample_command(resample=True)
            if self.height_command:  # a new height for the new command, after the command's draws
                self._draw_height()
        return self._observation('push'), sum(components.values()), fell, self.steps >= self.max_episode_steps, info

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.pelvis]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -15
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
