"""G1 command tracking, numerical observations and torque-limited PD control.

Controller values originate from the pinned Unitree deployment configuration.
This is a new training task, not a loader for Unitree's pretrained policies.
Observation, reward and command equations live in g1_tensor_math and are shared
with the CUDA environment, so both backends train the same task.
"""
import json
from functools import lru_cache
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
from .config import (ARM_PERTURB_RADIANS, ARM_PERTURB_RADIANS_RANGE, CRITIC_EXTRAS, CROUCH_HEIGHTS, FORCE_PUSH_RANGE,
                     MOTOR_FRICTION_RANGE, TERRAIN_GENERATOR, TERRAIN_LEVELS, TERRAIN_SEED, g1_commands, g1_gait,
                     g1_reward_weights, height_scan_setting, load_curve_setting, terrain_generator,
                     terrain_pelvis_support, terrain_seed, terrain_setting)
from . import g1_tensor_math as task
from . import g1_terrain

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
# hand_collision (task_config; iteration 33, getting up): our hands do not touch anything, the rubber_hand mesh has no
# collision (contype 0), so a robot leaning on its hands would rest on the wrist edge 13 cm short of the palm (critic,
# lab-critic/it33/hand_geom.py). With the setting load_model adds Unitree's hand capsule to each wrist_yaw_link exactly as
# unitree_rl_mjlab g1.xml has it (`left/right_hand_collision`, class collision: radius 0.035, fromto in the wrist_yaw
# frame, which is the same in both models, the right one mirrored in y), colliding with the floor and the robot as every
# Unitree collision geom (contype = conaffinity = 1). Their contact is Unitree's (g1_constants.py FULL_COLLISION for
# `.*_collision`): condim 1, priority 0 (CollisionCfg's default; only the foot capsules get 1), MuJoCo's default
# friction (g1.xml sets none); against our floor (condim 3, friction 1) and our collision meshes (condim 3) MuJoCo then
# takes the larger condim, 3, and friction 1. Unitree's head_collision sphere is not added: our head_link mesh already
# collides. Group 3 (hidden in MuJoCo's renderer and in the browser, webapp.shown_geoms), no mass (the wrist_yaw links
# have explicit inertia). The capsules are robot and arm geoms, so self_contact charges a hand pressing on the body.
# Part of the checkpoint identity (config.HAND_COLLISION_SUFFIX on the model revision); the 29-DoF G1 only (the
# legs-only one welds its arms to the torso).
HAND_CAPSULE_RADIUS = .035
HAND_CAPSULE_CONTACT = dict(condim=1, priority=0)
HAND_CAPSULES = {'left_wrist_yaw_link': (.07, 0., 0., .15, -.02, 0.), 'right_wrist_yaw_link': (.07, 0., 0., .15, .02, 0.)}
# task_config arm_targets: the pose libraries of the arm targets (scripts/lab/arm_pose_library.py): absolute angles of
# the ten free arm joints (names in `joints`), `poses` uniform over their ranges and `carry` the symmetric carry poses of
# task.ARM_CARRY (its ranges in `carry_ranges`), each kept when no arm geom touches the body, the other arm or its own
# arm, in the pose and halfway from home. Not under assets/robots/g1, so it is not part of the model revision.
ARM_POSES = Path(__file__).resolve().parent / 'data' / 'g1_arm_poses.npz'


@lru_cache(maxsize=4)
def arm_pose_library(path=ARM_POSES):
    """(joint names, free poses (L, 10), carry poses (C, 10), carry ranges (len(ARM_CARRY), 2)) of the arm pose
    library, float64; read once, not to be changed in place."""
    with np.load(path) as data:
        return (tuple(str(name) for name in data['joints']), data['poses'].astype(np.float64),
                data['carry'].astype(np.float64), data['carry_ranges'].astype(np.float64))

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


def load_model(legs_only=False, foot_collision=False, hand_collision=False, terrain=False, terrain_seed=TERRAIN_SEED,
               terrain_generator=TERRAIN_GENERATOR):
    """The G1 scene and the indices of its actuators in the 29-DoF controller.json order; foot_collision adds Unitree's
    foot capsules (FOOT_CAPSULES), which collide only foot with foot, hand_collision Unitree's hand capsules
    (HAND_CAPSULES; 29-DoF only), which collide with the floor and the robot, terrain the rough-ground heightfield
    beside the plane (g1_terrain), its map g1_terrain.terrain_grid(terrain_seed, terrain_generator). Without any, the
    files as they are."""
    path = str(ASSETS / 'scene_29dof.xml')
    if legs_only and hand_collision:
        raise ValueError('hand_collision needs the arms of the 29-DoF G1')
    if not legs_only and not foot_collision and not hand_collision and not terrain:
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
    if hand_collision:
        for body, fromto in HAND_CAPSULES.items():  # Unitree's names: left_hand_collision, right_hand_collision
            spec.body(body).add_geom(
                name=f'{body.split("_")[0]}_hand_collision', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[HAND_CAPSULE_RADIUS, 0., 0.], fromto=fromto, contype=1, conaffinity=1,
                condim=HAND_CAPSULE_CONTACT['condim'], priority=HAND_CAPSULE_CONTACT['priority'], group=3, density=0.,
                rgba=[.2, .6, .2, .2])
    if terrain:
        g1_terrain.add_terrain(spec, terrain_seed, terrain_generator)
        model = spec.compile()
        g1_terrain.write_terrain(model, terrain_seed, terrain_generator)
        return model, kept
    return spec.compile(), kept


class G1VelocityEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, max_episode_steps=1000, render_mode=None, reward_weights=None, heading_hold=False,
                 commands=None, legs_only=False, gait=None, action_limit=1., real_robot=False, critic_extras=False,
                 training_realism=1., history=1, stand_heading_hold=True, phase_stop=False, sole_contact=False,
                 random_phase=False, action_delay=0., arm_perturb=0., lock_wrists=False, knee_floor=None,
                 force_push=0., feet_flat_stance=False, foot_collision=False, side_heading_hold=True,
                 motor_friction=0., height_command=False, upper_body_slot=False,
                 arm_perturb_radians=ARM_PERTURB_RADIANS, arm_perturb_contact_mask=False, arm_blind=False,
                 hand_collision=False, arm_targets=False, load_curve=None, terrain=None, height_scan=None):
        self.real_robot = bool(real_robot)
        # height_scan: the ground's heights for the actor (a grid ahead, noised in training) and the critic (a grid
        # around the pelvis, exact), observation terms after the actor's and after the critic extras
        # (config.height_scan_setting, task.height_scan / critic_scan); needs terrain, the deploy layout and the critic
        # extras. None: the old layout, bit for bit.
        self.height_scan = height_scan_setting(height_scan)
        if self.height_scan is not None and (terrain is None or not real_robot or not critic_extras):
            raise ValueError('height_scan requires terrain, real_robot and critic_extras')
        # terrain: the rough-ground heightfield beside the plane (g1_terrain), its curriculum settings
        # (config.terrain_setting) and its map's seed (config.terrain_seed, g1_terrain.terrain_grid; the CUDA task uses
        # this model); None: the plane alone, bit for bit.
        self.terrain = terrain_setting(terrain)
        # load_curve: the full-GPU trainer's block rule for the hand load ceiling (TensorG1Env.rollout_metrics); kept
        # here for it, nothing on the CPU changes (this task keeps the ceiling it is given).
        self.load_curve = load_curve_setting(load_curve)
        if self.load_curve is not None and not arm_targets:
            raise ValueError('load_curve requires arm_targets')
        # arm_targets: training episodes draw an arm mode (task.ARM_MODES); with flag 1 the slot carries the arm targets,
        # the arm action is a residual on them and the hands carry loads. Needs the slot, the locked wrists and the
        # arms seen; real robot (the deploy layout), 29-DoF (checked below).
        self.arm_targets = bool(arm_targets)
        if self.arm_targets and (not real_robot or not upper_body_slot or not lock_wrists or arm_blind):
            raise ValueError('arm_targets requires real_robot, upper_body_slot and lock_wrists, not arm_blind')
        # arm_blind: the policy's normalization holds the arm columns constant (fixed_normalization, arm_columns); the
        # observation itself is unchanged. The deploy layout's columns, so real robot only (the 29-DoF G1).
        self.arm_blind = bool(arm_blind)
        if self.arm_blind and not self.real_robot:
            raise ValueError('arm_blind requires real_robot')
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
        # The hands collide with the floor and the robot through Unitree's hand capsules (load_model; 29-DoF only).
        self.hand_collision = bool(hand_collision)
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
        self.model, kept = load_model(self.legs_only, self.foot_collision, self.hand_collision,
                                      self.terrain is not None, terrain_seed(self.terrain),
                                      terrain_generator(self.terrain))
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
        # hand_collision: the hand capsules, left then right ((2,) geom indices; (0,) without it).
        self.hand_capsules = np.array([self.model.geom(f'{name.split("_")[0]}_hand_collision').id
                                       for name in HAND_CAPSULES] if self.hand_collision else [], dtype=np.int64)
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
        # Hand loads (set_hand_load; arm_targets' flag-1 episodes): the hand bodies (wrist_yaw links, 29-DoF), their own
        # mass and COM, every body's own subtree mass and which hand lies in its subtree (task.hand_subtree_mass), the
        # loads now (kg, left and right; 0: the model's own numbers, untouched).
        self.hand_bodies = None
        self.hand_load = np.zeros(2)
        if not self.legs_only:
            self.hand_bodies = np.array([self.model.body(name).id for name in task.HAND_LOAD_BODIES])
            self.hand_mass = self.model.body_mass[self.hand_bodies].copy()
            self.hand_ipos = self.model.body_ipos[self.hand_bodies].copy()
            self.subtree_mass = self.model.body_subtreemass.copy()
            self.hand_above = np.zeros((self.model.nbody, 2))
            for side, body in enumerate(self.hand_bodies):
                while body >= 0:
                    self.hand_above[body, side] = 1.
                    body = self.model.body_parentid[body] if body else -1
        self._setup_arm_targets(names)
        self._setup_terrain()
        self.action_space = spaces.Box(-self.action_limit, self.action_limit, (self.model.nu,), dtype=np.float32)
        # Simulation profile: body angular+linear velocity, gravity, command, joint offsets/velocities,
        # previous action, gait clock. Real robot: the controller's layout without linear velocity, then the extra
        # terms (height_command, upper_body_slot).
        self.frame_size = (11 if self.real_robot else 14) + 3*self.model.nu + sum(self.extra_sizes)
        size = self.frame_size * self.history
        # height_scan: the actor's grid (scan_grid, its points after the history) and the critic's (critic_grid, after
        # the critic extras), (k, 2) offsets in the pelvis yaw frame; (0, 2) without it. Whether this episode's scan
        # is noised (a training reset; set_command ends it).
        self.scan_grid = self.critic_grid = torch.zeros((0, 2), dtype=torch.float64)
        if self.height_scan is not None:
            self.scan_grid = task.scan_grid(self.height_scan['x'], self.height_scan['y'], self.height_scan['step'])
            self.critic_grid = task.critic_scan_grid()
        self.scan_noised = False
        size += len(self.scan_grid)
        self.actor_inputs = size  # what the robot measures (and the height scan); critic extras follow
        self.frames = None  # (1, history, frame size) torch buffer, oldest first; filled at reset
        size += CRITIC_EXTRAS if self.critic_extras else 0
        size += len(self.critic_grid)
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

    def _setup_arm_targets(self, names):
        """task_config arm_targets: the joints it tracks (shoulders, elbows, wrist roll: the arm joints but the locked
        wrists) and the posture joints without the arms (flag 1), the pose library as offsets from home, the carry
        poses' joints, parts and mirror signs (symmetry.model_joint_mirror), and the episode state: the mode, the flag
        (the arms follow the slot), whether the targets were set by hand (set_arm_targets: the generator stops), the
        segment (start, end, clock, move, hold: task.arm_target_advance), this step's offsets d, the offsets of the step
        before (the observed last action) and those the delay queue holds, the episode's load shares and the load
        ceiling m_max (kg; the full-GPU trainer moves it, task.load_curve_update; here it is restored from a stage or
        stays), and the arm process's own random stream. Without the setting only the empty state."""
        self.track_joints = tuple(i for i in self.arm_joints if i not in self.wrist_joints)
        self.arm_posture_joints = tuple(i for i in self.posture_joints if i not in self.arm_joints)
        self.arm_mode, self.arm_flag, self.arm_fixed = task.ARM_FREE, False, False
        self.arm_offset = np.zeros(self.model.nu)
        self.previous_flag, self.previous_offset = False, np.zeros(self.model.nu)
        self.delay_offsets = []
        self.perturb_training = False
        self.load_share = np.zeros(2)
        self.load_max = 0.
        self.arm_random = None
        if not self.arm_targets:
            return
        if len(self.track_joints) != 10 or self.hand_bodies is None:
            raise ValueError('arm_targets needs the arms of the 29-DoF G1')
        joint_names, poses, carry, ranges = arm_pose_library()
        columns = [names.index(name) for name in joint_names]
        if sorted(columns) != list(self.track_joints):
            raise ValueError('The arm pose library is not over the free arm joints')
        if not np.array_equal(ranges, [row[1:] for row in task.ARM_CARRY]):
            raise ValueError('The carry poses of the arm pose library are not those of task.ARM_CARRY: make it again')
        self.arm_poses, self.arm_carry = (torch.from_numpy(self._arm_offsets(columns, rows)) for rows in (poses, carry))
        self.arm_state = self._arm_still(np.zeros(self.model.nu))

    def _setup_terrain(self):
        """task_config terrain: the ground's heights as MuJoCo's heightfield has them (g1_terrain.model_heights, for
        task.terrain_height), each tile's centre and spawn height (its platform's), the generator's tile types
        (terrain_names) and their shares, and the episode state: its tile (type index into terrain_names and level;
        None: the plane at the origin), the level a training reset takes (terrain_level: 0 fresh, a stage's
        restore_training_state, set_terrain_level; the CPU task runs no curriculum), the path commanded and walked along
        the commands this episode (task.terrain_progress) and the terrain's own random stream (the tile type of a
        training reset). Without the setting only the empty state."""
        self.terrain_type, self.terrain_level, self.tile_level = None, 0, 0
        self.terrain_path = np.zeros(2)
        self.terrain_random = None
        if self.terrain is None:
            return
        spec = g1_terrain.generator(terrain_generator(self.terrain))
        self.terrain_names, self.terrain_shares = spec.types, spec.shares
        heights, self.terrain_corner, self.terrain_cell, self.terrain_lift = g1_terrain.model_heights(self.model)
        self.terrain_heights = torch.from_numpy(heights)
        self.tile_centres = g1_terrain.tile_centres(spec.number)
        self.tile_spawn = self._ground(self.tile_centres.reshape(-1, 2)).reshape(self.tile_centres.shape[:2])

    def _ground(self, xy):
        """Ground height (m) under world points xy (..., 2) (numpy or torch), a float64 tensor (task.terrain_height)."""
        return task.terrain_height(torch.as_tensor(np.asarray(xy, dtype=np.float64)), self.terrain_heights,
                                   self.terrain_corner, self.terrain_cell, self.terrain_lift)

    def set_terrain_level(self, level):
        """The level later training resets put the robot on (task_config terrain; probes, tests): a whole number in
        [0, TERRAIN_LEVELS)."""
        if self.terrain is None:
            raise ValueError('a terrain level needs task_config terrain')
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or not 0 <= level < TERRAIN_LEVELS:
            raise ValueError(f'terrain level must be a whole number in [0, {TERRAIN_LEVELS})')
        self.terrain_level = int(level)

    def _terrain_reset(self, options, seed):
        """task_config terrain, at a reset: the tile and the start on it. A training reset (no command given) takes a
        type drawn by its generator's shares from the terrain's own stream and terrain_level; options terrain_type
        (a name) and terrain_level put any reset on that tile (the other one as a training reset's); a reset with a
        command and neither starts on the plane at the origin, as without the setting. On a tile the robot starts at
        its centre, as high above the platform as above the plane."""
        if seed is not None or self.terrain_random is None:
            self.terrain_random = np.random.default_rng(None if seed is None else (seed, task.TERRAIN_STREAM))
        kind, level = options.get('terrain_type'), options.get('terrain_level')
        self.terrain_path = np.zeros(2)
        if kind is None and level is None and options.get('command') is not None:
            self.terrain_type = None
            return
        if kind is None:
            self.terrain_type = int(task.terrain_type_draw(lambda n: torch.from_numpy(self.terrain_random.random(n)), 1,
                                                           self.terrain_shares)[0])
        elif kind in self.terrain_names:
            self.terrain_type = self.terrain_names.index(kind)
        else:
            raise ValueError('terrain_type must be one of ' + ', '.join(self.terrain_names))
        if level is None:
            level = self.terrain_level
        if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or not 0 <= level < TERRAIN_LEVELS:
            raise ValueError(f'terrain_level must be a whole number in [0, {TERRAIN_LEVELS})')
        self.tile_level = int(level)
        self.data.qpos[:2] = self.tile_centres[self.tile_level, self.terrain_type]
        self.data.qpos[2] += float(self.tile_spawn[self.tile_level, self.terrain_type])

    def _cap_command(self):
        """terrain: a training command on a tile of slow_level or above at most slow_speed
        (task.terrain_command_cap)."""
        if self.terrain is not None and self.terrain_type is not None and not self.fixed_command:
            capped = task.terrain_command_cap(_batch(self.command.copy()), torch.tensor([self.tile_level]),
                                              self.terrain['slow_level'], self.terrain['slow_speed'])
            self.command[:] = capped[0].numpy()

    def _ground_terms(self):
        """task_config terrain: the reward's heights above the ground under the body and the feet's ground normals,
        reward_components' height (1,), foot_height (1, 2), sole spheres' height above the ground (1, 2, k) and
        floor_normal (1, 2, 3)."""
        feet = self.data.xpos[self.feet]
        spheres = self.data.geom_xpos[self.soles]
        return dict(height=_batch(self.pelvis_height()),
                    foot_height=_batch(feet[:, 2]) - self._ground(feet[:, :2])[None],
                    spheres=_batch(spheres[..., 2]) - self._ground(spheres[..., :2])[None],
                    floor_normal=task.terrain_normal(torch.from_numpy(feet[None, :, :2].copy()), self.terrain_heights,
                                                     self.terrain_corner, self.terrain_cell))

    def pelvis_height(self):
        """The pelvis height (m) the reward, the fall and the critic see: with task_config terrain above the ground
        under it, or with its pelvis_support above the mean ground under the feet (task.pelvis_ground), else its world
        height."""
        if self.terrain is None:
            return float(self.data.qpos[2])
        ground = task.pelvis_ground(torch.as_tensor(np.asarray(self.data.qpos[:2], dtype=np.float64)),
                                    torch.as_tensor(np.asarray(self.data.xpos[self.feet, :2], dtype=np.float64)),
                                    self.terrain_heights, self.terrain_corner, self.terrain_cell, self.terrain_lift,
                                    terrain_pelvis_support(self.terrain))
        return float(self.data.qpos[2] - float(ground))

    def _arm_offsets(self, columns, poses):
        """(L, nu) offsets from home of library poses (L, len(columns)) of the joints `columns`; 0 elsewhere."""
        offsets = np.zeros((len(poses), self.model.nu))
        offsets[:, columns] = poses - self.home[columns]
        return offsets

    def _arm_still(self, offsets):
        """A segment holding offsets (nu,) from its first step: (start, end, clock, move, hold) of batch 1."""
        held = _batch(offsets)
        return held, held.clone(), torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long), \
            torch.zeros(1, dtype=torch.long)

    def _arm_uniform(self, n):
        return torch.from_numpy(self.arm_random.random(n))

    def _arm_draws(self):
        """The arm process's draws for task functions: (uniform, free poses, carry poses, joints, dt)."""
        return self._arm_uniform, self.arm_poses, self.arm_carry, self.track_joints, self.dt

    def _arm_episode(self, training):
        """This episode's arm mode, targets and hand loads (arm_targets): a training reset draws the mode
        (task.arm_mode_draw), with flag 1 the two hands' load shares, in the targets mode the first segment from home
        (task.arm_segment_draw), all from the arm process's own stream; a reset with a command has free arms and no
        load. Arm perturbation segments run only in a flag-0 training episode."""
        self.arm_mode = int(task.arm_mode_draw(self._arm_uniform, 1)[0]) if training else task.ARM_FREE
        self.arm_flag, self.arm_fixed = self.arm_mode != task.ARM_FREE, False
        self.load_share = self.arm_random.random(2) if self.arm_flag else np.zeros(2)
        self.arm_state = self._arm_still(np.zeros(self.model.nu))
        if self.arm_mode == task.ARM_TARGET:
            start = self.arm_state[0]
            end, move, hold = task.arm_segment_draw(self._arm_uniform, start, *self._arm_draws()[1:])
            self.arm_state = (start, end, torch.zeros(1, dtype=torch.long), move, hold)
        self.arm_offset = task.arm_target_offset(*self.arm_state[:4])[0].numpy()
        self.previous_flag, self.previous_offset = False, np.zeros(self.model.nu)
        self.delay_offsets = [np.zeros(self.model.nu)] * len(self.delay_actions)
        self.perturb_live = self.perturb_training and not self.arm_flag
        self._apply_hand_load(self.load_share * self.load_max)

    def _advance_arm_targets(self):
        """One control step of this episode's targets, after the reward (task.arm_target_advance in the targets mode
        unless set by hand; otherwise the clock alone, which carries a set_arm_targets transition), then this step's
        offsets."""
        if not self.arm_flag:
            return
        if self.arm_mode == task.ARM_TARGET and not self.arm_fixed:
            self.arm_state = task.arm_target_advance(self.arm_state, torch.tensor([True]), *self._arm_draws())
        else:
            start, end, clock, move, hold = self.arm_state
            self.arm_state = (start, end, clock + 1, move, hold)
        self.arm_offset = task.arm_target_offset(*self.arm_state[:4])[0].numpy()

    def _apply_hand_load(self, load):
        """Loads (kg, left and right) at the palms (task.hand_load_mass, HAND_LOAD_POSITION) into the model: the hand
        bodies' mass and COM and every body's subtree mass; 0 restores the model's own numbers exactly."""
        load = np.asarray(load, dtype=np.float64).reshape(2)
        mass, ipos = task.hand_load_mass(self.hand_mass, self.hand_ipos, load[None], task.HAND_LOAD_POSITION)
        self.model.body_mass[self.hand_bodies] = mass[0]
        self.model.body_ipos[self.hand_bodies] = ipos[0]
        self.model.body_subtreemass[:] = task.hand_subtree_mass(self.subtree_mass, self.hand_above, load[None])[0]
        self.hand_load = load.copy()

    def set_arm_targets(self, q=None, flag=True, seconds=0.):
        """Arm command of this episode from now on (tests, probes; task_config arm_targets), until the next reset: flag
        1, the arms follow q ((nu,) absolute joint angles; its free arm joints, clipped to their ranges; None: home), in
        a quintic over `seconds` from the current targets (0: at once); flag 0, free arms (the policy's own, the slot
        zeros; arm perturbation as the episode had it). The generator stops; the hand loads stay (set_hand_load).
        Re-read the observation as after set_command."""
        if not self.arm_targets:
            raise ValueError('arm targets need task_config arm_targets')
        if isinstance(flag, (bool, np.bool_)):
            flag = bool(flag)
        elif isinstance(flag, (int, np.integer)) and flag in (0, 1):
            flag = bool(flag)
        else:
            raise ValueError('flag must be 0 or 1')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) \
                or seconds < 0:
            raise ValueError('seconds must be a finite non-negative number')
        target = self.home if q is None else np.asarray(q, dtype=float)
        if target.shape != (self.model.nu,) or not np.isfinite(target).all():
            raise ValueError('Expected finite (actuator_count,) arm targets')
        end = np.zeros(self.model.nu)
        if flag:
            joints = list(self.track_joints)
            low, high = self.model.jnt_range[self.joints[joints]].T
            end[joints] = np.clip(target[joints], low, high) - self.home[joints]
        self.arm_state = self._arm_still(end)
        if flag and seconds > 0:
            start = self.arm_offset if self.arm_flag else np.zeros(self.model.nu)
            move = task.arm_move_steps(_batch(start), _batch(end), self.track_joints, self.dt, seconds=seconds)
            self.arm_state = (_batch(start), _batch(end), torch.zeros(1, dtype=torch.long), move,
                              torch.zeros(1, dtype=torch.long))
        self.arm_mode, self.arm_flag, self.arm_fixed = task.ARM_TARGET if flag else task.ARM_FREE, flag, True
        self.arm_offset = task.arm_target_offset(*self.arm_state[:4])[0].numpy()
        if flag:  # no perturbation while the arms follow targets
            self.perturb_live, self.perturb_steps = False, 0
        else:
            self.perturb_live = self.perturb_training

    def set_hand_load(self, load):
        """Loads (kg) on the hands from the next physics step until the next reset (tests, probes): one number for both
        or (left, right), at the palms (HAND_LOAD_POSITION on the wrist_yaw links); also without arm_targets (29-DoF)."""
        if self.hand_bodies is None:
            raise ValueError('hand loads need the arms of the 29-DoF G1')
        load = np.asarray(load, dtype=float)
        if load.shape not in ((), (2,)) or not np.isfinite(load).all() or (load < 0).any():
            raise ValueError('Expected one or two finite non-negative loads (kg)')
        self._apply_hand_load(np.broadcast_to(load, (2,)))

    @property
    def keeps_training_state(self):
        """Whether stages keep this task's training state (train.keeps_training_state): the load ceiling of arm_targets,
        the terrain level of terrain."""
        return self.arm_targets or self.terrain is not None

    def training_state(self):
        """What a stage keeps of this training beyond the weights: the hand load ceiling m_max (arm_targets); the level
        training resets take and the current tile type (terrain; one world, as the full-GPU task's per-world lists)."""
        state = {'load_max': [self.load_max]} if self.arm_targets else {}
        if self.terrain is not None:
            state.update(terrain_level=[self.terrain_level],  # on the plane (a reset with a command): flat, 0
                         terrain_type=[0 if self.terrain_type is None else self.terrain_type])
        return state

    def restore_training_state(self, state, index=0, count=1):
        """The hand load ceiling from a stage's training_state (their mean, within [0, LOAD_LIMIT]); the terrain level
        (saved_terrain_level: the median of the worlds' levels); nothing without the settings or saved values."""
        if self.arm_targets and 'load_max' in state:
            self.load_max = saved_load(state)
        if self.terrain is not None and 'terrain_level' in state:
            self.terrain_level = saved_terrain_level(state)

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
        self.scan_noised = False  # and the height scan is exact

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
        frame's entries of the extra terms (task.TERM_NORMALIZATION), then with arm_blind the arm columns (arm_columns)
        at variance task.ARM_BLIND_SCALE² with mean NaN: the statistics' own, kept as they are (−z·ARM_BLIND_SCALE
        from warm_start.blind_arms; 0 in a fresh normalizer), or with arm_targets at mean 0 and the variances of
        task.ARM_TARGET_VARIANCE per term (warm_start.unblind_arms). None without any. warm_start.insert_terms writes
        the set values into a stage's statistics, TensorNormalize and the CPU training path (train.FrozenMoments) keep
        all of them frozen."""
        positions, mean, variance = [], [], []
        for name, _ in self.extra_terms:
            columns = task.term_positions(self.model.nu, self.history, self.extra_terms, name)
            positions += columns
            mean += [task.TERM_NORMALIZATION[name][0]] * len(columns)
            variance += [task.TERM_NORMALIZATION[name][1]] * len(columns)
        if self.arm_blind:
            columns = self.arm_columns
            positions += columns
            mean += [np.nan] * len(columns)
            variance += [task.ARM_BLIND_SCALE ** 2] * len(columns)
        if self.arm_targets:  # arm_columns' order: term by term, every frame of each
            columns = self.arm_columns
            positions += columns
            mean += [0.] * len(columns)
            variance += [task.ARM_TARGET_VARIANCE[term] for term in task.ARM_BLIND_TERMS
                         for _ in range(len(columns) // len(task.ARM_BLIND_TERMS))]
        if self.height_scan is not None:  # the actor's scan, then the critic's, at mean 0 (task.HEIGHT_SCAN_VARIANCE)
            for columns, value in zip(self.scan_columns, (task.HEIGHT_SCAN_VARIANCE, task.CRITIC_SCAN_VARIANCE)):
                positions += columns
                mean += [0.] * len(columns)
                variance += [value] * len(columns)
        if not positions:
            return None
        return np.array(positions, dtype=np.int64), np.array(mean), np.array(variance)

    @property
    def scan_columns(self):
        """height_scan: positions in the observation of the actor's scan (right after the history) and of the critic's
        (last), two lists; both empty without it."""
        actor = self.frame_size * self.history
        critic = self.observation_space.shape[0] - len(self.critic_grid)
        return (list(range(actor, actor + len(self.scan_grid))),
                list(range(critic, critic + len(self.critic_grid))))

    @property
    def arm_columns(self):
        """Positions in the observation of the arm joints' (arm_joints) joint_pos_rel, joint_vel_rel and last_action
        entries, every history frame (task.joint_positions, ARM_BLIND_TERMS order): the columns task_config arm_blind
        holds constant; the deploy layout's, so real robot only."""
        if not self.real_robot:
            raise ValueError('arm columns are those of the real-robot (deploy) observation layout')
        return task.joint_positions(self.model.nu, self.history, self.extra_sizes, task.ARM_BLIND_TERMS,
                                    self.arm_joints)

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
        if self.terrain is not None:  # terrain: the cap of slow_level's tiles (deterministic, no draw)
            self._cap_command()
            command = _batch(self.command.copy())
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
        """(1, 2) lowest sole point of each foot, m; with task_config terrain above the ground under each sphere."""
        if self.terrain is not None:
            spheres = self.data.geom_xpos[self.soles]
            return task.sole_height(_batch(spheres[..., 2]) - self._ground(spheres[..., :2])[None], self.sole_radius)
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
        previous = _batch(self.previous_action)
        if self.previous_flag:  # arm_targets: the last action as the ONNX graph output it, clip(a) + d/scale
            previous = task.observed_action(previous, _batch(self.previous_offset), torch.from_numpy(self.scale),
                                            torch.tensor([True]))
        parts = (velocity, rotation, command, _batch(self.data.qpos[self.qadr]), _batch(self.data.qvel[self.vadr]),
                 _batch(self.home), previous, self._phase())
        observed = (task.deploy_observation(*parts, phase_stop=self.phase_stop) if self.real_robot
                    else task.observation(*parts))
        if self.extra_terms:  # height_command [c − 0.78], upper_body_slot zeros (arm_targets, flag 1: the arm command)
            height = torch.tensor([self.height], dtype=torch.float64) if self.height_command else None
            upper_body = (task.upper_body_command(torch.tensor([True]), _batch(self.arm_offset)) if self.arm_flag
                          else None)
            observed = torch.cat((observed, task.command_terms(height, self.upper_body_slot, observed, upper_body)),
                                 dim=1)
        if self.history > 1:
            if history == 'fill' or self.frames is None:
                self.frames = observed[:, None].repeat(1, self.history, 1)
            elif history == 'push':
                self.frames = torch.cat((self.frames[:, 1:], observed[:, None]), dim=1)
            else:
                self.frames[:, -1] = observed
            observed = task.history_observation(self.frames, self.model.nu, self.extra_sizes)
        if self.height_scan is not None:  # the newest frame only, noised in a training episode (np_random)
            scan = task.height_scan(*self._scan_pose(), self.scan_grid, *self._terrain_grid())
            if self.scan_noised and self.height_scan['noise']:
                scan = scan + task.scan_noise(lambda shape: torch.from_numpy(self.np_random.random(shape)), scan.shape,
                                              self.height_scan['noise'])
            observed = torch.cat((observed, scan), dim=1)
        if self.critic_extras:  # the pelvis height above the ground under it with terrain (pelvis_height)
            height = _batch(self.data.qpos[2]) if self.terrain is None else _batch(self.pelvis_height())
            observed = torch.cat((observed, task.critic_extras(velocity, height)), dim=1)
        if self.height_scan is not None:  # exact, around the pelvis, above the feet's ground
            feet = torch.from_numpy(self.data.xpos[self.feet, :2].copy())[None]
            observed = torch.cat((observed, task.critic_scan(*self._scan_pose(), feet, self.critic_grid,
                                                             *self._terrain_grid())), dim=1)
        return observed[0].numpy().astype(np.float32)

    def _scan_pose(self):
        """height_scan: the pelvis position (1, 3) and rotation (1, 3, 3) its grids are laid around."""
        return _batch(self.data.qpos[:3]), _batch(self.data.xmat[self.pelvis].reshape(3, 3))

    def _terrain_grid(self):
        """task.terrain_height's heights, corner, cell and lift of task_config terrain."""
        return self.terrain_heights, self.terrain_corner, self.terrain_cell, self.terrain_lift

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # arm_targets: the arm process's own stream, seeded with the reset's seed (never drawn from np_random, so the
        # task's own draws stay the old ones).
        if self.arm_targets and (seed is not None or self.arm_random is None):
            self.arm_random = np.random.default_rng(None if seed is None else (seed, task.ARM_TARGET_STREAM))
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
        # arm_targets: only with free arms (flag 0, _arm_episode).
        self.perturb_live = self.perturb_training = bool(self.arm_perturb) and options.get('command') is None
        self.perturb_steps = 0
        self.perturb_action = np.zeros(self.model.nu)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0., 0., .8]
        self.data.qpos[3:7] = [1., 0., 0., 0.]
        self.data.qpos[self.qadr] = self.home + self.np_random.uniform(-.01, .01, self.model.nu)
        self.data.qvel[:] = 0
        if self.terrain is not None:  # its own stream, so the draws above and below stay the old ones
            self._terrain_reset(options, seed)
        self.steps = 0
        self.previous_action[:] = 0
        if (options or {}).get('command') is not None:
            self.set_command(options['command'])
        else:
            self.fixed_command = False
            self._sample_command()
        # arm_targets: the episode's arm mode, targets and hand loads (its own stream); a load set_hand_load left from the
        # episode before comes off.
        if self.arm_targets:
            self._arm_episode(options.get('command') is None)
        elif self.hand_load.any():
            self._apply_hand_load(np.zeros(2))
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
        # height_scan: a training reset (no command given) noises the actor's scan, drawn with np_random in every
        # observation from this one on, after every other draw; set_command ends it.
        self.scan_noised = self.height_scan is not None and options.get('command') is None
        return self._observation('fill'), {'x_position': float(self.data.qpos[0])}

    def _target(self, action, offset=None):
        """PD target of a clipped action, within the joint ranges, the knees at knee_floor or above (task.pd_target
        within target_low / target_high, as the CUDA task's PD); offset (nu,): task_config arm_targets' d, home moved
        by it (home + d + scale·clip(a)); None: home itself."""
        home = self.home if offset is None else self.home + offset
        return task.pd_target(torch.from_numpy(action), torch.from_numpy(home), torch.from_numpy(self.scale),
                              torch.from_numpy(self.target_low), torch.from_numpy(self.target_high),
                              self.action_limit).numpy()

    def applied_action(self, action):
        """The action this control step applies for a policy output: clipped to the action box, the arm joints
        following the running perturbation segment's held action (task.arm_perturb_apply; none: untouched), then with
        lock_wrists the wrist pitch and yaw entries at 0, the home pose (task.lock_joints), also in a segment. It
        replaces the policy's in everything after it (PD, delay queue, reward, observed previous action). Changes no
        state: step draws and counts off the segments around it. Probes running their own PD apply what step does
        through it, and bound its target by target_low / target_high: knee_floor acts there (_target), not on the
        action, so the observed previous action keeps a knee action below the floor, as g1_ctrl's last_action. Under
        arm_targets with flag 1 the target is _target(applied, arm_offset): home moved by this step's arm targets."""
        action = np.clip(np.asarray(action, dtype=float), -self.action_limit, self.action_limit)
        if self.perturb_steps:
            action = task.arm_perturb_apply(_batch(action), torch.tensor([self.perturb_steps]),
                                            _batch(self.perturb_action), self.arm_mask)[0].numpy()
        if self.lock_wrists:
            action = task.lock_joints(_batch(action), self.wrist_mask)[0].numpy()
        return action

    def _before_physics(self, pushes):
        """A hook right before the step's physics, after the step's draws (world model collector, scripts/lab/wm):
        pushes (frame_skip, 3) the pelvis force of each substep, None without a push. Does nothing here."""

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
        # arm_targets, flag 1: the arm targets of this step move home (offset d); flag 0 and off: home itself.
        offset = self.arm_offset.copy() if self.arm_flag else None
        target = self._target(action) if offset is None else self._target(action, offset)
        # action_delay: substep j follows the target of the action delay_lags[j] steps old (all 0 without delay); under
        # arm_targets each with the offsets of its own step.
        if self.arm_targets:
            targets = [target, *(self._target(past, past_offset) if past_offset.any() else self._target(past)
                                 for past, past_offset in zip(self.delay_actions, self.delay_offsets))]
        else:
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
        self._before_physics(pushes)
        feet_before = self.data.xpos[self.feet, :2].copy()
        pelvis_before = self.data.qpos[:2].copy()
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
        # arm_targets: the flag of this step and, with flag 1, the arms' tracking of home + d after the step
        # (task.arm_track; arm_press and the arms' posture are then off in reward_components); whether this step's
        # targets hold (task.arm_holding) and, while the arm_error weight is non-zero, the arms' error for its term.
        arms = {}
        if self.arm_targets:
            target_pose = _batch(self.home if offset is None else self.home + offset)
            tracked = (task.arm_track(_batch(self.data.qpos[self.qadr]), target_pose, self.track_joints)
                       if offset is not None else torch.zeros(1, dtype=torch.float64))
            error = (task.arm_error(_batch(self.data.qpos[self.qadr]), target_pose, self.track_joints)
                     if offset is not None else None)
            hold = bool(task.arm_holding(*self.arm_state[:4])[0])
            arms = dict(arm_flag=torch.tensor([offset is not None]), arm_track=tracked,
                        arm_posture_joints=self.arm_posture_joints)
            if self.reward_weights['arm_error']:
                arms.update(arm_error=torch.zeros(1, dtype=torch.float64) if error is None else error,
                            arm_hold=torch.tensor([hold]))
        # terrain: the heights above the ground under the body and the feet's ground normals.
        ground = {}
        if self.terrain is not None:
            ground = self._ground_terms()
            sole = task.sole_height(ground['spheres'], self.sole_radius)
        # Rewarded for the command the policy observed when it chose this action.
        components, fell = task.reward_components(
            _batch(velocity), _batch(self.data.xmat[self.pelvis].reshape(3, 3)),
            ground['height'] if ground else _batch(self.data.qpos[2]),
            _batch(self.observed_command), _batch(self.data.ctrl), _batch(action), _batch(self.previous_action),
            ground['foot_height'] if ground else _batch(self.data.xpos[self.feet, 2]), self._phase(),
            _batch(self.data.qpos[self.qadr]),
            _batch(self.home), _batch(self.data.xmat[self.torso].reshape(3, 3)), weights=self.reward_weights,
            posture_joints=self.posture_joints, heading_target=_batch(self.heading_target),
            stance_fraction=self.gait['stance_fraction'], hip_joints=self.hip_joints,
            foot_speed=_batch(np.linalg.norm(self.data.xpos[self.feet, :2] - feet_before, axis=1) / self.dt),
            foot_rotation=_batch(self.data.xmat[self.feet].reshape(len(self.feet), 3, 3)),
            sole_height=((sole if ground else self._sole_height())
                         if self.sole_contact or self.reward_weights['clearance'] else None),
            sole_contact=self.sole_contact,
            yaw_drift_error=yaw_drift, requested_command=_batch(self.command.copy()),
            self_contact=None if self_force is None else task.self_contact_share(_batch(self_force)),
            arm_press=arm_press, feet_flat_stance=self.feet_flat_stance, height_command=height_command,
            stand_home=stand_home, **arms, **({'floor_normal': ground['floor_normal']} if ground else {}))
        components = {key: float(value[0]) for key, value in components.items()}
        fell = bool(fell[0])
        info = dict(x_position=float(self.data.qpos[0]), fell=fell,
                    command=self.observed_command.copy(), velocity=velocity[3:5].copy(), reward_components=components)
        if self.height_command:
            info['height_command'] = self.height
        if self.terrain is not None:  # the tile (None: the plane), and the path commanded and walked along it, m
            commanded, walked = task.terrain_progress(_batch(self.data.qpos[:2] - pelvis_before),
                                                      _batch(self.data.xmat[self.pelvis].reshape(3, 3)),
                                                      _batch(self.command.copy()), self.dt)
            self.terrain_path += [float(commanded[0]), float(walked[0])]
            info.update(terrain_type=None if self.terrain_type is None else self.terrain_names[self.terrain_type],
                        terrain_level=None if self.terrain_type is None else self.tile_level,
                        terrain_path=self.terrain_path.copy(), pelvis_height=float(ground['height'][0]))
        if self.arm_targets:  # the flag, the arms' mean |error| (rad) with flag 1, the hand loads (kg), the hold
            info.update(arm_flag=offset is not None, hand_load=self.hand_load.copy(),
                        arm_error=None if error is None else float(error[0]), arm_track=float(tracked[0]),
                        arm_hold=offset is not None and hold)
        self.previous_action[:] = action
        if self.arm_targets:  # the offsets this step followed, for the observed last action and the delay queue
            self.previous_flag = offset is not None
            self.previous_offset = np.zeros(self.model.nu) if offset is None else offset
        if self.delay_actions:
            self.delay_actions = [action, *self.delay_actions[:-1]]
            if self.arm_targets:
                self.delay_offsets = [self.previous_offset, *self.delay_offsets[:-1]]
        # The reward above belongs to the command the action was taken under.
        if not self.fixed_command and self.np_random.random() < self.dt / task.COMMAND_RESAMPLE_SECONDS:
            self._sample_command(resample=True)
            if self.height_command:  # a new height for the new command, after the command's draws
                self._draw_height()
        # arm_targets: the targets move on (their own stream), shown in the next observation.
        self._advance_arm_targets()
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


def saved_terrain_level(state):
    """The level (task_config terrain) the CPU task takes from a stage's training_state: the median of the worlds'
    levels, checked whole numbers within [0, TERRAIN_LEVELS)."""
    values = np.asarray(state['terrain_level'])
    if values.ndim != 1 or not len(values) or not np.isin(values, np.arange(TERRAIN_LEVELS)).all():
        raise ValueError('Invalid saved terrain levels')
    return int(np.median(values) // 1)


def restored_terrain_levels(state, types):
    """(n,) levels (task_config terrain) for worlds of tile types `types` (n,) from a stage's training_state, type by
    type: the saved levels of worlds of the same type (all saved levels if none had it; level 0 for a type the state
    lists in terrain_new_types, carried_terrain_state's mark of a generator's new types) spread over this type's worlds
    in order, so another world count keeps each type's distribution; checked as saved_terrain_level."""
    saved_terrain_level(state)
    levels = np.asarray(state['terrain_level'], dtype=np.int64)
    kinds = np.asarray(state.get('terrain_type', np.zeros(len(levels))), dtype=np.int64)
    if kinds.shape != levels.shape:
        raise ValueError('Invalid saved terrain types')
    new = set(state.get('terrain_new_types', ()))
    types = np.asarray(types, dtype=np.int64)
    result = np.zeros(len(types), dtype=np.int64)
    for kind in np.unique(types):
        worlds = np.flatnonzero(types == kind)
        own = levels[kinds == kind]
        own = own if len(own) else np.zeros(1, np.int64) if int(kind) in new else levels
        result[worlds] = own[np.arange(len(worlds)) * len(own) // len(worlds)]
    return result


def carried_terrain_state(state, old_names, new_names):
    """A stage's training_state (task_config terrain) moved from a generator with tile types old_names to one with
    new_names (warm_start.carry_over): each world's tile type mapped by name (the same index for the types generator 3
    shares with 2), worlds of a type the new generator lacks left out, and terrain_new_types listing the new types no
    saved world had, which restored_terrain_levels starts at level 0. The same names: the state as it is."""
    if not state or 'terrain_level' not in state or tuple(old_names) == tuple(new_names):
        return state
    levels = list(state['terrain_level'])
    kinds = list(state.get('terrain_type', [0] * len(levels)))
    if len(kinds) != len(levels) or any(not 0 <= kind < len(old_names) for kind in kinds):
        raise ValueError('Invalid saved terrain types')
    kept = [(level, new_names.index(old_names[kind])) for level, kind in zip(levels, kinds)
            if old_names[kind] in new_names]
    if not kept:  # no world on a shared type: the levels start afresh
        return {key: value for key, value in state.items()
                if key not in ('terrain_level', 'terrain_type', 'terrain_new_types')}
    return {**{key: value for key, value in state.items() if key != 'terrain_new_types'},
            'terrain_level': [level for level, _ in kept], 'terrain_type': [kind for _, kind in kept],
            'terrain_new_types': [k for k, name in enumerate(new_names) if name not in old_names]}


def saved_load(state):
    """The hand load ceiling m_max (kg) in a stage's training_state (task_config arm_targets): the mean of its values
    (one per world), checked finite and non-negative, within [0, LOAD_LIMIT] (the CUDA task's restore too)."""
    values = np.asarray(state['load_max'], dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid saved hand load ceiling')
    return float(min(values.mean(), task.LOAD_LIMIT))
