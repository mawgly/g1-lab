"""G1 getting up after a fall (task g1_getup, iteration 33), CPU.

The walking candidates' robot and connector (config.G1_GETUP_PROFILE: the real-robot model with Unitree's foot and hand
capsules, locked wrists, the knee floor, the ±10 action box; the 490 actor inputs of the walking layout with the command
always zero and the gait phase stopped), so a policy that gets up hands over to the walking one through the same
g1_ctrl. Episodes start from the pose bank (g1_getup_bank) and last 10 s; only a non-finite state or a joint faster
than 50 rad/s ends one early. The reward (g1_tensor_math.getup_reward_components) pays for the height of the head and an
upright torso, a bonus for standing; a vertical assist force on the pelvis (assist_force) helps while it is nearly
upright and steps down as the robot learns to stand (assist_update at each reset). With task_config pd_bound
(iteration 34) each motor target stays within a share c of its peak torque from the observed joint angle
(g1_tensor_math.pd_bound), c stepping down the same way (pd_bound_update). The reward weight impact (iteration 35)
charges the speed at which a body other than the feet hits the floor, read after every physics substep
(g1_tensor_math.impact_contacts / approach_speed / impact_penalty). task_config stand_pose_std (iteration 36) sets the
width σ of the standing pose term; action_change_unbounded (iteration 37) charges the action_change term on the action
before pd_bound. The reward weight sole_impact (iteration 44) charges the feet the same way, the sole spheres above
GETUP_SOLE_IMPACT_SPEED; stance_width (after branch 46) the feet's width beyond GETUP_STANCE_WIDTH while the pelvis is
up and both soles are down (g1_tensor_math.stance_width / stance_active); arm_press (iteration 52) standing, the free
arm joints' PD target before pd_bound against their angles (g1_tensor_math.arm_press); style (iteration 53) the
nearness of the frame to a human one getting up while the head is 0.3-1.1 m up (task_config style_library,
g1_getup_style; g1_tensor_math.style_features / style_distance); head_rise (iteration 57) the head top rising faster
than GETUP_HEAD_RISE_SPEED (g1_tensor_math.head_rise). With task_config track_clip (iteration 64) the task is motion
tracking instead (BeyondMimic / mjlab): the policy follows one human clip getting up (g1_getup_track), then holds its
last frame; the tracking terms replace the getting-up ones, mjlab's terminations end an episode early, the critic also
sees the clip's phase and the anchor's errors (_clip_reset, _clip_step). With task_config settle_clip (iteration 66)
the task is settling instead: from the bank's fall classes the policy rolls and lays its limbs into the nearest of the
clips' first frames, where a tracker of that clip takes over (_settle_reset, _settle_step; the success the switch).
Equations live in g1_tensor_math, shared with the CUDA task.
"""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch

from . import g1_getup_bank, g1_getup_human, g1_getup_style, g1_getup_track
from . import g1_tensor_math as task
from .config import (G1_GETUP_DEFAULTS, G1_GETUP_PROFILE, G1_GETUP_SETTLE_WEIGHTS, G1_GETUP_TRACK_WEIGHTS,
                     GETUP_CRITIC_EXTRAS, TRACK_CRITIC_EXTRAS, g1_getup_config)
from .g1_getup_bank import instabilities
from .g1_velocity import G1VelocityEnv

EPISODE_STEPS = round(task.GETUP_EPISODE_SECONDS / .02)  # 500 at the walking control step
SETTLE_STEPS = round(task.SETTLE_EPISODE_SECONDS / .02)  # 350: settle_clip's episodes
# human_starts on CUDA: the human start draws come from a generator of their own, seeded seed + this.
HUMAN_START_STREAM = 56_000_000
HIP_ROLL_PART = 'hip_roll'  # task_config hip_roll_out's joints, by name


def outward_signs(model, joints, home):
    """(the hip roll joints' indices among the actuated `joints` (qpos addresses' order), their outward signs): +1 where a
    positive angle swings the foot of its own leg away from the body's midline, −1 where a negative one does, found by
    the kinematics of the home pose (as symmetry.Mirror reads the left / right pairs from the model: by name and
    geometry, not by a table). On the G1 the left leg's +, the right leg's −."""
    names = [model.joint(j).name for j in joints]
    data = mujoco.MjData(model)
    found, signs = [], []
    for index, name in enumerate(names):
        if HIP_ROLL_PART not in name:
            continue
        foot = model.body(f"{name.split('_')[0]}_ankle_roll_link").id
        width = []
        for delta in (0., .1):
            mujoco.mj_resetData(model, data)
            data.qpos[3] = 1.
            data.qpos[[model.jnt_qposadr[j] for j in joints]] = home
            data.qpos[model.jnt_qposadr[joints[index]]] += delta
            mujoco.mj_kinematics(model, data)
            width.append(abs(float(data.xpos[foot, 1] - data.xpos[model.body('pelvis').id, 1])))
        found.append(index)
        signs.append(1 if width[1] > width[0] else -1)
    return tuple(found), tuple(signs)


class G1GetupEnv(G1VelocityEnv):
    """g1_getup on the CPU: one world of the CUDA task's batch, in float64.

    Reset options (probes, evaluation): start (a bank row; default a random training row), assist_force (N, this episode
    and on; default the world's own, after assist_update), action_delay_steps, motor_friction_scale, body_friction
    (default drawn as in training), pd_bound (the PD error bound c, this episode and on: a share of the peak torque,
    also without the setting, as a probe; None: no bound; 'min': the setting's lowest, no bound without it; default
    the world's own, after pd_bound_update). With track_clip also track_frame (a clip frame to start from, its
    velocities: reference state init) and track_noise (default true: mjlab's start noise on a clip start); start is
    then a bank row the reference is aligned to (track_align); without any of them a training reset draws its start
    (TRACK_START_SHARES), an evaluation one (assist force or bound set) starts from the clip's first frame with noise.
    With settle_clip: start (any bank row; default a training row of SETTLE_CLASSES), assist_force, pd_bound and the
    randomization options as above."""

    keeps_training_state = True  # stages save its assist force and bound (training_state), resumed runs restore them
    asymmetric_critic = True  # the actor sees the first actor_inputs only (policies.CriticExtrasPolicy), on CPU too

    def __init__(self, max_episode_steps=EPISODE_STEPS, render_mode=None, reward_weights=None,
                 assist_force=G1_GETUP_DEFAULTS['assist_force'], action_delay=G1_GETUP_DEFAULTS['action_delay'],
                 motor_friction=G1_GETUP_DEFAULTS['motor_friction'], body_friction=G1_GETUP_DEFAULTS['body_friction'],
                 pd_bound=G1_GETUP_DEFAULTS['pd_bound'], bank=None,
                 stand_pose_std=G1_GETUP_DEFAULTS['stand_pose_std'],
                 action_change_unbounded=G1_GETUP_DEFAULTS['action_change_unbounded'],
                 style_library=G1_GETUP_DEFAULTS['style_library'], style_judge=G1_GETUP_DEFAULTS['style_judge'],
                 human_starts=G1_GETUP_DEFAULTS['human_starts'], human_bank=None,
                 stance_pelvis=G1_GETUP_DEFAULTS['stance_pelvis'], hip_roll_out=G1_GETUP_DEFAULTS['hip_roll_out'],
                 track_clip=G1_GETUP_DEFAULTS['track_clip'], track_bank=None,
                 settle_clip=G1_GETUP_DEFAULTS['settle_clip'], settle_bank=None):
        # style_judge and the amp weight act on the full-GPU path only (g1_getup_tensor_env, getup_amp): validated
        # here, nothing on the CPU. human_bank: the human starts (g1_getup_human.load's dict; tests) instead of the file
        # human_starts names; track_bank likewise the clip (g1_getup_track's arrays) instead of the file track_clip names;
        # settle_bank the clips (a list, in settle_clip's order) instead of the files settle_clip names.
        settings = g1_getup_config(dict(reward_weights=reward_weights or {}, assist_force=assist_force,
                                        action_delay=action_delay, motor_friction=motor_friction,
                                        body_friction=list(body_friction), pd_bound=pd_bound,
                                        stand_pose_std=stand_pose_std,
                                        action_change_unbounded=action_change_unbounded,
                                        style_library=style_library, style_judge=style_judge,
                                        human_starts=human_starts, stance_pelvis=stance_pelvis,
                                        hip_roll_out=hip_roll_out, track_clip=track_clip, settle_clip=settle_clip))
        if settings['settle_clip'] is not None:  # settling episodes are shorter (SETTLE_EPISODE_SECONDS)
            max_episode_steps = min(max_episode_steps, SETTLE_STEPS)
        super().__init__(max_episode_steps, render_mode, action_delay=settings['action_delay'],
                         motor_friction=settings['motor_friction'], **G1_GETUP_PROFILE)
        self.getup_weights = settings['reward_weights']
        self.stand_pose_std = settings['stand_pose_std']  # σ of the standing pose term (rad)
        self.stance_pelvis = settings['stance_pelvis']  # the stance_width term's pelvis threshold (m)
        # hip_roll_out: the hip roll joints, each with the sign of its outward rotation (outward_signs), and with the
        # setting the PD target's bound on that side at the limit (target_low / target_high, as knee_floor: _target, the
        # CUDA task's joint_low / joint_high, export_g1's clip); without it the joint ranges, bit for bit.
        self.hip_roll_out = settings['hip_roll_out']
        self.hip_roll_joints, self.hip_roll_signs = outward_signs(self.model, self.joints, self.home)
        if self.hip_roll_out is not None:
            self.target_low, self.target_high = self.target_low.copy(), self.target_high.copy()
            for joint, sign in zip(self.hip_roll_joints, self.hip_roll_signs):
                low, high = self.target_low[joint], self.target_high[joint]
                if not (low < sign * self.hip_roll_out < high):
                    raise ValueError('hip_roll_out must lie within the hip roll joint range')
                if sign > 0:
                    self.target_high[joint] = self.hip_roll_out
                else:
                    self.target_low[joint] = -self.hip_roll_out
        # action_change_unbounded: the action_change term's action is the one before pd_bound (box clip, lock_wrists;
        # unbounded_action) and its own one of the step before, zero at a reset; None while off.
        self.change_unbounded = settings['action_change_unbounded']
        self.previous_unbounded = np.zeros(self.model.nu) if self.change_unbounded else None
        self.body_friction = tuple(settings['body_friction'])
        # The critic's inputs: the walking ones, then the head height and the assist force (task.getup_critic_extras).
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.actor_inputs + GETUP_CRITIC_EXTRAS,),
                                            dtype=np.float32)
        # Joint limit term: the legs' soft limits (0.9 of the joint range, task.soft_limits).
        low, high = self.model.jnt_range[self.joints].T
        self.soft_low, self.soft_high = task.soft_limits(low, high)
        # Body friction: one sliding friction per episode for every robot geom that collides and the floor (with equal
        # priority MuJoCo takes the pair's larger one, so the floor follows; the sole spheres and the foot capsules take
        # their own by priority), the files' values kept for reference.
        floor = self.model.geom('floor').id
        colliding = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        self.friction_geoms = np.flatnonzero((self.robot_geoms & colliding) | (np.arange(self.model.ngeom) == floor))
        self.nominal_friction = self.model.geom_friction[:, 0].copy()
        self.friction = float(self.model.geom_friction[floor, 0])
        # impact reward weight: the robot geoms whose floor contacts it reads (all but the sole spheres) and, of the
        # last control step, each body's largest approach speed per physics substep ((frame_skip, nbody), 0 without a
        # floor contact; None while the weight is 0: nothing is read).
        self.floor = floor
        self.impact_geoms = self.robot_geoms.copy()
        self.impact_geoms[self.soles.ravel()] = False
        self.impact_substeps = np.zeros((self.frame_skip, self.model.nbody)) if self.getup_weights['impact'] else None
        # sole_impact reward weight: the same for the sole spheres, per foot ((frame_skip, 2), left then right; None
        # while the weight is 0).
        self.sole_geoms = np.zeros(self.model.ngeom, dtype=bool)
        self.sole_geoms[self.soles.ravel()] = True
        self.sole_impact_substeps = np.zeros((self.frame_skip, 2)) if self.getup_weights['sole_impact'] else None
        # stance_width reward weight: whether the feet's width (task.stance_width) is read after each step.
        self.stance_weighted = bool(self.getup_weights['stance_width'])
        # arm_press reward weight: the arm joints but the locked wrists (15-19, 22-26), whose target before pd_bound is
        # compared with their angle after the step; read only while weighted.
        self.press_joints = tuple(j for j in self.arm_joints if not bool(self.wrist_mask[j]))
        self.press_weighted = bool(self.getup_weights['arm_press'])
        # sole_slip reward weight: whether the feet's speeds over each step are read.
        self.slip_weighted = bool(self.getup_weights['sole_slip'])
        # style reward weight: the human frame library (g1_getup_style.load of task_config style_library) and the
        # joints of its features (the free ones, all but the locked wrists); None, nothing loaded, while the weight is
        # 0.
        self.style_joints = tuple(int(j) for j in np.flatnonzero(~self.wrist_mask.numpy()))
        self.style = None
        if self.getup_weights['style']:
            self.style = g1_getup_style.load(settings['style_library'])
            if tuple(self.style['joints'].tolist()) != self.style_joints:
                raise ValueError('style library joints do not match the task (the free joints)')
        # The assist force of this world (N) and the head heights of the last GETUP_ASSIST_SECONDS, for assist_update.
        self.assist = settings['assist_force']
        self.lift = 0.
        self.head_window = deque(maxlen=round(task.GETUP_ASSIST_SECONDS / self.dt))
        # pd_bound: [start, lowest] or None, and this world's bound c (a share of the peak torque; None: no bound),
        # which steps down at resets as the assist does (task.pd_bound_update). The bound acts on every joint but the
        # locked wrists, around the newest observed joint_pos_rel (observed_position, set with each observation; the
        # actor's float32 numbers, as the export reads them).
        self.pd_bound = settings['pd_bound']
        self.bound = None if self.pd_bound is None else self.pd_bound[0]
        self.peak = self.motors[2]  # y1, the motor model's peak torque
        self.bound_free = ~self.wrist_mask.numpy()
        self.position_columns = np.array(task.newest_frame_columns(self.model.nu, self.history, self.extra_sizes))
        self.observed_position = np.zeros(self.model.nu)
        self._bank = bank  # a path, a loaded bank (g1_getup_bank.load) or None: g1_getup_bank.BANK, read at reset
        self.training_rows = None
        self.start_row, self.start_class = None, None
        # human_starts: {share, md5} or None; the human starts (read at the first reset that needs them) and whether
        # this episode started from one (human_start: its row, None for a bank start), whose end steps neither the
        # assist nor the bound.
        self.human_starts = settings['human_starts']
        self._human = human_bank
        self.human_start = None
        self.getup_at, self.getup_held, self.hold_steps = None, None, 0
        self.head = 0.
        # track_clip (iteration 64): the human clip the task follows (g1_getup_track.load of the setting's file, its md5
        # checked; track_bank: a clip's arrays instead, tests); None while off: nothing loaded, the task the old one.
        self.track_clip = settings['track_clip']
        self.clip = None
        if self.track_clip is not None:
            self._setup_clip(track_bank if track_bank is not None
                             else g1_getup_track.load(self.track_clip['file'], self.track_clip['md5']))
        # settle_clip (iteration 66): the clips whose first frames are the settling targets (each md5 checked);
        # None while off: nothing loaded, the task the old one.
        self.settle_clip = settings['settle_clip']
        self.settle = None
        if self.settle_clip is not None:
            self._setup_settle(settle_bank if settle_bank is not None
                               else [g1_getup_track.load(entry['file'], entry['md5']) for entry in self.settle_clip])

    def _setup_settle(self, clips):
        """settle_clip's constants and state: per clip its first frame's pelvis gravity direction and joints (the
        targets, settle_gravity (K, 3), settle_position (K, 29)) and name; the free joints (all but the locked wrists),
        the weights (G1_GETUP_SETTLE_WEIGHTS with reward_weights' impact and sole_impact), the hand geoms
        (hand_force), the steps of a success (SETTLE_DONE_SECONDS); no assist. Per episode: the steps on end near a
        target and still (settle_count), when the success came (settle_at, s; None before) and into which target."""
        if len(clips) != len(self.settle_clip):
            raise ValueError('settle_clip: one clip per setting entry')
        gravity, position = [], []
        for clip in clips:
            g1_getup_track.check(clip)
            first = np.asarray(clip['qpos'][0], dtype=np.float64)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, first[3:7] / np.linalg.norm(first[3:7]))
            gravity.append(task.projected_gravity(rotation.reshape(1, 3, 3))[0])
            position.append(first[7:])
        self.settle = dict(gravity=np.array(gravity), position=np.array(position),
                           names=tuple(entry['file'] for entry in self.settle_clip))
        self.settle_joints = self.style_joints  # the free joints
        self.settle_weights = {**G1_GETUP_SETTLE_WEIGHTS, 'impact': self.getup_weights['impact'],
                               'sole_impact': self.getup_weights['sole_impact']}
        self.settle_hold = round(task.SETTLE_DONE_SECONDS / self.dt)
        self.self_force = np.zeros(self.frame_skip) if self.settle_weights['self_contact'] else None
        names = [self.model.body(b).name for b in range(self.model.nbody)]
        self.hand_geoms = self.robot_geoms & np.array([task.TRACK_HAND_PART in names[b] for b in self.model.geom_bodyid])
        self.assist = 0.
        self.settle_rows = None  # the bank's training rows of SETTLE_CLASSES, read at a reset
        self.settle_count, self.settle_at, self.settle_target = 0, None, None

    def _setup_clip(self, clip):
        """track_clip's constants and state: the clip and its reference frames (g1_getup_track.hold_frames), its last
        frame and the clock that ends an episode (TRACK_HOLD_SECONDS later), the tracked bodies (TRACK_BODIES' ids, the
        anchor's and the ends' indices among them), the hand geoms (rollout/hand_force: bodies named TRACK_HAND_PART),
        the weights (G1_GETUP_TRACK_WEIGHTS with reward_weights' impact and sole_impact), the critic's wider
        observation; no assist (0 N, no curriculum). Per episode: the clip clock (control steps from the clip's start),
        the reference's yaw and xy offset, how it started and whether the clock reached the clip's last frame."""
        g1_getup_track.check(clip)
        self.clip = clip
        self.clip_frames = g1_getup_track.hold_frames(clip)
        self.clip_last = len(clip['qpos']) - 1
        self.clip_end = self.clip_last + round(task.TRACK_HOLD_SECONDS / self.dt)
        self.clip_grace = round(task.TRACK_GRACE_SECONDS / self.dt)  # a bank start's steps without the ends' rule
        names =[self.model.body(b).name for b in range(self.model.nbody)]
        self.track_bodies = np.array([self.model.body(name).id for name in task.TRACK_BODIES])
        self.track_anchor = task.TRACK_BODIES.index(task.TRACK_ANCHOR)
        self.track_ends = tuple(task.TRACK_BODIES.index(name) for name in task.TRACK_END_BODIES)
        self.track_root = int(self.model.body_rootid[self.pelvis])
        self.hand_geoms = self.robot_geoms & np.array([task.TRACK_HAND_PART in names[b] for b in self.model.geom_bodyid])
        self.track_weights = {**G1_GETUP_TRACK_WEIGHTS, 'impact': self.getup_weights['impact'],
                              'sole_impact': self.getup_weights['sole_impact']}
        self.self_force = np.zeros(self.frame_skip) if self.track_weights['self_contact'] else None
        # The joint ranges a noisy clip start keeps the joints in (task.track_start_pose).
        self.range_low, self.range_high = (self.model.jnt_range[self.joints, 0].copy(),
                                           self.model.jnt_range[self.joints, 1].copy())
        self.observation_space = spaces.Box(-np.inf, np.inf, (
            self.actor_inputs + GETUP_CRITIC_EXTRAS + TRACK_CRITIC_EXTRAS,), dtype=np.float32)
        self.assist = 0.
        self.clip_rows = None  # the bank's training rows lying on the back (TRACK_BANK_CLASS), read at a reset
        self.clip_clock, self.clip_yaw, self.clip_offset = 0, 0., np.zeros(2)
        self.clip_start, self.clip_frame, self.clip_reached = None, 0, False

    def _clip_reference(self):
        """(1, B, …) the reference at this episode's clip clock (task.clip_reference)."""
        return task.clip_reference(self.clip_frames, np.array([self.clip_clock]), np.array([self.clip_yaw]),
                                   self.clip_offset[None])

    def _clip_robot(self):
        """(1, B, …) the tracked bodies now: frame origins, rotations, their world linear and angular velocities."""
        bodies = self.track_bodies
        com = np.broadcast_to(self.data.subtree_com[self.track_root], (len(bodies), 3))
        linear, angular = task.body_point_velocity(self.data.cvel[bodies], self.data.xpos[bodies], com)
        return (self.data.xpos[bodies][None].copy(), self.data.xmat[bodies].reshape(1, -1, 3, 3).copy(), linear[None],
                angular[None].copy())

    def _clip_extras(self):
        """(1, TRACK_CRITIC_EXTRAS) the critic's track_clip inputs (task.track_critic_extras)."""
        anchor = self.track_bodies[self.track_anchor]
        position, rotation, _, _ = self._clip_reference()
        return task.track_critic_extras(np.array([self.clip_clock]), self.clip_last, self.data.xpos[anchor][None],
                                        self.data.xmat[anchor].reshape(1, 3, 3), position[:, self.track_anchor],
                                        rotation[:, self.track_anchor])

    def _hand_force(self):
        """The floor's force on the hands at this physics step (N, the sum of the contact-frame force norms of their
        floor contacts; rollout/hand_force's number, the probe's)."""
        total = 0.
        if self.data.ncon:
            selected, _, _ = task.impact_contacts(self.data.contact.geom, self.floor, self.hand_geoms)
            for k in np.flatnonzero(selected):
                mujoco.mj_contactForce(self.model, self.data, int(k), self._contact_force)
                total += float(np.linalg.norm(self._contact_force[:3]))
        return total

    @property
    def bank(self):
        """The start bank (g1_getup_bank.load), read at the first reset; training draws its rows without holdout."""
        if not isinstance(self._bank, dict):
            self._bank = g1_getup_bank.load(g1_getup_bank.BANK if self._bank is None else self._bank)
        if self.training_rows is None:
            self.training_rows = np.flatnonzero(~self._bank['holdout'])
        return self._bank

    @property
    def human(self):
        """The human starts (g1_getup_human.load of the file human_starts names, md5 checked), read when first needed."""
        if self._human is None:
            self._human = g1_getup_human.load(None if self.human_starts is None else self.human_starts['md5'])
        return self._human

    @property
    def fixed_normalization(self):
        """The command and gait phase columns (task.zero_term_columns, always 0 here) at mean 0, variance 1."""
        columns = task.zero_term_columns(self.model.nu, self.history, self.extra_sizes)
        return np.array(columns, dtype=np.int64), np.zeros(len(columns)), np.ones(len(columns))

    def set_command(self, command):
        """g1_getup has no command: only (0, 0, 0) is accepted (the walking scenario runners set one)."""
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or np.any(command != 0):
            raise ValueError('g1_getup has no command: the policy always sees (0, 0, 0)')
        super().set_command(command)

    def set_body_friction(self, friction):
        """The sliding friction of every colliding robot geom and of the floor until the next reset."""
        if isinstance(friction, bool) or not isinstance(friction, (int, float, np.floating)) or not np.isfinite(friction) \
                or friction <= 0:
            raise ValueError('body_friction must be a positive number')
        self.friction = float(friction)
        self.model.geom_friction[self.friction_geoms, 0] = self.friction

    def training_state(self):
        """What a stage keeps of this world's training beyond the weights: its assist force, with pd_bound its bound."""
        state = {'assist_force': [self.assist]}
        if self.pd_bound is not None:
            state['pd_bound'] = [self.bound]
        return state

    def restore_training_state(self, state, index=0, count=1):
        """This world's assist force from a stage's training_state (task.assist_resample over `count` worlds), with
        pd_bound its bound too, within the setting's [lowest, start] (a stage without bounds: the start)."""
        forces = np.asarray(state['assist_force'], dtype=np.float64)
        if forces.ndim != 1 or not len(forces) or not np.isfinite(forces).all() or (forces < 0).any():
            raise ValueError('Invalid saved assist forces')
        self.assist = 0. if self.settle is not None else float(task.assist_resample(forces, count)[index])
        if self.pd_bound is not None and 'pd_bound' in state:
            self.bound = float(np.clip(task.assist_resample(saved_bounds(state), count)[index], self.pd_bound[1],
                                       self.pd_bound[0]))

    def set_pd_bound(self, fraction):
        """This world's PD error bound c from the next step on, until a reset changes it (tests, probes; the reset
        option pd_bound): a share of the peak torque (> 0, also without the setting), None (no bound) or 'min' (the
        setting's lowest; no bound without it)."""
        if isinstance(fraction, str) and fraction == 'min':
            fraction = None if self.pd_bound is None else self.pd_bound[1]
        elif fraction is not None and (isinstance(fraction, bool) or not isinstance(
                fraction, (int, float, np.floating, np.integer)) or not np.isfinite(fraction) or fraction <= 0):
            raise ValueError("pd_bound must be a positive number, None or 'min'")
        self.bound = None if fraction is None else float(fraction)

    def applied_action(self, action):
        """The action this control step applies for a policy output: clipped to the action box, with a bound c the PD
        error bound around the observed angles (task.pd_bound), then lock_wrists (G1VelocityEnv.applied_action, whose
        box clip leaves a bounded action as it is). Changes no state; probes with their own PD apply what step does
        through it."""
        action = np.clip(np.asarray(action, dtype=float), -self.action_limit, self.action_limit)
        if self.bound is not None:
            action = task.pd_bound(action[None], self.observed_position[None], np.array([self.bound]), self.peak,
                                   self.kp, self.scale, self.bound_free)[0]
        return super().applied_action(action)

    def unbounded_action(self, action):
        """The action before pd_bound for a policy output: clipped to the action box, then lock_wrists
        (G1VelocityEnv.applied_action; no arm perturbation here), what the action_change term charges under
        action_change_unbounded. Without a bound it is applied_action's, bit for bit. Changes no state."""
        return super().applied_action(action)

    def _head_mean(self):
        """Mean head height over the last GETUP_ASSIST_SECONDS of the episode; NaN when it was shorter."""
        full = len(self.head_window) == self.head_window.maxlen
        return float(np.mean(self.head_window)) if full else float('nan')

    def style_features(self):
        """(1, f) the style term's features of the current state (task.style_features: the free joints' angles and
        speeds, the gravity in the pelvis frame, the pelvis height and vertical speed). Probes read it too."""
        return task.style_features(self.data.qpos[self.qadr][None], self.data.qvel[self.vadr][None],
                                   self.data.xmat[self.pelvis].reshape(1, 3, 3), self.data.qpos[2:3],
                                   self.data.qvel[2:3], self.style_joints)

    def _head(self):
        return task.head_height(self.data.xpos[self.torso][None], self.data.xmat[self.torso].reshape(1, 3, 3))

    def _impact_speeds(self, geoms=None):
        """(nbody,) per body the largest approach speed (m/s, task.approach_speed) of its floor contacts at this
        physics step, read after mj_step: its contacts, cvel and subtree_com are those of the state the step started
        from, so the speed before the hit. The contacts task.impact_contacts picks (the floor and a geom of `geoms`,
        (ngeom,) bool; by default impact's, any robot geom but the sole spheres; sole_impact's sole_geoms); 0 for a
        body without one."""
        speed = np.zeros(self.model.nbody)
        if self.data.ncon:
            contact = self.data.contact
            selected, robot, sign = task.impact_contacts(contact.geom, self.floor,
                                                         self.impact_geoms if geoms is None else geoms)
            if selected.any():
                body = self.model.geom_bodyid[robot[selected]]
                np.maximum.at(speed, body, task.approach_speed(
                    contact.frame[selected, :3], sign[selected], contact.pos[selected], self.data.cvel[body],
                    self.data.subtree_com[self.model.body_rootid[body]]))
        return speed

    def _observation(self, history='replace'):
        """The walking task's actor inputs (the command 0, the phase stopped), then task.getup_critic_extras. Keeps the
        newest frame's joint_pos_rel as the actor reads it (observed_position, pd_bound's angles)."""
        actor = super()._observation(history)[:self.actor_inputs]
        self.observed_position = actor[self.position_columns].astype(np.float64)
        extras = task.getup_critic_extras(self._velocity()[None], self.data.qpos[2:3], self._head(),
                                          np.array([self.assist]))
        if self.clip is not None:  # track_clip: the clip's phase and the anchor's errors, after the six
            return np.concatenate((actor, extras[0].astype(np.float32), self._clip_extras()[0].astype(np.float32)))
        return np.concatenate((actor, extras[0].astype(np.float32)))

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if self.clip is not None:
            return self._clip_reset(options)
        if self.settle is not None:
            return self._settle_reset(options)
        bank = self.bank
        # The episode that just ended steps this world's assist down (assist_update); a set force replaces it. An
        # episode from a human start (human_starts) steps neither the assist nor the bound.
        head_mean = np.array([self._head_mean()])
        curriculum = self.human_start is None
        if 'assist_force' in options:
            force = options['assist_force']
            if isinstance(force, bool) or not isinstance(force, (int, float, np.floating)) or not np.isfinite(force) \
                    or force < 0:
                raise ValueError('assist_force must be a finite non-negative number')
            self.assist = float(force)
        elif curriculum:
            self.assist = float(task.assist_update(np.array([self.assist]), head_mean)[0])
        # pd_bound: the same for this world's bound (pd_bound_update, no draw); a set one replaces it.
        if 'pd_bound' in options:
            self.set_pd_bound(options['pd_bound'])
        elif self.pd_bound is not None and self.bound is not None and curriculum:
            self.bound = float(task.pd_bound_update(np.array([self.bound]), head_mean, self.pd_bound[1])[0])
        human = None
        if 'start' in options:
            row = options['start']
            if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(bank['qpos']):
                raise ValueError('start must be a row of the start bank')
        elif 'human_start' in options:  # a row of the human starts (tests, probes), also without the setting
            row, human = -1, options['human_start']
            if isinstance(human, bool) or not isinstance(human, (int, np.integer)) \
                    or not 0 <= human < len(self.human['qpos']):
                raise ValueError('human_start must be a row of the human starts')
        else:
            row = int(self.training_rows[self.np_random.integers(len(self.training_rows))])
            # human_starts: a training reset (no start, assist force or bound set: evaluation and probes set them)
            # starts from a human row with probability share, drawn after the bank row; nothing drawn without it.
            if self.human_starts is not None and 'assist_force' not in options and 'pd_bound' not in options \
                    and self.np_random.random() < self.human_starts['share']:
                row, human = -1, int(self.np_random.integers(len(self.human['qpos'])))
        self.human_start = None if human is None else int(human)
        if human is None:
            self.start_row, self.start_class = int(row), bank['classes'][int(bank['start_class'][row])]
        else:
            self.start_row, self.start_class = -1, 'human'
        # Domain randomization, drawn after the start: delay (substeps), motor friction multiplier, body friction.
        if 'action_delay_steps' in options:
            delay_steps = options['action_delay_steps']
        else:
            delay_steps = int(self.np_random.integers(self.max_delay_steps + 1)) if self.max_delay_steps else 0
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, (int, np.integer)) or delay_steps < 0:
            raise ValueError('action_delay_steps must be a non-negative integer')
        self.delay_steps = int(delay_steps)
        self.delay_lags = task.action_lags(_tensor(self.delay_steps), self.frame_skip)[0].numpy()
        self.delay_actions = [np.zeros(self.model.nu)] * int(self.delay_lags[0])
        if 'motor_friction_scale' in options:
            scale = options['motor_friction_scale']
        else:
            scale = float(task.motor_friction_draw(lambda n: _uniform(self, n), 1, self.motor_friction)[0]) \
                if self.motor_friction else 1.
        low, high = self.body_friction
        friction = options['body_friction'] if 'body_friction' in options else low + (high - low) * self.np_random.random()
        mujoco.mj_resetData(self.model, self.data)
        self.set_motor_friction(scale)
        self.set_body_friction(friction)
        starts = bank if human is None else self.human
        self.data.qpos[:] = starts['qpos'][row if human is None else human]
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        self.data.qvel[:] = starts['qvel'][row if human is None else human]
        mujoco.mj_forward(self.model, self.data)
        # No command, no perturbation, no pushes: the walking task's state at rest.
        self.command[:] = 0
        self.fixed_command = True
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        self.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.phase_steps = 0
        self.perturb_live, self.perturb_steps, self.perturb_action = False, 0, np.zeros(self.model.nu)
        self.push_live, self.push_timer, self.push_left, self.push_force = False, 0, 0, np.zeros(3)
        self.height = task.BASE_HEIGHT
        self.steps = 0
        self.previous_action[:] = 0
        if self.previous_unbounded is not None:
            self.previous_unbounded[:] = 0
        self.lift = 0.
        self.head_window.clear()
        self.getup_at, self.getup_held, self.hold_steps = None, None, 0
        self.head = float(self._head()[0])
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    assist_force=self.assist, body_friction=self.friction, head_height=self.head, pd_bound=self.bound)
        if self.human_start is not None:
            info['human_start'] = self.human_start
        return self._observation('fill'), info

    def _randomize(self, options):
        """This episode's delay (substeps), motor friction multiplier and body friction, drawn in that order unless
        options set them (the getting-up reset's draws, for track_clip's reset); mj_resetData, then the frictions set."""
        if 'action_delay_steps' in options:
            delay_steps = options['action_delay_steps']
        else:
            delay_steps = int(self.np_random.integers(self.max_delay_steps + 1)) if self.max_delay_steps else 0
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, (int, np.integer)) or delay_steps < 0:
            raise ValueError('action_delay_steps must be a non-negative integer')
        self.delay_steps = int(delay_steps)
        self.delay_lags = task.action_lags(_tensor(self.delay_steps), self.frame_skip)[0].numpy()
        self.delay_actions = [np.zeros(self.model.nu)] * int(self.delay_lags[0])
        if 'motor_friction_scale' in options:
            scale = options['motor_friction_scale']
        else:
            scale = float(task.motor_friction_draw(lambda n: _uniform(self, n), 1, self.motor_friction)[0]) \
                if self.motor_friction else 1.
        low, high = self.body_friction
        friction = options['body_friction'] if 'body_friction' in options else low + (high - low) * self.np_random.random()
        mujoco.mj_resetData(self.model, self.data)
        self.set_motor_friction(scale)
        self.set_body_friction(friction)

    def _clip_reset(self, options):
        """track_clip's reset: the start (a bank row, a clip frame or a training draw, TRACK_START_SHARES), its noise
        (task.track_start_pose / track_start_speed; then raised out of the floor by its deepest contact), the
        reference's alignment (a bank row: task.track_align; a clip start: none) and the clock at its frame; the
        domain randomization as the getting-up reset's; no assist unless a probe sets one, no curriculum."""
        bank = self.bank
        if 'assist_force' in options:
            force = options['assist_force']
            if isinstance(force, bool) or not isinstance(force, (int, float, np.floating)) or not np.isfinite(force) \
                    or force < 0:
                raise ValueError('assist_force must be a finite non-negative number')
            self.assist = float(force)
        if 'pd_bound' in options:
            self.set_pd_bound(options['pd_bound'])
        noisy = options.get('track_noise', True)
        if type(noisy) is not bool:
            raise ValueError('track_noise must be true or false')
        row, frame = -1, 0
        if 'start' in options:
            kind, row = 'bank', options['start']
            if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(bank['qpos']):
                raise ValueError('start must be a row of the start bank')
        elif 'track_frame' in options:
            kind, frame = 'frame', options['track_frame']
            if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame <= self.clip_last:
                raise ValueError('track_frame must be a frame of the clip')
        elif 'assist_force' in options or 'pd_bound' in options:  # evaluation, probes: the clip's first frame
            kind = 'clip'
        else:  # a training reset draws its start
            draw = self.np_random.random()
            shares = np.cumsum(task.TRACK_START_SHARES)
            kind = task.TRACK_START_KINDS[int(np.searchsorted(shares, draw, side='right'))]
            if kind == 'frame':
                frame = int(self.np_random.integers(self.clip_last + 1))
            elif kind == 'bank':
                if self.clip_rows is None:
                    self.clip_rows = np.flatnonzero(~bank['holdout'] & (
                        bank['start_class'] == bank['classes'].index(task.TRACK_BANK_CLASS)))
                row = int(self.clip_rows[self.np_random.integers(len(self.clip_rows))])
        noise = None
        if kind != 'bank' and noisy:
            noise = (self.np_random.uniform(-1., 1., task.TRACK_POSE_NOISE),
                     self.np_random.uniform(-1., 1., task.TRACK_SPEED_NOISE))
        self._randomize(options)
        if kind == 'bank':
            qpos, qvel = bank['qpos'][row].copy(), bank['qvel'][row].copy()
            qpos[3:7] /= np.linalg.norm(qpos[3:7])
        else:
            qpos = self.clip['qpos'][frame].copy()
            qvel = self.clip['qvel'][frame].copy() if kind == 'frame' else np.zeros(self.model.nv)
            if noise is not None:
                qpos = task.track_start_pose(qpos[None], noise[0][None], self.range_low, self.range_high)[0]
                qvel = task.track_start_speed(qvel[None], noise[1][None])[0]
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, self.data)
        raised = 0.
        if kind != 'bank':  # out of the floor by the deepest contact the noise (or the clip) left
            raised = -g1_getup_track.floor_depth(self.model, self.data, self.robot_geoms, self.floor)
            if raised > 0:
                self.data.qpos[2] += raised
                mujoco.mj_forward(self.model, self.data)
        if kind == 'bank':
            yaw, offset = task.track_align(self.data.xpos[self.pelvis][None],
                                           self.data.xmat[self.pelvis].reshape(1, 3, 3), self.clip_frames['pos'][0, 0],
                                           self.clip_frames['rot'][0, 0])
            self.clip_yaw, self.clip_offset = float(yaw[0]), offset[0].copy()
        else:
            self.clip_yaw, self.clip_offset = 0., np.zeros(2)
        self.clip_clock, self.clip_frame, self.clip_start = int(frame), int(frame), kind
        self.clip_reached = self.clip_clock >= self.clip_last
        self.human_start = None
        if kind == 'bank':
            self.start_row, self.start_class = int(row), bank['classes'][int(bank['start_class'][row])]
        else:
            self.start_row, self.start_class = -1, kind
        self.command[:] = 0
        self.fixed_command = True
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        self.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.phase_steps = 0
        self.perturb_live, self.perturb_steps, self.perturb_action = False, 0, np.zeros(self.model.nu)
        self.push_live, self.push_timer, self.push_left, self.push_force = False, 0, 0, np.zeros(3)
        self.height = task.BASE_HEIGHT
        self.steps = 0
        self.previous_action[:] = 0
        if self.previous_unbounded is not None:
            self.previous_unbounded[:] = 0
        self.lift = 0.
        self.head_window.clear()
        self.getup_at, self.getup_held, self.hold_steps = None, None, 0
        self.head = float(self._head()[0])
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    assist_force=self.assist, body_friction=self.friction, head_height=self.head, pd_bound=self.bound,
                    track_start=kind, track_frame=self.clip_frame, track_lift=raised)
        return self._observation('fill'), info

    def _clip_step(self, action):
        """track_clip's step: the getting-up step's action, delay queue, PD and substeps (the assist only if a probe set
        one), each substep's impact / sole_impact speeds while weighted, its self-contact force (self_contact) and the
        floor's force on the hands; then the clock on, the tracking errors against the reference there
        (task.track_errors), the reward (task.track_reward_components), mjlab's terminations (task.track_terminated)
        and, after the hold, the end."""
        unbounded = self.unbounded_action(action) if self.change_unbounded else None
        action = self.applied_action(action)
        target = self._target(action)
        if self.bound is not None and self.steps == 0 and self.delay_actions:
            self.delay_actions = [self.applied_action(np.zeros(self.model.nu))] * len(self.delay_actions)
        targets = [target, *(self._target(past) for past in self.delay_actions)]
        self.lift = float(task.assist_force(self.data.xmat[self.pelvis].reshape(1, 3, 3), np.array([self.assist]))[0])
        unstable = instabilities(self.data)
        hand = np.zeros(self.frame_skip)
        for substep, lag in enumerate(self.delay_lags):
            joint_velocity = self.data.qvel[self.vadr]
            torque = self.kp * (targets[lag] - self.data.qpos[self.qadr]) - self.kd * joint_velocity
            friction = {} if self._friction_scale is None else dict(friction_scale=self._friction_scale)
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                       *self._motor_tensors, **friction).numpy()
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            self.data.xfrc_applied[self.pelvis, 2] = self.lift
            mujoco.mj_step(self.model, self.data)
            if self.impact_substeps is not None:
                self.impact_substeps[substep] = self._impact_speeds()
            if self.sole_impact_substeps is not None:
                self.sole_impact_substeps[substep] = self._impact_speeds(self.sole_geoms)[self.feet]
            if self.self_force is not None:  # self_contact: this substep's largest robot-robot contact force
                self.self_force[substep] = self._self_contact_force()
            hand[substep] = self._hand_force()
        self.data.xfrc_applied[self.pelvis, 2] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.clip_clock += 1
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or instabilities(self.data) != unstable):
            info = dict(x_position=0., bad_state=True, start_class=self.start_class, reward_components={},
                        getup_seconds=self.getup_at, getup_held=False, assist_force=self.assist,
                        track_start=self.clip_start, track_done=self.clip_reached, track_failed=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0., True, False, info
        robot = self._clip_robot()
        reference = self._clip_reference()
        errors = task.track_errors(*robot, *reference, self.track_anchor)
        impact = None if self.impact_substeps is None else task.impact_penalty(self.impact_substeps.max(0)[None])
        sole_impact = None if self.sole_impact_substeps is None else task.impact_penalty(
            self.sole_impact_substeps.max(0)[None], task.GETUP_SOLE_IMPACT_SPEED)
        self_contact = None if self.self_force is None else task.self_contact_share(
            torch.from_numpy(self.self_force[None].copy())).numpy()
        components = task.track_reward_components(
            errors, action[None], self.previous_action[None], self.data.qpos[self.qadr][None], self.soft_low,
            self.soft_high, self.track_weights, self_contact=self_contact, impact=impact, sole_impact=sole_impact)
        components = {key: float(value[0]) for key, value in components.items()}
        grace = self.clip_start == 'bank' and self.steps <= self.clip_grace  # a bank start's limbs: not yet
        failed = bool(task.track_terminated(robot[0], robot[1], reference[0], reference[1], self.track_anchor,
                                            self.track_ends, np.array([not grace]))[0])
        self.clip_reached = self.clip_reached or self.clip_clock >= self.clip_last
        head = self._head()
        height, rotation = self.data.qpos[2:3], self.data.xmat[self.torso].reshape(1, 3, 3)
        self.head = float(head[0])
        self.head_window.append(self.head)
        soles = self._sole_height().numpy()
        self._track(bool(task.getup_success_pose(head, height, rotation, soles)[0]),
                    bool(task.getup_success_pose(head, height, rotation)[0]))
        standing = bool(task.getup_standing(head, height, rotation, soles)[0])  # track_standing, at the episode's end
        self.previous_action[:] = action
        if unbounded is not None:
            self.previous_unbounded[:] = unbounded
        if self.delay_actions:
            self.delay_actions = [action, *self.delay_actions[:-1]]
        terminated = failed or bool(np.abs(self.data.qvel[self.vadr]).max() > task.GETUP_MAX_JOINT_SPEED)
        truncated = self.steps >= self.max_episode_steps or self.clip_clock >= self.clip_end
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    head_height=self.head, pelvis_height=float(height[0]), assist_force=self.assist, lift=self.lift,
                    getup_seconds=self.getup_at, getup_held=self.getup_held, pd_bound=self.bound,
                    reward_components=components, track_start=self.clip_start, track_clock=self.clip_clock,
                    track_done=self.clip_reached, track_failed=failed, track_standing=standing,
                    hand_force=float(hand.max()),
                    track_errors={key: float(np.sqrt(value[0])) for key, value in errors.items()})
        return self._observation('push'), sum(components.values()), terminated, truncated, info

    def settle_errors(self):
        """(angle, pose, deviation), each (1, K): the current state against settle_clip's targets (task.settle_errors:
        the pelvis gravity angle, mean (q − q0)² and mean |q − q0| over the free joints). Probes read it too."""
        return task.settle_errors(task.projected_gravity(self.data.xmat[self.pelvis].reshape(1, 3, 3)),
                                  self.data.qpos[self.qadr][None], self.settle['gravity'], self.settle['position'],
                                  self.settle_joints)

    def _settle_reset(self, options):
        """settle_clip's reset: a bank row (option start, any row; else a training row of SETTLE_CLASSES), the domain
        randomization as the getting-up reset's (_randomize), no assist unless a probe sets one, no curriculum."""
        bank = self.bank
        if 'assist_force' in options:
            force = options['assist_force']
            if isinstance(force, bool) or not isinstance(force, (int, float, np.floating)) or not np.isfinite(force) \
                    or force < 0:
                raise ValueError('assist_force must be a finite non-negative number')
            self.assist = float(force)
        if 'pd_bound' in options:
            self.set_pd_bound(options['pd_bound'])
        if 'start' in options:
            row = options['start']
            if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(bank['qpos']):
                raise ValueError('start must be a row of the start bank')
        else:
            if self.settle_rows is None:
                classes = [bank['classes'].index(kind) for kind in task.SETTLE_CLASSES if kind in bank['classes']]
                self.settle_rows = np.flatnonzero(~bank['holdout'] & np.isin(bank['start_class'], classes))
            row = int(self.settle_rows[self.np_random.integers(len(self.settle_rows))])
        self._randomize(options)
        qpos = bank['qpos'][row].copy()
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        self.data.qpos[:], self.data.qvel[:] = qpos, bank['qvel'][row]
        mujoco.mj_forward(self.model, self.data)
        self.human_start = None
        self.start_row, self.start_class = int(row), bank['classes'][int(bank['start_class'][row])]
        self.command[:] = 0
        self.fixed_command = True
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        self.heading_target = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.phase_steps = 0
        self.perturb_live, self.perturb_steps, self.perturb_action = False, 0, np.zeros(self.model.nu)
        self.push_live, self.push_timer, self.push_left, self.push_force = False, 0, 0, np.zeros(3)
        self.height = task.BASE_HEIGHT
        self.steps = 0
        self.previous_action[:] = 0
        if self.previous_unbounded is not None:
            self.previous_unbounded[:] = 0
        self.lift = 0.
        self.head_window.clear()
        self.getup_at, self.getup_held, self.hold_steps = None, None, 0
        self.settle_count, self.settle_at, self.settle_target = 0, None, None
        self.head = float(self._head()[0])
        angle, pose, deviation = self.settle_errors()
        nearest = int(task.settle_nearest(angle, pose)[0])
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    assist_force=self.assist, body_friction=self.friction, head_height=self.head, pd_bound=self.bound,
                    settle_angle=float(np.degrees(angle[0, nearest])), settle_deviation=float(deviation[0, nearest]))
        return self._observation('fill'), info

    def _settle_step(self, action):
        """settle_clip's step: the getting-up step's action, delay queue, PD and substeps (the assist only if a probe set
        one), each substep's impact / sole_impact speeds while weighted, its self-contact force (self_contact) and the
        floor's force on the hands; then the errors against every target (settle_errors), the reward of the nearest
        (task.settle_reward_components), the success clock (task.settle_update) and the episode end (no early end but
        a non-finite state or a joint above GETUP_MAX_JOINT_SPEED; the episode goes on after the success)."""
        unbounded = self.unbounded_action(action) if self.change_unbounded else None
        action = self.applied_action(action)
        target = self._target(action)
        if self.bound is not None and self.steps == 0 and self.delay_actions:
            self.delay_actions = [self.applied_action(np.zeros(self.model.nu))] * len(self.delay_actions)
        targets = [target, *(self._target(past) for past in self.delay_actions)]
        self.lift = float(task.assist_force(self.data.xmat[self.pelvis].reshape(1, 3, 3), np.array([self.assist]))[0])
        unstable = instabilities(self.data)
        hand = np.zeros(self.frame_skip)
        for substep, lag in enumerate(self.delay_lags):
            joint_velocity = self.data.qvel[self.vadr]
            torque = self.kp * (targets[lag] - self.data.qpos[self.qadr]) - self.kd * joint_velocity
            friction = {} if self._friction_scale is None else dict(friction_scale=self._friction_scale)
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                       *self._motor_tensors, **friction).numpy()
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            self.data.xfrc_applied[self.pelvis, 2] = self.lift
            mujoco.mj_step(self.model, self.data)
            if self.impact_substeps is not None:
                self.impact_substeps[substep] = self._impact_speeds()
            if self.sole_impact_substeps is not None:
                self.sole_impact_substeps[substep] = self._impact_speeds(self.sole_geoms)[self.feet]
            if self.self_force is not None:  # self_contact: this substep's largest robot-robot contact force
                self.self_force[substep] = self._self_contact_force()
            hand[substep] = self._hand_force()
        self.data.xfrc_applied[self.pelvis, 2] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or instabilities(self.data) != unstable):
            info = dict(x_position=0., bad_state=True, start_class=self.start_class, reward_components={},
                        getup_seconds=None, getup_held=False, assist_force=self.assist,
                        settle_done=self.settle_at is not None, settle_seconds=self.settle_at,
                        settle_target=self.settle_target)
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0., True, False, info
        position, joint_velocity = self.data.qpos[self.qadr][None], self.data.qvel[self.vadr][None]
        angle, pose, deviation = self.settle_errors()
        impact = None if self.impact_substeps is None else task.impact_penalty(self.impact_substeps.max(0)[None])
        sole_impact = None if self.sole_impact_substeps is None else task.impact_penalty(
            self.sole_impact_substeps.max(0)[None], task.GETUP_SOLE_IMPACT_SPEED)
        self_contact = None if self.self_force is None else task.self_contact_share(
            torch.from_numpy(self.self_force[None].copy())).numpy()
        components, nearest, near = task.settle_reward_components(
            angle, pose, deviation, joint_velocity, self._velocity()[None], action[None], self.previous_action[None],
            position, self.soft_low, self.soft_high, self.settle_joints, self.settle_weights, self_contact=self_contact,
            impact=impact, sole_impact=sole_impact)
        components = {key: float(value[0]) for key, value in components.items()}
        nearest = int(nearest[0])
        self.settle_count = int(task.settle_update(np.array([self.settle_count]), near, joint_velocity,
                                                   self.settle_joints)[0])
        if self.settle_at is None and self.settle_count >= self.settle_hold:
            self.settle_at, self.settle_target = self.steps * self.dt, self.settle['names'][nearest]
        self.head = float(self._head()[0])
        self.head_window.append(self.head)
        self.previous_action[:] = action
        if unbounded is not None:
            self.previous_unbounded[:] = unbounded
        if self.delay_actions:
            self.delay_actions = [action, *self.delay_actions[:-1]]
        terminated = bool(np.abs(self.data.qvel[self.vadr]).max() > task.GETUP_MAX_JOINT_SPEED)
        truncated = self.steps >= self.max_episode_steps
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    head_height=self.head, pelvis_height=float(self.data.qpos[2]), assist_force=self.assist,
                    lift=self.lift, getup_seconds=None, getup_held=None, pd_bound=self.bound,
                    reward_components=components, hand_force=float(hand.max()), settle_nearest=nearest,
                    settle_angle=float(np.degrees(angle[0, nearest])), settle_deviation=float(deviation[0, nearest]),
                    settle_near=bool(near[0]), settle_count=self.settle_count, settle_done=self.settle_at is not None,
                    settle_seconds=self.settle_at, settle_target=self.settle_target)
        return self._observation('push'), sum(components.values()), terminated, truncated, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid action shape or non-finite action')
        if self.clip is not None:
            return self._clip_step(action)
        if self.settle is not None:
            return self._settle_step(action)
        # action_change_unbounded: the action_change term's action, before pd_bound (box clip, lock_wrists).
        unbounded = self.unbounded_action(action) if self.change_unbounded else None
        # arm_press: the same action before pd_bound, whose PD target the term compares with the arms' angles.
        pressed = None if not self.press_weighted else (
            unbounded if unbounded is not None else self.unbounded_action(action))
        # The applied action (box clip, pd_bound, lock_wrists) replaces the policy's in all below, as in walking.
        action = self.applied_action(action)
        target = self._target(action)
        # sole_slip: the feet's horizontal positions before the step, only while weighted.
        feet_before = self.data.xpos[self.feet, :2].copy() if self.slip_weighted else None
        if self.bound is not None and self.steps == 0 and self.delay_actions:
            # pd_bound: the actions still in flight at the episode's first step are the zero action bounded around the
            # start pose as observed (unbounded, home: a full-torque jerk from lying under the delay).
            self.delay_actions = [self.applied_action(np.zeros(self.model.nu))] * len(self.delay_actions)
        targets = [target, *(self._target(past) for past in self.delay_actions)]
        # Assist: this world's force while the pelvis is nearly upright (task.assist_force), on every substep.
        self.lift = float(task.assist_force(self.data.xmat[self.pelvis].reshape(1, 3, 3), np.array([self.assist]))[0])
        unstable = instabilities(self.data)
        for substep, lag in enumerate(self.delay_lags):
            joint_velocity = self.data.qvel[self.vadr]
            torque = self.kp * (targets[lag] - self.data.qpos[self.qadr]) - self.kd * joint_velocity
            friction = {} if self._friction_scale is None else dict(friction_scale=self._friction_scale)
            torque = task.motor_torque(torch.from_numpy(torque), torch.from_numpy(joint_velocity.copy()),
                                       *self._motor_tensors, **friction).numpy()
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            self.data.xfrc_applied[self.pelvis, 2] = self.lift
            mujoco.mj_step(self.model, self.data)
            if self.impact_substeps is not None:  # impact: this substep's floor contacts, only while weighted
                self.impact_substeps[substep] = self._impact_speeds()
            if self.sole_impact_substeps is not None:  # sole_impact: the soles' ones per foot, only while weighted
                self.sole_impact_substeps[substep] = self._impact_speeds(self.sole_geoms)[self.feet]
        self.data.xfrc_applied[self.pelvis, 2] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        # A non-finite state, or one MuJoCo had to reset (an instability warning), ends the episode.
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or instabilities(self.data) != unstable):
            info = dict(x_position=0., bad_state=True, start_class=self.start_class, reward_components={},
                        getup_seconds=self.getup_at, getup_held=False, assist_force=self.assist)
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0., True, False, info
        velocity = self._velocity()
        head = self._head()
        position, rotation = self.data.qpos[self.qadr][None], self.data.xmat[self.torso].reshape(1, 3, 3)
        height = self.data.qpos[2:3]
        sole = self._sole_height().numpy()
        # impact: per body its largest approach speed over the substeps (task.impact_penalty), only while weighted.
        impact = None if self.impact_substeps is None else task.impact_penalty(self.impact_substeps.max(0)[None])
        # sole_impact: per foot its largest sole approach speed over the substeps, above GETUP_SOLE_IMPACT_SPEED.
        sole_impact = None if self.sole_impact_substeps is None else task.impact_penalty(
            self.sole_impact_substeps.max(0)[None], task.GETUP_SOLE_IMPACT_SPEED)
        # stance_width: the feet's width across the pelvis heading after the step, only while weighted.
        width = None if not self.stance_weighted else task.stance_width(
            self.data.xpos[self.feet][None], self.data.xmat[self.pelvis].reshape(1, 3, 3))
        # arm_press: Σ over the free arm joints of (target before pd_bound − angle after the step)², while weighted.
        press = None if pressed is None else task.arm_press(
            torch.from_numpy(self._target(pressed))[None], torch.from_numpy(self.data.qpos[self.qadr].copy())[None],
            self.press_joints).numpy()
        # style: the distance of the frame after the step to the nearest library frame, while weighted, computed only
        # with the head in the band, where the term pays (outside it d is +inf: the term is 0 either way).
        style = nearest = None
        band = self.style is not None and bool(task.style_band(head)[0])
        if band:
            style, nearest = task.style_distance(self.style_features(), self.style['features'], self.style['norms'])
        elif self.style is not None:
            style = np.full(1, np.inf)
        # head_rise: the head top's vertical speed over the step (self.head: its height after the step before, or at
        # the reset), only while weighted.
        head_speed = (head - self.head) / self.dt if self.getup_weights['head_rise'] else None
        # sole_slip: each foot's horizontal speed over the step (foot_speed), Σ of its square over the feet in contact.
        slip = foot_speed = None
        if feet_before is not None:
            foot_speed = np.linalg.norm(self.data.xpos[self.feet, :2] - feet_before, axis=1)[None] / self.dt
            slip = task.sole_slip(foot_speed, sole)
        # action_change: the applied action and the one before, under action_change_unbounded the action before
        # pd_bound and its own previous one.
        changed, before = (action, self.previous_action) if unbounded is None else (unbounded, self.previous_unbounded)
        components, standing = task.getup_reward_components(
            head, height, rotation, velocity[None], position, self.home, self.soft_low, self.soft_high,
            range(task.LEGS), sole, self.data.ctrl[None], self.data.qvel[self.vadr][None], changed[None],
            before[None], weights=self.getup_weights, impact=impact, stand_pose_std=self.stand_pose_std,
            sole_impact=sole_impact, stance_width=width, arm_press=press, style=style, head_speed=head_speed,
            stance_pelvis=self.stance_pelvis, sole_slip=slip)
        components = {key: float(value[0]) for key, value in components.items()}
        self.head = float(head[0])
        self.head_window.append(self.head)
        self._track(bool(task.getup_success_pose(head, height, rotation, sole)[0]),
                    bool(task.getup_success_pose(head, height, rotation)[0]))
        self.previous_action[:] = action
        if unbounded is not None:
            self.previous_unbounded[:] = unbounded
        if self.delay_actions:
            self.delay_actions = [action, *self.delay_actions[:-1]]
        terminated = bool(np.abs(self.data.qvel[self.vadr]).max() > task.GETUP_MAX_JOINT_SPEED)
        truncated = self.steps >= self.max_episode_steps
        info = dict(x_position=float(self.data.qpos[0]), start=self.start_row, start_class=self.start_class,
                    head_height=self.head, pelvis_height=float(height[0]), standing=bool(standing[0]),
                    assist_force=self.assist, lift=self.lift, getup_seconds=self.getup_at, getup_held=self.getup_held,
                    pd_bound=self.bound, reward_components=components)
        if width is not None:  # stance_width: the width it charged and whether it counted (task.stance_active)
            info.update(stance_width=float(width[0]), stance_active=bool(
                task.stance_active(height, sole, self.stance_pelvis)[0]))
        if press is not None:  # arm_press: the unweighted Σ, charged only standing
            info['arm_press'] = float(press[0])
        if head_speed is not None:  # head_rise: the head top's vertical speed it charged above GETUP_HEAD_RISE_SPEED
            info['head_speed'] = float(head_speed[0])
        if slip is not None:  # sole_slip: the unweighted Σ and each foot's speed
            info.update(sole_slip=float(slip[0]), foot_speed=foot_speed[0].tolist())
        if style is not None:  # style: whether the head was in the band, there d and the nearest library frame
            info['style_band'] = band
            if band:
                info.update(style_dist=float(style[0]), style_frame=int(nearest[0]))
        return self._observation('push'), sum(components.values()), terminated, truncated, info

    def _track(self, up, holding):
        """Evaluation's success: up (task.getup_success_pose with both feet) within GETUP_SUCCESS_WITHIN seconds, then
        GETUP_SUCCESS_HOLD seconds holding (the pose without the feet); getup_held True / False once decided."""
        seconds = self.steps * self.dt
        if self.getup_at is None:
            if up and seconds <= task.GETUP_SUCCESS_WITHIN + 1e-9:
                self.getup_at = seconds
            elif seconds > task.GETUP_SUCCESS_WITHIN:
                self.getup_held = False
        elif self.getup_held is None:
            if not holding:
                self.getup_held = False
            else:
                self.hold_steps += 1
                if self.hold_steps >= round(task.GETUP_SUCCESS_HOLD / self.dt):
                    self.getup_held = True


def saved_bounds(state):
    """The worlds' PD error bounds in a stage's training_state (task_config pd_bound), (n,) float64, checked: finite and
    positive (the CUDA task's restore_training_state too)."""
    bounds = np.asarray(state['pd_bound'], dtype=np.float64)
    if bounds.ndim != 1 or not len(bounds) or not np.isfinite(bounds).all() or (bounds <= 0).any():
        raise ValueError('Invalid saved PD error bounds')
    return bounds


def _tensor(value):
    return torch.tensor([value])


def _uniform(env, n):
    return torch.from_numpy(env.np_random.random(n))
