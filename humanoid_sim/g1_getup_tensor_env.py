"""g1_getup on CUDA (iteration 33, part B): the CPU task tasks/g1_getup.G1GetupEnv batched in MuJoCo Warp.

TensorG1Env's physics (the getting-up model: real robot with Unitree's foot and hand capsules), per-substep PD, motor
model, action delay queue and observation (the walking connector, command 0, the gait phase stopped), with the task of
G1GetupEnv in place of walking: every world starts from a training row of the start bank (g1_getup_bank, the classes in
their bank shares, never a held-out row), episodes of 500 control steps ended early only by a non-finite state or a
joint faster than 50 rad/s, the reward and the assist of g1_tensor_math (getup_reward_components, assist_force,
assist_update), the critic's extras getup_critic_extras. Per world: its assist force (written into the pelvis
xfrc_applied before every substep's graph launch, which reads that array in place, as force_push's pushes), stepped
down at its reset after an episode up; its sliding friction of every colliding robot geom and of the floor, drawn at its
reset (put_model's batched geom_friction, read in place by the collision kernels); its delay and motor friction
multiplier, as the walking task's. No command, pushes, arm perturbation or self_contact. With the real robot's
training realism (observation_noise, randomize; as every real-robot CUDA run): sensor noise and encoder bias, and the
pelvis COM offset per world (the walking task's foot friction randomisation is overwritten by the body friction).
With task_config pd_bound (iteration 34) each world's bound c keeps every motor target but the locked wrists' within
c·y1/kp of the angles its policy observed (the newest history frame, noise and bias included; task.pd_bound), c stepping
down at its reset as the assist does (task.pd_bound_update). With the reward weight impact (iteration 35) every
substep's floor contacts give each world's bodies their approach speeds (task.approach_speed), the step's largest ones
charged by task.impact_penalty. task_config stand_pose_std (iteration 36): the standing pose term's σ, as on the CPU.
task_config action_change_unbounded (iteration 37): the action_change term charges each world's action before pd_bound
(the box clip and the locked wrists) against its own one of the step before, as on the CPU. With the reward weight
sole_impact (iteration 44) the same pass over the floor contacts gives each world's feet their sole approach speeds,
charged above GETUP_SOLE_IMPACT_SPEED, and counts the world-steps in a flight (rollout/flight). The reward weight
stance_width (after branch 46): each world's feet width after the step (task.stance_width), as on the CPU. arm_press
(iteration 52): standing, each world's free arm joints' target before pd_bound against their angles (task.arm_press).
style (iteration 53): each world's distance to the nearest human frame of the CPU task's style library, one (n_envs,
library rows) product per step (task.style_distance), paid while its head is in task.STYLE_HEAD_BAND; with
task_config style_judge or the amp weight computed and logged, not paid. amp (iteration 55): each world's pair of
states before and after the step (getup_amp.amp_state) and its band, for the learner's discriminator reward
(getup_amp.GetupAMP in TensorPPO.collect). task_config human_starts (iteration 56): a share of training resets starts a
world from a human frame (g1_getup_human), drawn with a generator of its own; such an episode steps neither the assist
nor the bound. head_rise (iteration 57): each world's head top speed over the step from its height a step before (at
its reset: the start's), charged above task.GETUP_HEAD_RISE_SPEED. sole_slip (iteration 62): each world's feet's
horizontal speed over the step, squared and summed over the feet in contact (task.sole_slip). task_config track_clip
(iteration 64): motion tracking of one human clip (the CPU task's): per world its clip clock, start kind and the
reference's yaw and offset, the starts of task.TRACK_START_SHARES drawn at its reset (noisy ones raised out of the floor
by their deepest contact, read from Warp's contact distances), the tracking reward, terminations and critic inputs of
g1_tensor_math, every substep's contact forces decoded for self_contact and the hands' floor force (_clip_step).
task_config settle_clip (iteration 66): settling into the nearest of the clips' first frames from the bank's fall
classes (task.SETTLE_CLASSES): per world the success clock, its time and target, the settling reward of g1_tensor_math
and the same contact decode (_settle_step); no assist.

Episodes out of step: a full reset() cuts each world's first episode at a random step of 1..max_episode_steps
(episode_limit; rsl_rl's init_at_random_ep_len), every later one runs the whole length. Without it the worlds, which
hardly ever end early here, kept their episodes in lockstep: all of them ended together every 500 steps, so only about
one rollout in 21 (24 steps each) had an episode end, and the critic saw every world at the same point of its episode
(the first root, exp-9e0bafcc272d4b51: 42 of 290 rollouts with an ep_len_mean, 24.10).
"""
import math

import numpy as np
import torch

from . import getup_amp
from .g1_tensor_env import ENCODER_BIAS, TensorG1Env
from .tasks import g1_getup_style
from .tasks import g1_tensor_math as task
from .tasks.config import G1_GETUP_DEFAULTS, G1_GETUP_PROFILE
from .tasks.g1_getup import EPISODE_STEPS, HUMAN_START_STREAM, SETTLE_STEPS, G1GetupEnv, saved_bounds
from .tasks.g1_tensor_math import pd_torque


def floor_entries(reference, impact, sole_impact):
    """(ngeom,) int64, per geom of the CPU task `reference` (G1GetupEnv) its entry in a world's row of floor approach
    speeds (TensorG1GetupEnv._floor_speeds), −1 where no weighted term reads its floor contacts: with impact its body
    (0 … nbody − 1) for every robot geom but the sole spheres (impact_geoms), with sole_impact nbody + its foot (0 left,
    1 right) for the sole spheres (soles)."""
    model = reference.model
    entry = np.full(model.ngeom, -1, dtype=np.int64)
    if impact:
        geoms = np.flatnonzero(reference.impact_geoms)
        entry[geoms] = model.geom_bodyid[geoms]
    if sole_impact:
        for foot, geoms in enumerate(reference.soles):
            entry[geoms] = model.nbody + foot
    return entry


class TensorG1GetupEnv(TensorG1Env):
    backend_metadata = dict(TensorG1Env.backend_metadata, task='g1_getup',
                            variant='CUDA reset RNG; float32; per-substep PD; start bank, assist and body friction '
                                    'per world')

    def __init__(self, n_envs, seed=0, max_episode_steps=EPISODE_STEPS, nconmax=128, njmax=512, observation_noise=None,
                 randomize=None, bank=None, reward_weights=None, assist_force=G1_GETUP_DEFAULTS['assist_force'],
                 action_delay=G1_GETUP_DEFAULTS['action_delay'], motor_friction=G1_GETUP_DEFAULTS['motor_friction'],
                 body_friction=G1_GETUP_DEFAULTS['body_friction'], pd_bound=G1_GETUP_DEFAULTS['pd_bound'],
                 stand_pose_std=G1_GETUP_DEFAULTS['stand_pose_std'],
                 action_change_unbounded=G1_GETUP_DEFAULTS['action_change_unbounded'],
                 style_library=G1_GETUP_DEFAULTS['style_library'], style_judge=G1_GETUP_DEFAULTS['style_judge'],
                 human_starts=G1_GETUP_DEFAULTS['human_starts'], human_bank=None,
                 stance_pelvis=G1_GETUP_DEFAULTS['stance_pelvis'], hip_roll_out=G1_GETUP_DEFAULTS['hip_roll_out'],
                 track_clip=G1_GETUP_DEFAULTS['track_clip'], track_bank=None,
                 settle_clip=G1_GETUP_DEFAULTS['settle_clip'], settle_bank=None):
        """The CPU task's settings (task_config: reward_weights, assist_force, action_delay, motor_friction,
        body_friction, pd_bound, stand_pose_std, action_change_unbounded, style_library, style_judge, human_starts,
        stance_pelvis, hip_roll_out, track_clip, settle_clip; bank, human_bank, track_bank and settle_bank as
        G1GetupEnv's); observation_noise and randomize as TensorG1Env's (tests switch them off to compare with the CPU
        task)."""
        self.clip = None  # track_clip's reference frames (set up at the end), None while off
        self.settle = None  # settle_clip's targets (set up at the end), None while off
        self._getup = dict(reward_weights=reward_weights, assist_force=assist_force, action_delay=action_delay,
                           motor_friction=motor_friction, body_friction=list(body_friction), pd_bound=pd_bound,
                           stand_pose_std=stand_pose_std, action_change_unbounded=action_change_unbounded, bank=bank,
                           style_library=style_library, style_judge=style_judge, human_starts=human_starts,
                           human_bank=human_bank, stance_pelvis=stance_pelvis, hip_roll_out=hip_roll_out,
                           track_clip=track_clip, track_bank=track_bank, settle_clip=settle_clip,
                           settle_bank=settle_bank)
        if settle_clip is not None:  # settling episodes are shorter, as on the CPU task
            max_episode_steps = min(max_episode_steps, SETTLE_STEPS)
        super().__init__(n_envs, seed=seed, max_episode_steps=max_episode_steps, nconmax=nconmax, njmax=njmax,
                         resample_commands=False, push_velocity=0., observation_noise=observation_noise,
                         randomize=randomize, action_delay=action_delay, motor_friction=motor_friction,
                         **G1_GETUP_PROFILE)
        ref = self.reference
        device = self.device

        def constant(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=device)
        # The start bank on the device: every row (set_starts takes held-out ones too), training draws its own rows.
        bank = ref.bank
        self.bank_qpos, self.bank_qvel = constant(bank['qpos']), constant(bank['qvel'])
        self.training_rows = constant(ref.training_rows, torch.long)
        self.start_row = torch.zeros(n_envs, dtype=torch.long, device=device)
        # human_starts (iteration 56): the share of training resets that start from a human row (None: off, nothing
        # drawn), drawn with a generator of its own (seed + HUMAN_START_STREAM), so the task's own stream stays the
        # old one; each world's episode from a human start (human_episode), whose end steps neither the assist nor the
        # bound; the human rows on the device (_human_rows, loaded when first needed); the episodes ended since the
        # last rollout_metrics that started from one, and those from the bank ended up, for the logs.
        self.human_share = None if ref.human_starts is None else ref.human_starts['share']
        self.human_episode = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.human_qpos = self.human_qvel = None
        if self.human_share is not None:
            self._human_rows()
            self.human_generator = torch.Generator(device=device).manual_seed(seed + HUMAN_START_STREAM)
            self._episodes_human = torch.zeros((), device=device)
            self._episodes_bank_up = torch.zeros((), device=device)
        # Each world's episode length: max_episode_steps, the first episode after a full reset() a random one.
        self.episode_limit = torch.full((n_envs,), self.max_episode_steps, dtype=torch.long, device=device)
        # Reward constants of the CPU task: the legs' soft joint limits, the weights, the pose term's σ (a float).
        self.soft_low, self.soft_high = constant(ref.soft_low), constant(ref.soft_high)
        self.limit_joints = range(task.LEGS)
        self.weights = ref.getup_weights
        self.stand_pose_std = ref.stand_pose_std
        self.stance_pelvis = ref.stance_pelvis  # the stance_width term's pelvis threshold (m, a float)
        # action_change_unbounded: each world's action before pd_bound of the step before (box clip, lock_wrists),
        # what the action_change term charges against, zeroed at its reset as previous_action; None while off.
        self.previous_unbounded = torch.zeros_like(self.previous_action) if ref.change_unbounded else None
        # Assist: each world's force (N) and this step's lift (assist_force of it, 0 unless nearly upright); the head
        # heights of its last GETUP_ASSIST_SECONDS (a ring indexed by the episode's step), for assist_update.
        self.assist = torch.full((n_envs,), float(ref.assist), device=device)
        self.lift = torch.zeros(n_envs, device=device)
        self.head_steps = round(task.GETUP_ASSIST_SECONDS / self.dt)
        self.head_window = torch.zeros((n_envs, self.head_steps), device=device)
        # Body friction: the CPU task's geoms (every colliding robot geom and the floor) in Warp's batched model.
        self.friction_geoms = constant(ref.friction_geoms, torch.long)
        self.geom_friction = self.wp.to_torch(self.model.geom_friction)  # (n_envs, ngeom, 3), read in place
        self.body_friction = tuple(float(v) for v in ref.body_friction)
        self.friction = torch.zeros(n_envs, device=device)
        # pd_bound: [start, lowest] or None; each world's bound c (bounding: whether the step applies it, also after
        # set_pd_bound without the setting), the motors' peaks y1, the joints it bounds (all but the locked wrists) and
        # where the newest history frame keeps joint_pos_rel (the observed angles, as the actor read them).
        self.pd_bound = ref.pd_bound
        self.bounding = self.pd_bound is not None
        self.bound = torch.full((n_envs,), 1. if self.pd_bound is None else self.pd_bound[0], device=device)
        self.peak = constant(ref.peak)
        self.bound_free = constant(ref.bound_free, torch.bool)
        self._free_joints = int(ref.bound_free.sum())
        start, end = task.deploy_terms(ref.model.nu, ref.extra_sizes)[task.JOINT_POSITION_TERM]
        self.position_entries = slice(start, end)
        # Bounded free joint-steps and all free joint-steps since the last rollout_metrics.
        self._bounded = torch.zeros((), device=device)
        self._bounded_count = 0
        # Contact and constraint demand (rollout_metrics): per world the most constraint rows (njmax per world) and
        # contacts (Warp's contact arrays are shared by all worlds, naconmax = nconmax·n_envs) of any substep.
        self.nefc = self.wp.to_torch(self.data.nefc)
        self.nacon = self.wp.to_torch(self.data.nacon)
        self.contact_world = self.wp.to_torch(self.data.contact.worldid)
        self._contact_slots = torch.arange(self.data.naconmax, device=device)
        self._contact_ones = torch.ones(self.data.naconmax, dtype=torch.long, device=device)
        self._nefc_max = torch.zeros((), dtype=self.nefc.dtype, device=device)
        self._ncon_max = torch.zeros((), dtype=torch.long, device=device)
        # impact and sole_impact reward weights (only while one is weighted): after each physics step every contact slot
        # between the floor and a robot geom a weighted term reads (task.impact_contacts on floor_geoms: impact's, every
        # robot geom but the sole spheres; sole_impact's, the sole spheres) gives the approach speed of its point
        # (task.approach_speed from the body's cvel and its root's subtree_com: the state the step started from, as the
        # CPU task reads them after mj_step), and Torch keeps, per world and entry (floor_entries: impact's per body,
        # then sole_impact's per foot), the largest one, both terms in one pass (scatter_reduce amax, as self_contact's
        # forces; stale slots past nacon and other pairs go to spare entries, one per slot, so the atomics of the ~95 %
        # dead slots do not queue on one address; no host sync). impact_substeps (n_envs, frame_skip, nbody) and
        # sole_impact_substeps (n_envs, frame_skip, 2) for the current step; the penalties and the world-steps above
        # their speeds are summed for rollout_metrics. With sole_impact also, per world, whether any of its geoms
        # touched the floor in any substep of the step (touching) and the steps since one last did (airborne), for
        # rollout/flight.
        self.impact_substeps = self.sole_impact_substeps = self.floor_entry = None
        impact, sole_impact = bool(self.weights['impact']), bool(self.weights['sole_impact'])
        if impact or sole_impact:
            self.floor = int(ref.floor)
            entry = floor_entries(ref, impact, sole_impact)
            self.floor_entry = constant(entry, torch.long)
            self.floor_geoms = constant(entry >= 0, torch.bool)
            self.body_of_geom = constant(ref.model.geom_bodyid, torch.long)
            self.root_of_body = constant(ref.model.body_rootid, torch.long)
            self.nbody = int(ref.model.nbody)
            self.floor_width = self.nbody + len(ref.soles)  # a world's entries: its bodies, then its feet
            self.contact_geom = self.wp.to_torch(self.data.contact.geom)  # (naconmax, 2)
            self.contact_pos = self.wp.to_torch(self.data.contact.pos)  # (naconmax, 3)
            self.contact_frame = self.wp.to_torch(self.data.contact.frame)  # (naconmax, 3, 3), the normal first
            self._floor_flat = torch.zeros(n_envs * self.floor_width + self.data.naconmax, device=device)
            self._floor_spare = n_envs * self.floor_width + self._contact_slots  # each slot's own spare entry
        if impact:
            self.impact_substeps = torch.zeros((n_envs, ref.frame_skip, self.nbody), device=device)
            self._impact_sum = torch.zeros((), device=device)
            self._impact_hits = torch.zeros((), device=device)
            self._impact_count = 0
        if sole_impact:
            self.sole_impact_substeps = torch.zeros((n_envs, ref.frame_skip, len(ref.soles)), device=device)
            self._sole_impact_sum = torch.zeros((), device=device)
            self._sole_impact_hits = torch.zeros((), device=device)
            self._sole_impact_count = 0
            self.touching = torch.zeros(n_envs, dtype=torch.bool, device=device)
            self.airborne = torch.zeros(n_envs, dtype=torch.long, device=device)
            self.flight_steps = max(1, round(task.GETUP_FLIGHT_SECONDS / self.dt))
            self._flight_sum = torch.zeros((), device=device)
        # stance_width reward weight (only while weighted): each world's feet width after the step (task.stance_width);
        # its sum and the world-steps where the term counts (task.stance_active), for rollout/stance_width.
        # arm_press reward weight (only while weighted): the CPU task's free arm joints (no locked wrists); the summed
        # Σ and the standing world-steps, for rollout/arm_press.
        self.press_weighted = bool(self.weights['arm_press'])
        if self.press_weighted:
            self.press_joints = constant(ref.press_joints, torch.long)
            self._press_sum = torch.zeros((), device=device)
            self._press_count = torch.zeros((), device=device)
        self.stance_weighted = bool(self.weights['stance_width'])
        if self.stance_weighted:
            self._stance_sum = torch.zeros((), device=device)
            self._stance_count = torch.zeros((), device=device)
        # style reward weight (only while weighted, or judged: task_config style_judge or the amp weight): the style
        # library on the device (float32 rows and their norms; the CPU task's while weighted) and feature joints; the
        # summed d and the world-steps with the head in the band, and all world-steps, for rollout/style_dist and
        # rollout/style_band. Judged only, d is computed and logged but not paid.
        self.style_weighted = ref.style is not None
        self.style_judged = self.style_weighted or bool(style_judge) or bool(self.weights['amp'])
        if self.style_judged:
            style = ref.style if self.style_weighted else g1_getup_style.load(style_library)
            if tuple(style['joints'].tolist()) != tuple(ref.style_joints):
                raise ValueError('style library joints do not match the task (the free joints)')
            self.style_rows = constant(style['features'])
            self.style_norms = task.style_norms(self.style_rows)
            self.style_joints = constant(ref.style_joints, torch.long)
            self._style_sum = torch.zeros((), device=device)
            self._style_band = torch.zeros((), device=device)
            self._style_count = 0
        # head_rise reward weight (iteration 57; only while weighted): each world's head top height after its last step
        # or at its reset (previous_head), the summed unweighted excess and world-steps, and per step the head's
        # vertical speeds in the band (NaN outside it) for rollout/head_rise and rollout/head_speed_p90.
        # sole_slip reward weight (iteration 62; only while weighted): the summed unweighted Σ and world-steps, and per
        # step the speeds of the feet in contact (NaN otherwise), for rollout/sole_slip and rollout/sole_slip_speed.
        self.slip_weighted = bool(self.weights['sole_slip'])
        if self.slip_weighted:
            self._slip_sum = torch.zeros((), device=device)
            self._slip_count = 0
            self._slip_speeds = []
        self.rise_weighted = bool(self.weights['head_rise'])
        self.previous_head = None
        if self.rise_weighted:
            self.previous_head = torch.zeros(n_envs, device=device)
            self._rise_sum = torch.zeros((), device=device)
            self._rise_count = 0
            self._band_speeds = []
        # amp reward weight (iteration 55; paid by the learner, getup_amp.GetupAMP in TensorPPO): each world's
        # getup_amp.amp_state of its current state (amp_state, set at its reset too), after each step the pair (state
        # before, state after) (amp_pair, (n_envs, 2·54)) and where it pays (amp_band: the head after the step in
        # task.STYLE_HEAD_BAND and a finite state); the learner reads the library (amp_library) and the free joints
        # (amp_joints). Off (weight 0): None, nothing computed.
        self.amp_weight = float(self.weights['amp'])
        self.amp_library = style_library
        self.amp_joints = tuple(int(j) for j in ref.style_joints)
        self.amp_state = self.amp_pair = self.amp_band = None
        if self.amp_weight:
            bodies, offsets = getup_amp.point_bodies(ref.model)
            self.amp_bodies, self.amp_offsets = constant(bodies, torch.long), constant(offsets)
            self._amp_joints = constant(self.amp_joints, torch.long)
            self.amp_state = torch.zeros((n_envs, getup_amp.state_size(len(self.amp_joints))), device=device)
        # Episodes ended since the last rollout_metrics, those with the head above GETUP_ASSIST_HEAD at the end, and
        # world-steps ended by a non-finite state.
        self._episodes = torch.zeros((), device=device)
        self._episodes_up = torch.zeros((), device=device)
        self._bad_states = torch.zeros((), device=device)
        # track_clip (iteration 64): None while off (nothing set up, the task the old one).
        if ref.clip is not None:
            self._setup_clip()
        # settle_clip (iteration 66): None while off (nothing set up, the task the old one).
        if ref.settle is not None:
            self._setup_settle()

    def _setup_settle(self):
        """settle_clip on the device: the CPU task's targets (pelvis gravity directions, joints), free joints, weights
        and hand geoms; the training rows narrowed to SETTLE_CLASSES (the draw of _masked_reset unchanged); no assist;
        per world the steps on end near a target and still, the episode step of its success (−1 before) and its
        target (−1); per-substep self-contact (self_contact weighted) and hand floor forces; the counters of
        rollout_metrics."""
        ref, device = self.reference, self.device

        def constant(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=device)
        if not hasattr(self, 'contact_force_graph'):  # built by TensorG1Env for foot_collision, which g1_getup has
            raise ValueError('settle_clip needs the contact force decode (foot_collision)')
        n = self.n_envs
        self.settle = dict(gravity=constant(ref.settle['gravity']), position=constant(ref.settle['position']))
        self.settle_names = ref.settle['names']
        self.settle_joints = tuple(ref.settle_joints)
        self.settle_weights = ref.settle_weights
        self.settle_hold = ref.settle_hold
        self.track_floor = int(ref.floor)
        self.hand_geoms = constant(ref.hand_geoms, torch.bool)
        bank = ref.bank
        classes = [bank['classes'].index(kind) for kind in task.SETTLE_CLASSES if kind in bank['classes']]
        self.training_rows = constant(np.flatnonzero(~bank['holdout'] & np.isin(bank['start_class'], classes)),
                                      torch.long)
        self.assist.zero_()
        self.settle_count = torch.zeros(n, dtype=torch.long, device=device)
        self.settle_at = torch.full((n,), -1, dtype=torch.long, device=device)
        self.settle_target = torch.full((n,), -1, dtype=torch.long, device=device)
        self.self_contact_force = (torch.zeros((n, ref.frame_skip), device=device) if self.settle_weights['self_contact']
                                   else None)
        self.hand_substeps = torch.zeros((n, ref.frame_skip), device=device)
        self._settle_ended = torch.zeros((), device=device)
        self._settle_done = torch.zeros((), device=device)
        self._settle_seconds = torch.zeros((), device=device)
        self._settle_targets = torch.zeros(len(self.settle_names), device=device)
        self._settle_angle = torch.zeros((), device=device)
        self._settle_deviation = torch.zeros((), device=device)
        self._hand_sum = torch.zeros((), device=device)
        self._settle_steps = torch.zeros((), device=device)

    def _setup_clip(self):
        """track_clip on the device: the CPU task's reference frames, clip coordinates, tracked bodies, weights, joint
        ranges and the bank's rows lying on the back; per world the clip clock, start kind (TRACK_START_KINDS' index),
        the reference's yaw and xy offset and whether the clock reached the clip's last frame; the per-substep
        self-contact forces (self_contact weighted) and hand floor forces, Warp's contact distances (the start lift);
        the counters of rollout_metrics."""
        ref, device = self.reference, self.device

        def constant(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=device)
        if not hasattr(self, 'contact_force_graph'):  # built by TensorG1Env for foot_collision, which g1_getup has
            raise ValueError('track_clip needs the contact force decode (foot_collision)')
        n = self.n_envs
        self.clip = {key: constant(value) for key, value in ref.clip_frames.items()}
        self.clip_qpos, self.clip_qvel = constant(np.array(ref.clip['qpos'])), constant(np.array(ref.clip['qvel']))
        self.clip_last, self.clip_end, self.clip_grace = ref.clip_last, ref.clip_end, ref.clip_grace
        self.track_bodies = constant(ref.track_bodies, torch.long)
        self.track_anchor, self.track_ends, self.track_root = ref.track_anchor, ref.track_ends, ref.track_root
        self.track_weights = ref.track_weights
        self.track_floor = int(ref.floor)
        self.hand_geoms = constant(ref.hand_geoms, torch.bool)
        self.range_low, self.range_high = constant(ref.range_low), constant(ref.range_high)
        bank = ref.bank
        rows = np.flatnonzero(~bank['holdout'] & (bank['start_class'] == bank['classes'].index(task.TRACK_BANK_CLASS)))
        self.clip_rows = constant(rows if len(rows) else ref.training_rows, torch.long)
        self.contact_dist = self.wp.to_torch(self.data.contact.dist)
        self.clip_clock = torch.zeros(n, dtype=torch.long, device=device)
        self.clip_kind = torch.zeros(n, dtype=torch.long, device=device)
        self.clip_yaw = torch.zeros(n, device=device)
        self.clip_offset = torch.zeros((n, 2), device=device)
        self.clip_reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.self_contact_force = (torch.zeros((n, ref.frame_skip), device=device) if self.track_weights['self_contact']
                                   else None)
        self.hand_substeps = torch.zeros((n, ref.frame_skip), device=device)
        self._clip_done = torch.zeros(len(task.TRACK_START_KINDS), device=device)  # ended, reached the last frame
        self._clip_ended = torch.zeros(len(task.TRACK_START_KINDS), device=device)
        self._clip_failed = torch.zeros((), device=device)
        self._clip_standing = torch.zeros((), device=device)  # ended by the hold's end, standing (task.getup_standing)
        self._clip_errors = {key: torch.zeros((), device=device) for key in task.TRACK_SIGMA}
        self._hand_sum = torch.zeros((), device=device)
        self._clip_count = torch.zeros((), device=device)

    def _clip_reference(self):
        return task.clip_reference(self.clip, self.clip_clock, self.clip_yaw, self.clip_offset)

    def _clip_robot(self):
        """(n_envs, B, …) the tracked bodies: frame origins, rotations, world linear and angular velocities."""
        bodies = self.track_bodies
        position = self.xpos[:, bodies]
        com = self.subtree_com[:, self.track_root][:, None].expand(-1, len(bodies), -1)
        linear, angular = task.body_point_velocity(self.cvel[:, bodies], position, com)
        return position, self.xmat[:, bodies].reshape(self.n_envs, -1, 3, 3), linear, angular

    def _clip_extras(self):
        anchor = self.track_bodies[self.track_anchor]
        position, rotation, _, _ = self._clip_reference()
        return task.track_critic_extras(self.clip_clock, self.clip_last, self.xpos[:, anchor],
                                        self.xmat[:, anchor].reshape(-1, 3, 3), position[:, self.track_anchor],
                                        rotation[:, self.track_anchor])

    def _hand_forces(self):
        """(n_envs,) the floor's force on each world's hands at the last physics step (the sum of the contact force
        norms of the live floor-hand slots; the CPU task's _hand_force), from the contact forces _contact_forces just
        decoded."""
        geom = self.contact_geom.long()
        live = (self._contact_slots < self.nacon) & (geom >= 0).all(dim=1)
        selected, _, _ = task.impact_contacts(geom.clamp(min=0), self.track_floor, self.hand_geoms)
        counted = live & selected
        force = torch.where(counted, self.contact_force[:, :3].norm(dim=1), 0.)
        world = torch.where(counted, self.contact_world.long(), self.n_envs)
        total = torch.zeros(self.n_envs + 1, device=self.device)
        return total.scatter_add_(0, world, force)[:-1]

    def _floor_depth(self):
        """(n_envs,) per world its deepest robot-floor contact distance (m, ≤ 0; 0 without a penetrating one) in Warp's
        contacts now (after a forward)."""
        geom = self.contact_geom.long()
        live = (self._contact_slots < self.nacon) & (geom >= 0).all(dim=1)
        selected, _, _ = task.impact_contacts(geom.clamp(min=0), self.track_floor, self.robot_geoms)
        counted = live & selected
        world = torch.where(counted, self.contact_world.long(), self.n_envs)
        depth = torch.zeros(self.n_envs + 1, device=self.device)
        return depth.scatter_reduce_(0, world, torch.where(counted, self.contact_dist, 0.), 'amin')[:-1]

    def _clip_masked_reset(self, mask, rows=None, frames=None, noise=True):
        """track_clip's reset of the masked worlds, as the CPU task's _clip_reset: rows (n_envs,) bank rows for every
        one (the reference aligned to each, task.track_align), or frames (n_envs,) clip frames (their velocities: the
        CPU reset option track_frame), else per world a training draw (TRACK_START_SHARES: the clip's first frame, a
        uniform frame, a bank row lying on the back); noise on the clip starts unless `noise` is false (the CPU
        option track_noise); then the getting-up reset's randomization (encoder bias, delay, motor and body friction),
        a forward, the clip starts raised by their deepest floor contact and another forward. No assist update."""
        n, generator = self.n_envs, self.generator
        self.reset_mask.copy_(mask)
        self.wp.capture_launch(self.reset_graph)
        if rows is not None:
            kinds, frames = torch.full((n,), 2, dtype=torch.long, device=self.device), torch.zeros_like(rows)
        elif frames is not None:
            kinds, rows = torch.ones_like(frames), torch.zeros_like(frames)
        else:
            draw = torch.rand(n, device=self.device, generator=generator)
            first, second = task.TRACK_START_SHARES[0], task.TRACK_START_SHARES[0] + task.TRACK_START_SHARES[1]
            kinds = (draw >= first).long() + (draw >= second).long()
            frames = torch.randint(0, self.clip_last + 1, (n,), device=self.device, generator=generator)
            frames = torch.where(kinds == 1, frames, 0)
            rows = self.clip_rows[torch.randint(0, len(self.clip_rows), (n,), device=self.device, generator=generator)]
        pose = 2 * torch.rand((n, task.TRACK_POSE_NOISE), device=self.device, generator=generator) - 1
        speed = 2 * torch.rand((n, task.TRACK_SPEED_NOISE), device=self.device, generator=generator) - 1
        bank = kinds == 2
        positions, velocities = self.clip_qpos[frames], torch.where((kinds == 1)[:, None], self.clip_qvel[frames], 0.)
        if noise:
            noisy = ~bank[:, None]
            positions = torch.where(noisy, task.track_start_pose(positions, pose, self.range_low, self.range_high),
                                    positions)
            velocities = torch.where(noisy, task.track_start_speed(velocities, speed), velocities)
        positions = torch.where(bank[:, None], self.bank_qpos[rows], positions)
        velocities = torch.where(bank[:, None], self.bank_qvel[rows], velocities)
        positions[:, 3:7] /= positions[:, 3:7].norm(dim=1, keepdim=True)
        self.qpos.copy_(torch.where(mask[:, None], positions, self.qpos))
        self.qvel.copy_(torch.where(mask[:, None], velocities, self.qvel))
        self.start_row.copy_(torch.where(mask, torch.where(bank, rows, -1), self.start_row))
        self.previous_action.masked_fill_(mask[:, None], 0)
        if self.previous_unbounded is not None:
            self.previous_unbounded.masked_fill_(mask[:, None], 0)
        self.command.masked_fill_(mask[:, None], 0)
        if self.observation_noise:
            bias = ((2 * torch.rand(self.encoder_bias.shape, device=self.device, generator=generator) - 1)
                    * ENCODER_BIAS * self.realism)
            self.encoder_bias.copy_(torch.where(mask[:, None], bias, self.encoder_bias))
        self.lengths.masked_fill_(mask, 0)
        self.head_window.masked_fill_(mask[:, None], 0)
        if self.sole_impact_substeps is not None:
            self.airborne.masked_fill_(mask, 0)
        if self.delay_depth:
            delay = torch.randint(0, self.reference.max_delay_steps + 1, (n,), device=self.device, generator=generator)
            self.delay_steps.copy_(torch.where(mask, delay, self.delay_steps))
            self.delay_lags.copy_(task.action_lags(self.delay_steps, self.reference.frame_skip))
            self.delay_actions.masked_fill_(mask[:, None, None], 0)
        if self.motor_frictions:
            scale = (task.motor_friction_draw(self._uniform, n, self.reference.motor_friction)
                     if self.reference.motor_friction else torch.ones_like(self.friction_scale))
            self.friction_scale.copy_(torch.where(mask, scale, self.friction_scale))
        low, high = self.body_friction
        self._set_friction(torch.where(mask, low + (high - low) * self._uniform(n), self.friction))
        self.returns.masked_fill_(mask, 0)
        self.wp.capture_launch(self.forward_graph)
        raised = torch.where(mask & ~bank, (-self._floor_depth()).clamp(min=0), 0.)
        self.qpos[:, 2] += raised
        self.wp.capture_launch(self.forward_graph)
        yaw, offset = task.track_align(self.qpos[:, 0:3], self.xmat[:, self.pelvis].reshape(-1, 3, 3),
                                       self.clip['pos'][0, 0], self.clip['rot'][0, 0])
        aligned = mask & bank
        self.clip_yaw.copy_(torch.where(aligned, yaw, torch.where(mask, 0., self.clip_yaw)))
        self.clip_offset.copy_(torch.where(aligned[:, None], offset, torch.where(mask[:, None], 0., self.clip_offset)))
        self.clip_clock.copy_(torch.where(mask, frames, self.clip_clock))
        self.clip_kind.copy_(torch.where(mask, kinds, self.clip_kind))
        self.clip_reached.copy_(torch.where(mask, frames >= self.clip_last, self.clip_reached))
        remaining = (self.clip_end - frames).clamp(min=1, max=self.max_episode_steps)
        self.episode_limit.copy_(torch.where(mask, remaining, self.episode_limit))

    @torch.no_grad()
    def set_clip_starts(self, frames, noise=False):
        """Reset every world now from clip frames[w] ((n_envs,) or one frame) with the clip's velocities there, without
        noise unless `noise` (the CPU task's reset options track_frame, track_noise; tests, probes); track_clip only.
        Returns the fresh observation."""
        if self.clip is None:
            raise ValueError('clip starts need task_config track_clip')
        frames = torch.as_tensor(frames, dtype=torch.long, device=self.device).expand(self.n_envs)
        if bool((frames < 0).any()) or bool((frames > self.clip_last).any()):
            raise ValueError('frames must be frames of the clip')
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._clip_masked_reset(torch.ones(self.n_envs, dtype=torch.bool, device=self.device), frames=frames.clone(),
                                    noise=noise)
            return self._obs('fill')

    def _make_reference(self, max_episode_steps, **settings):
        """G1GetupEnv with this task's settings; it fixes the robot and connector (config.G1_GETUP_PROFILE) itself."""
        return G1GetupEnv(max_episode_steps, **self._getup)

    def _batched_fields(self):
        """The body friction is drawn per world at every reset, so geom_friction is always batched; with randomize the
        pelvis COM too (set once per world, as the walking task)."""
        return {'geom_friction': self.n_envs, **({'body_ipos': self.n_envs} if self.randomize else {})}

    def _head(self):
        return task.head_height(self.xpos[:, self.torso], self.xmat[:, self.torso].reshape(-1, 3, 3))

    def _amp_state(self):
        """(n_envs, 54) every world's getup_amp.amp_state now (the amp weight's discriminator input, one frame)."""
        points = getup_amp.amp_points(self.xpos, self.xmat, self.amp_bodies, self.amp_offsets)
        return getup_amp.amp_state(self.qpos[:, self.qadr], self.qpos[:, :3], self.xmat[:, self.pelvis].reshape(-1, 3, 3),
                                   self._head(), points, self._amp_joints)

    def _obs(self, history='replace', mask=None):
        """The walking task's actor inputs (command 0, phase stopped), then task.getup_critic_extras."""
        actor = super()._obs(history, mask)[:, :self.reference.actor_inputs]
        extras = task.getup_critic_extras(self._velocity(), self.qpos[:, 2], self._head(), self.assist)
        if self.clip is not None:  # track_clip: the clip's phase and the anchor's errors, after the six
            return torch.cat((actor, extras, self._clip_extras()), dim=1)
        return torch.cat((actor, extras), dim=1)

    def _human_rows(self):
        """The human starts on the device (the CPU task's, g1_getup_human), loaded once."""
        if self.human_qpos is None:
            human = self.reference.human
            self.human_qpos = torch.as_tensor(human['qpos'], dtype=torch.float32, device=self.device)
            self.human_qvel = torch.as_tensor(human['qvel'], dtype=torch.float32, device=self.device)
        return self.human_qpos, self.human_qvel

    def _masked_reset(self, mask, rows=None, human_rows=None):
        """Resets the masked worlds: the episode that ended steps its assist down first (task.assist_update on the
        mean head height of its last GETUP_ASSIST_SECONDS; NaN, no step, for a shorter one; not after an episode from a
        human start), then a start from the bank (rows: (n_envs,) bank rows; None: a training row drawn per world,
        with human_starts a human row instead with probability share; human_rows: (n_envs,) human rows for every
        masked world), and the episode's delay, motor friction and body friction, as the CPU task's reset. With
        track_clip: _clip_masked_reset (rows: bank starts)."""
        if self.clip is not None:
            return self._clip_masked_reset(mask, rows=rows)
        full = self.lengths >= self.head_steps
        head_mean = torch.where(full, self.head_window.mean(dim=1), torch.full_like(self.assist, float('nan')))
        stepping = mask if self.human_share is None else mask & ~self.human_episode
        self.assist.copy_(torch.where(stepping, task.assist_update(self.assist, head_mean), self.assist))
        if self.pd_bound is not None:  # the bound by the same rule (no draw); nothing without the setting
            stepped = task.pd_bound_update(self.bound, head_mean, self.pd_bound[1])
            self.bound.copy_(torch.where(stepping, stepped, self.bound))
        self.reset_mask.copy_(mask)
        self.wp.capture_launch(self.reset_graph)  # also zeroes the reset worlds' xfrc_applied
        human = None  # (n_envs,) bool: the worlds starting from a human row
        if rows is None:
            draw = torch.randint(0, len(self.training_rows), (self.n_envs,), device=self.device,
                                 generator=self.generator)
            rows = self.training_rows[draw]
            if self.human_share is not None:  # human_starts: its own draws, after the bank rows
                human = torch.rand(self.n_envs, device=self.device, generator=self.human_generator) < self.human_share
                human_rows = torch.randint(0, len(self.human_qpos), (self.n_envs,), device=self.device,
                                           generator=self.human_generator)
        elif human_rows is not None:  # set_human_starts: every masked world from its human row
            human = torch.ones_like(mask)
        positions = self.bank_qpos[rows].clone()
        velocities = self.bank_qvel[rows]
        if human is not None:
            human_qpos, human_qvel = self._human_rows()
            positions = torch.where(human[:, None], human_qpos[human_rows], positions)
            velocities = torch.where(human[:, None], human_qvel[human_rows], velocities)
            rows = torch.where(human, -1, rows)
        if human is not None or self.human_share is not None:
            started = torch.zeros_like(mask) if human is None else human
            self.human_episode.copy_(torch.where(mask, started, self.human_episode))
        positions[:, 3:7] /= positions[:, 3:7].norm(dim=1, keepdim=True)
        self.qpos.copy_(torch.where(mask[:, None], positions, self.qpos))
        self.qvel.copy_(torch.where(mask[:, None], velocities, self.qvel))
        self.start_row.copy_(torch.where(mask, rows, self.start_row))
        self.previous_action.masked_fill_(mask[:, None], 0)
        if self.previous_unbounded is not None:
            self.previous_unbounded.masked_fill_(mask[:, None], 0)
        self.command.masked_fill_(mask[:, None], 0)  # no command, ever
        if self.observation_noise:
            bias = ((2 * torch.rand(self.encoder_bias.shape, device=self.device, generator=self.generator) - 1)
                    * ENCODER_BIAS * self.realism)
            self.encoder_bias.copy_(torch.where(mask[:, None], bias, self.encoder_bias))
        self.lengths.masked_fill_(mask, 0)
        self.episode_limit.masked_fill_(mask, self.max_episode_steps)
        self.head_window.masked_fill_(mask[:, None], 0)
        if self.sole_impact_substeps is not None:  # rollout/flight: a fresh episode starts on the floor's count
            self.airborne.masked_fill_(mask, 0)
        if self.delay_depth:
            delay = torch.randint(0, self.reference.max_delay_steps + 1, (self.n_envs,), device=self.device,
                                  generator=self.generator)
            self.delay_steps.copy_(torch.where(mask, delay, self.delay_steps))
            self.delay_lags.copy_(task.action_lags(self.delay_steps, self.reference.frame_skip))
            self.delay_actions.masked_fill_(mask[:, None, None], 0)  # zero actions: the home pose
        if self.motor_frictions:
            scale = (task.motor_friction_draw(self._uniform, self.n_envs, self.reference.motor_friction)
                     if self.reference.motor_friction else torch.ones_like(self.friction_scale))
            self.friction_scale.copy_(torch.where(mask, scale, self.friction_scale))
        low, high = self.body_friction
        self._set_friction(torch.where(mask, low + (high - low) * self._uniform(self.n_envs), self.friction))
        self.returns.masked_fill_(mask, 0)
        self.wp.capture_launch(self.forward_graph)
        if self.previous_head is not None:  # head_rise: a fresh episode's first speed is from its start pose
            self.previous_head.copy_(torch.where(mask, self._head(), self.previous_head))
        if self.amp_state is not None:  # amp: the reset worlds' first state, the start of their first pair
            self.amp_state.copy_(torch.where(mask[:, None], self._amp_state(), self.amp_state))
        if self.settle is not None:  # settle_clip: a fresh episode has not settled yet
            self.settle_count.masked_fill_(mask, 0)
            self.settle_at.masked_fill_(mask, -1)
            self.settle_target.masked_fill_(mask, -1)

    @torch.no_grad()
    def reset(self):
        """Every world from the bank (TensorG1Env.reset), its first episode cut at a random step of
        1..max_episode_steps (episode_limit, drawn after the reset's draws), so the worlds' episodes end spread out
        (with track_clip within what is left of the clip and its hold)."""
        obs = super().reset()
        cut = torch.randint(1, self.max_episode_steps + 1, (self.n_envs,), device=self.device, generator=self.generator)
        self.episode_limit.copy_(cut if self.clip is None else torch.minimum(cut, self.episode_limit))
        return obs

    def _set_friction(self, friction):
        self.friction.copy_(friction)
        self.geom_friction[:, self.friction_geoms, 0] = friction[:, None]

    def _count_constraints(self):
        """This substep's demand: the most constraint rows and contacts of any world, kept for rollout_metrics."""
        torch.maximum(self._nefc_max, self.nefc.max(), out=self._nefc_max)
        world = torch.where(self._contact_slots < self.nacon, self.contact_world.long(), self.n_envs)
        counts = torch.zeros(self.n_envs + 1, dtype=torch.long, device=self.device)
        counts.scatter_add_(0, world, self._contact_ones)
        torch.maximum(self._ncon_max, counts[:-1].max(), out=self._ncon_max)

    def _floor_speeds(self):
        """(n_envs, floor_width) per world and entry (floor_entries: impact's bodies, then sole_impact's feet) the
        largest approach speed (m/s) of its floor contacts at the last physics step, 0 without one (the CPU task's
        _impact_speeds): the contact slots task.impact_contacts picks on floor_geoms, live (below nacon, geoms, not
        flex), reduced per world and entry. With sole_impact it also marks the worlds with any live floor contact at
        all, whatever the geom (touching, for rollout/flight)."""
        geom = self.contact_geom.long()
        live = (self._contact_slots < self.nacon) & (geom >= 0).all(dim=1)
        selected, robot, sign = task.impact_contacts(geom.clamp(min=0), self.floor, self.floor_geoms)
        world = self.contact_world.long().clamp(0, self.n_envs - 1)  # stale slots: any world, they are not live
        body = self.body_of_geom[robot]
        speed = task.approach_speed(self.contact_frame[:, 0], sign, self.contact_pos, self.cvel[world, body],
                                    self.subtree_com[world, self.root_of_body[body]])
        index = torch.where(live & selected, world * self.floor_width + self.floor_entry[robot], self._floor_spare)
        flat = self._floor_flat.zero_().scatter_reduce_(0, index, speed, 'amax')
        if self.sole_impact_substeps is not None:
            floor = live & ((geom[:, 0] == self.floor) | (geom[:, 1] == self.floor))
            counts = torch.zeros(self.n_envs + 1, dtype=torch.long, device=self.device)
            counts.scatter_add_(0, torch.where(floor, world, self.n_envs), self._contact_ones)
            self.touching.logical_or_(counts[:-1] > 0)
        return flat[:self.n_envs * self.floor_width].view(self.n_envs, self.floor_width)

    @torch.no_grad()
    def set_starts(self, rows):
        """Reset every world now from bank rows[w] ((n_envs,) or one row; held-out rows too), as the CPU task's reset
        option start (tests, probes); delay, motor and body friction drawn as at any reset (set_action_delay_steps,
        set_motor_friction, set_body_friction override them). Returns the fresh observation."""
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).expand(self.n_envs)
        if bool((rows < 0).any()) or bool((rows >= len(self.bank_qpos)).any()):
            raise ValueError('start rows must be rows of the start bank')
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._masked_reset(torch.ones(self.n_envs, dtype=torch.bool, device=self.device), rows.clone())
            return self._obs('fill')

    @torch.no_grad()
    def set_human_starts(self, rows):
        """Reset every world now from human start rows[w] ((n_envs,) or one row; g1_getup_human), as the CPU task's
        reset option human_start (tests, probes), also without the setting; the rest drawn as at any reset. Returns
        the fresh observation."""
        human_qpos, _ = self._human_rows()
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).expand(self.n_envs)
        if bool((rows < 0).any()) or bool((rows >= len(human_qpos)).any()):
            raise ValueError('human start rows must be rows of the human starts')
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            everyone = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
            self._masked_reset(everyone, torch.zeros_like(rows), rows.clone())
            return self._obs('fill')

    def set_body_friction(self, friction):
        """The sliding friction of every colliding robot geom and of the floor in each world ((n_envs,) or one number)
        until its next reset, as the CPU task's set_body_friction."""
        friction = torch.as_tensor(friction, dtype=torch.float32, device=self.device).expand(self.n_envs)
        if not bool(torch.isfinite(friction).all()) or bool((friction <= 0).any()):
            raise ValueError('expected finite positive body frictions')
        self._set_friction(friction.clone())

    def set_assist(self, force):
        """Each world's assist force (N, (n_envs,) or one number), as the CPU task's reset option assist_force; the
        next observation's critic extras show it."""
        force = torch.as_tensor(force, dtype=torch.float32, device=self.device).expand(self.n_envs)
        if not bool(torch.isfinite(force).all()) or bool((force < 0).any()):
            raise ValueError('expected finite non-negative assist forces')
        self.assist.copy_(force)

    def training_state(self):
        """What a stage keeps of this training beyond the weights: every world's assist force, with pd_bound every
        world's bound (save_stage)."""
        state = {'assist_force': [float(v) for v in self.assist.cpu().tolist()]}
        if self.pd_bound is not None:
            state['pd_bound'] = [float(v) for v in self.bound.cpu().tolist()]
        return state

    def restore_training_state(self, state):
        """The worlds' assist forces from a stage's training_state, spread over this run's world count
        (task.assist_resample), with pd_bound their bounds too, within [lowest, start] (a stage without bounds: the
        start), as the CPU task's restore_training_state."""
        forces = np.asarray(state['assist_force'], dtype=np.float64)
        if forces.ndim != 1 or not len(forces) or not np.isfinite(forces).all() or (forces < 0).any():
            raise ValueError('Invalid saved assist forces')
        self.assist.copy_(torch.as_tensor(task.assist_resample(forces, self.n_envs), dtype=torch.float32,
                                          device=self.device))
        if self.settle is not None:  # settle_clip: no assist, whatever the stage it starts from had
            self.assist.zero_()
        if self.pd_bound is not None and 'pd_bound' in state:
            bounds = np.clip(task.assist_resample(saved_bounds(state), self.n_envs), self.pd_bound[1], self.pd_bound[0])
            self.bound.copy_(torch.as_tensor(bounds, dtype=torch.float32, device=self.device))

    def set_pd_bound(self, fraction):
        """Each world's PD error bound c ((n_envs,) or one number, a share of the peak torque, > 0) from the next step
        on, until its reset steps it (with the setting) or for good (without: tests, probes), as the CPU task's
        set_pd_bound; None: no bound in any world."""
        if fraction is None:
            self.bounding = False
            return
        fraction = torch.as_tensor(fraction, dtype=torch.float32, device=self.device).expand(self.n_envs)
        if not bool(torch.isfinite(fraction).all()) or bool((fraction <= 0).any()):
            raise ValueError('expected finite positive PD error bounds')
        self.bound.copy_(fraction)
        self.bounding = True

    @torch.no_grad()
    def step(self, actions):
        if actions.device != self.device or actions.shape != self.previous_action.shape:
            raise ValueError('Expected CUDA actions matching (n_envs, actuator_count)')
        if self.clip is not None:
            return self._clip_step(actions)
        if self.settle is not None:
            return self._settle_step(actions)
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._bad_state_seen.logical_or_(~torch.isfinite(actions).all())
            limit = self.reference.action_limit
            # The applied action (box clip, pd_bound, lock_wrists) replaces the policy's in all below, as on the CPU
            # task.
            action = actions.clamp(-limit, limit)
            # action_change_unbounded: the action_change term's action, before pd_bound (box clip, lock_wrists).
            unbounded = task.lock_joints(action, self.wrist_mask) if self.previous_unbounded is not None else None
            if self.bounding:  # around the angles the policy observed: the newest frame of its history
                position = self.frames[:, -1, self.position_entries]
                bounded = task.pd_bound(action, position, self.bound, self.peak, self.kp, self.scale, self.bound_free)
                self._bounded.add_((bounded != action).sum())
                self._bounded_count += self.n_envs * self._free_joints
                action = bounded
            action = task.lock_joints(action, self.wrist_mask)
            # Assist: each world's force while its pelvis is nearly upright, on every substep of this step.
            self.lift.copy_(task.assist_force(self.xmat[:, self.pelvis].reshape(-1, 3, 3), self.assist))
            if self.delay_depth:  # this step's action, then the older ones the motors may still follow
                if self.bounding:
                    # The actions still in flight at an episode's first step: the zero action bounded around the
                    # observed start pose (unbounded, home: a full-torque jerk from lying under the delay), as the CPU
                    # task.
                    zero = task.lock_joints(task.pd_bound(torch.zeros_like(action), position, self.bound, self.peak,
                                                          self.kp, self.scale, self.bound_free), self.wrist_mask)
                    first = (self.lengths == 0)[:, None, None]
                    self.delay_actions.copy_(torch.where(first, zero[:, None], self.delay_actions))
                queue = torch.cat((action[:, None], self.delay_actions), dim=1)
            friction = dict(friction_scale=self.friction_scale) if self.motor_frictions else {}
            if self.sole_impact_substeps is not None:
                self.touching.zero_()
            # sole_slip: the feet's horizontal positions before the step, only while weighted.
            feet_before = self.xpos[:, self.feet, :2].clone() if self.slip_weighted else None
            for substep in range(self.reference.frame_skip):
                applied = queue[self._worlds, self.delay_lags[:, substep]] if self.delay_depth else action
                joint_velocity = self.qvel[:, self.vadr]
                torque = pd_torque(applied, self.qpos[:, self.qadr], joint_velocity, self.home, self.scale, self.kp,
                                   self.kd, self.joint_low, self.joint_high, -self.unclamped, self.unclamped, limit)
                torque = task.motor_torque(torque, joint_velocity, *self.motors, **friction).clamp(min=self.torque_low,
                                                                                                   max=self.torque_high)
                self.ctrl.copy_(torque)
                self.xfrc_applied[:, self.pelvis, 2] = self.lift
                self.wp.capture_launch(self.step_graph)
                self._overflow_seen.bitwise_or_(self.overflow)
                self._count_constraints()
                if self.floor_entry is not None:  # impact, sole_impact: this substep's floor contacts, while weighted
                    speeds = self._floor_speeds()
                    if self.impact_substeps is not None:
                        self.impact_substeps[:, substep] = speeds[:, :self.nbody]
                    if self.sole_impact_substeps is not None:
                        self.sole_impact_substeps[:, substep] = speeds[:, self.nbody:]
            self.xfrc_applied[:, self.pelvis, 2] = 0.
            self.wp.capture_launch(self.forward_graph)
            self.lengths.add_(1)
            # A non-finite state ends its world's episode with reward 0 and a zero terminal observation (the CPU
            # task's bad_state); the other worlds go on, so training does not stop for one world.
            bad = ~(torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1))
            rotation = self.xmat[:, self.torso].reshape(-1, 3, 3)
            head = self._head()
            sole = task.sole_height(self.geom_xpos[:, self.soles, 2], self.sole_radius)
            impact = None
            if self.impact_substeps is not None:  # per body its largest approach speed over the substeps
                speed = self.impact_substeps.amax(dim=1)
                impact = task.impact_penalty(speed)
                self._impact_sum.add_(torch.where(bad, 0., impact).sum())
                self._impact_hits.add_((task.impact_hit(speed) & ~bad).sum())
                self._impact_count += self.n_envs
            sole_impact = None
            if self.sole_impact_substeps is not None:  # per foot its largest sole approach speed over the substeps
                speed = self.sole_impact_substeps.amax(dim=1)
                sole_impact = task.impact_penalty(speed, task.GETUP_SOLE_IMPACT_SPEED)
                self._sole_impact_sum.add_(torch.where(bad, 0., sole_impact).sum())
                self._sole_impact_hits.add_((task.impact_hit(speed, task.GETUP_SOLE_IMPACT_SPEED) & ~bad).sum())
                self._sole_impact_count += self.n_envs
                # rollout/flight: the steps since the world last touched the floor, a flight from GETUP_FLIGHT_SECONDS.
                self.airborne.copy_(torch.where(self.touching, 0, self.airborne + 1))
                self._flight_sum.add_(((self.airborne >= self.flight_steps) & ~bad).sum())
            width = None
            if self.stance_weighted:  # the feet's width across the pelvis heading after the step
                width = task.stance_width(self.xpos[:, self.feet], self.xmat[:, self.pelvis].reshape(-1, 3, 3))
                counted = task.stance_active(self.qpos[:, 2], sole, self.stance_pelvis) & ~bad
                self._stance_sum.add_(torch.where(counted, width, 0.).sum())
                self._stance_count.add_(counted.sum())
            # action_change: the applied action and the one before, under action_change_unbounded the action before
            # pd_bound and its own previous one.
            changed, before = ((action, self.previous_action) if unbounded is None
                               else (unbounded, self.previous_unbounded))
            press = None
            if self.press_weighted:  # the target before pd_bound (box clip, locked wrists) against the angle after
                pre = unbounded if unbounded is not None else task.lock_joints(actions.clamp(-limit, limit),
                                                                               self.wrist_mask)
                target = task.pd_target(pre, self.home, self.scale, self.joint_low, self.joint_high, limit)
                press = task.arm_press(target, self.qpos[:, self.qadr], self.press_joints)
            style = None
            if self.style_judged:  # each world's distance to the nearest human frame, after the step
                features = task.style_features(self.qpos[:, self.qadr], self.qvel[:, self.vadr],
                                               self.xmat[:, self.pelvis].reshape(-1, 3, 3), self.qpos[:, 2],
                                               self.qvel[:, 2], self.style_joints)
                style, _ = task.style_distance(features, self.style_rows, self.style_norms)
                counted = task.style_band(head) & ~bad
                self._style_sum.add_(torch.where(counted, style, 0.).sum())
                self._style_band.add_(counted.sum())
                self._style_count += self.n_envs
            slip = None
            if feet_before is not None:  # sole_slip: each foot's speed over the step, Σ of its square in contact
                foot_speed = (self.xpos[:, self.feet, :2] - feet_before).norm(dim=-1) / self.dt
                slip = task.sole_slip(foot_speed, sole)
                self._slip_sum.add_(torch.where(bad, 0., slip).sum())
                self._slip_count += self.n_envs
                contact = (sole < task.SOLE_CONTACT_HEIGHT) & ~bad[:, None]
                self._slip_speeds.append(torch.where(contact, foot_speed, float('nan')).flatten())
            head_speed = None
            if self.rise_weighted:  # head_rise: the head top's vertical speed over the step
                head_speed = (head - self.previous_head) / self.dt
                self._rise_sum.add_(torch.where(bad, 0., task.head_rise(head_speed)).sum())
                self._rise_count += self.n_envs
                self._band_speeds.append(torch.where(task.style_band(head) & ~bad, head_speed, float('nan')))
                self.previous_head.copy_(head)  # a world that resets below takes its start's height there
            if self.amp_state is not None:  # amp: the pair (state before the step, after it), paid in the band
                after = self._amp_state()
                self.amp_pair = torch.cat((self.amp_state, after), dim=1)
                self.amp_band = task.style_band(head) & ~bad
                self.amp_state.copy_(after)  # a world that resets below takes its first state there
            components, standing = task.getup_reward_components(
                head, self.qpos[:, 2], rotation, self._velocity(), self.qpos[:, self.qadr], self.home, self.soft_low,
                self.soft_high, self.limit_joints, sole, self.ctrl, self.qvel[:, self.vadr], changed, before,
                weights=self.weights, impact=impact, stand_pose_std=self.stand_pose_std, sole_impact=sole_impact,
                stance_width=width, arm_press=press, style=style if self.style_weighted else None,
                head_speed=head_speed, stance_pelvis=self.stance_pelvis, sole_slip=slip)
            if press is not None:  # rollout/arm_press: the unweighted Σ over the standing world-steps
                counted = standing & ~bad
                self._press_sum.add_(torch.where(counted, press, 0.).sum())
                self._press_count.add_(counted.sum())
            reward = torch.where(bad, 0., sum(components.values()))
            self.head_window[self._worlds, (self.lengths - 1) % self.head_steps] = torch.where(bad, 0., head)
            self.previous_action.copy_(action)
            if unbounded is not None:
                self.previous_unbounded.copy_(unbounded)
            if self.delay_depth:
                self.delay_actions.copy_(queue[:, :-1])
            terminal_obs = torch.where(bad[:, None], 0., self._obs('push'))
            self.returns.add_(reward)
            terminated = bad | (self.qvel[:, self.vadr].abs().amax(dim=1) > task.GETUP_MAX_JOINT_SPEED)
            truncated = self.lengths >= self.episode_limit
            done = terminated | truncated
            self._episodes.add_(done.sum())
            self._episodes_up.add_((done & ~bad & (head > task.GETUP_ASSIST_HEAD)).sum())
            if self.human_share is not None:  # human_starts: the ended episodes by their start
                self._episodes_human.add_((done & self.human_episode).sum())
                bank_up = done & ~self.human_episode & ~bad & (head > task.GETUP_ASSIST_HEAD)
                self._episodes_bank_up.add_(bank_up.sum())
            self._bad_states.add_(bad.sum())
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._overflow_seen.bitwise_or_(self.overflow)
            self._masked_reset(done)
            # Worlds that reset start a fresh history filled with their first frame; the others keep theirs.
            return torch.where(done[:, None], self._obs('fill', done), terminal_obs), reward, terminated, truncated, \
                terminal_obs

    @torch.no_grad()
    def _clip_step(self, actions):
        """track_clip's step, the CPU task's _clip_step per world: the getting-up step's action, delay queue, PD and
        substeps, each substep's contact forces decoded (self_contact while weighted, the hands' floor force) and the
        impact / sole_impact speeds while weighted; then the clocks on, the tracking errors, reward and terminations,
        the logs, and the ended worlds reset."""
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._bad_state_seen.logical_or_(~torch.isfinite(actions).all())
            limit = self.reference.action_limit
            action = actions.clamp(-limit, limit)
            unbounded = task.lock_joints(action, self.wrist_mask) if self.previous_unbounded is not None else None
            if self.bounding:
                position = self.frames[:, -1, self.position_entries]
                action = task.pd_bound(action, position, self.bound, self.peak, self.kp, self.scale, self.bound_free)
            action = task.lock_joints(action, self.wrist_mask)
            self.lift.copy_(task.assist_force(self.xmat[:, self.pelvis].reshape(-1, 3, 3), self.assist))
            if self.delay_depth:
                if self.bounding:
                    zero = task.lock_joints(task.pd_bound(torch.zeros_like(action), position, self.bound, self.peak,
                                                          self.kp, self.scale, self.bound_free), self.wrist_mask)
                    first = (self.lengths == 0)[:, None, None]
                    self.delay_actions.copy_(torch.where(first, zero[:, None], self.delay_actions))
                queue = torch.cat((action[:, None], self.delay_actions), dim=1)
            friction = dict(friction_scale=self.friction_scale) if self.motor_frictions else {}
            if self.sole_impact_substeps is not None:
                self.touching.zero_()
            for substep in range(self.reference.frame_skip):
                applied = queue[self._worlds, self.delay_lags[:, substep]] if self.delay_depth else action
                joint_velocity = self.qvel[:, self.vadr]
                torque = pd_torque(applied, self.qpos[:, self.qadr], joint_velocity, self.home, self.scale, self.kp,
                                   self.kd, self.joint_low, self.joint_high, -self.unclamped, self.unclamped, limit)
                torque = task.motor_torque(torque, joint_velocity, *self.motors, **friction).clamp(min=self.torque_low,
                                                                                                   max=self.torque_high)
                self.ctrl.copy_(torque)
                self.xfrc_applied[:, self.pelvis, 2] = self.lift
                self.wp.capture_launch(self.step_graph)
                self._overflow_seen.bitwise_or_(self.overflow)
                self._count_constraints()
                robot, _ = self._contact_forces()  # this substep's contact forces, decoded once
                if self.self_contact_force is not None:
                    self.self_contact_force[:, substep] = robot
                self.hand_substeps[:, substep] = self._hand_forces()
                if self.floor_entry is not None:
                    speeds = self._floor_speeds()
                    if self.impact_substeps is not None:
                        self.impact_substeps[:, substep] = speeds[:, :self.nbody]
                    if self.sole_impact_substeps is not None:
                        self.sole_impact_substeps[:, substep] = speeds[:, self.nbody:]
            self.xfrc_applied[:, self.pelvis, 2] = 0.
            self.wp.capture_launch(self.forward_graph)
            self.lengths.add_(1)
            self.clip_clock.add_(1)
            bad = ~(torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1))
            robot = self._clip_robot()
            reference = self._clip_reference()
            errors = task.track_errors(*robot, *reference, self.track_anchor)
            impact = sole_impact = None
            if self.impact_substeps is not None:
                speed = self.impact_substeps.amax(dim=1)
                impact = task.impact_penalty(speed)
                self._impact_sum.add_(torch.where(bad, 0., impact).sum())
                self._impact_hits.add_((task.impact_hit(speed) & ~bad).sum())
                self._impact_count += self.n_envs
            if self.sole_impact_substeps is not None:
                speed = self.sole_impact_substeps.amax(dim=1)
                sole_impact = task.impact_penalty(speed, task.GETUP_SOLE_IMPACT_SPEED)
                self._sole_impact_sum.add_(torch.where(bad, 0., sole_impact).sum())
                self._sole_impact_hits.add_((task.impact_hit(speed, task.GETUP_SOLE_IMPACT_SPEED) & ~bad).sum())
                self._sole_impact_count += self.n_envs
                self.airborne.copy_(torch.where(self.touching, 0, self.airborne + 1))
                self._flight_sum.add_(((self.airborne >= self.flight_steps) & ~bad).sum())
            self_contact = None
            if self.self_contact_force is not None:
                self_contact = task.self_contact_share(self.self_contact_force)
                self._self_contact_sum.add_(torch.where(bad, 0., self_contact).sum())
                self._self_contact_count += self.n_envs
            components = task.track_reward_components(
                errors, action, self.previous_action, self.qpos[:, self.qadr], self.soft_low, self.soft_high,
                self.track_weights, self_contact=self_contact, impact=impact, sole_impact=sole_impact)
            grace = (self.clip_kind == 2) & (self.lengths <= self.clip_grace)  # a bank start's limbs: not yet
            failed = task.track_terminated(robot[0], robot[1], reference[0], reference[1], self.track_anchor,
                                           self.track_ends, ~grace) & ~bad
            self.clip_reached.logical_or_(self.clip_clock >= self.clip_last)
            reward = torch.where(bad, 0., sum(components.values()))
            head = self._head()
            self.head_window[self._worlds, (self.lengths - 1) % self.head_steps] = torch.where(bad, 0., head)
            for key, value in errors.items():
                self._clip_errors[key].add_(torch.where(bad, 0., value.clamp(min=0).sqrt()).sum())
            self._hand_sum.add_(torch.where(bad, 0., self.hand_substeps.amax(dim=1)).sum())
            self._clip_count.add_((~bad).sum())
            self.previous_action.copy_(action)
            if unbounded is not None:
                self.previous_unbounded.copy_(unbounded)
            if self.delay_depth:
                self.delay_actions.copy_(queue[:, :-1])
            terminal_obs = torch.where(bad[:, None], 0., self._obs('push'))
            self.returns.add_(reward)
            terminated = bad | failed | (self.qvel[:, self.vadr].abs().amax(dim=1) > task.GETUP_MAX_JOINT_SPEED)
            truncated = (self.lengths >= self.episode_limit) | (self.clip_clock >= self.clip_end)
            done = terminated | truncated
            self._episodes.add_(done.sum())
            self._episodes_up.add_((done & ~bad & (head > task.GETUP_ASSIST_HEAD)).sum())
            self._bad_states.add_(bad.sum())
            kinds = torch.nn.functional.one_hot(self.clip_kind, len(task.TRACK_START_KINDS)).float()
            self._clip_ended.add_((kinds * done[:, None]).sum(0))
            self._clip_done.add_((kinds * (done & self.clip_reached & ~bad)[:, None]).sum(0))
            self._clip_failed.add_((done & failed).sum())
            standing = task.getup_standing(head, self.qpos[:, 2], self.xmat[:, self.torso].reshape(-1, 3, 3),
                                           task.sole_height(self.geom_xpos[:, self.soles, 2], self.sole_radius))
            self._clip_standing.add_((truncated & ~terminated & standing).sum())  # standing at the hold's end
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._overflow_seen.bitwise_or_(self.overflow)
            self._masked_reset(done)
            return torch.where(done[:, None], self._obs('fill', done), terminal_obs), reward, terminated, truncated, \
                terminal_obs

    @torch.no_grad()
    def _settle_step(self, actions):
        """settle_clip's step, the CPU task's _settle_step per world: the getting-up step's action, delay queue, PD and
        substeps, each substep's contact forces decoded (self_contact while weighted, the hands' floor force) and the
        impact / sole_impact speeds while weighted; then the errors against every target, the nearest one's reward
        (task.settle_reward_components), the success clock (task.settle_update), the logs, and the ended worlds
        reset."""
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._bad_state_seen.logical_or_(~torch.isfinite(actions).all())
            limit = self.reference.action_limit
            action = actions.clamp(-limit, limit)
            unbounded = task.lock_joints(action, self.wrist_mask) if self.previous_unbounded is not None else None
            if self.bounding:  # only a probe's set_pd_bound: the setting does not go with settle_clip
                position = self.frames[:, -1, self.position_entries]
                action = task.pd_bound(action, position, self.bound, self.peak, self.kp, self.scale, self.bound_free)
            action = task.lock_joints(action, self.wrist_mask)
            self.lift.copy_(task.assist_force(self.xmat[:, self.pelvis].reshape(-1, 3, 3), self.assist))
            if self.delay_depth:
                if self.bounding:
                    zero = task.lock_joints(task.pd_bound(torch.zeros_like(action), position, self.bound, self.peak,
                                                          self.kp, self.scale, self.bound_free), self.wrist_mask)
                    first = (self.lengths == 0)[:, None, None]
                    self.delay_actions.copy_(torch.where(first, zero[:, None], self.delay_actions))
                queue = torch.cat((action[:, None], self.delay_actions), dim=1)
            friction = dict(friction_scale=self.friction_scale) if self.motor_frictions else {}
            if self.sole_impact_substeps is not None:
                self.touching.zero_()
            for substep in range(self.reference.frame_skip):
                applied = queue[self._worlds, self.delay_lags[:, substep]] if self.delay_depth else action
                joint_velocity = self.qvel[:, self.vadr]
                torque = pd_torque(applied, self.qpos[:, self.qadr], joint_velocity, self.home, self.scale, self.kp,
                                   self.kd, self.joint_low, self.joint_high, -self.unclamped, self.unclamped, limit)
                torque = task.motor_torque(torque, joint_velocity, *self.motors, **friction).clamp(min=self.torque_low,
                                                                                                   max=self.torque_high)
                self.ctrl.copy_(torque)
                self.xfrc_applied[:, self.pelvis, 2] = self.lift
                self.wp.capture_launch(self.step_graph)
                self._overflow_seen.bitwise_or_(self.overflow)
                self._count_constraints()
                robot, _ = self._contact_forces()  # this substep's contact forces, decoded once
                if self.self_contact_force is not None:
                    self.self_contact_force[:, substep] = robot
                self.hand_substeps[:, substep] = self._hand_forces()
                if self.floor_entry is not None:
                    speeds = self._floor_speeds()
                    if self.impact_substeps is not None:
                        self.impact_substeps[:, substep] = speeds[:, :self.nbody]
                    if self.sole_impact_substeps is not None:
                        self.sole_impact_substeps[:, substep] = speeds[:, self.nbody:]
            self.xfrc_applied[:, self.pelvis, 2] = 0.
            self.wp.capture_launch(self.forward_graph)
            self.lengths.add_(1)
            bad = ~(torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1))
            position, joint_velocity = self.qpos[:, self.qadr], self.qvel[:, self.vadr]
            angle, pose, deviation = task.settle_errors(
                task.projected_gravity(self.xmat[:, self.pelvis].reshape(-1, 3, 3)), position, self.settle['gravity'],
                self.settle['position'], self.settle_joints)
            impact = sole_impact = None
            if self.impact_substeps is not None:
                speed = self.impact_substeps.amax(dim=1)
                impact = task.impact_penalty(speed)
                self._impact_sum.add_(torch.where(bad, 0., impact).sum())
                self._impact_hits.add_((task.impact_hit(speed) & ~bad).sum())
                self._impact_count += self.n_envs
            if self.sole_impact_substeps is not None:
                speed = self.sole_impact_substeps.amax(dim=1)
                sole_impact = task.impact_penalty(speed, task.GETUP_SOLE_IMPACT_SPEED)
                self._sole_impact_sum.add_(torch.where(bad, 0., sole_impact).sum())
                self._sole_impact_hits.add_((task.impact_hit(speed, task.GETUP_SOLE_IMPACT_SPEED) & ~bad).sum())
                self._sole_impact_count += self.n_envs
                self.airborne.copy_(torch.where(self.touching, 0, self.airborne + 1))
                self._flight_sum.add_(((self.airborne >= self.flight_steps) & ~bad).sum())
            self_contact = None
            if self.self_contact_force is not None:
                self_contact = task.self_contact_share(self.self_contact_force)
                self._self_contact_sum.add_(torch.where(bad, 0., self_contact).sum())
                self._self_contact_count += self.n_envs
            components, nearest, near = task.settle_reward_components(
                angle, pose, deviation, joint_velocity, self._velocity(), action, self.previous_action, position,
                self.soft_low, self.soft_high, self.settle_joints, self.settle_weights, self_contact=self_contact,
                impact=impact, sole_impact=sole_impact)
            # The success clock: steps on end near the nearest target and still; the first time it reaches
            # settle_hold is the world's success (its episode step and target), kept until its reset.
            self.settle_count.copy_(task.settle_update(self.settle_count, near & ~bad, joint_velocity,
                                                       self.settle_joints))
            fresh = (self.settle_at < 0) & (self.settle_count >= self.settle_hold) & ~bad
            self.settle_at.copy_(torch.where(fresh, self.lengths, self.settle_at))
            self.settle_target.copy_(torch.where(fresh, nearest, self.settle_target))
            reward = torch.where(bad, 0., sum(components.values()))
            head = self._head()
            self.head_window[self._worlds, (self.lengths - 1) % self.head_steps] = torch.where(bad, 0., head)
            nearest_angle = angle.gather(1, nearest[:, None])[:, 0]
            nearest_deviation = deviation.gather(1, nearest[:, None])[:, 0]
            self._settle_angle.add_(torch.where(bad, 0., nearest_angle).sum())
            self._settle_deviation.add_(torch.where(bad, 0., nearest_deviation).sum())
            self._hand_sum.add_(torch.where(bad, 0., self.hand_substeps.amax(dim=1)).sum())
            self._settle_steps.add_((~bad).sum())
            self.previous_action.copy_(action)
            if unbounded is not None:
                self.previous_unbounded.copy_(unbounded)
            if self.delay_depth:
                self.delay_actions.copy_(queue[:, :-1])
            terminal_obs = torch.where(bad[:, None], 0., self._obs('push'))
            self.returns.add_(reward)
            terminated = bad | (self.qvel[:, self.vadr].abs().amax(dim=1) > task.GETUP_MAX_JOINT_SPEED)
            truncated = self.lengths >= self.episode_limit
            done = terminated | truncated
            self._episodes.add_(done.sum())
            self._episodes_up.add_((done & ~bad & (head > task.GETUP_ASSIST_HEAD)).sum())
            self._bad_states.add_(bad.sum())
            settled = done & (self.settle_at >= 0)
            self._settle_ended.add_(done.sum())
            self._settle_done.add_(settled.sum())
            self._settle_seconds.add_(torch.where(settled, self.settle_at * self.dt, 0.).sum())
            targets = torch.nn.functional.one_hot(self.settle_target.clamp(min=0), len(self.settle_names)).float()
            self._settle_targets.add_((targets * settled[:, None]).sum(0))
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._overflow_seen.bitwise_or_(self.overflow)
            self._masked_reset(done)
            return torch.where(done[:, None], self._obs('fill', done), terminal_obs), reward, terminated, truncated, \
                terminal_obs

    def _settle_metrics(self):
        """rollout_metrics' settle_clip numbers, the counters zeroed: rollout/settle_done, the share of the ended
        episodes that settled (near the nearest target and still, the success g1_ctrl switches on); rollout/settle_seconds,
        their mean time to it (s); rollout/settle_target<k>, the share of them that settled into clip k's first frame
        (settle_clip's order); rollout/settle_angle and rollout/settle_deviation, the mean gravity angle (deg) and mean
        |q − q0| (rad) to the nearest target per world-step; rollout/hand_force, the mean of each world-step's largest
        floor force on the hands (N); rollout/self_contact while weighted."""
        ended, done = self._settle_ended.item(), self._settle_done.item()
        count = self._settle_steps.item()
        metrics = {'rollout/settle_done': done / ended if ended else float('nan'),
                   'rollout/settle_seconds': self._settle_seconds.item() / done if done else float('nan'),
                   'rollout/settle_angle': math.degrees(self._settle_angle.item() / count) if count else float('nan'),
                   'rollout/settle_deviation': self._settle_deviation.item() / count if count else float('nan'),
                   'rollout/hand_force': self._hand_sum.item() / count if count else float('nan')}
        for k, value in enumerate(self._settle_targets.cpu().tolist()):
            metrics[f'rollout/settle_target{k}'] = value / done if done else float('nan')
        if self.self_contact_force is not None:
            calls = self._self_contact_count
            metrics['rollout/self_contact'] = self._self_contact_sum.item() / calls if calls else float('nan')
            self._self_contact_sum.zero_()
            self._self_contact_count = 0
        for value in (self._settle_ended, self._settle_done, self._settle_seconds, self._settle_targets,
                      self._settle_angle, self._settle_deviation, self._hand_sum, self._settle_steps):
            value.zero_()
        return {key: float(value) for key, value in metrics.items()}

    def rollout_metrics(self):
        """Task numbers since the last call, for the training log (TensorPPO.collect reads them once per rollout):
        rollout/assist, the worlds' mean assist force now (N); rollout/getup_success, the share of episodes ended with
        the head above GETUP_ASSIST_HEAD (1 m; NaN without an ended episode); rollout/nefc_max and rollout/ncon_max,
        the most constraint rows (Warp allows njmax per world) and contacts of one world at any substep;
        rollout/bad_states, world-steps ended by a non-finite state. While the PD error bound acts (pd_bound):
        rollout/pd_bound, the worlds' mean bound c now; rollout/pd_bound_share, the share of free joint-steps (all but
        the locked wrists) whose action it changed. While the impact term is weighted: rollout/impact, its mean
        unweighted penalty per world-step (task.impact_penalty); rollout/impact_hit, the share of world-steps with a
        body above GETUP_IMPACT_SPEED (worlds ended by a non-finite state count as 0). While the sole_impact term is
        weighted: rollout/sole_impact and rollout/sole_impact_hit, the same for the feet (above
        GETUP_SOLE_IMPACT_SPEED); rollout/flight, the share of world-steps in a flight, the world without any floor
        contact in every substep of the last GETUP_FLIGHT_SECONDS (this step and the one before; within its episode)
        (worlds ended by a non-finite state count as 0). While the stance_width term is weighted: rollout/stance_width,
        the mean feet width (m) over the world-steps where the term counts (task.stance_active; NaN without one).
        While the arm_press term is weighted: rollout/arm_press, its mean unweighted Σ (rad²) over the standing
        world-steps (task.getup_standing; NaN without one). While the style term is weighted or judged (style_judge,
        the amp weight): rollout/style_dist, the mean d over the world-steps with the head in task.STYLE_HEAD_BAND (NaN
        without one), and rollout/style_band, the share of world-steps there (worlds ended by a non-finite state count
        as outside). While the head_rise term is weighted: rollout/head_rise, its mean unweighted excess (m/s) per
        world-step (a non-finite world's 0), and rollout/head_speed_p90, the 90th percentile of the head top's vertical
        speed over the world-steps with the head in task.STYLE_HEAD_BAND (NaN without one). While the sole_slip term is
        weighted: rollout/sole_slip, its mean unweighted Σ (m²/s²) per world-step (a non-finite world's 0), and
        rollout/sole_slip_speed, the median horizontal speed of the feet in contact (m/s). With human_starts:
        rollout/human_start_share, the share of the ended episodes that started from a human row, and
        rollout/getup_success_bank, rollout/getup_success of the others. With track_clip: rollout/track_done, the share
        of the ended episodes whose clip clock reached the clip's last frame (no tracking termination before it),
        rollout/track_done_clip of those started from the clip's first frame (TRACK_START_KINDS' 'clip'),
        rollout/track_failed the share ended by a tracking termination, rollout/track_standing the share that ended by
        the hold's end (truncated, not terminated) standing (task.getup_standing: head top above 1.15 m, torso tilt
        below 15°, pelvis above 0.70 m, both soles down; track_done passes a robot left on one knee within the
        anchor's 0.25 m); rollout/track_err_<term>, the mean √e of each
        tracking error per world-step (m, rad, m/s, rad/s; track_errors); rollout/hand_force, the mean of each
        world-step's largest floor force on the hands (N); rollout/self_contact while weighted. With settle_clip:
        _settle_metrics' rollout/settle_done, settle_seconds, settle_target<k>, settle_angle, settle_deviation,
        hand_force and self_contact."""
        if self.clip is not None:
            clip = self._clip_metrics()
        elif self.settle is not None:
            clip = self._settle_metrics()
        episodes = self._episodes.item()
        metrics = {'rollout/assist': self.assist.mean().item(),
                   'rollout/getup_success': self._episodes_up.item() / episodes if episodes else float('nan'),
                   'rollout/nefc_max': int(self._nefc_max.item()), 'rollout/ncon_max': int(self._ncon_max.item()),
                   'rollout/bad_states': int(self._bad_states.item())}
        if self.human_share is not None:
            human = self._episodes_human.item()
            metrics['rollout/human_start_share'] = human / episodes if episodes else float('nan')
            metrics['rollout/getup_success_bank'] = (self._episodes_bank_up.item() / (episodes - human)
                                                     if episodes - human else float('nan'))
            self._episodes_human.zero_(), self._episodes_bank_up.zero_()
        if self.bounding:
            metrics['rollout/pd_bound'] = self.bound.mean().item()
            metrics['rollout/pd_bound_share'] = (self._bounded.item() / self._bounded_count if self._bounded_count
                                                 else float('nan'))
        self._bounded_count = 0
        for value in (self._episodes, self._episodes_up, self._bad_states, self._nefc_max, self._ncon_max,
                      self._bounded):
            value.zero_()
        if self.impact_substeps is not None:
            count = self._impact_count
            metrics['rollout/impact'] = self._impact_sum.item() / count if count else float('nan')
            metrics['rollout/impact_hit'] = self._impact_hits.item() / count if count else float('nan')
            self._impact_sum.zero_(), self._impact_hits.zero_()
            self._impact_count = 0
        if self.sole_impact_substeps is not None:
            count = self._sole_impact_count
            metrics['rollout/sole_impact'] = self._sole_impact_sum.item() / count if count else float('nan')
            metrics['rollout/sole_impact_hit'] = self._sole_impact_hits.item() / count if count else float('nan')
            metrics['rollout/flight'] = self._flight_sum.item() / count if count else float('nan')
            self._sole_impact_sum.zero_(), self._sole_impact_hits.zero_(), self._flight_sum.zero_()
            self._sole_impact_count = 0
        if self.stance_weighted:
            count = self._stance_count.item()
            metrics['rollout/stance_width'] = self._stance_sum.item() / count if count else float('nan')
            self._stance_sum.zero_(), self._stance_count.zero_()
        if self.press_weighted:
            count = self._press_count.item()
            metrics['rollout/arm_press'] = self._press_sum.item() / count if count else float('nan')
            self._press_sum.zero_(), self._press_count.zero_()
        if self.slip_weighted:
            count = self._slip_count
            metrics['rollout/sole_slip'] = self._slip_sum.item() / count if count else float('nan')
            speeds = torch.cat(self._slip_speeds) if self._slip_speeds else torch.zeros(0, device=self.device)
            speeds = speeds[~speeds.isnan()]
            metrics['rollout/sole_slip_speed'] = speeds.median().item() if len(speeds) else float('nan')
            self._slip_sum.zero_()
            self._slip_count, self._slip_speeds = 0, []
        if self.rise_weighted:
            count = self._rise_count
            metrics['rollout/head_rise'] = self._rise_sum.item() / count if count else float('nan')
            speeds = torch.cat(self._band_speeds) if self._band_speeds else torch.zeros(0, device=self.device)
            speeds = speeds[~speeds.isnan()]
            metrics['rollout/head_speed_p90'] = torch.quantile(speeds, .9).item() if len(speeds) else float('nan')
            self._rise_sum.zero_()
            self._rise_count, self._band_speeds = 0, []
        if self.style_judged:
            band, count = self._style_band.item(), self._style_count
            metrics['rollout/style_dist'] = self._style_sum.item() / band if band else float('nan')
            metrics['rollout/style_band'] = band / count if count else float('nan')
            self._style_sum.zero_(), self._style_band.zero_()
            self._style_count = 0
        if self.clip is not None or self.settle is not None:
            metrics.update(clip)
        return metrics

    def _clip_metrics(self):
        """rollout_metrics' track_clip numbers (read before the shared episode counters are zeroed), the counters
        zeroed."""
        ended, done = self._clip_ended.cpu().numpy(), self._clip_done.cpu().numpy()
        count = self._clip_count.item()
        episodes = ended.sum()
        metrics = {'rollout/track_done': done.sum() / episodes if episodes else float('nan'),
                   'rollout/track_done_clip': done[0] / ended[0] if ended[0] else float('nan'),
                   'rollout/track_failed': self._clip_failed.item() / episodes if episodes else float('nan'),
                   'rollout/track_standing': self._clip_standing.item() / episodes if episodes else float('nan'),
                   'rollout/hand_force': self._hand_sum.item() / count if count else float('nan')}
        for key, value in self._clip_errors.items():
            metrics[f'rollout/track_err_{key}'] = value.item() / count if count else float('nan')
            value.zero_()
        if self.self_contact_force is not None:
            calls = self._self_contact_count
            metrics['rollout/self_contact'] = self._self_contact_sum.item() / calls if calls else float('nan')
            self._self_contact_sum.zero_()
            self._self_contact_count = 0
        for value in (self._clip_ended, self._clip_done, self._clip_failed, self._clip_standing, self._hand_sum,
                      self._clip_count):
            value.zero_()
        return {key: float(value) for key, value in metrics.items()}

