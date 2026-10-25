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
stance_width (after branch 46): each world's feet width after the step (task.stance_width), as on the CPU.

Episodes out of step: a full reset() cuts each world's first episode at a random step of 1..max_episode_steps
(episode_limit; rsl_rl's init_at_random_ep_len), every later one runs the whole length. Without it the worlds, which
hardly ever end early here, kept their episodes in lockstep: all of them ended together every 500 steps, so only about
one rollout in 21 (24 steps each) had an episode end, and the critic saw every world at the same point of its episode
(the first root, exp-9e0bafcc272d4b51: 42 of 290 rollouts with an ep_len_mean, 24.10).
"""
import numpy as np
import torch

from .g1_tensor_env import ENCODER_BIAS, TensorG1Env
from .tasks import g1_tensor_math as task
from .tasks.config import G1_GETUP_DEFAULTS, G1_GETUP_PROFILE
from .tasks.g1_getup import EPISODE_STEPS, G1GetupEnv, saved_bounds
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
                 action_change_unbounded=G1_GETUP_DEFAULTS['action_change_unbounded']):
        """The CPU task's settings (task_config: reward_weights, assist_force, action_delay, motor_friction,
        body_friction, pd_bound, stand_pose_std, action_change_unbounded; bank as G1GetupEnv's); observation_noise and
        randomize as TensorG1Env's (tests switch them off to compare with the CPU task)."""
        self._getup = dict(reward_weights=reward_weights, assist_force=assist_force, action_delay=action_delay,
                           motor_friction=motor_friction, body_friction=list(body_friction), pd_bound=pd_bound,
                           stand_pose_std=stand_pose_std, action_change_unbounded=action_change_unbounded, bank=bank)
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
        # Each world's episode length: max_episode_steps, the first episode after a full reset() a random one.
        self.episode_limit = torch.full((n_envs,), self.max_episode_steps, dtype=torch.long, device=device)
        # Reward constants of the CPU task: the legs' soft joint limits, the weights, the pose term's σ (a float).
        self.soft_low, self.soft_high = constant(ref.soft_low), constant(ref.soft_high)
        self.limit_joints = range(task.LEGS)
        self.weights = ref.getup_weights
        self.stand_pose_std = ref.stand_pose_std
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
        self.stance_weighted = bool(self.weights['stance_width'])
        if self.stance_weighted:
            self._stance_sum = torch.zeros((), device=device)
            self._stance_count = torch.zeros((), device=device)
        # Episodes ended since the last rollout_metrics, those with the head above GETUP_ASSIST_HEAD at the end, and
        # world-steps ended by a non-finite state.
        self._episodes = torch.zeros((), device=device)
        self._episodes_up = torch.zeros((), device=device)
        self._bad_states = torch.zeros((), device=device)

    def _make_reference(self, max_episode_steps, **settings):
        """G1GetupEnv with this task's settings; it fixes the robot and connector (config.G1_GETUP_PROFILE) itself."""
        return G1GetupEnv(max_episode_steps, **self._getup)

    def _batched_fields(self):
        """The body friction is drawn per world at every reset, so geom_friction is always batched; with randomize the
        pelvis COM too (set once per world, as the walking task)."""
        return {'geom_friction': self.n_envs, **({'body_ipos': self.n_envs} if self.randomize else {})}

    def _head(self):
        return task.head_height(self.xpos[:, self.torso], self.xmat[:, self.torso].reshape(-1, 3, 3))

    def _obs(self, history='replace', mask=None):
        """The walking task's actor inputs (command 0, phase stopped), then task.getup_critic_extras."""
        actor = super()._obs(history, mask)[:, :self.reference.actor_inputs]
        extras = task.getup_critic_extras(self._velocity(), self.qpos[:, 2], self._head(), self.assist)
        return torch.cat((actor, extras), dim=1)

    def _masked_reset(self, mask, rows=None):
        """Resets the masked worlds: the episode that ended steps its assist down first (task.assist_update on the
        mean head height of its last GETUP_ASSIST_SECONDS; NaN, no step, for a shorter one), then a start from the
        bank (rows: (n_envs,) bank rows; None: a training row drawn per world), and the episode's delay, motor
        friction and body friction, as the CPU task's reset."""
        full = self.lengths >= self.head_steps
        head_mean = torch.where(full, self.head_window.mean(dim=1), torch.full_like(self.assist, float('nan')))
        self.assist.copy_(torch.where(mask, task.assist_update(self.assist, head_mean), self.assist))
        if self.pd_bound is not None:  # the bound by the same rule (no draw); nothing without the setting
            stepped = task.pd_bound_update(self.bound, head_mean, self.pd_bound[1])
            self.bound.copy_(torch.where(mask, stepped, self.bound))
        self.reset_mask.copy_(mask)
        self.wp.capture_launch(self.reset_graph)  # also zeroes the reset worlds' xfrc_applied
        if rows is None:
            draw = torch.randint(0, len(self.training_rows), (self.n_envs,), device=self.device,
                                 generator=self.generator)
            rows = self.training_rows[draw]
        positions = self.bank_qpos[rows].clone()
        positions[:, 3:7] /= positions[:, 3:7].norm(dim=1, keepdim=True)
        self.qpos.copy_(torch.where(mask[:, None], positions, self.qpos))
        self.qvel.copy_(torch.where(mask[:, None], self.bank_qvel[rows], self.qvel))
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

    @torch.no_grad()
    def reset(self):
        """Every world from the bank (TensorG1Env.reset), its first episode cut at a random step of
        1..max_episode_steps (episode_limit, drawn after the reset's draws), so the worlds' episodes end spread out."""
        obs = super().reset()
        self.episode_limit.copy_(torch.randint(1, self.max_episode_steps + 1, (self.n_envs,), device=self.device,
                                               generator=self.generator))
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
                counted = task.stance_active(self.qpos[:, 2], sole) & ~bad
                self._stance_sum.add_(torch.where(counted, width, 0.).sum())
                self._stance_count.add_(counted.sum())
            # action_change: the applied action and the one before, under action_change_unbounded the action before
            # pd_bound and its own previous one.
            changed, before = ((action, self.previous_action) if unbounded is None
                               else (unbounded, self.previous_unbounded))
            components, _ = task.getup_reward_components(
                head, self.qpos[:, 2], rotation, self._velocity(), self.qpos[:, self.qadr], self.home, self.soft_low,
                self.soft_high, self.limit_joints, sole, self.ctrl, self.qvel[:, self.vadr], changed, before,
                weights=self.weights, impact=impact, stand_pose_std=self.stand_pose_std, sole_impact=sole_impact,
                stance_width=width)
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
            self._bad_states.add_(bad.sum())
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._overflow_seen.bitwise_or_(self.overflow)
            self._masked_reset(done)
            # Worlds that reset start a fresh history filled with their first frame; the others keep theirs.
            return torch.where(done[:, None], self._obs('fill', done), terminal_obs), reward, terminated, truncated, \
                terminal_obs

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
        the mean feet width (m) over the world-steps where the term counts (task.stance_active; NaN without one)."""
        episodes = self._episodes.item()
        metrics = {'rollout/assist': self.assist.mean().item(),
                   'rollout/getup_success': self._episodes_up.item() / episodes if episodes else float('nan'),
                   'rollout/nefc_max': int(self._nefc_max.item()), 'rollout/ncon_max': int(self._ncon_max.item()),
                   'rollout/bad_states': int(self._bad_states.item())}
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
        return metrics

