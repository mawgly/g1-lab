"""G1 Warp physics with CUDA task state. Experimental until GPU parity gates pass.

PD is recomputed every physics substep. Each physics step is a Warp graph;
Torch dispatch between steps is deliberately retained for correctness first.
"""
import numpy as np
import torch
from .tasks.config import CROUCH_HEIGHTS
from .tasks.g1_velocity import G1VelocityEnv
from .tasks import g1_tensor_math as task
from .tasks.g1_tensor_math import observation, reward_components, pd_torque, body_velocity

# Real-robot training noise and randomisation (mjlab velocity task; docs/g1-real-robot-plan.md).
# Uniform observation noise per deploy-observation block, joint encoder bias per episode.
NOISE = dict(angular_velocity=.2, gravity=.05, joint_position=.01, joint_velocity=1.5)
ENCODER_BIAS = .015
FOOT_FRICTION_RANGE = (.3, 1.2)
COM_OFFSET = (.025, .025, .03)  # pelvis centre of mass, m
# Pushes on average every 2 s (mjlab: 1-3 s): linear x, y, z (m/s) and angular roll, pitch, yaw (rad/s).
REAL_PUSH_INTERVAL = 2.
REAL_PUSH = (.5, .5, .4, .52, .52, .78)


class TensorG1Env:
    backend_metadata = dict(backend='mujoco-warp-torch', device='cuda:0', precision='float32',
                            solver='Newton', integrator='implicitfast', task='g1_velocity',
                            variant='CUDA reset RNG; float32; per-substep PD; random pelvis pushes')

    def __init__(self, n_envs, seed=0, max_episode_steps=1000, nconmax=128, njmax=512, reward_weights=None,
                 resample_commands=True, push_velocity=task.PUSH_VELOCITY, heading_hold=False, commands=None,
                 legs_only=False, gait=None, action_limit=1., real_robot=False, observation_noise=None,
                 randomize=None, critic_extras=False, training_realism=1., history=1, stand_heading_hold=True,
                 phase_stop=False, sole_contact=False, random_phase=False, action_delay=0., arm_perturb=0.,
                 lock_wrists=False, knee_floor=None, force_push=0., feet_flat_stance=False, foot_collision=False,
                 side_heading_hold=True, motor_friction=0., height_command=False, upper_body_slot=False,
                 arm_perturb_radians=task.ARM_PERTURB_RADIANS, arm_perturb_contact_mask=False, arm_blind=False,
                 hand_collision=False, arm_targets=False, load_curve=None):
        if type(n_envs) is not int or n_envs < 1 or type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('n_envs and max_episode_steps must be positive integers')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch is required; use .venv-fullgpu')
        import mujoco_warp as mjw
        import warp as wp
        self.mjw, self.wp = mjw, wp
        self.device = torch.device('cuda:0')
        self.n_envs = self.num_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self.reference = ref = self._make_reference(
            max_episode_steps, reward_weights=reward_weights, heading_hold=heading_hold, commands=commands,
            legs_only=legs_only, gait=gait, action_limit=action_limit, real_robot=real_robot,
            critic_extras=critic_extras, training_realism=training_realism, history=history,
            stand_heading_hold=stand_heading_hold, phase_stop=phase_stop, sole_contact=sole_contact,
            random_phase=random_phase, action_delay=action_delay, arm_perturb=arm_perturb, lock_wrists=lock_wrists,
            knee_floor=knee_floor, force_push=force_push, feet_flat_stance=feet_flat_stance,
            foot_collision=foot_collision, side_heading_hold=side_heading_hold, motor_friction=motor_friction,
            height_command=height_command, upper_body_slot=upper_body_slot, arm_perturb_radians=arm_perturb_radians,
            arm_perturb_contact_mask=arm_perturb_contact_mask, arm_blind=arm_blind, hand_collision=hand_collision,
            arm_targets=arm_targets, load_curve=load_curve)
        # Warp gets the reference's model (put_model below), so foot_collision's capsules, their contact filter and
        # the sole spheres are the CPU task's; a foot-foot contact is a robot-robot one for self_contact. So are
        # hand_collision's capsules (a hand on the body is a robot-robot contact too).
        # Curriculum: 0 = no noise, randomisation or pushes; 1 = Unitree's full training setup.
        self.realism = float(training_realism)
        self.heading_hold = bool(heading_hold)
        self.real_robot = ref.real_robot
        # Training-time realism of the real-robot profile; tests switch it off to compare with the CPU task.
        self.observation_noise = self.real_robot if observation_noise is None else bool(observation_noise)
        self.randomize = (self.real_robot if randomize is None else bool(randomize)) and self.real_robot
        self.observation_space, self.action_space = ref.observation_space, ref.action_space
        self.dt = ref.dt
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        def constant(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        self.qadr, self.vadr = constant(ref.qadr, torch.long), constant(ref.vadr, torch.long)
        for name in ('home', 'scale', 'kp', 'kd'):
            setattr(self, name, constant(getattr(ref, name)))
        # PD target bounds (pd_torque, arm_press): the joint ranges, with knee_floor the knees' lower one raised to it
        # (the CPU task's target_low); the action and the observed previous action are not raised, as in g1_ctrl.
        self.joint_low, self.joint_high = constant(ref.target_low), constant(ref.target_high)
        self.torque_low, self.torque_high = constant(ref.model.actuator_ctrlrange[:,0]), constant(ref.model.actuator_ctrlrange[:,1])
        self.pelvis = ref.pelvis
        self.torso = ref.torso
        self.feet = constant(ref.feet, torch.long)
        self.soles, self.sole_radius = constant(ref.soles, torch.long), constant(ref.sole_radius)  # (2, k) spheres
        self.posture_joints = constant(ref.posture_joints, torch.long)
        self.hip_joints = constant(ref.hip_joints, torch.long)
        self.arm_joints = constant(ref.arm_joints, torch.long)  # arm_press term (none on the legs-only G1)
        self.motors = None if ref.motors is None else tuple(constant(row) for row in ref.motors)
        self.unclamped = torch.full_like(self.torque_low, float('inf'))
        self.encoder_bias = torch.zeros((n_envs, ref.model.nu), device=self.device)
        # Observation history (real robot): the last frames the robot measured, oldest first, as on the CPU task.
        self.history = ref.history
        self.frames = torch.zeros((n_envs, self.history, ref.frame_size), device=self.device)
        self.resample_commands = resample_commands
        self.push_velocity = push_velocity
        self.root = int(ref.model.body_rootid[self.pelvis])
        ref.reset(seed=seed)
        self.initial_qpos = constant(ref.data.qpos)
        self.initial_qpos[self.qadr] = self.home
        self.command = torch.zeros((n_envs,3),device=self.device)
        self.observed_command = torch.zeros_like(self.command)
        # Reset pose has identity orientation, so the heading to hold is world +x for every world.
        self.heading_target = torch.zeros(n_envs,device=self.device)
        # yaw_drift reward: mean yaw-rate error under each world's command (task.yaw_drift_update), as on the CPU
        # task; updated only while that weight is non-zero, zeroed at reset and when a world's command is resampled.
        # Its mean |e| on straight commands is summed for rollout_metrics().
        self.yaw_drift_error = torch.zeros(n_envs,device=self.device)
        self._yaw_drift_sum = torch.zeros((),device=self.device)
        self._yaw_drift_count = torch.zeros((),device=self.device)
        # self_contact: summed share of each world-step's substeps with a self-contact above the threshold.
        self._self_contact_sum = torch.zeros((),device=self.device)
        self._self_contact_count = 0
        # foot_collision: world-steps with a foot-foot contact above the threshold, for rollout/foot_contact.
        self._foot_contact_sum = torch.zeros((),device=self.device)
        self._foot_contact_count = 0
        # arm_press: summed Σ over the arm joints of (PD target − angle)² of each world-step, while it is weighted.
        self._arm_press_sum = torch.zeros((),device=self.device)
        self._arm_press_count = 0
        self.previous_action = torch.zeros((n_envs,ref.model.nu),device=self.device)
        self.lengths = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        # Gait clock start of each world's episode (random_phase: a random step of the cycle, else 0). Every episode
        # used to start at phase 0, left foot first, so a walk started half a cycle on was never trained.
        self.period_steps = round(ref.gait['period']/ref.dt)
        self.phase_offset = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        # action_delay: each world's delay k (physics substeps, drawn at its reset from 0..max_delay_steps), the age
        # of the action each substep's target comes from (task.action_lags) and the older actions still in flight,
        # newest first. The substeps are separate graph launches with the PD in Torch, so k differs per world.
        self.delay_depth = -(-ref.max_delay_steps // ref.frame_skip)
        self.delay_steps = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        self.delay_lags = torch.zeros((n_envs,ref.frame_skip),dtype=torch.long,device=self.device)
        self.delay_actions = torch.zeros((n_envs,self.delay_depth,ref.model.nu),device=self.device)
        # arm_perturb: per world, the control steps left in its perturbation segment (0: none) and the segment's held
        # action (task.arm_perturb_draw / arm_perturb_apply), drawn with self.generator, ended by _masked_reset.
        # set_arm_perturb (tests) also runs this path without the setting; then no segment ever starts by itself.
        self.perturb_arms = ref.arm_perturb > 0
        self.arm_mask = constant(ref.arm_mask,torch.bool)
        self.perturb_steps = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        self.perturb_action = torch.zeros((n_envs,ref.model.nu),device=self.device)
        self._arm_perturb_sum = torch.zeros((),device=self.device)  # perturbed world-steps, for rollout_metrics
        self._arm_perturb_count = 0
        # arm_perturb_contact_mask: per world, whether this control step's arms follow a segment, so the contacts
        # touching an arm link leave its self_contact share (task.self_contact_pairs; the CPU task's arm_contacts_masked).
        self.arm_contacts_masked = torch.zeros(n_envs,dtype=torch.bool,device=self.device)
        # lock_wrists: the wrist pitch and yaw joints, whose applied action is always 0 (task.lock_joints).
        self.lock_wrists = ref.lock_wrists
        self.wrist_mask = constant(ref.wrist_mask,torch.bool)
        # force_push: per world, the control steps to its next push, the substeps of the running push still to come
        # (0: none) and its world force (task.force_push_draw / force_push_substeps), drawn with self.generator and
        # written into Warp's xfrc_applied of the pelvis before every physics substep: the captured step graph reads
        # that array in place (fwd_acceleration's xfrc_accumulate), as it reads ctrl. set_force_push (tests, probes)
        # also runs this path without the setting; then no push ever starts by itself.
        self.force_pushes = ref.force_push > 0
        self.push_body = ref.push_body
        self.push_timer = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        self.push_left = torch.zeros(n_envs,dtype=torch.long,device=self.device)
        self.push_force = torch.zeros((n_envs,3),device=self.device)
        # motor_friction: per world, the multiplier m of the motor model's joint friction (1: as measured), drawn with
        # self.generator at its reset (task.motor_friction_draw) and handed to task.motor_torque every substep.
        # set_motor_friction (tests, probes) also runs this path without the setting; then a reset puts m back to 1.
        self.motor_frictions = ref.motor_friction > 0
        self.friction_scale = torch.ones(n_envs,device=self.device)
        # height_command: per world the pelvis height command c, drawn with self.generator at its reset and at each
        # command change (task.height_draw, after the command's draws), an observation term and the reward's height,
        # fall and stand pose (task.stand_target on the crouch joints); set_height (tests, probes) fixes it. BASE_HEIGHT
        # and unused without the setting. Crouched world-steps and their pelvis above c, for rollout_metrics.
        self.crouch = ref.height_command
        self.height_command = torch.full((n_envs,),task.BASE_HEIGHT,device=self.device)
        self.crouch_joints = constant(ref.crouch_joints,torch.long)
        self.crouch_settings = {key: ref.commands[key] for key in ('crouch_share','crouch_min','crouch_speed')}
        self._crouch_sum = torch.zeros((),device=self.device)
        self._height_excess_sum = torch.zeros((),device=self.device)
        self._crouch_count = 0
        # Fixed normalization of the extra terms and of arm_blind's or arm_targets' arm columns (the CPU task's), for
        # TensorNormalize. arm_blind changes nothing else here: the observation is the same, the blindness is the
        # normalization's.
        self.fixed_normalization = ref.fixed_normalization
        # arm_targets: per world the episode's arm mode, flag (the arms follow the slot), whether set_arm_targets set the
        # targets (the generator stops), the target segment (start, end, clock, move, hold; task.arm_target_advance),
        # this step's offsets d, the flag and offsets of the step before (the observed last action) and those the delay
        # queue holds, the hand loads (kg; their shares of the ceiling load_max, which rollout_metrics moves by
        # task.load_curve_update, and whether set_hand_load fixed them), all drawn with a generator of their own
        # (arm_generator), so self.generator's stream is the old one. Flag-1 world-steps, their tracking and mean |error|
        # and the flag-1 falls since the last rollout_metrics.
        self.arm_targets = ref.arm_targets
        if self.arm_targets:
            if ref.hand_load.any():  # the reference's model goes to Warp as it is: without a load
                ref.set_hand_load(0.)
            self.arm_generator = torch.Generator(device=self.device).manual_seed(seed + task.ARM_TARGET_STREAM)
            nu = ref.model.nu
            self.track_joints = constant(ref.track_joints, torch.long)
            self.arm_posture_joints = constant(ref.arm_posture_joints, torch.long)
            self.arm_poses, self.arm_carry = constant(ref.arm_poses), constant(ref.arm_carry)
            self.arm_mode = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.arm_flag = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            self.arm_fixed = torch.zeros_like(self.arm_flag)
            self.arm_start, self.arm_end = torch.zeros((n_envs, nu), device=self.device), torch.zeros((n_envs, nu),
                                                                                                       device=self.device)
            self.arm_clock = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.arm_move, self.arm_hold = torch.ones_like(self.arm_clock), torch.zeros_like(self.arm_clock)
            self.arm_offset = torch.zeros((n_envs, nu), device=self.device)
            self.previous_flag, self.previous_offset = torch.zeros_like(self.arm_flag), torch.zeros_like(self.arm_offset)
            self.delay_offsets = torch.zeros_like(self.delay_actions)
            self.hand_bodies = constant(ref.hand_bodies, torch.long)
            self.hand_mass, self.hand_ipos = constant(ref.hand_mass), constant(ref.hand_ipos)
            self.subtree_mass, self.hand_above = constant(ref.subtree_mass), constant(ref.hand_above)
            self.hand_load, self.load_share = torch.zeros((n_envs, 2), device=self.device), torch.zeros((n_envs, 2),
                                                                                                        device=self.device)
            self.load_fixed = torch.zeros_like(self.arm_flag)
            self.load_max = 0.
            self._arm_steps = torch.zeros((), device=self.device)
            self._arm_track_sum = torch.zeros((), device=self.device)
            self._arm_error_sum = torch.zeros((), device=self.device)
            self._hold_steps = torch.zeros((), device=self.device)  # flag-1 world-steps on a hold, rollout/arm_error_hold
            self._hold_error_sum = torch.zeros((), device=self.device)
            self._arm_falls = torch.zeros((), device=self.device)
            # Flag-0 world-steps and falls since the last rollout_metrics (rollout/free_falls, logged only); with
            # task_config load_curve the block of windows the ceiling decides on (task.LoadCurveBlock, None without).
            self._free_steps = torch.zeros((), device=self.device)
            self._free_falls = torch.zeros((), device=self.device)
            self.load_block = None if ref.load_curve is None else task.LoadCurveBlock(**ref.load_curve)
        self._worlds = torch.arange(n_envs,device=self.device)
        self.returns = torch.zeros(n_envs,device=self.device)
        self.last_episode_lengths, self.last_episode_returns = self.lengths.clone(), self.returns.clone()
        self.reset_mask = torch.zeros(n_envs,dtype=torch.bool,device=self.device)
        wp.init()
        self._stream = wp.Stream('cuda:0')
        torch.cuda.synchronize(self.device)
        with wp.ScopedStream(self._stream):
            self.model = mjw.put_model(ref.model, batch_sizes=self._batched_fields())
            if self.randomize:
                # Once per world, as mjlab's startup events: foot friction shared by both feet, pelvis COM offset.
                friction = wp.to_torch(self.model.geom_friction)
                nominal = float(ref.model.geom_friction[ref.foot_geoms[0], 0])
                low, high = (nominal + (bound-nominal)*self.realism for bound in FOOT_FRICTION_RANGE)
                foot = constant(ref.foot_geoms, torch.long)
                friction[:, foot, 0] = (low + (high-low)*torch.rand((n_envs, 1), device=self.device,
                                                                     generator=self.generator))
                ipos = wp.to_torch(self.model.body_ipos)
                offset = ((2*torch.rand((n_envs, 3), device=self.device, generator=self.generator)-1)
                          * constant(COM_OFFSET) * self.realism)
                ipos[:, self.pelvis] += offset
            if self.arm_targets:  # the per-world model arrays the hand loads go into, read in place by the graphs
                self.body_mass = wp.to_torch(self.model.body_mass)
                self.body_ipos = wp.to_torch(self.model.body_ipos)
                self.body_subtreemass = wp.to_torch(self.model.body_subtreemass)
            self.data = mjw.make_data(ref.model,nworld=n_envs,nconmax=nconmax,njmax=njmax)
            mjw.forward(self.model,self.data)
            mjw.step(self.model,self.data)
            wp.synchronize_device()
            with wp.ScopedCapture() as capture:
                mjw.step(self.model,self.data)
            self.step_graph = capture.graph
            self._warp_reset_mask = wp.from_torch(self.reset_mask)
            mjw.reset_data(self.model,self.data,self._warp_reset_mask)
            with wp.ScopedCapture() as capture:
                mjw.reset_data(self.model,self.data,self._warp_reset_mask)
            self.reset_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mjw.forward(self.model,self.data)
            self.forward_graph = capture.graph
            for name in ('qpos','qvel','ctrl','cvel','xpos','xipos','xmat','ximat','subtree_com','geom_xpos','overflow',
                         'xfrc_applied'):
                setattr(self,name,wp.to_torch(getattr(self.data,name)))
            # self_contact reward (only while weighted): after each physics step mjw.contact_force decodes every
            # contact slot's constraint force (contact frame, as mj_contactForce on the CPU task) and Torch keeps,
            # per world, the largest one between two robot geoms. Contacts of all worlds share one array, so slots
            # past nacon (stale) and pairs touching the floor are masked out without a host sync. With foot_collision
            # the same decode also keeps, per world, the largest contact between a left and a right foot capsule, for
            # rollout/foot_contact (logged only; the reward charges foot-foot contacts through self_contact).
            self.self_contact_force = None
            self.foot_contact_force = None
            if ref.reward_weights['self_contact'] or ref.foot_collision:
                naconmax = self.data.naconmax
                self._contact_ids = wp.array(np.arange(naconmax,dtype=np.int32),dtype=int)
                self._contact_force = wp.zeros(naconmax,dtype=wp.spatial_vector)
                mjw.contact_force(self.model,self.data,self._contact_ids,False,self._contact_force)
                wp.synchronize_device()
                with wp.ScopedCapture() as capture:
                    mjw.contact_force(self.model,self.data,self._contact_ids,False,self._contact_force)
                self.contact_force_graph = capture.graph
                self.contact_force = wp.to_torch(self._contact_force)  # (naconmax, 6): force, then torque
                self.contact_geom = wp.to_torch(self.data.contact.geom)  # (naconmax, 2)
                self.contact_world = wp.to_torch(self.data.contact.worldid)
                self.nacon = wp.to_torch(self.data.nacon)
                self._contact_slots = torch.arange(naconmax,device=self.device)
                self.robot_geoms = constant(ref.robot_geoms,torch.bool)
                # arm_perturb_contact_mask: the arm links' geoms, whose contacts a perturbed world does not count.
                self.arm_geoms = constant(ref.arm_geoms,torch.bool) if ref.arm_perturb_contact_mask else None
                # Largest robot-robot contact force at each substep of the current step, (n_envs, frame_skip).
                if ref.reward_weights['self_contact']:
                    self.self_contact_force = torch.zeros((n_envs,ref.frame_skip),device=self.device)
                if ref.foot_collision:  # per geom: +1 a left foot capsule, -1 a right one, 0 the rest
                    side = np.zeros(ref.model.ngeom,dtype=np.int64)
                    side[ref.foot_capsules[0]], side[ref.foot_capsules[1]] = 1, -1
                    self.foot_side = constant(side,torch.long)
                    # Largest foot-foot contact force at each substep of the current step, (n_envs, frame_skip).
                    self.foot_contact_force = torch.zeros((n_envs,ref.frame_skip),device=self.device)
            wp.synchronize_device()
        self._overflow_seen = torch.zeros_like(self.overflow)
        self._bad_state_seen = torch.zeros((),dtype=torch.bool,device=self.device)

    def _make_reference(self, max_episode_steps, **settings):
        """The CPU task this one batches (model, constants, settings); the g1_getup task builds its own
        (g1_getup_tensor_env)."""
        return G1VelocityEnv(max_episode_steps, **settings)

    def _batched_fields(self):
        """Model fields Warp keeps per world (put_model batch_sizes): with randomize the foot friction and the pelvis
        COM, set once per world below; with arm_targets the bodies' mass, COM and subtree mass (the hand loads, set at
        resets and when the load ceiling moves); None: one model for all worlds."""
        fields = {'geom_friction': self.n_envs, 'body_ipos': self.n_envs} if self.randomize else {}
        if self.arm_targets:
            fields.update(body_mass=self.n_envs, body_ipos=self.n_envs, body_subtreemass=self.n_envs)
        return fields or None

    def _velocity(self):
        return body_velocity(self.cvel[:,self.pelvis],self.xipos[:,self.pelvis],
                             self.subtree_com[:,self.root],self.ximat[:,self.pelvis].reshape(-1,3,3))

    def _phase(self):
        return task.gait_phase(self.lengths+self.phase_offset, self.dt, self.reference.gait['period'])

    def _largest_contacts(self, selected):
        """(n_envs,) per world the largest contact force norm (N) at the last physics step over the contact slots
        where `selected` (naconmax,) holds, 0 without one."""
        force = torch.where(selected, self.contact_force[:,:3].norm(dim=1), 0.)
        world = torch.where(selected, self.contact_world.long(), self.n_envs)  # the rest goes to a spare slot
        largest = torch.zeros(self.n_envs+1,device=self.device)
        return largest.scatter_reduce_(0,world,force,'amax')[:-1]

    def _contact_forces(self):
        """Decodes the last physics step's contact forces once: (robot-robot, foot-foot) per world, each the largest
        contact force norm (N) between two robot geoms (the GPU side of the CPU task's _self_contact_force; None
        unless self_contact is weighted; with arm_perturb_contact_mask a world in an arm perturbation segment leaves out
        the contacts touching an arm link, task.self_contact_pairs) and between a left and a right foot capsule (None
        without foot_collision)."""
        self.wp.capture_launch(self.contact_force_graph)
        geom = self.contact_geom.long()
        live = (self._contact_slots < self.nacon) & (geom >= 0).all(dim=1)
        geom = geom.clamp(min=0)
        robot = None
        if self.self_contact_force is not None:
            masked = () if self.arm_geoms is None else (  # each slot's world (stale slots: any, they are not live)
                self.arm_geoms, self.arm_contacts_masked[self.contact_world.long().clamp(0,self.n_envs-1)])
            robot = self._largest_contacts(live & task.self_contact_pairs(geom,self.robot_geoms,*masked))
        feet = None if self.foot_contact_force is None else self._largest_contacts(
            live & (self.foot_side[geom[:,0]]*self.foot_side[geom[:,1]] < 0))
        return robot, feet

    def _obs(self, history='replace', mask=None):
        """history: 'push' adds this frame (after a step), 'fill' fills the buffer of the masked worlds (reset),
        'replace' rewrites the newest frame. Without history (1 frame) the three are the same."""
        rotation = self.xmat[:,self.pelvis].reshape(-1,3,3)
        command = (task.effective_command(self.command,rotation,self.heading_target,self.reference.stand_heading_hold,
                                          self.reference.side_heading_hold)
                   if self.heading_hold else self.command)
        self.observed_command.copy_(command)
        if not self.real_robot:
            return observation(self._velocity(),rotation,command,
                               self.qpos[:,self.qadr],self.qvel[:,self.vadr],self.home,self.previous_action,
                               self._phase())
        position, joint_velocity = self.qpos[:,self.qadr], self.qvel[:,self.vadr]
        velocity = self._velocity()
        if self.observation_noise:
            def noise(value, scale):
                return value + (2*torch.rand(value.shape,device=self.device,generator=self.generator)-1)*scale*self.realism
            velocity = torch.cat((noise(velocity[:,:3], NOISE['angular_velocity']), velocity[:,3:]), dim=1)
            position = noise(position + self.encoder_bias, NOISE['joint_position'])
            joint_velocity = noise(joint_velocity, NOISE['joint_velocity'])
            rotation = rotation.clone()
            rotation[:,2,:] = noise(rotation[:,2,:], NOISE['gravity'])  # projected gravity is -rotation[:,2,:]
        previous, upper_body = self.previous_action, None
        if self.arm_targets:  # the last action as the ONNX graph output it (clip(a) + d/scale), the slot's arm command
            previous = task.observed_action(previous,self.previous_offset,self.scale,self.previous_flag)
            upper_body = task.upper_body_command(self.arm_flag,self.arm_offset)
        observed = task.deploy_observation(velocity,rotation,command,position,joint_velocity,self.home,
                                           previous,self._phase(),phase_stop=self.reference.phase_stop)
        if self.reference.extra_terms:  # height_command [c − 0.78] (never noised, as the command), upper_body_slot zeros
            observed = torch.cat((observed,task.command_terms(self.height_command if self.crouch else None,
                                                              self.reference.upper_body_slot,observed,
                                                              upper_body)),dim=1)
        if self.history > 1:
            if history == 'fill':
                fresh = observed[:,None].expand(-1,self.history,-1)
                self.frames = fresh.clone() if mask is None else torch.where(mask[:,None,None],fresh,self.frames)
            elif history == 'push':
                self.frames = torch.cat((self.frames[:,1:],observed[:,None]),dim=1)
            else:
                self.frames[:,-1] = observed
            observed = task.history_observation(self.frames,self.previous_action.shape[1],self.reference.extra_sizes)
        if self.reference.critic_extras:  # exact, never noised: only the critic reads them
            observed = torch.cat((observed, task.critic_extras(velocity, self.qpos[:,2])), dim=1)
        return observed

    def _uniform(self, n):
        return torch.rand((n,),device=self.device,generator=self.generator)

    def _masked_reset(self, mask):
        self.reset_mask.copy_(mask)
        self.wp.capture_launch(self.reset_graph)
        positions = self.initial_qpos.expand(self.n_envs,-1).clone()
        positions[:,self.qadr] += .02*torch.rand(self.previous_action.shape,device=self.device,generator=self.generator)-.01
        commands = task.sample_commands(self.n_envs, self._uniform, **self.reference.commands)
        self.qpos.copy_(torch.where(mask[:,None],positions,self.qpos))
        self.qvel.masked_fill_(mask[:,None],0)
        self.previous_action.masked_fill_(mask[:,None],0)
        self.command.copy_(torch.where(mask[:,None],commands,self.command))
        self.heading_target.masked_fill_(mask,0)  # every episode starts facing +x
        self.yaw_drift_error.masked_fill_(mask,0)
        if self.observation_noise:
            bias = (2*torch.rand(self.encoder_bias.shape,device=self.device,generator=self.generator)-1)*ENCODER_BIAS*self.realism
            self.encoder_bias.copy_(torch.where(mask[:,None],bias,self.encoder_bias))
        self.lengths.masked_fill_(mask,0)
        if self.reference.random_phase:
            offset = torch.randint(0,self.period_steps,(self.n_envs,),device=self.device,generator=self.generator)
            self.phase_offset.copy_(torch.where(mask,offset,self.phase_offset))
        if self.delay_depth:  # no draw without action_delay, so the random stream stays the original one
            delay = torch.randint(0,self.reference.max_delay_steps+1,(self.n_envs,),device=self.device,
                                  generator=self.generator)
            self.delay_steps.copy_(torch.where(mask,delay,self.delay_steps))
            self.delay_lags.copy_(task.action_lags(self.delay_steps,self.reference.frame_skip))
            self.delay_actions.masked_fill_(mask[:,None,None],0)  # zero actions: the home pose
        if self.perturb_arms:  # a new episode starts outside a perturbation segment
            self.perturb_steps.masked_fill_(mask,0)
            self.perturb_action.masked_fill_(mask[:,None],0)
        if self.force_pushes:  # and without a push (reset_data zeroed xfrc_applied), its first one 2-4 s on
            self.push_left.masked_fill_(mask,0)
            self.push_force.masked_fill_(mask[:,None],0)
            if self.reference.force_push:
                timer = task.force_push_interval(self._uniform,self.n_envs,self.dt)
                self.push_timer.copy_(torch.where(mask,timer,self.push_timer))
        if self.motor_frictions:  # drawn last, so the draws above stay the old ones; nothing without motor_friction
            scale = (task.motor_friction_draw(self._uniform,self.n_envs,self.reference.motor_friction)
                     if self.reference.motor_friction else torch.ones_like(self.friction_scale))
            self.friction_scale.copy_(torch.where(mask,scale,self.friction_scale))
        if self.crouch:  # after every other draw, for the commands just drawn; nothing without height_command
            height = task.height_draw(self._uniform,self.command,**self.crouch_settings)
            self.height_command.copy_(torch.where(mask,height,self.height_command))
        if self.arm_targets:  # with the arm process's own generator, so the draws above stay the old ones
            self._arm_reset(mask)
        self.returns.masked_fill_(mask,0)
        self.wp.capture_launch(self.forward_graph)

    def _arm_uniform(self, n):
        return torch.rand((n,),device=self.device,generator=self.arm_generator)

    def _arm_reset(self, mask):
        """arm_targets: the masked worlds' new episodes (the CPU task's _arm_episode): the mode (task.arm_mode_draw),
        with flag 1 the hands' load shares of the ceiling, in the targets mode the first segment from home
        (task.arm_segment_draw); arm_generator draws them for every world, the masked ones take them."""
        n = self.n_envs
        mode = task.arm_mode_draw(self._arm_uniform,n)
        share = self._arm_uniform(2*n).reshape(n,2)
        home = torch.zeros_like(self.arm_start)
        end,move,hold = task.arm_segment_draw(self._arm_uniform,home,self.arm_poses,self.arm_carry,self.track_joints,
                                              self.dt)
        targets, flag = mode == task.ARM_TARGET, mode != task.ARM_FREE
        self.arm_mode.copy_(torch.where(mask,mode,self.arm_mode))
        self.arm_flag.copy_(torch.where(mask,flag,self.arm_flag))
        self.arm_fixed.masked_fill_(mask,False)
        self.arm_start.masked_fill_(mask[:,None],0)
        self.arm_end.copy_(torch.where(mask[:,None],torch.where(targets[:,None],end,home),self.arm_end))
        self.arm_clock.masked_fill_(mask,0)
        self.arm_move.copy_(torch.where(mask,torch.where(targets,move,1),self.arm_move))
        self.arm_hold.copy_(torch.where(mask,torch.where(targets,hold,0),self.arm_hold))
        self.arm_offset.masked_fill_(mask[:,None],0)  # clock 0: the segment's start, home
        self.previous_flag.masked_fill_(mask,False)
        self.previous_offset.masked_fill_(mask[:,None],0)
        self.delay_offsets.masked_fill_(mask[:,None,None],0)
        self.load_share.copy_(torch.where(mask[:,None],torch.where(flag[:,None],share,0.),self.load_share))
        self.load_fixed.masked_fill_(mask,False)
        self._set_loads(torch.where(mask[:,None],self.load_share*self.load_max,self.hand_load))

    def _set_loads(self, load):
        """Every world's hand loads (n_envs, 2) kg into its model arrays: the hand bodies' mass and COM
        (task.hand_load_mass) and every body's subtree mass (task.hand_subtree_mass); 0 the model's own numbers."""
        self.hand_load.copy_(load)
        mass,ipos = task.hand_load_mass(self.hand_mass,self.hand_ipos,load,task.HAND_LOAD_POSITION)
        self.body_mass[:,self.hand_bodies] = mass
        self.body_ipos[:,self.hand_bodies] = ipos
        self.body_subtreemass.copy_(task.hand_subtree_mass(self.subtree_mass,self.hand_above,load))

    def _rescale_loads(self):
        """The hand loads after the ceiling moved: each world's shares times load_max, but where set_hand_load fixed
        them."""
        self._set_loads(torch.where(self.load_fixed[:,None],self.hand_load,self.load_share*self.load_max))

    def set_arm_targets(self, q=None, flag=True, seconds=0.):
        """Every world's arm command from now on until its next reset (tests, probes), as the CPU task's
        set_arm_targets: flag ((n_envs,) or one) 1, the arms follow q ((n_envs, nu) or (nu,) absolute angles; their free
        arm joints, clipped to the joint ranges; None: home), in a quintic over `seconds` from the current targets (0: at
        once); 0, free arms. The generator stops in every world; the hand loads stay (set_hand_load). The next
        observation shows it."""
        if not self.arm_targets:
            raise ValueError('arm targets need task_config arm_targets')
        ref = self.reference
        flag = torch.as_tensor(flag,device=self.device).expand(self.n_envs)
        if flag.dtype != torch.bool and bool(((flag != 0) & (flag != 1)).any()):
            raise ValueError('flags must be 0 or 1')
        flag = flag.bool()
        number = isinstance(seconds,(int,float)) and not isinstance(seconds,bool) and np.isfinite(seconds)
        if not number or seconds < 0:
            raise ValueError('seconds must be a finite non-negative number')
        target = self.home.expand(self.n_envs,-1) if q is None else torch.as_tensor(
            q,dtype=torch.float32,device=self.device).expand(self.n_envs,-1)
        if target.shape != self.arm_offset.shape or not bool(torch.isfinite(target).all()):
            raise ValueError('expected finite (n_envs, actuator_count) arm targets')
        joints = self.track_joints
        low,high = (torch.as_tensor(bound,dtype=torch.float32,device=self.device)
                    for bound in ref.model.jnt_range[ref.joints[list(ref.track_joints)]].T)
        end = torch.zeros_like(self.arm_offset)
        end[:,joints] = torch.maximum(torch.minimum(target[:,joints],high),low)-self.home[joints]
        end = torch.where(flag[:,None],end,0.)
        if seconds > 0:  # from the current targets (home where the arms were free)
            self.arm_start.copy_(torch.where((flag & self.arm_flag)[:,None],self.arm_offset,0.))
        else:
            self.arm_start.copy_(end)
        self.arm_end.copy_(end)
        self.arm_clock.zero_()
        self.arm_move.copy_(task.arm_move_steps(end,end,joints,self.dt,seconds=seconds) if seconds > 0
                            else torch.ones_like(self.arm_move))
        self.arm_hold.zero_()
        self.arm_mode.copy_(torch.where(flag,torch.full_like(self.arm_mode,task.ARM_TARGET),
                                        torch.full_like(self.arm_mode,task.ARM_FREE)))
        self.arm_flag.copy_(flag)
        self.arm_fixed.fill_(True)
        self.arm_offset.copy_(torch.where(flag[:,None],task.arm_target_offset(self.arm_start,self.arm_end,self.arm_clock,
                                                                              self.arm_move),0.))
        self.perturb_steps.masked_fill_(flag,0)  # no perturbation while the arms follow targets

    def set_hand_load(self, load):
        """Every world's hand loads (kg; (n_envs, 2), (2,) or one number) until its next reset (tests, probes), as the
        CPU task's set_hand_load; the load ceiling's moves leave them alone."""
        if not self.arm_targets:
            raise ValueError('hand loads on CUDA need task_config arm_targets (the batched body masses)')
        load = torch.as_tensor(load,dtype=torch.float32,device=self.device).expand(self.n_envs,2)
        if not bool(torch.isfinite(load).all()) or bool((load < 0).any()):
            raise ValueError('expected finite non-negative hand loads')
        self.load_fixed.fill_(True)
        self._set_loads(load.clone())

    def training_state(self):
        """What a stage keeps of this training beyond the weights (save_stage): with arm_targets the hand load ceiling;
        None without (nothing kept, as before)."""
        return {'load_max': [self.load_max]} if self.arm_targets else None

    def restore_training_state(self, state):
        """The hand load ceiling from a stage's training_state (arm_targets; g1_velocity.saved_load), the worlds' loads
        scaled to it; nothing without the setting or a saved ceiling."""
        if self.arm_targets and 'load_max' in state:
            from .tasks.g1_velocity import saved_load
            self.load_max = saved_load(state)
            self._rescale_loads()

    @torch.no_grad()
    def reset(self):
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._masked_reset(torch.ones(self.n_envs,dtype=torch.bool,device=self.device))
            self._overflow_seen.zero_()
            self._bad_state_seen.zero_()
            return self._obs('fill')

    def set_action_delay_steps(self, steps):
        """Fix every world's delay k (physics substeps, at most max_delay_steps) until its next reset, as the CPU
        task's reset option action_delay_steps (tests, probes)."""
        steps = torch.as_tensor(steps,dtype=torch.long,device=self.device).expand(self.n_envs)
        if bool((steps < 0).any()) or bool((steps > self.reference.max_delay_steps).any()):
            raise ValueError('action delay steps must be within 0..max_delay_steps')
        self.delay_steps.copy_(steps)
        self.delay_lags.copy_(task.action_lags(self.delay_steps,self.reference.frame_skip))

    def set_arm_perturb(self, steps, action):
        """Start a perturbation segment in every world now (tests, probes), as the CPU task's set_arm_perturb: for
        the next steps[w] control steps (0: none) world w's arm joints follow action[w] (its arm entries, clipped to
        the action box) until the segment or the episode ends."""
        steps = torch.as_tensor(steps,dtype=torch.long,device=self.device).expand(self.n_envs)
        action = torch.as_tensor(action,dtype=torch.float32,device=self.device).expand(self.n_envs,-1)
        if bool((steps < 0).any()) or action.shape != self.perturb_action.shape:
            raise ValueError('expected non-negative steps and (n_envs, actuator_count) actions')
        limit = self.reference.action_limit
        self.perturb_steps.copy_(steps)
        self.perturb_action.copy_(action.clamp(-limit,limit))
        self.perturb_arms = True

    def set_force_push(self, velocity, seconds=.1):
        """Push every world's pelvis from the next control step on (tests, probes), as the CPU task's set_force_push:
        the horizontal force M·Δv/T for T = `seconds` in whole physics substeps, Δv = velocity[w] (forward, left; m/s,
        (n_envs, 2) or (2,)) in world w's pelvis heading frame now, fixed in the world while it lasts; a zero Δv pushes
        nothing. Until the push or the episode ends."""
        velocity = torch.as_tensor(velocity,dtype=torch.float32,device=self.device).expand(self.n_envs,2)
        number = isinstance(seconds,(int,float)) and not isinstance(seconds,bool) and np.isfinite(seconds)
        substeps = round(seconds/self.reference.model.opt.timestep) if number else 0
        if substeps < 1 or not bool(torch.isfinite(velocity).all()):
            raise ValueError('expected finite (n_envs, 2) push velocities lasting at least one physics substep')
        heading = task.heading(self.xmat[:,self.pelvis].reshape(-1,3,3))
        steps = torch.full((self.n_envs,),substeps,dtype=torch.long,device=self.device)
        self.push_left.copy_(steps)
        self.push_force.copy_(task.force_push_force(velocity,heading,self.reference.mass,steps,
                                                    self.reference.model.opt.timestep))
        self.force_pushes = True

    def set_motor_friction(self, scale):
        """Scale every world's motor joint friction (static and viscous) by scale[w] ((n_envs,) or one number) until
        its next reset (tests, probes), as the CPU task's set_motor_friction; real robot only."""
        if self.motors is None:
            raise ValueError('motor friction needs the real-robot motor model')
        scale = torch.as_tensor(scale,dtype=torch.float32,device=self.device).expand(self.n_envs)
        if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
            raise ValueError('expected finite non-negative motor friction scales')
        self.friction_scale.copy_(scale)
        self.motor_frictions = True

    def set_height(self, height):
        """Fix every world's pelvis height command c (m, (n_envs,) or one number, within config.CROUCH_HEIGHTS) until
        its next reset or, with resample_commands, its next command change (tests, probes), as the CPU task's
        set_height; height_command only. The next observation shows it."""
        if not self.crouch:
            raise ValueError('a height command needs task_config height_command')
        height = torch.as_tensor(height,dtype=torch.float32,device=self.device).expand(self.n_envs)
        low, high = CROUCH_HEIGHTS
        if not bool(torch.isfinite(height).all()) or bool((height < low).any()) or bool((height > high).any()):
            raise ValueError(f'expected heights in [{low:g}, {high:g}] m')
        self.height_command.copy_(height)

    @torch.no_grad()
    def step(self, actions):
        if actions.device != self.device or actions.shape != self.previous_action.shape:
            raise ValueError('Expected CUDA actions matching (n_envs, actuator_count)')
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._bad_state_seen.logical_or_(~torch.isfinite(actions).all())
            limit = self.reference.action_limit
            action = actions.clamp(-limit,limit)
            if self.perturb_arms:
                # arm_perturb: segments start, run and end per world; in one the arm joints follow its held action,
                # and this applied action replaces the policy's in everything below (PD, delay queue, reward,
                # observed previous action). PPO keeps the policy's own for its log-probability.
                if self.reference.arm_perturb:
                    steps,held = task.arm_perturb_draw(self.perturb_steps,self.perturb_action,self._uniform,
                                                       self.reference.arm_perturb,self.dt,self.scale,limit,
                                                       self.reference.arm_perturb_radians)
                    self.perturb_steps.copy_(steps)
                    self.perturb_action.copy_(held)
                if self.arm_targets:  # no segment in a world whose arms follow targets (drawn all the same)
                    self.perturb_steps.masked_fill_(self.arm_flag,0)
                action = task.arm_perturb_apply(action,self.perturb_steps,self.perturb_action,self.arm_mask)
                perturbed = self.perturb_steps > 0
                if self.reference.arm_perturb_contact_mask:  # this step's arm contacts leave self_contact
                    self.arm_contacts_masked.copy_(perturbed)
                self._arm_perturb_sum.add_(perturbed.sum())
                self._arm_perturb_count += self.n_envs
                self.perturb_steps.sub_(perturbed.long())
            if self.lock_wrists:  # wrist pitch and yaw of the applied action: 0 (home), also in a segment
                action = task.lock_joints(action,self.wrist_mask)
            if self.push_velocity > 0 and self.randomize:
                # Free-joint qvel: linear velocity in world x/y/z, then angular velocity in the pelvis frame.
                push = self._uniform(self.n_envs) < self.dt / REAL_PUSH_INTERVAL
                kick = (2*torch.rand((self.n_envs,6),device=self.device,generator=self.generator)-1)
                self.qvel[:,:6] += kick*torch.as_tensor(REAL_PUSH,device=self.device)*self.realism*push[:,None]
            elif self.push_velocity > 0:
                # Free-joint qvel[0:2] is the pelvis linear velocity in world x/y.
                push = self._uniform(self.n_envs) < self.dt / task.PUSH_INTERVAL_SECONDS
                kick = (2*torch.rand((self.n_envs,2),device=self.device,generator=self.generator)-1)*self.push_velocity
                self.qvel[:,:2] += kick*push[:,None]
            if self.force_pushes:
                # force_push: pushes start per world (task.force_push_draw); a running one's force goes into the
                # pelvis xfrc_applied before each substep's graph launch, 0 where none runs.
                if self.reference.force_push:
                    heading = task.heading(self.xmat[:,self.pelvis].reshape(-1,3,3))
                    timer,left,force = task.force_push_draw(self.push_timer,self.push_left,self.push_force,heading,
                                                            self._uniform,self.reference.force_push,
                                                            self.reference.mass,self.dt,
                                                            self.reference.model.opt.timestep)
                    self.push_timer.copy_(timer)
                    self.push_left.copy_(left)
                    self.push_force.copy_(force)
                pushes,left = task.force_push_substeps(self.push_force,self.push_left,self.reference.frame_skip)
                self.push_left.copy_(left)
            feet_before = self.xpos[:,self.feet,:2].clone()
            # arm_targets: home moved by each world's arm targets of this step (flag 1; 0 elsewhere), and the delay
            # queue's targets each with the offsets of its own step; without the setting home itself.
            home = self.home
            if self.arm_targets:
                offset = torch.where(self.arm_flag[:,None],self.arm_offset,0.)
                home = self.home+offset
                if self.delay_depth:
                    homes = self.home+torch.cat((offset[:,None],self.delay_offsets),dim=1)
            if self.delay_depth:  # this step's action, then the older ones the motors may still follow
                queue = torch.cat((action[:,None],self.delay_actions),dim=1)
            for substep in range(self.reference.frame_skip):
                applied = queue[self._worlds,self.delay_lags[:,substep]] if self.delay_depth else action
                held = homes[self._worlds,self.delay_lags[:,substep]] if self.arm_targets and self.delay_depth else home
                if self.motors is None:
                    torque = pd_torque(applied,self.qpos[:,self.qadr],self.qvel[:,self.vadr],held,self.scale,
                                       self.kp,self.kd,self.joint_low,self.joint_high,self.torque_low,self.torque_high,
                                       limit)
                else:  # same order as the CPU task: PD, motor curve and friction, then the actuator range
                    joint_velocity = self.qvel[:,self.vadr]
                    torque = pd_torque(applied,self.qpos[:,self.qadr],joint_velocity,held,self.scale,self.kp,
                                       self.kd,self.joint_low,self.joint_high,-self.unclamped,self.unclamped,limit)
                    friction = dict(friction_scale=self.friction_scale) if self.motor_frictions else {}
                    torque = task.motor_torque(torque,joint_velocity,*self.motors,**friction).clamp(min=self.torque_low,
                                                                                                    max=self.torque_high)
                self.ctrl.copy_(torque)
                if self.force_pushes:
                    self.xfrc_applied[:,self.push_body,:3] = pushes[:,substep]
                self.wp.capture_launch(self.step_graph)
                self._overflow_seen.bitwise_or_(self.overflow)
                if self.self_contact_force is not None or self.foot_contact_force is not None:
                    robot, feet = self._contact_forces()  # this substep's contacts and constraint forces
                    if robot is not None:
                        self.self_contact_force[:,substep] = robot
                    if feet is not None:
                        self.foot_contact_force[:,substep] = feet
            self.wp.capture_launch(self.forward_graph)
            self.lengths.add_(1)
            velocity = self._velocity()
            yaw_drift = None
            if self.reference.reward_weights['yaw_drift']:  # off: no state, so the reward is the one without the term
                self.yaw_drift_error.copy_(task.yaw_drift_update(self.yaw_drift_error,velocity,self.observed_command,
                                                                 self.dt))
                yaw_drift = self.yaw_drift_error.clone()
                straight = task.straight_walking(self.command)
                self._yaw_drift_sum.add_((yaw_drift.abs()*straight).sum())
                self._yaw_drift_count.add_(straight.sum())
            self_contact = None
            if self.self_contact_force is not None:
                self_contact = task.self_contact_share(self.self_contact_force)
                self._self_contact_sum.add_(self_contact.sum())
                self._self_contact_count += self.n_envs
            if self.foot_contact_force is not None:  # world-steps with a foot-foot contact above 10 N
                self._foot_contact_sum.add_(task.contact_step(self.foot_contact_force).sum())
                self._foot_contact_count += self.n_envs
            arm_press = None
            if self.reference.reward_weights['arm_press']:  # the policy's own target (not a delayed one), as on CPU
                target = task.pd_target(action,home,self.scale,self.joint_low,self.joint_high,limit)
                arm_press = task.arm_press(target,self.qpos[:,self.qadr],self.arm_joints)
                self._arm_press_sum.add_(arm_press.sum())
                self._arm_press_count += self.n_envs
            height_command = stand_home = None
            if self.crouch:  # the height command observed with the action (it changes only below, or by set_height)
                height_command = self.height_command.clone()
                stand_home = task.stand_target(self.home,height_command,self.crouch_joints)
                crouched = height_command < task.BASE_HEIGHT
                self._crouch_sum.add_(crouched.sum())
                self._height_excess_sum.add_(((self.qpos[:,2]-height_command).clamp(min=0)*crouched).sum())
                self._crouch_count += self.n_envs
            # arm_targets: each world's flag and its arms' tracking of home + d after the step (task.arm_track; the
            # reward then leaves arm_press and the arms' posture out where the flag is set), summed for rollout_metrics.
            # Whether each world's targets hold (task.arm_holding) and its arms' error there, logged, and while the
            # arm_error weight is non-zero charged by that term.
            arms = {}
            if self.arm_targets:
                position = self.qpos[:,self.qadr]
                tracked = task.arm_track(position,home,self.track_joints)
                error = task.arm_error(position,home,self.track_joints)
                flag = self.arm_flag.clone()
                hold = task.arm_holding(self.arm_start,self.arm_end,self.arm_clock,self.arm_move)
                self._arm_steps.add_(flag.sum())
                self._free_steps.add_((~flag).sum())
                self._arm_track_sum.add_((tracked*flag).sum())
                self._arm_error_sum.add_((error*flag).sum())
                self._hold_steps.add_((flag & hold).sum())
                self._hold_error_sum.add_((error*(flag & hold)).sum())
                arms = dict(arm_flag=flag,arm_track=tracked,arm_posture_joints=self.arm_posture_joints)
                if self.reference.reward_weights['arm_error']:
                    arms.update(arm_error=error,arm_hold=hold)
            # Rewarded for the command observed when the action was chosen (set by the previous _obs()).
            components,terminated = reward_components(velocity,self.xmat[:,self.pelvis].reshape(-1,3,3),
                self.qpos[:,2],self.observed_command.clone(),self.ctrl,action,self.previous_action,
                self.xpos[:,self.feet,2],self._phase(),self.qpos[:,self.qadr],self.home,
                self.xmat[:,self.torso].reshape(-1,3,3),weights=self.reference.reward_weights,
                posture_joints=self.posture_joints,heading_target=self.heading_target,
                stance_fraction=self.reference.gait['stance_fraction'],hip_joints=self.hip_joints,
                foot_speed=(self.xpos[:,self.feet,:2]-feet_before).norm(dim=-1)/self.dt,
                foot_rotation=self.xmat[:,self.feet].reshape(self.n_envs,len(self.feet),3,3),
                sole_height=(task.sole_height(self.geom_xpos[:,self.soles,2],self.sole_radius)
                             if self.reference.sole_contact or self.reference.reward_weights['clearance']
                             else None),sole_contact=self.reference.sole_contact,
                yaw_drift_error=yaw_drift,requested_command=self.command.clone(),self_contact=self_contact,
                arm_press=arm_press,feet_flat_stance=self.reference.feet_flat_stance,height_command=height_command,
                stand_home=stand_home,**arms)
            reward = sum(components.values())
            self.previous_action.copy_(action)
            if self.arm_targets:  # the offsets this step followed: the observed last action, the delay queue
                self.previous_flag.copy_(self.arm_flag)
                self.previous_offset.copy_(offset)
            if self.delay_depth:
                self.delay_actions.copy_(queue[:,:-1])
                if self.arm_targets:
                    self.delay_offsets.copy_(torch.cat((offset[:,None],self.delay_offsets),dim=1)[:,:-1])
            if self.resample_commands:
                # Same per-step resampling probability as the CPU task; applies from the next action.
                # side_stop reads the command each world is replacing (a reset draws without one).
                resample = self._uniform(self.n_envs) < self.dt / task.COMMAND_RESAMPLE_SECONDS
                fresh = task.sample_commands(self.n_envs, self._uniform, previous=self.command,
                                             **self.reference.commands)
                self.command.copy_(torch.where(resample[:,None],fresh,self.command))
                self.yaw_drift_error.masked_fill_(resample,0)  # a new command starts a new mean
                if self.reference.commands.get('straight_probability'):  # walk straight on from here
                    here = task.heading(self.xmat[:,self.pelvis].reshape(-1,3,3)).to(self.heading_target.dtype)
                    straight = resample & task.straight_walking(fresh)
                    self.heading_target.copy_(torch.where(straight,here,self.heading_target))
                if self.crouch:  # a new height for each new command, after the command's draws
                    height = task.height_draw(self._uniform,fresh,**self.crouch_settings)
                    self.height_command.copy_(torch.where(resample,height,self.height_command))
            if self.arm_targets:  # the targets move on (arm_generator), shown in the next observation
                live = (self.arm_mode == task.ARM_TARGET) & ~self.arm_fixed
                state = task.arm_target_advance((self.arm_start,self.arm_end,self.arm_clock,self.arm_move,self.arm_hold),
                                                live,self._arm_uniform,self.arm_poses,self.arm_carry,self.track_joints,
                                                self.dt)
                for current,new in zip((self.arm_start,self.arm_end,self.arm_clock,self.arm_move,self.arm_hold),state):
                    current.copy_(new)
                self.arm_offset.copy_(torch.where(self.arm_flag[:,None],task.arm_target_offset(*state[:4]),0.))
            terminal_obs = self._obs('push')
            self.returns.add_(reward)
            truncated = self.lengths >= self.max_episode_steps
            done = terminated | truncated
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._overflow_seen.bitwise_or_(self.overflow)
            if self.arm_targets:  # falls of flag-1 episodes, for the load curve, and of flag-0 ones, logged
                self._arm_falls.add_((terminated & self.arm_flag).sum())
                self._free_falls.add_((terminated & ~self.arm_flag).sum())
            self._masked_reset(done)
            # Worlds that reset start a fresh history filled with their first frame; the others keep theirs.
            return torch.where(done[:,None],self._obs('fill',done),terminal_obs),reward,terminated,truncated,terminal_obs

    def rollout_metrics(self):
        """Task means since the last call, for the training log (TensorPPO.collect reads them once per rollout):
        rollout/yaw_drift_abs, mean |e| of the yaw_drift term over steps on straight commands, while it is on;
        rollout/self_contact, mean share of physics substeps with a self-contact above the threshold, while that
        term is on (as the reward counts it: with arm_perturb_contact_mask, perturbed world-steps without their arm
        contacts); rollout/foot_contact, share of world-steps with a contact between the two feet's capsules above
        the threshold in any substep (task.contact_step), while foot_collision is on (those contacts are in
        rollout/self_contact too); rollout/arm_press, mean Σ over the arm joints of (PD target − angle)² per
        world-step (rad², unweighted), while that term is on; rollout/arm_perturb, share of world-steps whose arms
        followed a perturbation segment, while arm_perturb is on; rollout/motor_friction, the worlds' mean motor
        friction multiplier now, while motor_friction is on; rollout/crouch_share, share of world-steps under a crouch
        command (c < 0.78), and rollout/height_excess, mean max(0, h − c) (m) over those world-steps, while
        height_command is on. With arm_targets the window also moves the hand load ceiling (task.load_curve_update, the
        worlds' loads scaled to it): rollout/arm_track, the mean unweighted arm_track over the flag-1 world-steps;
        rollout/arm_error, their arms' mean |q − target| (rad), and rollout/arm_error_hold, the same over the world-steps
        whose targets hold (task.arm_holding, where the arm_error term charges it); rollout/arm_falls, flag-1 falls per episode of
        max_episode_steps (falls / (flag-1 world-steps / episode length)); rollout/free_falls, the same of flag 0 (logged
        only); rollout/load_max, the ceiling after the move (kg). With task_config load_curve the ceiling moves only
        once its block of windows is full (task.LoadCurveBlock), and that window logs rollout/arm_falls_block, the
        block's flag-1 falls per episode."""
        metrics = {}
        if self.arm_targets:
            steps, falls, track_sum = self._arm_steps.item(), self._arm_falls.item(), self._arm_track_sum.item()
            track = track_sum / steps if steps else float('nan')
            metrics['rollout/arm_track'] = track
            metrics['rollout/arm_error'] = self._arm_error_sum.item() / steps if steps else float('nan')
            hold_steps = self._hold_steps.item()
            metrics['rollout/arm_error_hold'] = self._hold_error_sum.item() / hold_steps if hold_steps else float('nan')
            metrics['rollout/arm_falls'] = falls * self.max_episode_steps / steps if steps else float('nan')
            free_steps = self._free_steps.item()
            metrics['rollout/free_falls'] = (self._free_falls.item() * self.max_episode_steps / free_steps
                                             if free_steps else float('nan'))
            level = self.load_max
            if self.load_block is None:  # the per-window rule
                level = task.load_curve_update(self.load_max, steps, falls, track, self.max_episode_steps)
            elif self.load_block.add(steps, falls, track_sum):  # load_curve: once a block is full
                level, metrics['rollout/arm_falls_block'] = self.load_block.decide(self.load_max,
                                                                                    self.max_episode_steps)
            if level != self.load_max:
                self.load_max = level
                self._rescale_loads()
            metrics['rollout/load_max'] = self.load_max
            for value in (self._arm_steps, self._arm_track_sum, self._arm_error_sum, self._hold_steps,
                          self._hold_error_sum, self._arm_falls, self._free_steps, self._free_falls):
                value.zero_()
        if self.reference.reward_weights['yaw_drift']:
            count = self._yaw_drift_count.item()
            metrics['rollout/yaw_drift_abs'] = self._yaw_drift_sum.item() / count if count else float('nan')
            self._yaw_drift_sum.zero_()
            self._yaw_drift_count.zero_()
        if self.self_contact_force is not None:
            count = self._self_contact_count
            metrics['rollout/self_contact'] = self._self_contact_sum.item() / count if count else float('nan')
            self._self_contact_sum.zero_()
            self._self_contact_count = 0
        if self.foot_contact_force is not None:
            count = self._foot_contact_count
            metrics['rollout/foot_contact'] = self._foot_contact_sum.item() / count if count else float('nan')
            self._foot_contact_sum.zero_()
            self._foot_contact_count = 0
        if self.reference.reward_weights['arm_press']:
            count = self._arm_press_count
            metrics['rollout/arm_press'] = self._arm_press_sum.item() / count if count else float('nan')
            self._arm_press_sum.zero_()
            self._arm_press_count = 0
        if self.reference.arm_perturb:
            count = self._arm_perturb_count
            metrics['rollout/arm_perturb'] = self._arm_perturb_sum.item() / count if count else float('nan')
            self._arm_perturb_sum.zero_()
            self._arm_perturb_count = 0
        if self.reference.motor_friction:
            metrics['rollout/motor_friction'] = self.friction_scale.mean().item()
        if self.crouch:
            crouched = self._crouch_sum.item()
            metrics['rollout/crouch_share'] = crouched / self._crouch_count if self._crouch_count else float('nan')
            metrics['rollout/height_excess'] = self._height_excess_sum.item() / crouched if crouched else float('nan')
            self._crouch_sum.zero_()
            self._height_excess_sum.zero_()
            self._crouch_count = 0
        return metrics

    def check_health(self):
        overflow = self.mjw.OverflowType
        # Solver iteration limits and the EPA horizon (convex-mesh contact precision; first seen when G1 learned
        # from scratch and fell a lot) do not lose state; buffer capacity overflows do.
        capacity_mask = int(overflow.ALL) & ~int(overflow.ITERATIONS | overflow.LS_ITERATIONS | overflow.EPA_HORIZON)
        if bool((self._overflow_seen & capacity_mask).any().item()):
            raise RuntimeError('G1 Warp contact/constraint allocation overflow')
        if bool(self._bad_state_seen.item()):
            raise RuntimeError('Nonfinite G1 simulation state, action or reward')
        self._overflow_seen.zero_()
        self._bad_state_seen.zero_()

    def close(self):
        self.reference.close()
