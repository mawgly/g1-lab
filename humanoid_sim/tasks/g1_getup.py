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
up and both soles are down (g1_tensor_math.stance_width / stance_active). Equations live in g1_tensor_math, shared with
the CUDA task.
"""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch

from . import g1_getup_bank
from . import g1_tensor_math as task
from .config import G1_GETUP_DEFAULTS, G1_GETUP_PROFILE, GETUP_CRITIC_EXTRAS, g1_getup_config
from .g1_getup_bank import instabilities
from .g1_velocity import G1VelocityEnv

EPISODE_STEPS = round(task.GETUP_EPISODE_SECONDS / .02)  # 500 at the walking control step


class G1GetupEnv(G1VelocityEnv):
    """g1_getup on the CPU: one world of the CUDA task's batch, in float64.

    Reset options (probes, evaluation): start (a bank row; default a random training row), assist_force (N, this episode
    and on; default the world's own, after assist_update), action_delay_steps, motor_friction_scale, body_friction
    (default drawn as in training), pd_bound (the PD error bound c, this episode and on: a share of the peak torque,
    also without the setting, as a probe; None: no bound; 'min': the setting's lowest, no bound without it; default
    the world's own, after pd_bound_update)."""

    keeps_training_state = True  # stages save its assist force and bound (training_state), resumed runs restore them
    asymmetric_critic = True  # the actor sees the first actor_inputs only (policies.CriticExtrasPolicy), on CPU too

    def __init__(self, max_episode_steps=EPISODE_STEPS, render_mode=None, reward_weights=None,
                 assist_force=G1_GETUP_DEFAULTS['assist_force'], action_delay=G1_GETUP_DEFAULTS['action_delay'],
                 motor_friction=G1_GETUP_DEFAULTS['motor_friction'], body_friction=G1_GETUP_DEFAULTS['body_friction'],
                 pd_bound=G1_GETUP_DEFAULTS['pd_bound'], bank=None,
                 stand_pose_std=G1_GETUP_DEFAULTS['stand_pose_std'],
                 action_change_unbounded=G1_GETUP_DEFAULTS['action_change_unbounded']):
        settings = g1_getup_config(dict(reward_weights=reward_weights or {}, assist_force=assist_force,
                                        action_delay=action_delay, motor_friction=motor_friction,
                                        body_friction=list(body_friction), pd_bound=pd_bound,
                                        stand_pose_std=stand_pose_std,
                                        action_change_unbounded=action_change_unbounded))
        super().__init__(max_episode_steps, render_mode, action_delay=settings['action_delay'],
                         motor_friction=settings['motor_friction'], **G1_GETUP_PROFILE)
        self.getup_weights = settings['reward_weights']
        self.stand_pose_std = settings['stand_pose_std']  # σ of the standing pose term (rad)
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
        self.getup_at, self.getup_held, self.hold_steps = None, None, 0
        self.head = 0.

    @property
    def bank(self):
        """The start bank (g1_getup_bank.load), read at the first reset; training draws its rows without holdout."""
        if not isinstance(self._bank, dict):
            self._bank = g1_getup_bank.load(g1_getup_bank.BANK if self._bank is None else self._bank)
        if self.training_rows is None:
            self.training_rows = np.flatnonzero(~self._bank['holdout'])
        return self._bank

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
        self.assist = float(task.assist_resample(forces, count)[index])
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
        return np.concatenate((actor, extras[0].astype(np.float32)))

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        bank = self.bank
        # The episode that just ended steps this world's assist down (assist_update); a set force replaces it.
        head_mean = np.array([self._head_mean()])
        if 'assist_force' in options:
            force = options['assist_force']
            if isinstance(force, bool) or not isinstance(force, (int, float, np.floating)) or not np.isfinite(force) \
                    or force < 0:
                raise ValueError('assist_force must be a finite non-negative number')
            self.assist = float(force)
        else:
            self.assist = float(task.assist_update(np.array([self.assist]), head_mean)[0])
        # pd_bound: the same for this world's bound (pd_bound_update, no draw); a set one replaces it.
        if 'pd_bound' in options:
            self.set_pd_bound(options['pd_bound'])
        elif self.pd_bound is not None and self.bound is not None:
            self.bound = float(task.pd_bound_update(np.array([self.bound]), head_mean, self.pd_bound[1])[0])
        if 'start' in options:
            row = options['start']
            if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(bank['qpos']):
                raise ValueError('start must be a row of the start bank')
        else:
            row = int(self.training_rows[self.np_random.integers(len(self.training_rows))])
        self.start_row, self.start_class = int(row), bank['classes'][int(bank['start_class'][row])]
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
        self.data.qpos[:] = bank['qpos'][row]
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        self.data.qvel[:] = bank['qvel'][row]
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
        return self._observation('fill'), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Invalid action shape or non-finite action')
        # action_change_unbounded: the action_change term's action, before pd_bound (box clip, lock_wrists).
        unbounded = self.unbounded_action(action) if self.change_unbounded else None
        # The applied action (box clip, pd_bound, lock_wrists) replaces the policy's in all below, as in walking.
        action = self.applied_action(action)
        target = self._target(action)
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
        # action_change: the applied action and the one before, under action_change_unbounded the action before
        # pd_bound and its own previous one.
        changed, before = (action, self.previous_action) if unbounded is None else (unbounded, self.previous_unbounded)
        components, standing = task.getup_reward_components(
            head, height, rotation, velocity[None], position, self.home, self.soft_low, self.soft_high,
            range(task.LEGS), sole, self.data.ctrl[None], self.data.qvel[self.vadr][None], changed[None],
            before[None], weights=self.getup_weights, impact=impact, stand_pose_std=self.stand_pose_std,
            sole_impact=sole_impact, stance_width=width)
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
            info.update(stance_width=float(width[0]), stance_active=bool(task.stance_active(height, sole)[0]))
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
