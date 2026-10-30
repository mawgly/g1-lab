"""Ladder step 1: G1 standing and walking in one world (g1_velocity, the real-robot profile; walking candidate 8 =
exp-28691193718e4d0e @ 7221510144, its commands stand, walk, turns, side steps, the training's pushes and
randomisation), collected on CUDA (TensorG1Env): the feet's contacts, a cyclic gait.

State: world_model.Layout (the G1's free root); the applied action (box clip, arm_perturb, lock_wrists) is the next
state's previous action. ξ: the foot friction, the motor friction multiplier, the delay (substeps / frame_skip) and the
pelvis COM offset (m). Per-step inputs: the root qvel kick the CUDA task's REAL_PUSH added before the physics (6, qvel
convention; the CPU task has none) and the force push's Δv of the step (Σ F·timestep / M, world, m/s), both recorded by
the tasks' no-op _before_physics hook. The reward is task.reward_components with the CUDA step's arguments made from
states; effort (the motor torques) and self_contact (contact forces) are not a state's, so they are excluded and their
share printed; heading hold, the yaw-drift EMA and the gait clock are teacher-forced from the logged fields."""
import numpy as np

from . import Adapter, Field, Terms, Thresholds
from .contacts import CONTACT_GROUPS, GROUP_PARTS, LOCKED_WRISTS
from .getup import FAMILIES

TIMESTEP, FRAME_SKIP = .002, 10  # the G1 task's physics substep (s) and substeps per control step
BODIES = ('pelvis', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link',
          'right_wrist_yaw_link')


class WalkTerms(Terms):
    """g1_velocity's reward and fall of canonical states by task.reward_components, the constants from the CPU task
    `reference` (G1VelocityEnv of the data's task_config)."""

    def __init__(self, reference, device='cpu', dtype=None):
        import torch
        from .. import world_model as wm
        dtype = torch.float32 if dtype is None else dtype

        def constant(value, kind=dtype):
            return torch.as_tensor(np.asarray(value), dtype=kind, device=device)
        model = reference.model
        self.layout = wm.Layout(model.nu)
        self.kinematics = wm.Kinematics(model, device, dtype)
        for name in ('home', 'scale', 'target_low', 'target_high'):
            setattr(self, name, constant(getattr(reference, name)))
        self.limit = float(reference.action_limit)
        self.weights = dict(reference.reward_weights)
        self.arm_joints = constant(reference.arm_joints, torch.long)
        self.posture_joints = constant(reference.posture_joints, torch.long)
        self.hip_joints = constant(reference.hip_joints, torch.long)
        self.stance_fraction, self.period = reference.gait['stance_fraction'], reference.gait['period']
        self.dt = float(reference.dt)
        self.sole_contact, self.feet_flat_stance = bool(reference.sole_contact), bool(reference.feet_flat_stance)
        self.use_soles = self.sole_contact or bool(self.weights['clearance'])  # the env's rule for sole_height
        self.feet, self.torso = [int(body) for body in reference.feet], int(reference.torso)
        soles = np.asarray(reference.soles)
        self.sole_shape = soles.shape
        self.sole_bodies = model.geom_bodyid[soles.reshape(-1)]
        self.sole_local = constant(model.geom_pos[soles.reshape(-1)])
        self.sole_radius = constant(reference.sole_radius)
        self.ipos = constant(model.body_ipos[reference.pelvis])  # the pelvis COM in its frame, nominal
        self.body_ids = constant([model.body(name).id for name in BODIES], torch.long)

    def velocity(self, state, com):
        """(m, 6) the reward's pelvis velocity (task.body_velocity, mj_objectVelocity BODY local): angular and linear at
        the pelvis COM (nominal + com (m, 3), the context's offset) in its inertial frame."""
        from ..world_model import _rotate, _unrotate
        import torch
        layout = self.layout
        rotation = layout.rotation(state)
        angular = _rotate(rotation, layout.part(state, 'angular'))
        linear = _rotate(rotation, layout.part(state, 'linear')) + torch.linalg.cross(
            angular, _rotate(rotation, self.ipos.to(state.dtype) + com), dim=-1)
        inertial = rotation @ self.kinematics.root_inertial.to(state.dtype)
        return torch.cat((_unrotate(inertial, angular), _unrotate(inertial, linear)), -1)

    def goal(self, batch):
        def goal(j, state):
            return self.goal_rows(state.detach(), None)
        return goal

    def goal_rows(self, following, rows):
        """(m, 2) the fall's margins: pelvis height − FALL_HEIGHT, R[2, 2] − FALL_UPRIGHT."""
        import torch
        from ..tasks import g1_tensor_math as task
        rotation = self.layout.rotation(following)
        return torch.stack((self.layout.part(following, 'position')[:, 2] - task.FALL_HEIGHT,
                            rotation[:, 2, 2] - task.FALL_UPRIGHT), -1)

    def reward(self, following, state, flat):
        import torch
        from ..tasks import g1_tensor_math as task
        layout = self.layout
        m = following.shape[0]
        velocity = self.velocity(following, flat['context'][:, 3:6].to(following.dtype))
        rotation = layout.rotation(following)
        position, turned = self.kinematics.poses(following, layout, [*self.feet, self.torso])
        before, _ = self.kinematics.poses(state, layout, self.feet)
        joints = layout.part(following, 'joints')
        action, previous = layout.part(following, 'previous'), layout.part(state, 'previous')
        sole_height = None
        if self.use_soles:
            centres = self.kinematics.points(following, layout, self.sole_bodies, self.sole_local)[..., 2]
            sole_height = task.sole_height(centres.reshape(m, *self.sole_shape), self.sole_radius)
        command = flat['observed_command'].to(following.dtype)
        yaw_drift = None
        if self.weights['yaw_drift']:  # the env's EMA, teacher-forced from the logged value before the step
            yaw_drift = task.yaw_drift_update(flat['yaw_drift_before'].to(following.dtype), velocity, command, self.dt)
        arm_press = None
        if self.weights['arm_press']:
            target = task.pd_target(action, self.home, self.scale, self.target_low, self.target_high, self.limit)
            arm_press = task.arm_press(target, joints, self.arm_joints)
        components, _ = task.reward_components(
            velocity, rotation, layout.part(following, 'position')[:, 2], command, torch.zeros_like(action), action,
            previous, position[:, :2, 2], task.gait_phase(flat['phase_steps'], self.dt, self.period), joints,
            self.home, turned[:, 2], weights=self.weights, posture_joints=self.posture_joints,
            heading_target=flat['heading_target'].to(following.dtype), stance_fraction=self.stance_fraction,
            hip_joints=self.hip_joints, foot_speed=(position[:, :2, :2] - before[..., :2]).norm(dim=-1) / self.dt,
            foot_rotation=turned[:, :2], sole_height=sole_height, sole_contact=self.sole_contact,
            yaw_drift_error=yaw_drift, requested_command=flat['command'].to(following.dtype), self_contact=None,
            arm_press=arm_press, feet_flat_stance=self.feet_flat_stance)
        return sum(components.values()), components

    def terminated(self, following, flat):
        """(m,) the fall of reward_components without height_command."""
        from ..tasks import g1_tensor_math as task
        rotation = self.layout.rotation(following)
        return (self.layout.part(following, 'position')[:, 2] < task.FALL_HEIGHT) | (rotation[:, 2, 2]
                                                                                        < task.FALL_UPRIGHT)

    def phase_masks(self, batch):
        """By what the gait does under the requested command, exclusive: stand (no motion asked), side (a purely
        sideways step), walk (‖v_xy‖ above STAND_SPEED, |ω_z| not), walk_turn (both above it: walking along an arc,
        most of candidate 8's turns, whose sampler draws vx for every walking command), turn (a turn in place: ω_z
        alone above it)."""
        from ..tasks import g1_tensor_math as task
        command = batch['command']
        n, k = command.shape[:2]
        flat = command.reshape(n * k, 3)
        moving = task.command_moving(flat)
        side = moving & task.pure_sideways(flat)
        rest = moving & ~side
        speed = flat[:, :2].norm(dim=1) > task.STAND_SPEED
        turning = flat[:, 2].abs() > task.STAND_SPEED
        return {key: value.reshape(n, k) for key, value in (
            ('stand', ~moving), ('walk', rest & speed & ~turning), ('walk_turn', rest & speed & turning),
            ('turn', rest & ~speed), ('side', side))}

    def window_sets(self, batch):
        """pushed_windows: a kick or a push on a scored step of the window (alive: the window's first episode; a push's
        timing within its step is lost)."""
        return dict(pushed_windows=((batch['inputs'] != 0).any(-1) & batch['alive']).any(-1))

    def bodies(self, states):
        position, rotation, linear, angular = self.kinematics(states, self.layout)
        index = self.body_ids
        return position[:, index], rotation[:, index], linear[:, index], angular[:, index]


class G1Walk(Adapter):
    """g1_velocity, real robot (module docstring)."""

    name, robot_id, task_id = 'g1_walk', 'g1_29dof', 'g1_velocity'
    ladder = Thresholds(step=1, phase='all', error=.02, skill=.8, reward_r=.9, reward_set='contact_steps', auroc=.9,
                        positives=20, gating=('error', 'skill', 'reward', 'auroc'))
    backends = frozenset({'cpu', 'cuda'})
    nu, kick, push = 29, True, True
    context = ('foot_friction', 'motor_friction', 'delay', 'com_x', 'com_y', 'com_z')
    inputs = ('kick_vx', 'kick_vy', 'kick_vz', 'kick_wx', 'kick_wy', 'kick_wz', 'push_x', 'push_y', 'push_z')
    bias = 29
    fields = (Field('observed_command', 'f4', (3,), 'before'), Field('command', 'f4', (3,), 'before'),
              Field('heading_target', 'f4', (), 'before'), Field('yaw_drift_before', 'f4', (), 'before'),
              Field('phase_steps', 'i4'), Field('steps', 'i4'))
    contact_groups, group_parts, families = CONTACT_GROUPS, GROUP_PARTS, FAMILIES
    phases = (('stand', 'стоит'), ('walk', 'идёт прямо'), ('walk_turn', 'идёт с поворотом'),
              ('turn', 'поворот на месте'), ('side', 'вбок'))
    agreed = ('action_change',)
    excluded = ('effort', 'self_contact')
    goal = 2

    def __init__(self, task_config=None):
        from .. import world_model as wm
        config = task_config or {}
        if task_config is not None:
            for key in ('arm_targets', 'height_command'):
                if config.get(key):
                    raise ValueError(f'модель мира: ходьба с {key} пока не поддержана')
            if not config.get('real_robot'):
                raise ValueError('модель мира: ходьба только в профиле реального робота (task_config real_robot)')
            if round((config.get('action_delay') or 0.) / TIMESTEP) > FRAME_SKIP:
                raise ValueError('модель мира: задержка больше шага управления: в состоянии только одно предыдущее '
                                 'действие')
        super().__init__(task_config)
        if config.get('lock_wrists'):
            self.held = LOCKED_WRISTS
        self.defaults = dict(cuda=dict(hidden=wm.HIDDEN, batch='auto', minutes=60.),
                             cpu=dict(hidden=(512, 512, 512), batch=512, minutes=10.))

    def check(self, device):
        super().check(device)
        config = self.task_config or {}
        if device == 'cuda' and not ((config.get('reward_weights') or {}).get('self_contact')
                                     or config.get('foot_collision')):
            raise ValueError('модель мира: сбору ходьбы на CUDA нужен разбор касаний (вес self_contact или '
                             'foot_collision)')

    def bind(self, reference, device, dtype=None):
        if reference.max_delay_steps > reference.frame_skip:
            raise ValueError('модель мира: задержка больше шага управления: в состоянии только одно предыдущее '
                             'действие')
        return WalkTerms(reference, device, dtype)

    @property
    def cuda_base(self):
        from ..g1_tensor_env import TensorG1Env
        return TensorG1Env

    # --- CPU: G1VelocityEnv --------------------------------------------------------------------------------------

    def cpu_context(self, env):
        motor = 1. if env._friction_scale is None else float(env._friction_scale[0])
        return np.array([float(env.model.geom_friction[env.foot_geoms[0], 0]), motor,
                         env.delay_steps / env.frame_skip, 0., 0., 0.])

    def cpu_before(self, env):
        return dict(observed_command=env.observed_command.copy(), command=env.command.copy(),
                    heading_target=float(env.heading_target), yaw_drift_before=float(env.yaw_drift_error[0]))

    def cpu_after(self, env, info, raw_before, action):
        components = info.get('reward_components', {})
        fields = dict(phase_steps=env.steps + env.phase_steps, steps=env.steps)
        return fields, sum(components.get(key, 0.) for key in self.excluded), env.previous_action.copy()

    def cpu_inputs(self, env, raw_before):
        """(9,) the kick (the CPU task has none: zeros, measured all the same) and the push's Δv of the step."""
        seen = getattr(env, 'exogenous', None) or {}
        nq = env.model.nq
        kick = np.zeros(6) if seen.get('qvel') is None else seen['qvel'] - raw_before[nq:nq + 6]
        pushes = seen.get('pushes')
        push = np.zeros(3) if pushes is None else np.asarray(pushes).sum(0) * env.model.opt.timestep / env.mass
        return np.concatenate([kick, push])

    # --- CUDA: the training's TensorG1Env ------------------------------------------------------------------------

    def cuda_context(self, env):
        import torch
        ref = env.reference
        n = env.n_envs
        friction = env.wp.to_torch(env.model.geom_friction)
        foot = int(ref.foot_geoms[0])
        foot = (friction[:, foot, 0] if friction.dim() == 3
                else torch.full((n,), float(ref.model.geom_friction[foot, 0]), device=env.device))
        ipos = env.wp.to_torch(env.model.body_ipos)
        if ipos.dim() == 3:  # randomize: the pelvis COM per world
            nominal = torch.as_tensor(ref.model.body_ipos[env.pelvis], dtype=ipos.dtype, device=env.device)
            com = ipos[:, env.pelvis] - nominal
        else:
            com = torch.zeros((n, 3), device=env.device)
        delay = env.delay_steps.float() / ref.frame_skip
        return torch.cat((foot[:, None], env.friction_scale[:, None], delay[:, None], com), 1)

    def cuda_before(self, env):
        return dict(observed_command=env.observed_command.clone(), command=env.command.clone(),
                    heading_target=env.heading_target.clone(), yaw_drift_before=env.yaw_drift_error.clone())

    def cuda_after(self, env):
        """Read in _masked_reset's hook: after lengths += 1, before the ended worlds reset."""
        return dict(phase_steps=env.lengths + env.phase_offset, steps=env.lengths.clone())

    def cuda_excluded(self, env):
        """(n,) the step's effort and self_contact terms (weighted, negative), as the step paid them."""
        from ..tasks import g1_tensor_math as task
        weights = env.reference.reward_weights
        part = -weights['effort'] * env.ctrl.square().sum(1)
        if env.self_contact_force is not None:
            part = part - weights['self_contact'] * task.self_contact_share(env.self_contact_force)
        return part

    def cuda_inputs(self, env, state):
        import torch
        seen = getattr(env, 'exogenous', None) or {}
        nq = env.qpos.shape[1]
        kick = seen['qvel'] - state[:, nq:nq + 6] if seen.get('qvel') is not None else torch.zeros(
            (env.n_envs, 6), device=env.device)
        pushes = seen.get('pushes')
        push = (torch.zeros((env.n_envs, 3), device=env.device) if pushes is None
                else pushes.sum(1) * env.reference.model.opt.timestep / env.reference.mass)
        return torch.cat((kick, push), 1)
