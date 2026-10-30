"""A learned G1 dynamics model (SuperMimic, arXiv 2610.28480 §3.2, appendices C and D; docs/lab-knowledge.md
«SuperMimic»), stage 1 of the world model experiment (Pavel, 27.10): can a learned model predict our getting up, the
contacts included, well enough to train a policy through it. Collected by scripts/lab/wm/collect.py, fitted and judged
by scripts/lab/wm/fit.py.

Since 27.10 this module is the task-free part of a general tool ("copies of any world", climbed as a ladder: the
SO-101 arm, G1 walking, G1 getting up): what makes the state, the context, the per-step inputs, the reward and the
termination of a task is its adapter's (humanoid_sim/world_model_adapters). The layouts below are the G1's free root
(Layout) and a fixed base (FixedLayout, the arm); Kinematics takes either; WorldModel takes optional per-step inputs (a
root velocity kick applied to the state analytically, a push's Δv). The paragraphs below describe the G1 getting up,
the first case; its numbers are unchanged.

State s. The collector writes the raw state (RAW): qpos (root position, quaternion w x y z, 29 joints), qvel (root
linear velocity in the world, angular velocity in the pelvis frame, joint speeds; MuJoCo's free joint) and the previous
applied action (the observation's last_action and, under action_delay, the target the motors still follow for the
step's first k substeps, so it is part of the physical state). Inside the model (Layout's canonical state) the root
orientation is the 6D of its rotation (the first two columns, Gram-Schmidt back; Zhou et al. 2019: continuous, no sign
ambiguity, no normalization to keep) and the root velocities are in the pelvis frame. The 14 TRACK_BODIES (every body)
are not part of the state: Kinematics computes their frames and frame-origin velocities from the root, the joints and
their speeds as mj_kinematics and g1_tensor_math.body_point_velocity do, differentiably, so the tracking reward, its
terminations (TrackTerms) and the policy's observation (policy_frame) follow from s by g1_tensor_math's formulas.
Predicting the bodies as well would give the model 14·18 more numbers to keep consistent with the joints.

Action a_t: the applied action, what the task's step passes to the PD and keeps as previous_action (the ±action_limit
box, the locked wrists' zeros), not the policy's sample: the physics sees the applied one, and the policy's sample maps
to it by a known function that imagination applies before the model. The delay is in the context and a_{t−1} in s.

Context ξ (CONTEXT, per episode: what the CUDA g1_getup task draws at a reset and the physics depends on): the body
friction (every colliding robot geom and the floor), the motor friction multiplier, the action delay (substeps /
frame_skip), the pelvis COM offset (m, x y z), the assist force (N / ASSIST_SCALE). The encoder bias is not in ξ: it
changes what the policy reads (policy_frame), not the physics; the collector keeps it apart.

Network (WorldModel): the invariant features of s (no xy, no yaw: the dynamics do not depend on them; pelvis height,
gravity in the pelvis frame, its velocities in its frame, joints relative to home, joint speeds, previous action), a_t
and ξ, each normalized by the data's mean and std, into an MLP (2048, 2048, 2048) with SiLU; out come s_{t+1} − s_t in
the pelvis frame of t (position, 6D(R_tᵀR_{t+1}) − 6D(I), joints, velocities), denormalized by the data's deltas; s_{t+1}'s
previous action is a_t, copied. A termination head (MLP 256, 256) on the features of s_{t+1} and goal features (for the
tracking task the termination margins of TrackTerms) gives the logit of d_{t+1}.

Loss (window_loss, the paper's Eq. 1 without the reward and observation heads, which are analytic here): k = WINDOW
steps rolled out autoregressively from a logged state on the logged actions, the sum over the steps of the mean squared
state error (Layout.error) in units of each coordinate's one-step change (error_scale), plus BCE of the termination;
steps after an episode boundary inside the window are masked. Adam 2e-4, windows of BATCH_WINDOWS, BATCHES_PER_UPDATE of
them per update, gradient norm clipped to 1 (their table A.5)."""
import numpy as np
import torch
from torch import nn

from .tasks import g1_tensor_math as task

VERSION = 1
WINDOW = 8
HIDDEN = (2048, 2048, 2048)
TERMINATION_HIDDEN = (256, 256)
LEARNING_RATE = 2e-4
BATCH_WINDOWS = 2048
BATCHES_PER_UPDATE = 10
GRAD_CLIP = 1.
CONTEXT = ('body_friction', 'motor_friction', 'delay', 'com_x', 'com_y', 'com_z', 'assist')
ASSIST_SCALE = 100.  # N: the assist force in ξ, so its column is of the same order as the others
# TrackTerms.margins: the anchor's height, its projected gravity's z and the four ends' heights against the reference,
# the fastest joint over GETUP_MAX_JOINT_SPEED, a bank start's grace (1 while its ends do not count).
TRACK_GOAL = 8
STD_FLOOR = 1e-6  # a column whose std is below this is constant in the data: left unscaled
ERROR_FLOOR = .1  # error_scale: at least this share of its group's median one-step change


def quat_matrix(quat):
    """(..., 3, 3) the rotations of unit quaternions w x y z (..., 4)."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(
                            *quat.shape[:-1], 3, 3)


def six_d(rotation):
    """(..., 6) the first two columns of rotations (..., 3, 3)."""
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), -1)


def from_six_d(six):
    """(..., 3, 3) the rotations of 6D numbers (..., 6) by Gram-Schmidt: the first column normalized, the second made
    orthogonal to it and normalized, the third their cross product."""
    first = nn.functional.normalize(six[..., :3], dim=-1)
    second = six[..., 3:]
    second = nn.functional.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack((first, second, torch.linalg.cross(first, second, dim=-1)), -1)


def vee(matrix):
    """(..., 3) the axial vector of the skew part of matrices (..., 3, 3): for a rotation the axis times the sine of
    its angle, the angle itself for small ones."""
    return .5 * torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0],
                             matrix[..., 1, 0] - matrix[..., 0, 1]), -1)


def _rotate(rotation, vector):
    return torch.einsum('...ij,...j->...i', rotation, vector)


def _unrotate(rotation, vector):
    return torch.einsum('...ji,...j->...i', rotation, vector)


class Layout:
    """The state's layouts for nu actuated hinge joints under a free root (the G1: 29): RAW as the collector writes it
    (qpos nq = 7 + nu, qvel nv = 6 + nu, previous action nu) and the model's canonical one (position 3, rotation 6,
    joints nu, root linear 3 and angular 3 velocity in the pelvis frame, joint speeds nu, previous action nu); the
    network's features and deltas and the error coordinates (position 3 in the world, rotation 3, joints, linear 3,
    angular 3, joint speeds), each with its slices. Every method takes any leading dimensions."""

    free_root = True

    def __init__(self, nu):
        self.nu = nu
        self.nq, self.nv = 7 + nu, 6 + nu
        self.raw = self.nq + self.nv + nu
        self.slices, start = {}, 0
        for name, size in (('position', 3), ('rotation', 6), ('joints', nu), ('linear', 3), ('angular', 3),
                           ('speeds', nu), ('previous', nu)):
            self.slices[name] = slice(start, start + size)
            start += size
        self.canonical = start
        self.features = 1 + 3 + 3 + 3 + 3 * nu
        self.deltas = 3 + 6 + nu + 3 + 3 + nu
        self.errors = 3 + 3 + nu + 3 + 3 + nu
        self.error_slices = dict(position=slice(0, 3), rotation=slice(3, 6), joints=slice(6, 6 + nu),
                                 linear=slice(6 + nu, 9 + nu), angular=slice(9 + nu, 12 + nu),
                                 speeds=slice(12 + nu, 12 + 2 * nu))
        # The report's groups of error coordinates: the root's pose, the joints, every velocity.
        self.error_groups = dict(root=(self.error_slices['position'], self.error_slices['rotation']),
                                 joints=(self.error_slices['joints'],),
                                 velocities=(self.error_slices['linear'], self.error_slices['angular'],
                                             self.error_slices['speeds']))

    def part(self, state, name):
        return state[..., self.slices[name]]

    def rotation(self, state):
        return from_six_d(self.part(state, 'rotation'))

    def from_raw(self, raw):
        """Canonical states of raw ones (..., RAW); the quaternion is normalized."""
        qpos, qvel, previous = raw[..., :self.nq], raw[..., self.nq:self.nq + self.nv], raw[..., self.nq + self.nv:]
        rotation = quat_matrix(nn.functional.normalize(qpos[..., 3:7], dim=-1))
        return torch.cat((qpos[..., :3], six_d(rotation), qpos[..., 7:], _unrotate(rotation, qvel[..., :3]),
                          qvel[..., 3:6], qvel[..., 6:], previous), -1)

    def network_features(self, state, home):
        """(..., features) the dynamics' view of a state, invariant to the root's xy and yaw: pelvis height, gravity in
        the pelvis frame, its linear and angular velocity in its frame, joints − home, joint speeds, previous action."""
        rotation = self.rotation(state)
        return torch.cat((self.part(state, 'position')[..., 2:3], -rotation[..., 2, :], self.part(state, 'linear'),
                          self.part(state, 'angular'), self.part(state, 'joints') - home, self.part(state, 'speeds'),
                          self.part(state, 'previous')), -1)

    def deltas_of(self, state, following):
        """(..., deltas) what the network predicts for the step state → following: the position change in the pelvis
        frame of `state`, 6D(RᵀR′) − 6D(I), and the joints', velocities' and joint speeds' changes."""
        rotation, turned = self.rotation(state), self.rotation(following)
        relative = rotation.transpose(-1, -2) @ turned
        identity = six_d(torch.eye(3, dtype=state.dtype, device=state.device))
        return torch.cat((_unrotate(rotation, self.part(following, 'position') - self.part(state, 'position')),
                          six_d(relative) - identity,
                          *(self.part(following, name) - self.part(state, name)
                            for name in ('joints', 'linear', 'angular', 'speeds'))), -1)

    def advance(self, state, delta, action):
        """The state after `state` by the deltas `delta` (deltas_of's layout), its previous action `action`."""
        nu = self.nu
        rotation = self.rotation(state)
        identity = six_d(torch.eye(3, dtype=state.dtype, device=state.device))
        turned = rotation @ from_six_d(identity + delta[..., 3:9])
        position = self.part(state, 'position') + _rotate(rotation, delta[..., :3])
        rest = 9
        parts = []
        for name, size in (('joints', nu), ('linear', 3), ('angular', 3), ('speeds', nu)):
            parts.append(self.part(state, name) + delta[..., rest:rest + size])
            rest += size
        return torch.cat((position, six_d(turned), *parts, action), -1)

    def error(self, predicted, true):
        """(..., errors) predicted − true: the root position in the world, the rotation vee(R_trueᵀ R_predicted), the
        joints, the root's linear and angular velocity in its own frame, the joint speeds."""
        rotation = vee(self.rotation(true).transpose(-1, -2) @ self.rotation(predicted))
        return torch.cat((self.part(predicted, 'position') - self.part(true, 'position'), rotation,
                          *(self.part(predicted, name) - self.part(true, name)
                            for name in ('joints', 'linear', 'angular', 'speeds'))), -1)

    def kicked(self, state, kick):
        """The states after a root velocity kick (..., 6) in MuJoCo's free-joint qvel convention (linear in the world,
        angular in the pelvis frame), as the CUDA task's REAL_PUSH adds it to qvel before the physics."""
        linear = self.part(state, 'linear') + _unrotate(self.rotation(state), kick[..., :3])
        angular = self.part(state, 'angular') + kick[..., 3:6]
        return torch.cat((state[..., :self.slices['linear'].start], linear, angular,
                          state[..., self.slices['angular'].stop:]), -1)

    def spread(self, rows, motion):
        """The report's «разброс» scale per error coordinate of states rows (m, canonical): the pelvis height's,
        joints' and velocities' std in the data, motion (the coordinate's change over the window) for xy and the
        rotation, which have no spread of their own."""
        spread = motion.clone()
        s = self.error_slices
        spread[s['position'].start + 2] = self.part(rows, 'position')[:, 2].std()
        for name in ('joints', 'linear', 'angular', 'speeds'):
            spread[s[name]] = self.part(rows, name).std(0)
        return spread


class FixedLayout:
    """The state's layouts for a fixed base (the SO-101 arm): `joints` hinge joints, `actions` actuators, RAW =
    canonical = qpos (joints) | qvel (joints) | the previous applied action (only with `previous`: a state whose physics
    keep it, as the G1's delay). Features: joints − home, speeds, previous action; deltas and errors: joints, speeds."""

    free_root = False

    def __init__(self, joints, actions, previous=False):
        self.nu = actions
        self.nq = self.nv = joints
        self.previous = previous
        self.raw = self.canonical = 2 * joints + (actions if previous else 0)
        self.slices = dict(joints=slice(0, joints), speeds=slice(joints, 2 * joints))
        if previous:
            self.slices['previous'] = slice(2 * joints, 2 * joints + actions)
        self.features = self.canonical
        self.deltas = self.errors = 2 * joints
        self.error_slices = dict(joints=slice(0, joints), speeds=slice(joints, 2 * joints))
        self.error_groups = dict(joints=(self.error_slices['joints'],), velocities=(self.error_slices['speeds'],))

    def part(self, state, name):
        return state[..., self.slices[name]]

    def rotation(self, state):
        raise ValueError('a fixed base has no root rotation')

    def from_raw(self, raw):
        return raw.clone()

    def network_features(self, state, home):
        parts = [self.part(state, 'joints') - home, self.part(state, 'speeds')]
        if self.previous:
            parts.append(self.part(state, 'previous'))
        return torch.cat(parts, -1)

    def deltas_of(self, state, following):
        return following[..., :self.deltas] - state[..., :self.deltas]

    def advance(self, state, delta, action):
        moved = state[..., :self.deltas] + delta
        return torch.cat((moved, action), -1) if self.previous else moved

    def error(self, predicted, true):
        return predicted[..., :self.errors] - true[..., :self.errors]

    def kicked(self, state, kick):
        speeds = self.part(state, 'speeds') + kick
        part = self.slices['speeds']
        return torch.cat((state[..., :part.start], speeds, state[..., part.stop:]), -1)

    def spread(self, rows, motion):
        spread = motion.clone()
        for name in ('joints', 'speeds'):
            spread[self.error_slices[name]] = self.part(rows, name).std(0)
        return spread


def layout_of(nu, root='free', joints=None, previous=True):
    """The state layout of a world: a free root (Layout, the G1: joints = nu, the previous action kept) or a fixed base
    (FixedLayout of `joints` hinges, nu actuators)."""
    if root == 'free':
        if (joints is not None and joints != nu) or not previous:
            raise ValueError('a free root layout has nu joints and keeps the previous action')
        return Layout(nu)
    if root != 'fixed':
        raise ValueError(f'unknown root {root}')
    return FixedLayout(nu if joints is None else joints, nu, previous)


class Kinematics:
    """Forward kinematics of a MuJoCo model whose bodies hang off one free root (the G1) or off the world (a fixed base,
    the SO-101) by at most one hinge each, in torch and differentiable: every body's frame origin, rotation, and the
    world linear and angular velocity of its frame origin from a canonical state, as mj_kinematics (xpos, xmat; a hinge
    turns by q − qpos0 about its axis through its anchor) and body_point_velocity of cvel at xpos give them. Mocap
    bodies (a fixed base's target marker) are not part of the state: __call__ gives them NaN, poses and points refuse
    them."""

    def __init__(self, model, device='cpu', dtype=torch.float32):
        import mujoco
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) > 1:
            raise ValueError('Kinematics needs at most one free joint')
        self.root = int(model.jnt_bodyid[free[0]]) if free else None
        free_q, free_v = (7, 6) if free else (0, 0)
        self.nbody = model.nbody
        self.mocap = {body for body in range(1, model.nbody) if model.body_mocapid[body] >= 0}

        def constant(value):
            return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=dtype, device=device)
        self.parent, self.offset, self.turn, self.joint = [], [], [], []
        hinge_index, hinge_zero, hinge_skew, hinge_square, hinge_turn = [], [], [], [], []
        for body in range(1, model.nbody):
            if self.root is not None and body != self.root and model.body_rootid[body] != self.root:
                raise ValueError('every body but the world must hang off the free root')
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body])
            self.parent.append(int(model.body_parentid[body]))
            self.offset.append(constant(model.body_pos[body]))
            self.turn.append(constant(rotation.reshape(3, 3)))
            joint = None
            if body != self.root:
                if model.body_jntnum[body] > 1:
                    raise ValueError('at most one joint per body')
                if model.body_jntnum[body] == 1:
                    j = int(model.body_jntadr[body])
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError('only hinge joints below the root')
                    index = int(model.jnt_qposadr[j]) - free_q
                    if index != int(model.jnt_dofadr[j]) - free_v:
                        raise ValueError('joint position and speed indices differ')
                    axis = model.jnt_axis[j] / np.linalg.norm(model.jnt_axis[j])
                    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    joint = dict(index=index, axis=constant(axis), anchor=constant(model.jnt_pos[j]),
                                 reach=constant(model.body_pos[body] + rotation.reshape(3, 3) @ model.jnt_pos[j]),
                                 slot=len(hinge_index))
                    hinge_turn.append(rotation.reshape(3, 3))
                    hinge_index.append(index)
                    hinge_zero.append(float(model.qpos0[model.jnt_qposadr[j]]))
                    hinge_skew.append(skew)
                    hinge_square.append(skew @ skew)
            self.joint.append(joint)
        # Every hinge's rotation of one state is computed in one pass (Rodrigues: I + sin·K + (1 − cos)·K²), slot by slot.
        self.hinge_index = torch.as_tensor(hinge_index, dtype=torch.long, device=device)
        self.hinge_zero = constant(hinge_zero)
        self.hinge_skew, self.hinge_square = constant(np.array(hinge_skew)), constant(np.array(hinge_square))
        self.hinge_turn = constant(np.array(hinge_turn))  # each hinge body's fixed rotation against its parent
        # The pelvis inertial frame against its body frame (the observation's angular velocity is in ximat).
        if self.root is not None:
            inertial = np.zeros(9)
            mujoco.mju_quat2Mat(inertial, model.body_iquat[self.root])
            self.root_inertial = constant(inertial.reshape(3, 3))
        self.dtype, self.device = dtype, device
        self._points = {}  # points' body lists: (the bodies posed, each point's index among them)

    def _hinges(self, state, layout):
        """(n, hinges, 3, 3) every hinge's rotation about its axis by q − qpos0, slot order."""
        angle = layout.part(state, 'joints')[:, self.hinge_index] - self.hinge_zero.to(state.dtype)
        return (torch.eye(3, dtype=state.dtype, device=state.device)
                + torch.sin(angle)[..., None, None] * self.hinge_skew.to(state.dtype)
                + (1 - torch.cos(angle))[..., None, None] * self.hinge_square.to(state.dtype))

    def ancestors(self, bodies):
        """The sorted body ids (≥ 1) `bodies` hang from, themselves included: what poses must walk through."""
        needed = set()
        for body in bodies:
            body = int(body)
            while body > 0 and body not in needed:
                needed.add(body)
                body = self.parent[body - 1]
        return sorted(needed)

    def poses(self, state, layout, bodies):
        """(position (n, len(bodies), 3), rotation (n, len(bodies), 3, 3)) of the bodies `bodies` (ids, in that
        order) of canonical states (n, canonical): __call__'s frames without the velocities and only along the bodies'
        chains (the termination margins of every rollout step need the anchor and the four ends alone)."""
        # A hinge body's frame: R = R_parent·turn·hinge, x = x_parent + R_parent·(offset + turn·anchor) − R·anchor; the
        # turn·hinge products of every hinge in one batched product.
        needed = self.ancestors(bodies)
        if self.mocap.intersection(needed):
            raise ValueError('mocap bodies are not part of the state')
        turned = self.hinge_turn.to(state.dtype) @ self._hinges(state, layout)
        position, rotation = {}, {}
        if self.root is None:  # a fixed base: the world's frame
            n = state.shape[0]
            position[0] = torch.zeros((n, 3), dtype=state.dtype, device=state.device)
            rotation[0] = torch.eye(3, dtype=state.dtype, device=state.device).expand(n, 3, 3)
        for body in needed:
            k = body - 1
            if body == self.root:
                position[body], rotation[body] = layout.part(state, 'position'), layout.rotation(state)
                continue
            x_parent, r_parent = position[self.parent[k]], rotation[self.parent[k]]
            joint = self.joint[k]
            if joint is None:
                x, r = x_parent + _rotate(r_parent, self.offset[k]), r_parent @ self.turn[k]
            else:
                r = r_parent @ turned[:, joint['slot']]
                x = x_parent + _rotate(r_parent, joint['reach']) - _rotate(r, joint['anchor'])
            position[body], rotation[body] = x, r
        bodies = [int(body) for body in bodies]
        return (torch.stack([position[body] for body in bodies], 1),
                torch.stack([rotation[body] for body in bodies], 1))

    def points(self, state, layout, bodies, local):
        """(n, P, 3) world positions of points fixed in bodies (P ids) at local offsets (P, 3) in their frames: sites
        (site_bodyid, site_pos) and geom centres (geom_bodyid, geom_pos) as mj_kinematics places them."""
        key = tuple(int(body) for body in bodies)
        if key not in self._points:  # made once, so a CUDA graph that calls this copies nothing from the host
            unique = sorted(set(key))
            self._points[key] = (unique, torch.as_tensor([unique.index(body) for body in key], dtype=torch.long,
                                                         device=self.device))
        unique, index = self._points[key]
        if not torch.is_tensor(local):
            local = torch.as_tensor(np.asarray(local, dtype=np.float64), device=state.device)
        position, rotation = self.poses(state, layout, unique)
        return position[:, index] + _rotate(rotation[:, index], local.to(state.dtype))

    def __call__(self, state, layout):
        """(position, rotation, linear, angular) of every body (n, nbody, 3) / (n, nbody, 3, 3), body id order (the
        world's zeros and identity first), of canonical states (n, canonical)."""
        n = state.shape[0]
        zero = torch.zeros((n, 3), dtype=state.dtype, device=state.device)
        identity = torch.eye(3, dtype=state.dtype, device=state.device).expand(n, 3, 3)
        position, rotation, linear, angular = [zero], [identity], [zero], [zero]
        speeds = layout.part(state, 'speeds')
        hinges = self._hinges(state, layout)
        for k, body in enumerate(range(1, self.nbody)):
            if body in self.mocap:  # not part of the state
                position.append(torch.full_like(zero, float('nan')))
                rotation.append(torch.full_like(identity, float('nan')))
                linear.append(torch.full_like(zero, float('nan')))
                angular.append(torch.full_like(zero, float('nan')))
                continue
            if body == self.root:
                turned = layout.rotation(state)
                position.append(layout.part(state, 'position'))
                rotation.append(turned)
                linear.append(_rotate(turned, layout.part(state, 'linear')))
                angular.append(_rotate(turned, layout.part(state, 'angular')))
                continue
            parent = self.parent[k]
            x_parent, r_parent = position[parent], rotation[parent]
            v_parent, w_parent = linear[parent], angular[parent]
            x = x_parent + _rotate(r_parent, self.offset[k])
            r = r_parent @ self.turn[k]
            joint = self.joint[k]
            if joint is None:
                w = w_parent
                v = v_parent + torch.linalg.cross(w_parent, x - x_parent, dim=-1)
            else:
                anchor = x + _rotate(r, joint['anchor'])
                r = r @ hinges[:, joint['slot']]
                x = anchor - _rotate(r, joint['anchor'])
                w = w_parent + _rotate(r, joint['axis']) * speeds[:, joint['index'], None]
                v = (v_parent + torch.linalg.cross(w_parent, anchor - x_parent, dim=-1)
                     + torch.linalg.cross(w, x - anchor, dim=-1))
            position.append(x)
            rotation.append(r)
            linear.append(v)
            angular.append(w)
        return torch.stack(position, 1), torch.stack(rotation, 1), torch.stack(linear, 1), torch.stack(angular, 1)


def policy_frame(state, layout, kinematics, home, encoder_bias=None):
    """(n, frame) the g1_getup actor's newest deploy_observation frame (command 0, phase stopped) of canonical states
    (n, canonical), before sensor noise: the pelvis angular velocity in its inertial frame, projected gravity, joints
    (+ encoder_bias (n, nu) when given) − home, joint speeds, previous action. The policy's input is the last
    `history` such frames (task.history_observation), noised and normalized as in training."""
    rotation = layout.rotation(state)
    n = state.shape[0]
    zero = torch.zeros((n, 3), dtype=state.dtype, device=state.device)
    angular = _unrotate(kinematics.root_inertial.to(state.dtype), layout.part(state, 'angular'))
    position = layout.part(state, 'joints')
    if encoder_bias is not None:
        position = position + encoder_bias
    return task.deploy_observation(torch.cat((angular, zero), -1), rotation, zero, position,
                                   layout.part(state, 'speeds'), home, layout.part(state, 'previous'),
                                   torch.zeros(n, dtype=state.dtype, device=state.device), phase_stop=True)


# g1_getup settings whose rules TrackTerms does not have (track_clips: many clips and the critic's future inputs;
# track_end_reach: an ankle's 3D distance ends an episode; track_handover_starts: hand-over starts with their own
# grace): data of such a stage would carry terminations the model's goal and the report cannot explain.
UNSUPPORTED_TRACK = ('track_clips', 'track_end_reach', 'track_handover_starts')


def check_track_config(task_config):
    """ValueError (Russian, for the lab) unless the g1_getup task_config is one TrackTerms knows: track_clip set and
    none of UNSUPPORTED_TRACK."""
    config = task_config or {}
    if not config.get('track_clip'):
        raise ValueError('модель мира: нужен этап слежения за одним клипом (task_config track_clip)')
    present = [key for key in UNSUPPORTED_TRACK if config.get(key) is not None]
    if present:
        raise ValueError(f'модель мира: правил {", ".join(present)} TrackTerms пока не знает (награда и обрывы '
                         f'такого этапа не совпадут с задачей); возьмите этап без них')


class TrackTerms:
    """g1_getup track_clip's reward and terminations from model states by g1_tensor_math's formulas (track_errors,
    track_reward_components, track_terminated), the tracked bodies by Kinematics: the CPU task `reference`
    (G1GetupEnv with the clip) gives the clip's frames, bodies, weights (track_weights' overrides included) and soft
    limits. The contact terms (self_contact, impact, sole_impact) need contact forces, which a state does not give: they
    are left out (zero). A task with one of UNSUPPORTED_TRACK is refused (check_track_config)."""

    def __init__(self, reference, device='cpu', dtype=torch.float32):
        if reference.clip is None:
            raise ValueError('модель мира: нужен этап слежения за одним клипом (task_config track_clip)')
        check_track_config(dict(track_clip=True, track_clips=getattr(reference, 'clips', None),
                                track_end_reach=getattr(reference, 'track_reach', None),
                                track_handover_starts=getattr(reference, 'track_handover', None)))

        def constant(value):
            return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=dtype, device=device)
        self.layout = Layout(reference.model.nu)
        self.kinematics = Kinematics(reference.model, device, dtype)
        self.frames = {key: constant(value) for key, value in reference.clip_frames.items()}
        self.bodies = torch.as_tensor(reference.track_bodies, dtype=torch.long, device=device)
        self.anchor, self.ends = reference.track_anchor, tuple(reference.track_ends)
        self.weights = dict(reference.track_weights)
        self.soft_low, self.soft_high = constant(reference.soft_low), constant(reference.soft_high)
        self.grace = int(reference.clip_grace)
        self.clip_last = int(reference.clip_last)  # the clip's last frame: clocks above it are the hold
        self.home = constant(reference.home)
        # margins' bodies (the anchor, then the ends; MuJoCo ids) and, per reference frame, what they are compared with:
        # the anchor's height, its rotation's R[2, 2] and the ends' heights. The reference's yaw and xy offset turn it
        # about z and move it in xy, which leaves heights and R[2, 2] as they are, so a frame's numbers are its own.
        self.margin_bodies = [int(reference.track_bodies[index]) for index in (self.anchor, *self.ends)]
        ends = list(self.ends)
        self.margin_table = torch.cat((self.frames['pos'][:, self.anchor, 2:3],
                                       self.frames['rot'][:, self.anchor, 2, 2:3],
                                       self.frames['pos'][:, ends, 2]), -1)

    def robot(self, state):
        """The tracked bodies of canonical states (n, canonical): positions, rotations, linear, angular velocities."""
        position, rotation, linear, angular = self.kinematics(state, self.layout)
        return position[:, self.bodies], rotation[:, self.bodies], linear[:, self.bodies], angular[:, self.bodies]

    def reference(self, clock, yaw, offset):
        dtype = self.frames['pos'].dtype
        return task.clip_reference(self.frames, clock.long(), yaw.to(dtype), offset.to(dtype))

    def reward(self, following, state, clock, yaw, offset):
        """(reward (n,), components) of the steps state → following (canonical (n, canonical)): the tracking terms of
        `following` against the reference at its clock (after the step), action_rate between following's previous
        action (the step's) and state's, joint_limit of following's joints; contact terms 0."""
        errors = task.track_errors(*self.robot(following), *self.reference(clock, yaw, offset), self.anchor)
        components = task.track_reward_components(
            errors, self.layout.part(following, 'previous'), self.layout.part(state, 'previous'),
            self.layout.part(following, 'joints'), self.soft_low, self.soft_high, self.weights)
        return sum(components.values()), components

    def margins(self, following, clock, yaw, offset, grace):
        """(n, TRACK_GOAL) what track_terminated and the joint speed rule compare, of canonical states (n, canonical)
        against the reference at clock: anchor height − reference's, its projected gravity z − reference's, the ends'
        heights − reference's (4), the fastest joint / GETUP_MAX_JOINT_SPEED, grace (n,) as 1 / 0. Only the anchor's and
        the ends' poses are computed (Kinematics.poses), and the reference's numbers come from margin_table: yaw and
        offset do not change them (kept in the signature, as reference's)."""
        position, rotation = self.kinematics.poses(following, self.layout, self.margin_bodies)
        index = clock.long().clamp(0, len(self.margin_table) - 1)
        reference = self.margin_table.to(following.dtype)[index]
        speed = self.layout.part(following, 'speeds').abs().amax(-1) / task.GETUP_MAX_JOINT_SPEED
        return torch.cat((position[:, :1, 2] - reference[:, :1], (rotation[:, 0, 2, 2] - reference[:, 1])[:, None],
                          position[:, 1:, 2] - reference[:, 2:], speed[:, None],
                          grace.to(following.dtype)[:, None]), -1)

    def grace_of(self, kind, steps):
        """(…) bool, a bank start's grace (its ends' rule off) of collect.py's start kind and episode steps after the
        step, as the task's _clip_step: kind 'bank' and steps ≤ clip_grace."""
        return (kind == task.TRACK_START_KINDS.index('bank')) & (steps <= self.grace)

    def terminated(self, following, clock, yaw, offset, grace):
        """(n,) bool, the task's termination of canonical states: track_terminated (the ends' rule off in a bank
        start's grace) or a joint faster than GETUP_MAX_JOINT_SPEED."""
        position, rotation, _, _ = self.robot(following)
        reference, turned, _, _ = self.reference(clock, yaw, offset)
        speed = self.layout.part(following, 'speeds').abs().amax(-1) > task.GETUP_MAX_JOINT_SPEED
        return task.track_terminated(position, rotation, reference, turned, self.anchor, self.ends, ~grace) | speed


def _mlp(inputs, hidden, outputs):
    layers, size = [], inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.SiLU()]
        size = width
    layers.append(nn.Linear(size, outputs))
    nn.init.zeros_(layers[-1].weight)  # starts at the data's mean delta / an even termination guess
    nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """s_{t+1} = f(s_t, a_t, ξ) and the logit of d_{t+1} = g(s_{t+1}, goal): canonical states (Layout), applied actions
    (nu), contexts (CONTEXT) and goal features (`goal` of them, 0: none). Normalization buffers are set from data by
    fit_normalization; home is the robot's default pose (the features' joints are relative to it).

    root, joints, previous: the state layout (layout_of; the G1's free root by default, 'fixed' for the arm). Per-step
    inputs (the collector's exogenous columns, world_model_adapters' Adapter.inputs): with kick their first 6 are a root
    qvel kick the task gave before the physics, applied to the state analytically (Layout.kicked) before the network
    sees it; with push the next 3 are a push's Δv in the world (m/s over the step), given to the network in the pelvis
    frame. A setting is written into `settings` only when it is not its default, so a getting-up model's file is as
    before and old files load."""

    def __init__(self, nu=29, context=len(CONTEXT), goal=0, hidden=HIDDEN, termination_hidden=TERMINATION_HIDDEN,
                 root='free', joints=None, previous=True, kick=False, push=False, history=1):
        super().__init__()
        self.layout = layout = layout_of(nu, root, joints, previous)
        self.settings = dict(version=VERSION, nu=nu, context=context, goal=goal, hidden=list(hidden),
                             termination_hidden=list(termination_hidden))
        for key, value, default in (('root', root, 'free'), ('joints', joints, None), ('previous', previous, True),
                                    ('kick', kick, False), ('push', push, False), ('history', history, 1)):
            if value != default:
                self.settings[key] = value
        self.kick, self.push = bool(kick), bool(push)
        self.push_columns = slice(6 if kick else 0, (6 if kick else 0) + 3)
        # history > 1: the dynamics also see the features of the history − 1 states before (oldest first), so what
        # one state hides (the solver's warm start, an acceleration) can be read from the change.
        self.history = int(history)
        inputs = layout.features * self.history + nu + context + 3 * self.push
        self.dynamics = _mlp(inputs, hidden, layout.deltas)
        self.termination = _mlp(layout.features + goal, termination_hidden, 1)
        for name, size in (('input', inputs), ('delta', layout.deltas), ('goal', layout.features + goal)):
            self.register_buffer(f'{name}_mean', torch.zeros(size))
            self.register_buffer(f'{name}_std', torch.ones(size))
        self.register_buffer('error_scale', torch.ones(layout.errors))
        self.register_buffer('home', torch.zeros(nu))

    def _exogenous(self, state, inputs):
        """(state after the kick, the push block or None) of states and their per-step inputs (…, E) or None (zeros)."""
        if not (self.kick or self.push):
            return state, None
        if inputs is None:
            width = (6 if self.kick else 0) + 3 * self.push
            inputs = torch.zeros((*state.shape[:-1], width), dtype=state.dtype, device=state.device)
        if self.kick:
            state = self.layout.kicked(state, inputs[..., :6])
        push = _unrotate(self.layout.rotation(state), inputs[..., self.push_columns]) if self.push else None
        return state, push

    def past_features(self, past):
        """(n, (history − 1)·features) of the earlier canonical states past (n, history − 1, canonical), oldest first."""
        return self.layout.network_features(past, self.home).flatten(-2)

    def network_inputs(self, state, action, context, push=None, past=None):
        parts = (self.layout.network_features(state, self.home), action, context)
        if past is not None:
            parts = (self.past_features(past), *parts)
        return torch.cat(parts if push is None else (*parts, push), -1)

    def step(self, state, action, context, inputs=None, past=None):
        """The next canonical states (n, canonical) of states (n, canonical) under applied actions (n, nu) in contexts
        (n, context), with per-step inputs (n, E) when the model takes them (None: zeros) and, with history > 1, the
        earlier states past (n, history − 1, canonical), oldest first."""
        if self.history > 1:
            if past is None:
                raise ValueError('world model with history: the earlier states are needed')
            kicked, push = self._exogenous(state, inputs)
            inputs = self.network_inputs(kicked, action, context, push, past)
            delta = self.dynamics((inputs - self.input_mean) / self.input_std) * self.delta_std + self.delta_mean
            return self.layout.advance(kicked, delta, action)
        if inputs is None and not (self.kick or self.push):
            inputs = torch.cat((self.layout.network_features(state, self.home), action, context), -1)
            delta = self.dynamics((inputs - self.input_mean) / self.input_std) * self.delta_std + self.delta_mean
            return self.layout.advance(state, delta, action)
        state, push = self._exogenous(state, inputs)
        inputs = self.network_inputs(state, action, context, push)
        delta = self.dynamics((inputs - self.input_mean) / self.input_std) * self.delta_std + self.delta_mean
        return self.layout.advance(state, delta, action)

    def termination_logit(self, state, goal=None):
        """(n,) the logit that canonical states (n, canonical) end their episodes; goal (n, goal) or None (zeros)."""
        features = self.layout.network_features(state, self.home)
        size = self.settings['goal']
        if size:
            goal = torch.zeros((len(state), size), dtype=state.dtype, device=state.device) if goal is None else goal
            features = torch.cat((features, goal), -1)
        return self.termination((features - self.goal_mean) / self.goal_std)[..., 0]

    def rollout(self, state, actions, context, goal=None, inputs=None, past=None):
        """(states (n, k, canonical), termination logits (n, k)) of k steps from states (n, canonical) on actions (n, k,
        nu) in contexts (n, context); goal(j, state_j) gives step j's goal features (n, goal) or None; inputs (n, k, E)
        the steps' per-step inputs or None."""
        states, logits = [], []
        for j in range(actions.shape[1]):
            before = state
            state = self.step(state, actions[:, j], context, None if inputs is None else inputs[:, j], past)
            if past is not None:  # the history slides: the state just left is the newest earlier one
                past = torch.cat((past[:, 1:], before[:, None]), 1)
            states.append(state)
            logits.append(self.termination_logit(state, None if goal is None else goal(j, state)))
        return torch.stack(states, 1), torch.stack(logits, 1)

    def save(self, path, **extra):
        torch.save(dict(settings=self.settings, state=self.state_dict(), extra=extra), path)

    @classmethod
    def load(cls, path, device='cpu'):
        """(model, extra) of a file save wrote; another VERSION is refused."""
        saved = torch.load(path, map_location=device, weights_only=False)
        settings = dict(saved['settings'])
        if settings.pop('version', None) != VERSION:
            raise ValueError('world model of another version')
        model = cls(**settings).to(device)
        model.load_state_dict(saved['state'])
        return model, saved.get('extra', {})



class Ensemble(nn.Module):
    """World models fitted on the same data (another seed, size or loss) as one: a step is the members' mean next state
    (the 6D rotation averaged, from_six_d orthonormalizes it), the termination logit their mean; disagreement gives the
    members' spread of a step, which says where the model should not be trusted. Same layout and inputs; the history
    may differ (the ensemble's is the longest, each member gets the newest of the earlier states it reads)."""

    def __init__(self, members):
        super().__init__()
        members = list(members)
        first = members[0]
        for member in members[1:]:
            if (member.settings['nu'], member.settings['context'], member.settings['goal'],
                    member.kick, member.push, type(member.layout)) != (
                    first.settings['nu'], first.settings['context'], first.settings['goal'],
                    first.kick, first.push, type(first.layout)):
                raise ValueError('ensemble members of other layouts')
        self.members = nn.ModuleList(members)
        self.layout, self.kick, self.push = first.layout, first.kick, first.push
        self.history = max(member.history for member in members)
        self.settings = dict(first.settings, members=len(members))
        self.register_buffer('error_scale', first.error_scale.clone())
        self.register_buffer('home', first.home.clone())

    def predictions(self, state, action, context, inputs=None, past=None):
        """(members, n, canonical) every member's next states."""
        return torch.stack([member.step(state, action, context, inputs,
                                        None if member.history == 1 else past[:, past.shape[1] - member.history + 1:])
                            for member in self.members])

    def step(self, state, action, context, inputs=None, past=None):
        return self.predictions(state, action, context, inputs, past).mean(0)

    def disagreement(self, state, action, context, inputs=None, past=None):
        """(n, errors) the members' spread of the next state, each error coordinate: the RMS of each member's error
        against the mean prediction."""
        every = self.predictions(state, action, context, inputs, past)
        mean = every.mean(0)
        return torch.stack([self.layout.error(one, mean) for one in every]).square().mean(0).sqrt()

    def termination_logit(self, state, goal=None):
        return torch.stack([member.termination_logit(state, goal) for member in self.members]).mean(0)

    rollout = WorldModel.rollout

    @classmethod
    def load(cls, paths, device='cpu'):
        return cls(WorldModel.load(path, device)[0] for path in paths)

def _spread(values):
    """(mean, std) per column of samples (m, k), a std below STD_FLOOR set to 1 (a constant column stays unscaled)."""
    mean, std = values.mean(0), values.std(0)
    return mean, torch.where(std < STD_FLOOR, torch.ones_like(std), std)


@torch.no_grad()
def fit_normalization(model, state, action, context, following, goal=None, home=None, inputs=None, past=None):
    """Sets the model's normalization from transitions within episodes (state → following, canonical (m, canonical);
    actions (m, nu), contexts (m, context), goal features (m, goal) of following or None, per-step inputs (m, E) or
    None): the inputs' and deltas' mean and std (of the kicked state, as step sees it), the termination head's, and
    error_scale, each error coordinate's one-step change std, at least ERROR_FLOOR of its group's median (a nearly still
    coordinate, the locked wrists, would weigh a thousand times the rest)."""
    layout = model.layout
    if home is not None:
        model.home.copy_(torch.as_tensor(home, dtype=model.home.dtype))
    home = model.home.to(state.dtype)
    kicked = state
    if inputs is None and not (model.kick or model.push) and past is None:
        inputs = torch.cat((layout.network_features(state, home), action, context), -1)
    else:
        kicked, push = model._exogenous(state, inputs)
        inputs = model.network_inputs(kicked, action, context, push, past)
    mean, std = _spread(inputs)
    model.input_mean.copy_(mean), model.input_std.copy_(std)
    mean, std = _spread(layout.deltas_of(kicked, following))
    model.delta_mean.copy_(mean), model.delta_std.copy_(std)
    features = layout.network_features(following, home)
    if model.settings['goal']:
        features = torch.cat((features, goal), -1)
    mean, std = _spread(features)
    model.goal_mean.copy_(mean), model.goal_std.copy_(std)
    change = layout.error(following, state).std(0)
    scale = change.clone()
    for part in layout.error_slices.values():
        floor = ERROR_FLOOR * change[part].median()
        scale[part] = torch.maximum(change[part], floor.clamp(min=STD_FLOOR))
    model.error_scale.copy_(scale)


def window_loss(model, state, actions, context, targets, terminated, alive, goal=None, inputs=None, past=None):
    """(loss, parts, predicted states) of windows: states (n, canonical), actions (n, k, nu), contexts (n, context),
    the true following states (n, k, canonical; anything where not alive, a non-finite state too), termination labels
    (n, k) and alive (n, k), whether step j is still in the window's first episode (0 from the step after a boundary);
    per-step inputs (n, k, E) or None. Σ over the steps of the mean squared error in error_scale units, plus Σ over the
    steps of BCE, each averaged over the windows."""
    states, logits = model.rollout(state, actions, context, goal, inputs, past)
    # Where not alive the target is the prediction itself, so a non-finite logged state gives no NaN gradient.
    targets = torch.where(alive[..., None], torch.nan_to_num(targets), states.detach())
    error = model.layout.error(states, targets) / model.error_scale
    weight = alive.to(error.dtype)
    squared = torch.where(alive, error.square().mean(-1), 0.)
    mse = squared.sum(1).mean()
    bce = (nn.functional.binary_cross_entropy_with_logits(logits, terminated.to(logits.dtype), reduction='none')
           * weight).sum(1).mean()
    return mse + bce, dict(mse=mse.detach(), bce=bce.detach()), states, logits
