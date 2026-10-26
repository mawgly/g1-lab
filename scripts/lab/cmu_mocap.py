"""The CMU motion capture database's ASF skeleton and AMC motion files (mocap.cs.cmu.edu; Acclaim format), read as text
and turned into world positions and rotations of the bones, frame by frame. Nothing in the files is executed: each line
is split into a name and numbers.

ASF: the bones' rest direction (unit, world axes of the rest pose), length, local axis (Euler XYZ in degrees) and
degrees of freedom, the hierarchy. AMC: per frame (a line with its number) one line per bone, the root's TX TY TZ RX RY
RZ and each bone's rotations in the order of its dof. Forward kinematics as the Acclaim format defines it (and the
database's own viewer): a bone's rotation is parent · C · Rz Ry Rx · C⁻¹ (C its axis, Rx Ry Rz its frame's angles), its
start the parent's end, its end start + length · rotation · direction. The root's rotation is C · Rz Ry Rx · C⁻¹ of its
own angles.

Lengths and the root position are in the file's units; `:units length 0.45` means value / 0.45 inches, so metres are
value · 0.0254 / 0.45. The data is Y up; `points` returns Z up (x, y, z) = (z_cmu, x_cmu, y_cmu), a right-handed turn.
"""
import math
from pathlib import Path

import mujoco
import numpy as np

INCH = 0.0254
CMU_TO_ZUP = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # world z up <- CMU y up


def _euler(angles_deg):
    """Rz · Ry · Rx of angles (x, y, z) in degrees (Acclaim's XYZ order: x first)."""
    x, y, z = (math.radians(a) for a in angles_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _numbers(words):
    return [float(w) for w in words]


def read_asf(path):
    """{name, length_scale (metres per file unit), root_order, bones: {name: dict(direction, length, axis, dof)},
    children: {name: [names]}, order: names parents first} of an ASF file."""
    lines = [line.split('#', 1)[0].strip() for line in Path(path).read_text(encoding='latin-1').splitlines()]
    section, bones, bone, children, scale, root_order = None, {}, None, {}, 1., []
    for line in lines:
        if not line:
            continue
        words = line.split()
        if words[0].startswith(':'):
            section = words[0][1:]
            if section == 'root':
                bones['root'] = dict(direction=np.zeros(3), length=0., axis=np.zeros(3), dof=[])
            continue
        if section == 'units' and words[0] == 'length':
            scale = INCH / float(words[1])
        elif section == 'root':
            if words[0] == 'order':
                root_order = [w.lower() for w in words[1:]]
            elif words[0] == 'axis' and words[1].upper() != 'XYZ':
                raise ValueError(f'{path}: root axis order {words[1]}, only XYZ is read')
        elif section == 'bonedata':
            if words[0] == 'begin':
                bone = {}
            elif words[0] == 'end':
                bones[bone['name']] = dict(direction=np.array(bone['direction']), length=bone['length'],
                                           axis=np.array(bone['axis']), dof=bone.get('dof', []))
                bone = None
            elif words[0] == 'name':
                bone['name'] = words[1]
            elif words[0] == 'direction':
                bone['direction'] = _numbers(words[1:4])
            elif words[0] == 'length':
                bone['length'] = float(words[1])
            elif words[0] == 'axis':
                if words[4].upper() != 'XYZ':
                    raise ValueError(f'{path}: bone {bone["name"]} axis order {words[4]}, only XYZ is read')
                bone['axis'] = _numbers(words[1:4])
            elif words[0] == 'dof':
                bone['dof'] = [w.lower() for w in words[1:]]
        elif section == 'hierarchy' and words[0] not in ('begin', 'end'):
            children.setdefault(words[0], []).extend(words[1:])
    if root_order[:6] != ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']:
        raise ValueError(f'{path}: root order {root_order}, expected TX TY TZ RX RY RZ')
    order, stack = [], ['root']
    while stack:
        name = stack.pop(0)
        order.append(name)
        stack.extend(children.get(name, []))
    missing = set(bones) - set(order)
    if missing:
        raise ValueError(f'{path}: bones outside the hierarchy {sorted(missing)}')
    return dict(length_scale=scale, bones=bones, children=children, order=order)


def read_amc(path):
    """[(frame number, {bone: [values]})] of an AMC file, in file order."""
    frames, current = [], None
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith(':'):
            continue
        words = line.split()
        if len(words) == 1 and words[0].isdigit():
            current = {}
            frames.append((int(words[0]), current))
        elif current is not None:
            current[words[0]] = _numbers(words[1:])
        else:
            raise ValueError(f'{path}: values before the first frame number: {line[:40]!r}')
    return frames


def pose(skeleton, values):
    """{bone: (start (3,), end (3,), rotation (3, 3))} of one AMC frame, in metres, CMU axes (y up)."""
    bones, scale = skeleton['bones'], skeleton['length_scale']
    parent = {child: name for name, kids in skeleton['children'].items() for child in kids}
    out = {}
    for name in skeleton['order']:
        bone = bones[name]
        c = _euler(bone['axis'])
        if name == 'root':
            root = values['root']
            rotation = c @ _euler(root[3:6]) @ c.T
            position = np.array(root[:3]) * scale
            out[name] = (position, position, rotation)
            continue
        angles = [0., 0., 0.]
        for axis, value in zip(bone['dof'], values.get(name, [])):
            angles['xyz'.index(axis[1])] = value
        rotation = out[parent[name]][2] @ c @ _euler(angles) @ c.T
        start = out[parent[name]][1]
        out[name] = (start, start + bone['length'] * scale * rotation @ bone['direction'], rotation)
    return out


def points(skeleton, frames):
    """(names, starts (F, B, 3), ends (F, B, 3), rotations (F, B, 3, 3)) of the frames' poses, metres, Z up. A bone's
    rotation turns its rest pose into the frame's (world axes), so the identity is the ASF rest pose: upright, facing
    +x (the toes), left +y, as the G1's pelvis, torso and feet in the home pose."""
    names = list(skeleton['order'])
    starts, ends, rotations = [], [], []
    for _, values in frames:
        p = pose(skeleton, values)
        starts.append([CMU_TO_ZUP @ p[n][0] for n in names])
        ends.append([CMU_TO_ZUP @ p[n][1] for n in names])
        rotations.append([CMU_TO_ZUP @ p[n][2] @ CMU_TO_ZUP.T for n in names])
    return names, np.array(starts), np.array(ends), np.array(rotations)


# Retargeting onto the G1 29 DoF (our own IK in the getting-up model, no GMR), per frame:
# - the legs' segments and the upper arms point where the person's do (hip roll -> knee -> ankle roll origins against
#   hip -> knee -> ankle; shoulder roll -> elbow origins against shoulder -> elbow);
# - each hand (wrist roll origin) sits where the person's does relative to the shoulder, scaled by the arms' length
#   ratio: a position, not a forearm direction, since the G1's straight arm is not straight between these origins;
# - the pelvis turns as the person's root; the torso as the person's thorax, less the spine's bend in a neutral frame
#   (the person lying flat; the subject's spine reads 15-20° bent there and standing); each foot's forward axis as
#   the person's ankle -> ball of the foot (measured from the ASF rest pose, the G1's home orientation; a foot's roll
#   is left to the pull toward home);
# - where the person touches the floor (CONTACTS), the G1's matching body's lowest point is on it, and no robot geom
#   is below it: the G1's limbs are shorter than the person's by different ratios (arms 0.58, legs 0.80), so matching
#   directions alone left it hanging on one hand or floating above its back;
# - the root's translation follows the person's root scaled by the legs' ratio (lightly: the contacts set it);
# - two robot geoms stay SELF_GAP apart; the joints within the task's soft limits;
# - every joint is pulled lightly toward home (the wrist roll, which nothing else sees; the ankle roll harder); the
#   locked wrists stay at home.
# (name, G1 body at the start, at the end, the CMU bones starting at each joint, weight)
SEGMENTS = (
    ('left_thigh', 'left_hip_roll_link', 'left_knee_link', 'lfemur', 'ltibia', 2.),
    ('left_shank', 'left_knee_link', 'left_ankle_roll_link', 'ltibia', 'lfoot', 2.),
    ('right_thigh', 'right_hip_roll_link', 'right_knee_link', 'rfemur', 'rtibia', 2.),
    ('right_shank', 'right_knee_link', 'right_ankle_roll_link', 'rtibia', 'rfoot', 2.),
    ('left_upper_arm', 'left_shoulder_roll_link', 'left_elbow_link', 'lhumerus', 'lradius', 1.),
    ('right_upper_arm', 'right_shoulder_roll_link', 'right_elbow_link', 'rhumerus', 'rradius', 1.),
)
# (name, G1 body at the shoulder, at the point, the CMU bones there, weight per metre)
REACHES = (
    ('left_hand', 'left_shoulder_roll_link', 'left_wrist_roll_link', 'lhumerus', 'lwrist', 5.),
    ('right_hand', 'right_shoulder_roll_link', 'right_wrist_roll_link', 'rhumerus', 'rwrist', 5.),
)
# (name, G1 body, CMU bone whose rotation it follows, weight)
ORIENTED = (('pelvis', 'pelvis', 'root', 2.), ('torso', 'torso_link', 'thorax', 1.5))
# (name, G1 body, CMU bone at the start of the person's direction, at its end, weight); the body's axis is that
# direction in the rest pose
AXES = (('left_foot', 'left_ankle_roll_link', 'lfoot', 'ltoes', .5),
        ('right_foot', 'right_ankle_roll_link', 'rfoot', 'rtoes', .5))
ANKLE_ROLL_PULL = .3  # the ankle roll's pull toward home, per rad
LEG_BONES = ('lfemur', 'ltibia', 'rfemur', 'rtibia')
ARM_BONES = ('lhumerus', 'lradius', 'rhumerus', 'rradius')
G1_LEGS = (('left_hip_roll_link', 'left_knee_link'), ('left_knee_link', 'left_ankle_roll_link'),
           ('right_hip_roll_link', 'right_knee_link'), ('right_knee_link', 'right_ankle_roll_link'))
G1_ARMS = (('left_shoulder_roll_link', 'left_elbow_link'), ('left_elbow_link', 'left_wrist_roll_link'),
           ('right_shoulder_roll_link', 'right_elbow_link'), ('right_elbow_link', 'right_wrist_roll_link'))
# Floor contacts: where the person touches the floor, the G1's matching body does too (its lowest point at z = 0).
# (name, G1 bodies, the person's points (bone, 'start' | 'end'), the height of the lowest point above the person's
# floor (m) below which the contact counts fully, above which not at all; linear between). The heights are the joint
# centres' when touching (subject 140: pelvis and lower back 0.08-0.09 m lying, the balls and tips of the feet
# 0.05-0.11 standing flat, 0.08-0.11 on the toes, wrists and hands 0.03-0.07, elbows ~0). A foot, a knee or a hand
# (PLANTED) counts only while still: its points' mean moves horizontally slower than STILL_SPEED (smoothed over
# ±STILL_FRAMES), fading out by STILL_SPEED[1], so a foot swept low over the floor is not on it.
CONTACTS = (
    ('pelvis', ('pelvis',), (('root', 'start'),), .11, .15),
    ('back', ('torso_link',), (('upperback', 'start'),), .1, .14),
    ('left_foot', ('left_ankle_roll_link',), (('ltoes', 'start'), ('ltoes', 'end')), .11, .15),
    ('right_foot', ('right_ankle_roll_link',), (('rtoes', 'start'), ('rtoes', 'end')), .11, .15),
    ('left_knee', ('left_knee_link',), (('ltibia', 'start'),), .09, .13),
    ('right_knee', ('right_knee_link',), (('rtibia', 'start'),), .09, .13),
    ('left_elbow', ('left_elbow_link',), (('lradius', 'start'),), .03, .06),
    ('right_elbow', ('right_elbow_link',), (('rradius', 'start'),), .03, .06),
    ('left_hand', ('left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link'),
     (('lwrist', 'start'), ('lhand', 'start'), ('lhand', 'end')), .06, .09),
    ('right_hand', ('right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link'),
     (('rwrist', 'start'), ('rhand', 'start'), ('rhand', 'end')), .06, .09),
)
PLANTED = ('left_foot', 'right_foot', 'left_knee', 'right_knee', 'left_hand', 'right_hand')
STILL_SPEED, STILL_FRAMES = (.25, .5), 6  # m/s: full weight below, none above; frames at the source rate
HOME_PULL = .03  # weight of every joint's pull toward home, per rad
DAMPING = 1e-3  # Levenberg-Marquardt damping of a step
FIRST_ITERATIONS, ITERATIONS = 200, 30
SELF_GAP, SELF_WEIGHT = .01, 50.  # m, per m: two robot geoms closer than the gap push apart
MAX_STEP = .1  # rad (m for the root): the largest coordinate change of one iteration
CONTACT_WEIGHT, FLOOR_WEIGHT = 30., 50.  # per m: a contact's lowest point to the floor; any geom out of it
ROOT_WEIGHT, ROOT_HEIGHT_WEIGHT = 1., .1  # per m: the root's xy to the person's (scaled), its height to theirs
FLOOR_PERCENTILE = 1.  # the person's floor: this percentile of their lowest joint over the take
TARGETS = tuple(t[0] for t in SEGMENTS + REACHES + ORIENTED + AXES)
REACH_NAMES = tuple(t[0] for t in REACHES)  # their errors are distances (m), the others' angles (rad)
# retarget's errors: TARGETS', then the largest |height| of a counted contact's lowest point and the deepest floor
# penetration, m
ERRORS = TARGETS + ('contacts', 'floor', 'support')
DISTANCES = REACH_NAMES + ('contacts', 'floor', 'support')
# Physical doability (iteration 67, the critic's lab-critic/it67: the first CMU clips had the centre of mass 10-25 cm
# off the support while the support changed, hands lifted with a knee still down, a knee asked for 185 N·m of 111, IK
# branch jumps of the wrist roll by 2.4 rad a frame; scripts/lab/probes/clip_feasibility.py judges a clip).
# On (measured on 140_01 frames 235-571 against the iteration-64 IK: acceleration noise 57.8 -> 6 rad/s², the fastest
# joint 42.9 -> 5.4 rad/s, the worst torque need 21 -> 4 times what the joint has; still not doable):
# - continuity: every joint pulled toward the frame before (PREVIOUS_PULL) and no faster than JOINT_SPEED, the root no
#   faster than ROOT_SPEED / ROOT_TURN, so a straight arm's free twist and a hand's roll on the floor no longer jump
#   between IK branches; the joints smoothed over SMOOTH_SECONDS (smooth) before the clip goes to 50 Hz;
# - a foot, a knee or a hand counts as on the floor only while still (STILL_SPEED);
# - settle by IK (SETTLE_IK): both feet flat (all eight sole spheres on the floor), pinned on their way to the home
#   stance, the centre of mass COM_MARGIN inside the soles.
# Off by default (implemented; in this one-frame-at-a-time Gauss-Newton IK each made 140_01 worse, the support moving
# under conflicting one-sided terms: the centre of mass 44-59 cm off the support instead of 27-32; kept for the next
# attempt, which needs a planner, not more weights; set the weight or the switch to try one):
# - support (COM_WEIGHT): every robot geom touching the floor, its points within SUPPORT_TOUCH (support_points); the
#   centre of mass at least COM_MARGIN inside their convex hull (support_term: the hull's own motion counted);
# - a flat foot (FLAT_WEIGHT): where the person's foot is down with its sole within FLAT_TILT of level, all four sole
#   spheres of the G1's foot on the floor, both toe spheres where it is on its toes (TOES_UP); SQUAT_FLAT: a squat on
#   the toes made flat-footed, that leg's targets weaker (SQUAT_LEG);
# - pins (PIN_WEIGHT): a flat foot, a knee or a hand on the floor keeps the xy where it came down;
# - weight transfer (PAUSE_SECONDS): when the person lifts a hand, a knee or a foot (PLANTED) that the rest of the
#   robot's support would not carry with the centre of mass RELEASE_MARGIN inside it, the person's clock waits on the
#   frame before (the contact still down, their pose's targets PAUSE_TARGETS weaker) while the centre of mass moves
#   over the rest at TRANSFER_SPEED (TRANSFER_WEIGHT), at most PAUSE_SECONDS; the clip gets longer by those frames;
# - a limit zone (LIMIT_WEIGHT): a joint within LIMIT_ZONE of its IK limit pushed back inside the solve;
# - torques (ROUNDS > 1): after each frame, the static torques of the pose (clip_feasibility.FrameChecker) against
#   what each joint has; a joint that needs more than its PD gives at the stop (kp·(limit − q), the waist's and
#   ankles' kp is 28) gets its IK limit moved off that stop by TORQUE_MARGIN·τ/kp, spread over ±LIMIT_SECONDS of the
#   person's frames, and the take is solved again;
# - project (PROJECT): after retarget, each frame solved again as settle is, pulled toward the retargeted one, with
#   its touching parts kept on the floor, pinned, flat or on the toes, the support and the weight transfers; on 140_01
#   the hands then hover 1-3 cm over the floor in some frames and the centre of mass ends 63 cm off the support.
RETARGET_VERSION = 'cmu-ik-v2'  # in a clip's header (v1: the iteration-64 IK)
SUPPORT_TOUCH = .005  # m (the check's touch is 1 cm: the IK's support is the stricter)
COM_MARGIN, COM_WEIGHT = .03, 0.  # m; per m of shortfall (200 to try)
TRANSFER_WEIGHT, TRANSFER_SPEED, PAUSE_SECONDS = 2000., .25, 0.  # per m; m/s; the longest wait, s (2 to try)
FLAT_TILT = math.radians(35.)
FLAT_WEIGHT = 0.  # per m: each sole sphere's lowest point to the floor (CONTACT_WEIGHT to try)
SQUAT_FLAT, SQUAT_LEG = False, .2
TOES_UP = .2  # a foot down but not flat, its sole facing at least this much down (z of its up): both toe spheres down
LIMIT_ZONE, LIMIT_WEIGHT = .02, 0.  # rad; per rad
PAUSE_TARGETS = .2
RELEASE_MARGIN = .01  # m
PIN_WEIGHT = 0.  # per m (20 to try)
HANDS = ('left_hand', 'right_hand')
FEET = ('left_foot', 'right_foot')
PREVIOUS_PULL = .3  # per rad
ROOT_SPEED, ROOT_TURN = 1.5, 4.  # m/s, rad/s: the root's fastest from one source frame to the next
JOINT_SPEED = 8.  # rad/s: the fastest a joint moves from one source frame to the next (LAFAN's clip peaks at 8.3)
SMOOTH_SECONDS = .025  # σ of the Gaussian over the joints at the source rate
TORQUE_MARGIN = 1.25
LIMIT_SECONDS = .25
ROUNDS = 1
SETTLE_IK = True  # settle by IK with flat feet (False: the iteration-64 interpolation)
SETTLE_PIN, SETTLE_FLAT, SETTLE_COM = 20., 30., 200.  # per m: settle's pins, flat soles, centre of mass
SETTLE_RAMP, SETTLE_HOME = .3, .005  # the share of settle over which flat and support weights come in; rad / m:
# the last frame is put at the home pose exactly when it is that close
SETTLE_PULL, SETTLE_ANKLE_PULL = 1., .02  # per rad: settle's pull toward the smoothstep pose, the ankles'
# project: the parts whose floor contact it keeps (name, G1 bodies, the first the pinned one)
PARTS = (('left_foot', ('left_ankle_roll_link',)), ('right_foot', ('right_ankle_roll_link',)),
         ('left_knee', ('left_knee_link',)), ('right_knee', ('right_knee_link',)),
         ('left_hand', ('left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link')),
         ('right_hand', ('right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link')))
PROJECT = False  # run project after retarget and smooth (off: see the comment above RETARGET_VERSION)
PROJECT_TOUCH = .01  # m above the frame's lowest geom: a part touching in the source frame
PROJECT_PULL, PROJECT_ANKLE_PULL = 1., .1  # per rad toward the source frame's joints
PROJECT_PIN, PROJECT_FLAT, PROJECT_COM = 20., 30., 200.  # per m
PAUSE_SECONDS_PROJECT = 2.
SOURCE_RATE = 120.  # frames/s of the CMU takes (the stillness, the pauses and the limits' spread count in them)


def length_ratio(model, data, home_qpos, skeleton, g1_pairs, bones):
    """The G1's limb length (the origins of g1_pairs, home) over the person's (the bones' lengths), mean of both
    sides."""
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    g1 = sum(np.linalg.norm(data.xpos[model.body(b).id] - data.xpos[model.body(a).id]) for a, b in g1_pairs)
    return g1 / (sum(skeleton['bones'][b]['length'] for b in bones) * skeleton['length_scale'])


def _rotation_error(current, target):
    """The world rotation vector e with current = exp(e) · target (3x3 matrices)."""
    q, t, error = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q, current.ravel())
    mujoco.mju_mat2Quat(t, target.ravel())
    mujoco.mju_subQuat(error, q, t)  # in target's frame
    return target @ error


def _skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def retarget_frame(model, data, qpos, targets, free, limits, home, joints, iterations, robot=None, floor=None,
                   previous=None, home_weights=None):
    """Gauss-Newton with damping: qpos (nq) moved (root rotation and the `free` joint ids) toward the frame's targets,
    clipped to limits (low, high of the free joints); home (len(joints)) the pull's pose of the joint ids `joints`
    (home_weights its weights per rad, None: HOME_PULL, the ankle roll ANKLE_ROLL_PULL); previous (len(joints)) the
    frame before's joints, pulled toward with PREVIOUS_PULL (None: no pull).
    targets: segments [(a, b, unit direction, weight)], reaches [(a, b, offset (m), weight)], oriented [(body, rotation,
    weight)], axes [(body, local axis, unit direction, weight)], in TARGETS' order. With robot (bool per geom), the
    contacts between two robot geoms closer than SELF_GAP (MuJoCo's pairs; the model's robot geoms carry that margin)
    push apart. With floor (dict: geoms, the robot's floor geoms; contacts [(geom ids, weight 0-1)]; root (xy, z) or
    None; optional flat [geom ids], pins [(body, xy, weight)], support [geom ids]) the root moves too: each contact's
    lowest point goes to z = 0, no geom below it, the root toward root, every flat geom's lowest point to z = 0, each
    pinned body's origin to its xy, the centre of mass COM_MARGIN inside the support's hull (support_term). Returns
    (qpos, errors: TARGETS' angles (rad) of segments, oriented and axes, distances (m) of reaches; with floor also the
    largest |height| of a contact weighted 0.5 or more, the deepest point below the floor and the centre of mass's
    shortfall of COM_MARGIN, m)."""
    nv = model.nv
    columns = np.zeros(nv, bool)
    columns[3:6] = True
    if floor is not None:
        columns[:3] = True
    columns[np.asarray(model.jnt_dofadr)[free]] = True
    low, high = limits
    qadr = np.asarray(model.jnt_qposadr)[free]
    home_qadr, home_dof = np.asarray(model.jnt_qposadr)[joints], np.asarray(model.jnt_dofadr)[joints]
    weights = np.array([ANKLE_ROLL_PULL if model.joint(j).name.endswith('ankle_roll_joint') else HOME_PULL
                        for j in joints]) if home_weights is None else np.asarray(home_weights, float)
    select = np.zeros((len(joints), nv))
    select[np.arange(len(joints)), home_dof] = 1.
    pull = np.zeros((len(joints), nv))
    pull[np.arange(len(joints)), home_dof] = weights
    jacp, jacr, jacp2 = np.zeros((3, nv)), np.zeros((3, nv)), np.zeros((3, nv))
    qpos = qpos.copy()
    for it in range(iterations + 1):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rows, residual, errors = [], [], []
        for a, b, unit, weight in targets['segments']:
            delta = data.xpos[b] - data.xpos[a]
            length = np.linalg.norm(delta)
            mujoco.mj_jacBody(model, data, jacp, None, b)
            mujoco.mj_jacBody(model, data, jacp2, None, a)
            residual.append(weight * (delta / length - unit))
            rows.append(weight * (jacp - jacp2) / length)
            errors.append(float(np.arccos(np.clip(delta @ unit / length, -1, 1))))
        for a, b, offset, weight in targets['reaches']:
            delta = data.xpos[b] - data.xpos[a]
            mujoco.mj_jacBody(model, data, jacp, None, b)
            mujoco.mj_jacBody(model, data, jacp2, None, a)
            residual.append(weight * (delta - offset))
            rows.append(weight * (jacp - jacp2))
            errors.append(float(np.linalg.norm(delta - offset)))
        for body, rotation, weight in targets['oriented']:
            error = _rotation_error(data.xmat[body].reshape(3, 3), rotation)
            mujoco.mj_jacBody(model, data, None, jacr, body)
            residual.append(weight * error)
            rows.append(weight * jacr.copy())
            errors.append(float(np.linalg.norm(error)))
        for body, axis, unit, weight in targets['axes']:
            turned = data.xmat[body].reshape(3, 3) @ axis
            mujoco.mj_jacBody(model, data, None, jacr, body)
            residual.append(weight * (turned - unit))
            rows.append(weight * -_skew(turned) @ jacr)
            errors.append(float(np.arccos(np.clip(turned @ unit, -1, 1))))
        if floor is not None:
            lowest = {g: geom_lowest(model, data, g) for g in floor['geoms']}
            height = 0.
            for geoms, weight in floor['contacts']:
                g = min(geoms, key=lambda g: lowest[g][2])
                mujoco.mj_jac(model, data, jacp, None, lowest[g], model.geom_bodyid[g])
                residual.append(np.array([CONTACT_WEIGHT * weight * lowest[g][2]]))
                rows.append(CONTACT_WEIGHT * weight * jacp[2:3].copy())
                if weight >= .5:
                    height = max(height, abs(float(lowest[g][2])))
            for g, point in lowest.items():
                if point[2] < 0:
                    mujoco.mj_jac(model, data, jacp, None, point, model.geom_bodyid[g])
                    residual.append(np.array([FLOOR_WEIGHT * point[2]]))
                    rows.append(FLOOR_WEIGHT * jacp[2:3].copy())
            if floor.get('root') is not None:
                xy, z = floor['root']
                root = np.zeros((3, nv))
                root[:, :3] = np.diag([ROOT_WEIGHT, ROOT_WEIGHT, ROOT_HEIGHT_WEIGHT])
                residual.append(root[:, :3].diagonal() * (qpos[:3] - np.array([*xy, z])))
                rows.append(root)
            flat_weight = floor.get('flat_weight', FLAT_WEIGHT)
            for g in floor.get('flat', ()):  # a flat foot: every sole sphere on the floor
                point = lowest[g] if g in lowest else geom_lowest(model, data, g)
                mujoco.mj_jac(model, data, jacp, None, point, model.geom_bodyid[g])
                residual.append(np.array([flat_weight * point[2]]))
                rows.append(flat_weight * jacp[2:3].copy())
            for b, xy, weight in floor.get('pins', ()):
                mujoco.mj_jacBody(model, data, jacp, None, b)
                residual.append(weight * (data.xpos[b, :2] - xy))
                rows.append(weight * jacp[:2].copy())
            shortfall = 0.
            if floor.get('support'):
                term = support_term(model, data, floor['support'], floor.get('com_body', 1))
                com_weight = floor.get('com_weight', COM_WEIGHT)
                if term is not None and term[0] < COM_MARGIN:
                    shortfall = COM_MARGIN - term[0]
                    residual.append(np.array([com_weight * shortfall]))
                    rows.append(-com_weight * term[1][None])
            if floor.get('transfer'):  # the weight onto the support that is to carry it next, as far as the goal
                term = support_term(model, data, floor['transfer'], floor.get('com_body', 1))
                if term is not None:
                    goal = min(COM_MARGIN, floor.get('transfer_goal', COM_MARGIN))
                    if term[0] < goal:
                        residual.append(np.array([TRANSFER_WEIGHT * (goal - term[0])]))
                        rows.append(-TRANSFER_WEIGHT * term[1][None])
            errors += [height, -min(0., min(float(p[2]) for p in lowest.values())), shortfall]
        if it == iterations:
            break
        if robot is not None:
            mujoco.mj_collision(model, data)
            for k in range(data.ncon):
                contact = data.contact[k]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if not (robot[g1] and robot[g2]) or contact.dist >= SELF_GAP:
                    continue
                normal = contact.frame[:3]  # from geom1 to geom2
                mujoco.mj_jac(model, data, jacp, None, contact.pos, model.geom_bodyid[g2])
                mujoco.mj_jac(model, data, jacp2, None, contact.pos, model.geom_bodyid[g1])
                residual.append(np.array([SELF_WEIGHT * (contact.dist - SELF_GAP)]))
                rows.append(SELF_WEIGHT * (normal @ (jacp - jacp2))[None])
        residual.append(weights * (qpos[home_qadr] - home))
        rows.append(pull)
        if previous is not None:  # continuity: toward the frame before
            residual.append(PREVIOUS_PULL * (qpos[home_qadr] - previous))
            rows.append(PREVIOUS_PULL * select)
        # a joint near its limit is held off it within the solve (the clip after a step alone hides the limit from
        # the other joints, which then never take up what it cannot give)
        near = np.maximum(0., qpos[qadr] - (high - LIMIT_ZONE)) - np.maximum(0., (low + LIMIT_ZONE) - qpos[qadr])
        if LIMIT_WEIGHT and near.any() and previous is not None:  # not on a take's first frame, solved from home
            rows_limit = np.zeros((len(qadr), nv))
            rows_limit[np.arange(len(qadr)), np.asarray(model.jnt_dofadr)[free]] = LIMIT_WEIGHT
            residual.append(LIMIT_WEIGHT * near[near != 0])
            rows.append(rows_limit[near != 0])
        jac = np.concatenate(rows)[:, columns]
        r = np.concatenate(residual)
        step = np.zeros(nv)
        step[columns] = -np.linalg.solve(jac.T @ jac + DAMPING * np.eye(columns.sum()), jac.T @ r)
        largest = np.abs(step).max()
        if largest > MAX_STEP:  # the one-sided floor and self terms switch on and off: no jumps
            step *= MAX_STEP / largest
        mujoco.mj_integratePos(model, qpos, step, 1.)
        qpos[qadr] = np.clip(qpos[qadr], low, high)
    return qpos, np.array(errors)


def floor_geoms(model, robot_geoms, floor):
    """The robot geoms that collide with the floor (contype / conaffinity), ids."""
    return [g for g in range(model.ngeom) if robot_geoms[g] and (
        model.geom_contype[g] & model.geom_conaffinity[floor] or model.geom_conaffinity[g] & model.geom_contype[floor])]


def geom_lowest(model, data, g):
    """The lowest world point (3,) of geom g in the current kinematics (mesh: a vertex; sphere, capsule, cylinder,
    box: their shape; an ellipsoid's bounding box below)."""
    kind, size, centre = model.geom_type[g], model.geom_size[g], data.geom_xpos[g]
    rotation = data.geom_xmat[g].reshape(3, 3)
    up = rotation[2]  # world z in the geom's axes
    down = np.array([0., 0., 1.])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        return centre + rotation @ vertices[np.argmin(vertices @ up)]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return centre - size[0] * down
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return centre - math.copysign(size[1], up[2]) * rotation[:, 2] - size[0] * down
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:  # the lower end disc's rim, toward the floor (its centre when upright)
        axis = rotation[:, 2]
        across = down - axis[2] * axis
        rim = across / np.linalg.norm(across) * size[0] if np.linalg.norm(across) > 1e-9 else np.zeros(3)
        return centre - math.copysign(size[1], axis[2]) * axis - rim
    if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return centre - rotation @ (np.sign(up) * size)
    raise ValueError(f'geom {model.geom(g).name}: type {kind} not measured')


def support_points(model, data, g, touch=SUPPORT_TOUCH):
    """The world points (n, 3) of geom g within touch of the floor (z = 0) in the current kinematics: a mesh's
    vertices, a sphere's bottom, a capsule's two end spheres' bottoms, another shape's lowest point."""
    kind, size, centre = model.geom_type[g], model.geom_size[g], data.geom_xpos[g]
    rotation = data.geom_xmat[g].reshape(3, 3)
    down = np.array([0., 0., 1.])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        points = centre + vertices @ rotation.T
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        points = (centre - size[0] * down)[None]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        points = centre + np.outer([-1., 1.], rotation[:, 2]) * size[1] - size[0] * down
    else:
        points = geom_lowest(model, data, g)[None]
    return points[points[:, 2] < touch]


def hull_indices(points):
    """The indices of 2-D points (n, 2) on their convex hull, counter-clockwise (Andrew's monotone chain)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))

    def cross(o, a, b):
        return ((points[a][0] - points[o][0]) * (points[b][1] - points[o][1])
                - (points[a][1] - points[o][1]) * (points[b][0] - points[o][0]))
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 1e-12:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 1e-12:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def support_term(model, data, geoms, body=1, gradient=True):
    """(margin, d margin / d qvel (nv,) or None) of the centre of mass of body's subtree (1: the robot) over the convex
    hull of geoms' support_points: its signed distance (m, + inside) to the hull's nearest edge, and how it changes
    with the coordinates: along that edge's inward normal (outside: toward the nearest point), the centre of mass's
    motion less the nearest point's own (it moves with its bodies). None with fewer than 3 hull points (no area)."""
    points, owners = [], []
    for g in geoms:
        for point in support_points(model, data, g):
            points.append(point)
            owners.append(int(model.geom_bodyid[g]))
    if len(points) < 3:
        return None
    points = np.array(points)
    index = hull_indices(points[:, :2])
    if len(index) < 3:
        return None
    com = data.subtree_com[body][:2]
    inside, best = True, None
    for i in range(len(index)):
        a, b = index[i], index[(i + 1) % len(index)]
        edge = points[b, :2] - points[a, :2]
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        if edge[0] * (com[1] - points[a, 1]) - edge[1] * (com[0] - points[a, 0]) < 0:
            inside = False
        t = float(np.clip(np.dot(com - points[a, :2], edge) / length ** 2, 0., 1.))
        nearest = points[a, :2] + t * edge
        distance = float(np.linalg.norm(com - nearest))
        if best is None or distance < best[0]:
            best = (distance, a, b, t, nearest, np.array([-edge[1], edge[0]]) / length)
    distance, a, b, t, nearest, normal = best
    margin = distance if inside else -distance
    if not gradient:
        return margin, None
    if inside or distance < 1e-9:
        direction = normal
    else:
        direction = (nearest - com) / distance
    nv = model.nv
    jac_com, jac_a, jac_b = np.zeros((3, nv)), np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, body)
    mujoco.mj_jac(model, data, jac_a, None, points[a], owners[a])
    mujoco.mj_jac(model, data, jac_b, None, points[b], owners[b])
    moved = jac_com[:2] - ((1 - t) * jac_a[:2] + t * jac_b[:2])
    return margin, direction @ moved


def lowest_point(model, data, qpos, geoms):
    """The lowest point of the geoms (floor_geoms) at qpos, its height above the floor (z = 0), m."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return min(float(geom_lowest(model, data, g)[2]) for g in geoms)


def _floor_model(env):
    """env's model, a data of its own and the floor-touching robot geoms."""
    model = env.model
    floor = model.geom('floor').id
    if abs(model.geom_pos[floor, 2]) > 1e-12 or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError('the floor is not the plane z = 0')
    return model, mujoco.MjData(model), floor_geoms(model, env.robot_geoms, floor)


def home_qpos(env):
    qpos = np.zeros(env.model.nq)
    qpos[3] = 1.
    qpos[np.asarray(env.qadr)] = env.home
    return qpos


def spine_bend(skeleton, frame):
    """The thorax's rotation relative to the root in an AMC frame (3x3): the spine's bend the torso target leaves out."""
    names, _, _, rotations = points(skeleton, [frame])
    index = {n: i for i, n in enumerate(names)}
    return rotations[0, index['root']].T @ rotations[0, index['thorax']]


def _clipped(vector, largest):
    """vector scaled down to the norm largest if longer."""
    norm = float(np.linalg.norm(vector))
    return vector if norm <= largest else vector * (largest / norm)


def _checker(env):
    """clip_feasibility.FrameChecker of env (scripts/lab/probes)."""
    import sys
    from pathlib import Path
    probes = str(Path(__file__).resolve().parent / 'probes')
    if probes not in sys.path:
        sys.path.insert(0, probes)
    import clip_feasibility
    return clip_feasibility.FrameChecker(env, clip_feasibility.TOUCH)


def spread(values, frames):
    """values (F, …) ≥ 0 spread over ±frames: a running maximum, then a running mean (the ends clamped)."""
    k = len(values)
    index = np.clip(np.arange(k)[:, None] + np.arange(-frames, frames + 1)[None], 0, k - 1)
    return values[index].max(1)[index].mean(1)


def torque_moves(env, checker, qpos, free_index):
    """(len(free), 2) how far each free joint's (low, high) IK limit must move off its stop so that the PD can give
    TORQUE_MARGIN times the pose's static torque (checker.frame without velocities): for a joint whose torque points at
    a stop it sits near, kp·(stop − q) ≥ TORQUE_MARGIN·|τ| beyond the soft limit; 0 elsewhere."""
    from humanoid_sim.tasks import g1_tensor_math as task
    result = checker.frame(qpos)
    torque = result['torque']
    low, high = env.model.jnt_range[env.joints].T
    soft_low, soft_high = task.soft_limits(low, high)
    moves = np.zeros((len(free_index), 2))
    q = np.asarray(qpos)[np.asarray(env.qadr)]
    for i, j in enumerate(free_index):
        need = TORQUE_MARGIN * abs(torque[j]) / env.kp[j]
        if torque[j] > 0 and q[j] > env.target_high[j] - need:  # the PD cannot give it from here
            moves[i, 1] = max(0., soft_high[j] - (env.target_high[j] - need))
        elif torque[j] < 0 and q[j] < env.target_low[j] + need:
            moves[i, 0] = max(0., (env.target_low[j] + need) - soft_low[j])
    return moves


def person_contacts(contact_sets, starts, ends, index, ground, rate):
    """[{contact name: weight 0-1}] per frame of the person's floor contacts: CONTACTS' height rule, and for PLANTED
    ones the stillness of their points' mean (horizontal speed by central differences at rate frames/s, averaged over
    ±STILL_FRAMES, full weight below STILL_SPEED[0], none above STILL_SPEED[1])."""
    count = len(starts)
    slow, fast = STILL_SPEED or (math.inf, math.inf)
    out = [{} for _ in range(count)]
    for name, _, human, full, none in contact_sets:
        heights = np.min([(starts if end == 'start' else ends)[:, index[bone], 2] for bone, end in human], 0) - ground
        weight = np.clip((none - heights) / (none - full), 0., 1.)
        if name in PLANTED and count > 2 and STILL_SPEED:
            centre = np.mean([(starts if end == 'start' else ends)[:, index[bone], :2] for bone, end in human], 0)
            speed = np.zeros(count)
            speed[1:-1] = np.linalg.norm(centre[2:] - centre[:-2], axis=1) * rate / 2
            speed[0], speed[-1] = speed[1], speed[-2]
            near = np.clip(np.arange(count)[:, None] + np.arange(-STILL_FRAMES, STILL_FRAMES + 1)[None], 0, count - 1)
            weight = weight * np.clip((fast - speed[near].mean(1)) / (fast - slow), 0., 1.)
        for k in range(count):
            out[k][name] = float(weight[k])
    return out


def smooth(env, qpos, rate, seconds=SMOOTH_SECONDS):
    """qpos (F, 36) at rate frames/s smoothed by a Gaussian of σ seconds (the ends clamped): the root position and the
    joints, the root quaternion as the normalized mean of its sign-aligned neighbours; then each frame raised out of
    the floor (its lowest robot geom at z ≥ 0)."""
    sigma = seconds * rate
    if sigma <= 0:
        return qpos
    half = int(math.ceil(3 * sigma))
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    count = len(qpos)
    index = np.clip(np.arange(count)[:, None] + offsets[None], 0, count - 1)
    out = np.einsum('fkc,k->fc', qpos[index], weights)
    quaternions = qpos[index, 3:7]
    signs = np.sign(np.einsum('fkc,fc->fk', quaternions, qpos[:, 3:7]))
    signs[signs == 0] = 1.
    mean = np.einsum('fkc,k->fc', quaternions * signs[..., None], weights)
    out[:, 3:7] = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    model, data, geoms = _floor_model(env)
    for row in out:
        row[2] -= min(0., lowest_point(model, data, row, geoms))
    return out


def project(env, qpos, rate, report=None):
    """qpos (F, 36) at rate frames/s (retarget's, smoothed) made statically doable, frame by frame from the one before,
    as settle does: each joint pulled toward the source frame (PROJECT_PULL per rad; the ankles PROJECT_ANKLE_PULL, so
    they take up the feet's flatness), the pelvis toward its orientation; the source frame's touching parts (PARTS: a
    foot, a knee, a hand, within PROJECT_TOUCH of the floor) on the floor, pinned where they came down; a foot that is
    within FLAT_TILT of level in the source flat on all four sole spheres, else on both toe spheres; the centre of mass
    COM_MARGIN inside the support (every touching geom, SUPPORT_TOUCH). A part the source lifts while the rest would
    not carry the centre of mass RELEASE_MARGIN inside it waits: the source frame is held while the centre of mass moves
    over the rest at TRANSFER_SPEED (at most PAUSE_SECONDS_PROJECT); no joint faster than JOINT_SPEED, the root than
    ROOT_SPEED / ROOT_TURN. Returns the frames (more than F with waits); report gets 'pauses' [(source frame, frames
    waited)], 'gave_up' [source frames] and 'short' (the largest shortfall of COM_MARGIN, m)."""
    import copy
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data, geoms = _floor_model(env)
    robot = np.asarray(env.robot_geoms, bool)
    ik_model = copy.deepcopy(model)
    ik_model.geom_margin[robot] = SELF_GAP
    ik_data = mujoco.MjData(ik_model)
    joints = np.asarray(env.joints)
    free = joints[~env.wrist_mask.numpy()]
    limits = task.soft_limits(model.jnt_range[free, 0], model.jnt_range[free, 1])
    qadr, free_qadr = np.asarray(env.qadr), np.asarray(model.jnt_qposadr)[free]
    pull = np.array([PROJECT_ANKLE_PULL if 'ankle' in model.joint(j).name else PROJECT_PULL for j in joints])
    pelvis = model.body('pelvis').id
    named = {name: [g for g in geoms if model.body(model.geom_bodyid[g]).name in bodies] for name, bodies in PARTS}
    grouped = {g for gs in named.values() for g in gs}
    parts = dict(named)  # every other floor-touching body is a part of its own (thighs, the trunk lying), not pinned
    for g in geoms:
        if g not in grouped:
            parts.setdefault(model.body(model.geom_bodyid[g]).name, []).append(g)
    anchor_body = {name: model.body(bodies[0]).id for name, bodies in PARTS}
    soles = {'left_foot': [int(g) for g in env.soles[0]], 'right_foot': [int(g) for g in env.soles[1]]}
    toes = {foot: [g for g in gs if model.geom_pos[g, 0] > 0] for foot, gs in soles.items()}
    step, joint_step = TRANSFER_SPEED / rate, JOINT_SPEED / rate
    pause_frames = int(round(PAUSE_SECONDS_PROJECT * rate))

    def touching(row):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        lowest = {g: float(geom_lowest(model, data, g)[2]) for g in geoms}
        floor_z = min(0., min(lowest.values()))
        down = {name for name, gs in parts.items() if min(lowest[g] for g in gs) - floor_z < PROJECT_TOUCH}
        tilt = {foot: float(data.xmat[anchor_body[foot]].reshape(3, 3)[2, 2]) for foot in soles}
        return down, tilt

    out, pauses, gave_up, short = [], [], [], 0.
    row = qpos[0].copy()
    anchors, held_down, carried = {}, set(), {}
    k, paused, goal = 0, 0, None
    while k < len(qpos):
        down, tilt = touching(qpos[k])
        lifting = [n for n in held_down - down if not carried.get(n, True)] if out else []
        if lifting and paused < pause_frames:
            source, paused = k - 1, paused + 1
            down, tilt = touching(qpos[source])
            transfer = [g for g in geoms if not any(g in parts[n] for n in lifting)]
            if paused == 1:
                ik_data.qpos[:] = row
                mujoco.mj_kinematics(ik_model, ik_data)
                mujoco.mj_comPos(ik_model, ik_data)
                term = support_term(ik_model, ik_data, transfer, gradient=False)
                start = -1. if term is None else term[0]
            goal = start + step * paused
        else:
            if lifting:
                gave_up.append(k)
            if paused:
                pauses.append((k, paused))
            source, paused, transfer, k = k, 0, None, k + 1
        src = qpos[source]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, src[3:7])
        targets = dict(segments=[], reaches=[], oriented=[(pelvis, rotation.reshape(3, 3), 2.)], axes=[])
        flat = []
        for foot in soles:
            if foot in down and tilt[foot] >= TOES_UP:  # sole toward the floor (not a foot lying on its top)
                flat += soles[foot] if tilt[foot] >= math.cos(FLAT_TILT) else toes[foot]
        pins = [(anchor_body[n], anchors[n], PROJECT_PIN) for n in down if n in anchors and n in anchor_body]
        floor = dict(geoms=geoms, contacts=[(parts[n], 1.) for n in down], root=(src[:2], src[2]), flat=flat,
                     flat_weight=PROJECT_FLAT, pins=pins, support=geoms, com_weight=PROJECT_COM, transfer=transfer,
                     transfer_goal=goal if transfer else None)
        before = row.copy()
        if out:
            now = row[free_qadr]
            frame_limits = (np.minimum(np.maximum(limits[0], now - joint_step), now),
                            np.maximum(np.minimum(limits[1], now + joint_step), now))
        else:
            frame_limits = limits
        row, error = retarget_frame(ik_model, ik_data, row if out else src.copy(), targets, free, frame_limits,
                                    src[qadr], joints, FIRST_ITERATIONS if not out else ITERATIONS, robot, floor,
                                    None if not out else before[qadr], pull)
        if out:
            row[:3] = before[:3] + _clipped(row[:3] - before[:3], ROOT_SPEED / rate)
            turn = np.zeros(3)
            mujoco.mju_subQuat(turn, row[3:7], before[3:7])
            row[3:7] = before[3:7]
            mujoco.mju_quatIntegrate(row[3:7], _clipped(turn, ROOT_TURN / rate), 1.)
        row[2] -= min(0., lowest_point(model, data, row, geoms))
        short = max(short, float(error[-1]))
        ik_data.qpos[:] = row
        mujoco.mj_kinematics(ik_model, ik_data)
        mujoco.mj_comPos(ik_model, ik_data)
        for n in anchor_body:
            if n in down and n not in anchors:
                anchors[n] = ik_data.xpos[anchor_body[n], :2].copy()
            elif n not in down:
                anchors.pop(n, None)
        held_down = set(down)
        for n in down:  # would the rest carry the centre of mass without it?
            term = support_term(ik_model, ik_data, [g for g in geoms if g not in parts[n]], gradient=False)
            carried[n] = term is not None and term[0] >= RELEASE_MARGIN
        out.append(row.copy())
    if report is not None:
        report.update(pauses=[[int(a), int(b)] for a, b in pauses], gave_up=[int(g) for g in gave_up], short=short)
    return np.array(out)


def retarget(env, skeleton, frames, neutral=None, rounds=ROUNDS, report=None, rate=SOURCE_RATE):
    """(qpos (F, 36), errors (F, len(ERRORS)) as retarget_frame's, {'legs', 'arms'} length ratios) of AMC frames
    (read_amc's) on env, a G1VelocityEnv of the getting-up model; neutral the AMC frame whose spine counts as straight
    (None: the first of frames). Each frame starts from the one before (the first from home turned as the person's
    root). The joints stay within the task's soft limits (g1_tensor_math.soft_limits: its joint_limit term charges
    beyond them) and the robot's geoms SELF_GAP apart; the support, flat feet, pins, weight transfers, continuity and
    the torque moves of the limits as the comment above RETARGET_VERSION says, over `rounds` solves of the take (the
    last one's frames returned: with transfers more than len(frames)). report: a dict that gets per round the frames
    and joints whose limits moved, the pauses ((person frame, frames waited)) and the transfers given up, and of the
    last round per frame its person frame ('sources') and a trace. rate: frames/s of frames (the speed limits, the
    stillness and the waits count in it)."""
    import copy
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data, geoms = _floor_model(env)
    robot = np.asarray(env.robot_geoms, bool)
    ik_model = copy.deepcopy(model)
    ik_model.geom_margin[robot] = SELF_GAP
    ik_data = mujoco.MjData(ik_model)
    names, starts, ends, rotations = points(skeleton, frames)
    index = {n: i for i, n in enumerate(names)}
    ground = float(np.percentile(np.minimum(starts[..., 2], ends[..., 2]).min(1), FLOOR_PERCENTILE))
    contact_sets = [(name, [g for g in geoms if model.body(model.geom_bodyid[g]).name in bodies], human, full, none)
                    for name, bodies, human, full, none in CONTACTS]
    empty = [c[0] for c in contact_sets if not c[1]]
    if empty:
        raise ValueError(f'contacts without floor geoms: {empty}')
    bend = spine_bend(skeleton, frames[0] if neutral is None else neutral)
    rotations[:, index['thorax']] = rotations[:, index['thorax']] @ bend.T
    rest = dict(zip(names, points(skeleton, [(0, {'root': [0.] * 6})])[1][0]))  # joints of the rest pose
    joints = np.asarray(env.joints)
    free = joints[~env.wrist_mask.numpy()]
    limits = task.soft_limits(model.jnt_range[free, 0], model.jnt_range[free, 1])
    start = home_qpos(env)
    legs = length_ratio(model, data, start, skeleton, G1_LEGS, LEG_BONES)
    arms = length_ratio(model, data, start, skeleton, G1_ARMS, ARM_BONES)
    body = lambda name: model.body(name).id
    axes = []
    for _, name, bone_a, bone_b, weight in AXES:
        direction = rest[bone_b] - rest[bone_a]
        axes.append((body(name), direction / np.linalg.norm(direction), bone_a, bone_b, weight))
    named = {c[0]: c for c in contact_sets}
    pin_body = {name: body(bodies[0]) for name, bodies, *_ in CONTACTS}
    soles = {'left_foot': [int(g) for g in env.soles[0]], 'right_foot': [int(g) for g in env.soles[1]]}
    foot_bone = {'left_foot': 'lfoot', 'right_foot': 'rfoot'}
    toes = {foot: [g for g in geoms_ if model.geom_pos[g, 0] > 0] for foot, geoms_ in soles.items()}  # the front two
    free_index = np.flatnonzero(~env.wrist_mask.numpy())
    home_qadr = np.asarray(model.jnt_qposadr)[joints]
    checker = _checker(env) if rounds > 1 else None
    count = len(frames)
    moved = np.zeros((count, len(free), 2))  # how far each free joint's (low, high) limit is off its soft limit
    without = {name: [g for g in geoms if g not in named[name][1]] for name in PLANTED}  # the rest, without one
    knee_of = {'left_foot': 'left_knee', 'right_foot': 'right_knee'}
    segment_names = [s[0] for s in SEGMENTS]
    leg_segments = {f'{side}_foot': [segment_names.index(f'{side}_thigh'), segment_names.index(f'{side}_shank')]
                    for side in ('left', 'right')}
    person = person_contacts(contact_sets, starts, ends, index, ground, rate)
    pause_frames = int(round(PAUSE_SECONDS * rate))
    free_qadr = np.asarray(model.jnt_qposadr)[free]
    joint_step = (JOINT_SPEED or 0.) / rate
    step = TRANSFER_SPEED / rate
    for round_ in range(max(rounds, 1)):
        low = np.minimum(limits[0] + moved[:, :, 0], limits[1] - .02)
        high = np.maximum(limits[1] - moved[:, :, 1], low + .02)
        out, errors, sources, found, trace = [], [], [], np.zeros_like(moved), []
        pauses, gave_up = [], []
        qpos, previous = start.copy(), None
        active, anchors, carried = {}, {}, {name: True for name in PLANTED}
        k, paused = 0, 0
        while k < count:
            # the person lifts a hand, a knee or a foot the rest does not carry yet: their clock waits on the frame
            # before while the robot moves its centre of mass over the rest (at most PAUSE_SECONDS)
            lifting = [n for n in PLANTED if k > 0 and active.get(n) and person[k][n] < .5 and not carried[n]]
            if lifting and paused < pause_frames:
                frame, paused = k - 1, paused + 1
                transfer = [g for g in geoms if not any(g in named[n][1] for n in lifting)]
                if paused == 1:  # the centre of mass's margin over the rest when the wait began
                    term = support_term(ik_model, ik_data, transfer, gradient=False)
                    transfer_start = -1. if term is None else term[0]
                transfer_goal = transfer_start + step * paused
            else:
                if lifting:
                    gave_up.append(k)
                if paused:
                    pauses.append((k, paused))
                frame, paused, transfer, k = k, 0, None, k + 1
            joint = lambda bone: starts[frame, index[bone]]
            targets = dict(segments=[], reaches=[], oriented=[], axes=[])
            for _, a, b, bone_a, bone_b, weight in SEGMENTS:
                delta = joint(bone_b) - joint(bone_a)
                targets['segments'].append((body(a), body(b), delta / np.linalg.norm(delta), weight))
            for _, a, b, bone_a, bone_b, weight in REACHES:
                targets['reaches'].append((body(a), body(b), arms * (joint(bone_b) - joint(bone_a)), weight))
            for _, name, bone, weight in ORIENTED:
                targets['oriented'].append((body(name), rotations[frame, index[bone]], weight))
            for name, axis, bone_a, bone_b, weight in axes:
                delta = joint(bone_b) - joint(bone_a)
                targets['axes'].append((name, axis, delta / np.linalg.norm(delta), weight))
            if transfer:  # while the weight moves, the person's frozen pose gives way to the support
                for key in targets:
                    targets[key] = [(*t[:-1], t[-1] * PAUSE_TARGETS) for t in targets[key]]
            weights = person[frame]
            # a foot down stands flat where the person's does, and also where the person squats on the toes with
            # that knee up: the G1 holds a deep squat flat-footed (pelvis 0.25-0.3 m, knee 2.6, ankle -0.4…-0.75, the
            # centre of mass inside the soles), not on its toes (ankle kp 28); that leg then takes its own bend
            sole_up = {foot: float((rotations[frame, index[foot_bone[foot]]] @ [0., 0., 1.])[2]) for foot in FEET}
            flat = {foot: weights[foot] >= .5 and (sole_up[foot] >= math.cos(FLAT_TILT) or (
                SQUAT_FLAT and sole_up[foot] >= TOES_UP and weights[knee_of[foot]] < .5)) for foot in FEET}
            for foot in FEET:
                if flat[foot] and sole_up[foot] < math.cos(FLAT_TILT):
                    for i in leg_segments[foot]:
                        targets['segments'][i] = (*targets['segments'][i][:3], targets['segments'][i][3] * SQUAT_LEG)
            on_toes = {foot: weights[foot] >= .5 and not flat[foot] and sole_up[foot] >= TOES_UP for foot in FEET}
            now = {name: weights[name] >= .5 for name in PLANTED}
            pinned = {name: now[name] and (name not in flat or flat[name]) for name in PLANTED}
            pins = [(pin_body[name], anchors[name], PIN_WEIGHT) for name in PLANTED
                    if pinned[name] and name in anchors]
            contacts = [(named[name][1], w) for name, w in weights.items() if w > 0]
            root = (joint('root')[:2] * legs, (joint('root')[2] - ground) * legs)
            if not out:
                mujoco.mju_mat2Quat(qpos[3:7], rotations[0, index['root']].ravel())
                qpos[:2], qpos[2] = root
            floor = dict(geoms=geoms, contacts=contacts, root=None if transfer else root, pins=pins, support=geoms,
                         transfer=transfer, transfer_goal=transfer_goal if transfer else None,
                         flat=[g for foot in FEET for g in (soles[foot] if flat[foot] else
                                                            toes[foot] if on_toes[foot] else ())])
            frame_low, frame_high = low[frame], high[frame]
            if out and JOINT_SPEED:  # no joint faster than JOINT_SPEED: a frame's solve never jumps IK branches
                now_free = qpos[free_qadr]
                frame_low = np.minimum(np.maximum(frame_low, now_free - joint_step), now_free)
                frame_high = np.maximum(np.minimum(frame_high, now_free + joint_step), now_free)
            before = qpos.copy()
            qpos, error = retarget_frame(ik_model, ik_data, qpos, targets, free, (frame_low, frame_high), env.home,
                                         joints, FIRST_ITERATIONS if not out else ITERATIONS, robot, floor, previous)
            if out and ROOT_SPEED:  # nor the root faster than ROOT_SPEED / ROOT_TURN
                qpos[:3] = before[:3] + _clipped(qpos[:3] - before[:3], ROOT_SPEED / rate)
                turn = np.zeros(3)
                mujoco.mju_subQuat(turn, qpos[3:7], before[3:7])
                qpos[3:7] = before[3:7]
                mujoco.mju_quatIntegrate(qpos[3:7], _clipped(turn, ROOT_TURN / rate), 1.)
            qpos[2] -= min(0., lowest_point(model, data, qpos, geoms))
            previous = qpos[home_qadr].copy()
            ik_data.qpos[:] = qpos
            mujoco.mj_kinematics(ik_model, ik_data)
            mujoco.mj_comPos(ik_model, ik_data)
            for name in PLANTED:
                if pinned[name] and name not in anchors:
                    anchors[name] = ik_data.xpos[pin_body[name], :2].copy()
                elif not pinned[name]:
                    anchors.pop(name, None)
            active = now
            for name in PLANTED:  # would the rest carry the centre of mass without it?
                term = support_term(ik_model, ik_data, without[name], gradient=False) if now[name] else None
                carried[name] = not now[name] or (term is not None and term[0] >= RELEASE_MARGIN)
            if checker is not None and round_ < rounds - 1:
                found[frame] = np.maximum(found[frame], torque_moves(env, checker, qpos, free_index))
            out.append(qpos.copy())
            errors.append(error)
            sources.append(frame)
            if report is not None:
                trace.append(dict(frame=frame, contacts=sorted(n for n, w in weights.items() if w >= .5),
                                  lifting=sorted(lifting) if transfer else [], flat=sorted(f for f in FEET if flat[f]),
                                  short=float(error[-1]), touch=float(error[-3]),
                                  transfer=None if not transfer else (support_term(
                                      ik_model, ik_data, transfer, gradient=False) or (None,))[0]))
        if report is not None:
            report['frames'] = trace
            report['sources'] = sources
            report.setdefault('rounds', []).append(dict(
                moved_frames=int((moved.max(2) > 1e-9).any(1).sum()),
                joints=sorted({model.joint(free[i]).name[:-6] for i in np.flatnonzero((moved > 1e-9).any((0, 2)))}),
                pauses=[[int(a), int(b)] for a, b in pauses], gave_up=[int(g) for g in gave_up],
                support_short=float(np.array(errors)[:, -1].max())))
        if round_ == rounds - 1 or not found.any():
            break
        moved = np.maximum(moved, spread(found, int(round(LIMIT_SECONDS * rate))))
    return np.array(out), np.array(errors), dict(legs=legs, arms=arms)


def resample(qpos, rate, dt):
    """qpos (F, 36) at rate frames/s every dt seconds: positions and joints linearly, the root quaternion (w x y z) by
    slerp."""
    from getup_style_library import slerp
    n = len(qpos)
    t = np.arange(n) / rate
    out = np.arange(0., t[-1] + 1e-9, dt)
    index = np.minimum((out * rate).astype(np.int64), n - 2)
    s = np.clip(out * rate - index, 0., 1.)
    linear = np.stack([np.interp(out, t, qpos[:, k]) for k in range(qpos.shape[1])], 1)
    linear[:, 3:7] = slerp(qpos[index, 3:7], qpos[index + 1, 3:7], s)
    return linear


def settle(env, qpos, seconds):
    """qpos (K, 36) with `seconds` more frames at env.dt that take its last pose to the home pose standing: the joints
    by a smoothstep, the pelvis upright with its heading kept (slerp), the root moved so that the feet's midpoint stays
    where it was, its height putting the lowest robot geom on the floor. 0 seconds: qpos as it is.
    Every settling frame is an IK solve (retarget_frame) with both feet flat (all eight sole spheres on the floor:
    the ankles' pitch and roll, pulled only lightly toward the smoothstep, take up what the legs and the pelvis leave),
    each foot pinned on a straight line from where it stood to its place in the home stance about that midpoint, the
    centre of mass COM_MARGIN inside the soles; the last frame is the home pose exactly."""
    count = int(round(seconds / env.dt))
    if count <= 0:
        return qpos
    import copy
    from getup_style_library import slerp
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data, geoms = _floor_model(env)
    robot = np.asarray(env.robot_geoms, bool)
    ik_model = copy.deepcopy(model)
    ik_model.geom_margin[robot] = SELF_GAP
    ik_data = mujoco.MjData(ik_model)
    last = qpos[-1].copy()
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, last[3:7])
    heading = math.atan2(rotation[3], rotation[0])
    upright = np.array([math.cos(heading / 2), 0., 0., math.sin(heading / 2)])
    feet_ids = list(env.feet)

    def feet(row):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        return data.xpos[feet_ids, :2].copy()
    begin = feet(last)
    anchor = begin.mean(0)
    home = home_qpos(env)
    home[3:7] = upright
    if not SETTLE_IK:  # the iteration-64 settle: no IK, the feet's midpoint kept, the feet as the joints put them
        qadr, rows = np.asarray(env.qadr), []
        for k in range(1, count + 1):
            s = k / count
            s = s * s * (3 - 2 * s)
            row = last.copy()
            row[qadr] = (1 - s) * last[qadr] + s * env.home
            row[3:7] = slerp(last[None, 3:7], upright[None], np.array([s]))[0]
            row[:2] += anchor - feet(row).mean(0)
            row[2] -= lowest_point(model, data, row, geoms)
            rows.append(row)
        return np.concatenate([qpos, np.array(rows)])
    end = feet(home)
    end += anchor - end.mean(0)
    joints = np.asarray(env.joints)
    free = joints[~env.wrist_mask.numpy()]
    limits = task.soft_limits(model.jnt_range[free, 0], model.jnt_range[free, 1])
    qadr = np.asarray(env.qadr)
    pull = np.array([SETTLE_ANKLE_PULL if 'ankle' in model.joint(j).name else SETTLE_PULL for j in joints])
    soles = [int(g) for g in np.asarray(env.soles).ravel()]
    pelvis = model.body('pelvis').id
    home[:2] += anchor - feet(home).mean(0)
    home[2] -= lowest_point(model, data, home, geoms)
    rows, row = [], last.copy()
    for k in range(1, count + 1):
        s = k / count
        s = s * s * (3 - 2 * s)
        pose = (1 - s) * last[qadr] + s * env.home
        turn = slerp(last[None, 3:7], upright[None], np.array([s]))[0]
        target = np.zeros(9)
        mujoco.mju_quat2Mat(target, turn)
        targets = dict(segments=[], reaches=[], oriented=[(pelvis, target.reshape(3, 3), 2.)], axes=[])
        place = (1 - s) * begin + s * end
        ramp = min(1., k / max(1., SETTLE_RAMP * count))  # the feet flatten and the weight moves in, no jump
        floor = dict(geoms=geoms, contacts=[(soles, 1.)], root=None, flat=soles, support=soles,
                     flat_weight=SETTLE_FLAT * ramp, com_weight=SETTLE_COM * ramp,
                     pins=[(f, place[i], SETTLE_PIN) for i, f in enumerate(feet_ids)])
        previous = row[qadr].copy()
        row, _ = retarget_frame(ik_model, ik_data, row, targets, free, limits, pose, joints,
                                FIRST_ITERATIONS if k == count else ITERATIONS, robot, floor, previous, pull)
        row[2] -= lowest_point(model, data, row, geoms)
        rows.append(row.copy())
    if np.abs(rows[-1][qadr] - env.home).max() < SETTLE_HOME and np.abs(rows[-1][:3] - home[:3]).max() < SETTLE_HOME:
        rows[-1] = home  # converged: the home pose exactly
    return np.concatenate([qpos, np.array(rows)])
