"""CPU only. getup_track_clip.py [--source fallAndGetUp1_subject1.csv | cmu | bones] [--subject 140 | <folder>]
[--take 140_09] [--file <bones csv>] [--fit v1|v2] [--frames a b] [--neutral N] [--settle 1.0] [--rounds 1] [--name <name>] [--force] [--out <file.npz>] [--data <folder>]
The clip of g1_getup's task_config track_clip (iteration 64; the critic's lab-critic/it63, select63.py / track63.py):
runs/humanoid_lab/getup_track/<name>.npz (tasks/g1_getup_track describes the arrays).

Source LAFAN1 (the default): LAFAN1's fallAndGetUp clips retargeted to the G1 29 DoF by Unitree
(.tools/datasets/lafan1_g1_unitree/g1, 30 frames/s; per frame the root position, its quaternion x y z w and the 29
joints in our order). The default is subject 1, fallAndGetUp1, frames 2104-2209 (3.5 s: back -> roll over -> knees ->
standing); the spare A′ is 2147-2209. The frames are resampled to the 50 Hz control step (joints and the root position
linearly, the root quaternion by slerp: getup_style_library.resample). LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0: a
research prototype, never published, and scripts/robot/package.py refuses a stage trained on it.

Source CMU (--source cmu): the CMU motion capture database, subject --subject N (default 140 «getting up from the
ground»; .tools/datasets/cmu_mocap_<N>: <N>.asf and <N>_<take>.amc, 120 frames/s; or a folder), take --take (subject
140: 140_08, 140_09 from the back, 01, 02 from the front, 03, 04 from a side; others: 111_06-08, 113_08, 139_16-18; a
bare 06 is <N>_06; another subject's take and frames have no default), AMC frames --frames a b (the file's numbers,
inclusive), retargeted by our
own IK in the getting-up model (cmu_mocap.retarget: the legs' and upper arms' directions, the hands relative to the
shoulders, the pelvis, the torso less the spine's bend in frame --neutral (default the take's first, lying flat) and the
feet's forward axes), at 120 Hz, then resampled to 50 Hz (cmu_mocap.resample). The subject ends every take in a half
crouch (knees ~65°, trunk ~40° forward), so --settle seconds (default 1) are added that take the last pose to the home
pose standing (cmu_mocap.settle: joints by a smoothstep, pelvis upright with its heading, the feet's midpoint kept).
The CMU licence (.tools/datasets/cmu_mocap_<N>/LICENSE.txt) allows commercial products but not reselling the data, even
converted: the clip stays under runs/ (untracked) and a policy trained on it may go to a robot, with the database's
acknowledgement in the package's README.

Since iteration 67 (cmu_mocap.RETARGET_VERSION 'cmu-ik-v2') the CMU IK pulls each joint toward the frame before and
caps the joints' and the root's speed, counts a foot, knee or hand down only while still, smooths the joints before the
50 Hz resampling, and settles by IK with both feet flat; its other mechanisms (support, flat feet, pins, weight
transfer, torque rounds --rounds, project) are off by default (cmu_mocap's comment says why).

Source BONES-SEED (--source bones --file <csv>): Bones Studio's motion capture, retargeted to the G1 29 DoF by NVIDIA
(.tools/datasets/bones_seed/g1_getup, untracked; --file a name there or a path). A row per frame at 120 frames/s
(the metadata's move_duration_frames, README «@ 120 fps»): Frame, root_translateX/Y/Z in cm (Z up, the pelvis origin),
root_rotateX/Y/Z in degrees, Euler angles applied about the fixed world axes X, then Y, then Z (R = Rz·Ry·Rx, Maya's
xyz; of the six orders only this one keeps the body on the floor through a whole rise), then the 29 joints in degrees,
named <joint>_dof, which are our joints in our order with our zeros (read_bones refuses another header; standing they
give level soles). Frames --frames a b are the Frame numbers, inclusive (default the whole file); names ending _M are
mirrored copies. The retarget's G1 has no hand capsule: a palm on the floor puts ours 5-14 cm into it, so each arm in
the floor is first raised by its own free joints (fit_floor; --fit v2, the default since iteration 71: bounded steps,
an arm that was on the floor in the data put on ours, the correction smoothed in time, header bones_fit, the name's
suffix -v2; --fit v1 rebuilds the earlier clips byte for byte, branch 71's among them); the retarget also leaves some takes
standing up to 2 cm above the floor, so each frame is then lowered by the smoothed height of its lowest robot geom
above it (drop_to_floor, never into it), and finally lifted out as for the other sources. --settle seconds (default 1) take the last pose to home as for CMU. The
licence (LICENSE.md beside the data): a Qualifying Startup (revenue under $1M) may train and sell models (robot
control policies included), not redistribute or publish the data or make a competing dataset; a model needs the
attribution BONES_ATTRIBUTION. The clip stays under runs/ (untracked), and package.py takes a stage trained on it
with that attribution in the package's README.

Patch (--patch <preset> | --base <clip> [--hold-hand SIDE FROM HELD RELEASED] [--feet-down FROM TO] [--slow FROM TO
FACTOR] [--waist-max RAD FROM TO] [--clear-self [GAP]], iteration 72): a made clip changed where its human motion does
not suit the robot, every time in the base clip's seconds (patch_clip); the frames no change touches keep the base's
arrays exactly and the header's patch records the options, what each change did and the unchanged frames. Preset
a493-kneel makes bones-…A493-749-1237-v3 from its v2 (PATCHES says what and why). --clear-self (iteration 79) moves the
arms' shoulders and elbows until every arm geom is GAP (1 cm) from the legs and the trunk, a hand on the floor kept on
it (clear_self; why at CLEAR_GAP), and names the frames where the gap is out of reach.
--base <clip> --from-frame K (iteration 85, the critic's lab-critic/it85 frames85.py): the clip from its frame K on,
every frame array (and the hold frame) bit for bit the base's (cut_clip), the header's patch {from_frame, base,
base_md5, base_patch}; the default name <base>-f<K>. A493 v3 puts the right hand into the torso over frames 0-18
(6.1 -> 1.2 cm), so a settling target (task_config settle_clip) is its frame 21, the first one without.

Both: the locked wrists set to home, each frame raised by the smoothed depth of its deepest floor contact in the
getting-up model (Unitree's foot and hand capsules included; LAFAN's data sinks 5-12 cm), and the kinematics of the 14
tracked bodies taken (g1_getup_track.build). The header keeps the source file's md5, the dataset (lafan1 / cmu /
bones_seed), which
package.py reads, and the clip's physical check (scripts/lab/probes/clip_feasibility.py: the verdict, its reasons and
numbers). An existing clip is not overwritten (stages name it by md5) without --force; --out writes elsewhere (trials).
Prints the clip, the lift, the check and the file's md5, which the setting names.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'probes'))
import clip_feasibility  # noqa: E402
from getup_style_library import DATA, SOURCE_RATE, resample  # noqa: E402

SOURCE, FRAMES = 'fallAndGetUp1_subject1.csv', (2104, 2209)
CMU_DATASETS = ROOT / '.tools' / 'datasets'  # subject N in cmu_mocap_N: N.asf and N_<take>.amc
CMU_SUBJECT = '140'
CMU_RATE = 120.
CMU_TAKE, CMU_FRAMES, CMU_SETTLE = '140_09', (61, 571), 1.
LAFAN_NOTE = ('LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0), retargeted to the G1 by Unitree; research prototype, not for '
              'publication or a robot package')
CMU_NOTE = ('CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), subject {subject}, retargeted to the G1 by the '
            'lab (scripts/lab/cmu_mocap.py); the licence allows commercial products but not reselling the data, even in '
            'converted form; acknowledgement: The data used in this project was obtained from mocap.cs.cmu.edu. The '
            'database was created with funding from NSF EIA-0196217.')
BONES_DIR = ROOT / '.tools' / 'datasets' / 'bones_seed' / 'g1_getup'
BONES_RATE = 120.
BONES_ROOT_ORDER = 'extrinsic xyz'  # R = Rz(z)·Ry(y)·Rx(x), the angles in degrees
# fit_floor's limbs: (the limb's top body, the joints it may move: the free ones, not the locked wrists)
LIMBS = tuple((f'{side}_shoulder_pitch_link', tuple(f'{side}_{j}_joint' for j in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll'))) for side in ('left', 'right')) + tuple(
    (f'{side}_hip_pitch_link', tuple(f'{side}_{j}_joint' for j in (
        'hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))) for side in ('left', 'right'))
LIMB_ITERATIONS, LIMB_DAMPING, LIMB_CLEARANCE = 30, 1e-4, 1e-3
# fit_floor's versions (--fit): v1 the first (iteration 70; branch 71's clip), v2 since iteration 71 (the critic's
# lab-critic/it71 fitbug71.py / fitfix71.py: v1's unbounded steps threw a joint to its soft limit wherever the
# Jacobian was small, 150-620 rad/s after the fit against at most 12 in the data, and a palm the human leant on rose
# 34-45 cm off the floor). v2: each step at most FIT_STEP rad per joint, damping FIT_DAMPING, FIT_ITERATIONS a solve; an
# arm whose lowest point was in or on the floor in the data is put on it (two-sided, LIMB_CLEARANCE within
# FIT_TOLERANCE), any other limb only out of it; FIT_ROUNDS solves with the correction (fit − data) smoothed over
# ±FIT_SMOOTH frames (120 Hz) between them, legs too, so the last solve leaves only what the smoothing undid.
BONES_FITS, BONES_FIT = ('v1', 'v2'), 'v2'
FIT_STEP, FIT_DAMPING, FIT_ITERATIONS, FIT_TOLERANCE, FIT_ROUNDS, FIT_SMOOTH = .05, 1e-3, 80, 5e-4, 4, 6
HAND_CEILING = 200.  # N on one hand (clip_feasibility's dynamic estimate): a warning above it, v2 records it
# --patch / --base (iteration 72, the critic's lab-critic/it72): a made clip changed in place of its source, the
# header's patch says how (patch_clip). A493 front v2 kneels at 2.0-2.6 s with both hands already off the floor (0 N),
# the centre of mass 1-10 cm outside the knees' support, left_hip_pitch asked for up to 92 of 69 N·m, and the waist pitch
# at its stop (+0.52) over 2.0-3.8 s, a retarget artefact; branch 72's tracker (exp-13efe7e29fb24979) passes the clip
# from 2.7 s (29 of 30) and never from 2.0-2.4 s (0 of 8): it props itself on its left hand and stays there.
PATCH_RAMP = .1  # s: the waist's and the feet's blends in and out of their spans
PATCH_SLOW_RAMP = .2  # s of output time at each end of a slowed span over which the clock eases from speed 1
PATCH_TOUCH = .002  # m: a hand whose lowest point is this close to the floor touches it (LIMB_CLEARANCE 1 mm, fit noise)
PATCH_WAIST_SOFT = .02  # rad: the width of the soft minimum under the waist's ceiling (at the stop 0.52: 0.4 − 2e-4)
PATCH_ITERATIONS, PATCH_SMOOTH, PATCH_ROUNDS = 80, 3, 4  # a solve's steps; corrections averaged ±3 frames between solves
PATCH_REACH = .01  # m: a held hand this far from its point is out of the arm's reach, and the hold ends there
# m: a lowered foot's lowest point (a sole sphere) right on the floor: MuJoCo's broadphase misses a 5 mm sphere of a
# zero-margin geom 2 mm above the plane even within the floor's margin, so clip_feasibility (touch 1 cm) would not
# count a foot left at LIMB_CLEARANCE once it puts the frame's lowest geom (a knee) on the floor
PATCH_FOOT_CLEARANCE = 0.
PATCH_KNEE_NEAR = .015  # m: over --feet-down a knee this close to the floor is kept on it
PATCH_HAND_NEAR = .01  # m: under --waist-max a hand this close to the floor keeps its lowest point where it was
# --clear-self (iteration 79, the critic's lab-critic/it79 clippen79.py / cost79.py / rsi79.py): fit v2 keeps no gap
# between the robot's parts, and A473 v2 from the back puts the right hand into hip_yaw / hip_roll deeper than 1 cm in 85
# of 238 frames over 0.66-2.34 s (up to 9.2 cm): following the clip costs −5.4 a step (self_contact −8.6) where lying
# earns +2.5, and RSI starts on frames 60-119 die within 5 steps in 32 of 45 (hand-hip contact 7.9-21.7 kN). The arms
# (CLEAR_JOINTS; the wrists' pitch and yaw are locked) are moved until every arm geom (Unitree's hand capsule included)
# is the gap away from every leg and trunk geom and the other arm, as cmu_mocap's SELF_GAP.
CLEAR_GAP = .01  # m: the default gap
CLEAR_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')
CLEAR_TOLERANCE = 1e-4  # m: an arm counts as clear at the gap less this; each solve aims FIT_TOLERANCE beyond the gap
# A hand on the floor in the clip (its lowest point within PATCH_TOUCH) is held where it is; one within
# clip_feasibility.TOUCH of it (what the check counts as a support: A473 v2's right hand 3 mm up at 2.16 s, the centre
# of mass 21 cm off the support when it rose) may slide but not rise above its own height. CLEAR_ROUNDS rounds; in each,
# a frame not clear after the smoothing is solved from itself and from its neighbours' corrections carried over, and of
# the solutions without a penetration the one nearest to the neighbours' corrections is kept (a solve from the frame
# alone picks a side per frame: 4 rounds of that left the right arm swinging at 35-50 rad/s over 2.0-2.2 s).
# What this cannot do: A473 v2's right hand passes under the pelvis at 1.9-2.3 s (5, +4, −4 cm in the pelvis frame at
# 2.1 s), so the arm cleared of it changes from one side of the body to the other between two frames (shoulder yaw
# 57 rad/s at 2.18 s; any blend over 0.14-0.52 s goes 6-8 cm through the body): the record's joint_speed_faster says so.
CLEAR_ROUNDS = 8
A493_V2 = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v2'
PATCHES = {
    # The left hand stays where it last touched over 1.9-2.8 s, the feet go down onto the floor behind the knees while
    # the knees and that hand are still down, 2.0-2.8 s plays 1.5 times slower (0.8 -> 1.2 s), the hand is released
    # over 2.8-3.3 s as the pelvis rises, and the waist pitch stays at most 0.4 rad over 2.0-3.8 s (seconds of v2).
    'a493-kneel': dict(base=A493_V2, name=A493_V2.removesuffix('-v2') + '-v3', hold_hand=('left', 1.9, 2.8, 3.3),
                       feet_down=(2.0, 2.7), slow=(2.0, 2.8, 1.5), waist_max=(.4, 2.0, 3.8)),
}
BONES_ATTRIBUTION = ('Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the underlying '
                     'dataset is subject to the BONES Motion Capture Dataset License Agreement.')
BONES_NOTE = ('BONES-SEED (Bones Studio, bones.studio), retargeted to the G1 by NVIDIA; licence: a Qualifying Startup may '
              'train and commercialize models on it, not redistribute or publish the data or derived clips, nor build a '
              'competing dataset; attribution: ' + BONES_ATTRIBUTION)

def cmu_subject(subject, data=None):
    """(folder, skeleton file, subject number) of a CMU subject: --subject a number (the folder cmu_mocap_<N> under
    .tools/datasets) or a folder; --data, if given, is the folder (the old option). The skeleton is <N>.asf, else the
    folder's only .asf; the number is its stem."""
    folder = Path(data) if data else (CMU_DATASETS / f'cmu_mocap_{subject}' if str(subject).isdigit() else Path(subject))
    if not folder.is_dir():
        raise SystemExit(f'нет папки {folder}')
    named = folder / f'{subject}.asf'
    found = [named] if str(subject).isdigit() and named.is_file() else sorted(folder.glob('*.asf'))
    if len(found) != 1:
        raise SystemExit(f'{folder}: нужен один файл скелета .asf, найдено {len(found)}')
    return folder, found[0], found[0].stem


def cmu_take(take, number):
    """The take's file stem: <N>_<take> as given, or the subject's number added to a bare take (06 -> 111_06); no take
    is subject 140's CMU_TAKE, and refused for another subject."""
    if take is None:
        if number != CMU_SUBJECT:
            raise SystemExit(f'субъект {number}: укажите --take (например {number}_01)')
        return CMU_TAKE
    return take if '_' in take else f'{number}_{take}'


def feasibility_header(summary):
    """The header's record of clip_feasibility's summary: the verdict, its reasons and the numbers it was judged by."""
    keys = ('feasible', 'reasons', 'com_margin_min', 'com_margin_at', 'residual_force_max', 'residual_moment_max',
            'ratio_max', 'ratio_joint', 'ratio_at', 'hand_max', 'hand_over_seconds', 'noise', 'qd_max', 'float_max',
            'feasible_waist_free', 'reasons_waist_free', 'waist_free_ratio_max', 'waist_free_ratio_joint')
    out = {key: round(summary[key], 4) if isinstance(summary[key], float) else summary[key] for key in keys}
    out.update(check='scripts/lab/probes/clip_feasibility.py', touch=clip_feasibility.TOUCH,
               criteria=dict(com_outside=clip_feasibility.COM_OUTSIDE, residual_force=clip_feasibility.RESIDUAL_FORCE,
                             residual_arm=clip_feasibility.RESIDUAL_ARM, torque_slack=clip_feasibility.TORQUE_SLACK))
    return out


def lafan_frames(args, env):
    """(qpos at 50 Hz, header fields, name) of a LAFAN1 clip."""
    first, last = args.frames or FRAMES
    data = args.data or DATA
    source = data / args.source
    if not source.is_file():
        raise SystemExit(f'нет файла {source}')
    rows = np.loadtxt(source, delimiter=',')
    if rows.ndim != 2 or rows.shape[1] != 36 or not 0 <= first < last < len(rows):
        raise SystemExit(f'{source.name}: {rows.shape}, кадры {first}-{last} вне клипа')
    qpos, _, _ = resample(rows[first:last + 1].copy(), env.dt)
    header = dict(dataset='lafan1', source_rate=SOURCE_RATE, source=source.name,
                  source_md5=hashlib.md5(source.read_bytes()).hexdigest(), frames=[first, last], data=LAFAN_NOTE)
    return qpos, header, args.name or f'lafan1-{source.stem}-{first}-{last}'


def cmu_frames(args, env):
    """(qpos at 50 Hz, header fields, name) of a CMU take, retargeted (cmu_mocap.retarget), resampled and settled."""
    import cmu_mocap
    folder, skeleton_file, subject = cmu_subject(args.subject, args.data)
    take = cmu_take(args.take, subject)
    if args.frames is None and subject != CMU_SUBJECT:
        raise SystemExit(f'{take}: укажите --frames (окно подъёма, номера кадров AMC)')
    first, last = args.frames or CMU_FRAMES
    source = folder / f'{take}.amc'
    if not source.is_file():
        raise SystemExit(f'нет файла {source}')
    skeleton = cmu_mocap.read_asf(skeleton_file)
    frames = cmu_mocap.read_amc(source)
    numbers = [number for number, _ in frames]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))) or not numbers[0] <= first < last <= numbers[-1]:
        raise SystemExit(f'{source.name}: кадры {numbers[0]}-{numbers[-1]}, а просят {first}-{last}')
    neutral = numbers[0] if args.neutral is None else args.neutral
    if not numbers[0] <= neutral <= numbers[-1]:
        raise SystemExit(f'{source.name}: нейтрального кадра {neutral} нет')
    chosen = frames[first - numbers[0]:last - numbers[0] + 1]
    rounds = {}
    qpos, errors, ratios = cmu_mocap.retarget(env, skeleton, chosen, frames[neutral - numbers[0]],
                                              cmu_mocap.ROUNDS if args.rounds is None else args.rounds, rounds)
    qpos = cmu_mocap.smooth(env, qpos, CMU_RATE)
    projected = {}
    if cmu_mocap.PROJECT:
        qpos = cmu_mocap.project(env, qpos, CMU_RATE, projected)
        waited = sum(frames for _, frames in projected['pauses'])
        print(f'   опора: переносов веса {len(projected["pauses"])} ({waited / CMU_RATE:.2f} с ожидания, не удалось '
              f'{len(projected["gave_up"])}), нехватка запаса ЦМ до {projected["short"] * 100:.1f} см')
    qpos = cmu_mocap.settle(env, cmu_mocap.resample(qpos, CMU_RATE, env.dt), args.settle)
    report = {}
    for k, target in enumerate(cmu_mocap.ERRORS):
        scale = 100 if target in cmu_mocap.DISTANCES else 180 / np.pi
        report[target] = [round(float(np.median(errors[:, k])) * scale, 1), round(float(errors[:, k].max()) * scale, 1)]
    header = dict(dataset='cmu', subject=subject, source_rate=CMU_RATE, source=source.name,
                  source_md5=hashlib.md5(source.read_bytes()).hexdigest(), skeleton=skeleton_file.name,
                  skeleton_md5=hashlib.md5(skeleton_file.read_bytes()).hexdigest(), frames=[first, last],
                  neutral_frame=neutral, settle_seconds=args.settle,
                  ratios={key: round(float(value), 4) for key, value in ratios.items()},
                  retarget_error=report, retarget_units='cm for hands, contacts, floor and support, degrees otherwise: '
                                                        '[median, max]',
                  retarget=dict(version=cmu_mocap.RETARGET_VERSION, rounds=rounds.get('rounds', []),
                                smooth_seconds=cmu_mocap.SMOOTH_SECONDS, com_margin=cmu_mocap.COM_MARGIN,
                                project=projected or None),
                  data=CMU_NOTE.format(subject=subject))
    print(f'ИК {source.name} кадры {first}-{last}: длины ног × {ratios["legs"]:.3f}, рук × {ratios["arms"]:.3f}; '
          'ошибка цели медиана / max: ' + ', '.join(f'{k} {v[0]:g}/{v[1]:g}' for k, v in report.items()))
    for k, r in enumerate(rounds.get('rounds', [])):
        waited = sum(frames for _, frames in r['pauses'])
        print(f'   проход {k + 1}: переносов веса {len(r["pauses"])} ({waited} кадров ожидания, не удалось '
              f'{len(r["gave_up"])}), пределы сдвинуты в {r["moved_frames"]} кадрах ({", ".join(r["joints"]) or "—"}), '
              f'нехватка запаса ЦМ до {r["support_short"] * 100:.1f} см')
    return qpos, header, args.name or f'cmu-{take}-{first}-{last}'


def euler_quat(angles):
    """(n, 4) w x y z quaternions of BONES root angles (n, 3), degrees about X, Y, Z applied about the fixed world axes
    in that order (BONES_ROOT_ORDER: q = qz·qy·qx)."""
    import mujoco
    out = np.zeros((len(angles), 4))
    for k, row in enumerate(np.radians(np.asarray(angles, dtype=np.float64))):
        q = np.array([1., 0., 0., 0.])
        for axis, angle in zip(np.eye(3), row):
            turn = np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, axis, angle)
            mujoco.mju_mulQuat(q, turn.copy(), q.copy())
        out[k] = q
    return out


def read_bones(path, joint_names):
    """(Frame numbers (F,), qpos (F, 36): the root in m, w x y z, the joints in rad) of a BONES-SEED G1 CSV whose
    header is Frame, the root's three translations and rotations and joint_names + '_dof' in that order (refused
    otherwise: another joint order or set would need a mapping); the frames numbered consecutively."""
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    expected = ['Frame', *(f'root_translate{a}' for a in 'XYZ'), *(f'root_rotate{a}' for a in 'XYZ'),
                *(f'{name}_dof' for name in joint_names)]
    if header != expected:
        raise SystemExit(f'{path.name}: столбцы не те, что у G1 29 DoF в нашем порядке:\n  {header}')
    rows = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    numbers = rows[:, 0].astype(np.int64)
    if len(rows) < 2 or rows.shape[1] != 36 or not np.isfinite(rows).all() \
            or (numbers != np.arange(numbers[0], numbers[0] + len(numbers))).any():
        raise SystemExit(f'{path.name}: {rows.shape}, кадры не подряд или не числа')
    qpos = np.concatenate([rows[:, 1:4] / 100., euler_quat(rows[:, 4:7]), np.radians(rows[:, 7:])], 1)
    return numbers, qpos


def drop_to_floor(env, qpos, frames=None):
    """(qpos with its root lowered, the drop (F,), m) for frames whose lowest robot geom (cmu_mocap.lowest_point, the
    floor-touching geoms) is above the floor: the height h ≥ 0 per frame, a running minimum over ±frames, then a running
    mean over ±frames (g1_getup_track.smooth_lift's mirror), so no frame is lowered by more than its own h, none into
    the floor; frames already in it are left for build's lift."""
    import cmu_mocap
    from humanoid_sim.tasks import g1_getup_track
    frames = g1_getup_track.LIFT_FRAMES if frames is None else frames
    model, data, geoms = cmu_mocap._floor_model(env)
    height = np.array([max(cmu_mocap.lowest_point(model, data, row, geoms), 0.) for row in qpos])
    count = len(height)
    index = np.clip(np.arange(count)[:, None] + np.arange(-frames, frames + 1)[None], 0, count - 1)
    drop = height[index].min(1)[index].mean(1)
    out = np.array(qpos, dtype=np.float64)
    out[:, 2] -= drop
    return out, drop


def fit_floor(env, qpos, version=BONES_FIT):
    """(qpos, stats) of frames qpos (F, 36) fitted to our getting-up model's floor: the locked wrists at home; each
    frame raised by smooth_lift of its trunk's depth (the floor geoms outside the limbs: pelvis, torso, head); then, per
    frame and limb (LIMBS: each arm from its shoulder pitch link, each leg from its hip pitch link, Unitree's hand and
    foot capsules included), the limb's joints moved from the frame's own angles by damped Gauss-Newton steps on its
    lowest point's height (that point's mj_jac row), within the task's soft limits (_lift_limb).
    version v1: a limb reaching below the floor, each step the least change that puts its lowest point LIMB_CLEARANCE
    above the floor, unbounded, every frame on its own. v2 (BONES_FIT): each step at most FIT_STEP per joint; an arm
    whose lowest point was in or on the floor (height ≤ 0 after the trunk lift) put at LIMB_CLEARANCE (two-sided), any
    other limb only out of the floor; FIT_ROUNDS solves, the correction smoothed over ±FIT_SMOOTH frames between them.
    stats: trunk_lift (F,), limb_before / limb_after (F, 4) depths ≥ 0, m; v2 also arm_on_floor (F, 4) bool (the
    two-sided frames; legs False) and limb_height (F, 4), the lowest point after the fit, m. Why: the BONES retarget's
    G1 has neither hand nor foot capsules, so a palm on the floor puts our hand capsule 5-14 cm into it (the locked
    wrists up to 6 cm more) and a heel lying on the back ours 4-6 cm, and build's lift alone would raise the whole body
    by that (the pelvis 0.23 m lying); v2's why is at BONES_FITS."""
    import cmu_mocap
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    if version not in BONES_FITS:
        raise ValueError(f'fit_floor: version {version!r}, known {BONES_FITS}')
    model, data, geoms = cmu_mocap._floor_model(env)
    qpos = np.array(qpos, dtype=np.float64)
    wrists = env.wrist_mask.numpy()
    qpos[:, np.asarray(env.qadr)[wrists]] = env.home[wrists]
    limbs, in_limb = [], set()
    for top_name, joint_names in LIMBS:
        top = model.body(top_name).id
        bodies = {b for b in range(model.nbody) if b == top or _descends(model, b, top)}
        in_limb |= bodies
        joints = [model.joint(name).id for name in joint_names]
        low, high = task.soft_limits(*model.jnt_range[joints].T)
        limbs.append(([g for g in geoms if model.geom_bodyid[g] in bodies], model.jnt_qposadr[joints],
                      model.jnt_dofadr[joints], low, high))
    trunk = [g for g in geoms if model.geom_bodyid[g] not in in_limb]
    depth = np.array([max(-cmu_mocap.lowest_point(model, data, row, trunk), 0.) for row in qpos])
    lift = g1_getup_track.smooth_lift(depth)
    qpos[:, 2] += lift
    before, after = np.zeros((len(qpos), len(limbs))), np.zeros((len(qpos), len(limbs)))
    jacobian = np.zeros((3, model.nv))
    if version == 'v1':
        for k, row in enumerate(qpos):
            for a, limb in enumerate(limbs):
                height, before[k, a] = _lift_limb(model, data, row, limb, jacobian, LIMB_ITERATIONS, LIMB_DAMPING)
                after[k, a] = max(-height, 0.)
        return qpos, dict(trunk_lift=lift, limb_before=before, limb_after=after)
    raw = qpos.copy()
    first = np.zeros((len(raw), len(limbs)))  # each limb's lowest point in the data (after the trunk lift)
    for k, row in enumerate(raw):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        first[k] = [min(float(cmu_mocap.geom_lowest(model, data, g)[2]) for g in limb[0]) for limb in limbs]
    before = np.maximum(-first, 0.)
    arms = np.array([top.endswith('shoulder_pitch_link') for top, _ in LIMBS])
    on_floor = (first <= 0.) & arms[None]
    height = np.zeros_like(first)
    for round_ in range(FIT_ROUNDS):
        if round_:
            qpos[:, 7:] = raw[:, 7:] + running_mean(qpos[:, 7:] - raw[:, 7:], FIT_SMOOTH)
        for k, row in enumerate(qpos):
            for a, limb in enumerate(limbs):
                height[k, a], _ = _lift_limb(model, data, row, limb, jacobian, FIT_ITERATIONS, FIT_DAMPING,
                                             step=FIT_STEP, two_sided=bool(on_floor[k, a]))
    after = np.maximum(-height, 0.)
    return qpos, dict(trunk_lift=lift, limb_before=before, limb_after=after, arm_on_floor=on_floor,
                      limb_height=height)


def running_mean(values, frames):
    """values (F, …) averaged over ±frames, the ends repeated."""
    index = np.clip(np.arange(len(values))[:, None] + np.arange(-frames, frames + 1)[None], 0, len(values) - 1)
    return values[index].mean(1)


def _lift_limb(model, data, row, limb, jacobian, iterations, damping, step=None, two_sided=False):
    """(the limb's lowest point's height after, its depth ≥ 0 before, m) of row (36,), changed in place: damped
    Gauss-Newton steps of the limb's joints on its lowest point's height (that point's mj_jac row) toward
    LIMB_CLEARANCE, clipped to the soft limits; one-sided: until that point is out of the floor; two-sided: until it is
    within FIT_TOLERANCE of LIMB_CLEARANCE (above it the point is pulled down); step: each step's largest joint change
    (None: unbounded, fit v1)."""
    import cmu_mocap
    import mujoco
    limb_geoms, qadr, dofs, low, high = limb
    before = 0.
    for k in range(iterations + 1):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        points = [cmu_mocap.geom_lowest(model, data, g) for g in limb_geoms]
        lowest = int(np.argmin([p[2] for p in points]))
        height = float(points[lowest][2])
        if k == 0:
            before = max(-height, 0.)
        done = abs(height - LIMB_CLEARANCE) < FIT_TOLERANCE if two_sided else height >= 0.
        if done or k == iterations:
            break
        mujoco.mj_comPos(model, data)  # mj_jac reads cdof and subtree_com
        mujoco.mj_jac(model, data, jacobian, None, points[lowest], model.geom_bodyid[limb_geoms[lowest]])
        row_z = jacobian[2, dofs]
        change = row_z * (LIMB_CLEARANCE - height) / (row_z @ row_z + damping)
        if step is not None:
            change *= min(1., step / max(float(np.abs(change).max()), 1e-12))
        row[qadr] = np.clip(row[qadr] + change, low, high)
    return height, before


def _descends(model, body, top):
    """Whether body lies in top's subtree."""
    while body > 0:
        body = model.body_parentid[body]
        if body == top:
            return True
    return False


def bones_frames(args, env):
    """(qpos at 50 Hz, header fields, name) of a BONES-SEED G1 CSV: read (read_bones), fitted to our floor (fit_floor:
    the trunk lifted, the limbs raised by their own joints) and the body lowered onto it at 120 Hz (drop_to_floor),
    resampled to 50 Hz (cmu_mocap.resample) and settled to home (cmu_mocap.settle)."""
    import cmu_mocap
    from humanoid_sim.tasks.config import GETUP_TRACK_NAME
    from humanoid_sim.tasks.g1_getup_bank import joint_names
    if not args.file:
        raise SystemExit('--source bones: укажите --file <файл CSV> (в .tools/datasets/bones_seed/g1_getup или путь)')
    source = Path(args.file)
    if not source.is_file():
        source = BONES_DIR / (args.file if args.file.endswith('.csv') else f'{args.file}.csv')
    if not source.is_file():
        raise SystemExit(f'нет файла {args.file} (и {source})')
    numbers, qpos = read_bones(source, joint_names(env))
    first, last = args.frames or (int(numbers[0]), int(numbers[-1]))
    if not numbers[0] <= first < last <= numbers[-1]:
        raise SystemExit(f'{source.name}: кадры {numbers[0]}-{numbers[-1]}, а просят {first}-{last}')
    version = getattr(args, 'fit', None) or BONES_FIT
    data = qpos[first - numbers[0]:last - numbers[0] + 1]
    qpos, fit = fit_floor(env, data, version)
    speed = np.abs(np.diff(qpos[:, 7:], axis=0)).max(0) * BONES_RATE  # per joint, rad/s at 120 Hz
    speed_data = np.abs(np.diff(data[:, 7:], axis=0)).max() * BONES_RATE
    qpos, drop = drop_to_floor(env, qpos)
    before, after = fit['limb_before'], fit['limb_after']
    qpos = cmu_mocap.settle(env, cmu_mocap.resample(qpos, BONES_RATE, env.dt), args.settle)
    stem = source.stem
    header = dict(dataset='bones_seed', source_rate=BONES_RATE, source=source.name,
                  source_md5=hashlib.md5(source.read_bytes()).hexdigest(), frames=[first, last],
                  performer=stem.rsplit('__', 1)[-1].removesuffix('_M') if '__' in stem else None,
                  mirrored=stem.endswith('_M'), root_order=BONES_ROOT_ORDER, settle_seconds=args.settle,
                  drop_max=round(float(drop.max()), 4), drop_median=round(float(np.median(drop)), 4),
                  trunk_lift_max=round(float(fit['trunk_lift'].max()), 4),
                  limbs=[top for top, _ in LIMBS], limb_depth_before=[round(float(v), 4) for v in before.max(0)],
                  limb_depth_after=[round(float(v), 4) for v in after.max(0)],
                  limb_fix_frames=[int(v) for v in (before > 0).sum(0)], data=BONES_NOTE)
    names = joint_names(env)
    if version != 'v1':  # v1's header stays byte for byte (branch 71's clip is named by its md5)
        on_floor = fit['arm_on_floor']
        palm = [float(np.abs(fit['limb_height'][on_floor[:, a], a] - LIMB_CLEARANCE).max()) if on_floor[:, a].any()
                else 0. for a in range(len(LIMBS))]
        header['bones_fit'] = dict(
            version=version, step=FIT_STEP, damping=FIT_DAMPING, iterations=FIT_ITERATIONS, tolerance=FIT_TOLERANCE,
            rounds=FIT_ROUNDS, smooth_frames=FIT_SMOOTH, clearance=LIMB_CLEARANCE,
            arm_on_floor_frames=[int(v) for v in on_floor.sum(0)], palm_off_floor_max=[round(v, 4) for v in palm],
            joint_speed_max=round(float(speed.max()), 2), joint_speed_joint=names[int(speed.argmax())],
            data_joint_speed_max=round(float(speed_data), 2), speed_units='rad/s between 120 Hz frames, before the '
                                                                          'drop and the resampling')
    print(f'{source.name} кадры {first}-{last} (120 Гц), перенос на пол {version}: корпус поднят из пола до '
          f'{fit["trunk_lift"].max() * 100:.1f} см; конечности в полу (л. рука, п. рука, л. нога, п. нога) до '
          + ' / '.join(f'{v * 100:.1f}' for v in before.max(0)) + ' см, в '
          + ' / '.join(f'{v:.0%}' for v in (before > 0).mean(0)) + ' кадров, после поправки до '
          + ' / '.join(f'{v * 100:.1f}' for v in after.max(0))
          + f' см; опущено на пол до {drop.max() * 100:.1f} см (медиана {np.median(drop) * 100:.1f})')
    print(f'   скорость суставов после переноса max {speed.max():.1f} рад/с ({names[int(speed.argmax())]}), в данных '
          f'{speed_data:.1f}' + ('' if version == 'v1' else '; ладонь на полу (кадры, где рука в данных на полу или в '
                                 'нём): л. / п. ' + ' / '.join(f'{int(n)} кадров, отрыв до {v * 100:.2f} см' for n, v in
                                 zip(header['bones_fit']['arm_on_floor_frames'][:2],
                                     header['bones_fit']['palm_off_floor_max'][:2]))))
    name = args.name or f'bones-{stem}-{first}-{last}' + ('' if version == 'v1' else f'-{version}')
    if not re.fullmatch(GETUP_TRACK_NAME, name):
        raise SystemExit(f'имя клипа {name!r} не подходит ({GETUP_TRACK_NAME}): задайте --name')
    return qpos, header, name


def smoothstep(x):
    """3x² − 2x³ of x clipped to [0, 1]."""
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def slow_clock(count, dt, start, end, factor, ramp=PATCH_SLOW_RAMP):
    """(K',) the source frame (a float index into count frames dt apart) of each output frame when the source span
    start-end s plays factor times slower: the clock's speed is 1 up to start, eases (smoothstep) over ramp s of output
    time to c and back to 1 by the span's end, c chosen so that the span lasts (end − start)·factor s; after it the
    source frames one to one, shifted. Frames outside the span are whole indices, so they copy the source exactly."""
    a, b = start / dt, end / dt
    span, out = b - a, (end - start) * factor / dt
    r = ramp / dt
    if abs(a - round(a)) > 1e-6 or abs(b - round(b)) > 1e-6 or abs(out - round(out)) > 1e-6:
        raise SystemExit(f'--slow {start} {end} {factor}: начало, конец и новая длина — целые кадры по {dt} с')
    a, b, out = int(round(a)), int(round(b)), int(round(out))
    span = b - a
    if not (0 <= a < b <= count - 1 and factor > 1 and 2 * r < out):
        raise SystemExit(f'--slow {start} {end} {factor}: отрезок вне клипа, множитель ≤ 1 или отрезок короче '
                         f'двух разгонов по {ramp} с')
    c = (span - r) / (out - r)

    def eased(u):  # the source frames covered after u output frames of a speed ramp from 1 to c
        x = u / r
        return u - (1. - c) * r * (x ** 3 - x ** 4 / 2.)

    clock = []
    for f in range(count + out - span):
        u = f - a
        if u <= 0:
            clock.append(float(f))
        elif u >= out:
            clock.append(float(f - out + span))
        elif u <= r:
            clock.append(a + eased(u))
        elif u >= out - r:
            clock.append(b - eased(out - u))
        else:
            clock.append(a + eased(r) + c * (u - r))
    return np.array(clock)


def clock_frames(qpos, clock):
    """qpos (K, 36) at the float source frames clock (K',): the joints and the root position linearly, the root
    quaternion by slerp; a whole index copies its frame exactly."""
    from getup_style_library import slerp
    index = np.minimum(np.floor(clock).astype(np.int64), len(qpos) - 2)
    s = clock - index
    out = (1. - s)[:, None] * qpos[index] + s[:, None] * qpos[index + 1]
    out[:, 3:7] = slerp(qpos[index, 3:7], qpos[index + 1, 3:7], s)
    whole = s == 0.
    out[whole] = qpos[index[whole]]
    return out


def _window(t, start, end, ramp=PATCH_RAMP):
    """The weight at times t of a span start-end s: 0 → 1 by a smoothstep over ramp s before start, 1 to end, 1 → 0
    over ramp s after it."""
    t = np.asarray(t, dtype=np.float64)
    weight = np.where(t < start, smoothstep((t - start + ramp) / ramp),
                      np.where(t > end, 1. - smoothstep((t - end) / ramp), 1.))
    return np.where(weight < 1e-9, 0., weight)  # a frame at the ramp's edge (k·dt rounded) as it is


def _rotation_error(target, current):
    """The world rotation vector (3,) that turns rotation matrix current (3, 3) into target."""
    import mujoco
    quat, error = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quat, (target @ current.T).ravel())
    mujoco.mju_quat2Vel(error, quat, 1.)
    return error


def _solve(model, data, row, joints, goals, iterations=PATCH_ITERATIONS, damping=FIT_DAMPING, step=FIT_STEP):
    """row (36,) changed in place: damped Gauss-Newton steps of the joints (ids, within the task's soft limits or as far
    out as the row already is), each at most step rad per joint, toward goals, a list of ('rotation', body, R (3, 3)):
    the body's world rotation; ('point', body, p (3,)): its frame origin; ('lowest', geoms, target (3,), axes): the
    geoms' lowest point (cmu_mocap.geom_lowest) at target along axes (a list of world axes 0-2). Several goals are met
    in the least-squares sense. Returns the largest remaining error (rad or m)."""
    import cmu_mocap
    import mujoco
    from humanoid_sim.tasks import g1_tensor_math as task
    qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    low, high = task.soft_limits(*model.jnt_range[joints].T)
    low, high = np.minimum(low, row[qadr]), np.maximum(high, row[qadr])  # a clip joint beyond them is not pulled in
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    worst = 0.
    for k in range(iterations + 1):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # mj_jac reads cdof and subtree_com
        errors, rows = [], []
        for goal in goals:
            if goal[0] == 'rotation':
                _, body, target = goal
                errors.append(_rotation_error(target, data.xmat[body].reshape(3, 3)))
                mujoco.mj_jacBody(model, data, None, jacr, body)
                rows.append(jacr[:, dofs].copy())
            elif goal[0] == 'point':
                _, body, target = goal
                errors.append(target - data.xpos[body])
                mujoco.mj_jacBody(model, data, jacp, None, body)
                rows.append(jacp[:, dofs].copy())
            else:
                _, geoms, target, axes = goal
                points = [cmu_mocap.geom_lowest(model, data, g) for g in geoms]
                lowest = int(np.argmin([p[2] for p in points]))
                mujoco.mj_jac(model, data, jacp, None, points[lowest], model.geom_bodyid[geoms[lowest]])
                errors.append((np.asarray(target) - points[lowest])[axes])
                rows.append(jacp[axes][:, dofs].copy())
        error, jacobian = np.concatenate(errors), np.vstack(rows)
        worst = float(np.abs(error).max())
        if worst < FIT_TOLERANCE or k == iterations:
            break
        change = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(len(error)), error)
        change *= min(1., step / max(float(np.abs(change).max()), 1e-12))
        row[qadr] = np.clip(row[qadr] + change, low, high)
    return worst


def _leg_goals(model, data, leg, knee_height=None):
    """The goals (_solve) that keep a leg (joints, foot body, knee geoms) as it is in the current kinematics: the foot
    body's world rotation and position; knee_height: also its knee's lowest point at that height."""
    _, foot, knee = leg
    goals = [('rotation', foot, data.xmat[foot].reshape(3, 3).copy()), ('point', foot, data.xpos[foot].copy())]
    if knee_height is not None:
        goals.append(('lowest', knee, np.array([0., 0., knee_height]), [2]))
    return goals


def _knee_down(model, data, leg):
    """The leg's knee's lowest point height if within PATCH_TOUCH·2.5 (5 mm) of the floor, else None."""
    import cmu_mocap
    height = min(float(cmu_mocap.geom_lowest(model, data, g)[2]) for g in leg[2])
    return height if height <= 2.5 * PATCH_TOUCH else None


def _waist_ceiling(model, data, row, waist, ceiling, weight, legs, arms):
    """row (36,) changed in place: the waist pitch pulled below ceiling (a soft minimum PATCH_WAIST_SOFT wide) by
    weight (0-1), the torso's world rotation kept: the pelvis turned by the difference about the midpoint of the hip
    pitch joints (the legs hang there, so they move least), each leg's six joints solved so that its foot keeps its
    pose and a knee on the floor its height (least squares), and each arm (joints, hand geoms) whose hand is near the
    floor (PATCH_HAND_NEAR) solved so that the hand's lowest point stays where it was (the torso moves about 1 cm).
    Returns (the waist pitch's change, the legs' and the arms' largest joint change, rad)."""
    import cmu_mocap
    import mujoco
    adr = model.jnt_qposadr[waist]
    q = row[adr]
    soft = ceiling - PATCH_WAIST_SOFT * np.logaddexp(0., (ceiling - q) / PATCH_WAIST_SOFT)
    new = q + weight * (soft - q)
    if new == q:
        return 0., 0.
    torso = model.jnt_bodyid[waist]
    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    before = data.xmat[torso].reshape(3, 3).copy()
    pivot = data.xanchor[[leg[0][0] for leg in legs]].mean(0)
    goals = [(leg[0], _leg_goals(model, data, leg, _knee_down(model, data, leg))) for leg in legs]
    for joints, hand in arms:
        lowest = min((cmu_mocap.geom_lowest(model, data, g) for g in hand), key=lambda p: p[2])
        if lowest[2] <= PATCH_HAND_NEAR:
            goals.append((joints, [('lowest', hand, lowest.copy(), [0, 1, 2])]))
    row[adr] = new
    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    turn = before @ data.xmat[torso].reshape(3, 3).T  # world: the pelvis turned by this keeps the torso's rotation
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, turn.ravel())
    row[:3] = pivot + turn @ (row[:3] - pivot)
    root = np.zeros(4)
    mujoco.mju_mulQuat(root, quat, row[3:7].copy())
    row[3:7] = root / np.linalg.norm(root)
    qadr = model.jnt_qposadr[np.concatenate([joints for joints, _ in goals])]
    old = row[qadr].copy()
    for joints, goal in goals:
        _solve(model, data, row, joints, goal, step=.2)
    return float(new - q), float(np.abs(row[qadr] - old).max())


def _clear_model(env, gap):
    """(a copy of env's model with the robot geoms' margin at gap, its data, the arms, each arm's obstacles) for
    clear_self. An arm (per side): its CLEAR_JOINTS ids, its colliding robot geoms (the shoulder pitch link's subtree:
    the links and Unitree's hand capsule), its hand's floor geoms (the wrist links'), its floor geoms. Its obstacles:
    every other colliding robot geom, the legs, the trunk and the other arm (as it is when this arm is solved: an arm
    moved off a hip must not go into the other hand, A473 v2's left hand at 0.8 s)."""
    import copy
    import cmu_mocap
    import mujoco
    model, _, floor_geoms = cmu_mocap._floor_model(env)
    robot = np.asarray(env.robot_geoms, bool)
    margin = copy.deepcopy(model)
    margin.geom_margin[robot] = gap
    colliding = {g for g in range(model.ngeom) if robot[g] and (model.geom_contype[g] or model.geom_conaffinity[g])}
    arms = []
    for side in ('left', 'right'):
        top = model.body(f'{side}_shoulder_pitch_link').id
        bodies = {b for b in range(model.nbody) if b == top or _descends(model, b, top)}
        arms.append(([model.joint(f'{side}_{j}_joint').id for j in CLEAR_JOINTS],
                     {g for g in colliding if model.geom_bodyid[g] in bodies},
                     [g for g in floor_geoms if model.body(model.geom_bodyid[g]).name.startswith(f'{side}_wrist')],
                     [g for g in floor_geoms if model.geom_bodyid[g] in bodies]))
    return margin, mujoco.MjData(margin), arms, [colliding - arm[1] for arm in arms]


def _arm_contacts(model, data, geoms, others):
    """[(distance m, the arm geom, the other geom, contact)] of data's contacts (after mj_collision with the margin
    model) between one of geoms and one of others: the pairs closer than the margin."""
    out = []
    for c in range(data.ncon):
        contact = data.contact[c]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in geoms and g2 in others:
            out.append((float(contact.dist), g1, g2, contact))
        elif g2 in geoms and g1 in others:
            out.append((float(contact.dist), g2, g1, contact))
    return out


def _pair_name(model, first, second):
    return '-'.join(model.body(model.geom_bodyid[g]).name.removesuffix('_link') for g in (first, second))


def _arm_terms(model, data, row, arm, others, gap, hand, floor_z, jacp):
    """(clear: the arm meets every condition, the smallest arm-to-other distance (≤ gap, m), its pair or None, the
    hand's error (m), the residuals (E,) and their rows (E, arm joints)) of row: each arm-to-other pair within the
    margin model's gap toward gap + FIT_TOLERANCE along its contact normal; hand None (free), ('hold', p (3,)): the
    hand's lowest point at p, or ('ceiling', z): not above z; the arm's lowest point not below floor_z."""
    import cmu_mocap
    import mujoco
    joints, geoms, hand_geoms, floor = arm
    dofs = model.jnt_dofadr[joints]
    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)  # mj_jac reads cdof and subtree_com
    mujoco.mj_collision(model, data)
    errors, rows = [], []
    nearest, pair, hand_error = gap, None, 0.
    for dist, mine, other, contact in _arm_contacts(model, data, geoms, others):
        if dist < nearest:
            nearest, pair = dist, _pair_name(model, mine, other)
        # the normal points from geom1 to geom2: the arm geom moving along it (geom2) or against it (geom1) parts them
        sign = 1. if int(contact.geom2) == mine else -1.
        mujoco.mj_jac(model, data, jacp, None, contact.pos, model.geom_bodyid[mine])
        errors.append(gap + FIT_TOLERANCE - dist)
        rows.append(sign * contact.frame[:3] @ jacp[:, dofs])
    if hand is not None:
        point, g = min(((cmu_mocap.geom_lowest(model, data, g), g) for g in hand_geoms), key=lambda p: p[0][2])
        mujoco.mj_jac(model, data, jacp, None, point, model.geom_bodyid[g])
        if hand[0] == 'hold':
            offset = np.asarray(hand[1]) - point
            hand_error = float(np.linalg.norm(offset))
            errors += list(offset)
            rows += list(jacp[:, dofs].copy())
        else:
            hand_error = max(float(point[2]) - hand[1], 0.)
            if hand_error > 0.:
                errors.append(-hand_error)
                rows.append(jacp[2, dofs].copy())
    point, g = min(((cmu_mocap.geom_lowest(model, data, g), g) for g in floor), key=lambda p: p[0][2])
    below = floor_z - float(point[2])
    if below > FIT_TOLERANCE:
        mujoco.mj_jac(model, data, jacp, None, point, model.geom_bodyid[g])
        errors.append(below)
        rows.append(jacp[2, dofs].copy())
    clear = nearest >= gap - CLEAR_TOLERANCE and hand_error < FIT_TOLERANCE and below <= FIT_TOLERANCE
    return clear, nearest, pair, hand_error, np.array(errors), np.array(rows).reshape(len(errors), len(dofs))


def _clear_arm(model, data, row, arm, others, gap, hand=None, floor_z=0.):
    """row (36,) changed in place: damped Gauss-Newton steps of the arm's CLEAR_JOINTS (within the task's soft limits
    or as far out as the row already is), each at most FIT_STEP rad per joint, until _arm_terms' conditions hold: every
    arm geom gap away from every geom of others, the hand as `hand` says (held where it is on the floor, or not above
    its height), the arm's lowest point not below floor_z. Returns (the smallest arm-to-other distance left, at most
    gap, m; its pair's body names or None; the hand's error left, m)."""
    from humanoid_sim.tasks import g1_tensor_math as task
    joints = arm[0]
    qadr = model.jnt_qposadr[joints]
    low, high = task.soft_limits(*model.jnt_range[joints].T)
    low, high = np.minimum(low, row[qadr]), np.maximum(high, row[qadr])
    jacp = np.zeros((3, model.nv))
    for k in range(FIT_ITERATIONS + 1):
        clear, nearest, pair, hand_error, error, jacobian = _arm_terms(model, data, row, arm, others, gap, hand,
                                                                       floor_z, jacp)
        if clear or k == FIT_ITERATIONS:
            break
        change = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + FIT_DAMPING * np.eye(len(error)), error)
        change *= min(1., FIT_STEP / max(float(np.abs(change).max()), 1e-12))
        row[qadr] = np.clip(row[qadr] + change, low, high)
    return nearest, pair, hand_error


def _spans(frames):
    """[[first, last]] of the runs of consecutive frame numbers."""
    runs = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = int(f)
        else:
            runs.append([int(f), int(f)])
    return runs


def clear_self(env, qpos, gap=CLEAR_GAP):
    """(qpos (K, 36) with the arms moved, changed (K,) bool: the frames whose arm joints differ, the header's record)
    of frames qpos in env (the getting-up model): per frame and arm _clear_arm (every arm geom gap away from the legs,
    the trunk and the other arm; a hand on the floor in the frame held where it is, one touching it not above its
    height (CLEAR_ROUNDS' comment); the arm not lower than the frame's own lowest point or the floor), CLEAR_ROUNDS
    solves with the correction (arm − frame) smoothed over ±FIT_SMOOTH frames between them, as fit v2. A frame that
    meets every condition and gets no correction from its neighbours is not touched. The record: the frames closer
    than the gap and deeper than clip_feasibility.SELF_DEEP (any two robot geoms) before and after, the frames where
    the gap was not reached (spans with the smallest distance left and its pair), the hands on the floor (frames held
    and their error, frames touching and how far they rose, the slide of the lowest point), the largest joint
    change."""
    import cmu_mocap
    import mujoco
    model, data, arms, others = _clear_model(env, gap)
    plain, plain_data, _ = cmu_mocap._floor_model(env)
    robot = np.asarray(env.robot_geoms, bool)
    reference = np.array(qpos, dtype=np.float64)
    rows = reference.copy()
    count = len(rows)
    qadr = np.concatenate([model.jnt_qposadr[arm[0]] for arm in arms])
    home = cmu_mocap.home_qpos(env)

    def lowest(row, geoms):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        return min((cmu_mocap.geom_lowest(model, data, g) for g in geoms), key=lambda p: p[2])

    def measure(frames):
        """(K, 2) the smallest arm-to-other distance (≤ gap) per arm, [K] its pair, (K,) the deepest penetration."""
        nearest, pairs, deep = np.full((count, 2), gap), [[None, None] for _ in range(count)], np.zeros(count)
        for f, row in enumerate(frames):
            data.qpos[:] = row
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            for a, arm in enumerate(arms):
                for dist, mine, other, _ in _arm_contacts(model, data, arm[1], others[a]):
                    if dist < nearest[f, a]:
                        nearest[f, a], pairs[f][a] = dist, _pair_name(model, mine, other)
            plain_data.qpos[:] = row
            mujoco.mj_kinematics(plain, plain_data)
            mujoco.mj_collision(plain, plain_data)
            deep[f] = clip_feasibility.self_penetration(plain, plain_data, robot)[0]
        return nearest, pairs, deep

    hand = np.array([[lowest(row, arm[2]) for arm in arms] for row in reference])  # (K, 2, 3)
    held = hand[:, :, 2] <= PATCH_TOUCH  # CLEAR_ROUNDS' comment: on the floor, held where it is
    touching = ~held & (hand[:, :, 2] <= clip_feasibility.TOUCH)  # a support for the check: not above its height
    floor_z = np.minimum(np.array([[lowest(row, arm[3])[2] for arm in arms] for row in reference]), 0.)
    before, _, deep_before = measure(reference)
    jacp = np.zeros((3, model.nv))
    for round_ in range(CLEAR_ROUNDS):
        if round_:
            rows[:, qadr] = reference[:, qadr] + running_mean(rows[:, qadr] - reference[:, qadr], FIT_SMOOTH)
        for f in range(count):
            for a, arm in enumerate(arms):
                rule = ('hold', hand[f, a]) if held[f, a] else ('ceiling', hand[f, a, 2]) if touching[f, a] else None
                if _arm_terms(model, data, rows[f], arm, others[a], gap, rule, floor_z[f, a], jacp)[0]:
                    continue
                # CLEAR_ROUNDS' comment: solved from the frame as it is, from its neighbours' corrections carried over
                # and from the arm at home; of the solutions that reach the gap (the hand where it must be), the one
                # nearest to its neighbours' corrections, else of those without a penetration the one with the least
                # left
                arm_qadr = model.jnt_qposadr[arm[0]]
                near = [n for n in (f - 1, f + 1) if 0 <= n < count]
                starts = [rows[f].copy()]
                for n in near:
                    starts.append(rows[f].copy())
                    starts[-1][arm_qadr] = reference[f, arm_qadr] + rows[n, arm_qadr] - reference[n, arm_qadr]
                starts.append(rows[f].copy())
                starts[-1][arm_qadr] = home[arm_qadr]
                solved = []
                for start in starts:
                    nearest, _, error = _clear_arm(model, data, start, arm, others[a], gap, rule, float(floor_z[f, a]))
                    short = max(-nearest, 0.) + (error if error >= FIT_TOLERANCE else 0.)
                    left = max(gap - CLEAR_TOLERANCE - nearest, 0.) + (error if error >= FIT_TOLERANCE else 0.)
                    jump = max((float(np.abs(start[arm_qadr] - reference[f, arm_qadr] - rows[n, arm_qadr]
                                             + reference[n, arm_qadr]).max()) for n in near), default=0.)
                    solved.append((left > 0., short > 0., left if left > 0. else jump, start))
                rows[f, arm_qadr] = min(solved, key=lambda s: s[:3])[3][arm_qadr]
    after, pairs, deep_after = measure(rows)
    changed = (rows[:, qadr] != reference[:, qadr]).any(1)
    moved = np.array([[lowest(row, arm[2]) for arm in arms] for row in rows])
    on_floor = held | touching
    unreached = []
    for a, side in enumerate(('left', 'right')):
        for first, last in _spans(np.flatnonzero(after[:, a] < gap - CLEAR_TOLERANCE)):
            worst = first + int(np.argmin(after[first:last + 1, a]))
            unreached.append(dict(side=side, frames=[first, last], nearest=round(float(after[worst, a]), 4),
                                  at=worst, pair=pairs[worst][a]))
    closer = lambda values: [int(v) for v in (values < gap - CLEAR_TOLERANCE).sum(0)]
    error = np.where(held, np.linalg.norm(moved - hand, axis=2), 0.)
    rise = np.where(touching, np.maximum(moved[:, :, 2] - hand[:, :, 2], 0.), 0.)
    slide = np.where(on_floor, np.linalg.norm(moved[:, :, :2] - hand[:, :, :2], axis=2), 0.)
    record = dict(gap=gap, joints=list(CLEAR_JOINTS), step=FIT_STEP, damping=FIT_DAMPING, iterations=FIT_ITERATIONS,
                  rounds=CLEAR_ROUNDS, smooth_frames=FIT_SMOOTH, tolerance=CLEAR_TOLERANCE, hold=PATCH_TOUCH,
                  touch=clip_feasibility.TOUCH,
                  frames_changed=int(changed.sum()), closer_frames_before=closer(before),
                  closer_frames_after=closer(after), nearest_before=[round(float(v), 4) for v in before.min(0)],
                  nearest_after=[round(float(v), 4) for v in after.min(0)], deep=clip_feasibility.SELF_DEEP,
                  deep_frames_before=int((deep_before > clip_feasibility.SELF_DEEP).sum()),
                  deep_frames_after=int((deep_after > clip_feasibility.SELF_DEEP).sum()),
                  depth_max_before=round(float(deep_before.max()), 4), depth_max_after=round(float(deep_after.max()), 4),
                  unreached=unreached, hand_held_frames=[int(v) for v in held.sum(0)],
                  hand_touch_frames=[int(v) for v in touching.sum(0)],
                  hand_held_error_max=[round(float(v), 4) for v in error.max(0)],
                  hand_rise_max=[round(float(v), 4) for v in rise.max(0)],
                  hand_slide_max=[round(float(v), 4) for v in slide.max(0)],
                  arm_change_max=round(float(np.abs(rows[:, qadr] - reference[:, qadr]).max()), 4))
    return rows, changed, record


def patch_clip(env, clip, hold_hand=None, feet_down=None, slow=None, waist_max=None, clear=None):
    """(clip arrays, the header's record) of a track clip (g1_getup_track.load's dict) with these changes, in this
    order, every time in the source clip's seconds:
    slow (start, end, factor): the span played factor times slower (slow_clock), the frames after it shifted;
    waist_max (ceiling, start, end): from start to end the waist pitch at most ceiling (_waist_ceiling: the torso's world
    rotation kept, its lean given to the hips, the feet, knees and hands on the floor kept), blended in and out over
    PATCH_RAMP s outside the span;
    feet_down (start, end): each foot whose lowest point is above the floor lowered onto it (PATCH_FOOT_CLEARANCE) by
    its leg's six joints, a knee on the floor kept at its height, blended in and out over PATCH_RAMP s; a foot the clip
    itself puts down is left alone (the ankles alone cannot: the ankle pitch is at its soft limit, the toes 1.4-3.6 cm
    up);
    hold_hand (side, start, held, released): from start the arm's hand (its wrist links' lowest point) kept on the floor
    by the arm's free joints alone: where the clip's hand is, until its last touch (the last frame within PATCH_TOUCH of
    the floor), then at that point, until held or the first frame where the arm no longer reaches it (PATCH_REACH);
    from there to released the arm's correction at that frame fades (smoothstep) into the clip's arm.
    The feet's and the hand's corrections are solved PATCH_ROUNDS times, smoothed over ±PATCH_SMOOTH frames between
    the solves; then clear (a gap, m): the arms moved off the legs and the trunk (clear_self), last.
    Frames no change touches keep every array of the source frame exactly."""
    import cmu_mocap
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    model, data, geoms = cmu_mocap._floor_model(env)
    dt = env.dt
    base = np.asarray(clip['qpos'], dtype=np.float64)
    count = len(base)
    clock = slow_clock(count, dt, *slow) if slow else np.arange(count, dtype=np.float64)
    t = clock * dt
    rows = clock_frames(base, clock)
    changed = clock != np.floor(clock)
    record = {}
    joint = lambda name: model.joint(name).id
    if slow:
        record['slow'] = dict(seconds=list(slow[:2]), factor=slow[2], ramp=PATCH_SLOW_RAMP,
                              frames_added=int(len(clock) - count), slowest=round(float(np.diff(clock).min()), 4))
    legs = []  # (the leg's six joints, its foot body, its knee's floor geoms) per side
    for side in ('left', 'right'):
        knee = model.body(f'{side}_knee_link').id
        legs.append(([joint(f'{side}_{j}_joint') for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch',
                                                            'ankle_roll')], model.body(f'{side}_ankle_roll_link').id,
                     [g for g in geoms if model.geom_bodyid[g] == knee]))
    arms = {}  # side: (the arm's free joints, its hand's floor geoms: the wrist links')
    for side in ('left', 'right'):
        bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(f'{side}_wrist')}
        arms[side] = ([joint(f'{side}_{j}_joint') for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                                                            'wrist_roll')], [g for g in geoms if model.geom_bodyid[g] in bodies])
    if waist_max:
        ceiling, start, end = waist_max
        waist = joint('waist_pitch_joint')
        weight = _window(t, start, end)
        waist_change, leg_change = 0., 0.
        for f in np.flatnonzero(weight > 0):
            dw, dl = _waist_ceiling(model, data, rows[f], waist, ceiling, float(weight[f]), legs, arms.values())
            changed[f] |= dw != 0.
            waist_change, leg_change = min(waist_change, dw), max(leg_change, dl)
        inside = (t >= start) & (t <= end)
        record['waist_max'] = dict(ceiling=ceiling, seconds=[start, end], ramp=PATCH_RAMP, soft=PATCH_WAIST_SOFT,
                                   waist_pitch_max=round(float(rows[inside, model.jnt_qposadr[waist]].max()), 4),
                                   waist_pitch_change_max=round(-waist_change, 4), leg_change_max=round(leg_change, 4))
    reference = rows.copy()  # after the slowing and the waist: the feet and the hand solve from it
    solves = []  # (frames, joints, goals per frame, after-solve hook)
    if feet_down:
        start, end = feet_down
        weight = _window(t, start, end)
        lowered = []
        for side, leg in zip(('left', 'right'), legs):
            bodies = {model.body(f'{side}_ankle_pitch_link').id, leg[1]}
            foot = [g for g in geoms if model.geom_bodyid[g] in bodies]
            heights = np.array([cmu_mocap.lowest_point(model, data, row, foot) for row in reference])
            knees = np.array([cmu_mocap.lowest_point(model, data, row, leg[2]) for row in reference])
            drop = weight * np.maximum(heights - PATCH_FOOT_CLEARANCE, 0.)
            # a knee near the floor (PATCH_KNEE_NEAR) is put on it by the same weight: it leaves the floor when the
            # span ends, not before
            knee_drop = np.where(knees <= PATCH_KNEE_NEAR, weight * np.maximum(knees, 0.), 0.)
            frames = np.flatnonzero((drop > 0) | (knee_drop > 0))
            goals = {}
            for f in frames:
                goals[f] = [('lowest', foot, np.array([0., 0., heights[f] - drop[f]]), [2])] + (
                    [('lowest', leg[2], np.array([0., 0., knees[f] - knee_drop[f]]), [2])]
                    if knees[f] <= PATCH_KNEE_NEAR else [])
            solves.append((frames, leg[0], goals, None))
            lowered.append(round(float(drop.max()), 4))
            changed[frames] = True
        record['feet_down'] = dict(seconds=[start, end], ramp=PATCH_RAMP, lowered_max=lowered,
                                   frames=[int(len(s[0])) for s in solves])
    if hold_hand:
        side, start, held, released = hold_hand
        arm, hand = arms[side]
        points = []
        for row in reference:
            data.qpos[:] = row
            mujoco.mj_kinematics(model, data)
            points.append(min((cmu_mocap.geom_lowest(model, data, g) for g in hand), key=lambda p: p[2]))
        points = np.array(points)
        first = int(np.flatnonzero(t >= start - 1e-9)[0])
        if points[first, 2] > PATCH_TOUCH:
            raise SystemExit(f'--hold-hand {side} {start}: в {start} с кисть клипа не на полу '
                             f'({points[first, 2] * 100:.1f} см)')
        contact = first
        while contact + 1 < len(t) and t[contact + 1] <= held and points[contact + 1, 2] <= PATCH_TOUCH:
            contact += 1
        point = np.array([*points[contact, :2], LIMB_CLEARANCE])
        frames = np.flatnonzero((t > start + 1e-9) & (t <= released + 1e-9))  # the frame at start as it is
        goals = {f: [('lowest', hand, np.array([*points[f, :2], LIMB_CLEARANCE]) if f <= contact else point,
                      [0, 1, 2])] for f in frames}
        # The hold ends where the arm no longer reaches the point (the body moves on over the hand): from there the
        # arm's correction at that frame fades (smoothstep) into the clip's arm by released, no more solves.
        # Or where holding it would move an arm joint faster than any joint of the source clip moves (near full reach
        # the solution swings): each frame solved from the one before, as the hold's solves below.
        arm_qadr = model.jnt_qposadr[arm]
        fastest = float(np.abs(np.diff(base[:, 7:], axis=0)).max() / dt)
        last = int(frames[t[frames] <= held + 1e-9][-1])
        previous = None
        for f in frames[t[frames] <= held + 1e-9]:
            row = rows[f].copy()
            if previous is not None:
                row[arm_qadr] = reference[f, arm_qadr] + previous - reference[f - 1, arm_qadr]
            error = _solve(model, data, row, arm, goals[f])
            if previous is not None and f > contact and (
                    error > PATCH_REACH or np.abs(row[arm_qadr] - rows_before[arm_qadr]).max() / dt > fastest):
                last = int(f - 1)
                break
            previous, rows_before = row[arm_qadr].copy(), row
        reach = float(t[last])
        blend = smoothstep((t - reach) / (released - reach))
        for f in frames[frames > last]:
            goals[f] = None
        held_correction = {}

        def release(f, row):
            if f == last:
                held_correction['arm'] = row[arm_qadr] - reference[f, arm_qadr]
            elif f > last:
                row[arm_qadr] = reference[f, arm_qadr] + (1. - blend[f]) * held_correction['arm']

        solves.append((frames, arm, goals, release))
        changed[frames] = True
        record['hold_hand'] = dict(side=side, seconds=[start, held, released], touch=PATCH_TOUCH,
                                   reach=PATCH_REACH, held_until=round(reach, 4),
                                   last_touch_seconds=round(float(t[contact]), 4),
                                   last_touch_source_frame=round(float(clock[contact]), 4),
                                   point=[round(float(v), 4) for v in point])
    errors = [np.zeros(len(t)) for _ in solves]
    span = np.flatnonzero(changed)
    for round_ in range(PATCH_ROUNDS):
        if round_ and len(span):
            correction = rows[:, 7:] - reference[:, 7:]
            smoothed = running_mean(correction, PATCH_SMOOTH)
            rows[span, 7:] = reference[span, 7:] + smoothed[span]
        for (frames, joints, goals, hook), error in zip(solves, errors):
            qadr = model.jnt_qposadr[joints]
            for f in frames:
                if not round_ and f - 1 in goals:  # the first solve starts from the frame before's correction
                    rows[f, qadr] = reference[f, qadr] + rows[f - 1, qadr] - reference[f - 1, qadr]
                if goals[f] is not None:
                    error[f] = _solve(model, data, rows[f], joints, goals[f])
                if hook is not None:
                    hook(f, rows[f])
    if hold_hand:
        frames = solves[-1][0]
        offset = []
        for f in frames[frames <= last]:  # the held frames: the hand's lowest point against its target
            data.qpos[:] = rows[f]
            mujoco.mj_kinematics(model, data)
            lowest = min((cmu_mocap.geom_lowest(model, data, g) for g in hand), key=lambda p: p[2])
            offset.append(float(np.linalg.norm(lowest - goals[f][0][2])))
        record['hold_hand'].update(held_error_max=round(max(offset), 4), frames=int(len(frames)))
    if feet_down:
        record['feet_down']['error_max'] = [round(float(e.max()), 4) for e in errors[:2]]
    if clear is not None:
        rows, cleared, record['clear_self'] = clear_self(env, rows, clear)
        changed |= cleared
    qvel = g1_getup_track.velocities(rows, dt)
    bodies = [model.body(name).id for name in task.TRACK_BODIES]
    body_pos, body_rot, body_lin, body_ang = g1_getup_track.kinematics(env.model, env.data, rows, qvel, bodies)
    index = np.minimum(np.floor(clock).astype(np.int64), count - 2)
    s = clock - index
    lift = {key: (1. - s) * np.asarray(clip[key])[index] + s * np.asarray(clip[key])[index + 1]
            for key in ('lift', 'raw_lift')}
    out = dict(qpos=rows, qvel=qvel, body_pos=body_pos, body_rot=body_rot, body_lin=body_lin, body_ang=body_ang,
               **lift)
    kept = np.flatnonzero(~changed)
    for key in g1_getup_track.ARRAYS:
        out[key] = np.array(out[key])
        out[key][kept] = np.asarray(clip[key])[clock[kept].astype(np.int64)]
    g1_getup_track.check(out)
    speed_base = np.abs(np.diff(base[:, 7:], axis=0)).max(0) / dt
    speed = np.abs(np.diff(rows[:, 7:], axis=0)).max(0) / dt
    names = [model.joint(j).name for j in env.joints]
    faster = {names[j]: [round(float(speed_base[j]), 2), round(float(speed[j]), 2)]
              for j in np.flatnonzero(speed > speed_base + 1e-6)}
    runs, previous = [], None
    for f in kept:
        if previous is not None and f == previous + 1:
            runs[-1][1] = int(f)
        else:
            runs.append([int(f), int(f)])
        previous = f
    record.update(frames=int(len(rows)), source_frames=int(count), unchanged_frames=runs,
                  unchanged_source_frames=[[int(clock[a]), int(clock[b])] for a, b in runs],
                  joint_speed_max=[round(float(speed_base.max()), 2), round(float(speed.max()), 2)],
                  joint_speed_faster=faster, speed_units='rad/s between 50 Hz frames: [source, patched]')
    return out, record


def cut_clip(clip, start):
    """(clip arrays, the header's record) of a track clip (g1_getup_track.load's dict) from its frame `start` on: every
    array of g1_getup_track.ARRAYS from that frame, bit for bit (qvel, the kinematics and the lift of the source's
    frames; its hold frame, the last one, unchanged), so its first frame is the source's frame `start`."""
    from humanoid_sim.tasks import g1_getup_track
    count = len(clip['qpos'])
    if not 0 < start <= count - 2:
        raise SystemExit(f'--from-frame {start}: от 1 до {count - 2} (в клипе {count} кадров, после среза не меньше 2)')
    out = {key: np.array(clip[key][start:]) for key in g1_getup_track.ARRAYS}
    g1_getup_track.check(out)
    return out, dict(from_frame=int(start), frames=int(len(out['qpos'])), source_frames=int(count))


def patched(args, env):
    """(clip arrays, header fields, name) of --patch / --base: the base clip with patch_clip's changes, or with
    --from-frame cut from that frame (cut_clip; no other change goes with it)."""
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks.config import GETUP_TRACK_NAME
    if args.from_frame is not None:
        if args.patch or not args.base or any(getattr(args, key) is not None for key in (
                'hold_hand', 'feet_down', 'slow', 'waist_max', 'clear_self')):
            raise SystemExit('--from-frame: только с --base <клип>, без других правок')
        clip = g1_getup_track.load(args.base)
        out, record = cut_clip(clip, args.from_frame)
        header = {key: value for key, value in clip['header'].items()
                  if key not in ('version', 'bodies', 'anchor', 'rate', 'seconds', 'lift_frames', 'locked_wrists',
                                 'feasibility', 'patch')}
        # The base's own patch record, if any, goes inside this one.
        header['patch'] = dict(name=None, base=args.base, base_md5=clip['md5'], options={'from_frame': args.from_frame},
                               base_patch=clip['header'].get('patch'), **record)
        name = args.name or f'{args.base}-f{args.from_frame}'
        if not re.fullmatch(GETUP_TRACK_NAME, name):
            raise SystemExit(f'имя клипа {name!r} не подходит ({GETUP_TRACK_NAME}): задайте --name')
        print(f'срез клипа {args.base} ({clip["md5"]}) с кадра {args.from_frame} ({args.from_frame * env.dt:.2f} с): '
              f'{record["source_frames"]} → {record["frames"]} кадров, массивы кадров без изменений')
        return out, header, name
    preset = PATCHES.get(args.patch, {}) if args.patch else {}
    if args.patch and not preset:
        raise SystemExit(f'--patch {args.patch}: известны {", ".join(PATCHES)}')
    base = args.base or preset.get('base')
    if not base:
        raise SystemExit('правка клипа: укажите --base <клип> или --patch <имя>')
    options = {key: getattr(args, key) if getattr(args, key) is not None else preset.get(key)
               for key in ('hold_hand', 'feet_down', 'slow', 'waist_max')}
    if options['hold_hand']:
        side, *times = options['hold_hand']
        if side not in ('left', 'right') or len(times) != 3:
            raise SystemExit('--hold-hand left|right <от> <держать до> <отпустить к>')
        options['hold_hand'] = (side, *map(float, times))
    for key in ('feet_down', 'slow', 'waist_max'):
        if options[key]:
            options[key] = tuple(map(float, options[key]))
    clear = args.clear_self if args.clear_self is not None else preset.get('clear_self')
    if clear is not None and not 0. < clear <= .05:
        raise SystemExit(f'--clear-self {clear}: зазор от 0 до 0,05 м')
    if not any(options.values()) and clear is None:
        raise SystemExit('правка клипа: ни одной правки (--hold-hand, --feet-down, --slow, --waist-max, --clear-self)')
    clip = g1_getup_track.load(base)
    out, record = patch_clip(env, clip, **options, clear=clear)
    header = {key: value for key, value in clip['header'].items()
              if key not in ('version', 'bodies', 'anchor', 'rate', 'seconds', 'lift_frames', 'locked_wrists',
                             'feasibility')}
    named = {key: list(value) if value else None for key, value in options.items()}
    if clear is not None:  # only then: the clips made before --clear-self keep their headers byte for byte
        named['clear_self'] = clear
    record = dict(name=args.patch, base=base, base_md5=clip['md5'], options=named, **record)
    header['patch'] = record
    name = args.name or preset.get('name') or f'{base}-patched'
    if not re.fullmatch(GETUP_TRACK_NAME, name):
        raise SystemExit(f'имя клипа {name!r} не подходит ({GETUP_TRACK_NAME}): задайте --name')
    print(f'правка {args.patch or "—"} клипа {base} ({clip["md5"]}): {record["source_frames"]} → {record["frames"]} '
          f'кадров; без изменений кадры ' + ', '.join(f'{a}-{b}' for a, b in record['unchanged_frames']))
    if 'slow' in record:
        print(f'   замедление {options["slow"][0]}–{options["slow"][1]} с × {options["slow"][2]:g}: +'
              f'{record["slow"]["frames_added"]} кадров')
    if 'waist_max' in record:
        w = record['waist_max']
        print(f'   пояс ≤ {w["ceiling"]} рад на {w["seconds"][0]}–{w["seconds"][1]} с: max {w["waist_pitch_max"]:.3f}, '
              f'снят до {w["waist_pitch_change_max"]:.3f} рад, суставы ног сдвинуты до {w["leg_change_max"]:.3f} рад')
    if 'feet_down' in record:
        f = record['feet_down']
        print(f'   ступни на пол на {f["seconds"][0]}–{f["seconds"][1]} с: опущены до '
              + ' / '.join(f'{v * 100:.1f}' for v in f['lowered_max']) + ' см (л. / п.), остаток до '
              + ' / '.join(f'{v * 1000:.1f}' for v in f['error_max']) + ' мм')
    if 'hold_hand' in record:
        h = record['hold_hand']
        print(f'   кисть {h["side"]} на полу с {h["seconds"][0]} с, последнее касание {h["last_touch_seconds"]:.2f} с '
              f'в точке ({h["point"][0]:.3f}, {h["point"][1]:.3f}), держится до {h["held_until"]:.2f} с (просили до '
              f'{h["seconds"][1]} с; отклонение до {h["held_error_max"] * 1000:.1f} мм), отпущена к {h["seconds"][2]} с')
    if 'clear_self' in record:
        c = record['clear_self']
        print(f'   руки от ног и корпуса на {c["gap"] * 100:g} см: изменено кадров {c["frames_changed"]}; ближе зазора (л. / '
              f'п.) {" / ".join(map(str, c["closer_frames_before"]))} → {" / ".join(map(str, c["closer_frames_after"]))} '
              f'кадров, наименьшее расстояние {" / ".join(f"{v * 100:.1f}" for v in c["nearest_before"])} → '
              f'{" / ".join(f"{v * 100:.1f}" for v in c["nearest_after"])} см; проникновение глубже '
              f'{c["deep"] * 100:g} см (любые части) {c["deep_frames_before"]} → {c["deep_frames_after"]} кадров, до '
              f'{c["depth_max_before"] * 100:.1f} → {c["depth_max_after"] * 100:.1f} см; суставы рук сдвинуты до '
              f'{c["arm_change_max"]:.3f} рад')
        print(f'   кисть на полу (л. / п.): удержана на месте {" / ".join(map(str, c["hand_held_frames"]))} кадров '
              f'(отклонение до {" / ".join(f"{v * 1000:.1f}" for v in c["hand_held_error_max"])} мм), в касании '
              f'{" / ".join(map(str, c["hand_touch_frames"]))} кадров (выше своей высоты до '
              f'{" / ".join(f"{v * 1000:.1f}" for v in c["hand_rise_max"])} мм); сдвиг по полу до '
              f'{" / ".join(f"{v * 100:.1f}" for v in c["hand_slide_max"])} см')
        for u in c['unreached']:
            print(f'   ЗАЗОР НЕ ДОСТИГНУТ: {u["side"]} рука, кадры {u["frames"][0]}-{u["frames"][1]} '
                  f'({u["frames"][0] * env.dt:.2f}–{u["frames"][1] * env.dt:.2f} с), осталось {u["nearest"] * 100:.1f} см '
                  f'(кадр {u["at"]}, {u["pair"]})')
    print(f'   скорость суставов max {record["joint_speed_max"][0]:.1f} → {record["joint_speed_max"][1]:.1f} рад/с'
          + ('; быстрее, чем в исходном: ' + ', '.join(f'{k} {v[0]:.1f} → {v[1]:.1f}' for k, v in
                                                         record['joint_speed_faster'].items())
             if record['joint_speed_faster'] else ''))
    return out, header, name


def main():
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument('--source', default=SOURCE,
                        help='файл клипа LAFAN1 в папке --data, cmu (запись CMU) или bones (BONES-SEED, --file)')
    parser.add_argument('--file', help='bones: файл CSV G1 из .tools/datasets/bones_seed/g1_getup (имя или путь)')
    parser.add_argument('--fit', choices=BONES_FITS, default=BONES_FIT,
                        help='bones: версия переноса на наш пол (v1 — как до итерации 71, клип ветки 71; v2 — шаг '
                             'ограничен, ладонь на полу, поправка сглажена); имя по умолчанию у v2 с суффиксом -v2')
    parser.add_argument('--subject', default=CMU_SUBJECT,
                        help='CMU: номер субъекта (папка .tools/datasets/cmu_mocap_<N>) или путь к папке с <N>.asf и '
                             'записями <N>_<take>.amc (по умолчанию 140)')
    parser.add_argument('--take', help='запись CMU (файл <take>.amc; 06 — то же, что <N>_06), для --source cmu; '
                                       'по умолчанию 140_09 у субъекта 140')
    parser.add_argument('--frames', type=int, nargs=2, metavar=('FROM', 'TO'),
                        help='первый и последний кадр включительно: LAFAN1 30 Гц (по умолчанию 2104 2209), CMU — '
                             'номера кадров AMC, 120 Гц (по умолчанию 61 571), bones — столбец Frame, 120 Гц (по '
                             'умолчанию весь файл)')
    parser.add_argument('--neutral', type=int, help='CMU: кадр AMC с прямой спиной (по умолчанию первый в записи)')
    parser.add_argument('--settle', type=float, default=CMU_SETTLE,
                        help='CMU и bones: секунд доводки последней позы до стойки home (0 — без неё)')
    parser.add_argument('--rounds', type=int, default=None,
                        help='CMU: проходов ИК с поправкой пределов по моментам (по умолчанию cmu_mocap.ROUNDS)')
    parser.add_argument('--name', help='имя клипа (файл runs/humanoid_lab/getup_track/<имя>.npz); по умолчанию '
                                       'lafan1-<файл>-<от>-<до> или cmu-<запись>-<от>-<до>')
    parser.add_argument('--force', action='store_true', help='перезаписать клип с тем же именем')
    parser.add_argument('--out', type=Path, help='записать клип в этот файл, а не в runs/humanoid_lab/getup_track (пробы)')
    parser.add_argument('--data', type=Path, help='папка с данными (LAFAN1 для G1 или CMU; для CMU то же, что '
                                                  '--subject <папка>)')
    parser.add_argument('--patch', help='готовая правка сделанного клипа: ' + ', '.join(PATCHES) + ' (её клип, имя и '
                                        'правки; опции ниже заменяют её значения)')
    parser.add_argument('--base', help='править сделанный клип runs/humanoid_lab/getup_track/<имя>.npz (а не делать из '
                                       'источника); время правок — в секундах этого клипа')
    parser.add_argument('--hold-hand', nargs=4, metavar=('SIDE', 'FROM', 'HELD', 'RELEASED'),
                        help='правка: кисть left|right остаётся на полу с FROM (до последнего касания там, где она в '
                             'клипе, потом в его точке; ИК только руки) до HELD, к RELEASED рука плавно возвращается '
                             'к клипу')
    parser.add_argument('--feet-down', type=float, nargs=2, metavar=('FROM', 'TO'),
                        help='правка: ступни, висящие над полом, опущены на него голеностопами (вход и выход по '
                             f'{PATCH_RAMP} с)')
    parser.add_argument('--slow', type=float, nargs=3, metavar=('FROM', 'TO', 'FACTOR'),
                        help='правка: отрезок FROM-TO с проигрывается в FACTOR раз медленнее, остальное сдвигается')
    parser.add_argument('--waist-max', type=float, nargs=3, metavar=('RAD', 'FROM', 'TO'),
                        help='правка: на FROM-TO с наклон пояса не больше RAD, наклон корпуса в мире сохранён (его '
                             'берут бёдра)')
    parser.add_argument('--clear-self', type=float, nargs='?', const=CLEAR_GAP, metavar='GAP',
                        help='правка: руки (плечо и локоть) отведены от ног и корпуса, пока зазор между их геомами '
                             f'(с капсулой кисти) не станет не меньше GAP м (по умолчанию {CLEAR_GAP}); кисть на полу '
                             'остаётся на полу')
    parser.add_argument('--from-frame', type=int, metavar='K',
                        help='правка: клип --base с его кадра K (50 Гц), массивы кадров и кадр удержания без '
                             'изменений; имя по умолчанию <base>-f<K>')
    args = parser.parse_args()
    env = getup_model_env()
    if env.model.nq != 36:
        raise SystemExit(f'модель с {env.model.nq} координатами, нужен G1 29 DoF')
    if args.patch or args.base or args.from_frame is not None:
        clip, fields, name = patched(args, env)
    else:
        frames = dict(cmu=cmu_frames, bones=bones_frames).get(args.source, lafan_frames)
        qpos, fields, name = frames(args, env)
        clip = g1_getup_track.build(env, qpos)
    save_clip(env, clip, fields, name, args.out, args.force)


def save_clip(env, clip, fields, name, out=None, force=False):
    """Writes the clip arrays with their header (version, bodies, rate, length, the source's fields, clip_feasibility's
    verdict) to runs/humanoid_lab/getup_track/<name>.npz (or out), refusing an existing file without force, prints
    the clip, its lift, the check and the md5; returns (path, md5, clip_feasibility's summary)."""
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    # What is left of the floor penetration after the lift, per frame (m, ≤ 0).
    floor = env.model.geom('floor').id
    left = []
    for row in clip['qpos']:
        env.data.qpos[:] = row
        mujoco.mj_forward(env.model, env.data)
        left.append(g1_getup_track.floor_depth(env.model, env.data, env.robot_geoms, floor))
    head = task.head_height(clip['body_pos'][:, task.TRACK_BODIES.index(task.TRACK_ANCHOR)],
                            clip['body_rot'][:, task.TRACK_BODIES.index(task.TRACK_ANCHOR)])
    checked = clip_feasibility.check_clip(clip, env)
    feasibility = checked['summary']
    over = float((np.array([r['hands'] for r in checked['rows']]).max(1) > HAND_CEILING).sum() * env.dt)
    if 'bones_fit' in fields:
        fields['bones_fit'].update(hand_ceiling=HAND_CEILING, hand_over_ceiling_seconds=round(over, 2))
    header = dict(version=g1_getup_track.TRACK_VERSION, bodies=list(task.TRACK_BODIES), anchor=task.TRACK_ANCHOR,
                  rate=1 / env.dt, seconds=round((len(clip['qpos']) - 1) * env.dt, 3),
                  lift_frames=g1_getup_track.LIFT_FRAMES, locked_wrists='home', **fields,
                  feasibility=feasibility_header(feasibility))
    path = g1_getup_track.clip_path(name) if out is None else Path(out)
    if path.exists() and not force:
        raise SystemExit(f'{path} уже есть: этапы ссылаются на клип по md5, новый — под другим --name (или --force)')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.pending-{path.name}')
    with open(temporary, 'wb') as out:
        np.savez(out, **clip, header=np.array(json.dumps(header)))
    os.replace(temporary, path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    g1_getup_track.load(name, digest, path=path)
    first, last = fields['frames']
    print(f'{path}: {fields["source"]} кадры {first}-{last} → {len(clip["qpos"])} кадров 50 Гц ({header["seconds"]} с); '
          f'таз {clip["qpos"][0, 2]:.2f} → {clip["qpos"][-1, 2]:.2f} м, макушка {head[0]:.2f} → {head[-1]:.2f} м')
    print(f'подъём из пола: проникание до {clip["raw_lift"].max() * 100:.1f} см (медиана '
          f'{np.median(clip["raw_lift"]) * 100:.1f}), подъём после сглаживания до {clip["lift"].max() * 100:.1f} см; '
          f'осталось проникания до {abs(min(left)) * 100:.1f} см')
    print(f'выполнимость (clip_feasibility): {clip_feasibility.verdict_text(feasibility)}')
    print(f'без правила моментов пояса: {clip_feasibility.verdict_text(feasibility, waist_free=True)}')
    if over > 0:
        print(f'ВНИМАНИЕ: нагрузка на одну кисть до {feasibility["hand_max"]:.0f} Н, выше потолка {HAND_CEILING:.0f} Н '
              f'{over:.2f} с (clip_feasibility): опора на кисть тяжелее, чем выдержит Dex3-1')
    print(f'md5 {digest}')
    print(json.dumps({'track_clip': {'file': name, 'md5': digest}}))
    return path, digest, feasibility


if __name__ == '__main__':
    main()
