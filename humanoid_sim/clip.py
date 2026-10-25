"""Pre-rendered clips of stages side by side for the journal («Дневник») and the site, watchable without a live race.

    .venv\\Scripts\\python.exe -m humanoid_sim.clip exp-…[:steps] … [--titles "было,стало"] [--scenario walk]
        [--focus body|torso|arms|hands|feet] [--view side|front|three_quarter|top] [--seconds S]
        [--pose NAME|random] [--draw N] [--amplitude A] [--load KG] [--start N] [--lang ru|en|zh] [--size 640x480]
        --out c.webp

Each lane is the same deterministic CPU episode as `supervise` and `collage` (seed 73000, the scenario's fixed
commands), filmed by a camera following the pelvis. Lanes are tiles side by side, at most four per row, labelled with
their title, time and the scenario's measure (metres forward, back or sideways, degrees turned, the pelvis displacement
after a push, the arms' error and sag, the head height); a fallen robot's tile freezes, greyed, with the fall time.

Scenarios (CLIPS):
  walking   the evaluation ones (stand, slow, walk, brisk, stop_start, sprint) and clip-only ones (back, side_left /
            side_right, turn_left / turn_right, walk_turn, side_stop, and force pushes push_back / push_forward /
            push_left / push_right / push_walk as G1VelocityEnv.set_force_push gives them, marked in the tile for
            MARK_SECONDS); side_left_fast / side_right_fast, side steps at 1 m/s filmed on the feet from the front, the
            footer giving the smallest gap so far between the feet's capsules (Unitree's, g1_velocity.FOOT_CAPSULES,
            placed on the ankle_roll links with or without task_config foot_collision; negative: they overlap).
  held pose any 29-DoF walking stage: arm_pose_stand / arm_pose_walk (0.5 m/s), the arms held from 1 s on in one
            pose as training's arm_perturb segments hold them (set_arm_perturb, renewed every control step; the
            policy's arm actions unused): a named pose (--pose, export_g1.NAMED_ARM_POSES) or random (the default),
            U(±A rad) around home per arm joint over the action scale, numpy default_rng(--draw), A = --amplitude
            (1 rad), as scripts/lab/probes/arm_probe.py; the footer gives the displacement (standing) or metres
            forward and the heading change.
  height    stages with task_config height_command: crouch_stand (c 0.78, 0.64 from 2 s, 0.78 from 6 s) and
            crouch_walk (0.5 m/s, c 0.70 from 1 s), set_height; the footer gives the pelvis height and c.
  arms      stages with task_config arm_targets: arms_stand (extended), arms_push (extended, pushed forward 1 m/s at
            4 s), arms_walk (carry_mid at 0.5 m/s) and arms_poses (carry_low, carry_mid, extended, carry_wide,
            arms_side). The arms follow targets from the reset (flag 1, home), each named pose of
            export_g1.NAMED_ARM_POSES in the training's quintic; --pose replaces the pose of a one-pose scenario,
            --load puts kg on each hand (set_hand_load). A translucent green ball marks where each palm would be in
            the target pose (the targets' forward kinematics on the body as it stands), an amber block the load in the
            palm; the footer gives the arms' mean error over the ten free joints and the shoulder pitch sag, q − target
            (rad, positive: the arm hangs lower than asked).
  getup     task g1_getup: getup_back / getup_front / getup_side start from a held-out row of the start bank's class
            (--start: the class's n-th, 0 first), without the assist and at the lowest PD bound (the task's
            evaluation options), no delay, the motor model as measured and the files' friction (getup_probe's ours),
            8 s; the footer gives the head height and when it got up (the task's success pose within 6 s).
Camera (--focus): body (the default: the pelvis from 2.5 m, as before; getting up the centre of mass, from further
off), or closer on the centre of the torso, the arms (shoulders, elbows, palms), the hands (palms) or the feet
(ankle_roll links), following the robot with the focus's offset from it smoothed over FOCUS_SECONDS, so a swinging
foot or hand does not shake the picture. --view: side (from the robot's right, forward to the right), front,
three_quarter (front right) or top.
Texts in Russian (the default, decimal comma), English or Chinese; the overlays scale with the tile. Saved as an
animated WebP (browsers play it in an <img>, looping) plus a JPEG poster frame.
"""
import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .checkpoint_schema import checkpoint_identity
from .collage import _font
from .core import DEFAULT_RUN, load_stage, read_json, stage_task_config
from .evaluation.scenarios import SCENARIOS, Scenario
from .lab import Laboratory
from .supervise import find_stage, rollout

TILE = (320, 270)
PER_ROW = 4
FPS = 25
MARK_SECONDS = .6  # a push stays marked in the tile this long from its start


# Azimuth per view (deg): side looks along +y (forward is to the right); front faces the robot; three_quarter looks at
# it from the front right; top looks down (TOP_ELEVATION) with forward to the right, as side.
VIEWS = {'side': 90., 'front': 180., 'three_quarter': 135., 'top': 90.}
TOP_ELEVATION = -89.  # -90 leaves the azimuth undefined
# Camera per focus: the bodies whose centre it looks at (a wrist_yaw link at its palm, HAND_LOAD_POSITION), distance
# (m) and elevation (deg). body is the camera of the walking clips before foci: the pelvis from 2.8 × the playback
# target height, looking at 0.7 of it from -8°; a getting-up lane looks at the centre of mass from GETUP_CAMERA.
FOCI = {
    'body': ((), None, None),
    'torso': (('torso_link',), 1.6, -10.),
    'arms': (('left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_elbow_link', 'right_elbow_link',
              'left_wrist_yaw_link', 'right_wrist_yaw_link'), 1.5, -12.),
    'hands': (('left_wrist_yaw_link', 'right_wrist_yaw_link'), 1., -15.),
    'feet': (('left_ankle_roll_link', 'right_ankle_roll_link'), 1.1, -20.),
}
FOCUS_SECONDS = .4  # time constant of the smoothing of the focus's offset from the robot
GETUP_CAMERA = (.55, 3., -12.)  # look-at height (m), distance (m), elevation (deg)
HAND_LOAD_POSITION = (.11, 0., 0.)  # the palm in the wrist_yaw frame: g1_tensor_math.HAND_LOAD_POSITION (test)
# Arms scenarios' marks: the target palm's ball (radius m, RGBA) and the load's block (half side m per kg^(1/3), RGBA).
TARGET_MARK = (.04, (.2, .9, .35, .45))
LOAD_MARK = (.025, (.95, .55, .1, 1.))
LOAD_LIMIT = 5.  # kg per hand

# Tile texts by language. The measure (the footer's number) is positive in the named direction. Russian is the
# original, decimal comma; Chinese follows the site's catalogs and docs/i18n-glossary.md (units as symbols, 跌倒).
TEXTS = {
    'ru': dict(time='{t:4.1f} с', fell='упал на {t:.1f} с', forward='вперёд {v:+.1f} м', back='назад {v:+.1f} м',
               left='влево {v:+.1f} м', right='вправо {v:+.1f} м', turn_left='поворот влево {v:+.0f}°',
               turn_right='поворот вправо {v:+.0f}°', push='смещение {v:.2f} м', mark='толчок {d} {v} м/с',
               million=' млн', arms='ошибка рук {v[0]:.2f} · провис плеча {v[1]:+.2f} рад',
               getup='голова {v:.2f} м', getup_up='встал за {t:.1f} с · голова {v:.2f} м',
               cutoff='отсечка на {t:.1f} с', load='груз {v} кг на кисть',
               creep='смещение {v[0]:.2f} м · курс {v[1]:+.0f}°', walk_heading='вперёд {v[0]:+.1f} м · курс {v[1]:+.0f}°',
               height='таз {v[0]:.2f} м · команда {v[1]:.2f} м',
               left_gap='влево {v[0]:+.1f} м · наименьший зазор стоп {v[1]:+.1f} см',
               right_gap='вправо {v[0]:+.1f} м · наименьший зазор стоп {v[1]:+.1f} см',
               pose='навязанная поза рук: {v}', random='навязанная поза рук ±{v} рад'),
    'en': dict(time='{t:4.1f} s', fell='fell at {t:.1f} s', forward='forward {v:+.1f} m', back='back {v:+.1f} m',
               left='left {v:+.1f} m', right='right {v:+.1f} m', turn_left='turned left {v:+.0f}°',
               turn_right='turned right {v:+.0f}°', push='displaced {v:.2f} m', mark='push {d} {v} m/s',
               million=' M', arms='arm error {v[0]:.2f} · shoulder sag {v[1]:+.2f} rad',
               getup='head {v:.2f} m', getup_up='up in {t:.1f} s · head {v:.2f} m',
               cutoff='cut off at {t:.1f} s', load='load {v} kg per hand',
               creep='displaced {v[0]:.2f} m · heading {v[1]:+.0f}°',
               walk_heading='forward {v[0]:+.1f} m · heading {v[1]:+.0f}°',
               height='pelvis {v[0]:.2f} m · command {v[1]:.2f} m',
               left_gap='left {v[0]:+.1f} m · smallest foot gap {v[1]:+.1f} cm',
               right_gap='right {v[0]:+.1f} m · smallest foot gap {v[1]:+.1f} cm',
               pose='forced arm pose: {v}', random='forced arm pose ±{v} rad'),
    'zh': dict(time='{t:4.1f} s', fell='在 {t:.1f} s 时跌倒', forward='前进 {v:+.1f} m', back='后退 {v:+.1f} m',
               left='向左 {v:+.1f} m', right='向右 {v:+.1f} m', turn_left='左转 {v:+.0f}°', turn_right='右转 {v:+.0f}°',
               push='位移 {v:.2f} m', mark='{d}推 {v} m/s', million=' M',
               arms='手臂误差 {v[0]:.2f} · 肩部下垂 {v[1]:+.2f} rad', getup='头部 {v:.2f} m',
               getup_up='{t:.1f} s 内起身 · 头部 {v:.2f} m', cutoff='在 {t:.1f} s 时中止', load='每只手负载 {v} kg',
               creep='位移 {v[0]:.2f} m · 航向 {v[1]:+.0f}°', walk_heading='前进 {v[0]:+.1f} m · 航向 {v[1]:+.0f}°',
               height='骨盆 {v[0]:.2f} m · 指令 {v[1]:.2f} m',
               left_gap='向左 {v[0]:+.1f} m · 双脚最小间隙 {v[1]:+.1f} cm',
               right_gap='向右 {v[0]:+.1f} m · 双脚最小间隙 {v[1]:+.1f} cm',
               pose='强制手臂姿态：{v}', random='强制手臂姿态 ±{v} rad'),
}
DIRECTIONS = {'ru': dict(forward='вперёд', back='назад', left='влево', right='вправо'),
              'en': dict(forward='forward', back='back', left='left', right='right'),
              'zh': dict(forward='向前', back='向后', left='向左', right='向右')}
# collage's font (Segoe UI) has no CJK glyphs; Chinese tiles use Microsoft YaHei (or another Windows CJK font).
CJK_FONTS = ('msyh.ttc', 'msyh.ttf', 'simhei.ttf', 'simsun.ttc')


@dataclass(frozen=True)
class ClipScenario:
    """A filmed scenario: its commands (an evaluation Scenario, so supervise.rollout plays it), the footer's measure
    (a TEXTS key) and an optional force push (seconds from the start, (forward, left) Δv m/s in the pelvis heading
    frame, duration s), as G1VelocityEnv.set_force_push. Arms scenarios (task_config arm_targets): arms, ((seconds,
    pose name of export_g1.NAMED_ARM_POSES), ...) the arms move to from the reset on, and load, kg on each hand.
    Getting-up scenarios (task g1_getup): getup, the start bank's class, and start, which of its held-out rows. Held
    poses: held, (seconds from which, a pose name or 'random'), with draw and amplitude (rad) of a random one. Height:
    heights, ((seconds, c m), ...). focus, view and seconds are the clip's defaults (FOCI, VIEWS, length)."""
    scenario: Scenario
    measure: str = 'forward'
    push: tuple = None
    arms: tuple = None
    load: float = 0.
    getup: str = None
    start: int = 0
    focus: str = 'body'
    view: str = 'side'
    seconds: float = 10.
    held: tuple = None
    draw: int = 0
    amplitude: float = 1.
    heights: tuple = None

    @property
    def episode(self):
        """Whether clip._play films it (arms, getting up, held poses, height), not supervise.rollout."""
        return bool(self.arms or self.getup or self.held or self.heights)


def _clip(name, schedule, measure, push=None):
    return ClipScenario(Scenario(name, schedule), measure, push)


def _arms(name, schedule, poses, push=None, seconds=10.):
    """An arms scenario: flag 1 at home from the reset, then `poses` ((seconds, name), ...)."""
    return ClipScenario(Scenario(name, schedule), 'arms', push, arms=((0., 'home'), *poses), focus='arms',
                        seconds=seconds)


STAND = (0., 0., 0.)
PUSH = 1.25  # m/s: the acceptance probes' and the sim2sim push patch's Δv (force_push's working value), for 0.1 s
ARM_PUSH = 1.  # m/s forward for 0.1 s: carry_probe's push of a robot with its arms extended
HELD_FROM = 1.  # s: held pose scenarios hold the arms from then on
HELD_AMPLITUDE = 2.  # rad: the largest amplitude of a random held pose (arm_probe.py's bound)
# Heading hold steers only a zero yaw rate, so the turn commands pass unchanged; the sideways ones and the walks keep
# the start heading, and a stand command stays zero (stand_heading_hold false). Pushes come mid-segment: a new command
# ends a running push.
CLIPS = {name: ClipScenario(scenario) for name, scenario in SCENARIOS.items()}
CLIPS.update((clip.scenario.name, clip) for clip in (
    _clip('back', ((0., (-.5, 0., 0.)),), 'back'),
    _clip('side_left', ((0., (0., .3, 0.)),), 'left'),
    _clip('side_right', ((0., (0., -.3, 0.)),), 'right'),
    _clip('turn_left', ((0., (0., 0., .5)),), 'turn_left'),
    _clip('turn_right', ((0., (0., 0., -.5)),), 'turn_right'),
    _clip('walk_turn', ((0., (.5, 0., .3)),), 'turn_left'),
    _clip('side_stop', ((0., (0., .3, 0.)), (4., STAND)), 'left'),
    _clip('push_back', ((0., STAND),), 'push', (2., (-PUSH, 0.), .1)),
    _clip('push_forward', ((0., STAND),), 'push', (2., (PUSH, 0.), .1)),
    _clip('push_left', ((0., STAND),), 'push', (2., (0., PUSH), .1)),
    _clip('push_right', ((0., STAND),), 'push', (2., (0., -PUSH), .1)),
    _clip('push_walk', ((0., (.5, 0., 0.)),), 'left', (3., (0., 1.), .1)),
    _arms('arms_stand', ((0., STAND),), ((.5, 'extended'),)),
    _arms('arms_push', ((0., STAND),), ((.5, 'extended'),), (4., (ARM_PUSH, 0.), .1)),
    _arms('arms_walk', ((0., (.5, 0., 0.)),), ((.5, 'carry_mid'),)),
    _arms('arms_poses', ((0., STAND),), ((.5, 'carry_low'), (2.5, 'carry_mid'), (4.5, 'extended'),
                                         (6.5, 'carry_wide'), (8.5, 'arms_side')), seconds=11.),
    *(ClipScenario(Scenario(f'getup_{kind}', ((0., STAND),)), 'getup', getup=kind, seconds=8.)
      for kind in ('back', 'front', 'side')),
    ClipScenario(Scenario('side_left_fast', ((0., (0., 1., 0.)),)), 'left_gap', focus='feet', view='front'),
    ClipScenario(Scenario('side_right_fast', ((0., (0., -1., 0.)),)), 'right_gap', focus='feet', view='front'),
    ClipScenario(Scenario('arm_pose_stand', ((0., STAND),)), 'creep', held=(HELD_FROM, 'random')),
    ClipScenario(Scenario('arm_pose_walk', ((0., (.5, 0., 0.)),)), 'walk_heading', held=(HELD_FROM, 'random')),
    ClipScenario(Scenario('crouch_stand', ((0., STAND),)), 'height', heights=((0., .78), (2., .64), (6., .78))),
    ClipScenario(Scenario('crouch_walk', ((0., (.5, 0., 0.)),)), 'height', heights=((0., .78), (1., .70))),
))


def get_clip(name):
    if name not in CLIPS:
        raise ValueError(f'Неизвестный сценарий ролика: {name}')
    return CLIPS[name]


def configure(clip, load=None, pose=None, start=None, draw=None, amplitude=None):
    """The scenario with the command line's changes (None: the scenario's own): load, kg on each hand, and pose, the
    one named pose of a one-pose arms scenario or a held pose scenario's pose (there also 'random'), with draw and
    amplitude (rad) of a random held pose; start, which held-out row of a getting-up scenario's class."""
    arms = [name for name, other in CLIPS.items() if other.arms or other.held]
    if load is not None and not clip.arms:
        raise ValueError(f'Груз на кистях — только в сценариях рук: {", ".join(n for n in arms if CLIPS[n].arms)}')
    if pose is not None and not (clip.arms or clip.held):
        raise ValueError(f'Поза рук — только в сценариях рук: {", ".join(arms)}')
    if (draw is not None or amplitude is not None) and not (clip.held and (pose or clip.held[1]) == 'random'):
        raise ValueError('Вариант и размах случайной позы — только для случайной навязанной позы (arm_pose_*)')
    if load is not None:
        if not (math.isfinite(load) and 0 <= load <= LOAD_LIMIT):
            raise ValueError(f'Груз на кисть — от 0 до {LOAD_LIMIT:g} кг')
        clip = replace(clip, load=float(load))
    if pose is not None:
        from .export_g1 import NAMED_ARM_POSES
        names = [name for name, _, _ in NAMED_ARM_POSES] + (['random'] if clip.held else [])
        if pose not in names:
            raise ValueError(f'Неизвестная поза рук: {pose} (есть {", ".join(names)})')
        if clip.held:
            clip = replace(clip, held=(clip.held[0], pose))
        elif len(clip.arms) != 2:  # home, then the one pose
            raise ValueError(f'В сценарии {clip.scenario.name} несколько поз: --pose только для сценариев с одной позой')
        else:
            clip = replace(clip, arms=(clip.arms[0], (clip.arms[1][0], pose)))
    if draw is not None:
        if isinstance(draw, bool) or not isinstance(draw, int) or draw < 0:
            raise ValueError('Вариант случайной позы — целое число от 0')
        clip = replace(clip, draw=draw)
    if amplitude is not None:
        if not (math.isfinite(amplitude) and 0 < amplitude <= HELD_AMPLITUDE):
            raise ValueError(f'Размах случайной позы — больше 0 и до {HELD_AMPLITUDE:g} рад')
        clip = replace(clip, amplitude=float(amplitude))
    if start is not None:
        if not clip.getup:
            raise ValueError('Номер старта — только в сценариях вставания (getup_*)')
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError('Номер старта — целое число от 0')
        clip = replace(clip, start=start)
    return clip


def _tile_font(size, lang):
    if lang == 'zh':
        for name in CJK_FONTS:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    return _font(size)


def _measure(kind, focus, turned):
    """The footer's number, positive in the named direction: m along x or y, degrees turned, m from the start."""
    return {'forward': focus[0], 'back': -focus[0], 'left': focus[1], 'right': -focus[1],
            'turn_left': math.degrees(turned), 'turn_right': -math.degrees(turned),
            'push': math.hypot(focus[0], focus[1])}[kind]


def _keep_push(raw):
    """supervise.rollout sets the scenario's command before every control step, and G1VelocityEnv.set_command ends a
    running push: from now on an unchanged command lets it run on (a new command still ends it)."""
    set_command = raw.set_command

    def keep(command):
        running = (raw.push_left, raw.push_force) if np.array_equal(command, raw.command) else None
        set_command(command)
        if running is not None:
            raw.push_left, raw.push_force = running
    raw.set_command = keep


def _arrow(force, view):
    """→ or ← when the push runs across the picture (side view: forward or back), '' along the camera axis."""
    azimuth = math.radians(VIEWS[view])
    across = (force[0] * math.sin(azimuth) - force[1] * math.cos(azimuth)) / max(math.hypot(force[0], force[1]), 1e-9)
    return '→' if across > .7 else '←' if across < -.7 else ''


def _body(model, name):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        raise ValueError(f'У модели нет тела {name}')
    return body


def _palm(data, body):
    """A hand's palm in the world: HAND_LOAD_POSITION in its wrist_yaw link's frame."""
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ HAND_LOAD_POSITION


class _Aim:
    """Where a lane's camera looks for a focus of FOCI other than the walking body: the centre of the focus points (a
    getting-up lane's body: the robot's centre of mass at GETUP_CAMERA's height), following the robot (the pelvis;
    getting up, the centre of mass) with the focus's offset from it smoothed over FOCUS_SECONDS."""

    def __init__(self, model, focus, getup, fps):
        names = FOCI[focus][0]
        try:
            self.bodies = [_body(model, name) for name in names]
        except ValueError as error:
            raise ValueError(f'Фокус {focus} не для этого робота: {error}') from None
        self.palms = [name.endswith('wrist_yaw_link') for name in names]
        self.pelvis = _body(model, 'pelvis')
        self.getup = getup
        self.blend = 1. - math.exp(-1. / (fps * FOCUS_SECONDS))
        self.offset = None

    def __call__(self, data):
        anchor = (data.subtree_com if self.getup else data.xpos)[self.pelvis]
        if not self.bodies:
            return anchor[0], anchor[1], GETUP_CAMERA[0]
        centre = np.mean([_palm(data, body) if palm else data.xpos[body]
                          for body, palm in zip(self.bodies, self.palms)], axis=0)
        offset = centre - (anchor[0], anchor[1], 0.)
        self.offset = offset if self.offset is None else self.offset + self.blend * (offset - self.offset)
        return anchor[0] + self.offset[0], anchor[1] + self.offset[1], self.offset[2]


def _arm_measure(raw, shoulders):
    """(the arms' mean |q − target| over the ten free joints, the shoulder pitch sag: mean q − target of both), rad,
    of the control step just taken (its targets: home moved by the offsets it followed); NaN with free arms."""
    if not raw.previous_flag:
        return math.nan, math.nan
    error = raw.data.qpos[raw.qadr] - raw.home - raw.previous_offset
    return float(np.abs(error[list(raw.track_joints)]).mean()), float(error[shoulders].mean())


def _add_geom(scene, kind, size, position, rotation, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, dtype=np.float64),
                        np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _arm_marks(scene, raw, scratch, hands):
    """The arms scenarios' marks in a rendered scene: with arm targets a translucent ball where each palm would be in
    the target pose (the targets of the step just taken on the body as it stands: forward kinematics in `scratch`),
    and a block in each loaded palm, its side growing with the load's cube root."""
    if raw.previous_flag:
        joints = list(raw.track_joints)
        scratch.qpos[:] = raw.data.qpos
        scratch.qpos[raw.qadr[joints]] = (raw.home + raw.previous_offset)[joints]
        mujoco.mj_kinematics(raw.model, scratch)
        for body in hands:
            _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (TARGET_MARK[0], 0., 0.), _palm(scratch, body), np.eye(3),
                      TARGET_MARK[1])
    for load, body in zip(raw.hand_load, hands):
        if load > 0:
            _add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [LOAD_MARK[0] * load ** (1 / 3)] * 3, _palm(raw.data, body),
                      raw.data.xmat[body], LOAD_MARK[1])


def getup_start(bank, kind, index=0):
    """The bank row of the index-th held-out start of class `kind` (g1_getup_bank.load's bank)."""
    if kind not in bank['classes']:
        raise ValueError(f'В банке стартов нет класса {kind}')
    rows = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index(kind)))
    if not 0 <= index < len(rows):
        raise ValueError(f'В банке стартов {len(rows)} отложенных строк класса {kind}, номер {index} вне их')
    return int(rows[index])


def _offsets(offsets, joints):
    """(1, nu) float64 tensor of the offsets on `joints`, 0 elsewhere (task.arm_move_steps' start and end)."""
    import torch
    out = torch.zeros((1, len(offsets)), dtype=torch.float64)
    out[0, joints] = torch.as_tensor(np.asarray(offsets, dtype=np.float64)[joints])
    return out


def _held_action(raw, clip):
    """The action a held pose scenario holds the arm joints at (the rest 0, unused): a named pose's angles from home
    over the action scale, or a random pose as arm_probe.py draws it, U(±amplitude) rad per arm joint over the scale
    from numpy default_rng(draw). set_arm_perturb clips it to the action box, pd_target to the joint ranges."""
    from .export_g1 import named_arm_poses
    arms = list(raw.arm_joints)
    if not arms:
        raise ValueError('Навязанная поза рук — для G1 с руками (29 DoF)')
    held = np.zeros(raw.action_space.shape)
    if clip.held[1] == 'random':
        held[arms] = np.random.default_rng(clip.draw).uniform(-1, 1, len(arms)) * clip.amplitude / raw.scale[arms]
    else:
        q = dict((name, q) for name, _, q in named_arm_poses(raw))[clip.held[1]]
        held[arms] = (q - raw.home)[arms] / raw.scale[arms]
    return held


# Unitree's foot capsules (g1_velocity.FOOT_CAPSULES: per ankle_roll link, ends in its frame), radius m.
def _foot_capsules():
    from .tasks.g1_velocity import FOOT_CAPSULE_RADIUS, FOOT_CAPSULES
    return ({name: np.array(capsules, dtype=float).reshape(len(capsules), 2, 3)
             for name, capsules in FOOT_CAPSULES.items()}, FOOT_CAPSULE_RADIUS)


def _segment_distance(p1, q1, p2, q2):
    """Closest distance between the segments p1-q1 and p2-q2 (arrays broadcasting to (..., 3), no zero-length
    segment), Ericson's ClosestPtSegmentSegment (Real-Time Collision Detection, 5.1.9), as foot_overlap.py."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    denom = a * e - b * b
    general = denom > 1e-12 * a * e  # not parallel: the closest point of the infinite lines, clamped
    s = np.where(general, np.clip((b * f - c * e) / np.where(general, denom, 1.), 0., 1.), 0.)
    t = (b * s + f) / e
    s = np.where(t < 0., np.clip(-c / a, 0., 1.), np.where(t > 1., np.clip((b - c) / a, 0., 1.), s))
    t = np.clip(t, 0., 1.)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def feet_gap(model, data, capsules):
    """Smallest distance (m) between the left and the right foot's capsules (_foot_capsules) placed on the ankle_roll
    links as they stand; < 0: they overlap."""
    local, radius = capsules
    ends = []
    for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        body = _body(model, name)
        ends.append(data.xpos[body] + local[name] @ data.xmat[body].reshape(3, 3).T)
    left, right = ends
    distance = _segment_distance(left[:, None, 0], left[:, None, 1], right[None, :, 0], right[None, :, 1])
    return float(distance.min() - 2 * radius)


def _play(stage, identity, policy, norm, clip, seed, seconds, on_step):
    """One deterministic CPU episode of an arms, getting-up, held pose or height scenario (supervise.rollout plays
    the others), with on_step(seconds, raw env, None, ended) after every control step.

    Arms: the walking reset with the scenario's command, the hand loads, then from the first step the arms following
    targets (flag 1), each named pose at its time in the training's quintic (task.arm_move_steps), the history filled
    with the first frame as a training reset with targets has it. Getting up: the held-out start of the scenario's
    class with the task's evaluation options (no assist, the lowest PD bound), no delay, the motor model as measured
    and the files' friction. Held pose: the walking reset, the arms held by a one-step perturbation segment renewed
    before every control step from its time on (set_command, called every step as in supervise.rollout, ends a
    segment). Height: the walking reset, set_height at each time."""
    from .export_g1 import named_arm_poses
    from .tasks import g1_tensor_math as task
    from .tasks.catalog import make_task, resolve_task
    dt = identity['control_dt']
    limit = math.ceil(seconds / dt)
    env = make_task(identity['task_id'], identity['robot_id'], {**stage_task_config(stage), 'max_episode_steps': limit})
    try:
        raw = env.unwrapped
        moves, heights, held = {}, {}, None
        if clip.getup:
            options = dict(resolve_task(identity['robot_id'], identity['task_id']).evaluation_options or {})
            options.update(start=getup_start(raw.bank, clip.getup, clip.start), action_delay_steps=0,
                           motor_friction_scale=1.)
            obs, _ = env.reset(seed=seed, options=options)
            raw.model.geom_friction[:, 0] = raw.nominal_friction
        else:
            obs, _ = env.reset(seed=seed, options={'command': clip.scenario.command_at(0.)})
        if clip.arms:
            raw.set_hand_load(clip.load)
            poses = {name: q for name, _, q in named_arm_poses(raw)}
            moves = {round(at / dt): poses[name] for at, name in clip.arms}
        if clip.held:
            held, held_from = _held_action(raw, clip), round(clip.held[0] / dt)
        if clip.heights:
            heights = {round(at / dt): height for at, height in clip.heights}
        joints = list(raw.track_joints)
        for step in range(limit):
            if not clip.getup:
                raw.set_command(clip.scenario.command_at(step * dt))
            if step in moves:
                end = _offsets(moves[step] - raw.home, joints)
                start = _offsets(raw.arm_offset if raw.arm_flag else np.zeros(raw.model.nu), joints)
                steps = int(task.arm_move_steps(start, end, joints, dt)[0])
                raw.set_arm_targets(moves[step], 1, seconds=steps * dt if step else 0.)
            if step in heights:
                raw.set_height(heights[step])
            if held is not None and step >= held_from:
                raw.set_arm_perturb(1, held)
            if not clip.getup:
                obs = raw._observation('fill' if step == 0 and clip.arms else 'replace')
            action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
            obs, _, terminated, truncated, _ = env.step(action)
            on_step((step + 1) * dt, raw, None, terminated)
            if terminated or truncated:
                break
    finally:
        env.close()


def _check_stage(clip, identity, stage):
    """Getting-up scenarios film g1_getup stages and only them; arms scenarios need task_config arm_targets, height
    scenarios height_command."""
    getup = identity['task_id'] == 'g1_getup'
    if bool(clip.getup) != getup:
        raise ValueError(f'Этапы задачи g1_getup снимают сценарии вставания (getup_*), другие этапы — остальные '
                         f'сценарии: {clip.scenario.name}, {stage}')
    if clip.arms and not stage_task_config(stage).get('arm_targets'):
        raise ValueError(f'Сценарий рук {clip.scenario.name} — для этапов с task_config arm_targets: {stage}')
    if clip.heights and not stage_task_config(stage).get('height_command'):
        raise ValueError(f'Сценарий высоты {clip.scenario.name} — для этапов с task_config height_command: {stage}')


def film_lane(stage, scenario, seconds=10., fps=FPS, size=TILE, view=None, focus=None):
    """[(rgb frame, seconds, measure, fell, push mark, tag)] at `fps` for one episode, padded with the last frame.

    scenario: a ClipScenario, or an evaluation Scenario (measured forward). The push mark is ((forward, left) Δv,
    on-screen arrow) for MARK_SECONDS from the push start, else None; the tag (tag_label) is ('load', kg per hand),
    ('pose', name) or ('random', amplitude) while the arms are held, else None. view and focus: None, the scenario's.
    The measure is a number, for arms (error, sag), getting up (head height, seconds to get up or None), held poses
    (displacement or metres forward, heading change in degrees), height (pelvis height, c) and fast side steps
    (metres sideways, the smallest feet gap so far in cm); fell is a getting-up lane's cut-off."""
    from .tasks.catalog import resolve_task
    clip = scenario if isinstance(scenario, ClipScenario) else ClipScenario(scenario)
    view, focus = view or clip.view, focus or clip.focus
    if view not in VIEWS or focus not in FOCI:
        raise ValueError(f'Неизвестный вид или фокус камеры: {view}, {focus}')
    identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
    _check_stage(clip, identity, stage)
    playback = resolve_task(identity['robot_id'], identity['task_id']).playback
    every = max(1, round(1 / (fps * identity['control_dt'])))
    push_step = None if clip.push is None else round(clip.push[0] / identity['control_dt'])
    marked = round(MARK_SECONDS / identity['control_dt'])
    policy, norm = load_stage(stage)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.8 * playback.target[2], -8., VIEWS[view]
    walking_body = focus == 'body' and not clip.getup  # the camera of the walking clips before foci, bit for bit
    if not walking_body:
        camera.distance, camera.elevation = GETUP_CAMERA[1:] if focus == 'body' else FOCI[focus][1:]
    if view == 'top':
        camera.elevation = TOP_ELEVATION
    state = {'renderer': None, 'step': 0, 'yaw': 0., 'turned': 0., 'mark': None, 'gap': math.inf}
    held_step = None if clip.held is None else round(clip.held[0] / identity['control_dt'])
    capsules = _foot_capsules() if clip.measure in ('left_gap', 'right_gap') else None
    frames = []

    def on_step(now, raw, lean, fell):
        state['step'] += 1
        if clip.measure in ('turn_left', 'turn_right', 'creep', 'walk_heading'):  # unwrapped heading change
            rotation = raw.data.xmat[mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, playback.follow_body)]
            yaw = math.atan2(rotation[3], rotation[0])
            state['turned'] += math.remainder(yaw - state['yaw'], math.tau)
            state['yaw'] = yaw
        if capsules is not None:  # every control step, so the smallest gap is not only the filmed frames'
            state['gap'] = min(state['gap'], feet_gap(raw.model, raw.data, capsules))
        if state['step'] == push_step:  # pushes from the next control step on
            _keep_push(raw)
            raw.set_force_push(clip.push[1], clip.push[2])
            state['mark'] = (clip.push[1], _arrow(raw.push_force, view))
        if state['step'] % every and not fell:
            return
        if state['renderer'] is None:
            visual = raw.model.vis.global_  # the offscreen buffer is 640x480 unless the model asks for more
            visual.offwidth, visual.offheight = max(visual.offwidth, size[0]), max(visual.offheight, size[1])
            state['renderer'] = mujoco.Renderer(raw.model, height=size[1], width=size[0])
            if not walking_body:
                state['aim'] = _Aim(raw.model, focus, bool(clip.getup), fps)
            if clip.arms:
                names = [raw.model.joint(joint).name for joint in raw.joints]
                state['shoulders'] = [names.index(f'{side}_shoulder_pitch_joint') for side in ('left', 'right')]
                state['hands'] = [_body(raw.model, f'{side}_wrist_yaw_link') for side in ('left', 'right')]
                state['scratch'] = mujoco.MjData(raw.model)
        focus_at = playback.focus(raw.model, raw.data)
        camera.lookat[:] = (focus_at[0], focus_at[1], .7 * playback.target[2]) if walking_body else state['aim'](raw.data)
        state['renderer'].update_scene(raw.data, camera=camera)
        turned = math.degrees(state['turned'])
        if clip.arms:
            _arm_marks(state['renderer'].scene, raw, state['scratch'], state['hands'])
            value = _arm_measure(raw, state['shoulders'])
        elif clip.getup:
            value = (raw.head, raw.getup_at)
        elif clip.measure == 'creep':
            value = (math.hypot(focus_at[0], focus_at[1]), turned)
        elif clip.measure == 'walk_heading':
            value = (focus_at[0], turned)
        elif clip.measure == 'height':
            value = (float(raw.data.qpos[2]), raw.height)
        elif capsules is not None:
            value = (focus_at[1] if clip.measure == 'left_gap' else -focus_at[1], 100 * state['gap'])
        else:
            value = float(_measure(clip.measure, focus_at, state['turned']))
        mark = state['mark'] if push_step is not None and 0 <= state['step'] - push_step < marked else None
        tag = None
        if clip.arms and clip.load:
            tag = ('load', clip.load)
        elif held_step is not None and state['step'] > held_step:
            tag = ('random', clip.amplitude) if clip.held[1] == 'random' else ('pose', clip.held[1])
        frames.append((state['renderer'].render().copy(), now, value, bool(fell), mark, tag))

    try:
        if clip.episode:
            _play(stage, identity, policy, norm, clip, 73000, seconds, on_step)
        else:
            rollout(stage, identity, policy, norm, clip.scenario, 73000, seconds, on_step)
    finally:
        norm.close()
        if state['renderer'] is not None:
            state['renderer'].close()
    total = math.ceil(seconds * fps)
    if frames and len(frames) < total:  # fell (or a length between two frames): hold the last frame
        image, now, value, fell, _, tag = frames[-1]
        frames += [(image, now, value, fell, None, tag)] * (total - len(frames))
    return frames[:total]


def _number(value, lang):
    """A load or a push speed as written in a tile: no trailing zeros, a decimal comma in Russian."""
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    return text.replace('.', ',') if lang == 'ru' else text


def footer(now, value, fell, measure='forward', lang='ru'):
    """The tile's bottom line: time (or the fall time; getting up, the cut-off time) and the scenario's measure."""
    texts = TEXTS[lang]
    status = texts[('cutoff' if measure == 'getup' else 'fell') if fell else 'time'].format(t=now)
    if measure == 'getup':
        head, up = value
        detail = texts['getup'].format(v=head) if up is None else texts['getup_up'].format(t=up, v=head)
    else:
        detail = texts[measure].format(v=value)
    line = f'{status} · {detail}'
    return line.replace('.', ',') if lang == 'ru' else line


def push_label(mark, lang='ru'):
    """The push mark's text: direction in the robot's frame, Δv and the on-screen arrow, if any."""
    (forward, left), arrow = mark
    side = ('forward' if forward > 0 else 'back') if abs(forward) >= abs(left) else ('left' if left > 0 else 'right')
    speed = f'{math.hypot(forward, left):.2f}'
    text = TEXTS[lang]['mark'].format(d=DIRECTIONS[lang][side], v=speed[:-1] if speed.endswith('0') else speed)
    text += f' {arrow}' if arrow else ''
    return text.replace('.', ',') if lang == 'ru' else text


def tag_label(tag, lang='ru'):
    """A tile's tag (film_lane): the hand load, kg per hand; a held named pose; a held random pose's amplitude."""
    kind, value = tag
    return TEXTS[lang][kind].format(v=value if kind == 'pose' else _number(value, lang))


def compose(lanes, titles, size=TILE, lang='ru', measure='forward'):
    """One PIL frame per time step: lane tiles in rows of PER_ROW, each with title, live numbers, a push mark and a
    tag (a hand load, a held arm pose). Overlays scale with the tile (TILE gives the original sizes)."""
    scale = min(size[0] / TILE[0], size[1] / TILE[1])
    px = lambda value: max(1, round(value * scale))
    font, big = _tile_font(px(14), lang), _tile_font(px(16), lang)
    smaller = {}

    def fitted(draw, text):
        """The footer's font, or a smaller one (down to 0.7 of it) while the text is wider than the tile."""
        width = size[0] - px(16)
        if draw.textlength(text, font=font) <= width:
            return font
        for points in range(px(14) - 1, max(1, round(.7 * px(14))) - 1, -1):
            if points not in smaller:
                smaller[points] = _tile_font(points, lang)
            if draw.textlength(text, font=smaller[points]) <= width:
                break
        return smaller[points]

    columns = min(PER_ROW, len(lanes))
    rows = math.ceil(len(lanes) / columns)
    result = []
    for index in range(max(len(lane) for lane in lanes)):
        sheet = Image.new('RGB', (size[0] * columns, size[1] * rows), (24, 28, 34))
        draw = ImageDraw.Draw(sheet)
        for lane_index, (lane, title) in enumerate(zip(lanes, titles)):
            image, now, value, fell, *extra = lane[min(index, len(lane) - 1)]
            mark, tag = (list(extra) + [None, None])[:2]
            tile = Image.fromarray(image)
            if fell:
                tile = Image.blend(tile, Image.new('RGB', tile.size, (90, 90, 90)), .5)
            left, top = size[0] * (lane_index % columns), size[1] * (lane_index // columns)
            sheet.paste(tile, (left, top))
            draw.rectangle((left, top, left + size[0] - 1, top + px(26)), fill=(16, 18, 22))
            draw.text((left + px(8), top + px(4)), title, font=big, fill=(235, 238, 242))
            draw.rectangle((left, top + size[1] - px(24), left + size[0] - 1, top + size[1] - 1), fill=(16, 18, 22))
            text = footer(now, value, fell, measure, lang)
            line = fitted(draw, text)
            shift = (px(14) - getattr(line, 'size', px(14))) // 2  # a smaller font stays centred in the bar
            draw.text((left + px(8), top + size[1] - px(21) + shift), text, font=line,
                      fill=(255, 120, 110) if fell else (235, 238, 242))
            if mark is not None:
                text, at = push_label(mark, lang), (left + px(8), top + px(34))
                box = draw.textbbox(at, text, font=big)
                draw.rectangle((box[0] - px(5), box[1] - px(3), box[2] + px(5), box[3] + px(3)), fill=(245, 158, 11))
                draw.text(at, text, font=big, fill=(16, 18, 22))
            if tag is not None:  # top right, under the title bar
                text = tag_label(tag, lang)
                width = draw.textlength(text, font=font)
                at = (left + size[0] - px(8) - width, top + px(34))
                box = draw.textbbox(at, text, font=font)
                draw.rectangle((box[0] - px(5), box[1] - px(3), box[2] + px(5), box[3] + px(3)), fill=(16, 18, 22))
                draw.text(at, text, font=font, fill=(245, 158, 11))
            if fell:
                draw.rectangle((left, top, left + size[0] - 1, top + size[1] - 1), outline=(220, 60, 50), width=px(3))
            draw.line((left, top, left, top + size[1]), fill=(24, 28, 34), width=px(2))
        result.append(sheet)
    return result


def render(items, scenario_name, out, seconds=None, fps=FPS, size=TILE, view=None, lang='ru', focus=None, load=None,
           pose=None, start=None, draw=None, amplitude=None):
    """items: [(title, stage dir)]. Writes <out>.webp and the poster <out stem>-poster.jpg; returns both paths.
    seconds, view, focus: None, the scenario's; load, pose, start, draw, amplitude: configure's."""
    if lang not in TEXTS:
        raise ValueError(f'Неизвестный язык подписей: {lang}')
    clip = configure(get_clip(scenario_name), load, pose, start, draw, amplitude)
    seconds = clip.seconds if seconds is None else seconds
    lanes = [film_lane(stage, clip, seconds, fps, size, view, focus) for _, stage in items]
    frames = compose(lanes, [title for title, _ in items], size, lang, clip.measure)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, quality=60,
                   method=4)
    poster = out.with_name(out.stem + '-poster.jpg')
    frames[min(len(frames) - 1, int(2.5 * fps))].save(poster, quality=82)
    return out, poster


def _size(text):
    """'640x480' → (640, 480)."""
    width, _, height = text.lower().partition('x')
    if not (width.isdigit() and height.isdigit() and 64 <= int(width) <= 4096 and 64 <= int(height) <= 4096):
        raise argparse.ArgumentTypeError('ожидается ШИРИНАxВЫСОТА от 64 до 4096, например 640x480')
    return int(width), int(height)


def main():
    from .export_g1 import NAMED_ARM_POSES
    groups = {'walking': [name for name, clip in CLIPS.items() if not clip.episode],
              'arms (arm_targets stages)': [name for name, clip in CLIPS.items() if clip.arms],
              'held arm pose (any 29-DoF walking stage)': [name for name, clip in CLIPS.items() if clip.held],
              'height (height_command stages)': [name for name, clip in CLIPS.items() if clip.heights],
              'getting up (g1_getup stages)': [name for name, clip in CLIPS.items() if clip.getup]}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('branches', nargs='+', help='experiment id, optionally :steps of a stage')
    parser.add_argument('--titles', help='comma separated tile titles (default: branch and stage)')
    parser.add_argument('--scenario', default='walk', choices=list(CLIPS), metavar='SCENARIO',
                        help='; '.join(f'{group}: {", ".join(names)}' for group, names in groups.items()))
    parser.add_argument('--seconds', type=float,
                        help="clip length (default: the scenario's, 10 s; arms_poses 11 s, getting up 8 s)")
    parser.add_argument('--focus', choices=list(FOCI),
                        help="camera focus: body (the scenario's default; arms scenarios arms, fast side steps feet), "
                             "torso, arms (shoulders to palms), hands (palms), feet (ankle_roll links)")
    parser.add_argument('--view', choices=list(VIEWS),
                        help='camera side (default side: from the right; fast side steps front)')
    parser.add_argument('--pose', choices=[name for name, _, _ in NAMED_ARM_POSES] + ['random'],
                        help='arms scenarios with one pose: that pose instead (export_g1.NAMED_ARM_POSES); held arm '
                             'pose scenarios: the pose held, or random (the default)')
    parser.add_argument('--draw', type=int, help='held random pose: numpy default_rng seed (default 0)')
    parser.add_argument('--amplitude', type=float,
                        help=f'held random pose: ±rad per arm joint (default 1, up to {HELD_AMPLITUDE:g})')
    parser.add_argument('--load', type=float, help=f'arms scenarios: kg on each hand (0-{LOAD_LIMIT:g}, default 0)')
    parser.add_argument('--start', type=int, help="getting-up scenarios: which held-out start of the class (0 first)")
    parser.add_argument('--lang', default='ru', choices=list(TEXTS), help='tile text language')
    parser.add_argument('--size', type=_size, default=TILE, help='tile size WxH (default 320x270)')
    parser.add_argument('--out', type=Path, required=True, help='.webp path')
    args = parser.parse_args()
    lab = Laboratory(DEFAULT_RUN)
    stages = [find_stage(lab, item) for item in args.branches]
    titles = args.titles.split(',') if args.titles else [
        f"{item.partition(':')[0][4:12]} · {int(s.name[6:]) / 1e6:.0f}{TEXTS[args.lang]['million']}"
        for item, s in zip(args.branches, stages)]
    print(*render(list(zip(titles, stages)), args.scenario, args.out, args.seconds, size=args.size, view=args.view,
                  lang=args.lang, focus=args.focus, load=args.load, pose=args.pose, start=args.start, draw=args.draw,
                  amplitude=args.amplitude), sep='\n')


if __name__ == '__main__':
    main()
