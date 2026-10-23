"""Pre-rendered clips of stages side by side for the journal («Дневник») and the site, watchable without a live race.

    .venv\\Scripts\\python.exe -m humanoid_sim.clip exp-…[:steps] … [--titles "было,стало"] [--scenario walk]
        [--lang ru|en|zh] [--size 640x480] --out c.webp

Each lane is the same deterministic CPU episode as `supervise` and `collage` (seed 73000, the scenario's fixed
commands), filmed from the side by a camera following the pelvis. Lanes are tiles side by side, at most four
per row, labelled with their title, time and the scenario's measure (metres forward, back or sideways, degrees
turned, the pelvis displacement after a push); a fallen robot's tile freezes, greyed, with the fall time.
Scenarios (CLIPS): the evaluation ones (stand, slow, walk, brisk, stop_start, sprint) and clip-only ones (back,
side_left / side_right, turn_left / turn_right, walk_turn, side_stop, and force pushes push_back / push_forward /
push_left / push_right / push_walk as G1VelocityEnv.set_force_push gives them, marked in the tile for
MARK_SECONDS). Texts in Russian (the default, decimal comma), English or Chinese; the overlays scale with the tile.
Saved as an animated WebP (browsers play it in an <img>, looping) plus a JPEG poster frame.
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .checkpoint_schema import checkpoint_identity
from .collage import _font
from .core import DEFAULT_RUN, load_stage, read_json
from .evaluation.scenarios import SCENARIOS, Scenario
from .lab import Laboratory
from .supervise import find_stage, rollout

TILE = (320, 270)
PER_ROW = 4
FPS = 25
MARK_SECONDS = .6  # a push stays marked in the tile this long from its start


VIEWS = {'side': 90., 'front': 180.}  # side looks along +y (forward is to the right); front faces the robot

# Tile texts by language. The measure (the footer's number) is positive in the named direction. Russian is the
# original, decimal comma; Chinese follows the site's catalogs and docs/i18n-glossary.md (units as symbols, 跌倒).
TEXTS = {
    'ru': dict(time='{t:4.1f} с', fell='упал на {t:.1f} с', forward='вперёд {v:+.1f} м', back='назад {v:+.1f} м',
               left='влево {v:+.1f} м', right='вправо {v:+.1f} м', turn_left='поворот влево {v:+.0f}°',
               turn_right='поворот вправо {v:+.0f}°', push='смещение {v:.2f} м', mark='толчок {d} {v} м/с',
               million=' млн'),
    'en': dict(time='{t:4.1f} s', fell='fell at {t:.1f} s', forward='forward {v:+.1f} m', back='back {v:+.1f} m',
               left='left {v:+.1f} m', right='right {v:+.1f} m', turn_left='turned left {v:+.0f}°',
               turn_right='turned right {v:+.0f}°', push='displaced {v:.2f} m', mark='push {d} {v} m/s',
               million=' M'),
    'zh': dict(time='{t:4.1f} s', fell='在 {t:.1f} s 时跌倒', forward='前进 {v:+.1f} m', back='后退 {v:+.1f} m',
               left='向左 {v:+.1f} m', right='向右 {v:+.1f} m', turn_left='左转 {v:+.0f}°', turn_right='右转 {v:+.0f}°',
               push='位移 {v:.2f} m', mark='{d}推 {v} m/s', million=' M'),
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
    frame, duration s), as G1VelocityEnv.set_force_push."""
    scenario: Scenario
    measure: str = 'forward'
    push: tuple = None


def _clip(name, schedule, measure, push=None):
    return ClipScenario(Scenario(name, schedule), measure, push)


STAND = (0., 0., 0.)
PUSH = 1.25  # m/s: the acceptance probes' and the sim2sim push patch's Δv (force_push's working value), for 0.1 s
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
))


def get_clip(name):
    if name not in CLIPS:
        raise ValueError(f'Неизвестный сценарий ролика: {name}')
    return CLIPS[name]


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


def film_lane(stage, scenario, seconds=10., fps=FPS, size=TILE, view='side'):
    """[(rgb frame, seconds, measure, fell, push mark)] at `fps` for one episode, padded with the last frame.

    scenario: a ClipScenario, or an evaluation Scenario (measured forward). The push mark is ((forward, left) Δv,
    on-screen arrow) for MARK_SECONDS from the push start, else None."""
    from .tasks.catalog import resolve_task
    clip = scenario if isinstance(scenario, ClipScenario) else ClipScenario(scenario)
    identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
    playback = resolve_task(identity['robot_id'], identity['task_id']).playback
    every = max(1, round(1 / (fps * identity['control_dt'])))
    push_step = None if clip.push is None else round(clip.push[0] / identity['control_dt'])
    marked = round(MARK_SECONDS / identity['control_dt'])
    policy, norm = load_stage(stage)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.8 * playback.target[2], -8., VIEWS[view]
    state = {'renderer': None, 'step': 0, 'yaw': 0., 'turned': 0., 'mark': None}
    frames = []

    def on_step(now, raw, lean, fell):
        state['step'] += 1
        if clip.measure in ('turn_left', 'turn_right'):  # unwrapped heading change of the followed body
            rotation = raw.data.xmat[mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, playback.follow_body)]
            yaw = math.atan2(rotation[3], rotation[0])
            state['turned'] += math.remainder(yaw - state['yaw'], math.tau)
            state['yaw'] = yaw
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
        focus = playback.focus(raw.model, raw.data)
        camera.lookat[:] = (focus[0], focus[1], .7 * playback.target[2])
        state['renderer'].update_scene(raw.data, camera=camera)
        mark = state['mark'] if push_step is not None and 0 <= state['step'] - push_step < marked else None
        frames.append((state['renderer'].render().copy(), now, float(_measure(clip.measure, focus, state['turned'])),
                       bool(fell), mark))

    try:
        rollout(stage, identity, policy, norm, clip.scenario, 73000, seconds, on_step)
    finally:
        norm.close()
        if state['renderer'] is not None:
            state['renderer'].close()
    total = math.ceil(seconds * fps)
    if frames and len(frames) < total:  # fell: hold the fall frame
        image, now, value, _, _ = frames[-1]
        frames += [(image, now, value, True, None)] * (total - len(frames))
    return frames[:total]


def footer(now, value, fell, measure='forward', lang='ru'):
    """The tile's bottom line: time (or the fall time) and the scenario's measure."""
    texts = TEXTS[lang]
    line = f"{texts['fell' if fell else 'time'].format(t=now)} · {texts[measure].format(v=value)}"
    return line.replace('.', ',') if lang == 'ru' else line


def push_label(mark, lang='ru'):
    """The push mark's text: direction in the robot's frame, Δv and the on-screen arrow, if any."""
    (forward, left), arrow = mark
    side = ('forward' if forward > 0 else 'back') if abs(forward) >= abs(left) else ('left' if left > 0 else 'right')
    speed = f'{math.hypot(forward, left):.2f}'
    text = TEXTS[lang]['mark'].format(d=DIRECTIONS[lang][side], v=speed[:-1] if speed.endswith('0') else speed)
    text += f' {arrow}' if arrow else ''
    return text.replace('.', ',') if lang == 'ru' else text


def compose(lanes, titles, size=TILE, lang='ru', measure='forward'):
    """One PIL frame per time step: lane tiles in rows of PER_ROW, each with title, live numbers and a push mark.
    Overlays scale with the tile (TILE gives the original sizes)."""
    scale = min(size[0] / TILE[0], size[1] / TILE[1])
    px = lambda value: max(1, round(value * scale))
    font, big = _tile_font(px(14), lang), _tile_font(px(16), lang)
    columns = min(PER_ROW, len(lanes))
    rows = math.ceil(len(lanes) / columns)
    result = []
    for index in range(max(len(lane) for lane in lanes)):
        sheet = Image.new('RGB', (size[0] * columns, size[1] * rows), (24, 28, 34))
        draw = ImageDraw.Draw(sheet)
        for lane_index, (lane, title) in enumerate(zip(lanes, titles)):
            image, now, value, fell, *mark = lane[min(index, len(lane) - 1)]
            tile = Image.fromarray(image)
            if fell:
                tile = Image.blend(tile, Image.new('RGB', tile.size, (90, 90, 90)), .5)
            left, top = size[0] * (lane_index % columns), size[1] * (lane_index // columns)
            sheet.paste(tile, (left, top))
            draw.rectangle((left, top, left + size[0] - 1, top + px(26)), fill=(16, 18, 22))
            draw.text((left + px(8), top + px(4)), title, font=big, fill=(235, 238, 242))
            draw.rectangle((left, top + size[1] - px(24), left + size[0] - 1, top + size[1] - 1), fill=(16, 18, 22))
            draw.text((left + px(8), top + size[1] - px(21)), footer(now, value, fell, measure, lang),
                      font=font, fill=(255, 120, 110) if fell else (235, 238, 242))
            if mark and mark[0] is not None:
                text, at = push_label(mark[0], lang), (left + px(8), top + px(34))
                box = draw.textbbox(at, text, font=big)
                draw.rectangle((box[0] - px(5), box[1] - px(3), box[2] + px(5), box[3] + px(3)), fill=(245, 158, 11))
                draw.text(at, text, font=big, fill=(16, 18, 22))
            if fell:
                draw.rectangle((left, top, left + size[0] - 1, top + size[1] - 1), outline=(220, 60, 50), width=px(3))
            draw.line((left, top, left, top + size[1]), fill=(24, 28, 34), width=px(2))
        result.append(sheet)
    return result


def render(items, scenario_name, out, seconds=10., fps=FPS, size=TILE, view='side', lang='ru'):
    """items: [(title, stage dir)]. Writes <out>.webp and the poster <out stem>-poster.jpg; returns both paths."""
    if lang not in TEXTS:
        raise ValueError(f'Неизвестный язык подписей: {lang}')
    clip = get_clip(scenario_name)
    lanes = [film_lane(stage, clip, seconds, fps, size, view) for _, stage in items]
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('branches', nargs='+', help='experiment id, optionally :steps of a stage')
    parser.add_argument('--titles', help='comma separated tile titles (default: branch and stage)')
    parser.add_argument('--scenario', default='walk', choices=list(CLIPS), metavar='SCENARIO',
                        help=', '.join(CLIPS))
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--view', default='side', choices=sorted(VIEWS))
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
                  lang=args.lang), sep='\n')


if __name__ == '__main__':
    main()
