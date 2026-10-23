"""Contact sheet of stages: one picture per check shows posture over time instead of many screenshots.

    .venv\\Scripts\\python.exe -m humanoid_sim.collage exp-… [exp-…:228753408 …] [--scenario walk] [--views side,front]

Rows are branches (× seeds × views), columns are moments of one CPU episode under a fixed scenario command:
the same episodes as `supervise` (seed 73000 first). Each tile shows time, torso lean, pelvis height and
position; after a fall the remaining tiles repeat the fall frame, greyed, with the fall time. The last
column is the ground path seen from above with its widest bearing off the course, so circles and drift show.
"""
import argparse
import time
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from .checkpoint_schema import checkpoint_identity
from .core import DEFAULT_RUN, load_stage, read_json
from .lab import Laboratory
from .supervise import find_stage, rollout

TIMES = (.5, 1., 2., 3., 5., 7., 10.)
# Camera azimuth: 'side' looks along +y, so forward (+x) is to the right; 'front' faces a robot heading +x.
VIEWS = {'side': 90., 'front': 180.}
TILE = (200, 260)
HEADER = 150


def _font(size):
    for name in ('segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def film(stage, identity, policy, norm, scenario, seed, times=TIMES, views=('side',), size=TILE):
    """One episode: ({view: [(rgb array, label lines, fallen)]} per requested time, ground path, supervise row)."""
    from .tasks.catalog import resolve_task
    playback = resolve_task(identity['robot_id'], identity['task_id']).playback
    dt = identity['control_dt']
    frames = {view: [] for view in views}
    path = []
    state = {'renderer': None, 'next': 0}
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = 2.6 * playback.target[2], -5.

    def shoot(now, raw, lean):
        if state['renderer'] is None:
            state['renderer'] = mujoco.Renderer(raw.model, height=size[1], width=size[0])
        focus = playback.focus(raw.model, raw.data)
        camera.lookat[:] = (focus[0], focus[1], .72 * playback.target[2])  # horizontal follow, fixed height
        label = [f'{now:g} с' + ('' if lean is None else f' · корпус {lean:+.0f}°'),
                 f'таз {focus[2]:.2f} · x {focus[0]:+.1f} y {focus[1]:+.1f}']
        for view in views:
            camera.azimuth = VIEWS[view]
            state['renderer'].update_scene(raw.data, camera=camera)
            frames[view].append((state['renderer'].render().copy(), label, False))

    def on_step(now, raw, lean, fell):
        path.append(tuple(playback.focus(raw.model, raw.data)[:2]))
        while state['next'] < len(times) and now >= times[state['next']] - dt / 2:
            shoot(now, raw, lean)
            state['next'] += 1
        if fell and state['next'] < len(times):
            shoot(now, raw, lean)
            for view in views:
                image, label, _ = frames[view].pop()
                frames[view] += [(image, [f'упал на {now:.1f} с', label[1]], True)] * (len(times) - state['next'])

    try:
        row = rollout(stage, identity, policy, norm, scenario, seed, max(times), on_step)
    finally:
        if state['renderer'] is not None:
            state['renderer'].close()
    return frames, path, row


def path_tile(path, row, size, font):
    """Top view of the ground path: 1 m grid, target course along +x (dashed), start green, end white/red."""
    width, height = size
    tile = Image.new('RGB', size, (20, 33, 48))
    draw = ImageDraw.Draw(tile)
    points = path or [(0., 0.)]
    xs, ys = [0.] + [p[0] for p in points], [0.] + [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 2.) * 1.15
    scale = min(width - 16, height - 56) / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def screen(x, y):  # +x to the right, +y up
        return width / 2 + (x - cx) * scale, (height - 40) / 2 - (y - cy) * scale

    for metre in range(int(min(xs) - span), int(max(xs) + span) + 1):
        draw.line((*screen(metre, cy - span), *screen(metre, cy + span)), fill=(40, 58, 78))
    for metre in range(int(min(ys) - span), int(max(ys) + span) + 1):
        draw.line((*screen(cx - span, metre), *screen(cx + span, metre)), fill=(40, 58, 78))
    for start in range(int(min(xs) - span), int(max(xs) + span) + 1):
        draw.line((*screen(start, 0), *screen(start + .5, 0)), fill=(110, 140, 170))
    if len(points) > 1:
        draw.line([screen(*p) for p in points], fill=(240, 200, 90), width=2)
    for (x, y), colour in ((points[0], (90, 220, 120)),
                           (points[-1], (230, 70, 60) if row and row.get('fell_at') else (240, 240, 240))):
        sx, sy = screen(x, y)
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=colour)
    draw.rectangle((0, height - 38, width - 1, height - 1), fill=(16, 18, 22))
    end = points[-1]
    bearing = f" · уходит до {row['path_deg']:.0f}°" if row else ''
    draw.text((6, height - 36), f'путь сверху{bearing}', font=font, fill=(235, 238, 242))
    draw.text((6, height - 19), f'вперёд {end[0]:+.1f} м · вбок {end[1]:+.1f} м', font=font, fill=(235, 238, 242))
    return tile


def collage(items, scenario_name='walk', times=TIMES, seeds=1, views=('side',), seed_start=73000, size=TILE):
    """items: [(row title, stage dir)]. Returns one PIL image, a row per stage × seed × view."""
    from .evaluation.scenarios import get_scenario
    scenario = get_scenario(scenario_name)
    small, big = _font(13), _font(15)
    rows = []
    for title, stage in items:
        identity = checkpoint_identity(read_json(Path(stage) / 'metadata.json'))
        policy, norm = load_stage(stage)
        try:
            for seed in range(seed_start, seed_start + seeds):
                frames, path, row = film(stage, identity, policy, norm, scenario, seed, times, views, size)
                rows += [([title, f'{scenario.name} · seed {seed}', view], frames[view], path, row) for view in views]
        finally:
            norm.close()
    sheet = Image.new('RGB', (HEADER + size[0] * (len(times) + 1), size[1] * len(rows)), (24, 28, 34))
    draw = ImageDraw.Draw(sheet)
    for r, (header, tiles, path, row) in enumerate(rows):
        sheet.paste(path_tile(path, row, size, small), (HEADER + size[0] * len(times), r * size[1]))
        top = r * size[1]
        for i, line in enumerate(header):
            draw.text((10, top + 12 + 22 * i), line, fill=(235, 238, 242) if i == 0 else (160, 170, 180),
                      font=big if i == 0 else small)
        for c, (image, label, fallen) in enumerate(tiles):
            tile = Image.fromarray(image)
            if fallen:
                tile = Image.blend(tile, Image.new('RGB', tile.size, (90, 90, 90)), .55)
            left = HEADER + c * size[0]
            sheet.paste(tile, (left, top))
            draw.rectangle((left, top + size[1] - 38, left + size[0] - 1, top + size[1] - 1), fill=(16, 18, 22))
            for i, line in enumerate(label):
                draw.text((left + 6, top + size[1] - 36 + 17 * i), line, font=small,
                          fill=(255, 120, 110) if fallen and i == 0 else (235, 238, 242))
            if fallen:
                draw.rectangle((left, top, left + size[0] - 1, top + size[1] - 1), outline=(220, 60, 50), width=3)
            draw.line((left, top, left, top + size[1]), fill=(24, 28, 34), width=1)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('branches', nargs='+', help='experiment id, optionally :steps of a stage')
    parser.add_argument('--scenario', default='walk', help='stand, slow, walk, brisk, stop_start, sprint')
    parser.add_argument('--times', default=','.join(f'{t:g}' for t in TIMES), help='seconds, comma separated')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--views', default='side', help=f'comma separated: {", ".join(VIEWS)}')
    parser.add_argument('--out', type=Path, help='PNG path (default: <lab>/collages/<time>-<scenario>.png)')
    args = parser.parse_args()
    lab = Laboratory(DEFAULT_RUN)
    items = []
    for item in args.branches:
        stage = find_stage(lab, item)
        if stage is None:
            print(f'{item}: нет этапов', flush=True)
            continue
        items.append((f"{item.partition(':')[0][:8]} · {int(stage.name[6:]) / 1e6:.1f} млн", stage))
    if not items:
        raise SystemExit('нечего показывать')
    views = tuple(v.strip() for v in args.views.split(',') if v.strip())
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f'неизвестный вид: {", ".join(unknown)}')
    times = tuple(sorted(float(t) for t in args.times.split(',') if t.strip()))
    sheet = collage(items, args.scenario, times, args.seeds, views)
    out = args.out or lab.root / 'collages' / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.scenario}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
