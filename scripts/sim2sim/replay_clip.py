"""Clips of sim2sim getting-up runs replayed in the G1 model (29.10, Pavel: film every fall of the bridge, from the moment
the robot stands and falls to the moment it is up, without the band).

Usage: python scripts/sim2sim/replay_clip.py <replay .npz>... --out x.webp [--grid 3x2] [--size 320x240]

The .npz come from the bridge's lowstate.csv / sport.csv / events.csv (pelvis position, IMU quaternion and the 29
joints at 25 Hz from 0.5 s before the push to 2 s into Velocity, the passive lying cut to 1 s; fields q, quat, xyz,
phase, t, meta): nothing is simulated, the recorded state is drawn by mj_forward in assets/robots/g1. Each tile follows
its pelvis and names the state of the chain (fall, lying, 1164, the PD transition, the tracker, walking №8); a grid
of runs plays side by side, a shorter run holding its last frame."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_sim.clip import _font  # noqa: E402

SCENE = ROOT / 'assets' / 'robots' / 'g1' / 'scene_29dof.xml'
FPS = 25
PHASES = ('стоит', 'толчок, падение', 'падение, лежит', '1164: переворот', 'переход ПД', 'трекер: встаёт', 'ходьба №8')
LYING = {'живот': 'на живот', 'спина': 'на спину', 'левый бок': 'на левый бок', 'правый бок': 'на правый бок'}


def tile_frames(path, size, renderer, model, data):
    run = np.load(path, allow_pickle=False)
    meta = json.loads(str(run['meta']))
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 3.4, -12., 135.
    font, small = _font(15), _font(13)
    title = f'{meta["push"] or "с подвеса"} → {LYING.get(meta["lying"], meta["lying"])}'
    frames = []
    for k in range(len(run['t'])):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:36] = run['xyz'][k], run['quat'][k], run['q'][k]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = (data.qpos[0], data.qpos[1], 0.55)
        renderer.update_scene(data, camera)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, size[0], 22), fill=(0, 0, 0))
        draw.text((6, 3), title, font=font, fill=(255, 255, 255))
        phase = PHASES[int(run['phase'][k])]
        draw.rectangle((0, size[1] - 20, size[0], size[1]), fill=(0, 0, 0))
        draw.text((6, size[1] - 18), f'{run["t"][k]:4.1f} с · {phase}', font=small, fill=(255, 255, 255))
        frames.append(image)
    return frames, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--grid', default='3x2')
    parser.add_argument('--size', default='320x240')
    args = parser.parse_args(argv)
    cols, rows = (int(v) for v in args.grid.split('x'))
    size = tuple(int(v) for v in args.size.split('x'))
    if len(args.runs) > cols * rows:
        raise SystemExit(f'{len(args.runs)} прогонов в сетку {args.grid} не входят')
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth, model.vis.global_.offheight = max(size[0], 640), max(size[1], 480)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    tiles = [tile_frames(path, size, renderer, model, data) for path in args.runs]
    length = max(len(frames) for frames, _ in tiles)
    sheet = []
    for k in range(length):
        canvas = Image.new('RGB', (cols * size[0], rows * size[1]), (20, 20, 20))
        for i, (frames, _) in enumerate(tiles):
            canvas.paste(frames[min(k, len(frames) - 1)], ((i % cols) * size[0], (i // cols) * size[1]))
        sheet.append(canvas)
    out = Path(args.out)
    sheet[0].save(out, save_all=True, append_images=sheet[1:], duration=1000 // FPS, loop=0, quality=70, method=4)
    sheet[min(len(sheet) - 1, int(length * 0.75))].save(out.with_name(out.stem + '-poster.jpg'), quality=82)
    print(f'{out}: {len(tiles)} прогонов, {length} кадров ({length / FPS:.1f} с)')


if __name__ == '__main__':
    main()
