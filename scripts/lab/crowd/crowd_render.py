"""A clip of a recorded crowd (crowd_record.py): every robot in its own grid cell along its own recorded run, drawn into
one MuJoCo scene per frame (mjv_addGeoms), the camera flying from a close shot of a few robots up over the field.

    python scripts/lab/crowd/crowd_render.py <crowd.npz> --out x.mp4 [--robots N] [--size 1920x1080]
        [--fps 30] [--spacing 1.6] [--frames a,b,c --stills]   (--stills: just those frames as .jpg, a quick look)

A march recording (crowd_record.py --march-at) shows only the robots that got up and marched, each turned about the
point where it stood when the march began: with --pattern radial (default) to walk out from the field's centre along
its own ray, so the crowd opens out and no one crosses another; with column all along +x, the camera following. The field's edge never shows (Pavel 29.10: one should not see where it ends, and no fog): every frame the
camera's distance is cut so that the four corners of the picture meet the floor inside the outermost robots, and it
always looks down by more than half the field of view, so no horizon either.

Nothing is simulated here: each frame sets a robot's recorded qpos, moved to its cell, runs mj_kinematics and adds its
geoms. Renders on the GPU through OpenGL; minutes, not hours."""
import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / 'assets' / 'robots' / 'g1' / 'scene_29dof.xml'
RATE = 25  # recorded poses per second (crowd_record.SAMPLE)
MARGIN = 1.  # m inside the outermost robots where a corner of the picture may meet the floor


def smooth(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3 - 2 * x)


def turned(qpos, angle, pivot):
    """The robot's run turned by angle about pivot (xy)."""
    c, s = math.cos(angle), math.sin(angle)
    out = qpos.astype(np.float64).copy()
    xy = out[:, :2] - pivot
    out[:, 0], out[:, 1] = c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]
    turn = np.array([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
    for k in range(len(out)):
        mujoco.mju_mulQuat(out[k, 3:7], turn, out[k, 3:7].copy())
    out[:, :2] += pivot
    return out


def corners_inside(lookat, distance, elevation, azimuth, bounds, fovy, aspect):
    """Whether the four corner rays of the picture all meet the floor inside bounds (x0, x1, y0, y1)."""
    el, az = math.radians(elevation), math.radians(azimuth)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    eye = np.asarray(lookat) - distance * forward
    ty = math.tan(math.radians(fovy) / 2)
    tx = ty * aspect
    for sx in (-1, 1):
        for sy in (-1, 1):
            ray = forward + sx * tx * right + sy * ty * up
            if ray[2] >= -1e-6:
                return False
            point = eye - eye[2] / ray[2] * ray
            if not (bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]):
                return False
    return True


def largest_distance(lookat, wanted, elevation, azimuth, bounds, fovy, aspect):
    """The wanted distance, or the largest smaller one whose corners stay inside (bisection)."""
    if corners_inside(lookat, wanted, elevation, azimuth, bounds, fovy, aspect):
        return wanted
    lo, hi = 0.3, wanted
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if corners_inside(lookat, mid, elevation, azimuth, bounds, fovy, aspect) else (lo, mid)
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('crowd')
    parser.add_argument('--out', required=True)
    parser.add_argument('--robots', type=int, default=4096)
    parser.add_argument('--size', default='1920x1080')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--spacing', type=float, default=2.2)
    parser.add_argument('--pattern', choices=('radial', 'column'), default='radial',
                        help='after the march begins: out from the centre, or all along +x')
    parser.add_argument('--frames', help='render only these output frames (comma list)')
    parser.add_argument('--stills', action='store_true')
    args = parser.parse_args()
    # read every array once: an NpzFile decompresses an array again on each access (4096 times 464 MB took ages)
    crowd = dict(np.load(args.crowd, allow_pickle=False))
    march = 'marched' in crowd and float(crowd['march_at']) > 0
    live = 'kind' in crowd  # crowd_live.py: every robot its own programme, the camera always on the move
    if live and args.spacing == parser.get_default('spacing'):
        args.spacing = 3.2  # they roam up to ~3.5 m (p90) from their spot
    chosen = np.flatnonzero(crowd['marched']) if march else np.arange(len(crowd['qpos']))
    chosen = np.random.default_rng(29_10).permutation(chosen)[:args.robots]  # classes mixed over the field
    n = len(chosen)
    steps = crowd['qpos'].shape[1]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # cells: rows along the march (x), columns across it (y)
    cells = np.stack([(np.arange(n) // cols) - (rows - 1) / 2, (np.arange(n) % cols) - (cols - 1) / 2], 1) * args.spacing
    poses = np.empty((n, steps, crowd['qpos'].shape[2]))
    for i, k in enumerate(chosen):
        q = crowd['qpos'][k]
        if march:
            pivot = crowd['march_xy'][k]
            # radial: each walks out from the field's centre along its own ray, so neighbours part and no one
            # crosses another (they do not see each other); column: all along +x
            goal = math.atan2(cells[i, 1], cells[i, 0]) if args.pattern == 'radial' else 0.
            q = turned(q, goal - float(crowd['heading'][k]), pivot)
            q[:, :2] += cells[i] - pivot
        else:
            q = q.astype(np.float64).copy()
            q[:, :2] += cells[i] - q[0, :2]
        poses[i] = q
    # the column's travel per recorded pose (mean displacement from the cells), for the camera and the field's bounds
    shift = (poses[:, :, 0] - cells[:, None, 0]).mean(0)
    if args.pattern == 'radial':  # the crowd opens out about the centre, which stays put
        shift[:] = 0.
    print(f'{n} robots ({rows} x {cols}), {steps / RATE:.1f} s, column travel {shift[-1]:.1f} m', flush=True)
    width, height = (int(v) for v in args.size.split('x'))

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.map.zfar = 600.
    model.vis.quality.shadowsize = 2048  # 8192 took ~0.5 GB of the GPU the training shares (29.10 it died of OOM)
    fovy, aspect = float(model.vis.global_.fovy), width / height
    data = mujoco.MjData(model)
    per_robot = int(np.sum(model.geom_group != 3)) + 4
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=n * per_robot + 200)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.geomgroup[3] = 0  # collision capsules
    pert, camera = mujoco.MjvPerturb(), mujoco.MjvCamera()
    # the close shot: the robot nearest the field's centre, a few rows in front of it
    near = int(np.argmin(np.linalg.norm(cells - [args.spacing * 2, 0.], axis=1)))
    half_x, half_y = (rows - 1) / 2 * args.spacing - MARGIN, (cols - 1) / 2 * args.spacing - MARGIN

    seconds = steps / RATE
    total = int(seconds * args.fps)
    wanted = [int(v) for v in args.frames.split(',')] if args.frames else range(total)
    writer = None
    if not args.stills:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(args.out, (width, height), fps=args.fps, codec='libx264', quality=None,
                                             pix_fmt_out='yuv420p', output_params=['-crf', '20', '-preset', 'medium',
                                                                                    '-movflags', '+faststart', '-an'])
        writer.send(None)

    def view(t, k):
        """Camera at t s (recorded pose k): close and low over the lying robots, rising while they get up, then over
        the column, slowly swinging round to its front; the distance cut to keep the corners inside the field."""
        rise = smooth(t / 11.)
        centre = np.array([shift[k], 0., 0.5])
        close = np.array([*poses[near, k, :2], 0.4])
        lookat = (1 - rise) * close + rise * centre
        elevation = -27 - 15 * rise
        azimuth = 150 + 60 * t / seconds
        distance = 3. + 57. * rise
        bounds = (shift[k] - half_x, shift[k] + half_x, -half_y, half_y)
        return lookat, largest_distance(lookat, distance, elevation, azimuth, bounds, fovy, aspect), elevation, azimuth

    def roam(t, k):
        """Camera of a live field: never still; from close over a few robots it drifts across the field, its height
        breathing and its heading turning, the distance cut as in view()."""
        rise = smooth(t / 8.)
        lookat = np.array([14. * math.sin(.11 * t), 10. * math.sin(.08 * t + .6), .6 - .3 * rise])
        elevation = -30 - 12 * rise - 6 * math.sin(.21 * t)
        azimuth = 120 + 5. * t
        distance = 5. + 33. * rise + 9. * rise * math.sin(.17 * t + 1.)
        bounds = (-half_x, half_x, -half_y, half_y)
        return lookat, largest_distance(lookat, distance, elevation, azimuth, bounds, fovy, aspect), elevation, azimuth

    for f in wanted:
        t = f / args.fps
        k = min(int(t * RATE), steps - 1)
        lookat, distance, elevation, azimuth = (roam if live else view)(t, k)
        camera.lookat[:], camera.distance, camera.elevation, camera.azimuth = lookat, distance, elevation, azimuth
        data.qpos[:] = poses[0, k]
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, camera, scene_option=option)
        scene = renderer.scene
        for i in range(1, n):
            data.qpos[:] = poses[i, k]
            mujoco.mj_kinematics(model, data)
            mujoco.mjv_addGeoms(model, data, option, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        image = renderer.render()
        if writer is not None:
            writer.send(np.ascontiguousarray(image))
        else:
            Image.fromarray(image).save(Path(args.out).with_name(f'{Path(args.out).stem}-{f:04d}.jpg'), quality=88)
        if f % 30 == 0:
            print(f'frame {f} / {total}, distance {distance:.1f} m, geoms {scene.ngeom}', flush=True)
    if writer is not None:
        writer.close()
    print(args.out, n, 'robots', total, 'frames')


if __name__ == '__main__':
    main()
