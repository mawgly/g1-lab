"""A clip of a walking policy learned inside the world model (imagine_walk / dyna_walk policy.pt, not an SB3 stage,
so humanoid_sim.clip cannot film it) beside candidate 8, in the same CPU task with the same evaluation reset and
command; the camera follows the pelvis from the side, the footer gives the speed over the last second.

    .venv\\Scripts\\python.exe scripts\\lab\\wm\\walk_video.py --policy runs\\humanoid_lab\\world_model\\dyna-walk\\policy.pt
        [--command 0.5,0,0] [--seconds 8] --out <file>.webp"""
import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import imagine_walk as iw  # noqa: E402

SIZE = (420, 360)


def film(act, config, command, seconds):
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_velocity', 'g1_29dof', config).unwrapped
    observation, _ = env.reset(seed=0, options=dict(command=np.array(command)))
    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.6, 90., -10.
    frames, positions, fell = [], [], None
    for step in range(int(seconds / env.dt)):
        observation, _, terminated, _, _ = env.step(act(env, observation))
        positions.append(env.data.xpos[env.pelvis][:2].copy())
        if step % 2 == 0:
            camera.lookat[:] = env.data.xpos[env.pelvis]
            renderer.update_scene(env.data, camera=camera)
            recent = positions[-int(1 / env.dt):]
            speed = np.linalg.norm(recent[-1] - recent[0]) / max(len(recent) * env.dt, env.dt)
            frames.append((renderer.render().copy(), step * env.dt, speed, fell))
        if terminated:
            fell = step * env.dt
            break
    while len(frames) < int(seconds / env.dt / 2):
        image, _, speed, _ = frames[-1]
        frames.append((image, len(frames) * 2 * env.dt, speed, fell))
    renderer.close()
    env.close()
    return frames


def main():
    import torch
    from PIL import Image, ImageDraw, ImageFont
    parser = argparse.ArgumentParser(description='ролик ходьбы политики из модели мира рядом с кандидатом 8')
    parser.add_argument('--policy', required=True)
    parser.add_argument('--command', default='0.5,0,0')
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    command = [float(v) for v in args.command.split(',')]
    saved = torch.load(args.policy, map_location='cpu', weights_only=False)
    actor = iw.mlp(iw.OBSERVATION, iw.HIDDEN, 29, 'cpu')
    actor.load_state_dict({k: v.cpu() for k, v in saved['actor'].items()})
    mean, std = saved['mean'].cpu(), saved['std'].cpu()
    candidate, config = iw.candidate_policy()
    from humanoid_sim import world_model as wm
    from humanoid_sim.tasks import g1_tensor_math as task
    layout = wm.Layout(29)

    def ours(env, observation):
        raw = torch.as_tensor(np.concatenate([env.data.qpos, env.data.qvel, env.previous_action]),
                              dtype=torch.float32)[None]
        state = layout.from_raw(raw)
        rotation = layout.rotation(state)
        home = torch.as_tensor(env.home, dtype=torch.float32)
        phase = torch.tensor([env.steps + 1 + env.phase_steps])
        angle = 2 * math.pi * task.gait_phase(phase, env.dt, env.gait['period']).float()
        observed = torch.as_tensor(env.observed_command, dtype=torch.float32)[None]
        moving = task.command_moving(observed).float()[:, None]
        x = torch.cat((-rotation[..., 2, :], layout.part(state, 'linear'), layout.part(state, 'angular'),
                       layout.part(state, 'joints') - home, layout.part(state, 'speeds'),
                       layout.part(state, 'previous'), observed, moving * torch.stack((angle.sin(), angle.cos()), -1)),
                      -1)
        with torch.no_grad():
            return actor(((x - mean) / std).clamp(-10, 10))[0].clamp(-env.action_limit, env.action_limit).numpy()

    lanes = [('кандидат 8 (PPO)', film(lambda env, o: candidate(o), config, command, args.seconds)),
             ('выучен через модель мира', film(ours, config, command, args.seconds))]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    frames = []
    for k in range(min(len(frames_) for _, frames_ in lanes)):
        frame = Image.new('RGB', (SIZE[0] * 2, SIZE[1] + 46), (245, 245, 245))
        draw = ImageDraw.Draw(frame)
        for i, (title, lane) in enumerate(lanes):
            image, now, speed, fell = lane[k]
            frame.paste(Image.fromarray(image), (i * SIZE[0], 0))
            text = f'{title}: {speed:.2f} м/с' + (f', упал на {fell:.1f} с' if fell is not None else '')
            draw.text((10 + i * SIZE[0], SIZE[1] + 4), text, fill=(20, 20, 20), font=font)
        draw.text((10, SIZE[1] + 24), f'команда: вперёд {command[0]:.1f} м/с, вбок {command[1]:.1f}, поворот '
                                     f'{command[2]:.1f} рад/с; {lanes[0][1][k][1]:.1f} с', fill=(90, 90, 90), font=font)
        frames.append(frame)
    out = Path(args.out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0, quality=60, method=4)
    poster = out.with_name(out.stem + '-poster.jpg')
    frames[min(len(frames) - 1, 100)].save(poster, quality=82)
    print(f'ролик {out}, обложка {poster}, кадров {len(frames)}')


if __name__ == '__main__':
    main()
