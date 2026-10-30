"""A clip of the reach skill before it goes to the arm: left our MuJoCo SO-101 with sysid.py's fit running the policy
(physics), right the world model's imagined poses (drawn through the same model's kinematics), the same policy, the
same start (a recorded state) and the same held-out targets one after another; a green ball marks the target, the
footer gives the distance and the hit (2 cm, still, 5 ticks) per panel. The goal is rate-limited as on the arm.

    .venv\\Scripts\\python.exe scripts\\so101\\reach_video.py --policy runs\\so101_real\\reach-smooth
        [--fit runs\\humanoid_lab\\world_model\\real-arm\\fit-both] [--targets 6] [--out <file>.webp]"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'wm'))
sys.path.insert(0, str(ROOT / 'scripts' / 'so101'))

import reach_real  # noqa: E402
import sysid  # noqa: E402

TICKS, SIZE = 100, (400, 320)


def main():
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import fit as fitting
    import imagine
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_adapters.arm import ArmReach
    parser = argparse.ArgumentParser(description='ролик навыка «дотянуться»: симуляция и модель мира')
    parser.add_argument('--policy', required=True)
    parser.add_argument('--fit', default=str(ROOT / 'runs/humanoid_lab/world_model/real-arm/fit-both'))
    parser.add_argument('--data', default=str(ROOT / 'runs/humanoid_lab/world_model/real-arm/data-both'))
    parser.add_argument('--sysid', default=str(ROOT / 'runs/so101_real/sysid.json'))
    parser.add_argument('--targets', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    saved = torch.load(Path(args.policy) / 'policy.pt', map_location='cpu', weights_only=False)
    policy = imagine.Policy('cpu')
    policy.net.load_state_dict(saved['net'])
    step = float(saved['step'])
    model, _ = wm.WorldModel.load(Path(args.fit) / 'model.pt', 'cpu')
    model.eval()
    env = imagine.make_env()
    world = imagine.Imagination(model, env, 'cpu')
    arm = sysid.Arm()
    if Path(args.sysid).is_file():
        arm.apply(json.loads(Path(args.sysid).read_text(encoding='utf-8'))['fit'])
    rng = np.random.default_rng(args.seed)
    tips = np.array(saved['judge_tips'])
    targets = tips[rng.choice(len(tips), args.targets, replace=False)]
    meta, arrays = fitting.load(args.data)
    data = fitting.Data(arrays, 'cpu', .2, ArmReach({}))
    held = arrays['state'][:, data.held_worlds.numpy()].reshape(-1, arrays['state'].shape[-1])
    start = held[rng.integers(len(held))].astype(np.float64)
    low, high = reach_real.BOX[:, 0], reach_real.BOX[:, 1]

    noise = np.random.default_rng(args.seed + 1)
    smooth = bool(saved.get('smooth'))
    smoothed = [np.zeros(5), np.zeros(5)]  # per panel, as reach_real.run

    def act(q, speed, target, tip, panel=0):
        speed = speed + reach_real.SPEED_NOISE * noise.standard_normal(5)  # the arm's speeds are noisy differences
        observation = np.concatenate((np.sin(q), np.cos(q), .1 * speed, 3 * target, 3 * tip, 3 * (target - tip)))
        with torch.no_grad():
            action = policy.act(torch.as_tensor(observation, dtype=torch.float32)[None])[0].numpy().astype(np.float64)
        if smooth:
            smoothed[panel] = reach_real.SMOOTH * smoothed[panel] + (1 - reach_real.SMOOTH) * action
            return smoothed[panel]
        return action

    sim_model, sim = arm.model, arm.data
    sim.qpos[:] = env.home
    sim.qpos[arm.qadr] = start[:5]
    sim.qvel[arm.vadr] = start[6:11]
    sim.ctrl[:] = env.home
    mujoco.mj_forward(sim_model, sim)
    imagined = torch.as_tensor(start, dtype=torch.float32)[None]
    view = mujoco.MjData(env.model)
    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    sim_renderer = mujoco.Renderer(sim_model, SIZE[1], SIZE[0])
    frames, summary = [], []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)  # the default bitmap font has no Cyrillic
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.1, 0, .2]
    camera.distance, camera.azimuth, camera.elevation = .95, 150, -18
    sim_goal, model_goal = start[:5].copy(), start[:5].copy()
    for number, target in enumerate(targets):
        hits, holds = [None, None], [0, 0]
        for tick in range(TICKS):
            # physics
            q, speed = sim.qpos[arm.qadr].copy(), sim.qvel[arm.vadr].copy()
            tip = sim.site_xpos[env.tip_id].copy()
            sim_goal = reach_real.limited(q + step * act(q, speed, target, tip), sim_goal, low, high, reach_real.RATE)
            sim.ctrl[:5] = sim_goal
            sim.mocap_pos[0] = target
            mujoco.mj_step(sim_model, sim, nstep=sysid.SUBSTEPS)
            sim_distance = float(np.linalg.norm(sim.site_xpos[env.tip_id] - target))
            # the world model
            mq = imagined[0, :5].numpy().astype(np.float64)
            mspeed = imagined[0, 6:11].numpy().astype(np.float64)
            mtip = world.tip(imagined)[0].numpy().astype(np.float64)
            model_goal = reach_real.limited(mq + step * act(mq, mspeed, target, mtip, 1), model_goal, low, high,
                                            reach_real.RATE)
            u = torch.as_tensor(np.concatenate((model_goal - mq, [world.gripper_home - float(imagined[0, 5])])),
                                dtype=torch.float32)[None]
            with torch.no_grad():
                imagined = model.step(imagined, u, torch.zeros((1, 0)))
            model_distance = float(world.terms.distance(imagined, torch.as_tensor(target, dtype=torch.float32)[None]))
            for k, (distance, velocity) in enumerate(((sim_distance, sim.qvel[arm.vadr]),
                                                      (model_distance, imagined[0, 6:11].numpy()))):
                still = distance < .02 and np.abs(velocity).max() < reach_real.STILL
                holds[k] = holds[k] + 1 if still else 0
                if holds[k] >= 5 and hits[k] is None:
                    hits[k] = tick
            # frames
            sim_renderer.update_scene(sim, camera=camera)
            left = Image.fromarray(sim_renderer.render())
            view.qpos[:] = env.home
            view.qpos[arm.qadr] = imagined[0, :5].numpy()
            view.mocap_pos[0] = target
            mujoco.mj_forward(env.model, view)
            renderer.update_scene(view, camera=camera)
            right = Image.fromarray(renderer.render())
            frame = Image.new('RGB', (2 * SIZE[0], SIZE[1] + 40), (245, 245, 245))
            frame.paste(left, (0, 0))
            frame.paste(right, (SIZE[0], 0))
            draw = ImageDraw.Draw(frame)
            for k, (title, distance) in enumerate((('симуляция (подогнанная)', sim_distance),
                                                   ('модель мира руки', model_distance))):
                mark = f', есть за {hits[k] * .04:.1f} с' if hits[k] is not None else ''
                draw.text((10 + k * SIZE[0], SIZE[1] + 4), f'{title}: цель {number + 1}, {distance * 100:.1f} см{mark}',
                          fill=(20, 20, 20), font=font)
            draw.text((10, SIZE[1] + 22), f'{tick * .04:.1f} с', fill=(90, 90, 90), font=font)
            frames.append(frame)
        summary.append(dict(target=number + 1, sim_hit=hits[0], model_hit=hits[1], sim_cm=round(sim_distance * 100, 1),
                            model_cm=round(model_distance * 100, 1)))
        print(summary[-1], flush=True)
    out = Path(args.out) if args.out else Path(args.policy) / 'reach.webp'
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0, quality=70, method=4)
    print(f'ролик: {out} ({len(frames)} кадров)')


if __name__ == '__main__':
    main()
