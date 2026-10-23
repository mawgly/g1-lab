import time

import glfw
import gymnasium as gym
import mujoco
import numpy as np
import torch

from .core import StageSelection, list_stages, load_stage, read_json


class ViewControls:
    def __init__(self):
        self.selection = StageSelection()
        self.paused = False
        self.reset_requested = False

    def command(self, command, stages):
        if command in ('left', 'right'):
            self.selection.move(-1 if command == 'left' else 1, stages)
            self.reset_requested = True
        elif command == 'latest':
            self.selection.follow = True
            self.reset_requested = True
        elif command == 'pause':
            self.paused = not self.paused
        elif command == 'reset':
            self.reset_requested = True


def watch(args):
    from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
    torch.set_num_threads(2)
    stages = list_stages(args.run_dir)
    if not stages:
        raise ValueError('No stages yet. Start humanoid_train.cmd, then open humanoid_watch.cmd.')
    controls = ViewControls()
    if args.stage is not None:
        matches = [p for p in stages if int(p.name.split('_')[1]) == args.stage]
        if not matches:
            raise ValueError('Stage not found. Use the list command to see available stages.')
        controls.selection.current = matches[0]
        controls.selection.follow = False
    current = controls.selection.select(stages)
    policy, norm = load_stage(current)
    env = gym.make('Humanoid-v5')
    obs, initial = env.reset(seed=71000)
    viewer = None
    try:
        viewer = WindowViewer(env.unwrapped.model, env.unwrapped.data, width=1100, height=800)
        glfw.set_window_title(viewer.window, 'Humanoid - training stages | Left/Right: history | L: live')
        viewer._render_every_frame = True
        viewer._create_overlay = lambda: None
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        original_key = viewer._key_callback

        def key_callback(window, key, scancode, action, mods):
            if action != glfw.RELEASE:
                return
            mapping = {glfw.KEY_LEFT: 'left', glfw.KEY_RIGHT: 'right', glfw.KEY_L: 'latest',
                       glfw.KEY_SPACE: 'pause', glfw.KEY_R: 'reset'}
            if key in mapping:
                controls.command(mapping[key], stages)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_TAB:
                original_key(window, key, scancode, action, mods)

        glfw.set_key_callback(viewer.window, key_callback)
        start = time.monotonic()
        poll = 0.0
        episode_steps = 0
        distance = 0.0
        episode = 0
        done_at = None
        metrics = read_json(current / 'evaluation.json', {})
        status = {}
        print(f'Viewer ready: {current.name}. Left/Right history, L live, Space pause, R reset.', flush=True)
        while not glfw.window_should_close(viewer.window):
            tick = time.monotonic()
            if args.seconds is not None and tick - start >= args.seconds:
                break
            if tick - poll >= 1:
                stages = list_stages(args.run_dir)
                status = read_json(args.run_dir / 'status.json', {})
                metrics = read_json(current / 'evaluation.json', {})
                poll = tick
            if controls.reset_requested or (done_at is not None and tick >= done_at and not controls.paused):
                chosen = controls.selection.select(stages)
                if chosen != current:
                    next_policy, next_norm = load_stage(chosen)
                    norm.close()
                    policy, norm, current = next_policy, next_norm, chosen
                    metrics = read_json(current / 'evaluation.json', {})
                    print(f'Watching {current.name}', flush=True)
                episode += 1
                obs, initial = env.reset(seed=71000 + episode % 5)
                episode_steps = 0
                distance = 0.0
                done_at = None
                controls.reset_requested = False
            if not controls.paused and done_at is None:
                action = policy.predict(norm.normalize_obs(obs), deterministic=True)[0]
                obs, _, terminated, truncated, info = env.step(action)
                episode_steps += 1
                distance = float(info['x_position'] - initial['x_position'])
                if terminated or truncated:
                    done_at = tick + 0.5
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            viewer.cam.lookat[:] = [env.unwrapped.data.qpos[0], env.unwrapped.data.qpos[1], 1.0]
            stage_steps = int(current.name.split('_')[1])
            state = status.get('state', 'unknown')
            if state in ('training', 'saving', 'starting') and time.time() - status.get('updated_at', 0) > 30:
                state = 'no recent update - check training.log'
            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'Humanoid-v5\nWatching stage\nMode\nTraining\nCurrent episode\nForward movement',
                               f'PPO / MuJoCo\n{stage_steps:,} steps\n'
                               f'{"PAUSED" if controls.paused else "LIVE latest" if controls.selection.follow else "HISTORY"}\n'
                               f'{state}: {status.get("steps", 0):,} / {status.get("target_steps", 0):,}\n'
                               f'{episode_steps * env.unwrapped.dt:.2f} s\n{distance:.2f} m')
            viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               'Left / Right\nL\nSpace\nR\nEsc\nMouse / wheel',
                               'Earlier / later stage\nFollow latest stage\nPause / resume\nRestart attempt\nClose viewer\nRotate / zoom')
            if metrics:
                viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                   'Fixed-seed evaluation\nMean reward\nMean episode time\nMean forward distance',
                                   f'5 attempts\n{metrics["mean_return"]:.1f}\n{metrics["mean_seconds"]:.2f} s\n{metrics["mean_forward_m"]:.2f} m')
            capture_overlays = {key: value[:] for key, value in viewer._overlays.items()} if args.screenshot else None
            viewer.render()
            if args.screenshot:
                from PIL import Image
                width, height = glfw.get_framebuffer_size(viewer.window)
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                # WindowViewer swaps buffers before returning; redraw the back buffer
                # before reading it, including the same HUD used in the window.
                mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.con)
                for gridpos, (left, right) in capture_overlays.items():
                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos,
                                       viewer.viewport, left, right, viewer.con)
                mujoco.mjr_readPixels(rgb, None, viewer.viewport, viewer.con)
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.flipud(rgb)).save(args.screenshot)
                args.screenshot = None
            time.sleep(max(0, env.unwrapped.dt - (time.monotonic() - tick)))
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        norm.close()
