from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'so101'


class ArmReachEnv(gym.Env):
    """Five position-controlled arm axes, fixed open gripper, 3D reaching."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 25}

    def __init__(self, render_mode=None, max_steps=150):
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported render mode')
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        tree = ET.parse(MODEL_DIR / 'so101_new_calib.xml')
        root = tree.getroot()
        root.find('compiler').set('meshdir', str(MODEL_DIR / 'assets'))
        ET.SubElement(root, 'option', timestep='0.002', integrator='implicitfast')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth='960', offheight='720')
        ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.7 0.7 0.7')
        world = root.find('worldbody')
        ET.SubElement(world, 'light', pos='0.3 -0.5 1.5', dir='0 0 -1')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='1 1 0.02', rgba='0.16 0.21 0.27 1')
        target = ET.SubElement(world, 'body', name='target', mocap='true', pos='0.3 0 0.2')
        ET.SubElement(target, 'geom', type='sphere', size='0.015', rgba='0.1 0.9 0.5 0.8', contype='0', conaffinity='0')
        tip = root.find(".//site[@name='gripperframe']")
        tip.set('group', '0')
        tip.set('size', '0.007')
        tip.set('rgba', '1 0.25 0.15 1')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.probe = mujoco.MjData(self.model)
        self.tip_id = self.model.site('gripperframe').id
        self.home = np.array([0, -0.5, 0.5, 0.5, 0, 0.8])
        self.target = np.zeros(3)
        self.goal_qpos = self.home.copy()
        self.action_space = spaces.Box(-1, 1, (5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32)
        self.renderer = None
        self.steps = self.hold = 0
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.16, 0, 0.16]
        self.camera.distance = 0.95
        self.camera.azimuth = 135
        self.camera.elevation = -25

    def _observation(self):
        q = self.data.qpos[:5]
        tip = self.data.site_xpos[self.tip_id]
        return np.concatenate((np.sin(q), np.cos(q), self.data.qvel[:5] * 0.1,
                               self.target * 3, tip * 3, (self.target - tip) * 3)).astype(np.float32)

    def _info(self):
        return {'distance': float(np.linalg.norm(self.target - self.data.site_xpos[self.tip_id])),
                'is_success': bool(self.hold >= 5)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(1000):
            self.goal_qpos = self.home.copy()
            self.goal_qpos[:5] += self.np_random.uniform(-1, 1, 5) * [1.0, 0.65, 0.8, 0.8, 1.0]
            self.probe.qpos[:] = self.goal_qpos
            mujoco.mj_forward(self.model, self.probe)
            pos = self.probe.site_xpos[self.tip_id]
            if self.probe.ncon == 0 and 0.08 < pos[2] < 0.4 and pos[0] > 0.1:
                self.target[:] = pos
                break
        else:
            raise RuntimeError('Could not sample a collision-free reachable target')
        self.data.qpos[:] = self.home
        self.data.qpos[:5] += self.np_random.uniform(-0.05, 0.05, 5)
        self.data.ctrl[:] = self.data.qpos
        self.data.mocap_pos[0] = self.target
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=100)
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold = 0
        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError('Action must contain five finite numbers')
        action = np.clip(action, -1, 1)
        before = self._info()['distance']
        limits = self.model.actuator_ctrlrange[:5]
        self.data.ctrl[:5] = np.clip(self.data.qpos[:5] + action * 0.08, limits[:, 0], limits[:, 1])
        self.data.ctrl[5] = self.home[5]
        self.data.mocap_pos[0] = self.target
        mujoco.mj_step(self.model, self.data, nstep=20)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        distance = self._info()['distance']
        self.hold = self.hold + 1 if distance < 0.02 else 0
        success = self.hold >= 5
        reward = 20 * (before - distance) - 10 * distance - 0.005 * float(action @ action)
        reward += 0.5 * np.exp(-distance * 50) + (10 if success else 0)
        return self._observation(), float(reward), success, self.steps >= self.max_steps and not success, self._info()

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=960)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
