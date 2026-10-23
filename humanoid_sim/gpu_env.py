"""Batched CUDA physics for the default Gymnasium Humanoid-v5 task.

Physics uses MuJoCo Warp float32 with Newton (default Gym PGS unsupported). The SB3 boundary copies observations/actions
between host and device; policy training can remain CPU. Not bitwise CPU parity.
"""
import time
import gymnasium as gym
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from stable_baselines3.common.vec_env import VecEnv


class GPUHumanoidVecEnv(VecEnv):
    backend_metadata = {
        'backend': 'mujoco-warp', 'device': 'cuda:0', 'precision': 'float32',
        'solver': 'Newton', 'gym_default_solver': 'PGS', 'integrator': 'RK4',
        'task': 'Humanoid-v5', 'variant': 'Newton solver; CPU/GPU numerical differences',
        'host_device_transfer': 'numpy actions and observations each vector step',
    }

    def __init__(self, n_envs=256, seed=0, max_episode_steps=1000):
        if n_envs < 1 or max_episode_steps < 1:
            raise ValueError('n_envs and max_episode_steps must be positive')
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError('MuJoCo Warp requires an NVIDIA CUDA GPU')
        self.device = wp.get_device('cuda:0')
        self.reference = gym.make('Humanoid-v5').unwrapped
        # Warp does not implement Gym XML PGS; Newton is an explicit backend variant.
        self.reference.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.render_mode = None
        self.max_episode_steps = max_episode_steps
        self.base_seed = seed
        self.rngs = [np.random.default_rng(seed + i) for i in range(n_envs)]
        self.lengths = np.zeros(n_envs, dtype=np.int64)
        self.returns = np.zeros(n_envs, dtype=np.float64)
        self.started = time.perf_counter()
        super().__init__(n_envs, self.reference.observation_space, self.reference.action_space)
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.reference.model)
            self.data = mjw.make_data(self.reference.model, nworld=n_envs, nconmax=64, njmax=256)
            # Capture RK4 and its exact Gym frame skip without changing solver settings.
            mjw.forward(self.model, self.data)
            mjw.step(self.model, self.data)
            mjw.rne_postconstraint(self.model, self.data)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(self.reference.frame_skip):
                    mjw.step(self.model, self.data)
                mjw.rne_postconstraint(self.model, self.data)
            self.step_graph = capture.graph
            self.reset_mask = wp.zeros(n_envs, dtype=wp.bool)
            mjw.reset_data(self.model, self.data, self.reset_mask)
            with wp.ScopedCapture() as capture:
                mjw.reset_data(self.model, self.data, self.reset_mask)
            self.reset_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mjw.forward(self.model, self.data)
            self.forward_graph = capture.graph
        self._actions = None

    def _obs(self):
        d = self.data
        n = self.num_envs
        return np.concatenate((d.qpos.numpy()[:, 2:], d.qvel.numpy(),
            d.cinert.numpy()[:, 1:].reshape(n, -1),
            d.cvel.numpy()[:, 1:].reshape(n, -1),
            d.qfrc_actuator.numpy()[:, 6:],
            d.cfrc_ext.numpy()[:, 1:].reshape(n, -1)), axis=1)

    def _center(self):
        mass = self.reference.model.body_mass
        return (self.data.xipos.numpy() * mass[None, :, None]).sum(axis=1)[:, :2] / mass.sum()

    def _reset_indices(self, indices):
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[indices] = True
        self.reset_mask.assign(mask)
        wp.capture_launch(self.reset_graph)
        qpos, qvel = self.data.qpos.numpy(), self.data.qvel.numpy()
        for i in indices:
            qpos[i] = self.reference.init_qpos + self.rngs[i].uniform(-0.01, 0.01, self.reference.model.nq)
            qvel[i] = self.reference.init_qvel + self.rngs[i].uniform(-0.01, 0.01, self.reference.model.nv)
            self.lengths[i] = 0
            self.returns[i] = 0
            self.reset_infos[i] = {'x_position': float(qpos[i, 0]), 'y_position': float(qpos[i, 1])}
        self.data.qpos.assign(qpos)
        self.data.qvel.assign(qvel)
        wp.capture_launch(self.forward_graph)

    def reset(self):
        if any(s is not None for s in self._seeds):
            self.rngs = [np.random.default_rng(s) for s in self._seeds]
        self._reset_seeds()
        self._reset_options()
        self._reset_indices(np.arange(self.num_envs))
        return self._obs()

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 17) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions with shape (n_envs, 17)')
        self._actions = actions.copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError('step_async must precede step_wait')
        before = self._center()
        self.data.ctrl.assign(self._actions)
        wp.capture_launch(self.step_graph)
        # Capacity overflows invalidate transitions: fail instead of training silently.
        overflow = self.data.overflow.numpy()
        capacity_mask = int(mjw.OverflowType.ALL) & ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        if np.any(overflow & capacity_mask):
            raise RuntimeError('MuJoCo Warp allocation overflow; increase contact/constraint capacity')
        obs = self._obs()
        qpos = self.data.qpos.numpy()
        velocity = (self._center() - before) / self.reference.dt
        healthy = (qpos[:, 2] > 1.0) & (qpos[:, 2] < 2.0)
        forward = 1.25 * velocity[:, 0]
        survive = 5.0 * healthy
        control = 0.1 * np.square(self._actions).sum(axis=1)
        contact = np.clip(5e-7 * np.square(self.data.cfrc_ext.numpy()).sum(axis=(1, 2)), 0, 10)
        rewards = forward + survive - control - contact
        self.lengths += 1
        self.returns += rewards
        truncated = self.lengths >= self.max_episode_steps
        dones = ~healthy | truncated
        infos = []
        for i in range(self.num_envs):
            info = dict(x_position=float(qpos[i, 0]), y_position=float(qpos[i, 1]),
                        x_velocity=float(velocity[i, 0]), y_velocity=float(velocity[i, 1]),
                        reward_survive=float(survive[i]), reward_forward=float(forward[i]),
                        reward_ctrl=float(-control[i]), reward_contact=float(-contact[i]))
            info['TimeLimit.truncated'] = bool(truncated[i] and healthy[i])
            if dones[i]:
                info['terminal_observation'] = obs[i].copy()
                info['episode'] = {'r': float(self.returns[i]), 'l': int(self.lengths[i]),
                                   't': time.perf_counter() - self.started}
            infos.append(info)
        if dones.any():
            self._reset_indices(np.flatnonzero(dones))
            obs[dones] = self._obs()[dones]
        self._actions = None
        return obs, rewards.astype(np.float32), dones, infos

    def close(self):
        self.reference.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, getattr(self.reference, attr_name, None)) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise NotImplementedError('Per-world attribute changes are unsupported')
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError('Arbitrary Gym methods are unsupported on batched physics')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def create_gpu_env(n_envs=256, seed=0, max_episode_steps=1000):
    return GPUHumanoidVecEnv(n_envs=n_envs, seed=seed, max_episode_steps=max_episode_steps)

