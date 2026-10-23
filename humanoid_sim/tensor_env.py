"""CUDA-resident Humanoid task for an end-to-end torch learner.

Newton replaces unsupported PGS; RK4 and default Humanoid-v5 task equations
remain. Reset RNG is torch CUDA, intentionally different from Gym's NumPy RNG.
No host/device state transfers occur in reset/step; check_health synchronizes
once when explicitly called by the learner (typically once per rollout).
"""
import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


class TensorHumanoidEnv:
    backend_metadata = {
        'backend': 'mujoco-warp-torch', 'device': 'cuda:0', 'precision': 'float32',
        'solver': 'Newton', 'gym_default_solver': 'PGS', 'integrator': 'RK4',
        'task': 'Humanoid-v5', 'variant': 'Newton solver; CUDA reset RNG; float32',
        'host_device_transfer': 'none during reset/step',
    }

    def __init__(self, n_envs, seed=0, max_episode_steps=1000):
        if n_envs < 1 or max_episode_steps < 1:
            raise ValueError('n_envs and max_episode_steps must be positive')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch is required; use .venv-fullgpu')
        self.device = torch.device('cuda:0')
        self.n_envs = self.num_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self.reference = gym.make('Humanoid-v5').unwrapped
        self.reference.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.observation_space = self.reference.observation_space
        self.action_space = self.reference.action_space
        self.dt = self.reference.dt
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        wp.init()
        # CUDA cannot capture on Torch's legacy default (null) stream.
        # Build graphs on a dedicated Warp stream, then launch on the caller's
        # Torch stream at runtime. Startup synchronization bridges allocations.
        self._stream = wp.Stream('cuda:0')
        with wp.ScopedStream(self._stream):
            self.model = mjw.put_model(self.reference.model)
            self.data = mjw.make_data(self.reference.model, nworld=n_envs, nconmax=64, njmax=256)
            mjw.forward(self.model, self.data)
            mjw.step(self.model, self.data)
            mjw.rne_postconstraint(self.model, self.data)
            wp.synchronize_device()
            with wp.ScopedCapture() as capture:
                for _ in range(self.reference.frame_skip):
                    mjw.step(self.model, self.data)
                mjw.rne_postconstraint(self.model, self.data)
            self.step_graph = capture.graph
            self.reset_mask = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            self._warp_reset_mask = wp.from_torch(self.reset_mask)
            torch.cuda.synchronize(self.device)
            mjw.reset_data(self.model, self.data, self._warp_reset_mask)
            with wp.ScopedCapture() as capture:
                mjw.reset_data(self.model, self.data, self._warp_reset_mask)
            self.reset_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mjw.forward(self.model, self.data)
            self.forward_graph = capture.graph
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
            self.cinert = wp.to_torch(self.data.cinert)
            self.cvel = wp.to_torch(self.data.cvel)
            self.qfrc_actuator = wp.to_torch(self.data.qfrc_actuator)
            self.cfrc_ext = wp.to_torch(self.data.cfrc_ext)
            self.xipos = wp.to_torch(self.data.xipos)
            self.overflow = wp.to_torch(self.data.overflow)
            wp.synchronize_device()
        self.initial_qpos = torch.tensor(self.reference.init_qpos, dtype=torch.float32, device=self.device)
        self.initial_qvel = torch.tensor(self.reference.init_qvel, dtype=torch.float32, device=self.device)
        mass = torch.tensor(self.reference.model.body_mass, dtype=torch.float32, device=self.device)
        self.mass_weights = mass / mass.sum()
        self.lengths = torch.zeros(n_envs, dtype=torch.int64, device=self.device)
        self.returns = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        self.last_episode_lengths = self.lengths.clone()
        self.last_episode_returns = self.returns.clone()
        self._centers = torch.zeros((n_envs, 2), dtype=torch.float32, device=self.device)
        self._overflow_seen = torch.zeros_like(self.overflow)
        self._bad_state_seen = torch.zeros((), dtype=torch.bool, device=self.device)

    def _obs(self):
        return torch.cat((self.qpos[:, 2:], self.qvel,
            self.cinert[:, 1:].reshape(self.n_envs, -1),
            self.cvel[:, 1:].reshape(self.n_envs, -1),
            self.qfrc_actuator[:, 6:],
            self.cfrc_ext[:, 1:].reshape(self.n_envs, -1)), dim=1)

    def _center(self):
        return (self.xipos[:, :, :2] * self.mass_weights[None, :, None]).sum(dim=1)

    def _masked_reset(self, mask):
        self.reset_mask.copy_(mask)
        wp.capture_launch(self.reset_graph)
        qpos = self.initial_qpos + .02 * torch.rand(self.qpos.shape, generator=self.generator, device=self.device) - .01
        qvel = self.initial_qvel + .02 * torch.rand(self.qvel.shape, generator=self.generator, device=self.device) - .01
        self.qpos.copy_(torch.where(mask[:, None], qpos, self.qpos))
        self.qvel.copy_(torch.where(mask[:, None], qvel, self.qvel))
        self.lengths.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)
        # Forward is batched even when no worlds reset, avoiding a host done.any().
        wp.capture_launch(self.forward_graph)
        # Gym computes velocity from prior post-step derived COM. A forward on
        # survivors refreshes derived fields, so preserve that history explicitly.
        self._centers.copy_(torch.where(mask[:, None], self._center(), self._centers))

    @torch.no_grad()
    def reset(self):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self._masked_reset(torch.ones(self.n_envs, dtype=torch.bool, device=self.device))
            self._overflow_seen.zero_()
            self._bad_state_seen.zero_()
            return self._obs()

    @torch.no_grad()
    def step(self, actions):
        if actions.device != self.device or actions.shape != (self.n_envs, 17):
            raise ValueError('Expected CUDA:0 actions with shape (n_envs, 17)')
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            before = self._centers.clone()
            self.ctrl.copy_(actions)
            wp.capture_launch(self.step_graph)
            terminal_obs = self._obs()
            after = self._center()
            velocity = (after - before) / self.dt
            self._centers.copy_(after)
            healthy = (self.qpos[:, 2] > 1.0) & (self.qpos[:, 2] < 2.0)
            reward = 1.25 * velocity[:, 0] + 5.0 * healthy
            reward -= .1 * self.ctrl.square().sum(dim=1)
            reward -= (5e-7 * self.cfrc_ext.square().sum(dim=(1, 2))).clamp(0, 10)
            self.lengths.add_(1)
            self.returns.add_(reward)
            terminated = ~healthy
            truncated = self.lengths >= self.max_episode_steps
            done = terminated | truncated
            self.last_episode_lengths.copy_(self.lengths)
            self.last_episode_returns.copy_(self.returns)
            self._overflow_seen.bitwise_or_(self.overflow)
            self._bad_state_seen.logical_or_(~torch.isfinite(terminal_obs).all() | ~torch.isfinite(reward).all())
            self._masked_reset(done)
            next_obs = torch.where(done[:, None], self._obs(), terminal_obs)
            return next_obs, reward, terminated, truncated, terminal_obs

    def check_health(self):
        capacity_mask = int(mjw.OverflowType.ALL) & ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        if bool((self._overflow_seen & capacity_mask).any().item()):
            raise RuntimeError('MuJoCo Warp allocation overflow; increase contact/constraint capacity')
        if bool(self._bad_state_seen.item()):
            raise RuntimeError('Nonfinite simulation state or reward')
        self._overflow_seen.zero_()
        self._bad_state_seen.zero_()

    def close(self):
        self.reference.close()
