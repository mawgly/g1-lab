"""Torch-resident normalization, rollout and PPO; SB3-compatible policy weights."""
import math

import torch

# Soft limit of the bounds loss on the action mean, relative to the action box (actions are clipped to it).
BOUND = 1.1


class RunningMoments:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float64, device=device)

    def update(self, values):
        values = values.double()
        count = values.shape[0]
        mean, var = values.mean(0), values.var(0, unbiased=False)
        delta, total = mean - self.mean, self.count + count
        self.var = (self.var * self.count + var * count + delta.square() * self.count * count / total) / total
        self.mean = self.mean + delta * count / total
        self.count = total

    def import_rms(self, rms):
        for name in ('mean', 'var', 'count'):
            setattr(self, name, torch.as_tensor(getattr(rms, name), dtype=torch.float64, device=self.mean.device).clone())

    def export_rms(self, rms):
        rms.mean = self.mean.detach().cpu().numpy().copy()
        rms.var = self.var.detach().cpu().numpy().copy()
        rms.count = self.count.item()


class TensorNormalize:
    def __init__(self, obs_shape, n_envs, device, gamma=.99):
        self.obs = RunningMoments(obs_shape, device)
        self.ret = RunningMoments((), device)
        self.returns = torch.zeros(n_envs, dtype=torch.float64, device=device)
        self.gamma = gamma

    def observation(self, value, update=True):
        if update:
            self.obs.update(value)
        return ((value - self.obs.mean) / torch.sqrt(self.obs.var + 1e-8)).clamp(-10, 10).float()

    def reward(self, value, done):
        self.returns = self.returns * self.gamma + value
        self.ret.update(self.returns)
        normalized = (value / torch.sqrt(self.ret.var + 1e-8)).clamp(-10, 10).float()
        self.returns.masked_fill_(done, 0)
        return normalized

    def import_normalizer(self, normalizer):
        self.obs.import_rms(normalizer.obs_rms)
        if normalizer.gamma == self.gamma:
            self.ret.import_rms(normalizer.ret_rms)

    def export_normalizer(self, normalizer):
        self.obs.export_rms(normalizer.obs_rms)
        self.ret.export_rms(normalizer.ret_rms)
        normalizer.gamma = self.gamma


def generalized_advantage(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        alive = (~dones[step]).float()
        following = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * following * alive - values[step]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[step] = carry
    return advantages, advantages + values


class TensorPPO:
    def __init__(self, model, env, config, normalizer, mirror=None, bounds=None):
        """mirror: symmetry.Mirror of the task, required when config['mirror_coef'] > 0.
        bounds: (low, high) per action for the bounds loss instead of the action box (range_bounds)."""
        self.model, self.env, self.config, self.normalizer = model, env, config, normalizer
        self.policy = model.policy
        self.device = env.device
        if config.get('mirror_coef') and mirror is None:
            raise ValueError('mirror_coef requires the task mirror')
        self.mirror = mirror.to(self.device) if mirror is not None else None
        self.obs = normalizer.observation(env.reset())
        shape = (config['n_steps'], env.n_envs)
        self.obs_buffer = torch.empty((*shape, *env.observation_space.shape), device=self.device)
        self.actions = torch.empty((*shape, *env.action_space.shape), device=self.device)
        self.values = torch.empty(shape, device=self.device)
        self.log_probs = torch.empty(shape, device=self.device)
        self.rewards = torch.empty(shape, device=self.device)
        self.dones = torch.empty(shape, dtype=torch.bool, device=self.device)
        self.low = torch.as_tensor(env.action_space.low, device=self.device)
        self.high = torch.as_tensor(env.action_space.high, device=self.device)
        self.bound_low, self.bound_high = (self.low, self.high) if bounds is None else (
            torch.as_tensor(bound, dtype=self.low.dtype, device=self.device) for bound in bounds)

    @torch.no_grad()
    def collect(self):
        self.policy.set_training_mode(False)
        completed = torch.zeros((), device=self.device)
        episode_return = torch.zeros((), device=self.device)
        episode_length = torch.zeros((), device=self.device)
        for step in range(self.config['n_steps']):
            actions, values, log_probs = self.policy(self.obs)
            self.obs_buffer[step].copy_(self.obs)
            self.actions[step].copy_(actions)
            self.values[step].copy_(values.flatten())
            self.log_probs[step].copy_(log_probs)
            obs, reward, terminated, truncated, terminal_obs = self.env.step(actions.clamp(self.low, self.high))
            done = terminated | truncated
            self.obs = self.normalizer.observation(obs)
            rewards = self.normalizer.reward(reward, done)
            terminal_values = self.policy.predict_values(self.normalizer.observation(terminal_obs, update=False)).flatten()
            rewards = rewards + self.config['gamma'] * terminal_values * (truncated & ~terminated)
            self.rewards[step].copy_(rewards)
            self.dones[step].copy_(done)
            completed += done.sum()
            episode_return += (self.env.last_episode_returns * done).sum()
            episode_length += (self.env.last_episode_lengths * done).sum()
        next_value = self.policy.predict_values(self.obs).flatten()
        self.advantages, self.targets = generalized_advantage(self.rewards, self.values, self.dones, next_value,
                                                             self.config['gamma'], self.config['gae_lambda'])
        self.env.check_health()
        self.model.num_timesteps += self.rewards.numel()
        count = completed.item()
        metrics = {'rollout/ep_rew_mean': episode_return.item() / count if count else float('nan'),
                   'rollout/ep_len_mean': episode_length.item() / count if count else float('nan')}
        if hasattr(self.env, 'rollout_metrics'):  # task means over this rollout (G1: yaw_drift_abs, self_contact, arm_press)
            metrics.update(self.env.rollout_metrics())
        return metrics

    def update(self):
        self.policy.set_training_mode(True)
        cfg = self.config
        obs = self.obs_buffer.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        old_log = self.log_probs.flatten()
        advantages, returns = self.advantages.flatten(), self.targets.flatten()
        stats = []
        stop = False
        for epoch in range(cfg['n_epochs']):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), cfg['batch_size']):
                batch = indices[start:start + cfg['batch_size']]
                values, log_prob, entropy = self.policy.evaluate_actions(obs[batch], actions[batch])
                adv = advantages[batch]
                if len(batch) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = log_prob - old_log[batch]
                ratio = log_ratio.exp()
                pg = -torch.minimum(adv * ratio, adv * ratio.clamp(1 - cfg['clip_range'], 1 + cfg['clip_range'])).mean()
                vf = (returns[batch] - values.flatten()).square().mean()
                ent = -entropy.mean()
                loss = pg + .5 * vf + cfg['ent_coef'] * ent
                mirror_loss = torch.zeros((), device=self.device)
                if cfg.get('bounds_coef') or cfg.get('mirror_coef'):
                    mean = self.policy.get_distribution(obs[batch]).distribution.mean
                if cfg.get('bounds_coef'):
                    # rl_games bounds loss: actions are clipped to [-1, 1] before the environment, so a mean
                    # drifting far outside gets no corrective gradient and the joint becomes a bang-bang switch.
                    # With range_bounds the box is where the motor target reaches the joint's range.
                    outside = ((mean - BOUND * self.bound_high).clamp_min(0).square()
                               + (mean - BOUND * self.bound_low).clamp_max(0).square())
                    loss = loss + cfg['bounds_coef'] * outside.sum(dim=1).mean()
                if cfg.get('mirror_coef'):
                    # Mirror loss (rsl_rl symmetry): the mean action for the mirrored state should be the mirrored
                    # mean action. States are mirrored before normalization, whose statistics are not symmetric.
                    moments = self.normalizer.obs
                    scale = torch.sqrt(moments.var + 1e-8)
                    raw = obs[batch].double() * scale + moments.mean
                    mirrored = ((self.mirror.observation(raw) - moments.mean) / scale).clamp(-10, 10).float()
                    mirrored_mean = self.policy.get_distribution(mirrored).distribution.mean
                    mirror_loss = (mirrored_mean - self.mirror.action(mean).detach()).square().mean()
                    loss = loss + cfg['mirror_coef'] * mirror_loss
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > cfg['clip_range']).float().mean()
                stats.append(torch.stack((pg.detach(), vf.detach(), ent.detach(), kl, clip_fraction,
                                          mirror_loss.detach())))
                # A scalar safety check is intentionally allowed; trajectory tensors stay on CUDA.
                if cfg['target_kl'] is not None and kl.item() > 1.5 * cfg['target_kl']:
                    stop = True
                    break
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                self.policy.optimizer.step()
                if cfg.get('max_std'):
                    # With clipped actions a wider Gaussian costs nothing, so the entropy bonus can grow the
                    # state-independent std without bound (seen: sigma 1e10); cap it.
                    with torch.no_grad():
                        self.policy.log_std.clamp_(max=math.log(cfg['max_std']))
            self.model._n_updates += 1
            if stop:
                break
        means = torch.stack(stats).mean(0).detach().cpu().tolist()
        if not all(__import__('math').isfinite(x) for x in means):
            raise RuntimeError('Non-finite PPO update')
        return dict(zip(('train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss',
                         'train/approx_kl', 'train/clip_fraction', 'train/mirror_loss'), means))
