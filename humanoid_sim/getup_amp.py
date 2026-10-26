"""AMP style reward for g1_getup on the full-GPU path (iteration 55; the critic's spec; Peng et al., arXiv 2104.02180).

A discriminator D learns to tell pairs of neighbouring states (s_t, s_t+1) of a human getting up from the policy's, and
the policy is paid w·[0.3 < head < 1.1]·max(0, 1 − 0.25·(D − 1)²) for its step (reward weight amp of g1_getup's
task_config; LSGAN, human → +1, policy → −1). The reward is added by TensorPPO.collect after the task's own, before
the reward normalization; the task (g1_tensor_math, the CPU task) does not know it, so a stage's CPU evaluation does
not see it: style is judged by probes and by rollout/style_dist (the style library's nearest frame, kept on as a judge).

State s (amp_state, AMP_VERSION), 54 numbers a frame: the 25 free joints' angles (all but the locked wrists), the
gravity direction in the pelvis frame (3), the pelvis height (1), the head top's height (g1_tensor_math.head_height, 1)
and 8 points relative to the pelvis in its heading frame (3 each: the palms at HAND_LOAD_POSITION of the wrist_yaw
links, the elbow, knee and ankle_roll link origins; the heading as stance_width's). A pair is 108 numbers at the 50 Hz
control step.

The human pairs are a library runs/humanoid_lab/getup_style/<style_library>-amp.npz (scripts/lab/getup_amp_library.py,
from LAFAN1's fallAndGetUp clips retargeted to the G1 by Unitree, through mj_kinematics of the getting-up model; pairs
whose second frame has the head in the band; subjects 1-4, subject 5 held out). LAFAN1 is CC BY-NC-ND 4.0: the library,
the discriminator (amp.pt of a stage) and a policy trained with it are never published or put into a client's package.
Arrays only, read without pickle: human (m, 108), heldout (k, 108), joints (25,) int64, header (JSON text).

D: MLP 108 → 512 → 256 → 1 (ReLU), its input normalized by the human library's own mean and std (frozen; the policy's
normalization is untouched) and clipped to ±AMP_INPUT_CLIP. Loss per minibatch: mean((D(h) − 1)²) + mean((D(p) + 1)²)
+ AMP_GRADIENT_PENALTY·mean(‖∇ D(h)‖²) (the gradient on D's normalized input) + AMP_LOGIT_DECAY·‖W_out‖², its own Adam
(lr AMP_LEARNING_RATE). One D step after each PPO minibatch step, the human batch as large as the policy one, both drawn
with replacement: the policy's from a FIFO of the last AMP_BUFFER in-band pairs on the device, the human one from the
whole library on the device. Warm-up: while D has trained fewer than AMP_WARMUP_STEPS environment steps (a fresh D: no
amp.pt in the stage, or one of another feature version or library), the reward's weight is 0.

A stage keeps AMP_FILE (amp.pt: D's weights, its Adam, the input normalization, AMP_VERSION, the library's md5, the
steps D trained), written into the stage's .pending-* folder with the policy (core.save_stage extras), so a branch's
copy of the stage carries it; a resumed run restores D (another version or md5: a fresh D, warm-up again); the buffer is
not kept.
"""
import hashlib
import json
import math

import numpy as np
import torch

from .tasks import g1_getup_style
from .tasks.g1_tensor_math import GETUP_STANCE_HEADING, HAND_LOAD_POSITION, _module

AMP_VERSION = 'getup-amp-v1'  # the state layout below; a library or amp.pt of another one is refused / not restored
AMP_POINTS = (('left_wrist_yaw_link', HAND_LOAD_POSITION), ('right_wrist_yaw_link', HAND_LOAD_POSITION),
              ('left_elbow_link', (0., 0., 0.)), ('right_elbow_link', (0., 0., 0.)),
              ('left_knee_link', (0., 0., 0.)), ('right_knee_link', (0., 0., 0.)),
              ('left_ankle_roll_link', (0., 0., 0.)), ('right_ankle_roll_link', (0., 0., 0.)))
AMP_HIDDEN = (512, 256)
AMP_LEARNING_RATE = 1e-4
AMP_GRADIENT_PENALTY = 5.
AMP_LOGIT_DECAY = 1e-4
AMP_REWARD_SCALE = .25  # max(0, 1 − AMP_REWARD_SCALE·(D − 1)²)
AMP_BUFFER = 1_000_000  # policy pairs
AMP_WARMUP_STEPS = 5_000_000
AMP_INPUT_CLIP = 10.
AMP_MIN_STD = 1e-3  # floor of the library's per-feature std
AMP_FILE = 'amp.pt'


def state_size(joints):
    """Numbers in one frame's amp_state for `joints` free joints."""
    return joints + 5 + 3 * len(AMP_POINTS)


def point_bodies(model):
    """(bodies (8,) int64, offsets (8, 3)) of AMP_POINTS in a MuJoCo model."""
    return (np.array([model.body(name).id for name, _ in AMP_POINTS], dtype=np.int64),
            np.array([offset for _, offset in AMP_POINTS], dtype=np.float64))


def amp_points(xpos, xmat, bodies, offsets):
    """(n, 8, 3) world positions of AMP_POINTS: xpos (n, nbody, 3), xmat (n, nbody, 9 or 3, 3) world <- body,
    bodies / offsets of point_bodies (offsets in the body frame). numpy or torch alike."""
    xp = _module(xpos)
    rotation = xmat[:, bodies].reshape(xpos.shape[0], len(bodies), 3, 3)
    offsets = offsets if xp is np else torch.as_tensor(offsets, dtype=xpos.dtype, device=xpos.device)
    return xpos[:, bodies] + xp.einsum('nkij,kj->nki', rotation, offsets)


def heading_yaw(pelvis_rotation):
    """(n,) yaw of the pelvis heading: its x axis, or its y axis turned −90° where the x axis is within
    GETUP_STANCE_HEADING of vertical (g1_tensor_math.stance_width's rule). numpy or torch alike."""
    xp = _module(pelvis_rotation)
    forward_x, forward_y = pelvis_rotation[:, 0, 0], pelvis_rotation[:, 1, 0]
    steep = forward_x ** 2 + forward_y ** 2 < GETUP_STANCE_HEADING ** 2
    forward_x = xp.where(steep, pelvis_rotation[:, 1, 1], forward_x)
    forward_y = xp.where(steep, -pelvis_rotation[:, 0, 1], forward_y)
    return xp.arctan2(forward_y, forward_x)


def amp_state(position, pelvis_position, pelvis_rotation, head, points, joints):
    """(n, state_size(len(joints))) one frame's D input: the angles of `joints` (position (n, nq − 7) qpos, rad), the
    gravity in the pelvis frame (pelvis_rotation (n, 3, 3) world <- pelvis: Rᵀ·(0, 0, −1)), the pelvis height
    (pelvis_position (n, 3)), the head top height (head (n,)) and points (n, 8, 3) (amp_points) relative to the
    pelvis, turned into its heading frame (heading_yaw). numpy or torch alike."""
    xp = _module(position)
    yaw = heading_yaw(pelvis_rotation)
    cos, sin = xp.cos(yaw)[:, None], xp.sin(yaw)[:, None]
    relative = points - pelvis_position[:, None, :]
    dx, dy, dz = relative[..., 0], relative[..., 1], relative[..., 2]
    local = xp.stack((cos * dx + sin * dy, -sin * dx + cos * dy, dz), -1).reshape(position.shape[0], -1)
    joints = list(joints)
    parts = (position[:, joints], -pelvis_rotation[:, 2, :], pelvis_position[:, 2:3], head[:, None], local)
    if xp is torch:
        return torch.cat([part.to(position.dtype) for part in parts], dim=1)
    return np.concatenate([np.asarray(part, dtype=position.dtype) for part in parts], axis=1)


def library_path(name):
    """The AMP pair library of style library `name`: getup_style/<name>-amp.npz (g1_getup_style.library_path's
    folder and name rule)."""
    return g1_getup_style.library_path(name).with_name(f'{name}-amp.npz')


def load_library(name):
    """The AMP library of `name`: a dict of human, heldout (float32 pairs), joints (tuple), header (dict) and md5 (of
    the file). Raises FileNotFoundError with the command that makes it, ValueError for another layout."""
    path = library_path(name)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup AMP library {path} is missing: '
                                f'python scripts/lab/getup_amp_library.py --name {name}')
    data = path.read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        library = {key: arrays[key] for key in arrays.files}
    header = json.loads(str(library['header']))
    if header.get('version') != AMP_VERSION:
        raise ValueError(f'AMP library {path}: version {header.get("version")}, expected {AMP_VERSION}')
    joints = tuple(int(j) for j in library['joints'])
    width = 2 * state_size(len(joints))
    human, heldout = library['human'], library['heldout']
    if human.ndim != 2 or human.shape[1] != width or not len(human) or heldout.ndim != 2 \
            or heldout.shape[1] != width or not np.isfinite(human).all() or not np.isfinite(heldout).all():
        raise ValueError(f'AMP library {path}: bad pair arrays')
    return dict(human=human.astype(np.float32), heldout=heldout.astype(np.float32), joints=joints, header=header,
                md5=hashlib.md5(data).hexdigest())


class Discriminator(torch.nn.Sequential):
    def __init__(self, width):
        layers, size = [], width
        for hidden in AMP_HIDDEN:
            layers += [torch.nn.Linear(size, hidden), torch.nn.ReLU()]
            size = hidden
        super().__init__(*layers, torch.nn.Linear(size, 1))


class GetupAMP:
    """The discriminator, its training and the style reward of one full-GPU run (TensorPPO's amp)."""

    def __init__(self, weight, library, device, joints, stage=None, buffer=AMP_BUFFER, warmup=AMP_WARMUP_STEPS):
        """weight: the amp reward weight (> 0); library: load_library's dict; joints: the task's free joints (the
        library's must match); stage: the stage the run starts from (its AMP_FILE restored when it fits)."""
        if tuple(library['joints']) != tuple(joints):
            raise ValueError('AMP library joints do not match the task (the free joints)')
        self.weight, self.device, self.warmup = float(weight), torch.device(device), warmup
        self.md5 = library['md5']
        self.human = torch.as_tensor(library['human'], dtype=torch.float32, device=self.device)
        self.heldout = torch.as_tensor(library['heldout'], dtype=torch.float32, device=self.device)
        self.width = self.human.shape[1]
        self.mean = self.human.mean(0)
        self.std = self.human.std(0, unbiased=False).clamp_min(AMP_MIN_STD)
        self.disc = Discriminator(self.width).to(self.device)
        self.optimizer = torch.optim.Adam(self.disc.parameters(), lr=AMP_LEARNING_RATE)
        self.trained_steps = 0
        self.restored = False
        self.status = 'fresh discriminator (no amp.pt in the stage): warm-up'
        if stage is not None and (stage / AMP_FILE).is_file():
            self._restore(stage / AMP_FILE)
        # The policy's in-band pairs, a FIFO on the device (not saved), and this rollout's pairs and band.
        self.buffer = torch.empty((buffer, self.width), device=self.device)
        self.filled = self.cursor = 0
        self._pairs = self._band = None
        self._reward_sum = torch.zeros((), device=self.device)
        self._stats = []

    def _restore(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if saved.get('version') != AMP_VERSION or saved.get('library_md5') != self.md5:
            self.status = (f'amp.pt of {saved.get("version")} / library {saved.get("library_md5")} does not fit '
                           f'{AMP_VERSION} / {self.md5}: fresh discriminator, warm-up')
            return
        self.disc.load_state_dict(saved['disc'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.mean, self.std = saved['mean'].to(self.device), saved['std'].to(self.device)
        self.trained_steps = int(saved['trained_steps'])
        self.restored = True
        self.status = f'discriminator restored from {path} ({self.trained_steps:,} steps trained)'

    def save(self, folder):
        """Writes AMP_FILE into folder (a stage's .pending-* folder)."""
        torch.save({'version': AMP_VERSION, 'library_md5': self.md5, 'trained_steps': int(self.trained_steps),
                    'disc': self.disc.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'mean': self.mean.cpu(), 'std': self.std.cpu()}, folder / AMP_FILE)

    def describe(self):
        return {'weight': self.weight, 'version': AMP_VERSION, 'library_md5': self.md5, 'human_pairs': len(self.human),
                'heldout_pairs': len(self.heldout), 'trained_steps': self.trained_steps, 'warmup': self.warmup,
                'status': self.status}

    @property
    def weight_now(self):
        """The reward's weight now: 0 while D warms up."""
        return self.weight if self.trained_steps >= self.warmup else 0.

    def normalize(self, pairs):
        return ((pairs - self.mean) / self.std).clamp(-AMP_INPUT_CLIP, AMP_INPUT_CLIP)

    def score(self, pairs):
        """(n,) D of raw pairs (n, width)."""
        return self.disc(self.normalize(pairs)).squeeze(-1)

    def begin_rollout(self, steps, worlds):
        if self._pairs is None or self._pairs.shape[:2] != (steps, worlds):
            self._pairs = torch.empty((steps, worlds, self.width), device=self.device)
            self._band = torch.zeros((steps, worlds), dtype=torch.bool, device=self.device)
        self._reward_sum.zero_()

    @torch.no_grad()
    def reward(self, step, pairs, band):
        """(n,) the weighted style reward of rollout step `step`: pairs (n, width) the worlds' (s_t, s_t+1), band (n,)
        bool where it pays (head in the band, finite state). Keeps the pairs for the buffer."""
        pairs = torch.nan_to_num(pairs, nan=0., posinf=0., neginf=0.)  # a non-finite world is outside the band
        self._pairs[step].copy_(pairs)
        self._band[step].copy_(band)
        style = (1 - AMP_REWARD_SCALE * (self.score(pairs) - 1).square()).clamp_min(0)
        style = torch.where(band, style, 0.)
        self._reward_sum.add_(style.sum())
        return self.weight_now * style

    @torch.no_grad()
    def end_rollout(self):
        """Puts the rollout's in-band pairs into the FIFO; rollout/amp_reward, the unweighted reward's mean over the
        in-band world-steps (NaN without one)."""
        selected = self._pairs[self._band]
        count = len(selected)
        if count >= len(self.buffer):
            self.buffer.copy_(selected[-len(self.buffer):])
            self.cursor, self.filled = 0, len(self.buffer)
        elif count:
            index = (self.cursor + torch.arange(count, device=self.device)) % len(self.buffer)
            self.buffer[index] = selected
            self.cursor = (self.cursor + count) % len(self.buffer)
            self.filled = min(self.filled + count, len(self.buffer))
        return {'rollout/amp_reward': self._reward_sum.item() / count if count else float('nan')}

    def train_step(self, batch):
        """One D step on `batch` policy pairs from the FIFO and as many human ones (nothing while the FIFO is empty)."""
        if not self.filled:
            return
        policy = self.buffer[torch.randint(self.filled, (batch,), device=self.device)]
        human = self.normalize(self.human[torch.randint(len(self.human), (batch,), device=self.device)])
        human.requires_grad_(True)
        d_human = self.disc(human).squeeze(-1)
        d_policy = self.score(policy)
        gradient, = torch.autograd.grad(d_human.sum(), human, create_graph=True)
        penalty = gradient.square().sum(-1).mean()
        least = (d_human - 1).square().mean() + (d_policy + 1).square().mean()
        loss = least + AMP_GRADIENT_PENALTY * penalty + AMP_LOGIT_DECAY * self.disc[-1].weight.square().sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._stats.append(torch.stack((least.detach(), penalty.detach(), d_human.detach().mean(),
                                        d_policy.detach().mean())))

    @torch.no_grad()
    def end_update(self, steps):
        """After a PPO update: D has trained on `steps` more environment steps (if it trained at all); the update's
        train/disc_loss (the LSGAN part), train/disc_gp (mean ‖∇D(h)‖²), train/disc_human and train/disc_policy (mean
        D of the batches) and train/disc_heldout (mean D of the held-out subject's pairs)."""
        trained = bool(self._stats)
        means = torch.stack(self._stats).mean(0).cpu().tolist() if trained else [float('nan')] * 4
        self._stats = []
        heldout = self.score(self.heldout).mean().item()
        if (trained and not all(math.isfinite(value) for value in means)) or not math.isfinite(heldout):
            raise RuntimeError('Non-finite AMP discriminator update')
        if trained:
            self.trained_steps += int(steps)
        return dict(zip(('train/disc_loss', 'train/disc_gp', 'train/disc_human', 'train/disc_policy'), means),
                    **{'train/disc_heldout': heldout})
