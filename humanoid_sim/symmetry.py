"""Left–right mirror of the G1 real-robot task, for the PPO mirror loss (Mittal et al. 2024; rsl_rl symmetry).

PPO alone may settle on a lopsided gait: on G1 it kept a right-turning bias of 0.03–0.1 rad/s and turned left
weaker than right through every reward fix (20.10.2026). The robot is symmetric about its sagittal (x–z)
plane, so the policy should answer a mirrored state with the mirrored action; `TensorPPO` adds that as a loss
when `mirror_coef` > 0. Nothing about the task, observations or checkpoints changes.

Joints: left and right swap. Reflecting a rotation about axis a gives a rotation about (−ax, ay, −az), so a
joint keeps its sign when its counterpart's axis equals the reflected axis and flips otherwise (roll and yaw
joints flip, pitch joints keep); signs are read from the model's joint axes, not typed in. Observations
(Unitree deploy layout, see tasks/g1_tensor_math.deploy_observation): angular velocity (−x, y, −z), gravity
(x, −y, z), command (vx, −vy, −yaw rate), gait phase half a cycle later (−sin, −cos: the right foot runs half a
cycle after the left), joint positions, velocities and last action by the joint map; the height command as it is,
the upper-body slot's flag as it is and its waist and arm targets by the joint map; critic extras: linear velocity
(x, −y, z) and height.
"""
import numpy as np
import torch


def reflected(axis):
    return np.array([-axis[0], axis[1], -axis[2]])


def joint_mirror(env):
    """(permutation, signs) over the actuated joints: mirrored[i] = signs[i] · value[permutation[i]]."""
    raw = env.unwrapped
    names = [raw.model.joint(j).name for j in raw.joints]
    raw.reset(seed=0)  # the home pose (± 0.01 rad), so joint axes are those of a symmetric pose
    axes = [raw.data.xaxis[j].copy() for j in raw.joints]
    permutation, signs = [], []
    for i, name in enumerate(names):
        other = name.replace('left', '\0').replace('right', 'left').replace('\0', 'right')
        j = names.index(other)
        dot = float(reflected(axes[i]) @ axes[j])
        if abs(dot) < .9:
            raise ValueError(f'{name}: оси суставов не зеркальны ({dot:.2f})')
        permutation.append(j)
        signs.append(1. if dot > 0 else -1.)
    return np.array(permutation), np.array(signs)


class Mirror:
    """Index-and-sign maps for observations and actions of a G1 real-robot env (CPU or CUDA tensors)."""

    def __init__(self, env):
        raw = env.unwrapped
        if not getattr(raw, 'real_robot', False):
            raise ValueError('Зеркальная потеря есть только для профиля real_robot (раскладка наблюдений Unitree)')
        from .tasks import g1_tensor_math as task
        permutation, signs = joint_mirror(env)
        n = len(permutation)
        index, sign = list(range(11)), [-1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1]
        for block in range(3):  # joint positions, joint velocities, last action
            index += list(11 + block * n + permutation)
            sign += list(signs)
        for name, size in getattr(raw, 'extra_terms', ()):  # task_config height_command, upper_body_slot
            start = len(index)
            if name == task.HEIGHT_TERM:  # a height is the same on either side
                index.append(start)
                sign.append(1)
            elif name == task.UPPER_BODY_TERM and size == 1 + n - task.UPPER_BODY_FIRST and all(
                    permutation[task.UPPER_BODY_FIRST:] >= task.UPPER_BODY_FIRST):
                index.append(start)  # the flag
                sign.append(1)
                for joint in range(task.UPPER_BODY_FIRST, n):  # waist and arm targets relative to home, as joints
                    index.append(start + 1 + int(permutation[joint]) - task.UPPER_BODY_FIRST)
                    sign.append(signs[joint])
            else:
                raise ValueError(f'Зеркало не знает терма {name}')
        history = getattr(raw, 'history', 1)
        if history > 1:  # each term holds its frames oldest first (task.history_observation); mirror every frame
            position = task.history_index(n, history, getattr(raw, 'extra_sizes', ()))
            frame_index, frame_sign = index, sign
            index, sign = [0] * (history * len(frame_index)), [0] * (history * len(frame_index))
            for frame in range(history):
                for i in range(len(frame_index)):
                    index[position[frame][i]] = position[frame][frame_index[i]]
                    sign[position[frame][i]] = frame_sign[i]
        if raw.critic_extras:
            index += [len(index) + k for k in range(4)]
            sign += [1, -1, 1, 1]
        if len(index) != raw.observation_space.shape[0]:
            raise ValueError('Раскладка наблюдений не совпадает с зеркалом')
        self.obs_index, self.obs_sign = torch.tensor(index), torch.tensor(sign, dtype=torch.float32)
        self.action_index, self.action_sign = torch.tensor(permutation), torch.tensor(signs, dtype=torch.float32)

    def to(self, device):
        for name in ('obs_index', 'obs_sign', 'action_index', 'action_sign'):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def observation(self, values):
        return values[..., self.obs_index] * self.obs_sign.to(values.dtype)

    def action(self, values):
        return values[..., self.action_index] * self.action_sign.to(values.dtype)
