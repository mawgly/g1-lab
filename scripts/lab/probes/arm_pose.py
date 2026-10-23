"""CPU only: arm_pose.py <stage> [<stage> ...]. How each arm hangs: mean angle of every arm joint from the home pose
(degrees), standing 10 s and walking straight 0.5 and 1 m/s, deterministic, heading hold off, seed 73000, after the
first 2 s; left and right side by side, with the mirrored difference (left minus the mirror of right: 0 for a
symmetric pose; roll and yaw joints flip sign in the mirror, pitch joints keep it, as symmetry.joint_mirror)."""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.symmetry import joint_mirror  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def mean_pose(model, norm, config, command):
    env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 500})
    raw = env.unwrapped
    obs, _ = env.reset(seed=73000, options={'command': command})
    angles = []
    for step in range(500):
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if step >= 100:
            angles.append(raw.data.qpos[raw.qadr] - raw.home)
        if term:
            break
    names = [raw.model.joint(j).name for j in raw.joints]
    permutation, signs = joint_mirror(env)
    env.close()
    return np.degrees(np.mean(angles, axis=0)), names, permutation, signs


for arg in sys.argv[1:]:
    stage = Path(arg)
    model, norm = load_stage(stage)
    config = stage_task_config(stage)
    print(f'== {stage.parent.name[:12]} {stage.name}')
    for label, command in (('stand', [0., 0., 0.]), ('walk 0.5', [.5, 0., 0.]), ('walk 1.0', [1., 0., 0.])):
        pose, names, permutation, signs = mean_pose(model, norm, config, command)
        print(f'  {label}:')
        for part in PARTS:
            left, right = names.index(f'left_{part}_joint'), names.index(f'right_{part}_joint')
            mirrored = signs[left] * pose[permutation[left]]  # the right joint seen as a left one
            print(f'    {part:15s} L {pose[left]:+7.1f}  R {pose[right]:+7.1f}  L-mirror(R) {pose[left] - mirrored:+7.1f}')
    norm.close()
