"""The arm pose libraries of task_config arm_targets (iteration 40): arm_pose_library.py [--count 16384]
[--carry 8192] [--seed 40] [--out <npz>]

Poses of the ten free arm joints of the 29-DoF G1 (shoulder pitch, roll, yaw, elbow and wrist roll of both arms; the
locked wrists and the rest of the robot at home, the pelvis upright 0.8 m up): `poses` each joint uniform over its range,
`carry` the symmetric "carry in front" poses of task.ARM_CARRY (the left arm's absolute angles per part, uniform; the
right arm its mirror by the joint axes, symmetry.model_joint_mirror). A pose is kept when no geom of an arm link (the
shoulders, elbows, wrists and Unitree's hand capsules: the real robot has hands, ours touch nothing without
hand_collision) touches another robot geom, the body, the other arm or its own arm, both in the pose and halfway from
home (the middle of the quintic from home, quintic(0.5) = 0.5): MuJoCo CPU contacts, the model's own margins (0), as the
critic's lab-critic/it40/collide40.py. A margin of 1 cm would reject nearly every pose with the shoulder roll below
home: the shoulder roll link passes within 1 cm of the torso there, a near miss of the joint's own build.

Writes humanoid_sim/tasks/data/g1_arm_poses.npz (not under assets/robots/g1, so not part of the model revision): joints
(the ten names), poses (count, 10) and carry (carry, 10) float32 absolute angles (rad), carry_ranges (task.ARM_CARRY's),
seed, drawn and carry_drawn (poses tried); prints the shares kept."""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_sim.symmetry import model_joint_mirror  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.g1_velocity import ARM_POSES, G1VelocityEnv  # noqa: E402

COUNT, CARRY, SEED, BATCH = 16384, 8192, 40, 4096


class Touch:
    """Whether arm poses touch: the 29-DoF model with Unitree's foot and hand capsules."""

    def __init__(self):
        self.env = env = G1VelocityEnv(real_robot=True, foot_collision=True, hand_collision=True)
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        names = [self.model.joint(j).name for j in env.joints]
        self.free = [i for i in env.arm_joints if i not in env.wrist_joints]
        self.names = [names[i] for i in self.free]
        self.low, self.high = self.model.jnt_range[env.joints[self.free]].T
        self.home = env.home.copy()
        self.robot, self.arm = env.robot_geoms, env.arm_geoms
        # The carry poses: per ARM_CARRY part its left joint's place among the free ones and the right one's, with the
        # mirror sign (mirrored[right] = signs[right]·value[left]).
        permutation, signs = model_joint_mirror(self.model, env.joints, env.qadr, env.home)
        self.carry_left, self.carry_right, self.carry_sign = [], [], []
        for part, _, _ in task.ARM_CARRY:
            left = names.index(f'left_{part}_joint')
            right = int(np.flatnonzero(permutation == left)[0])
            self.carry_left.append(self.free.index(left))
            self.carry_right.append(self.free.index(right))
            self.carry_sign.append(float(signs[right]))

    def touches(self, free):
        """Whether the free arm joints at `free` (10,) bring an arm geom into contact with another robot geom."""
        data = self.data
        data.qpos[:] = self.model.qpos0
        data.qpos[2] = .8
        qpos = self.home.copy()
        qpos[self.free] = free
        data.qpos[self.env.qadr] = qpos
        mujoco.mj_forward(self.model, data)
        if not data.ncon:
            return False
        geom = data.contact.geom
        return bool((self.robot[geom].all(axis=1) & self.arm[geom].any(axis=1) & (data.contact.dist < 0)).any())

    def keeps(self, pose):
        """The libraries' rule: neither the pose nor its midpoint from home touches."""
        home = self.home[self.free]
        return not self.touches(pose) and not self.touches(home + .5 * (pose - home))

    def carry(self, values):
        """(n, 10) free joint angles of carry poses from the left arm's values (n, len(ARM_CARRY))."""
        poses = np.tile(self.home[self.free], (len(values), 1))
        poses[:, self.carry_left] = values
        poses[:, self.carry_right] = values * np.array(self.carry_sign)
        return poses

    def close(self):
        self.env.close()


def library(touch, draw, count):
    """count poses from draw(k) -> (k, 10) candidates that the rule keeps, and how many were tried."""
    kept, drawn = [], 0
    while len(kept) < count:
        for pose in draw(BATCH):
            drawn += 1
            if touch.keeps(pose):
                kept.append(pose)
                if len(kept) == count:
                    break
    return np.array(kept, dtype=np.float32), drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--carry', type=int, default=CARRY)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--out', type=Path, default=ARM_POSES)
    args = parser.parse_args()
    touch = Touch()
    try:
        if touch.touches(touch.home[touch.free]):
            raise SystemExit('the home pose touches: the rule is wrong for this model')
        rng = np.random.default_rng(args.seed)
        poses, drawn = library(touch, lambda k: rng.uniform(touch.low, touch.high, (k, len(touch.free))), args.count)
        ranges = np.array([row[1:] for row in task.ARM_CARRY])
        carry, carry_drawn = library(touch, lambda k: touch.carry(rng.uniform(ranges[:, 0], ranges[:, 1],
                                                                              (k, len(ranges)))), args.carry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, joints=np.array(touch.names), poses=poses, carry=carry, carry_ranges=ranges,
                            seed=args.seed, drawn=drawn, carry_drawn=carry_drawn)
        print(f'{len(poses)} free poses of {drawn} drawn kept ({len(poses) / drawn:.1%}), {len(carry)} carry poses of '
              f'{carry_drawn} ({len(carry) / carry_drawn:.1%}) -> {args.out}')
        print('joints:', ', '.join(touch.names))
        for name, rows in (('free', poses), ('carry', carry)):
            print(f'{name} range per joint (rad):',
                  ', '.join(f'{lo:+.2f}..{hi:+.2f}' for lo, hi in zip(rows.min(0), rows.max(0))))
    finally:
        touch.close()


if __name__ == '__main__':
    main()
