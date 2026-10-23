"""Crouch poses of the G1 height command (task_config height_command, iteration 28), CPU, no policy:

    .venv\\Scripts\\python.exe scripts\\lab\\crouch_table.py

For each pelvis height c from 0.60 to 0.78 m in 1 cm steps, the symmetric sagittal inverse kinematics of the legs (hip
pitch, knee and ankle pitch, the same on both sides; every other joint at home): pelvis upright at height c with the
lowest sole point on the floor, both feet flat, the whole robot's centre of mass over the middle of the soles (x of the
mean of the sole spheres). Newton's method with finite differences, each height started from the one above (the
critic's lab-critic/it28/crouch_geom.py at pelvis pitch 0). Prints g1_tensor_math.CROUCH_TABLE: per c, IK(c) − IK(0.78),
the offsets stand_target adds to home; tests/test_g1_height.py checks that the table in the code is this one to 1e-4
and that every pose meets the conditions.
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv  # noqa: E402

HEIGHTS = tuple(round(.60 + .01 * k, 2) for k in range(19))  # 0.60 ... 0.78
STEP = 1e-5  # finite difference, rad


class Kinematics:
    """The G1 model in a pose of the three sagittal leg joints (hip pitch, knee, ankle pitch, both legs alike), the
    pelvis upright and lowered so the lowest sole point is at z = 0."""

    def __init__(self):
        self.env = G1VelocityEnv()
        self.model, self.data = self.env.model, self.env.data
        names = [self.model.joint(j).name for j in self.env.joints]
        self.legs = [[names.index(f'{side}_{part}_joint') for part in task.CROUCH_JOINT_PARTS]
                     for side in ('left', 'right')]
        self.soles = self.env.soles
        self.radius = self.model.geom_size[self.soles, 0]
        ranges = self.model.jnt_range[self.env.joints]
        self.low, self.high = ranges[self.legs[0], 0], ranges[self.legs[0], 1]

    def pose(self, x):
        """Joint angles (actuator order) of leg angles x (3,), the model placed: soles on the floor, pelvis upright."""
        q = self.env.home.copy()
        for leg in self.legs:
            q[leg] = x
        model, data = self.model, self.data
        data.qpos[:] = 0
        data.qpos[2] = 1.
        data.qpos[3:7] = [1., 0., 0., 0.]
        data.qpos[self.env.qadr] = q
        data.qvel[:] = 0
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        data.qpos[2] = 1. - float((data.geom_xpos[self.soles, 2] - self.radius).min())
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return q

    def conditions(self, x):
        """(pelvis height m, mean sine of the feet's pitch, COM x − sole centre x m) in pose x."""
        self.pose(x)
        tilt = np.mean([self.data.xmat[foot].reshape(3, 3)[2, 0] for foot in self.env.feet])
        centre = self.data.geom_xpos[self.soles, 0].mean()
        return np.array([self.data.qpos[2], tilt, self.data.subtree_com[self.env.pelvis][0] - centre])

    def check(self, x):
        """(pelvis height m, worst foot pitch deg, COM x − sole centre x m) in pose x, for the tests."""
        self.pose(x)
        tilt = max(abs(math.degrees(math.asin(self.data.xmat[foot].reshape(3, 3)[2, 0]))) for foot in self.env.feet)
        centre = self.data.geom_xpos[self.soles, 0].mean()
        return float(self.data.qpos[2]), tilt, float(self.data.subtree_com[self.env.pelvis][0] - centre)

    def solve(self, height, start):
        """Leg angles (3,) at pelvis height `height` from `start`: Newton steps of at most 0.05 rad within the joint
        ranges, until every condition is met to 1e-9."""
        x = np.array(start, dtype=float)
        goal = np.array([height, 0., 0.])
        for _ in range(100):
            residual = self.conditions(x) - goal
            if np.abs(residual).max() < 1e-9:
                return x
            jacobian = np.zeros((3, 3))
            for k in range(3):
                shifted = x.copy()
                shifted[k] += STEP
                jacobian[:, k] = (self.conditions(shifted) - goal - residual) / STEP
            dx = -np.linalg.solve(jacobian, residual)
            largest = np.abs(dx).max()
            x = np.clip(x + (dx * (.05 / largest) if largest > .05 else dx), self.low, self.high)
        raise RuntimeError(f'IK did not converge at {height} m: residual {residual}')

    def close(self):
        self.env.close()


def solve_table():
    """{c: leg angles (3,)} for HEIGHTS, from the top down, and the kinematics they were solved with."""
    kinematics = Kinematics()
    start, poses = np.array([-.1, .3, -.2]), {}
    for height in reversed(HEIGHTS):
        start = poses[height] = kinematics.solve(height, start)
    return poses, kinematics


def offsets(poses):
    """CROUCH_TABLE rows (c, hip pitch, knee, ankle pitch): IK(c) − IK(BASE_HEIGHT)."""
    top = poses[task.BASE_HEIGHT]
    return tuple((height, *(float(v) for v in poses[height] - top)) for height in HEIGHTS)


def main():
    poses, kinematics = solve_table()
    try:
        top = poses[task.BASE_HEIGHT]
        print(f'IK({task.BASE_HEIGHT}): hip pitch {top[0]:.5f}, knee {top[1]:.5f}, ankle pitch {top[2]:.5f}')
        for height in HEIGHTS:
            pelvis, tilt, com = kinematics.check(poses[height])
            x = poses[height]
            print(f'c {height:.2f}: hip {x[0]:+.4f} knee {x[1]:+.4f} ankle {x[2]:+.4f} (ankle stop margin '
                  f'{x[2] - kinematics.low[2]:.3f}); pelvis {pelvis - height:+.1e} m, foot {tilt:.1e}°, COM {com:+.1e} m')
        print('CROUCH_TABLE = (')
        for row in offsets(poses):
            print(f'    ({row[0]:.2f}, {row[1]:.5f}, {row[2]:.5f}, {row[3]:.5f}),')
        print(')')
    finally:
        kinematics.close()


if __name__ == '__main__':
    main()
