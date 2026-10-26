"""scripts/lab/probes/clip_feasibility.py (iteration 67): the non-negative least squares against brute force, the convex
hull's signed distance, the torque each joint has at its stops and speeds, and the verdict on simple clips: standing at
home is doable, the same clip 5 cm in the air is not (a residual of m·g) unless lowered onto the floor, a robot pitched
onto its toes has its centre of mass off the support."""
import importlib
import itertools
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
feasibility = importlib.import_module('clip_feasibility')
cmu_mocap = importlib.import_module('cmu_mocap')
from humanoid_sim.tasks import g1_getup_track  # noqa: E402
from humanoid_sim.tasks.g1_getup_bank import getup_model_env  # noqa: E402


def _brute_force(a, b):
    """The exact non-negative least squares by trying every support set (small problems)."""
    best, value = None, math.inf
    n = a.shape[1]
    for size in range(n + 1):
        for chosen in itertools.combinations(range(n), size):
            x = np.zeros(n)
            if chosen:
                x[list(chosen)] = np.linalg.lstsq(a[:, list(chosen)], b, rcond=None)[0]
            if (x >= -1e-12).all():
                residual = float(np.sum((a @ x - b) ** 2))
                if residual < value - 1e-12:
                    best, value = x, residual
    return best, value


def test_nonnegative_least_squares_is_the_exact_minimum():
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = rng.normal(size=(7, 5))
        b = rng.normal(size=7)
        x = feasibility.nonnegative_least_squares(a, b)
        _, value = _brute_force(a, b)
        assert (x >= 0).all()
        assert float(np.sum((a @ x - b) ** 2)) == pytest.approx(value, rel=1e-8, abs=1e-10)


def test_hull_and_margin_are_the_signed_distance_to_the_convex_hull():
    square = feasibility.hull([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5], [1, .5]])
    assert len(square) == 4
    assert feasibility.margin([.5, .5], square) == pytest.approx(.5)
    assert feasibility.margin([.9, .5], square) == pytest.approx(.1)
    assert feasibility.margin([1.3, .5], square) == pytest.approx(-.3)
    assert feasibility.margin([1.3, 1.4], square) == pytest.approx(-.5)
    assert feasibility.margin([0, 2], feasibility.hull([[0, 0], [0, 1]])) == pytest.approx(-1.)  # a segment: no inside
    assert feasibility.margin([0, 0], np.zeros((0, 2))) == -math.inf


def test_available_torque_is_the_motor_curve_and_the_pd_at_its_stop():
    env = getup_model_env()
    low, high = env.target_low, env.target_high
    middle = (low + high) / 2
    up, down = feasibility.available_torque(env, middle, np.zeros(29))
    x1, x2, y1, y2, static, viscous = (np.asarray(row, float) for row in env.motors)
    np.testing.assert_allclose(up, np.minimum(y2, env.kp * (high - middle)), atol=1e-9)  # still: y2 (not the same way)
    np.testing.assert_allclose(down, np.minimum(y2, env.kp * (middle - low)), atol=1e-9)
    at_stop, _ = feasibility.available_torque(env, high, np.zeros(29))
    np.testing.assert_allclose(at_stop, 0., atol=1e-9)  # the target cannot pass the range: nothing more that way
    fast = np.full(29, 1e3)
    up, _ = feasibility.available_torque(env, middle, fast)
    assert (up < 0).all()  # beyond x2 the motor gives nothing, and friction and damping take more


def _standing_clip(env, frames=20, lift=0., pitch=0.):
    row = cmu_mocap.home_qpos(env)
    if pitch:
        mujoco.mju_axisAngle2Quat(row[3:7], np.array([0., 1., 0.]), pitch)
    model, data, geoms = cmu_mocap._floor_model(env)
    row[2] -= cmu_mocap.lowest_point(model, data, row, geoms) - lift
    qpos = np.repeat(row[None], frames, 0)
    return dict(qpos=qpos, qvel=np.zeros((frames, 35)))


def test_standing_is_doable_floating_is_not_and_a_tilted_robot_falls_off_its_support():
    env = getup_model_env()
    standing = feasibility.check_clip(_standing_clip(env), env)['summary']
    assert standing['feasible'], standing['reasons']
    assert standing['com_margin_min'] > .02 and standing['residual_force_max'] < 1e-3 and standing['hand_max'] == 0.
    assert standing['noise'] == 0. and standing['qd_max'] == 0.
    floating = _standing_clip(env, lift=.05)
    lowered = feasibility.check_clip(floating, env)['summary']
    assert lowered['feasible'] and lowered['float_max'] == pytest.approx(.05, abs=1e-6)
    as_is = feasibility.check_clip(floating, env, lower=False)['summary']
    assert not as_is['feasible'] and as_is['residual_force_max'] == pytest.approx(1., abs=1e-3)  # a harness: m·g
    tilted = feasibility.check_clip(_standing_clip(env, pitch=.5), env)['summary']
    assert not tilted['feasible'] and tilted['com_margin_min'] < -feasibility.COM_OUTSIDE
    assert any('центр масс' in reason for reason in tilted['reasons'])


def test_the_built_clip_kinematics_go_through_the_check():
    """A clip made by g1_getup_track.build (home, standing) is doable, with its velocities from central differences."""
    env = getup_model_env()
    clip = g1_getup_track.build(env, _standing_clip(env)['qpos'])
    summary = feasibility.check_clip(clip, env)['summary']
    assert summary['feasible'], summary['reasons']
    assert feasibility.verdict_text(summary) == 'годен'


def test_the_waist_free_verdict_leaves_out_only_the_waists_torque_rule():
    """A waist short of torque fails the full verdict and passes the waist-free one; a knee short of torque, the centre
    of mass off the support or a residual fail both."""
    env = getup_model_env()
    result = feasibility.check_clip(_standing_clip(env, frames=5), env)
    names, checker = result['names'], feasibility.FrameChecker(env)
    locked, peak = checker.locked, checker.peak

    def judged(joint=None, **changes):
        rows = [dict(r, **changes) for r in result['rows']]
        if joint is not None:
            k = names.index(joint)
            for r in rows:
                r['torque'] = list(r['torque'])
                r['torque'][k] = 10 * (abs(r['have'][k]) + peak[k])
                r['worst_ratio'], r['worst'], r['worst_torque'], r['worst_have'] = 5., joint, r['torque'][k], r['have'][k]
        return feasibility.summarize(rows, names, 0., 0., 0., locked, peak)
    plain = judged()
    assert plain['feasible'] and plain['feasible_waist_free'] and plain['waist_free_ratio_max'] < 1
    waist = judged('waist_pitch')
    assert not waist['feasible'] and waist['feasible_waist_free']
    assert feasibility.verdict_text(waist, waist_free=True) == 'годен'
    knee = judged('left_knee')
    assert not knee['feasible'] and not knee['feasible_waist_free'] and knee['waist_free_ratio_joint'] == 'left_knee'
    assert knee['waist_free_ratio_frames'] == 5 and 'left_knee' in feasibility.verdict_text(knee, waist_free=True)
    for changes in (dict(com_margin=-.2), dict(residual_force=.2), dict(residual_moment=.2)):
        off = judged('waist_pitch', **changes)
        assert not off['feasible'] and not off['feasible_waist_free']
