"""g1_getup task_config hip_roll_out (iteration 63), CPU: the hip roll PD target's limit outwards per leg (the left
leg's +, the right leg's −, found from the model's kinematics), through target_low / target_high as knee_floor (the
PD, the delay queue, the export's clip); the action, action_change and the observed last action unchanged. Its own
action contract (…-hiproll<v>). Off (missing or null) the task is bit for bit the old one. CPU–GPU:
tests/test_g1_gpu.py."""
import numpy as np
import pytest

from humanoid_sim.tasks.g1_getup import G1GetupEnv, outward_signs
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_stance_width import BEFORE_46

WORKING = .7
LEFT, RIGHT = 1, 7  # left_hip_roll_joint, right_hip_roll_joint


def test_the_setting_is_validated_labelled_and_its_own_action_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['hip_roll_out'] is None  # old stages: off
    assert validate_config({**base, 'task_config': {'hip_roll_out': WORKING}})['task_config']['hip_roll_out'] == .7
    for value in (.1, 2.6, float('nan'), '0.7', True, [.7]):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'hip_roll_out': value}})
    plain = task_identity('g1_29dof', 'g1_getup', {})
    assert task_identity('g1_29dof', 'g1_getup', {'hip_roll_out': None}) == plain
    limited = task_identity('g1_29dof', 'g1_getup', {'hip_roll_out': WORKING})
    assert limited['action_schema'] == plain['action_schema'] + '-hiproll0.7'
    assert {k: v for k, v in limited.items() if k != 'action_schema'} == {
        k: v for k, v in plain.items() if k != 'action_schema'}
    both = task_identity('g1_29dof', 'g1_getup', {**BRANCH_51, 'hip_roll_out': WORKING})
    assert both['action_schema'].endswith('-knee0.061-pdbound0.6-hiproll0.7')
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['hip_roll_out']
    assert field['group'] is None and field['default'] is None and (field['min'], field['max']) == (.2, 2.5)
    assert 'крена бедра наружу' in field['label']


def test_outward_signs_and_the_target_bounds():
    """outward_signs finds the two hip roll joints, + outwards on the left, − on the right (a positive left roll swings
    the left foot away from the midline); the limit moves exactly those two bounds, every other bound is the joint
    range."""
    plain, limited = G1GetupEnv(bank=drop_bank()), G1GetupEnv(bank=drop_bank(), hip_roll_out=WORKING)
    try:
        assert outward_signs(plain.model, plain.joints, plain.home) == ((LEFT, RIGHT), (1, -1))
        assert plain.hip_roll_joints == (LEFT, RIGHT) and plain.hip_roll_signs == (1, -1)
        low, high = plain.model.jnt_range[plain.joints].T
        np.testing.assert_array_equal(plain.target_high, high)
        expected_low, expected_high = plain.target_low.copy(), plain.target_high.copy()
        expected_high[LEFT], expected_low[RIGHT] = .7, -.7
        np.testing.assert_array_equal(limited.target_high, expected_high)
        np.testing.assert_array_equal(limited.target_low, expected_low)
        assert plain.target_low[RIGHT] < -2.9 and plain.target_high[LEFT] > 2.9  # the joint ranges: far outwards
        assert limited.target_low[LEFT] == plain.target_low[LEFT] < 0  # inwards: the joint range
        # The PD target of an action, by hand: home + scale·clip(a, ±10) within the bounds.
        for a in (-10., -3., -1., 0., 1., 3., 10.):
            action = np.zeros(29)
            action[[LEFT, RIGHT]] = a, -a  # both outwards for a > 0
            raw = plain.home + plain.scale * np.clip(action, -10, 10)
            target = limited._target(action)
            assert target[LEFT] == pytest.approx(min(max(raw[LEFT], expected_low[LEFT]), .7), abs=1e-12)
            assert target[RIGHT] == pytest.approx(max(min(raw[RIGHT], expected_high[RIGHT]), -.7), abs=1e-12)
            np.testing.assert_array_equal(np.delete(target, [LEFT, RIGHT]), np.delete(plain._target(action), [LEFT, RIGHT]))
    finally:
        plain.close(), limited.close()


def test_the_task_bounds_the_target_and_leaves_the_action_alone():
    """Lying on the back, with the 20 ms delay queue full: actions driving both hips outwards to the box edge. With the
    limit the hip roll angles stay near ±0.7 (within the PD's own error), without it they go far past; the observed
    last action and action_change are the same either way (the action is not changed). Standing and flailing, a fall
    can force a leg past the limit under the body's weight (the limit is the target's, not the angle's): there every
    hip roll torque beyond it pushes back inwards or the joint already moves back."""
    bank = drop_bank()
    plain, limited = G1GetupEnv(bank=bank), G1GetupEnv(bank=bank, hip_roll_out=WORKING)
    rng = np.random.default_rng(1)
    signs = np.array([1., -1.])
    try:
        for row in (0, 3):
            for env in (plain, limited):
                env.reset(seed=2, options=dict(OPTIONS, start=row, action_delay_steps=10))
            reach, beyond = {}, 0
            for k in range(60):
                action = rng.uniform(-2., 2., 29)
                action[[LEFT, RIGHT]] = 10., -10.
                (*_, info_p), (*_, info_l) = plain.step(action), limited.step(action)
                np.testing.assert_array_equal(plain.previous_action, limited.previous_action)
                assert info_p['reward_components']['action_change'] == info_l['reward_components']['action_change']
                for name, env in (('plain', plain), ('limited', limited)):
                    roll = env.data.qpos[env.qadr][[LEFT, RIGHT]] * signs
                    reach[name] = max(reach.get(name, -9), float(roll.max()))
                roll = limited.data.qpos[limited.qadr][[LEFT, RIGHT]] * signs
                speed = limited.data.qvel[limited.vadr][[LEFT, RIGHT]] * signs
                torque = limited.data.ctrl[[LEFT, RIGHT]] * signs
                for leg in range(2):
                    if roll[leg] > .7 + .02:  # past the limit: the PD pulls in, or the leg already comes back
                        beyond += 1
                        assert torque[leg] < 0 or speed[leg] < 0, (row, k, leg, roll, torque, speed)
            if row == 0:
                assert reach['limited'] < .7 + .15 and reach['plain'] > 1.2, reach
            else:
                assert reach['plain'] > reach['limited'], reach
    finally:
        plain.close(), limited.close()


def test_off_the_task_is_bit_for_bit_the_old_one():
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'hip_roll_out': None}, 'plain'), ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'hip_roll_out': None}, bank) == (OLD_STANDING_DIGEST, 59)
    assert getup_digest(BRANCH_51, bank) == BEFORE_46[0]
