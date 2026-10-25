"""g1_getup task_config action_change_unbounded (iteration 37), CPU: the action_change term −w·Σ(a − a_prev)² charges
the action before pd_bound (the box-clipped one with the locked wrists at 0) and its own one of the step before, zero at
a reset; the PD, the delay queue and the observed last action keep the applied, bounded action. Off (missing or false)
the task is bit for bit the old one; a reward setting only, so the contract and branching are unchanged; walking does
not know it."""
import numpy as np
import pytest

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.lab import validate_config
from humanoid_sim.tasks.config import (G1_GETUP_DEFAULTS, G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, g1_config,
                                       g1_getup_config)
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, STANDING, standing_digest

BOUND = [1., .6]  # pd_bound of branches 34-36: start, lowest
LOCKED = [20, 21, 27, 28]  # the wrist pitch and yaw joints (lock_wrists)
SHOULDER_YAW = 24  # right_shoulder_yaw_joint: range ±2.618, scale 0.44, the action's range edge ±5.95


def before_bound(action):
    """The action before pd_bound by hand, as the critic's noise_cost.py builds it: the ±10 box, the wrists at 0."""
    action = np.clip(np.asarray(action, dtype=float), -10., 10.)
    action[LOCKED] = 0.
    return action


def change_term(action, previous):
    """The action_change term as specified: −w·Σ(a − a_prev)²."""
    return -G1_GETUP_REWARD_WEIGHTS['action_change'] * float(((action - previous) ** 2).sum())


def test_the_setting_is_a_bool_spelled_out_and_leaves_the_contract_alone():
    assert G1_GETUP_DEFAULTS['action_change_unbounded'] is False
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['action_change_unbounded'] is False  # old stages and branches
    for given in (True, False):
        config = validate_config({**base, 'task_config': {'action_change_unbounded': given, 'pd_bound': BOUND}})
        assert config['task_config']['action_change_unbounded'] is given
    for bad in (1, 0, 1., 'true', None, [True]):
        with pytest.raises(ValueError):
            g1_getup_config({'action_change_unbounded': bad})
    with pytest.raises(ValueError):  # walking has no such setting
        g1_config({'action_change_unbounded': True})
    branch36 = {'pd_bound': BOUND, 'stand_pose_std': .7, 'reward_weights': {'impact': 2.}}
    for config in ({}, {'pd_bound': BOUND}, branch36):
        saved = task_identity('g1_29dof', 'g1_getup', config)
        unbounded = task_identity('g1_29dof', 'g1_getup', {**config, 'action_change_unbounded': True})
        assert unbounded == saved
        validate_resume(saved, unbounded)  # an ordinary branch of the same contract
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'action_change_unbounded')
    assert field['group'] is None and field['default'] is False and field['type'] == 'bool'
    assert 'резкие действия' in field['label'] and 'мягкости' in field['label']


def test_without_the_setting_the_task_is_bit_for_bit_the_old_one():
    """Missing or false: every observation, reward, end, qpos and term of training episodes is the code's before the
    setting (getup_digest's OLD_DIGESTS, also with pd_bound; standing_digest's OLD_STANDING_DIGEST). Without a bound the
    action before it is the applied one, so the setting on changes nothing either; with pd_bound it changes the
    rewards."""
    bank = drop_bank()
    for config, name in (({'action_change_unbounded': False}, 'plain'),
                         ({'pd_bound': BOUND, 'action_change_unbounded': False}, 'pd_bound'),
                         ({'action_change_unbounded': True}, 'plain')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'action_change_unbounded': False}, bank) == (OLD_STANDING_DIGEST, 59)
    assert getup_digest({'pd_bound': BOUND, 'action_change_unbounded': True}, bank) != OLD_DIGESTS['pd_bound']


def test_the_cpu_task_charges_the_action_before_the_bound_and_changes_that_term_only():
    """The same episodes with the setting and without, both bounded at c 0.15 (most joints) under a 10 ms delay, actions
    over and beyond the ±10 box from lying and standing: every observation, qpos, applied action (previous_action, what
    the PD, the delay queue and the observation use) and term but action_change is the same; with the setting the term
    is −w·Σ(a − a_prev)² of the box-clipped actions with the wrists at 0, by hand (the previous one 0 after a reset),
    without it that of the applied actions; the reward differs by the term alone."""
    bank = drop_bank()
    options = dict(OPTIONS, action_delay_steps=5, pd_bound=.15)
    unbounded, plain = (G1GetupEnv(bank=bank, pd_bound=BOUND, action_change_unbounded=flag) for flag in (True, False))
    rng = np.random.default_rng(37)
    bounded, steps = [], 0
    try:
        assert unbounded.change_unbounded and not plain.change_unbounded and plain.previous_unbounded is None
        assert np.flatnonzero(unbounded.wrist_mask.numpy()).tolist() == LOCKED
        for row in (0, 1, STANDING):
            unbounded.previous_unbounded[:] = 1.  # a reset clears it
            obs = [env.reset(seed=3, options=dict(options, start=row))[0] for env in (unbounded, plain)]
            np.testing.assert_array_equal(*obs)
            assert not unbounded.previous_unbounded.any()
            previous, applied_before = np.zeros(29), np.zeros(29)
            for k in range(25):
                action = rng.uniform(-12., 12., 29) * (.3 if k % 3 else 1.)
                applied = unbounded.applied_action(action)
                pre = before_bound(action)
                np.testing.assert_array_equal(unbounded.unbounded_action(action), pre)
                (obs_u, reward_u, end_u, cut_u, info_u), (obs_p, reward_p, end_p, cut_p, info_p) = (
                    env.step(action) for env in (unbounded, plain))
                np.testing.assert_array_equal(obs_u, obs_p)
                np.testing.assert_array_equal(unbounded.data.qpos, plain.data.qpos)
                np.testing.assert_array_equal(unbounded.previous_action, applied)
                np.testing.assert_array_equal(plain.previous_action, applied)
                np.testing.assert_array_equal(unbounded.previous_unbounded, pre)
                assert (end_u, cut_u) == (end_p, cut_p)
                terms_u, terms_p = info_u['reward_components'], info_p['reward_components']
                assert {key: v for key, v in terms_u.items() if key != 'action_change'} == {
                    key: v for key, v in terms_p.items() if key != 'action_change'}
                assert terms_u['action_change'] == pytest.approx(change_term(pre, previous), rel=1e-12, abs=1e-15)
                assert terms_p['action_change'] == pytest.approx(change_term(applied, applied_before), rel=1e-12,
                                                                 abs=1e-15)
                assert reward_u == pytest.approx(reward_p - terms_p['action_change'] + terms_u['action_change'],
                                                 rel=1e-12, abs=1e-12)
                bounded.append(np.mean(np.abs(applied - pre) > 1e-9))
                previous, applied_before = pre, applied
                steps += 1
                if end_u or cut_u:
                    break
    finally:
        unbounded.close(), plain.close()
    assert steps >= 60 and np.mean(bounded) > .3  # the bound acted, so the two terms differed


def test_a_mean_beyond_the_clamp_no_longer_hides_the_noise():
    """The critic's case (lab-critic/it37, noise_cost.py): standing at the lowest bound c 0.6, the right shoulder yaw's
    action at a mean of 8.42 with σ 1 noise, the other joints at 0. The arm swings to its stop (2.6 rad), where the
    clamp's edge is 5.95 + 0.6·y1/kp/scale = 8.31, so the bound pins that action in a third of the steps; without the
    setting the term charges the pinned action and sees about half of the noise, with it the whole of it, on the same
    trajectory."""
    bank = drop_bank()
    envs = [G1GetupEnv(bank=bank, pd_bound=BOUND, action_change_unbounded=flag) for flag in (True, False)]
    rng = np.random.default_rng(5)
    charged, pinned = {True: [], False: []}, []
    try:
        unbounded = envs[0]
        edge = BOUND[1] * unbounded.peak[SHOULDER_YAW] / unbounded.kp[SHOULDER_YAW] / unbounded.scale[SHOULDER_YAW]
        assert 5.95 + edge == pytest.approx(8.31, abs=.01)
        for env in envs:
            env.reset(seed=3, options=dict(OPTIONS, start=STANDING, pd_bound='min'))
        for k in range(40):
            action = np.zeros(29)
            action[SHOULDER_YAW] = 8.42 + rng.standard_normal()
            pinned.append(unbounded.applied_action(action)[SHOULDER_YAW] != action[SHOULDER_YAW])
            for env, flag in zip(envs, (True, False)):
                charged[flag].append(env.step(action)[-1]['reward_components']['action_change'])
        assert envs[0].data.qpos[envs[0].qadr][SHOULDER_YAW] > 2.5  # at the stop
    finally:
        for env in envs:
            env.close()
    hidden, whole = -np.mean(charged[False][15:]), -np.mean(charged[True][15:])
    assert .25 < np.mean(pinned[15:]) < .5
    assert whole > 1.5 * hidden and .008 < whole < .03  # 0.01·E[(Δa)²], 2σ² = 2 for the unpinned noise


def test_the_task_builds_from_a_saved_config_with_the_setting():
    """make_task (evaluation, playback, probes and the trainers build from the run's or stage's task_config): a config
    without the key gives the original term, with it the action before the bound."""
    from humanoid_sim.tasks.catalog import make_task
    for config, flag in (({}, False), ({'pd_bound': BOUND, 'action_change_unbounded': True}, True)):
        env = make_task('g1_getup', 'g1_29dof', validate_config(dict(robot_id='g1_29dof', task_id='g1_getup',
                                                                     task_config=config))['task_config'])
        try:
            assert env.unwrapped.change_unbounded is flag and (env.unwrapped.previous_unbounded is not None) is flag
        finally:
            env.close()
