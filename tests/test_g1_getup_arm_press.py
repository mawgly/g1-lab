"""g1_getup reward weight arm_press (iteration 52), CPU: −w·[standing]·Σ over the free arm joints (15-19, 22-26: the
locked wrists left out) of (T − q)², T the PD target of the action before pd_bound (pd_target of lock_joints of the
box-clipped policy output, the action action_change_unbounded charges), q the angle after the step; standing as the
standing bonus (getup_standing), so getting up costs nothing. Walking's g1_tensor_math.arm_press on other joints. Off
(weight 0) nothing is computed and the task is bit for bit the old one."""
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_sole_impact import BEFORE_DIGESTS, BRANCH_823, IMPACT, STANDING
from test_g1_getup_stance_width import BEFORE_46, BRANCH_46, stance_bank

WORKING = {'arm_press': .05}  # the critic's working weight
# Branch 51's recipe (exp-af603bd0d2974b93): branch 46's and stance_width 2; with drop_bank's episodes its digests are
# branch 46's (the code before arm_press, e04f996: the width term is 0 there).
BRANCH_51 = {**BRANCH_46, 'reward_weights': {**BRANCH_46['reward_weights'], 'stance_width': 2.}}
FREE_ARMS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['arm_press'] == 0
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['arm_press'] == 0  # old stages and branches: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': WORKING}})['task_config']
    assert weighted['reward_weights']['arm_press'] == .05
    for value in (-.1, 101., float('nan'), '2', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'arm_press': value}}})
    for config in ({}, BRANCH_51):  # the contract (and so branching) does not change
        weights = {**config.get('reward_weights', {}), **WORKING}
        assert task_identity('g1_29dof', 'g1_getup', {**config, 'reward_weights': weights}) == task_identity(
            'g1_29dof', 'g1_getup', config)
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'arm_press')
    assert field['group'] == 'reward_weights' and field['default'] == 0. and 'стойке' in field['label']


def test_the_joints_and_the_term():
    """The free arm joints are 15-19 and 22-26 by name; the term is −w·[standing]·Σ given one (numpy and torch), 0
    without, and the other terms do not depend on it."""
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    env = G1GetupEnv(bank=drop_bank(), reward_weights=WORKING)
    try:
        names = [env.model.joint(j).name for j in env.joints]
        assert env.press_joints == (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)
        assert all(any(part in names[j] for part in FREE_ARMS) for j in env.press_joints)
    finally:
        env.close()
    inputs = batch(n=64)
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'arm_press': 2.}
    press = np.random.default_rng(1).uniform(0, 8, 64)
    terms, standing = task.getup_reward_components(**inputs, weights=weights, arm_press=press)
    assert 5 < standing.sum() < 30
    np.testing.assert_array_equal(terms['arm_press'], -2. * standing * press)
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['arm_press'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'arm_press')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, arm_press=torch.as_tensor(press))
    np.testing.assert_allclose(torched['arm_press'].numpy(), terms['arm_press'], rtol=1e-12, atol=0)


def hand_press(env, policy_action):
    """Σ over 15-19, 22-26 of (target before pd_bound − angle)², by hand: the box clip, the four locked wrists at 0,
    home + scale·a within the joint ranges, against qpos after the step."""
    a = np.clip(policy_action, -10., 10.)
    a[[20, 21, 27, 28]] = 0.
    target = np.clip(env.home + env.scale * a, env.target_low, env.target_high)
    error = (target - env.data.qpos[env.qadr])[[15, 16, 17, 18, 19, 22, 23, 24, 25, 26]]
    return float((error ** 2).sum())


@pytest.mark.parametrize('bound', (None, .15))
def test_the_cpu_task_charges_the_target_before_the_bound_standing_only(bound):
    """Standing in home (small actions, some beyond the box) and from the back drop, with and without the weight, also
    under a PD error bound c 0.15 that clamps the arms' targets: every observation and other term the same, the term
    −0.05·[standing]·Σ of the target before the bound (by hand), 0 while not standing; the reward differs by it alone.
    Standing in home with zero action Σ is near 0."""
    bank = stance_bank()
    config = dict(reward_weights=WORKING, pd_bound=None if bound is None else [1., .6])
    plain_config = dict(pd_bound=config['pd_bound'])
    weighted, plain = G1GetupEnv(bank=bank, **config), G1GetupEnv(bank=bank, **plain_config)
    rng = np.random.default_rng(2)
    stood = charged = 0
    try:
        for row in (STANDING, 0):
            options = dict(OPTIONS, start=row, **({} if bound is None else {'pd_bound': bound}))
            obs = [env.reset(seed=5, options=options)[0] for env in (weighted, plain)]
            np.testing.assert_array_equal(*obs)
            for k in range(40):
                if row == STANDING:  # the legs and waist at home, the arms off it, now and then beyond the box
                    action = np.zeros(29)
                    action[15:] = rng.normal(0., 12. if k % 5 == 0 else 2., 14)
                else:
                    action = rng.uniform(-10, 10, 29)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                terms_w, terms_p = info_w['reward_components'], info_p['reward_components']
                assert terms_p['arm_press'] == 0 and 'arm_press' not in info_p
                assert {key: v for key, v in terms_w.items() if key != 'arm_press'} == {
                    key: v for key, v in terms_p.items() if key != 'arm_press'}
                press = hand_press(weighted, action)
                assert info_w['arm_press'] == pytest.approx(press, rel=1e-12, abs=1e-15)
                assert terms_w['arm_press'] == pytest.approx(-.05 * info_w['standing'] * press, rel=1e-12, abs=1e-15)
                assert reward_w == pytest.approx(reward_p + terms_w['arm_press'], rel=1e-12, abs=1e-12)
                stood += info_w['standing']
                charged += terms_w['arm_press'] < 0
        assert stood >= 10 and charged >= 10
        weighted.reset(seed=5, options=dict(OPTIONS, start=STANDING))
        for _ in range(20):
            info = weighted.step(np.zeros(29))[-1]
        assert info['standing'] and info['arm_press'] < .05
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    """Weight 0 (missing or explicit): task.arm_press never runs, the term is 0, and every observation, reward, end,
    qpos and other term of training episodes is the code's before the weight (the plain task, pd_bound, impact 2,
    branches 823, 46 and 51)."""
    def refuse(*args, **kwargs):
        raise AssertionError('arm_press computed with its weight at 0')
    monkeypatch.setattr(task, 'arm_press', refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'arm_press': 0.}}, 'plain'),
                         ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'reward_weights': {'arm_press': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    for config, (getup, standing) in ((IMPACT, BEFORE_DIGESTS['impact']), (BRANCH_823, BEFORE_DIGESTS['branch_823']),
                                      (BRANCH_46, BEFORE_46), (BRANCH_51, BEFORE_46)):
        assert getup_digest(config, bank) == getup, config
        assert standing_digest(config, bank) == (standing, 59), config


def test_the_task_builds_from_a_saved_config_with_the_weight():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    for weights, on in ((BRANCH_51['reward_weights'], False), ({**BRANCH_51['reward_weights'], **WORKING}, True)):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup',
                                      task_config={**BRANCH_51, 'reward_weights': weights}))['task_config']
        env = make_task('g1_getup', 'g1_29dof', config)
        try:
            assert env.unwrapped.press_weighted is on
        finally:
            env.close()
