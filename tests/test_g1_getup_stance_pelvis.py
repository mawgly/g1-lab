"""g1_getup task_config stance_pelvis (iteration 60), CPU: the stance_width term counts with the pelvis above it (and
both soles below 1 cm) instead of the fixed 0.5 m, so a split low down is charged too (stage 1164 jumps out of one).
Missing or 0.5 is the task before it, bit for bit. CPU–GPU: tests/test_g1_gpu.py."""
import numpy as np
import pytest

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import standing_digest
from test_g1_getup_stance_width import BEFORE_46, BRANCH_46, WIDE, hold, stance_bank

WORKING = .25  # the lead's working value


def test_the_setting_is_validated_labelled_and_leaves_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import GETUP_STANCE_PELVIS, G1_GETUP_PARAMETERS
    from humanoid_sim.tasks.identity import task_identity
    assert GETUP_STANCE_PELVIS == task.GETUP_STANCE_PELVIS == .5
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['stance_pelvis'] == .5  # old stages: the original threshold
    assert validate_config({**base, 'task_config': {'stance_pelvis': WORKING}})['task_config']['stance_pelvis'] == .25
    for value in (.05, .81, float('nan'), '0.25', True, None):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'stance_pelvis': value}})
    for config in ({}, BRANCH_51):
        assert task_identity('g1_29dof', 'g1_getup', {**config, 'stance_pelvis': WORKING}) \
            == task_identity('g1_29dof', 'g1_getup', config)
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['stance_pelvis']
    assert field['group'] is None and field['default'] == .5 and (field['min'], field['max']) == (.1, .8)
    assert 'таз выше' in field['label']


def test_the_term_against_a_hand_calculation():
    """−w·[pelvis > p, both soles < 1 cm]·max(0, W − 0.30) at p 0.25 and 0.5, numpy and torch alike."""
    import torch
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    inputs = batch(n=8)
    inputs['height'] = np.array([.2, .26, .3, .45, .49, .51, .7, .3])
    inputs['sole_height'] = np.array([[0., 0.]] * 7 + [[0., .05]])
    width = np.full(8, 1.05)  # a split
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'stance_width': 2.}
    for pelvis in (.25, .5):
        terms, _ = task.getup_reward_components(**inputs, weights=weights, stance_width=width, stance_pelvis=pelvis)
        active = (inputs['height'] > pelvis) & (inputs['sole_height'] < .01).all(1)
        np.testing.assert_allclose(terms['stance_width'], -2. * active * (1.05 - .30), rtol=1e-15, atol=0)
        torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights,
                                                  stance_width=torch.as_tensor(width), stance_pelvis=pelvis)
        np.testing.assert_allclose(torched['stance_width'].numpy(), terms['stance_width'], rtol=1e-12, atol=0)
    low = task.stance_active(inputs['height'], inputs['sole_height'], .25)
    assert low.tolist() == [False, True, True, True, True, True, True, False]
    np.testing.assert_array_equal(task.stance_active(inputs['height'], inputs['sole_height']),
                                  task.stance_active(inputs['height'], inputs['sole_height'], .5))
    default, _ = task.getup_reward_components(**inputs, weights=weights, stance_width=width)
    explicit, _ = task.getup_reward_components(**inputs, weights=weights, stance_width=width, stance_pelvis=.5)
    np.testing.assert_array_equal(default['stance_width'], explicit['stance_width'])


def test_the_cpu_task_takes_the_threshold():
    """The wide stance held by its PD targets: the pelvis about 0.75 m up, so charged at the default 0.5 and at 0.25,
    not at 0.8 (the pelvis below it): the setting reaches both the term and info['stance_active']."""
    weights = {'stance_width': 2.}
    for pelvis, active in ((None, True), (.25, True), (.8, False)):
        settings = {} if pelvis is None else {'stance_pelvis': pelvis}
        env = G1GetupEnv(bank=stance_bank(), reward_weights=weights, **settings)
        try:
            env.reset(seed=0, options=dict(OPTIONS, start=WIDE))
            for _ in range(10):
                info = env.step(hold(env))[-1]
                assert info['pelvis_height'] < .8 and info['stance_active'] is active, (pelvis, info['pelvis_height'])
                expected = -2. * max(0., info['stance_width'] - .30) if active else 0.
                assert info['reward_components']['stance_width'] == pytest.approx(expected, rel=1e-12, abs=1e-15)
                assert (expected < -.15) is active
        finally:
            env.close()


def test_missing_or_the_original_threshold_is_bit_for_bit_the_old_task():
    """Branch 46's recipe with the stance_width weight 2 gives the digests of the code before the weight (the drop
    episodes never stand wide); with the setting missing or at 0.5 the stance_width world-steps of flailing episodes
    are charged exactly as before (the term reads the same threshold)."""
    bank = drop_bank()
    recipe = {**BRANCH_46, 'reward_weights': {**BRANCH_46['reward_weights'], 'stance_width': 2.}}
    for config in (recipe, {**recipe, 'stance_pelvis': .5}):
        assert getup_digest(config, bank) == BEFORE_46[0], config
        assert standing_digest(config, bank) == (BEFORE_46[1], 59), config
    rng = np.random.default_rng(8)
    stance = stance_bank()
    envs = [G1GetupEnv(bank=stance, reward_weights={'stance_width': 2.}, **extra)
            for extra in ({}, {'stance_pelvis': .5})]
    try:
        for env in envs:
            env.reset(seed=3, options=dict(OPTIONS, start=WIDE))
        for k in range(40):
            action = rng.uniform(-10, 10, 29) * (.3 if k % 5 else 1.)
            (obs_a, reward_a, *_, info_a), (obs_b, reward_b, *_, info_b) = (env.step(action) for env in envs)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert reward_a == reward_b and info_a['reward_components'] == info_b['reward_components']
    finally:
        for env in envs:
            env.close()
