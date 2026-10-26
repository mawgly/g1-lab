"""g1_getup reward weight head_rise (iteration 57), CPU: −w·max(0, v_z − 0.8 m/s), v_z the head top's vertical speed
over the control step from its height a step before (at a reset: the start's), every step. Off (weight 0) nothing is
computed and the task is bit for bit the old one. CPU–GPU: tests/test_g1_gpu.py."""
import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_stance_width import BEFORE_46

WORKING = {'head_rise': 3.}


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['head_rise'] == 0 and 'head_rise' not in G1_REWARD_WEIGHTS
    assert task.GETUP_HEAD_RISE_SPEED == .8
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['head_rise'] == 0
    weighted = validate_config({**base, 'task_config': {'reward_weights': WORKING}})['task_config']
    assert weighted['reward_weights']['head_rise'] == 3.
    for value in (-.1, 101., float('nan'), '3', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'head_rise': value}}})
    for config in ({}, BRANCH_51):
        changed = {**config, 'reward_weights': {**config.get('reward_weights', {}), **WORKING}}
        assert task_identity('g1_29dof', 'g1_getup', changed) == task_identity('g1_29dof', 'g1_getup', config)
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['head_rise']
    assert field['group'] == 'reward_weights' and field['default'] == 0. and '0,8 м/с' in field['label']


def test_the_term_against_a_hand_calculation():
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    inputs = batch(n=64)
    speed = np.linspace(-3., 3., 64)
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'head_rise': 3.}
    terms, _ = task.getup_reward_components(**inputs, weights=weights, head_speed=speed)
    np.testing.assert_allclose(terms['head_rise'], -3. * np.maximum(0., speed - .8), rtol=1e-15, atol=0)
    assert (terms['head_rise'][speed <= .8] == 0).all() and (terms['head_rise'][speed > .8] < 0).all()
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['head_rise'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'head_rise')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, head_speed=torch.as_tensor(speed))
    np.testing.assert_allclose(torched['head_rise'].numpy(), terms['head_rise'], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(task.head_rise(np.array([.8, .81, -1.])), [0., .81 - .8, 0.])


def test_the_cpu_task_charges_the_head_speed_from_the_step_before_and_a_reset_starts_from_the_start():
    """Random actions from every drop row and standing, plus a standing start thrown upwards at 3 m/s: the term is
    −3·max(0, (head − head a step before)/dt − 0.8) by hand from the heads the task reports, the first step's speed from
    the reset's head; the reward differs from the unweighted task's by the term alone, every observation the same."""
    bank = drop_bank()
    weighted = G1GetupEnv(bank=bank, reward_weights=WORKING)
    plain = G1GetupEnv(bank=bank)
    rng = np.random.default_rng(4)
    charged = 0
    try:
        for row in (0, 1, 2, 3, 'thrown'):
            starts = []
            for env in (weighted, plain):
                obs, info = env.reset(seed=5, options=dict(OPTIONS, start=3 if row == 'thrown' else row))
                if row == 'thrown':  # the same pose, moving up: the head's height at the reset is unchanged
                    env.data.qvel[2] = 3.
                    mujoco.mj_forward(env.model, env.data)
                    obs = env._observation('fill')
                starts.append((obs, info['head_height']))
            np.testing.assert_array_equal(starts[0][0], starts[1][0])
            head = starts[0][1]
            assert head == pytest.approx(float(weighted._head()[0]), abs=0)  # the reset's own head
            for k in range(25):
                action = rng.uniform(-10, 10, 29) * (.3 if k % 4 else 1.)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                speed = (info_w['head_height'] - head) / weighted.dt
                assert info_w['head_speed'] == pytest.approx(speed, rel=1e-12, abs=1e-12) and 'head_speed' not in info_p
                term = info_w['reward_components']['head_rise']
                assert term == pytest.approx(-3. * max(0., speed - .8), rel=1e-12, abs=1e-12)
                assert info_p['reward_components']['head_rise'] == 0
                assert reward_w == pytest.approx(reward_p + term, rel=1e-12, abs=1e-12)
                charged += term < 0
                head = info_w['head_height']
        assert charged >= 3
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('head_rise computed with its weight at 0')
    monkeypatch.setattr(task, 'head_rise', refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'head_rise': 0.}}, 'plain'),
                         ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'reward_weights': {'head_rise': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    assert getup_digest(BRANCH_51, bank) == BEFORE_46[0]
