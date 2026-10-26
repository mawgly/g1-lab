"""g1_getup reward weight sole_slip (iteration 62), CPU: −w·Σ over the feet with the lowest sole point below 1 cm of
their horizontal speed² (the ankle_roll origin's displacement over the control step / dt, walking's foot_speed). The
position before a step is read at its start, so a reset starts from its own pose. Off (weight 0) nothing is computed
and the task is bit for bit the old one. CPU–GPU: tests/test_g1_gpu.py."""
import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_stance_width import BEFORE_46, WIDE, hold, stance_bank

WORKING = {'sole_slip': .1}


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['sole_slip'] == 0 and 'sole_slip' not in G1_REWARD_WEIGHTS
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['sole_slip'] == 0
    assert validate_config({**base, 'task_config': {'reward_weights': WORKING}})['task_config'][
        'reward_weights']['sole_slip'] == .1
    for value in (-.1, 101., float('nan'), '0.1', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'sole_slip': value}}})
    for config in ({}, BRANCH_51):
        changed = {**config, 'reward_weights': {**config.get('reward_weights', {}), **WORKING}}
        assert task_identity('g1_29dof', 'g1_getup', changed) == task_identity('g1_29dof', 'g1_getup', config)
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['sole_slip']
    assert field['group'] == 'reward_weights' and field['default'] == 0. and 'скольжение' in field['label']


def test_the_term_against_a_hand_calculation():
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    rng = np.random.default_rng(0)
    inputs = batch(n=32)
    inputs['sole_height'] = rng.choice([-.001, .005, .0099, .01, .02, .3], (32, 2))
    speed = rng.uniform(0., 4., (32, 2))
    contact = inputs['sole_height'] < .01
    expected = (contact * speed ** 2).sum(1)
    np.testing.assert_allclose(task.sole_slip(speed, inputs['sole_height']), expected, rtol=1e-15, atol=0)
    np.testing.assert_allclose(task.sole_slip(torch.as_tensor(speed), torch.as_tensor(inputs['sole_height'])).numpy(),
                               expected, rtol=1e-12, atol=0)
    assert 0 < contact.sum() < 64
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'sole_slip': .1}
    terms, _ = task.getup_reward_components(**inputs, weights=weights, sole_slip=expected)
    np.testing.assert_allclose(terms['sole_slip'], -.1 * expected, rtol=1e-15, atol=0)
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['sole_slip'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'sole_slip')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, sole_slip=torch.as_tensor(expected))
    np.testing.assert_allclose(torched['sole_slip'].numpy(), terms['sole_slip'], rtol=1e-12, atol=0)


def test_the_cpu_task_charges_the_feet_in_contact_from_the_step_start():
    """Flailing from the wide stance, a drop and standing, and a standing start slid sideways at 2 m/s: the term is
    −0.1·Σ [sole < 1 cm]·(|Δxy| / dt)² by hand from the feet's positions before and after each step (the first one
    from the reset's pose), the reward differs from the unweighted task's by it alone."""
    bank = stance_bank()
    weighted = G1GetupEnv(bank=bank, reward_weights=WORKING)
    plain = G1GetupEnv(bank=bank)
    rng = np.random.default_rng(3)
    charged = 0
    try:
        for row, kind in ((WIDE, 'hold'), (0, 'random'), (3, 'slid'), (WIDE, 'random')):
            for env in (weighted, plain):
                env.reset(seed=5, options=dict(OPTIONS, start=row))
                if kind == 'slid':  # the same pose, sliding sideways: nothing moved yet
                    env.data.qvel[1] = 2.
                    mujoco.mj_forward(env.model, env.data)
                    env._observation('fill')
            for k in range(20):
                action = hold(weighted) if kind == 'hold' else rng.uniform(-10, 10, 29) * (.3 if k % 4 else 1.)
                before = weighted.data.xpos[weighted.feet, :2].copy()
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                speed = np.linalg.norm(weighted.data.xpos[weighted.feet, :2] - before, axis=1) / weighted.dt
                sole = weighted._sole_height().numpy()[0]
                expected = float(((sole < .01) * speed ** 2).sum())
                np.testing.assert_allclose(info_w['foot_speed'], speed, rtol=1e-12, atol=1e-12)
                assert info_w['sole_slip'] == pytest.approx(expected, rel=1e-12, abs=1e-12)
                assert info_w['reward_components']['sole_slip'] == pytest.approx(-.1 * expected, rel=1e-12, abs=1e-12)
                assert info_p['reward_components']['sole_slip'] == 0 and 'sole_slip' not in info_p
                assert reward_w == pytest.approx(reward_p + info_w['reward_components']['sole_slip'], rel=1e-12,
                                                 abs=1e-12)
                charged += expected > .01
                if kind == 'slid' and k == 0:
                    assert expected > 1.  # the sliding feet are charged from the reset's own pose
        assert charged >= 10
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('sole_slip computed with its weight at 0')
    monkeypatch.setattr(task, 'sole_slip', refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'sole_slip': 0.}}, 'plain'),
                         ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'reward_weights': {'sole_slip': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    assert getup_digest(BRANCH_51, bank) == BEFORE_46[0]
