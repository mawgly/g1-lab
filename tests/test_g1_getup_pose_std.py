"""g1_getup task_config stand_pose_std (iteration 36), CPU: the width σ of the standing pose term
w·[standing]·exp(−mean((q − home)²)/σ²). Missing, it is GETUP_POSE_STD 0.1 and the task is bit for bit the old one; a
reward setting only, so the contract and branching are unchanged; walking does not know it."""
import hashlib
import math

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.lab import validate_config
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import (G1_GETUP_DEFAULTS, G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, g1_config,
                                       g1_getup_config)
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_math import as_torch, batch

WORKING = .7  # the critic's working value (lab-critic/it36)
STANDING = 3  # drop_bank's row standing in home on the floor
# Reward terms added after standing_digest's code, always among the components (0 without their weight): left out.
LATER_TERMS = ('sole_impact', 'stance_width')


def pose_term(position, home, std, standing=True):
    """The pose term as specified: w·[standing]·exp(−mean over the joints of (q − home)²/σ²)."""
    mean = ((np.asarray(position) - home) ** 2).mean(-1)
    return G1_GETUP_REWARD_WEIGHTS['stand_pose'] * np.asarray(standing, dtype=float) * np.exp(-mean / std ** 2)


def standing_digest(config, bank, seed=36, episodes=4, steps=60):
    """sha256 of every observation, reward, end flag, standing state, qpos and reward term (but LATER_TERMS) of a few
    episodes that stand: from drop_bank's standing row (no assist, delay or friction draws) and from training resets,
    small random actions (σ 0.3) that move the joints off home without a fall, so the pose term is live (getup_digest's
    episodes never stand). Returns (digest, standing steps)."""
    env = G1GetupEnv(max_episode_steps=steps, bank=bank, **config)
    rng = np.random.default_rng(seed)
    digest, stood = hashlib.sha256(), 0
    try:
        for episode in range(episodes):
            options = dict(OPTIONS, start=STANDING) if episode % 2 == 0 else None
            obs, _ = env.reset(seed=seed + episode, options=options)
            digest.update(obs.tobytes())
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(rng.normal(0., .3, 29))
                digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes(
                    [terminated, truncated, info['standing']]) + env.data.qpos.tobytes())
                for key in sorted(info['reward_components']):
                    if key not in LATER_TERMS:
                        digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
                stood += info['standing']
                if terminated or truncated:
                    break
    finally:
        env.close()
    return digest.hexdigest(), stood


def test_the_setting_is_bounded_spelled_out_and_leaves_the_contract_alone():
    assert G1_GETUP_DEFAULTS['stand_pose_std'] == task.GETUP_POSE_STD == .1
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['stand_pose_std'] == .1  # old stages and branches: the old term
    for given in (WORKING, 1, .02, 2.):
        kept = validate_config({**base, 'task_config': {'stand_pose_std': given}})['task_config']['stand_pose_std']
        assert kept == float(given) and type(kept) is float
    for bad in (.01, 2.5, 0., -.7, float('nan'), float('inf'), '0.7', True, None, [.7]):
        with pytest.raises(ValueError):
            g1_getup_config({'stand_pose_std': bad})
    with pytest.raises(ValueError):  # walking has no such setting
        g1_config({'stand_pose_std': WORKING})
    for config in ({}, {'pd_bound': [1., .6]}, {'reward_weights': {'impact': 2.}, 'pd_bound': [1., .6]}):
        saved = task_identity('g1_29dof', 'g1_getup', config)
        wider = task_identity('g1_29dof', 'g1_getup', {**config, 'stand_pose_std': WORKING})
        assert wider == saved
        validate_resume(saved, wider)  # an ordinary branch of the same contract
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'stand_pose_std')
    assert field['group'] is None and field['default'] == .1 and (field['min'], field['max']) == (.02, 2.)
    assert 'σ' in field['label'] and '0,1' in field['label']


def test_numpy_and_torch_give_the_same_pose_term_and_the_default_is_the_old_one():
    inputs = batch()
    old, old_standing = task.getup_reward_components(**inputs)
    explicit, _ = task.getup_reward_components(**inputs, stand_pose_std=.1)
    assert all(np.array_equal(old[key], explicit[key]) for key in old)  # the default, bit for bit
    np.testing.assert_allclose(old['stand_pose'], pose_term(inputs['position'], inputs['home'], .1, old_standing),
                               rtol=1e-12, atol=0)
    for std in (.02, .3, WORKING, 2.):
        expected, standing = task.getup_reward_components(**inputs, stand_pose_std=std)
        np.testing.assert_array_equal(standing, old_standing)
        assert all(np.array_equal(expected[key], old[key]) for key in old if key != 'stand_pose')  # only the pose term
        np.testing.assert_allclose(expected['stand_pose'], pose_term(inputs['position'], inputs['home'], std, standing),
                                   rtol=1e-12, atol=0)
        assert not expected['stand_pose'][~standing].any()
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
            got, got_standing = task.getup_reward_components(**as_torch(inputs, dtype), stand_pose_std=std)
            np.testing.assert_array_equal(got_standing.numpy(), standing)
            assert got['stand_pose'].dtype == dtype
            for key in expected:
                np.testing.assert_allclose(got[key].numpy(), expected[key], rtol=tolerance, atol=tolerance,
                                           err_msg=f'{key} at σ {std}')
    wide, _ = task.getup_reward_components(**inputs, stand_pose_std=WORKING)
    standing = old_standing
    assert (wide['stand_pose'][standing] > old['stand_pose'][standing]).all()  # wider: more for any pose off home


def test_the_term_on_given_poses():
    """Standing in home: w = 0.5 at any σ. The critic's poses (lab-critic/it36/post_pose.txt, raw_targets.py): after
    getting up Σ over the 29 joints of (q − home)² ≈ 23 rad², at σ 0.1 e^−79 (dead), at 0.7 0.099 per standing step;
    candidate 8 standing 0.18 rad²: 0.494 at 0.7. Home + 0.1 rad on every joint: 0.5·e^−1 at 0.1."""
    inputs = batch(n=4)
    home = inputs['home']
    rng = np.random.default_rng(7)
    position = np.stack([home, home + .1, home, home])
    for row, total in ((2, 23.), (3, .18)):  # random directions, the given Σ (q − home)²
        offset = rng.normal(size=29)
        position[row] = home + offset * math.sqrt(total / (offset ** 2).sum())
    np.testing.assert_allclose(((position - home) ** 2).sum(-1), [0., .29, 23., .18], rtol=1e-12, atol=1e-15)
    inputs.update(head=np.full(4, 1.33), height=np.full(4, .79), torso_rotation=np.stack([np.eye(3)] * 4),
                  velocity=np.zeros((4, 6)), position=position, sole_height=np.full((4, 2), -.0006))
    terms = {std: task.getup_reward_components(**inputs, stand_pose_std=std) for std in (.1, WORKING)}
    assert all(standing.all() for _, standing in terms.values())
    old, wide = terms[.1][0]['stand_pose'], terms[WORKING][0]['stand_pose']
    assert old[0] == wide[0] == .5
    assert old[1] == pytest.approx(.5 * math.exp(-1), rel=1e-12)
    assert wide[1] == pytest.approx(.5 * math.exp(-.01 / .49), rel=1e-12)
    assert old[2] == pytest.approx(.5 * math.exp(-23 / 29 / .01), rel=1e-9) and old[2] < 1e-34
    assert wide[2] == pytest.approx(.5 * math.exp(-23 / 29 / .49), rel=1e-12) and .09 < wide[2] < .10
    assert wide[3] == pytest.approx(.494, abs=5e-4) and old[3] == pytest.approx(.5 * math.exp(-.18 / 29 / .01))


def test_the_cpu_task_changes_the_pose_term_only():
    """The same episodes with σ 0.7 and without the setting: every observation and every other term the same, the pose
    term w·[standing]·exp(−mean((q − home)²)/σ²) of the step's own joint angles with each σ (0.1 without), and the
    reward differs by it alone. Standing in home on the floor (zero action) earns it; lying, nothing."""
    bank = drop_bank()
    wide, plain = G1GetupEnv(bank=bank, stand_pose_std=WORKING), G1GetupEnv(bank=bank)
    rng = np.random.default_rng(3)
    stood = 0
    try:
        assert (wide.stand_pose_std, plain.stand_pose_std) == (WORKING, .1)
        for row in (STANDING, 0, 1):
            obs = [env.reset(seed=2, options=dict(OPTIONS, start=row))[0] for env in (wide, plain)]
            np.testing.assert_array_equal(*obs)
            for k in range(30):
                action = np.zeros(29) if row == STANDING else rng.uniform(-10., 10., 29) * (.3 if k % 3 else 1.)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (wide, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                terms_w, terms_p = info_w['reward_components'], info_p['reward_components']
                assert {key: v for key, v in terms_w.items() if key != 'stand_pose'} == {
                    key: v for key, v in terms_p.items() if key != 'stand_pose'}
                assert info_w['standing'] == info_p['standing']
                position = wide.data.qpos[wide.qadr]
                np.testing.assert_array_equal(position, plain.data.qpos[plain.qadr])
                assert terms_w['stand_pose'] == pytest.approx(
                    float(pose_term(position, wide.home, WORKING, info_w['standing'])), rel=1e-12, abs=0)
                assert terms_p['stand_pose'] == pytest.approx(
                    float(pose_term(position, plain.home, .1, info_p['standing'])), rel=1e-12, abs=0)
                assert reward_w == pytest.approx(reward_p - terms_p['stand_pose'] + terms_w['stand_pose'],
                                                 rel=1e-12, abs=1e-12)
                if row == STANDING:
                    assert info_w['standing'] and terms_w['stand_pose'] > .49
                    stood += 1
                elif not info_w['standing']:
                    assert terms_w['stand_pose'] == terms_p['stand_pose'] == 0
        assert stood == 30
    finally:
        wide.close(), plain.close()


# standing_digest({}) of the code before the setting (commit edc7646), with drop_bank(): 59 standing steps. Another
# MuJoCo, numpy or torch build changes it: then make it again with that commit's code.
OLD_STANDING_DIGEST = 'b6fbe081ac7b8f0fdd88cdb9f52e65d1e7855759ecd1e312b132fc969ff44a71'


def test_without_the_setting_the_task_is_bit_for_bit_the_old_one():
    """Missing or 0.1: every observation, reward, end, qpos and term of training episodes is the code's before the
    setting: getup_digest's (OLD_DIGESTS, also with pd_bound; they never stand) and standing_digest's, whose 59 standing
    steps earn the pose term (OLD_STANDING_DIGEST). σ 0.7 changes the rewards of those, not the episodes."""
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'stand_pose_std': .1}, 'plain'),
                         ({'pd_bound': [1., .6], 'stand_pose_std': .1}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    for config in ({}, {'stand_pose_std': .1}):
        assert standing_digest(config, bank) == (OLD_STANDING_DIGEST, 59), config
    digest, stood = standing_digest({'stand_pose_std': WORKING}, bank)
    assert digest != OLD_STANDING_DIGEST and stood == 59


def test_the_task_builds_from_a_saved_config_with_the_setting():
    """make_task (evaluation, playback, probes and the trainers build from the run's or stage's task_config): a config
    without the key gives 0.1, with it the given σ."""
    from humanoid_sim.tasks.catalog import make_task
    for config, std in (({}, .1), ({'stand_pose_std': WORKING, 'pd_bound': [1., .6]}, WORKING)):
        env = make_task('g1_getup', 'g1_29dof', validate_config(dict(robot_id='g1_29dof', task_id='g1_getup',
                                                                     task_config=config))['task_config'])
        try:
            assert env.unwrapped.stand_pose_std == std
        finally:
            env.close()
