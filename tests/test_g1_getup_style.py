"""g1_getup reward weight style (iteration 53), CPU: +w·[0.3 < head < 1.1]·exp(−d/2), d the distance of the frame's
features (the free joints' angles, the gravity in the pelvis frame, 3·pelvis height, 0.1·the free joints' speeds, the
pelvis vertical speed) to the nearest frame of the style library (task_config style_library, a file under
runs/humanoid_lab/getup_style read without pickle). Off (weight 0) nothing is loaded or computed and the task is bit for
bit the old one."""
import json

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_style
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_sole_impact import BEFORE_DIGESTS, BRANCH_823, IMPACT
from test_g1_getup_stance_width import BEFORE_46, BRANCH_46

WORKING = {'style': .5}  # the critic's working weight
FREE = tuple(j for j in range(29) if j not in (20, 21, 27, 28))  # all but the locked wrists


def write_library(folder, name, features, joints=FREE, version=task.STYLE_VERSION, holdout=None, **extra):
    """A style library file as getup_style_library.py writes it."""
    features = np.asarray(features, dtype=np.float64)
    holdout = features[:3] if holdout is None else holdout
    arrays = dict(features=features, holdout=holdout, joints=np.array(joints, dtype=np.int64),
                  source=np.zeros(len(features), dtype=np.int64), time=np.arange(len(features)) * .02,
                  header=np.array(json.dumps(dict(version=version))), **extra)
    np.savez(folder / f'{name}.npz', **arrays)


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A library 'test' of 400 frames spread around the drop rows' start features (random, seed 3)."""
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    env = G1GetupEnv(bank=drop_bank())
    try:
        centres = []
        for row in range(3):
            env.reset(seed=1, options=dict(OPTIONS, start=row))
            centres.append(env.style_features()[0])
    finally:
        env.close()
    rng = np.random.default_rng(3)
    features = np.concatenate([centre + rng.normal(0., .4, (400 // 3 + 1, len(centre))) for centre in centres])[:400]
    write_library(tmp_path, 'test', features)
    return features


def test_the_weight_and_the_library_setting_are_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['style'] == 0 and 'style' not in G1_REWARD_WEIGHTS
    assert task.STYLE_HEAD_BAND == (.3, 1.1) and task.STYLE_KERNEL == 2.
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    plain = validate_config(base)['task_config']
    assert plain['reward_weights']['style'] == 0 and plain['style_library'] == 'lafan1'  # old stages: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': WORKING, 'style_library': 'mine-2'}})
    assert weighted['task_config']['reward_weights']['style'] == .5
    assert weighted['task_config']['style_library'] == 'mine-2'
    for value in (-.1, 101., float('nan'), '2', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'style': value}}})
    for name in ('', '../lafan1', 'a/b', 'a\\b', 'C:x', 'x.npz', 'x' * 65, 5, None, True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'style_library': name}})
        with pytest.raises(ValueError):
            g1_getup_style.library_path(name)
    for config in ({}, BRANCH_51):  # the contract (and so branching) does not change
        changed = {**config, 'style_library': 'other', 'reward_weights': {**config.get('reward_weights', {}),
                                                                           **WORKING}}
        assert task_identity('g1_29dof', 'g1_getup', changed) == task_identity('g1_29dof', 'g1_getup', config)
    fields = {p['key']: p for p in G1_GETUP_PARAMETERS}
    assert fields['style']['group'] == 'reward_weights' and fields['style']['default'] == 0.
    assert 'стиль' in fields['style']['label'] and '0,3–1,1 м' in fields['style']['label']
    assert fields['style_library']['group'] is None and fields['style_library']['default'] == 'lafan1'
    assert fields['style_library']['type'] == 'text' and 'getup_style' in fields['style_library']['label']


def test_the_features_the_distance_and_the_term():
    """style_features as the layout says (numpy and torch alike), style_distance against every row by brute force, the
    term w·[band]·exp(−d/2) (0 without d, the other terms unchanged)."""
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    rng = np.random.default_rng(0)
    n = 32
    position, speed = rng.normal(0, 1, (n, 29)), rng.normal(0, 5, (n, 29))
    rotation = np.linalg.qr(rng.normal(0, 1, (n, 3, 3)))[0]
    height, vertical = rng.uniform(.1, .8, n), rng.normal(0, 1, n)
    features = task.style_features(position, speed, rotation, height, vertical, FREE)
    assert features.shape == (n, 55)
    for k in range(n):
        gravity = rotation[k].T @ np.array([0., 0., -1.])
        expected = np.concatenate([position[k, list(FREE)], gravity, [3 * height[k]], .1 * speed[k, list(FREE)],
                                   [vertical[k]]])
        np.testing.assert_allclose(features[k], expected, rtol=0, atol=1e-15)
    torched = task.style_features(*(torch.as_tensor(v) for v in (position, speed, rotation, height, vertical)), FREE)
    np.testing.assert_array_equal(torched.numpy(), features)
    rows = rng.normal(0, 1, (500, 55))
    distance, index = task.style_distance(features, rows, task.style_norms(rows))
    brute = np.sqrt(((features[:, None] - rows[None]) ** 2).sum(-1))
    np.testing.assert_array_equal(index, brute.argmin(1))
    np.testing.assert_allclose(distance, brute.min(1), rtol=1e-12)
    t_rows = torch.as_tensor(rows)
    t_distance, t_index = task.style_distance(torch.as_tensor(features), t_rows, task.style_norms(t_rows))
    np.testing.assert_array_equal(t_index.numpy(), index)
    np.testing.assert_allclose(t_distance.numpy(), distance, rtol=1e-12)
    f32, _ = task.style_distance(torch.as_tensor(features, dtype=torch.float32), t_rows.float(),
                                 task.style_norms(t_rows.float()))  # the CUDA task's float32
    np.testing.assert_allclose(f32.numpy(), distance, rtol=1e-4)
    inputs = batch(n=64)
    inputs['head'] = np.linspace(.1, 1.3, 64)
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'style': 2.}
    d = rng.uniform(0, 8, 64)
    terms, _ = task.getup_reward_components(**inputs, weights=weights, style=d)
    band = (inputs['head'] > .3) & (inputs['head'] < 1.1)
    assert 20 < band.sum() < 50
    np.testing.assert_allclose(terms['style'], 2. * band * np.exp(-d / 2), rtol=1e-15, atol=0)
    assert np.array_equal(task.style_band(np.array([.3, .30001, 1.09999, 1.1])), [False, True, True, False])
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['style'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'style')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, style=torch.as_tensor(d))
    np.testing.assert_allclose(torched['style'].numpy(), terms['style'], rtol=1e-12, atol=0)


def test_the_library_is_read_without_pickle_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='getup_style_library.py'):
        g1_getup_style.load('absent')
    rows = np.random.default_rng(1).normal(0, 1, (20, 55))
    write_library(tmp_path, 'good', rows)
    library = g1_getup_style.load('good')
    np.testing.assert_array_equal(library['features'], rows)
    np.testing.assert_allclose(library['norms'], (rows ** 2).sum(1))
    assert tuple(library['joints'].tolist()) == FREE and library['header']['version'] == task.STYLE_VERSION
    assert not library['features'].flags.writeable
    write_library(tmp_path, 'pickled', rows, extra=np.array([{'a': 1}], dtype=object))
    with pytest.raises(ValueError):  # an object array needs pickle, which np.load refuses
        g1_getup_style.load('pickled')
    write_library(tmp_path, 'old', rows, version='getup-style-v0')
    with pytest.raises(ValueError, match='version'):
        g1_getup_style.load('old')
    write_library(tmp_path, 'narrow', rows[:, :50])
    with pytest.raises(ValueError, match='feature'):
        g1_getup_style.load('narrow')
    write_library(tmp_path, 'joints', rows[:, :53], joints=FREE[:-1])
    with pytest.raises(ValueError, match='joints'):
        G1GetupEnv(bank=drop_bank(), reward_weights=WORKING, style_library='joints')


def test_the_cpu_task_pays_the_nearness_with_the_head_in_the_band(library):
    """The drop rows (the head starts ~0.4 m up, then lies below 0.3) and the standing row under random actions, with
    and without the weight: every observation and other term the same, d the brute-force distance of the frame after
    the step (by hand from qpos, qvel and the pelvis rotation) to the library, the term 0.5·[band]·exp(−d/2); the
    reward differs by it alone."""
    bank = drop_bank()
    weighted = G1GetupEnv(bank=bank, reward_weights=WORKING, style_library='test')
    plain = G1GetupEnv(bank=bank)
    rng = np.random.default_rng(2)
    paid = outside = 0
    try:
        assert weighted.style_joints == FREE and plain.style is None
        for row in (0, 1, 2, 3):
            obs = [env.reset(seed=5, options=dict(OPTIONS, start=row))[0] for env in (weighted, plain)]
            np.testing.assert_array_equal(*obs)
            for k in range(30):
                action = rng.uniform(-10, 10, 29) * (.3 if k % 4 else 1.)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                terms_w, terms_p = info_w['reward_components'], info_p['reward_components']
                assert terms_p['style'] == 0 and 'style_dist' not in info_p
                assert {key: v for key, v in terms_w.items() if key != 'style'} == {
                    key: v for key, v in terms_p.items() if key != 'style'}
                band = .3 < info_w['head_height'] < 1.1
                assert info_w['style_band'] is band
                if band:
                    d = weighted.data
                    rotation = d.xmat[weighted.pelvis].reshape(3, 3)
                    x = np.concatenate([d.qpos[7:][list(FREE)], -rotation[2], [3 * d.qpos[2]],
                                        .1 * d.qvel[6:][list(FREE)], [d.qvel[2]]])
                    brute = np.sqrt(((library - x) ** 2).sum(1))
                    assert info_w['style_dist'] == pytest.approx(brute.min(), rel=1e-9)
                    assert info_w['style_frame'] == int(brute.argmin())
                    assert terms_w['style'] == pytest.approx(.5 * np.exp(-brute.min() / 2), rel=1e-9, abs=0)
                    assert terms_w['style'] > 0
                else:  # outside the band nothing is computed
                    assert 'style_dist' not in info_w and terms_w['style'] == 0
                assert reward_w == pytest.approx(reward_p + terms_w['style'], rel=1e-12, abs=1e-12)
                paid += band
                outside += not band
        assert paid >= 10 and outside >= 10
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_loaded_or_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    """Weight 0 (missing or explicit, any library name, also one that does not exist): no library is read,
    task.style_features / style_distance never run, the term is 0, and every observation, reward, end, qpos and other
    term of training episodes is the code's before the weight (the plain task, pd_bound, impact 2, branches 823, 46 and
    51)."""
    def refuse(*args, **kwargs):
        raise AssertionError('style computed or loaded with its weight at 0')
    for name in ('style_features', 'style_distance'):
        monkeypatch.setattr(task, name, refuse)
    monkeypatch.setattr(g1_getup_style, 'load', refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'style': 0.}}, 'plain'),
                         ({'style_library': 'absent'}, 'plain'), ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'reward_weights': {'style': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    for config, (getup, standing) in ((IMPACT, BEFORE_DIGESTS['impact']), (BRANCH_823, BEFORE_DIGESTS['branch_823']),
                                      (BRANCH_46, BEFORE_46), (BRANCH_51, BEFORE_46)):
        assert getup_digest(config, bank) == getup, config
        assert standing_digest(config, bank) == (standing, 59), config


def test_the_task_builds_from_a_saved_config_with_the_weight(library):
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    for weights, on in ((BRANCH_51['reward_weights'], False), ({**BRANCH_51['reward_weights'], **WORKING}, True)):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', task_config={
            **BRANCH_51, 'reward_weights': weights, 'style_library': 'test'}))['task_config']
        env = make_task('g1_getup', 'g1_29dof', config)
        try:
            assert (env.unwrapped.style is not None) is on
        finally:
            env.close()


@pytest.mark.skipif(not g1_getup_style.library_path('lafan1').is_file(),
                    reason='the lab library runs/humanoid_lab/getup_style/lafan1.npz is made locally')
def test_the_lab_library():
    """The library getup_style_library.py makes: subjects 1-4 (~15.7 k frames at 50 Hz), the held-out subject 5 near
    it (median d ~1.6; knn53.py's pose-only 1.26), the free joints."""
    library = g1_getup_style.load('lafan1')
    assert tuple(library['joints'].tolist()) == FREE and library['features'].shape[1] == 55
    assert 10_000 < len(library['features']) < 25_000 and library['header']['holdout_subject'] == 5
    distance, _ = task.style_distance(library['holdout'], library['features'], library['norms'])
    assert 1. < np.median(distance) < 2.2
