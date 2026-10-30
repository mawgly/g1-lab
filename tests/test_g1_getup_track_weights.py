"""g1_getup task_config track_weights (iteration 80; SuperMimic, table A.2: no self-contact penalty, joint limits at 2 on
the BONES-SEED clips), CPU: a partial dict of G1_GETUP_TRACK_WEIGHTS' keys over the constant. The setting (bounds,
unknown keys, needs a clip, not part of the contract); off (missing or null), or every weight at the constant's, the
task is bit for bit the old one; {"self_contact": 0, "joint_limit": 2} changes those two terms alone (the same
episodes, the other terms equal, joint_limit by hand from the joints, self_contact 0 with no contact forces read);
track_clips and world_model.TrackTerms take the task's weights. The CUDA task reads the reference's track_weights
(g1_getup_tensor_env._setup_clip, the sampler's G_max), so no GPU test of its own."""
import math

import numpy as np
import pytest

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_TRACK_WEIGHTS, TRACK_WEIGHTS_RANGE
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_track import SETTING, synthetic_clip, track_env
from test_g1_getup_track_clips import SETTINGS, front_clip
from test_g1_getup_track_sampling import OLD_EPISODES_DIGEST, episodes_digest
from test_g1_getup_track_starts import OLD_STARTS_DIGEST, starts_digest

WORKING = {'self_contact': 0, 'joint_limit': 2}


def test_the_setting_is_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_weights'] is None  # old stages: the constant
    for tracking in ({'track_clip': SETTING}, {'track_clips': SETTINGS}):
        on = validate_config({**base, 'task_config': {**tracking, 'track_weights': WORKING}})['task_config']
        assert on['track_weights'] == {'joint_limit': 2., 'self_contact': 0.}  # partial, floats
        assert all(type(v) is float for v in on['track_weights'].values())
        full = validate_config({**base, 'task_config': {**tracking, 'track_weights': {
            key: 100 for key in G1_GETUP_TRACK_WEIGHTS}}})['task_config']['track_weights']
        assert full == {key: 100. for key in G1_GETUP_TRACK_WEIGHTS}
        assert validate_config({**base, 'task_config': {**tracking, 'track_weights': None}})['task_config'][
            'track_weights'] is None
        for value in ({'self_contact': -1}, {'joint_limit': 100.5}, {'body_pos': math.inf}, {'body_pos': math.nan},
                      {'body_pos': True}, {'body_pos': '1'}, {'impact': 1}, {'head': 1}, {'unknown': 0}, [0], 'x', 5):
            with pytest.raises(ValueError):
                validate_config({**base, 'task_config': {**tracking, 'track_weights': value}})
        assert task_identity('g1_29dof', 'g1_getup', {**tracking, 'track_weights': WORKING}) == task_identity(
            'g1_29dof', 'g1_getup', tracking)
    with pytest.raises(ValueError, match='track_clip'):
        validate_config({**base, 'task_config': {'track_weights': WORKING}})  # nothing to track
    assert TRACK_WEIGHTS_RANGE == (0., 100.)
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['track_weights']
    assert field['group'] is None and field['default'] is None and field['type'] == 'json'
    assert 'Веса следования' in field['label']


def test_off_the_task_is_bit_for_bit_the_old_one():
    """Missing or null: the old digests (getting up, track_clip's starts and its RSI episodes) and the constant's
    weights; every weight given at the constant's value: the same episodes."""
    assert getup_digest({'track_weights': None}, drop_bank()) == OLD_DIGESTS['plain']
    assert starts_digest(track_weights=None) == OLD_STARTS_DIGEST
    assert episodes_digest(track_weights=None) == OLD_EPISODES_DIGEST
    assert episodes_digest(track_weights=dict(G1_GETUP_TRACK_WEIGHTS)) == OLD_EPISODES_DIGEST
    env = track_env()
    try:
        assert env.track_weights == {**G1_GETUP_TRACK_WEIGHTS, 'impact': 0., 'sole_impact': 0.}
        assert env.self_force is not None  # self_contact 10: read
    finally:
        env.close()


def _run(env, seed=4, resets=6, steps=40):
    """(observations, reward components, rewards, joint positions) of training resets and random actions over a wide
    box, so the joints pass their soft limits and the limbs touch."""
    rng = np.random.default_rng(seed)
    out = []
    env.reset(seed=seed)
    for _ in range(resets):
        env.reset()
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-6., 6., 29))
            out.append((obs, info['reward_components'], reward, env.data.qpos[env.qadr].copy()))
            if terminated or truncated:
                break
    return out


def test_working_weights_change_only_their_terms():
    """{"self_contact": 0, "joint_limit": 2}: the same episodes as the constant's (observations bit for bit: the
    weights do not act on the physics), every other term equal, joint_limit −2·Σ excess beyond the soft limits (by
    hand, and 0.2 of the constant's), self_contact 0 with no contact forces read; the reward the terms' sum."""
    plain, working = track_env(), track_env(track_weights=WORKING)
    try:
        assert working.track_weights == {**G1_GETUP_TRACK_WEIGHTS, 'self_contact': 0., 'joint_limit': 2.,
                                         'impact': 0., 'sole_impact': 0.}
        assert working.self_force is None and plain.self_force is not None
        old, new = _run(plain), _run(working)
        assert len(old) == len(new) > 100
        charged_limit = charged_contact = 0
        for (obs, terms, _, _), (obs_new, terms_new, reward_new, position) in zip(old, new):
            np.testing.assert_array_equal(obs, obs_new)
            assert set(terms) == set(terms_new)
            for key in terms:
                if key not in WORKING:
                    assert terms_new[key] == terms[key], key
            excess = (np.clip(position - working.soft_high, 0, None) + np.clip(working.soft_low - position, 0, None))
            assert terms_new['joint_limit'] == pytest.approx(-2. * excess.sum(), rel=1e-12, abs=1e-15)
            assert terms_new['joint_limit'] == pytest.approx(.2 * terms['joint_limit'], rel=1e-12, abs=1e-15)
            assert terms_new['self_contact'] == 0.
            assert reward_new == pytest.approx(sum(terms_new.values()), rel=1e-12, abs=1e-12)
            charged_limit += terms['joint_limit'] < 0
            charged_contact += terms['self_contact'] < 0
        assert charged_limit > 10 and charged_contact > 10  # the episodes do exercise both terms
    finally:
        plain.close()
        working.close()


def test_track_clips_and_the_world_model_take_the_tasks_weights():
    from humanoid_sim.world_model import TrackTerms
    clips = G1GetupEnv(bank=drop_bank(), track_clips=SETTINGS, track_clips_bank=[synthetic_clip(), front_clip()],
                       track_weights={'body_pos': 3})
    env = track_env(track_weights=WORKING)
    try:
        assert clips.track_weights['body_pos'] == 3. and clips.track_weights['self_contact'] == 10.
        assert TrackTerms(env).weights == env.track_weights
        assert sum(env.track_weights[key] for key in task.TRACK_SIGMA) == sum(
            G1_GETUP_TRACK_WEIGHTS[key] for key in task.TRACK_SIGMA)  # the sampler's G_max: kernel weights unchanged
        env.reset(seed=0, options=dict(OPTIONS, track_frame=5, track_noise=False))
        *_, info = env.step(np.zeros(29))
        assert info['reward_components']['self_contact'] == 0.
    finally:
        clips.close()
        env.close()
