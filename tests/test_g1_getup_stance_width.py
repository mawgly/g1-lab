"""g1_getup reward weight stance_width (after branch 46), CPU: −w·[on the feet]·max(0, W − 0.30), W the left − right
foot (ankle_roll origins) across the pelvis heading (g1_tensor_math.stance_width, scripts/lab/probes/getup_stance.py's
width), on the feet: the pelvis above 0.5 m and both soles below 1 cm (stance_active), nothing asked of the head or
the tilt. Off (weight 0) nothing is computed and the task is bit for bit the old one. The CUDA task:
tests/test_g1_gpu.py."""
import math

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest
from test_g1_getup_sole_impact import (BEFORE_DIGESTS, BRANCH_823, IMPACT, ON_FLOOR, STAND_DROP, STANDING,
                                       sole_bank)

WORKING = {'stance_width': 2.}  # the lead's working weight
SPREAD = .15  # rad: row WIDE's hip roll outwards and ankle roll back, each leg (W 0.41 m; home 0.24)
WIDE = 7  # stance_bank's row: the wide stance on the floor
HIP_ROLL, ANKLE_ROLL = (1, 7), (5, 11)  # left, right
BRANCH_46 = {'reward_weights': {'impact': 2., 'sole_impact': 1.}, 'pd_bound': [1., .6], 'stand_pose_std': .3,
             'action_change_unbounded': True}  # exp-e9a0735be3e0418a
# getup_digest and standing_digest of branch 46's recipe with the code before the weight (commit cb9bfbd), drop_bank();
# BEFORE_DIGESTS and OLD_DIGESTS cover the recipes before it. Another MuJoCo, numpy or torch build changes them: then
# make them again with that commit's code.
BEFORE_46 = ('6254b90e1948cfe0deca4a4ed6e6f0224474ab510072e2fd20f66e8b1890353d',
             '234af346608d13f442104e07cec0692764690561c44acfc11ff95e41dfd2b95d')


def wide_pose(home):
    """Home with each hip roll SPREAD outwards (left +, right −) and the ankle roll back, so the soles stay flat."""
    q = home.copy()
    q[list(HIP_ROLL)] += [SPREAD, -SPREAD]
    q[list(ANKLE_ROLL)] += [-SPREAD, SPREAD]
    return q


def stance_bank():
    """sole_bank's rows (STAND_DROP, SOFT, STANDING among them), then the wide pose upright with the soles ON_FLOOR
    (row 7)."""
    bank = sole_bank()
    env = bank_module.getup_model_env()
    m, d = env.model, env.data
    floor = m.geom('floor').id
    mujoco.mj_resetData(m, d)
    d.qpos[2], d.qpos[3] = 1., 1.
    d.qpos[env.qadr] = wide_pose(env.home)
    mujoco.mj_forward(m, d)
    d.qpos[2] -= min(mujoco.mj_geomDistance(m, d, floor, g, 2., None) for g in env.soles.ravel()) - ON_FLOOR
    return dict(qpos=np.concatenate([bank['qpos'], [d.qpos.copy()]]),
                qvel=np.concatenate([bank['qvel'], np.zeros((1, m.nv))]),
                start_class=np.concatenate([bank['start_class'], [5]]).astype(np.uint8),
                holdout=np.zeros(len(bank['qpos']) + 1, bool), classes=bank['classes'])


def hold(env):
    """The action whose PD targets are the wide pose (pd_target: home + scale·a)."""
    return (wide_pose(env.home) - env.home) / env.scale


def probe_width(env):
    """getup_stance.py's width: left − right foot across the pelvis heading, from atan2 / sin / cos."""
    d = env.data
    rotation = d.xmat[env.pelvis].reshape(3, 3)
    h = math.atan2(rotation[1, 0], rotation[0, 0])
    dx, dy = d.xpos[env.feet[0], :2] - d.xpos[env.feet[1], :2]
    return -math.sin(h) * dx + math.cos(h) * dy


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['stance_width'] == 0
    assert (task.GETUP_STANCE_WIDTH, task.GETUP_STANCE_PELVIS) == (.30, .5)
    assert 'stance_width' not in G1_REWARD_WEIGHTS  # walking has no such weight
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['stance_width'] == 0  # old stages and branches: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': {'stance_width': 2}}})['task_config']
    assert weighted['reward_weights']['stance_width'] == 2. and weighted['reward_weights']['sole_impact'] == 0
    for value in (-.1, 101., float('nan'), '2', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'stance_width': value}}})
    for config in ({}, BRANCH_46):  # the contract (and so branching) does not change
        weights = {**config.get('reward_weights', {}), **WORKING}
        assert task_identity('g1_29dof', 'g1_getup', {**config, 'reward_weights': weights}) == task_identity(
            'g1_29dof', 'g1_getup', config)
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'stance_width')
    assert field['group'] == 'reward_weights' and field['default'] == 0. and '0,30 м' in field['label']


def test_the_width_and_the_term_in_numpy_and_torch():
    """W is the feet's offset along the pelvis heading's left (a yawed and pitched pelvis: the heading is its x axis's
    yaw); the term is −w·max(0, W − 0.30) where the pelvis is above 0.5 m and both soles below 1 cm (strictly), 0
    elsewhere and without W; the other terms do not depend on it; numpy and torch agree."""
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    rng = np.random.default_rng(3)
    n = 64
    yaw = rng.uniform(-math.pi, math.pi, n)
    pitch = rng.uniform(-.6, .6, n)
    rotation = np.einsum('nij,njk->nik', np.stack([[[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                                                    [0, 0, 1]] for a in yaw]),
                         np.stack([[[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]
                                   for b in pitch]))
    across, along = rng.uniform(-.2, .8, n), rng.uniform(-.3, .3, n)
    left = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
    forward = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    centre = rng.normal(size=(n, 3))
    feet = np.stack([centre, centre], 1)
    feet[:, 0, :2] += (across[:, None] * left + along[:, None] * forward) / 2
    feet[:, 1, :2] -= (across[:, None] * left + along[:, None] * forward) / 2
    feet[:, 1, 2] += rng.normal(0, .05, n)  # heights do not count
    width = task.stance_width(feet, rotation)
    np.testing.assert_allclose(width, across, rtol=0, atol=1e-12)
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        got = task.stance_width(torch.as_tensor(feet, dtype=dtype), torch.as_tensor(rotation, dtype=dtype))
        np.testing.assert_allclose(got.numpy(), width, rtol=0, atol=tolerance)
    height = np.array([.6, .5, .5 + 1e-9, .3, .79, .79, .79, .79])
    sole = np.array([[0., 0.], [0., 0.], [.0099, -.001], [0., 0.], [.01, 0.], [-.001, .02], [0., 0.], [0., 0.]])
    wide = np.array([.5, .5, .5, .5, .5, .5, .3, .2])
    np.testing.assert_array_equal(task.stance_active(height, sole), [True, False, True, False, False, False, True,
                                                                     True])
    inputs = batch(n=8)
    inputs.update(height=height, sole_height=sole)
    weights = {**G1_GETUP_REWARD_WEIGHTS, **WORKING}
    terms, _ = task.getup_reward_components(**inputs, weights=weights, stance_width=wide)
    np.testing.assert_allclose(terms['stance_width'], [-.4, 0., -.4, 0., 0., 0., 0., 0.], rtol=1e-12, atol=1e-15)
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['stance_width'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'stance_width')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, stance_width=torch.as_tensor(wide))
    np.testing.assert_allclose(torched['stance_width'].numpy(), terms['stance_width'], rtol=1e-12, atol=1e-15)
    single, _ = task.getup_reward_components(**as_torch(inputs, torch.float32), weights=weights,
                                             stance_width=torch.as_tensor(wide))  # float64 W, float32 terms
    assert single['stance_width'].dtype == torch.float32


def _rotation(yaw, pitch, roll):
    """Rz(yaw)·Ry(pitch)·Rx(roll)."""
    cz, sz, cy, sy, cx, sx = (f(a) for a in (yaw, pitch, roll) for f in (math.cos, math.sin))
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def test_the_heading_of_a_lying_pelvis_is_well_defined_in_float32():
    """The pelvis x axis within GETUP_STANCE_HEADING of vertical (pitched ±90°, ±85°, under any yaw and roll): the
    heading is its y axis's turned −90°, W the feet's offset along that axis's horizontal part, and float32 gives
    float64's W (the x axis's yaw from its near-zero horizontal part gave CUDA −0.18 against the CPU's +0.23 for the
    lying back drop). Lying on the back and the front (drop_bank's rows) likewise. From the threshold on (x axis
    tilted up to 84°) W is the x axis's, getup_stance.py's, unchanged."""
    rng = np.random.default_rng(8)
    rotations, feet, expected = [], [], []
    for pitch in (math.pi / 2, -math.pi / 2, math.radians(85), -math.radians(85), math.radians(84)):
        for _ in range(16):
            rotation = _rotation(rng.uniform(-math.pi, math.pi), pitch, rng.uniform(-math.pi, math.pi))
            horizontal_x = math.hypot(rotation[0, 0], rotation[1, 0])
            left = rotation[:2, 1] if horizontal_x < task.GETUP_STANCE_HEADING else np.array(
                [-rotation[1, 0], rotation[0, 0]])
            left = left / np.linalg.norm(left)
            across, along = rng.uniform(.1, .7), rng.uniform(-.2, .2)
            offset = across * left + along * np.array([left[1], -left[0]])
            centre = rng.normal(size=3)
            feet.append([centre + [*offset / 2, 0.], centre - [*offset / 2, 0.]])
            rotations.append(rotation)
            expected.append(across)
    feet, rotations = np.array(feet), np.array(rotations)
    width = task.stance_width(feet, rotations)
    np.testing.assert_allclose(width, expected, rtol=0, atol=1e-12)
    single = task.stance_width(torch.as_tensor(feet, dtype=torch.float32),
                               torch.as_tensor(rotations, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(single, width, rtol=0, atol=1e-5)
    env = G1GetupEnv(bank=stance_bank(), reward_weights=WORKING)
    try:
        for row in (0, 1, 2):
            env.reset(seed=0, options=dict(OPTIONS, start=row))
            d = env.data
            rotation = d.xmat[env.pelvis].reshape(1, 3, 3)
            assert math.hypot(rotation[0, 0, 0], rotation[0, 1, 0]) < task.GETUP_STANCE_HEADING, row
            position = d.xpos[env.feet][None]
            wanted = task.stance_width(position, rotation)
            got = task.stance_width(torch.as_tensor(position, dtype=torch.float32),
                                    torch.as_tensor(rotation, dtype=torch.float32)).numpy()
            np.testing.assert_allclose(got, wanted, rtol=0, atol=1e-5, err_msg=f'row {row}')
    finally:
        env.close()


def test_the_cpu_task_charges_a_wide_stance_not_the_home_one_nor_lying():
    """Standing in home (zero action): on the feet, W 0.24 m, nothing charged; the wide stance held by its PD targets:
    on the feet, W about 0.41, −2·(W − 0.30) each step; lying (the back drop): pelvis below 0.5 m, nothing. W is
    getup_stance.py's width and info carries it."""
    env = G1GetupEnv(bank=stance_bank(), reward_weights=WORKING)
    try:
        assert env.stance_weighted
        for row, action, low, high in ((STANDING, np.zeros(29), .22, .26), (WIDE, hold(env), .38, .45)):
            env.reset(seed=0, options=dict(OPTIONS, start=row))
            for _ in range(15):
                info = env.step(action)[-1]
                assert info['stance_active'] and low < info['stance_width'] < high, (row, info['stance_width'])
                assert info['stance_width'] == pytest.approx(probe_width(env), abs=1e-12)
                expected = -2. * max(0., info['stance_width'] - .30)
                assert info['reward_components']['stance_width'] == pytest.approx(expected, rel=1e-12, abs=1e-15)
                assert (expected < -.15) if row == WIDE else (expected == 0)
        env.reset(seed=0, options=dict(OPTIONS, start=0))
        for _ in range(30):
            info = env.step(np.zeros(29))[-1]
            assert not info['stance_active'] and info['reward_components']['stance_width'] == 0
    finally:
        env.close()


def test_the_cpu_task_changes_the_term_only():
    """The same flailing episodes (from the wide stance, standing and the standing drop) with and without the weight:
    every observation, qpos and other term the same, the term −2·[stance_active]·max(0, W − 0.30) of the step's own
    state, and the reward differs by it alone."""
    bank = stance_bank()
    weighted, plain = G1GetupEnv(bank=bank, reward_weights=WORKING), G1GetupEnv(bank=bank)
    rng = np.random.default_rng(6)
    charged = []
    try:
        assert not plain.stance_weighted
        for row in (WIDE, STANDING, STAND_DROP):
            obs = [env.reset(seed=3, options=dict(OPTIONS, start=row))[0] for env in (weighted, plain)]
            np.testing.assert_array_equal(*obs)
            for k in range(40):
                action = (hold(weighted) if row == WIDE else 0.) + rng.normal(0., 1. if k % 4 else 3., 29)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action)
                                                                              for env in (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                np.testing.assert_array_equal(weighted.data.qpos, plain.data.qpos)
                terms_w, terms_p = info_w['reward_components'], info_p['reward_components']
                assert terms_p['stance_width'] == 0 and 'stance_width' not in info_p
                assert {key: v for key, v in terms_w.items() if key != 'stance_width'} == {
                    key: v for key, v in terms_p.items() if key != 'stance_width'}
                assert reward_w == pytest.approx(reward_p + terms_w['stance_width'], rel=1e-12, abs=1e-12)
                d = weighted.data
                active = d.qpos[2] > .5 and bool((weighted._sole_height().numpy() < .01).all())
                assert info_w['stance_active'] == active
                assert terms_w['stance_width'] == pytest.approx(-2. * active * max(0., probe_width(weighted) - .3),
                                                                rel=1e-12, abs=1e-15)
                charged.append(terms_w['stance_width'])
        assert sum(c < 0 for c in charged) >= 10
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    """Weight 0 (missing or explicit): the width is never read, the term is 0, and every observation, reward, end, qpos
    and other term of training episodes is the code's before the weight: the plain task and pd_bound (OLD_DIGESTS,
    OLD_STANDING_DIGEST), impact 2 and branch 823's recipe (BEFORE_DIGESTS), branch 46's (BEFORE_46)."""
    def refuse(*args, **kwargs):
        raise AssertionError('stance_width computed with its weight at 0')
    for name in ('stance_width', 'stance_active'):
        monkeypatch.setattr(task, name, refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'stance_width': 0.}}, 'plain'),
                         ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    assert standing_digest({'reward_weights': {'stance_width': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    for config, (getup, standing) in ((IMPACT, BEFORE_DIGESTS['impact']), (BRANCH_823, BEFORE_DIGESTS['branch_823']),
                                      (BRANCH_46, BEFORE_46),
                                      ({**BRANCH_46, 'reward_weights': {**BRANCH_46['reward_weights'],
                                                                        'stance_width': 0.}}, BEFORE_46)):
        assert getup_digest(config, bank) == getup, config
        assert standing_digest(config, bank) == (standing, 59), config
    env = G1GetupEnv(bank=stance_bank())
    try:
        env.reset(seed=0, options=dict(OPTIONS, start=WIDE))
        for _ in range(10):
            info = env.step(hold(env))[-1]
            assert info['reward_components']['stance_width'] == 0 and 'stance_width' not in info
    finally:
        env.close()


def test_the_task_builds_from_a_saved_config_with_the_weight():
    """make_task (evaluation, playback, probes and the trainers build from the run's or stage's task_config): a config
    without the key reads no width, with it it does."""
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    for weights, on in (({}, False), (BRANCH_46['reward_weights'], False),
                        ({**BRANCH_46['reward_weights'], **WORKING}, True)):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup',
                                      task_config={**BRANCH_46, 'reward_weights': weights}))['task_config']
        env = make_task('g1_getup', 'g1_29dof', config)
        try:
            assert env.unwrapped.stance_weighted is on
        finally:
            env.close()
