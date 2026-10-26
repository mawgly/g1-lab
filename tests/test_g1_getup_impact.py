"""g1_getup reward weight impact (iteration 35), CPU: −w·Σ over the robot bodies touching the floor (any geom but the
sole spheres) of max(0, v − 1.5)², v the largest approach speed of the body's contact points over the control step's
physics substeps, read after each mj_step from the state it started with. Off (weight 0) nothing is computed and the
task is bit for bit the old one."""
import hashlib
import math

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv

DROP = .3  # m: the lowest robot geom but the sole spheres above the floor at the start of the drop rows
OPTIONS = dict(assist_force=0., action_delay_steps=0, motor_friction_scale=1., body_friction=.6)
SPIN = (-.5, 1., 3.)  # the spinning row: down at 0.5 m/s, rolling at 1 and pitching at 3 rad/s (body frame)


def _floor_geoms(env, soles=False):
    """The robot geoms whose contact filter passes with the floor's, without the sole spheres unless `soles`."""
    m = env.model
    floor = m.geom('floor').id
    sole = set(env.soles.ravel().tolist())
    return [g for g in range(m.ngeom) if env.robot_geoms[g] and (soles or g not in sole) and (
        m.geom_contype[g] & m.geom_conaffinity[floor] or m.geom_contype[floor] & m.geom_conaffinity[g])]


def drop_bank(height=DROP):
    """A start bank made without simulation (G1GetupEnv takes a loaded bank as a dict): the home pose on its back, on
    its front, on its front spinning (SPIN), each with the lowest robot geom but the soles `height` above the floor and
    the soles higher, then standing in home on the floor. No row held out."""
    env = bank_module.getup_model_env()
    m, d = env.model, env.data
    floor = m.geom('floor').id
    spin = np.zeros(m.nv)
    spin[2], spin[3:5] = SPIN[0], SPIN[1:]
    qpos, qvel = [], []
    for kind, velocity in (('back', np.zeros(m.nv)), ('front', np.zeros(m.nv)), ('front', spin)):
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 1.
        d.qpos[3:7] = bank_module.quat_axis(*bank_module.LYING[kind])
        d.qpos[env.qadr] = env.home
        mujoco.mj_forward(m, d)
        gap = min(mujoco.mj_geomDistance(m, d, floor, g, 2., None) for g in _floor_geoms(env))
        d.qpos[2] -= gap - height
        qpos.append(d.qpos.copy())
        qvel.append(velocity)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = .793
    d.qpos[3] = 1.
    d.qpos[env.qadr] = env.home
    qpos.append(d.qpos.copy())
    qvel.append(np.zeros(m.nv))
    return dict(qpos=np.array(qpos), qvel=np.array(qvel), start_class=np.array([0, 1, 1, 5], np.uint8),
                holdout=np.zeros(4, bool), classes=bank_module.CLASSES)


# Reward terms added with or after the impact weight, always among the components (0 without their weight): left out of
# getup_digest, which the code before them made.
NEWER_TERMS = ('impact', 'sole_impact', 'stance_width', 'arm_press', 'style', 'head_rise',
               'sole_slip')


def getup_digest(config, bank, seed=35, steps=240):
    """sha256 of every observation, reward, end flag, qpos and reward term but NEWER_TERMS of a few training episodes of
    the CPU task (80 steps each): training resets (random rows, delay, motor and body friction, the assist and pd_bound
    curricula) and random actions over the ±10 box."""
    env = G1GetupEnv(max_episode_steps=80, bank=bank, **config)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    try:
        obs, _ = env.reset(seed=seed)
        digest.update(obs.tobytes())
        for k in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-10., 10., 29) * (.3 if k % 7 else 1.))
            digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated])
                          + env.data.qpos.tobytes())
            for key in sorted(info['reward_components']):
                if key not in NEWER_TERMS:
                    digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + k + 1)
                digest.update(obs.tobytes())
    finally:
        env.close()
    return digest.hexdigest()


# getup_digest of the code before the impact weight (commit 4459fa3), with drop_bank(): the plain task and branch 34's
# pd_bound [1, 0.6]. Another MuJoCo, numpy or torch build changes them: then make them again with that commit's code.
OLD_DIGESTS = {'plain': '4d9fae680bf8458f84631664f4229e1d2e0f7d085cd5150777ed6bc0d5786e85',
               'pd_bound': '80bc359d561d015816539ae0284b9b7074cff01a10e5bcb0379e33d764bd2e1e'}


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['impact'] == 0 and task.GETUP_IMPACT_SPEED == 1.5
    assert 'impact' not in G1_REWARD_WEIGHTS  # walking has no such weight
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['impact'] == 0  # old stages and branches: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': {'impact': 2}}})['task_config']
    assert weighted['reward_weights']['impact'] == 2. and weighted['reward_weights']['head'] == 1.
    for value in (-.1, 101., float('nan'), '2', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'impact': value}}})
    for config in ({}, {'pd_bound': [1., .6]}):  # the contract (and so branching) does not change
        assert task_identity('g1_29dof', 'g1_getup', {**config, 'reward_weights': {'impact': 2.}}) == task_identity(
            'g1_29dof', 'g1_getup', config)
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'impact')
    assert field['group'] == 'reward_weights' and field['default'] == 0. and '1,5 м/с' in field['label']


def test_the_impact_functions_give_the_same_numbers_in_numpy_and_torch():
    rng = np.random.default_rng(0)
    k, ngeom, floor = 300, 12, 0
    geom = rng.integers(1, ngeom, (k, 2))
    geom[::3, 0] = floor
    geom[1::3, 1] = floor
    impact_geoms = rng.random(ngeom) < .7
    impact_geoms[floor] = False
    selected, other, sign = task.impact_contacts(geom, floor, impact_geoms)
    for i, (g1, g2) in enumerate(geom):
        robot = g2 if g1 == floor else g1
        assert selected[i] == (floor in (g1, g2) and impact_geoms[robot])
        assert (other[i], sign[i]) == (robot, 1 if g1 == floor else -1)
    assert 50 < selected.sum() < 200
    normal = rng.normal(size=(k, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    position, com, velocity = rng.normal(size=(k, 3)), rng.normal(size=(k, 3)), rng.normal(0, 2, (k, 6))
    speed = task.approach_speed(normal, sign, position, velocity, com)
    for i in range(k):
        point = velocity[i, 3:] + np.cross(velocity[i, :3], position[i] - com[i])
        assert speed[i] == pytest.approx(-sign[i] * normal[i] @ point, abs=1e-12)
    torched = task.impact_contacts(torch.as_tensor(geom), floor, torch.as_tensor(impact_geoms))
    for got, expected in zip(torched, (selected, other, sign)):
        np.testing.assert_array_equal(got.numpy(), expected)
    torched_speed = task.approach_speed(*(torch.as_tensor(v) for v in (normal, sign, position, velocity, com)))
    np.testing.assert_allclose(torched_speed.numpy(), speed, rtol=1e-12, atol=1e-12)
    bodies = np.array([[0., 1.5, 2.5, 3.], [1.4, -2., 0., 0.], [1.5 + 1e-9, 0., 0., 0.]])
    np.testing.assert_allclose(task.impact_penalty(bodies), [1. + 2.25, 0., 1e-18], rtol=1e-6, atol=0)
    assert task.impact_hit(bodies).tolist() == [True, False, True]
    np.testing.assert_allclose(task.impact_penalty(torch.as_tensor(bodies)).numpy(), task.impact_penalty(bodies))
    assert task.impact_hit(torch.as_tensor(bodies)).tolist() == [True, False, True]
    # The term: −w·penalty given one, 0 without; the other terms do not depend on it.
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    inputs, penalty = batch(n=3), task.impact_penalty(bodies)
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'impact': 2.}
    terms, _ = task.getup_reward_components(**inputs, weights=weights, impact=penalty)
    np.testing.assert_array_equal(terms['impact'], -2. * penalty)
    plain, _ = task.getup_reward_components(**inputs, weights=weights)
    assert not plain['impact'].any() and all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'impact')
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, impact=torch.as_tensor(penalty))
    np.testing.assert_allclose(torched['impact'].numpy(), -2. * penalty, rtol=1e-12)
    single, _ = task.getup_reward_components(**as_torch(inputs, torch.float32), weights=weights,
                                             impact=torch.as_tensor(penalty))  # a float64 penalty, float32 terms
    assert single['impact'].dtype == torch.float32


def test_the_approach_speed_is_mj_jacs_point_velocity_along_the_floor_normal():
    """Pressed 4 mm into the floor with random joint and base velocities: for every floor contact of a robot geom but
    the soles, approach_speed from the body's cvel is −n·(J_p q̇) (mj_jac at the contact point, the critic's
    impact35.py), n from the floor to the robot; the soles' contacts (standing) are never picked."""
    env = G1GetupEnv(bank=drop_bank())
    m, d = env.model, env.data
    floor = m.geom('floor').id
    soles = set(env.soles.ravel().tolist())
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    rng = np.random.default_rng(1)
    try:
        assert env.impact_substeps is None  # weight 0: no buffer, nothing read
        assert env.impact_geoms.sum() == env.robot_geoms.sum() - len(soles) and not env.impact_geoms[floor]
        for row in (0, 1, 2, 3):
            env.reset(seed=0, options=dict(OPTIONS, start=row))
            if row < 3:
                d.qpos[2] -= DROP + .004
            else:  # the soles 1 mm into the floor
                d.qpos[2] -= min(mujoco.mj_geomDistance(m, d, floor, g, 1., None) for g in soles) + .001
            d.qvel[:] = rng.normal(0, 1, m.nv)
            mujoco.mj_forward(m, d)
            contact = d.contact
            selected, robot, sign = task.impact_contacts(contact.geom, env.floor, env.impact_geoms)
            picked = 0
            for i in range(d.ncon):
                g1, g2 = (int(g) for g in contact.geom[i])
                other = g2 if g1 == floor else g1
                assert selected[i] == (floor in (g1, g2) and bool(env.robot_geoms[other]) and other not in soles)
                if not selected[i]:
                    continue
                body = m.geom_bodyid[other]
                mujoco.mj_jac(m, d, jacp, jacr, contact.pos[i], body)
                expected = -(1. if g1 == floor else -1.) * contact.frame[i, :3] @ (jacp @ d.qvel)
                got = task.approach_speed(contact.frame[i:i + 1, :3], sign[i:i + 1], contact.pos[i:i + 1],
                                          d.cvel[[body]], d.subtree_com[[m.body_rootid[body]]])[0]
                assert got == pytest.approx(expected, abs=1e-9)
                picked += 1
            if row < 3:
                assert picked >= 2
            else:  # standing in home: the soles touch, nothing else
                assert picked == 0 and any(floor in contact.geom[i] for i in range(d.ncon))
    finally:
        env.close()


@pytest.mark.parametrize('height', (DROP, .05))
def test_a_drop_hits_the_floor_at_the_free_fall_speed(height):
    """The home pose dropped flat (zero action holds it; no assist, delay or noise) with its lowest geom but the soles
    `height` up: its first floor contact closes at √(2gh) within 5 % on its back (the shoulders) and on its front (the
    thighs); the term is −w·Σ max(0, v − 1.5)² of the step's largest speeds (from 5 cm, 0.99 m/s: nothing). Standing on
    the soles charges nothing."""
    env = G1GetupEnv(bank=drop_bank(height), reward_weights={'impact': 2.})
    names = [env.model.body(b).name for b in range(env.model.nbody)]
    expected = math.sqrt(2 * 9.81 * height)
    try:
        for row in (0, 1):
            env.reset(seed=0, options=dict(OPTIONS, start=row))
            for _ in range(50):
                info = env.step(np.zeros(29))[-1]
                touched = np.flatnonzero(env.impact_substeps.max(1) > 0)
                if len(touched):
                    break
            else:
                pytest.fail('no floor contact')
            first = env.impact_substeps[touched[0]]
            bodies = np.flatnonzero(first)
            assert bodies.size and not any('ankle_roll' in names[b] for b in bodies)
            assert all(part in names[b] for b in bodies for part in (('shoulder',) if row == 0 else ('hip',)))
            np.testing.assert_allclose(first[bodies], expected, rtol=.05)
            charged = -2. * (np.clip(env.impact_substeps.max(0) - 1.5, 0, None) ** 2).sum()
            assert info['reward_components']['impact'] == pytest.approx(charged, rel=1e-12, abs=1e-12)
            assert (charged < -1.) if height == DROP else (charged == 0)
        env.reset(seed=0, options=dict(OPTIONS, start=3))
        for _ in range(25):
            assert env.step(np.zeros(29))[-1]['reward_components']['impact'] == 0
        assert not env.impact_substeps.any()
    finally:
        env.close()


def test_the_cpu_task_reads_each_substep_before_its_integration(monkeypatch):
    """With the weight on, flailing from the drop rows: each substep's per-body speeds are the critic's measure
    (impact35.py: mj_jac at the contact point after mj_step, times the velocities before it), the term charges the
    step's largest; the observations and every other term are those of the task without the weight (a reward weight
    only) and the reward differs by the term alone."""
    bank = drop_bank()
    weighted = G1GetupEnv(bank=bank, reward_weights={'impact': 2.})
    plain = G1GetupEnv(bank=bank)
    m = weighted.model
    floor = m.geom('floor').id
    soles = set(weighted.soles.ravel().tolist())
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    seen, original = [], mujoco.mj_step

    def step(model, data):
        before = data.qvel.copy()
        original(model, data)
        if model is not m:
            return
        speeds = np.zeros(model.nbody)
        for i in range(data.ncon):
            g1, g2 = (int(g) for g in data.contact.geom[i])
            other = g2 if g1 == floor else g1
            if floor not in (g1, g2) or not weighted.robot_geoms[other] or other in soles:
                continue
            body = model.geom_bodyid[other]
            mujoco.mj_jac(model, data, jacp, jacr, data.contact.pos[i], body)
            approach = (-1. if g1 == floor else 1.) * float((jacp @ before) @ data.contact.frame[i, :3])
            speeds[body] = max(speeds[body], approach)
        seen.append(speeds)
    monkeypatch.setattr(mujoco, 'mj_step', step)
    rng = np.random.default_rng(2)
    charged = []
    try:
        for row in (0, 1, 2):
            obs = [env.reset(seed=1, options=dict(OPTIONS, start=row))[0] for env in (weighted, plain)]
            np.testing.assert_array_equal(*obs)
            for k in range(40):
                action = rng.uniform(-10., 10., 29) * (.3 if k % 3 else 1.)
                (obs_w, reward_w, *_, info_w), (obs_p, reward_p, *_, info_p) = (env.step(action) for env in
                                                                              (weighted, plain))
                np.testing.assert_array_equal(obs_w, obs_p)
                terms_w, terms_p = info_w['reward_components'], info_p['reward_components']
                assert terms_p['impact'] == 0 and {key: v for key, v in terms_w.items() if key != 'impact'} == {
                    key: v for key, v in terms_p.items() if key != 'impact'}
                assert reward_w == pytest.approx(reward_p + terms_w['impact'], rel=1e-12, abs=1e-12)
                np.testing.assert_allclose(weighted.impact_substeps, np.array(seen[-10:]), rtol=0, atol=1e-9)
                largest = np.array(seen[-10:]).max(0)
                assert terms_w['impact'] == pytest.approx(-2. * (np.clip(largest - 1.5, 0, None) ** 2).sum(),
                                                          rel=1e-9, abs=1e-12)
                charged.append(terms_w['impact'])
        assert len(seen) == 3 * 40 * 10
        assert sum(c < 0 for c in charged) >= 3  # the drops at 2.4 m/s, then whatever the flailing hits
    finally:
        weighted.close(), plain.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    """Weight 0 (missing or explicit): the impact functions never run, the term is 0, and every observation, reward,
    end, qpos and other term of training episodes is the code's before the weight (OLD_DIGESTS), also with pd_bound."""
    def refuse(*args, **kwargs):
        raise AssertionError('impact computed with its weight at 0')
    for name in ('impact_contacts', 'approach_speed', 'impact_penalty', 'impact_hit'):
        monkeypatch.setattr(task, name, refuse)
    bank = drop_bank()
    for config, name in (({}, 'plain'), ({'reward_weights': {'impact': 0.}}, 'plain'),
                         ({'pd_bound': [1., .6]}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    env = G1GetupEnv(bank=bank)
    try:
        env.reset(seed=0, options=dict(OPTIONS, start=0))
        terms = [env.step(np.zeros(29))[-1]['reward_components']['impact'] for _ in range(20)]
        assert terms == [0.] * 20 and env.impact_substeps is None  # the drop hits at step 12: not charged
    finally:
        env.close()
