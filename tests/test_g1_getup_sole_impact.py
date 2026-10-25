"""g1_getup reward weight sole_impact (iteration 44), CPU: impact's charge on the sole spheres, the contacts impact
leaves out, −w·Σ over the two feet of max(0, v − 0.5)², v the largest approach speed of a foot's sole sphere floor
contacts over the control step's physics substeps (task.impact_contacts / approach_speed / impact_penalty with
GETUP_SOLE_IMPACT_SPEED), read after each mj_step from the state it started with. Off (weight 0) nothing is computed and
the task is bit for bit the old one, also with impact weighted. The CUDA task's reduction of both terms in one pass
(TensorG1GetupEnv._floor_speeds) is checked here on the CPU with torch against the CPU task's speeds; its physics in
tests/test_g1_gpu.py."""
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv, floor_entries
from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import DROP, OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_pose_std import OLD_STANDING_DIGEST, standing_digest

WORKING = {'sole_impact': 2.}  # the lead's working weight (iteration 44)
SOLE_DROP = .05  # m: the soles' lowest point above the floor in row STAND_DROP (√(2gh) = 0.99 m/s)
SOFT_DROP = .01  # m: in row SOFT (0.44 m/s, below GETUP_SOLE_IMPACT_SPEED)
ON_FLOOR = -.0005  # m: in row STANDING, the soles half a millimetre into the floor
STAND_DROP, SOFT, STANDING = 4, 5, 6  # sole_bank's rows: from SOLE_DROP, from SOFT_DROP, on the floor


def sole_bank():
    """drop_bank's rows (0-2 the home pose on its back, front and front spinning 0.3 m up; 3 upright in home, its soles
    8.8 mm up in this model), then the home pose upright with the soles' lowest point SOLE_DROP (4), SOFT_DROP (5) and
    ON_FLOOR (6) above the floor. No row held out."""
    bank = drop_bank()
    env = bank_module.getup_model_env()
    m, d = env.model, env.data
    floor = m.geom('floor').id
    qpos = []
    for height in (SOLE_DROP, SOFT_DROP, ON_FLOOR):
        mujoco.mj_resetData(m, d)
        d.qpos[2], d.qpos[3] = 1., 1.
        d.qpos[env.qadr] = env.home
        mujoco.mj_forward(m, d)
        d.qpos[2] -= min(mujoco.mj_geomDistance(m, d, floor, g, 2., None) for g in env.soles.ravel()) - height
        qpos.append(d.qpos.copy())
    return dict(qpos=np.concatenate([bank['qpos'], qpos]), qvel=np.concatenate([bank['qvel'], np.zeros((3, m.nv))]),
                start_class=np.concatenate([bank['start_class'], [5, 5, 5]]).astype(np.uint8),
                holdout=np.zeros(7, bool), classes=bank['classes'])


# getup_digest and standing_digest of the code before the weight (commit 5ef277a), with drop_bank(), impact weighted:
# impact 2 alone and branch 823's recipe (exp-f19f12fb1eff4ef3: impact 2, pd_bound [1, 0.6], stand_pose_std 0.3,
# action_change_unbounded); OLD_DIGESTS and OLD_STANDING_DIGEST cover the task without impact. Another MuJoCo, numpy or
# torch build changes them: then make them again with that commit's code.
IMPACT = {'reward_weights': {'impact': 2.}}
BRANCH_823 = {'reward_weights': {'impact': 2.}, 'pd_bound': [1., .6], 'stand_pose_std': .3,
              'action_change_unbounded': True}
BEFORE_DIGESTS = {
    'impact': ('e2e0c48068975e3607e021c0f49489f7faf351e69696404b4b8f097426141b3c',
               '8da2822564417f585c653e973294280b59801a4a6ce0dd6783cf88b7f92caea3'),
    'branch_823': ('7a7a92d3898f4d0a0d4c3198d488a4fbf60ee68dd933fd27921ea5905a85f330',
                   '011dc1b44168e63ff7e99fae9bd325ec4c3e2c745d71fd8fd0a2f354e30b2dff')}


def test_the_weight_is_a_reward_weight_only_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['sole_impact'] == 0 and task.GETUP_SOLE_IMPACT_SPEED == .5
    assert task.GETUP_IMPACT_SPEED == 1.5  # impact keeps its own
    assert 'sole_impact' not in G1_REWARD_WEIGHTS  # walking has no such weight
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['reward_weights']['sole_impact'] == 0  # old stages and branches: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': {'sole_impact': 2}}})['task_config']
    assert weighted['reward_weights']['sole_impact'] == 2. and weighted['reward_weights']['impact'] == 0
    for value in (-.1, 101., float('nan'), '2', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'sole_impact': value}}})
    for config in ({}, {'pd_bound': [1., .6]}, BRANCH_823):  # the contract (and so branching) does not change
        weights = {**config.get('reward_weights', {}), **WORKING}
        assert task_identity('g1_29dof', 'g1_getup', {**config, 'reward_weights': weights}) == task_identity(
            'g1_29dof', 'g1_getup', config)
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'sole_impact')
    assert field['group'] == 'reward_weights' and field['default'] == 0. and '0,5 м/с' in field['label']


def test_the_threshold_and_the_term_in_numpy_and_torch():
    """impact_penalty and impact_hit take the threshold (impact's 1.5 by default, unchanged); the sole_impact term is
    −w·penalty given one, 0 without, and the other terms do not depend on it."""
    from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
    from test_g1_getup_math import as_torch, batch
    feet = np.array([[0., .5], [.99, .3], [.5 + 1e-9, 2.5], [0., 0.]])
    expected = [0., .49 ** 2, 1e-18 + 4.]
    penalty = task.impact_penalty(feet, task.GETUP_SOLE_IMPACT_SPEED)
    np.testing.assert_allclose(penalty, expected + [0.], rtol=1e-6, atol=0)
    assert task.impact_hit(feet, task.GETUP_SOLE_IMPACT_SPEED).tolist() == [False, True, True, False]
    np.testing.assert_array_equal(task.impact_penalty(feet), [0., 0., 1., 0.])  # the default threshold, 1.5
    assert task.impact_hit(feet).tolist() == [False, False, True, False]
    np.testing.assert_allclose(task.impact_penalty(torch.as_tensor(feet), task.GETUP_SOLE_IMPACT_SPEED).numpy(),
                               penalty, rtol=1e-12)
    assert task.impact_hit(torch.as_tensor(feet), task.GETUP_SOLE_IMPACT_SPEED).tolist() == [False, True, True, False]
    inputs = batch(n=4)
    weights = {**G1_GETUP_REWARD_WEIGHTS, 'impact': 2., **WORKING}
    impact = np.array([.3, 0., 1., 2.])
    terms, _ = task.getup_reward_components(**inputs, weights=weights, impact=impact, sole_impact=penalty)
    np.testing.assert_array_equal(terms['sole_impact'], -2. * penalty)
    np.testing.assert_array_equal(terms['impact'], -2. * impact)
    plain, _ = task.getup_reward_components(**inputs, weights=weights, impact=impact)
    assert not plain['sole_impact'].any()
    assert all(np.array_equal(plain[key], terms[key]) for key in plain if key != 'sole_impact')
    old, _ = task.getup_reward_components(**inputs)  # no weight: the term is there, 0
    assert 'sole_impact' in old and not old['sole_impact'].any()
    torched, _ = task.getup_reward_components(**as_torch(inputs), weights=weights, impact=torch.as_tensor(impact),
                                              sole_impact=torch.as_tensor(penalty))
    np.testing.assert_allclose(torched['sole_impact'].numpy(), -2. * penalty, rtol=1e-12)
    single, _ = task.getup_reward_components(**as_torch(inputs, torch.float32), weights=weights,
                                             sole_impact=torch.as_tensor(penalty))  # a float64 penalty, float32 terms
    assert single['sole_impact'].dtype == torch.float32


@pytest.mark.parametrize('row, height', ((STAND_DROP, SOLE_DROP), (SOFT, SOFT_DROP)))
def test_a_standing_drop_lands_on_both_soles_at_the_free_fall_speed(row, height):
    """Upright in home with the soles `height` up (zero action holds the pose; no assist, delay or noise): each foot's
    first sole contact closes at √(2gh) within 5 % (both feet in the same control step), the term charges the step's
    largest per foot above 0.5 m/s (from 5 cm, 0.99 m/s: about −0.96 at weight 2; from 1 cm, 0.44 m/s: nothing), and
    impact, weighted too, charges nothing (no body but the feet touches). Standing on the floor charges nothing."""
    env = G1GetupEnv(bank=sole_bank(), reward_weights={'impact': 2., **WORKING})
    expected = math.sqrt(2 * 9.81 * height)
    try:
        assert env.sole_impact_substeps.shape == (10, 2) and env.sole_geoms.sum() == env.soles.size
        env.reset(seed=0, options=dict(OPTIONS, start=row))
        for _ in range(50):
            info = env.step(np.zeros(29))[-1]
            speeds = env.sole_impact_substeps.copy()
            if speeds.any():
                break
            assert info['reward_components']['sole_impact'] == 0
        else:
            pytest.fail('no sole contact')
        first = [int(np.flatnonzero(speeds[:, foot])[0]) for foot in range(2)]
        assert all(speeds[:, foot].any() for foot in range(2)), speeds  # both feet in this control step
        np.testing.assert_allclose([speeds[first[foot], foot] for foot in range(2)], expected, rtol=.05)
        charged = -2. * (np.clip(speeds.max(0) - .5, 0, None) ** 2).sum()
        assert info['reward_components']['sole_impact'] == pytest.approx(charged, rel=1e-12, abs=1e-12)
        assert (charged < -.8) if height == SOLE_DROP else (charged == 0)
        assert not env.impact_substeps.any() and info['reward_components']['impact'] == 0
        env.reset(seed=0, options=dict(OPTIONS, start=STANDING))
        for _ in range(25):
            assert env.step(np.zeros(29))[-1]['reward_components']['sole_impact'] == 0
        assert env.sole_impact_substeps.max() < .01  # both soles down, at rest
    finally:
        env.close()


def test_the_cpu_task_reads_each_substep_before_its_integration(monkeypatch):
    """Both weights on, flailing from the standing drop, from standing and from a back drop: each substep's per-foot
    speeds are the critic's measure (mj_jac at each sole contact point after mj_step, times the velocities before it),
    the term charges the step's largest per foot; impact's speeds are those of the task with impact alone, the
    observations and every other term those of the task without sole_impact (a reward weight only), and the reward
    differs by the term alone."""
    bank = sole_bank()
    both = G1GetupEnv(bank=bank, reward_weights={'impact': 2., **WORKING})
    impact = G1GetupEnv(bank=bank, reward_weights={'impact': 2.})
    m = both.model
    floor = m.geom('floor').id
    foot_of = {int(g): foot for foot in range(2) for g in both.soles[foot]}
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    seen, original = [], mujoco.mj_step

    def step(model, data):
        before = data.qvel.copy()
        original(model, data)
        if model is not m:
            return
        speeds = np.zeros(2)
        for i in range(data.ncon):
            g1, g2 = (int(g) for g in data.contact.geom[i])
            other = g2 if g1 == floor else g1
            if floor not in (g1, g2) or other not in foot_of:
                continue
            mujoco.mj_jac(model, data, jacp, jacr, data.contact.pos[i], model.geom_bodyid[other])
            approach = (-1. if g1 == floor else 1.) * float((jacp @ before) @ data.contact.frame[i, :3])
            speeds[foot_of[other]] = max(speeds[foot_of[other]], approach)
        seen.append(speeds)
    monkeypatch.setattr(mujoco, 'mj_step', step)
    rng = np.random.default_rng(4)
    charged = []
    try:
        for row in (STAND_DROP, STANDING, 0):
            obs = [env.reset(seed=1, options=dict(OPTIONS, start=row))[0] for env in (both, impact)]
            np.testing.assert_array_equal(*obs)
            for k in range(40):
                action = rng.uniform(-10., 10., 29) * (.3 if k % 3 else 1.)
                (obs_b, reward_b, *_, info_b), (obs_i, reward_i, *_, info_i) = (env.step(action) for env in
                                                                              (both, impact))
                np.testing.assert_array_equal(obs_b, obs_i)
                np.testing.assert_array_equal(both.impact_substeps, impact.impact_substeps)
                terms_b, terms_i = info_b['reward_components'], info_i['reward_components']
                assert terms_i['sole_impact'] == 0 and impact.sole_impact_substeps is None
                assert {key: v for key, v in terms_b.items() if key != 'sole_impact'} == {
                    key: v for key, v in terms_i.items() if key != 'sole_impact'}
                assert reward_b == pytest.approx(reward_i + terms_b['sole_impact'], rel=1e-12, abs=1e-12)
                np.testing.assert_allclose(both.sole_impact_substeps, np.array(seen[-10:]), rtol=0, atol=1e-9)
                largest = np.array(seen[-10:]).max(0)
                assert terms_b['sole_impact'] == pytest.approx(-2. * (np.clip(largest - .5, 0, None) ** 2).sum(),
                                                               rel=1e-9, abs=1e-12)
                charged.append(terms_b['sole_impact'])
        assert len(seen) == 3 * 40 * 10
        assert sum(c < 0 for c in charged) >= 3  # the drop at 0.99 m/s, then whatever the flailing stamps
    finally:
        both.close(), impact.close()


def test_off_nothing_is_computed_and_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    """Weight 0 (missing or explicit): the soles' speeds are never read, the term is 0, and every observation, reward,
    end, qpos and other term of training episodes is the code's before the weight: without impact (OLD_DIGESTS,
    OLD_STANDING_DIGEST; the impact functions never run), with impact 2 and with branch 823's recipe
    (BEFORE_DIGESTS)."""
    def refuse(*args, **kwargs):
        raise AssertionError('impact computed with its weight at 0')
    original = G1GetupEnv._impact_speeds

    def impact_only(self, geoms=None):
        assert geoms is None, 'sole_impact computed with its weight at 0'
        return original(self)
    monkeypatch.setattr(G1GetupEnv, '_impact_speeds', impact_only)
    bank = drop_bank()
    with monkeypatch.context() as patch:
        for name in ('impact_contacts', 'approach_speed', 'impact_penalty', 'impact_hit'):
            patch.setattr(task, name, refuse)
        for config, name in (({}, 'plain'), ({'reward_weights': {'sole_impact': 0.}}, 'plain'),
                             ({'pd_bound': [1., .6]}, 'pd_bound')):
            assert getup_digest(config, bank) == OLD_DIGESTS[name], config
        assert standing_digest({'reward_weights': {'sole_impact': 0.}}, bank) == (OLD_STANDING_DIGEST, 59)
    for config, name in ((IMPACT, 'impact'), (BRANCH_823, 'branch_823'),
                         ({**BRANCH_823, 'reward_weights': {'impact': 2., 'sole_impact': 0.}}, 'branch_823')):
        assert getup_digest(config, bank) == BEFORE_DIGESTS[name][0], config
        assert standing_digest(config, bank) == (BEFORE_DIGESTS[name][1], 59), config
    env = G1GetupEnv(bank=sole_bank(), reward_weights={'impact': 2.})
    try:
        env.reset(seed=0, options=dict(OPTIONS, start=STAND_DROP))
        terms = [env.step(np.zeros(29))[-1]['reward_components']['sole_impact'] for _ in range(20)]
        assert terms == [0.] * 20 and env.sole_impact_substeps is None  # the feet land at step ~6: not charged
    finally:
        env.close()


def test_the_weight_changes_the_rewards_only():
    """With the weight the same training episodes (getup_digest's) keep every observation, end and qpos; the digest,
    which has the rewards, differs."""
    bank = sole_bank()
    assert getup_digest({'reward_weights': WORKING}, bank) != getup_digest({}, bank)
    runs = []
    for weights in (WORKING, {}):
        env = G1GetupEnv(max_episode_steps=60, bank=bank, reward_weights=weights)
        rng = np.random.default_rng(9)
        record = []
        try:
            obs, _ = env.reset(seed=9, options=dict(OPTIONS, start=STAND_DROP))
            for _ in range(60):
                obs, reward, terminated, truncated, info = env.step(rng.normal(0., 2., 29))
                term = info['reward_components']['sole_impact']
                record.append((obs, env.data.qpos.copy(), terminated, truncated, term))
        finally:
            env.close()
        runs.append(record)
    for (obs_w, qpos_w, *ends_w, term), (obs_p, qpos_p, *ends_p, zero) in zip(*runs):
        np.testing.assert_array_equal(obs_w, obs_p)
        np.testing.assert_array_equal(qpos_w, qpos_p)
        assert ends_w == ends_p and zero == 0 and term <= 0
    assert any(term < 0 for *_, term in runs[0])


def _fake_cuda(envs, slots_extra=7, seed=0):
    """A stand-in for TensorG1GetupEnv with the CPU tasks `envs` (one world each, both weights on) as its worlds: their
    data.contact in shuffled slots of one array as Warp keeps them (worldid per slot, float32, int32 geoms), a live
    flex-like slot (geom −1) and stale slots past nacon with random contents; cvel and subtree_com per world."""
    rng = np.random.default_rng(seed)
    ref = envs[0]
    m = ref.model
    rows = [(w, env.data.contact.geom[i].copy(), env.data.contact.pos[i].copy(), env.data.contact.frame[i].copy())
            for w, env in enumerate(envs) for i in range(env.data.ncon)]
    rows.append((0, np.array([-1, ref.floor]), rng.normal(size=3), np.eye(3).ravel()))
    order = rng.permutation(len(rows))
    rows = [rows[k] for k in order]
    nacon = len(rows)
    for _ in range(slots_extra):  # stale: floor and sole pairs of any world, never live
        rows.append((int(rng.integers(len(envs))), np.array([ref.floor, int(rng.choice(ref.soles.ravel()))]),
                     rng.normal(size=3), np.eye(3).ravel()))
    n, naconmax = len(envs), len(rows)
    entry = floor_entries(ref, True, True)
    width = m.nbody + len(ref.soles)
    f32 = torch.float32
    return SimpleNamespace(
        contact_geom=torch.tensor(np.array([r[1] for r in rows]), dtype=torch.int32),
        contact_world=torch.tensor([r[0] for r in rows], dtype=torch.int32),
        contact_pos=torch.tensor(np.array([r[2] for r in rows]), dtype=f32),
        contact_frame=torch.tensor(np.array([r[3] for r in rows]), dtype=f32).reshape(naconmax, 3, 3),
        nacon=torch.tensor([nacon], dtype=torch.int32), _contact_slots=torch.arange(naconmax),
        _contact_ones=torch.ones(naconmax, dtype=torch.long), floor=int(ref.floor),
        floor_entry=torch.as_tensor(entry), floor_geoms=torch.as_tensor(entry >= 0),
        body_of_geom=torch.as_tensor(m.geom_bodyid, dtype=torch.long),
        root_of_body=torch.as_tensor(m.body_rootid, dtype=torch.long), nbody=int(m.nbody), floor_width=width,
        cvel=torch.tensor(np.stack([env.data.cvel for env in envs]), dtype=f32),
        subtree_com=torch.tensor(np.stack([env.data.subtree_com for env in envs]), dtype=f32), n_envs=n,
        _floor_flat=torch.zeros(n * width + naconmax), _floor_spare=n * width + torch.arange(naconmax),
        sole_impact_substeps=torch.zeros((n, 10, 2)), touching=torch.zeros(n, dtype=torch.bool),
        device=torch.device('cpu'))


def test_the_cuda_reduction_on_the_cpu_gives_the_cpu_tasks_speeds():
    """TensorG1GetupEnv._floor_speeds (both terms in one scatter, entries by floor_entries) run with torch on the CPU
    over four worlds' contacts as Warp keeps them (shuffled across worlds, a flex-like slot, stale slots): per world its
    bodies' speeds are the CPU task's impact speeds and its feet's the CPU task's sole speeds (float32), touching marks
    the worlds with a floor contact; floor_entries maps impact's geoms to their bodies and the soles to nbody + foot."""
    bank = sole_bank()
    envs = [G1GetupEnv(bank=bank, reward_weights={'impact': 2., **WORKING}) for _ in range(4)]
    rng = np.random.default_rng(5)
    try:
        ref = envs[0]
        entry = floor_entries(ref, True, True)
        nbody = ref.model.nbody
        assert (entry[ref.impact_geoms] == ref.model.geom_bodyid[ref.impact_geoms]).all()
        assert entry[ref.soles[0]].tolist() == [nbody] * ref.soles.shape[1]
        assert entry[ref.soles[1]].tolist() == [nbody + 1] * ref.soles.shape[1]
        assert (entry[~(ref.impact_geoms | ref.sole_geoms)] == -1).all()
        assert (floor_entries(ref, True, False)[ref.sole_geoms] == -1).all()
        assert (floor_entries(ref, False, True)[ref.impact_geoms] == -1).all()
        for env, row in zip(envs, (0, 1, STANDING, STAND_DROP)):
            env.reset(seed=0, options=dict(OPTIONS, start=row))
            m, d = env.model, env.data
            if row in (0, 1):  # lying: pressed 4 mm into the floor
                d.qpos[2] -= DROP + .004
            elif row == STANDING:  # the soles 1 mm into the floor; the last world stays 5 cm up
                gap = min(mujoco.mj_geomDistance(m, d, env.floor, g, 1., None) for g in env.soles.ravel())
                d.qpos[2] -= gap + .001
            env.data.qvel[:] = rng.normal(0, .3, env.model.nv)
            env.data.qvel[2] = -1.  # every world moving down, so its floor contacts close
            mujoco.mj_forward(env.model, env.data)
        fake = _fake_cuda(envs)
        speeds = TensorG1GetupEnv._floor_speeds(fake).numpy()
        assert speeds.shape == (4, nbody + 2)
        for world, env in enumerate(envs):
            np.testing.assert_allclose(speeds[world, :nbody], env._impact_speeds(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(speeds[world, nbody:], env._impact_speeds(env.sole_geoms)[env.feet], rtol=1e-5,
                                       atol=1e-5)
        assert fake.touching.tolist() == [True, True, True, False]
        assert (speeds[:2, :nbody] > 0).sum(1).min() >= 2 and not speeds[:2, nbody:].any()  # lying: bodies, no soles
        assert (speeds[2, nbody:] > .5).all() and not speeds[2, :nbody].any()  # standing: both soles, nothing else
        assert not speeds[3].any()
    finally:
        for env in envs:
            env.close()


def test_the_task_builds_from_a_saved_config_with_the_weight():
    """make_task (evaluation, playback, probes and the trainers build from the run's or stage's task_config): a config
    without the key reads no sole contacts, with it it does."""
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.catalog import make_task
    for weights, on in (({}, False), ({'impact': 2.}, False), (WORKING, True)):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup',
                                      task_config={**BRANCH_823, 'reward_weights': weights}))['task_config']
        env = make_task('g1_getup', 'g1_29dof', config)
        try:
            assert (env.unwrapped.sole_impact_substeps is not None) is on
        finally:
            env.close()
