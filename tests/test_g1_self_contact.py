from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-facb20de4bfe472c/stage_004975362048'
# The candidate's real-robot profile (iteration 7), without its reward weights.
PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, action_limit=10.,
               heading_hold=True, stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55))
WEIGHTS = dict(velocity=2., yaw=2., yaw_drift=5., alive=.15, tilt=0., orientation=2., height=10., torso=1., still=3.,
               stand_pose=1., slip=1., feet_flat=2., posture=.3, hips=1., action_change=.05, fall=100.)


def arms_pressed():
    """Shoulders rolled in and turned in, both sides: the upper arms lean on the torso (shoulder_yaw_link against
    torso_link, 60-80 N per contact on the CPU task once settled) while the robot keeps standing."""
    action = np.zeros(29)
    action[[16, 17]] = -4., 10.  # left shoulder roll, yaw
    action[[23, 24]] = 4., -10.  # right: mirrored signs
    return action


def _record(env):
    """Wrap env._self_contact_force to keep what each substep read."""
    seen, original = [], env._self_contact_force

    def record():
        seen.append(original())
        return seen[-1]

    env._self_contact_force = record
    return seen


def test_self_contact_share_counts_substeps_above_the_threshold_and_the_term_is_linear():
    assert task.SELF_CONTACT_FORCE == 10.
    force = torch.tensor([[0.] * 10, [10.] * 10, [10.01] + [0.] * 9, [60.] * 7 + [3.] * 3], dtype=torch.float64)
    share = task.self_contact_share(force)
    torch.testing.assert_close(share, torch.tensor([0., 0., .1, .7], dtype=torch.float64))  # strictly above 10 N
    n = len(share)
    args = (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.zeros(n, 3, dtype=torch.float64), torch.zeros(n, 29),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.full((n, 2), .034, dtype=torch.float64),
            torch.full((n,), .25), torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    weights = {**task.G1_REWARD_WEIGHTS, 'self_contact': 4.}
    charged = task.reward_components(*args, weights=weights, self_contact=share)[0]
    torch.testing.assert_close(charged['self_contact'], -4 * share)
    plain = task.reward_components(*args, weights=weights)[0]
    assert plain['self_contact'].abs().max() == 0  # not given: zero
    assert all(torch.equal(plain[key], charged[key]) for key in plain if key != 'self_contact')
    assert task.G1_REWARD_WEIGHTS['self_contact'] == 0
    off = task.reward_components(*args, self_contact=share)[0]
    assert off['self_contact'].abs().max() == 0  # absent unless weighted


def test_self_contact_off_keeps_the_reward_bit_for_bit_and_reads_no_contacts(monkeypatch):
    plain = G1VelocityEnv(reward_weights=WEIGHTS, **PROFILE)
    weighted = G1VelocityEnv(reward_weights={**WEIGHTS, 'self_contact': 4.}, **PROFILE)
    given, original = [], task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    reads = _record(plain)
    try:
        for env in (plain, weighted):
            env.reset(seed=3, options={'command': [0., 0., 0.]})
        rng = np.random.default_rng(1)
        charged = []
        for _ in range(12):
            action = arms_pressed() + rng.normal(0, .2, 29)
            (obs, reward, *_, info), (obs_w, reward_w, *_, info_w) = (env.step(action) for env in (plain, weighted))
            assert given[-2]['self_contact'] is None and given[-1]['self_contact'] is not None
            np.testing.assert_array_equal(obs, obs_w)  # a reward weight only: same physics and observations
            components, components_w = info['reward_components'], info_w['reward_components']
            assert components['self_contact'] == 0
            assert reward == sum(value for key, value in components.items() if key != 'self_contact')
            assert {k: v for k, v in components.items() if k != 'self_contact'} == {
                k: v for k, v in components_w.items() if k != 'self_contact'}
            assert reward_w == pytest.approx(reward + components_w['self_contact'], rel=1e-12, abs=1e-12)
            charged.append(components_w['self_contact'])
        assert reads == []  # off: no contact forces are read
        assert charged[-1] == -4.  # every substep of a settled press
    finally:
        for env in (plain, weighted):
            env.close()


def test_cpu_self_contact_reads_each_substep_and_ignores_the_floor():
    env = G1VelocityEnv(reward_weights={**WEIGHTS, 'self_contact': 1.}, **PROFILE)
    try:
        floor = env.model.geom('floor').id
        colliding = (env.model.geom_contype != 0) | (env.model.geom_conaffinity != 0)
        assert not env.robot_geoms[floor] and env.robot_geoms[colliding].sum() == colliding.sum() - 1
        env.reset(seed=4, options={'command': [0., 0., 0.]})
        seen = _record(env)
        floor_force = np.zeros(6)
        for _ in range(10):  # the home pose: the feet press on the floor with far more than 10 N, the arms are free
            info = env.step(np.zeros(29))[-1]
            assert info['reward_components']['self_contact'] == 0
        pressing = []
        for k in range(env.data.ncon):
            if floor in env.data.contact.geom[k]:
                mujoco.mj_contactForce(env.model, env.data, k, floor_force)
                pressing.append(floor_force[0])
        assert max(pressing) > 10. and seen == [0.] * 100
        shares = []
        for _ in range(8):
            info = env.step(arms_pressed())[-1]
            shares.append(-info['reward_components']['self_contact'])
            # The term is the share of this step's substeps (read after each mj_step) above 10 N.
            assert shares[-1] == pytest.approx(np.mean(np.array(seen[-10:]) > task.SELF_CONTACT_FORCE), abs=1e-12)
        assert len(seen) == 180 and shares[0] == 0 and shares[-1] == 1 and min(seen[-30:]) > 30.
        assert float(env.data.qpos[2]) > .7  # still standing
    finally:
        env.close()


def _candidate():
    if not (CANDIDATE / 'metadata.json').exists():
        pytest.skip('candidate stage is not on this machine')
    from humanoid_sim.core import load_stage, read_json, stage_task_config
    return (*load_stage(CANDIDATE), stage_task_config(CANDIDATE), read_json(CANDIDATE / 'metadata.json'))


def test_the_candidate_leans_on_itself_and_keeps_its_contract_with_the_term():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.tasks.catalog import make_task
    model, norm, config, metadata = _candidate()
    assert 'self_contact' not in metadata['task_config']['reward_weights']
    weighted = {**config, 'reward_weights': {**config['reward_weights'], 'self_contact': 1.}}
    validate_resume(checkpoint_identity(metadata), task_identity('g1_29dof', 'g1_velocity', weighted))
    assert task_identity('g1_29dof', 'g1_velocity', weighted) == task_identity('g1_29dof', 'g1_velocity', config)
    env = make_task('g1_velocity', 'g1_29dof', weighted)
    try:
        obs, _ = env.reset(seed=73000, options={'command': [0., 0., 0.]})
        terms = []
        for _ in range(40):  # standing: the shoulders lean on the torso and the hands on the hips from the start
            obs, _, fell, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            assert not fell
            terms.append(info['reward_components']['self_contact'])
        assert np.mean(terms[10:]) < -.9
    finally:
        env.close()


def test_self_contact_weight_is_validated_and_labelled():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['reward_weights']['self_contact'] == 0  # old branches keep theirs
    accepted = validate_config({**base, 'task_config': {'reward_weights': {'self_contact': 4.}}})
    assert accepted['task_config']['reward_weights']['self_contact'] == 4
    for value in (-.1, 101., float('nan'), '4'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'self_contact': value}}})
    assert '10 Н' in G1_LABELS['self_contact']
    assert [p['default'] for p in G1_PARAMETERS if p['key'] == 'self_contact'] == [0.]
