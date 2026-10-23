from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

ROOT = Path(__file__).resolve().parents[1]
# task_config of exp-89d499d1c441484c (iteration 4, sole_contact), the branch iteration 5 would start from.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, action_limit=10.,
              heading_hold=True, stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
              commands=dict(min_speed=-.5, max_speed=1., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                            straight_probability=.5),
              reward_weights=dict(velocity=2., yaw=2., alive=.15, orientation=2., height=10., torso=1., still=3.,
                                  stand_pose=1., slip=1., feet_flat=.5, posture=.3, hips=1., action_change=.05,
                                  fall=100.))


def _args(command, foot_height):
    n = len(command)
    return (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.tensor(command, dtype=torch.float64),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29), torch.tensor(foot_height, dtype=torch.float64),
            torch.full((n,), .25), torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))


def test_clearance_charges_moving_feet_by_sole_height_times_speed():
    assert task.CLEARANCE_HEIGHT == .10
    walk, turn, stand = [1., 0., 0.], [0., 0., .5], [0., 0., 0.]
    # Both feet planted and still; left foot travelling 1 m/s with its sole 2 cm up, at the target, 20 cm up at
    # 0.5 m/s; the same 2 cm swing while turning in place and while told to stand.
    command = [walk, walk, walk, walk, turn, stand]
    sole = torch.tensor([[0., 0.], [.02, 0.], [.10, 0.], [.20, 0.], [.02, 0.], [.02, 0.]], dtype=torch.float64)
    speed = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [.5, 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
    args = _args(command, [[.034, .034]] * len(command))
    weights = {**task.G1_REWARD_WEIGHTS, 'clearance': 2.}
    term = task.reward_components(*args, weights=weights, foot_speed=speed, sole_height=sole)[0]['clearance']
    torch.testing.assert_close(term, torch.tensor([0., -2 * .08, 0., -2 * .1 * .5, -2 * .08, 0.], dtype=torch.float64))
    # A moving foot is charged on both feet, each by its own height and speed.
    both = task.reward_components(*_args([walk], [[.034, .034]]), weights=weights,
                                  foot_speed=torch.tensor([[1., .5]], dtype=torch.float64),
                                  sole_height=torch.tensor([[.02, .06]], dtype=torch.float64))[0]['clearance']
    torch.testing.assert_close(both, torch.tensor([-2 * (.08 + .04 * .5)], dtype=torch.float64))
    assert task.G1_REWARD_WEIGHTS['clearance'] == 0
    assert task.reward_components(*args, foot_speed=speed, sole_height=sole)[0]['clearance'].abs().max() == 0
    assert task.reward_components(*args, weights=weights, foot_speed=speed)[0]['clearance'].abs().max() == 0
    assert task.reward_components(*args, weights=weights, sole_height=sole)[0]['clearance'].abs().max() == 0


def test_sole_height_given_for_clearance_does_not_switch_the_contact_rule():
    # Ankle_roll origin 20 cm up (in the air by the old rule), lowest sole point on the floor (support by sole).
    args = _args([[1., 0., 0.]], [[.2, .2]])
    sole, speed = torch.zeros(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
    weights = {**task.G1_REWARD_WEIGHTS, 'slip': 1., 'clearance': 2.}

    def slip(**kwargs):
        return float(task.reward_components(*args, weights=weights, foot_speed=speed, sole_height=sole,
                                            **kwargs)[0]['slip'][0])

    assert slip(sole_contact=False) == 0  # contact by the ankle_roll origin: both feet in the air, no slip
    assert slip(sole_contact=True) == slip() == -2  # by the sole (also when unspecified, as before)


def _spy(monkeypatch):
    given, original = [], task.reward_components

    def spy(*args, **kwargs):
        given.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, 'reward_components', spy)
    return given


def test_cpu_clearance_off_by_default_keeps_the_reward_and_reads_the_simulator(monkeypatch):
    weights = {'gait': 2., 'slip': 1., 'swing': 10.}
    plain, weighted = G1VelocityEnv(reward_weights=weights), G1VelocityEnv(reward_weights={**weights, 'clearance': 2.})
    given = _spy(monkeypatch)
    try:
        for env in (plain, weighted):
            env.reset(seed=3, options={'command': [1., 0., 0.]})
        rng = np.random.default_rng(1)
        charged = []
        for _ in range(30):
            action = rng.normal(0, .5, 29)
            before = weighted.data.xpos[weighted.feet, :2].copy()
            (obs, reward, *_, info), (obs_w, reward_w, *_, info_w) = (env.step(action) for env in (plain, weighted))
            assert given[-2]['sole_height'] is None and given[-1]['sole_height'] is not None
            assert given[-2]['sole_contact'] is False and given[-1]['sole_contact'] is False
            np.testing.assert_array_equal(obs, obs_w)  # a reward weight only: same physics and observations
            components, components_w = info['reward_components'], info_w['reward_components']
            # Off by default: the added term is zero, so the sum is bit for bit the one without it.
            assert components['clearance'] == 0
            assert reward == sum(value for key, value in components.items() if key != 'clearance')
            # The weight changes nothing else (contact still by the ankle_roll origin) ...
            assert {k: v for k, v in components.items() if k != 'clearance'} == {
                k: v for k, v in components_w.items() if k != 'clearance'}
            # ... and the term is |sole - 0.10| times each foot's horizontal speed over the step, from MuJoCo.
            sole = weighted._sole_height()[0].numpy()
            speed = np.linalg.norm(weighted.data.xpos[weighted.feet, :2] - before, axis=1) / weighted.dt
            assert components_w['clearance'] == pytest.approx(-2 * np.sum(np.abs(sole - .1) * speed), rel=1e-12)
            assert reward_w == pytest.approx(reward + components_w['clearance'], rel=1e-12, abs=1e-12)
            charged.append(components_w['clearance'])
        assert min(charged) < -.01  # the feet do travel under random actions
    finally:
        for env in (plain, weighted):
            env.close()


def test_clearance_with_sole_contact_uses_one_sole_height(monkeypatch):
    env = G1VelocityEnv(reward_weights={'clearance': 2.}, sole_contact=True)
    given = _spy(monkeypatch)
    try:
        env.reset(seed=0, options={'command': [0., 0., 0.]})
        for _ in range(5):
            info = env.step(np.zeros(29))[-1]
            assert given[-1]['sole_contact'] is True and given[-1]['sole_height'] is not None
            assert info['reward_components']['clearance'] == 0  # told to stand: not charged
    finally:
        env.close()


def test_clearance_weight_is_validated_labelled_and_keeps_the_checkpoint_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['reward_weights']['clearance'] == 0  # old branches keep their reward
    accepted = validate_config({**base, 'task_config': {'reward_weights': {'clearance': 2.}}})
    assert accepted['task_config']['reward_weights']['clearance'] == 2
    for value in (-.1, 101., float('nan'), '2'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'clearance': value}}})
    assert 'feet_clearance' in G1_LABELS['clearance'] and 'подошва' in G1_LABELS['clearance']
    assert [p['default'] for p in G1_PARAMETERS if p['key'] == 'clearance'] == [0.]
    with_term = {**RECIPE, 'reward_weights': {**RECIPE['reward_weights'], 'clearance': 2., 'swing': 0.}}
    saved = task_identity('g1_29dof', 'g1_velocity', RECIPE)
    assert task_identity('g1_29dof', 'g1_velocity', with_term) == saved
    validate_resume(saved, task_identity('g1_29dof', 'g1_velocity', with_term))


def test_the_iteration_4_stage_can_be_branched_with_clearance():
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import list_stages, read_json
    run = ROOT / 'runs' / 'humanoid_lab' / 'exp-89d499d1c441484c'
    stages = list_stages(run) if run.is_dir() else []
    if not stages:
        pytest.skip('local run exp-89d499d1c441484c is not here')
    metadata = read_json(stages[-1] / 'metadata.json')
    assert 'clearance' not in metadata['task_config']['reward_weights']
    weights = {**metadata['task_config']['reward_weights'], 'clearance': 2.}
    requested = task_identity('g1_29dof', 'g1_velocity', {**metadata['task_config'], 'reward_weights': weights})
    validate_resume(checkpoint_identity(metadata), requested)
