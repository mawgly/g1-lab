"""task_config height_command / upper_body_slot (iteration 28): the pelvis height command, the reserved upper-body slot,
the crouch pose table, the commands' height draw, the weight transfer warm_start.insert_terms and the fixed
normalization of the new columns."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

ROOT = Path(__file__).resolve().parents[1]
# Candidate 8, the parent of iteration 28's root.
CANDIDATE = ROOT / 'runs/humanoid_lab/exp-28691193718e4d0e/stage_007221510144'
# Candidate 8's recipe, the draws of every training reset included (random phase, arm perturbation, force pushes,
# motor friction), short episodes so that a run has many training resets.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
              stand_heading_hold=False, side_heading_hold=False, sole_contact=True, random_phase=True,
              arm_perturb=.1, lock_wrists=True, knee_floor=.061, force_push=1.25, foot_collision=True,
              motor_friction=1.25, gait={'period': .6, 'stance_fraction': .55},
              commands={'min_speed': -1., 'max_speed': 2., 'stand_probability': .1, 'lateral_speed': .5,
                        'yaw_rate': 1., 'straight_probability': .5, 'side_share': .2, 'side_stop': 1.,
                        'side_speed': 1.},
              reward_weights={'height': 10., 'stand_pose': 1., 'still': 3., 'yaw': 2., 'yaw_drift': 5., 'alive': .15,
                              'orientation': 2., 'torso': 1., 'slip': 1., 'feet_flat': 2., 'self_contact': 1.,
                              'arm_press': .1, 'posture': .3, 'hips': 1., 'action_change': .05, 'fall': 100.})
CROUCH = dict(height_command=True, upper_body_slot=True)
CROUCH_COMMANDS = {'crouch_share': [.5, .33], 'crouch_min': [.64, .70], 'crouch_speed': 1.}


def crouch_recipe(**commands):
    return {**RECIPE, **CROUCH, 'commands': {**RECIPE['commands'], **commands},
            'reward_weights': {**RECIPE['reward_weights'], 'height_ceiling': 2.}}


def actions(count, nu=29, seed=0, size=.3):
    return np.random.default_rng(seed).uniform(-size, size, (count, nu))


def crouch_table_module():
    spec = importlib.util.spec_from_file_location('crouch_table', ROOT / 'scripts/lab/crouch_table.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TwoDraws(G1VelocityEnv):
    """The task without the setting, discarding two draws where the height command draws them: after every training
    reset's draws and after every command change's."""

    def _sample_command(self, resample=False):
        super()._sample_command(resample)
        if resample:
            self.np_random.random(1), self.np_random.random(1)

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        if (options or {}).get('command') is None:
            self.np_random.random(1), self.np_random.random(1)
        return result


def test_without_the_settings_nothing_new_runs_and_the_layout_and_reward_are_the_old_ones(monkeypatch):
    """Off: no height draw, no stand target, no extra terms (each would raise here) through 200 steps with training
    resets; 494 inputs, the old schema names, and the height, stand pose and fall terms are the old expressions of the
    state (BASE_HEIGHT, home, FALL_HEIGHT); height_ceiling is 0."""
    def forbidden(*args, **kwargs):
        raise AssertionError('called without the setting')

    for name in ('height_draw', 'stand_target', 'command_terms', 'height_ceiling', 'fall_height'):
        monkeypatch.setattr(task, name, forbidden)
    env = G1VelocityEnv(max_episode_steps=40, **RECIPE)
    try:
        assert env.observation_space.shape == (494,) and env.extra_terms == () and env.fixed_normalization is None
        env.reset(seed=3)
        resets = 0
        for action in actions(200, seed=1):
            _, _, terminated, truncated, info = env.step(action)
            parts = info['reward_components']
            height = float(env.data.qpos[2])
            assert parts['height'] == pytest.approx(-10. * (height - .78) ** 2, abs=1e-15)
            assert parts['height_ceiling'] == 0
            moving = bool(task.command_moving(torch.from_numpy(info['command'])[None])[0])  # the rewarded command
            pose = float(((env.data.qpos[env.qadr] - env.home)[:12] ** 2).sum())
            assert parts['stand_pose'] == pytest.approx(0. if moving else -pose, abs=1e-12)
            upright = env.data.xmat[env.pelvis][8]
            assert info['fell'] == (height < .55 or upright < .5) and 'height_command' not in info
            if terminated or truncated:
                resets += 1
                env.reset()
        assert resets >= 3
    finally:
        env.close()
    config = {key: value for key, value in RECIPE.items() if key not in ('gait',)}
    assert task_identity('g1_29dof', 'g1_velocity', config)['observation_schema'] == \
        'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-critic4'
    assert task_identity('g1_29dof', 'g1_velocity', {**config, 'height_command': False, 'upper_body_slot': False}) == \
        task_identity('g1_29dof', 'g1_velocity', config)


def test_on_at_the_nominal_height_the_old_columns_reward_and_draws_are_the_old_ones():
    """Both settings on, no crouch share (c = 0.78 always): 200 steps with training resets give, bit for bit, the old
    task's observation on the old columns (critic extras moved to the end), zeros on the 95 new ones, the old reward
    and falls; the random stream is the old one plus two draws after every training reset and command change."""
    from humanoid_sim.warm_start import term_map
    old, new = TwoDraws(max_episode_steps=40, **RECIPE), G1VelocityEnv(max_episode_steps=40, **RECIPE, **CROUCH)
    try:
        assert new.observation_space.shape == (589,) and new.actor_inputs == 585 and new.frame_size == 117
        mapping = term_map(old, new)
        known = (mapping >= 0).numpy()
        assert known.sum() == 494 and (mapping[-4:].numpy() == np.arange(490, 494)).all()
        observed = [(old.reset(seed=3)[0], new.reset(seed=3)[0])]
        resets = 0
        for action in actions(200, seed=1):
            a, b = old.step(action), new.step(action)
            observed.append((a[0], b[0]))
            assert a[1] == b[1] and a[2:4] == b[2:4] and new.height == task.BASE_HEIGHT
            assert b[4]['reward_components']['height_ceiling'] == 0 and b[4]['height_command'] == .78
            if a[2] or a[3]:
                resets += 1
                observed.append((old.reset()[0], new.reset()[0]))
            assert old.np_random.bit_generator.state == new.np_random.bit_generator.state
        for a, b in observed:
            np.testing.assert_array_equal(b[known], a[mapping[known].numpy()])
            assert not b[~known].any()
        assert resets >= 3
    finally:
        old.close(), new.close()


def test_the_crouch_table_is_the_generator_s_and_its_poses_meet_the_conditions():
    """scripts/lab/crouch_table.py solves the table again (to 1e-4 of the one in the code); every pose stands the
    pelvis within 3 mm of c, the feet flat within 0.5° and the centre of mass within 5 mm of the middle of the soles;
    the critic's points (IK(0.78) = −0.136, 0.378, −0.242) hold."""
    module = crouch_table_module()
    poses, kinematics = module.solve_table()
    try:
        rows = module.offsets(poses)
        assert [row[0] for row in rows] == [row[0] for row in task.CROUCH_TABLE] == list(module.HEIGHTS)
        np.testing.assert_allclose(np.array(rows), np.array(task.CROUCH_TABLE), atol=1e-4, rtol=0)
        for height in module.HEIGHTS:
            pelvis, tilt, com = kinematics.check(poses[height])
            assert abs(pelvis - height) <= .003 and tilt <= .5 and abs(com) <= .005
        critic = {.78: (-.136, .378, -.242), .74: (-.362, .813, -.451), .70: (-.504, 1.090, -.585),
                  .66: (-.619, 1.313, -.695), .64: (-.669, 1.413, -.744)}
        for height, angles in critic.items():
            np.testing.assert_allclose(poses[height], angles, atol=1.5e-3)
    finally:
        kinematics.close()


def test_stand_target_is_home_at_the_nominal_height_and_the_table_below():
    env = G1VelocityEnv(**RECIPE, **CROUCH)
    try:
        assert env.crouch_joints == ((0, 3, 4), (6, 9, 10))
        home = torch.from_numpy(env.home)
        heights = torch.tensor([.78, .70, .64, .60, .695, .80, .55], dtype=torch.float64)
        target = task.stand_target(home, heights, env.crouch_joints)
        assert torch.equal(target[0], home) and torch.equal(target[5], home)  # 0.78 and above: home exactly
        table = {row[0]: torch.tensor(row[1:], dtype=torch.float64) for row in task.CROUCH_TABLE}
        for k, height in ((1, .70), (2, .64), (3, .60), (6, .60)):  # clamped to the table below 0.60
            for leg in env.crouch_joints:
                torch.testing.assert_close(target[k, list(leg)], home[list(leg)] + table[height], rtol=0, atol=1e-12)
        torch.testing.assert_close(target[4, [0, 3, 4]], home[[0, 3, 4]] + (table[.69] + table[.70]) / 2, rtol=0,
                                   atol=1e-9)
        others = [j for j in range(29) if j not in (0, 3, 4, 6, 9, 10)]
        assert torch.equal(target[:, others], home[others].expand(7, -1))
        float32 = task.stand_target(home.float(), torch.tensor([.78, .7]), env.crouch_joints)
        assert torch.equal(float32[0], home.float())
    finally:
        env.close()


def test_fall_height_and_the_ceiling_follow_the_command_and_vanish_at_the_nominal_height():
    command = torch.tensor([.78, .78, .70, .70, .77, .64], dtype=torch.float64)
    height = torch.tensor([.80, .70, .72, .69, .80, .64], dtype=torch.float64)
    assert float(task.fall_height(torch.tensor([.78], dtype=torch.float64))[0]) == task.FALL_HEIGHT == .55
    assert float(task.fall_height(torch.tensor([.78]))[0]) == np.float32(.55)
    torch.testing.assert_close(task.fall_height(command), command - .23, rtol=0, atol=1e-12)
    ceiling = task.height_ceiling(height, command)
    expected = [0., 0., 1 - math.exp(-4 * .02), 0., .5 * (1 - math.exp(-4 * .03)), 0.]
    torch.testing.assert_close(ceiling, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-12)


def test_the_height_draw_shares_and_ranges():
    """1e5 commands of each kind: standing crouches half the time to U(0.64, 0.78), walking up to 1 m/s a third of the
    time to U(0.70, 0.78), faster walking never; the rest is exactly 0.78; always two draws of n."""
    n = 100000
    generator = torch.Generator().manual_seed(9)
    calls = []

    def uniform(k):
        calls.append(k)
        return torch.rand(k, generator=generator, dtype=torch.float64)

    stand = torch.zeros((n, 3), dtype=torch.float64)
    stand[:, 0] = .05  # below STAND_SPEED: a stand command for the reward
    walk = torch.tensor([[.5, .3, .4]], dtype=torch.float64).expand(n, -1)
    turn = torch.tensor([[0., 0., .5]], dtype=torch.float64).expand(n, -1)  # turning on the spot is walking
    fast = torch.tensor([[1.5, 0., 0.]], dtype=torch.float64).expand(n, -1)
    for command, share, lowest in ((stand, .5, .64), (walk, .33, .70), (turn, .33, .70), (fast, 0., None)):
        calls.clear()
        height = task.height_draw(uniform, command, **CROUCH_COMMANDS)
        assert calls == [n, n] and height.dtype == torch.float64
        crouched = height < task.BASE_HEIGHT
        assert bool((height[~crouched] == task.BASE_HEIGHT).all())
        assert float(crouched.double().mean()) == pytest.approx(share, abs=.006)
        if lowest is not None:
            values = height[crouched]
            assert float(values.min()) >= lowest and float(values.min()) < lowest + .001
            assert float(values.mean()) == pytest.approx((lowest + .78) / 2, abs=.002)
    calls.clear()
    off = task.height_draw(uniform, walk)  # the defaults: no crouch share, still two draws
    assert calls == [n, n] and bool((off == task.BASE_HEIGHT).all())
    edge = torch.tensor([[1., 0., 0.], [1.01, 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    always = task.height_draw(lambda k: torch.zeros(k, dtype=torch.float64), edge, crouch_share=[1., 1.],
                              crouch_min=[.64, .70])
    assert always.tolist() == [.70, .78, .70]  # ‖v_xy‖ ≤ crouch_speed crouches


def test_cpu_task_draws_c_at_training_resets_and_command_changes_and_options_and_setter_fix_it(monkeypatch):
    monkeypatch.setattr(task, 'COMMAND_RESAMPLE_SECONDS', .04)  # a command change every other step on average
    env = G1VelocityEnv(max_episode_steps=1000, **crouch_recipe(**CROUCH_COMMANDS))
    try:
        heights, kinds = [], []
        env.reset(seed=0)
        for step in range(600):
            before = env.command.copy()
            obs, _, terminated, truncated, _ = env.step(np.zeros(29))
            if terminated or truncated:
                obs, _ = env.reset()  # a training reset draws for its own command
            elif np.array_equal(before, env.command):
                continue
            heights.append(env.height)
            moving = bool(task.command_moving(torch.from_numpy(env.command.copy())[None])[0])
            fast = np.linalg.norm(env.command[:2]) > 1
            kinds.append('fast' if moving and fast else 'walk' if moving else 'stand')
            lowest = {'stand': .64, 'walk': .70, 'fast': .78}[kinds[-1]]
            assert lowest <= env.height <= .78
            assert obs[494] == pytest.approx(env.height - .78, abs=1e-7)  # the newest frame's height term
        heights, kinds = np.array(heights), np.array(kinds)
        assert len(heights) > 150 and (heights[kinds == 'fast'] == .78).all()
        assert (heights[kinds == 'stand'] < .78).any() and (heights[kinds == 'walk'] < .78).any()
        env.reset(seed=1, options={'command': [0., 0., 0.]})  # a reset with a command: 0.78, nothing drawn
        assert env.height == .78
        state = env.np_random.bit_generator.state
        env.reset(seed=1, options={'command': [0., 0., 0.], 'height': .66})
        assert env.height == .66 and env.np_random.bit_generator.state == state
        env.set_command([.5, 0., 0.])  # the command does not touch the height
        assert env.height == .66
        env.set_height(.7)
        obs = env._observation()  # the newest frame shows it, the older ones keep what they saw
        assert obs[494] == pytest.approx(-.08, abs=1e-7) and obs[490:494] == pytest.approx(-.12, abs=1e-7)
        assert not obs[495:585].any()
        for bad in (.59, .79, float('nan'), True, '0.7', None):
            with pytest.raises(ValueError):
                env.set_height(bad)
    finally:
        env.close()
    plain = G1VelocityEnv(**RECIPE)
    try:
        with pytest.raises(ValueError, match='height_command'):
            plain.set_height(.7)
        with pytest.raises(ValueError, match='height_command'):
            plain.reset(seed=1, options={'height': .7})
    finally:
        plain.close()
    with pytest.raises(ValueError, match='real_robot'):
        G1VelocityEnv(height_command=True)


def test_the_reward_and_stand_pose_follow_the_height_command():
    """At c = 0.70 standing, the robot still at home (pelvis ~0.79): the height term −10·(h − 0.70)², the ceiling
    −2·(1 − exp(−4·(h − 0.70))) (g = 1), the stand pose against home plus the table's offsets."""
    env = G1VelocityEnv(**crouch_recipe())
    try:
        env.reset(seed=2, options={'command': [0., 0., 0.], 'height': .70})
        _, _, terminated, _, info = env.step(np.zeros(29))
        parts, height = info['reward_components'], float(env.data.qpos[2])
        assert parts['height'] == pytest.approx(-10 * (height - .70) ** 2, abs=1e-12) and height > .75
        assert parts['height_ceiling'] == pytest.approx(-2 * (1 - math.exp(-4 * (height - .70))), abs=1e-12)
        assert parts['height_ceiling'] < -.5 and info['height_command'] == .70 and not terminated
        stand = task.stand_target(torch.from_numpy(env.home), torch.tensor([.70], dtype=torch.float64),
                                  env.crouch_joints)[0].numpy()
        pose = ((env.data.qpos[env.qadr] - stand)[:12] ** 2).sum()
        assert parts['stand_pose'] == pytest.approx(-pose, abs=1e-12) and pose > .5  # home is far from the crouch
    finally:
        env.close()


def test_the_episode_ends_below_the_fall_height_of_its_command():
    """The shared reward on synthetic states: pelvis at 0.50 is a fall at the nominal height (below 0.55) and not
    under a 0.70 command (fall height 0.47); 0.46 is a fall under both; tilted past FALL_UPRIGHT always."""
    n = 4
    eye = torch.eye(3, dtype=torch.float64).expand(n, -1, -1).clone()
    eye[3] = torch.tensor([[1., 0., 0.], [0., .3, -.954], [0., .954, .3]], dtype=torch.float64)  # 72° roll
    height = torch.tensor([.50, .50, .46, .70], dtype=torch.float64)
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float64)
    common = (zeros(n, 6), eye, height, zeros(n, 3), zeros(n, 29), zeros(n, 29), zeros(n, 29), zeros(n, 2) + .1,
              zeros(n), zeros(n, 29), zeros(29), eye)
    _, plain = task.reward_components(*common)
    command = torch.tensor([.78, .70, .70, .70], dtype=torch.float64)
    parts, crouched = task.reward_components(*common, height_command=command)
    assert plain.tolist() == [True, True, True, True]
    assert crouched.tolist() == [True, False, True, True]
    assert parts['height_ceiling'].tolist() == [0., 0., 0., 0.]  # weight 0 by default


def test_config_is_validated_labelled_real_robot_only_and_its_own_observation_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_COMMAND_LABELS, G1_LABELS, G1_PARAMETERS, g1_config
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    real = {'real_robot': True, 'action_limit': 10., 'history': 5, 'phase_stop': True, 'critic_extras': True}
    config = validate_config(base)['task_config']
    assert (config['height_command'], config['upper_body_slot']) == (False, False)
    assert config['commands']['crouch_share'] == [0., 0.] and config['commands']['crouch_min'] == [.64, .70]
    assert config['commands']['crouch_speed'] == 1. and config['reward_weights']['height_ceiling'] == 0.
    checked = validate_config({**base, 'task_config': {**real, **CROUCH, 'commands': CROUCH_COMMANDS,
                                                       'reward_weights': {'height_ceiling': 2.}}})['task_config']
    assert checked['commands']['crouch_share'] == [.5, .33] and checked['reward_weights']['height_ceiling'] == 2.
    assert g1_config(checked) == checked  # spelled out, stable through a second pass (saved configs)
    for bad in ({'height_command': True}, {'upper_body_slot': True}, {**real, 'height_command': 1},
                {**real, 'commands': {'crouch_share': [.5, .33]}},  # a crouch needs the height command
                {**real, **CROUCH, 'commands': {'crouch_share': .5}},
                {**real, **CROUCH, 'commands': {'crouch_share': [.5]}},
                {**real, **CROUCH, 'commands': {'crouch_share': [1.5, .3]}},
                {**real, **CROUCH, 'commands': {'crouch_min': [.55, .7]}},
                {**real, **CROUCH, 'commands': {'crouch_min': [.64, .80]}},
                {**real, **CROUCH, 'commands': {'crouch_min': [.64, float('nan')]}},
                {**real, **CROUCH, 'commands': {'crouch_speed': 3.}}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': bad})
    keys = {p['key']: p for p in G1_PARAMETERS}
    assert keys['height_command']['default'] is False and keys['upper_body_slot']['type'] == 'bool'
    assert 'присед' in keys['height_command']['label'] and 'верха тела' in keys['upper_body_slot']['label']
    assert keys['crouch_share']['default'] == [0., 0.] and 'приседом' in G1_COMMAND_LABELS['crouch_share']
    assert keys['height_ceiling']['default'] == 0. and 'приседа' in G1_LABELS['height_ceiling']
    saved = task_identity('g1_29dof', 'g1_velocity', real)
    both = task_identity('g1_29dof', 'g1_velocity', {**real, **CROUCH, 'commands': CROUCH_COMMANDS})
    height = task_identity('g1_29dof', 'g1_velocity', {**real, 'height_command': True})
    assert both['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hcmd-ubslot-critic4'
    assert height['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hcmd-critic4'
    assert {k: v for k, v in both.items() if k != 'observation_schema'} == \
        {k: v for k, v in saved.items() if k != 'observation_schema'}
    with pytest.raises(ValueError, match='observation_schema'):
        validate_resume(saved, both)


def test_fixed_normalization_columns_and_the_tensor_normalizer_keeps_them():
    from humanoid_sim.tensor_ppo import TensorNormalize
    env = G1VelocityEnv(**RECIPE, **CROUCH)
    try:
        positions, mean, variance = env.fixed_normalization
        assert len(positions) == 5 * 19 and (positions >= 490).all() and (positions < 585).all()
        assert sorted(positions.tolist()) == list(range(490, 585))
        height = task.term_positions(29, 5, env.extra_terms, task.HEIGHT_TERM)
        assert height == [490, 491, 492, 493, 494] and (variance[:5] == .0025).all() and (variance[5:] == 1).all()
        assert not mean.any()
    finally:
        env.close()
    fixed = (positions, mean, variance)
    norm = TensorNormalize((589,), 4, 'cpu', fixed=fixed)
    assert float(norm.obs.var[490]) == .0025 and float(norm.obs.var[500]) == 1
    values = torch.randn(64, 589, dtype=torch.float64) * 3 + 1
    normalized = norm.observation(values)
    assert torch.equal(norm.obs.mean[positions], torch.zeros(95, dtype=torch.float64))
    assert torch.equal(norm.obs.var[positions], torch.from_numpy(variance))
    assert float(norm.obs.var[:490].min()) > 1  # the rest is estimated
    torch.testing.assert_close(normalized[:, 490], (values[:, 490] / math.sqrt(.0025 + 1e-8)).clamp(-10, 10).float())

    class Stats:  # a VecNormalize's statistics with other numbers on the fixed columns
        def __init__(self):
            self.obs_rms = type('RMS', (), dict(mean=np.full(589, 7.), var=np.full(589, 9.), count=7e9))()
            self.ret_rms = type('RMS', (), dict(mean=0., var=1., count=1.))()
            self.gamma = .99

    norm.import_normalizer(Stats())
    assert float(norm.obs.count) == 7e9 and float(norm.obs.mean[0]) == 7.
    assert torch.equal(norm.obs.var[positions], torch.from_numpy(variance)) and not norm.obs.mean[positions].any()
    plain = TensorNormalize((589,), 4, 'cpu')
    plain.observation(values)
    assert float(plain.obs.var[490]) > 1


def test_export_of_a_height_stage_has_the_two_terms_and_checks_crouched_observations(tmp_path):
    """policy.onnx takes [1, 585]; deploy.yaml lists height_command (params for g1_ctrl-height.patch) and
    upper_body_command after the 7 terms with history 5; check() compares onnxruntime with the stage on samples half
    of which come under a crouch command, and the height term moves the action (the untrained actor reads it)."""
    onnxruntime = pytest.importorskip('onnxruntime')
    yaml = pytest.importorskip('yaml')
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import load_stage, save_stage, write_json
    from humanoid_sim.export_g1 import TERMS, check, export, onnx_properties
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    task_config = {'real_robot': True, 'critic_extras': True, 'history': 5, 'phase_stop': True, 'action_limit': 10.,
                   'heading_hold': True, 'lock_wrists': True, 'knee_floor': .061, **CROUCH,
                   'commands': {'min_speed': -.5, 'max_speed': 1., 'lateral_speed': .5, 'yaw_rate': 1.,
                                **CROUCH_COMMANDS}}
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=task_config))
    run = tmp_path / 'run'
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 585})
        positions, mean, variance = env.envs[0].unwrapped.fixed_normalization
        env.obs_rms.mean[positions], env.obs_rms.var[positions] = mean, variance
        with torch.no_grad():
            model.policy.action_net.weight.mul_(10)
        stage = save_stage(run, model, env)
    finally:
        env.close()
    result = export(stage, tmp_path / 'out')
    assert result['actor_inputs'] == 585 and any('g1_ctrl-height.patch' in note for note in result['warnings'])
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 585]
    properties = onnx_properties(result['onnx'])
    assert properties['height_command'] == properties['upper_body_command'] == 'true'
    assert properties['observation_schema'].endswith('-phasestop-hcmd-ubslot-critic4')
    deploy = yaml.safe_load(result['deploy'].read_text(encoding='utf-8'))
    terms = deploy['observations']
    assert tuple(terms) == (*TERMS, 'height_command', 'upper_body_command')
    assert [len(term['scale']) for term in terms.values()] == [3, 3, 3, 2, 29, 29, 29, 1, 18]
    assert all(term['history_length'] == 5 for term in terms.values())
    assert terms['height_command']['params'] == {'nominal': .78, 'lowest': .64, 'walking_lowest': .70,
                                                 'walking_norm': .1}
    assert terms['upper_body_command']['params'] == {}
    assert 'g1_ctrl-height.patch' in result['deploy'].read_text(encoding='utf-8').splitlines()[1]
    verdict = check(stage, result['onnx'], samples=20, stride=2)
    assert verdict['ok'] and verdict['crouched'] == 10
    policy, norm = load_stage(stage)
    raw = norm.venv.envs[0].unwrapped
    try:
        obs, _ = raw.reset(seed=1, options={'command': [0., 0., 0.]})
        raw.set_height(.64)
        crouched = raw._observation()
        run = lambda o: session.run(['actions'], {'obs': o[None, :585].astype(np.float32)})[0][0]
        assert np.abs(run(crouched) - run(obs)).max() > 1e-3
    finally:
        norm.close()


def test_sim2sim_stick_gives_the_height_term_the_environment_observes(monkeypatch):
    """g1_ctrl-height.patch's rule as analyze.py replays it (c = 0.78 + 0.14·min(0, ry), at least 0.70 while the
    command's norm is ≥ 0.1, the export's height_params) and gamepad.py's stick for a height: the term is the CPU
    task's c − 0.78; the crouch scenario's segments stand at 0.70 and 0.64 and walk at 0.70."""
    from argparse import Namespace
    from humanoid_sim.export_g1 import height_params
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/sim2sim'))
    analyze, gamepad = importlib.import_module('analyze'), importlib.import_module('gamepad')
    params = height_params({'commands': CROUCH_COMMANDS})
    replay = type('Replay', (), dict(height=params))()
    term = lambda command, ry: float(analyze.Replay.height_term(replay, np.float32(command), ry)[0])
    for height in (.78, .74, .70, .66, .64):
        ry = gamepad.stick_for(height)
        assert gamepad.height_of(ry) == pytest.approx(height, abs=1e-9)
        assert term([0., 0., 0.], ry) == pytest.approx(height - .78, abs=1e-6)
        assert term([.5, 0., 0.], ry) == pytest.approx(max(height, .70) - .78, abs=1e-6)
    assert term([0., 0., 0.], 1.) == 0 and term([.05, 0., .05], -1.) == pytest.approx(-.14, abs=1e-6)
    env = G1VelocityEnv(**crouch_recipe())
    try:
        env.reset(seed=0, options={'command': [0., 0., 0.], 'height': .66})
        assert env._observation()[494] == pytest.approx(term([0., 0., 0.], gamepad.stick_for(.66)), abs=1e-6)
    finally:
        env.close()
    args = Namespace(scenario='crouch', stand_seconds=5., walk=.5, yaw=.5, side=.3, after_seconds=5.,
                     passive_walking=False, push_walking=False, ry=0.)
    plan = gamepad.segments(args)
    assert [(name, vx, round(gamepad.height_of(rest[0]), 2) if rest else None)
            for name, vx, _, _, _, *rest in plan] == [
        ('stand', 0., None), ('crouch', 0., .70), ('crouch', 0., .64), ('rise', 0., .78), ('walk', .5, .70),
        ('walk', 1., .70), ('stop', 0., .78)]


@pytest.mark.skipif(not (CANDIDATE / 'metadata.json').exists(), reason='candidate 8 is not on this machine')
def test_insert_terms_moves_candidate_8_to_the_height_contract_acting_and_valuing_as_before(tmp_path):
    """The real stage: the new root's stage 0 passes validate_resume for the new contract, keeps steps and updates,
    acts as candidate 8 to 1e-6 and values to 1e-5 on the same states (also under a crouch command: the new columns
    start at zero weights), with the old columns' statistics and the fixed ones on the new columns; Adam: m 0 and v
    the row mean of the old columns on the new ones."""
    from stable_baselines3 import PPO
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import load_stage, read_json, stage_task_config, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.warm_start import insert_terms
    parent = read_json(CANDIDATE.parents[0] / 'run_config.json')
    task_config = {**parent['task_config'], **CROUCH, 'commands': {**parent['task_config']['commands'],
                                                                 **CROUCH_COMMANDS},
                   'reward_weights': {**parent['task_config']['reward_weights'], 'height_ceiling': 2.}}
    config = validate_config({key: parent[key] for key in (
        'backend', 'envs', 'seed', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
        'ent_coef', 'clip_range', 'target_kl', 'robot_id', 'task_id', 'bounds_coef', 'max_std', 'mirror_coef',
        'range_bounds')} | {'task_config': task_config})
    run = tmp_path / 'root'
    write_json(run / 'run_config.json', config)
    stage = insert_terms(CANDIDATE, run)
    metadata = read_json(stage / 'metadata.json')
    validate_resume(checkpoint_identity(metadata), task_identity('g1_29dof', 'g1_velocity', task_config))
    assert metadata['steps'] == 7221510144 and stage.name == CANDIDATE.name
    info = read_json(stage / 'warm_start.json')
    assert info['inserted'] == ['height_command', 'upper_body_command'] and info['observation_size'] == [494, 589]
    old_model, old_norm = load_stage(CANDIDATE)
    new_model, new_norm = load_stage(stage)
    old_env, new_env = old_norm.venv.envs[0].unwrapped, new_norm.venv.envs[0].unwrapped
    try:
        assert new_model._n_updates == old_model._n_updates and new_model.num_timesteps == old_model.num_timesteps
        assert new_model.policy.actor_inputs == 585
        old_rms, new_rms = old_norm.obs_rms, new_norm.obs_rms
        assert new_rms.count == old_rms.count > 7e9
        np.testing.assert_array_equal(new_rms.mean[:490], old_rms.mean[:490])
        np.testing.assert_array_equal(new_rms.var[585:], old_rms.var[490:])
        assert not new_rms.mean[490:585].any() and (new_rms.var[490:495] == .0025).all()
        assert (new_rms.var[495:585] == 1).all()
        old_layer = old_model.policy.state_dict()['mlp_extractor.policy_net.0.weight']
        new_layer = new_model.policy.state_dict()['mlp_extractor.policy_net.0.weight']
        assert torch.equal(new_layer[:, :490], old_layer[:, :490]) and torch.equal(new_layer[:, 585:], old_layer[:, 490:])
        assert not new_layer[:, 490:585].any()
        names = [name for name, _ in new_model.policy.named_parameters()]
        number = names.index('mlp_extractor.value_net.0.weight')
        new_adam = new_model.policy.optimizer.state_dict()['state'][number]
        old_adam = old_model.policy.optimizer.state_dict()['state'][number]
        assert not new_adam['exp_avg'][:, 490:585].any()
        torch.testing.assert_close(new_adam['exp_avg_sq'][:, 490:585],
                                   old_adam['exp_avg_sq'].mean(dim=1, keepdim=True).expand(-1, 95))
        assert torch.equal(new_adam['exp_avg_sq'][:, 585:], old_adam['exp_avg_sq'][:, 490:])

        def value(model, norm, obs):
            with torch.no_grad():
                return float(model.policy.predict_values(torch.as_tensor(norm.normalize_obs(obs)[None]))[0, 0])

        worst_action = worst_value = 0.
        for number, command in enumerate(([0., 0., 0.], [1., 0., 0.], [.5, 0., .5])):
            a, _ = old_env.reset(seed=81000 + number, options={'command': command})
            b, _ = new_env.reset(seed=81000 + number, options={'command': command})
            for step in range(40):
                if step % 4 == 0:
                    act = old_model.predict(old_norm.normalize_obs(a), deterministic=True)[0]
                    for obs in (b, None):
                        if obs is None:  # the same state told to crouch
                            new_env.set_height(.66)
                            obs = new_env._observation()
                            assert obs[494] == pytest.approx(-.12, abs=1e-6)
                        got = new_model.predict(new_norm.normalize_obs(obs), deterministic=True)[0]
                        worst_action = max(worst_action, float(np.abs(got - act).max()))
                        worst_value = max(worst_value, abs(value(new_model, new_norm, obs)
                                                           - value(old_model, old_norm, a)))
                    new_env.height = .78
                    b = new_env._observation()
                action = old_model.predict(old_norm.normalize_obs(a), deterministic=True)[0]
                a, b = old_env.step(action)[0], new_env.step(action)[0]
        assert worst_action <= 1e-6 and worst_value <= 1e-5
        assert stage_task_config(stage)['height_command'] is True
    finally:
        old_norm.close(), new_norm.close()
    PPO.load(stage / 'policy.zip', device='cpu')  # an ordinary SB3 archive
