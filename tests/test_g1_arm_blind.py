"""task_config arm_blind (iteration 31): the arm columns' fixed normalization (the policy does not see its arms), its
freezing on both training paths, the branch warm_start.blind_arms (scripts/lab/blind_branch.py) and the export."""
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

ROOT = Path(__file__).resolve().parents[1]
# Stage 7741 of branch 29 (exp-04f74e3838184b82), the critic's blind candidate (lab-critic/it30 blind_const.py).
STAGE_7741 = ROOT / 'runs/humanoid_lab/exp-04f74e3838184b82/stage_007741440000'
# The critic's mixref z of 7741 (bc_7741_mix.json, reset seed 73000): first and last three columns, the sum.
CRITIC_Z_7741 = ((1.2025871276855469, 1.2337775230407715, -0.016044311225414276),
                 (0.09500929713249207, -0.5376829504966736, -0.058606911450624466), -13.14248082099948)
# Candidate 8's recipe with the draws of every training reset (random phase, arm perturbation, force pushes, motor
# friction), short episodes so that a run has many training resets.
RECIPE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
              stand_heading_hold=False, side_heading_hold=False, sole_contact=True, random_phase=True,
              arm_perturb=.25, arm_perturb_radians=1., arm_perturb_contact_mask=True, lock_wrists=True,
              knee_floor=.061, force_push=1.25, foot_collision=True, motor_friction=1.25,
              gait={'period': .6, 'stance_fraction': .55},
              commands={'min_speed': -1., 'max_speed': 2., 'stand_probability': .1, 'lateral_speed': .5,
                        'yaw_rate': 1., 'straight_probability': .5},
              reward_weights={'height': 10., 'stand_pose': 1., 'still': 3., 'yaw': 2., 'yaw_drift': 5., 'alive': .15,
                              'orientation': 2., 'torso': 1., 'slip': 1., 'feet_flat': 2., 'self_contact': 1.,
                              'arm_press': .1, 'posture': .3, 'hips': 1., 'action_change': .05, 'fall': 100.})
CROUCH = dict(height_command=True, upper_body_slot=True)
ARMS = tuple(range(15, 29))  # shoulders, elbows and wrists of the 29-DoF G1
SIMPLE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True)


def actions(count, nu=29, seed=0, size=.3):
    return np.random.default_rng(seed).uniform(-size, size, (count, nu))


def blinded(rms, columns, z):
    """A copy of SB3 statistics `rms` with the arm columns at mean −z·ARM_BLIND_SCALE, variance ARM_BLIND_SCALE²."""
    copy = rms.copy()
    copy.mean[columns], copy.var[columns] = -np.asarray(z) * task.ARM_BLIND_SCALE, task.ARM_BLIND_SCALE ** 2
    return copy


def test_without_the_setting_the_contract_and_the_fixed_normalization_are_the_old_ones():
    from humanoid_sim.tasks.config import G1_PARAMETERS, g1_config
    keys = {p['key']: p for p in G1_PARAMETERS}
    assert keys['arm_blind']['default'] is False and keys['arm_blind']['type'] == 'bool'
    assert 'Слепые руки' in keys['arm_blind']['label']
    assert g1_config({'real_robot': True})['arm_blind'] is False
    for values in ({'arm_blind': True}, {'real_robot': True, 'arm_blind': 1}):  # real robot only; a bool
        with pytest.raises(ValueError):
            g1_config(values)
    with pytest.raises(ValueError):
        G1VelocityEnv(arm_blind=True)
    plain = task_identity('g1_29dof', 'g1_velocity', RECIPE)
    assert task_identity('g1_29dof', 'g1_velocity', {**RECIPE, 'arm_blind': False}) == plain
    assert plain['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-critic4'
    blind = task_identity('g1_29dof', 'g1_velocity', {**RECIPE, 'arm_blind': True})
    crouch = task_identity('g1_29dof', 'g1_velocity', {**RECIPE, **CROUCH, 'arm_blind': True})
    assert blind['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-armblind-critic4'
    assert crouch['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hcmd-ubslot-armblind-critic4'
    assert {k: v for k, v in blind.items() if k != 'observation_schema'} == \
        {k: v for k, v in plain.items() if k != 'observation_schema'}
    with pytest.raises(ValueError, match='observation_schema'):
        validate_resume(plain, blind)
    env, terms = G1VelocityEnv(**RECIPE), G1VelocityEnv(**RECIPE, **CROUCH)
    try:
        assert env.fixed_normalization is None and env.arm_blind is False
        positions, mean, variance = terms.fixed_normalization
        assert sorted(positions.tolist()) == list(range(490, 585)) and not np.isnan(mean).any() and not mean.any()
    finally:
        env.close(), terms.close()


def test_on_the_observation_reward_and_draws_are_those_of_the_task_without_it():
    """The blindness is the normalization's: 200 steps with training resets give, bit for bit, the old observation,
    reward, components, falls and random stream."""
    old = G1VelocityEnv(max_episode_steps=40, **RECIPE)
    new = G1VelocityEnv(max_episode_steps=40, **RECIPE, arm_blind=True)
    try:
        np.testing.assert_array_equal(old.reset(seed=3)[0], new.reset(seed=3)[0])
        resets = 0
        for action in actions(200, seed=1):
            a, b = old.step(action), new.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4] and a[4]['reward_components'] == b[4]['reward_components']
            if a[2] or a[3]:
                resets += 1
                np.testing.assert_array_equal(old.reset()[0], new.reset()[0])
        assert resets >= 3
        assert old.np_random.bit_generator.state == new.np_random.bit_generator.state
    finally:
        old.close(), new.close()


def test_arm_columns_are_the_arm_entries_of_the_three_joint_terms_in_every_frame():
    """210 columns on the height layout: joint_pos_rel, joint_vel_rel and last_action of joints 15-28 in each of the 5
    frames, nothing else; fixed_normalization lists the 95 term columns as before, then these at variance 1e12 with
    NaN means."""
    env = G1VelocityEnv(**RECIPE, **CROUCH, arm_blind=True)
    try:
        names = [env.model.joint(int(j)).name for j in env.joints]
        assert env.arm_joints == ARMS and all(any(p in names[i] for p in task.ARM_JOINT_PARTS) for i in ARMS)
        columns = env.arm_columns
        assert len(columns) == len(set(columns)) == 14 * 3 * 5
        size, history = env.frame_size, env.history
        code = torch.arange(history * size, dtype=torch.float64).reshape(1, history, size)  # frame·size + entry
        observed = task.history_observation(code, 29, env.extra_sizes)[0].numpy()
        spans = task.deploy_terms(29, env.extra_sizes)
        wanted = {frame * size + spans[term][0] + joint for term in (4, 5, 6) for frame in range(history)
                  for joint in ARMS}
        assert {int(observed[c]) for c in columns} == wanted
        positions, mean, variance = env.fixed_normalization
        assert positions[:95].tolist() == G1VelocityEnv(**RECIPE, **CROUCH).fixed_normalization[0].tolist()
        assert positions[95:].tolist() == columns and np.isnan(mean[95:]).all() and not np.isnan(mean[:95]).any()
        assert (variance[95:] == 1e12).all()
    finally:
        env.close()


def test_blind_statistics_hold_the_arm_inputs_at_z_whatever_the_arms_do():
    """Arms held in random poses of ±1.2 rad (set_arm_perturb), moving between them: the normalized arm inputs stay z
    within 3e-5 while the raw ones move by radians, and every other column is the plain statistics' bit for bit."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**RECIPE, arm_blind=True)]), training=False)
    raw = env.envs[0].unwrapped
    try:
        rng = np.random.default_rng(2)
        rms = RunningMeanStd(shape=(494,))
        rms.mean, rms.var, rms.count = rng.normal(0, .3, 494), rng.uniform(.01, 2, 494), 8e9
        columns = np.asarray(raw.arm_columns)
        z = rng.uniform(-1.5, 1.5, len(columns))
        plain, blind = rms.copy(), blinded(rms, columns, z)
        obs, _ = raw.reset(seed=4, options={'command': [0., 0., 0.]})
        seen = []
        for step in range(240):
            if step % 40 == 0:
                held = np.zeros(29)
                held[list(ARMS)] = rng.uniform(-1.2, 1.2, 14) / raw.scale[list(ARMS)]
                raw.set_arm_perturb(40, held)
            seen.append(obs)
            obs, _, terminated, _, _ = raw.step(np.zeros(29))
            if terminated:
                obs, _ = raw.reset(seed=step)
        seen = np.array(seen)
        assert seen[:, columns].std(axis=0).max() > .3  # the arms did move
        env.obs_rms = blind
        got = env.normalize_obs(seen)
        env.obs_rms = plain
        before = env.normalize_obs(seen)
        assert np.abs(got[:, columns] - z.astype(np.float32)).max() < 3e-5
        others = np.setdiff1d(np.arange(494), columns)
        np.testing.assert_array_equal(got[:, others], before[:, others])
    finally:
        env.close()


def test_the_frozen_columns_hold_through_updates_on_both_training_paths_and_the_two_agree():
    """TensorNormalize (full GPU) and train.FrozenMoments (the CPU path's VecNormalize) with blind statistics on the
    height layout: through 20 updates with arm inputs of any size the term columns stay at their set values and the
    arm columns at −z·1e6 / 1e12 exactly, the rest is estimated alike on both, and both normalize alike; FrozenMoments
    pickles as SB3's plain RunningMeanStd."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from humanoid_sim.tensor_ppo import TensorNormalize
    from humanoid_sim.train import FrozenMoments
    env = G1VelocityEnv(**RECIPE, **CROUCH, arm_blind=True)
    try:
        fixed, columns = env.fixed_normalization, np.asarray(env.arm_columns)
    finally:
        env.close()
    rng = np.random.default_rng(5)
    rms = RunningMeanStd(shape=(589,))
    rms.mean, rms.var, rms.count = rng.normal(0, .3, 589), rng.uniform(.01, 2, 589), 8e9
    z = rng.uniform(-1.5, 1.5, len(columns))
    rms = blinded(rms, columns, z)
    tensor = TensorNormalize((589,), 4, 'cpu', fixed=fixed)
    tensor.import_normalizer(type('Stats', (), dict(obs_rms=rms, ret_rms=RunningMeanStd(shape=()), gamma=.99))())
    frozen = FrozenMoments(rms, fixed)
    terms = fixed[0][:95]
    for _ in range(20):
        batch = rng.normal(1., 3., (64, 589))
        batch[:, columns] *= 100.  # arm inputs far outside anything seen
        normalized = tensor.observation(torch.from_numpy(batch))
        frozen.update(batch)
        on_cpu = np.clip((batch - frozen.mean) / np.sqrt(frozen.var + 1e-8), -10, 10).astype(np.float32)
        np.testing.assert_allclose(normalized.numpy(), on_cpu, rtol=0, atol=1e-6)
        bound = np.abs(batch[:, columns]).max() / task.ARM_BLIND_SCALE + 1e-6  # z + x/1e6, |x| up to ~1.5e3 here
        assert np.abs(on_cpu[:, columns] - z.astype(np.float32)).max() <= bound
    for moments in ((tensor.obs.mean.numpy(), tensor.obs.var.numpy()), (frozen.mean, frozen.var)):
        np.testing.assert_array_equal(moments[0][columns], -z * task.ARM_BLIND_SCALE)
        assert (moments[1][columns] == 1e12).all()
        np.testing.assert_array_equal(moments[0][terms], fixed[1][:95])
        np.testing.assert_array_equal(moments[1][terms], fixed[2][:95])
    np.testing.assert_allclose(tensor.obs.mean.numpy(), frozen.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tensor.obs.var.numpy(), frozen.var, rtol=1e-12)
    assert float(tensor.obs.count) == frozen.count > 8e9
    again = pickle.loads(pickle.dumps(frozen))
    assert type(again) is RunningMeanStd and np.array_equal(again.mean, frozen.mean) and again.count == frozen.count
    fresh = TensorNormalize((589,), 4, 'cpu', fixed=fixed)  # no import: the NaN means keep the initial 0
    fresh.observation(torch.from_numpy(rng.normal(5., 1., (8, 589))))
    assert not fresh.obs.mean[torch.from_numpy(columns)].any()


def test_the_cpu_training_path_freezes_the_columns_in_its_vecnormalize(tmp_path):
    """train.freeze_normalization: a VecNormalize stepping in training mode keeps the blind columns; saved, it loads as
    the ordinary SB3 object with the frozen values; a task without fixed columns keeps its plain statistics."""
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.train import FrozenMoments, freeze_normalization
    config = {**SIMPLE, 'arm_blind': True}
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config)]))
    try:
        columns = np.asarray(env.envs[0].unwrapped.arm_columns)
        z = np.random.default_rng(6).uniform(-1, 1, len(columns))
        env.obs_rms = blinded(env.obs_rms, columns, z)
        freeze_normalization(env, 'g1_velocity', 'g1_29dof', config)
        assert isinstance(env.obs_rms, FrozenMoments)
        env.reset()
        for action in actions(30, seed=3):
            env.step(action[None])
        assert env.obs_rms.count > 30
        np.testing.assert_array_equal(env.obs_rms.mean[columns], -z * task.ARM_BLIND_SCALE)
        assert (env.obs_rms.var[columns] == 1e12).all() and env.obs_rms.var[:70].min() != 1
        env.save(tmp_path / 'normalize.pkl')
        loaded = VecNormalize.load(tmp_path / 'normalize.pkl', DummyVecEnv([lambda: G1VelocityEnv(**config)]))
        assert type(loaded.obs_rms) is RunningMeanStd
        np.testing.assert_array_equal(loaded.obs_rms.mean, env.obs_rms.mean)
        loaded.close()
    finally:
        env.close()
    plain = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**SIMPLE)]))
    try:
        freeze_normalization(plain, 'g1_velocity', 'g1_29dof', SIMPLE)
        assert type(plain.obs_rms) is RunningMeanStd
    finally:
        plain.close()


def _stage(path, task_config):
    """A small real-robot stage under run dir `path` (trained 16 steps, statistics off the defaults, an actor whose
    outputs follow its inputs)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=task_config))
    write_json(path / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        model.learn(16)
        env.obs_rms.mean[:] = np.random.default_rng(1).normal(0, .3, 494)
        with torch.no_grad():
            model.policy.action_net.weight.mul_(10)
        return config, save_stage(path, model, env)
    finally:
        env.close()


def _run(path, config, **changes):
    from humanoid_sim.core import write_json
    path.mkdir()
    write_json(path / 'run_config.json', {**config, 'task_config': {**config['task_config'], **changes}})
    return path


def test_blind_arms_branches_a_stage_that_acts_as_its_parent_with_arm_inputs_held_at_z(tmp_path):
    """The branch's stage: validate_resume takes it under arm_blind and refuses it under the parent's contract; the
    weights, steps and every other column's statistics are the parent's, the arm columns −z·1e6 / 1e12; it acts as
    the parent does with its normalized arm inputs replaced by z. Refused: a blind source, a task without arm_blind or
    with another change, and carry_over / insert_terms switching arm_blind."""
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import load_stage, read_json
    from humanoid_sim.warm_start import blind_arms, carry_over, check_carry_over, check_insert_terms
    (tmp_path / 'old').mkdir()
    config, stage = _stage(tmp_path / 'old', SIMPLE)
    z = np.random.default_rng(7).uniform(-1, 1, 210)
    moved = blind_arms(stage, _run(tmp_path / 'blind', config, arm_blind=True), z)
    metadata, old = read_json(moved / 'metadata.json'), read_json(stage / 'metadata.json')
    assert metadata['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-armblind-critic4'
    blind = task_identity('g1_29dof', 'g1_velocity', {**config['task_config'], 'arm_blind': True})
    validate_resume(checkpoint_identity(metadata), blind)
    with pytest.raises(ValueError, match='observation_schema'):
        validate_resume(checkpoint_identity(metadata), task_identity('g1_29dof', 'g1_velocity', config['task_config']))
    assert metadata['steps'] == old['steps'] and moved.name == stage.name
    warm = read_json(moved / 'warm_start.json')
    assert np.allclose(warm['z'], z, rtol=0, atol=0) and warm['reference_given'] and len(warm['columns']) == 210
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        columns = np.asarray(new_norm.venv.envs[0].unwrapped.arm_columns)
        assert new_norm.venv.envs[0].unwrapped.arm_blind and warm['columns'] == columns.tolist()
        others = np.setdiff1d(np.arange(494), columns)
        for name in ('mean', 'var'):
            np.testing.assert_array_equal(getattr(new_norm.obs_rms, name)[others],
                                          getattr(old_norm.obs_rms, name)[others])
        np.testing.assert_array_equal(new_norm.obs_rms.mean[columns], -z * 1e6)
        assert (new_norm.obs_rms.var[columns] == 1e12).all() and new_norm.obs_rms.count == old_norm.obs_rms.count
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        raw = new_norm.venv.envs[0].unwrapped
        obs, _ = raw.reset(seed=9, options={'command': [.5, 0., 0.]})
        held = np.zeros(29)
        held[list(ARMS)] = np.random.default_rng(8).uniform(-1, 1, 14) / raw.scale[list(ARMS)]
        raw.set_arm_perturb(60, held)
        worst = 0.
        for _ in range(60):
            replaced = old_norm.normalize_obs(obs)
            replaced[columns] = z
            want = before.predict(replaced, deterministic=True)[0]
            got = after.predict(new_norm.normalize_obs(obs), deterministic=True)[0]
            worst = max(worst, float(np.abs(got - want).max()))
            obs, _, terminated, _, _ = raw.step(got)
            if terminated:
                break
        assert worst < 1e-3
    finally:
        old_norm.close(), new_norm.close()
    refused = [(moved, dict(arm_blind=True)),  # blind already
               (stage, dict()),  # no arm_blind in the task
               (stage, dict(arm_blind=True, lock_wrists=True))]  # another contract change
    for number, (source, changes) in enumerate(refused):
        run = _run(tmp_path / f'refused{number}', config, **changes)
        with pytest.raises(ValueError):
            blind_arms(source, run, z)
        assert not any(run.glob('stage_*'))
    with pytest.raises(ValueError, match='blind_branch'):
        carry_over(stage, _run(tmp_path / 'carried', config, arm_blind=True))
    saved = checkpoint_identity(old)
    with pytest.raises(ValueError, match='blind_branch'):
        check_insert_terms(saved, task_identity('g1_29dof', 'g1_velocity', {**config['task_config'], **CROUCH,
                                                                             'arm_blind': True}))
    # Both blind, another contract change: carry_over may move it (the arm columns keep their statistics).
    check_carry_over(blind, task_identity('g1_29dof', 'g1_velocity', {**config['task_config'], 'arm_blind': True,
                                                                       'lock_wrists': True}))


def test_export_of_a_blind_stage_is_the_plain_graph_with_the_arm_inputs_constant(tmp_path):
    """policy.onnx of a blind stage: the same ops, inputs and deploy.yaml as its parent's (but the header), the property
    arm_blind; check() passes; whatever the arm inputs, the actions stay put."""
    onnxruntime = pytest.importorskip('onnxruntime')
    onnx = pytest.importorskip('onnx')
    yaml = pytest.importorskip('yaml')
    from humanoid_sim.export_g1 import check, export, onnx_properties
    from humanoid_sim.warm_start import blind_arms
    (tmp_path / 'old').mkdir()
    config, stage = _stage(tmp_path / 'old', SIMPLE)
    z = np.random.default_rng(7).uniform(-1, 1, 210)
    moved = blind_arms(stage, _run(tmp_path / 'blind', config, arm_blind=True), z)
    plain, blind = export(stage, tmp_path / 'plain'), export(moved, tmp_path / 'out')
    properties = onnx_properties(blind['onnx'])
    assert properties['arm_blind'] == 'true' and 'arm_blind' not in onnx_properties(plain['onnx'])
    ops = lambda result: [node.op_type for node in onnx.load(str(result['onnx'])).graph.node]
    assert ops(blind) == ops(plain)
    text = blind['deploy'].read_text(encoding='utf-8')
    assert 'arm_blind' in text.split('joint_ids_map')[0]
    body = lambda result: result['deploy'].read_text(encoding='utf-8').split('joint_ids_map', 1)[1]
    assert body(blind) == body(plain)
    assert yaml.safe_load(text)['observations']['joint_pos_rel']['history_length'] == 5
    verdict = check(moved, blind['onnx'], samples=20, stride=2)
    assert verdict['ok'], verdict
    session = onnxruntime.InferenceSession(str(blind['onnx']), providers=['CPUExecutionProvider'])
    env = G1VelocityEnv(**{**config['task_config'], 'arm_blind': True})
    try:
        obs, _ = env.reset(seed=3, options={'command': [.5, 0., 0.]})
        columns = np.asarray(env.arm_columns)
        rng = np.random.default_rng(4)
        run = lambda o: session.run(['actions'], {'obs': o[None, :490].astype(np.float32)})[0][0]
        reference = run(obs)
        for _ in range(5):
            other = obs.copy()
            other[columns] = rng.uniform(-3, 3, len(columns))
            assert np.abs(run(other) - reference).max() < 1e-3
        other = obs.copy()
        other[:3] += 1.  # a column the policy sees
        assert np.abs(run(other) - reference).max() > 1e-3
    finally:
        env.close()


@pytest.mark.skipif(not (STAGE_7741 / 'metadata.json').exists(), reason='stage 7741 is not on this machine')
def test_blind_reference_repeats_the_critics_mixref_of_stage_7741():
    """warm_start.blind_reference at the critic's reset seed 73000 gives his mixref z of 7741 (blind_const.py)."""
    from humanoid_sim.warm_start import blind_reference
    z, means = blind_reference(STAGE_7741, seed=73000)
    first, last, total = CRITIC_Z_7741
    assert z.shape == (210,) and len(means) == 3
    np.testing.assert_allclose(z[:3], first, atol=1e-5)
    np.testing.assert_allclose(z[-3:], last, atol=1e-5)
    assert float(z.sum()) == pytest.approx(total, abs=1e-3)
