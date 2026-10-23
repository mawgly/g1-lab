import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.core import load_stage, read_json, save_stage, write_json
from humanoid_sim.export_g1 import (TERMS, TRIM_FILE, ExportedPolicy, check, command_inputs, deploy_target, export,
                                    mirror_maps, onnx_properties, reference_action, stand_trim, write_yaml)
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

onnxruntime = pytest.importorskip('onnxruntime')
yaml = pytest.importorskip('yaml')

REAL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=True, history=5)


def test_export_matches_the_stage_policy_and_writes_the_g1_ctrl_config(tmp_path):
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=REAL))  # no phase_stop: the export warns
    (tmp_path / 'run').mkdir()
    write_json(tmp_path / 'run' / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        rng = np.random.default_rng(1)  # statistics far from the defaults, so normalization and its clip matter
        env.obs_rms.mean[:] = rng.normal(0, 1, 494)
        env.obs_rms.var[:] = rng.uniform(.01, 4, 494)
        with torch.no_grad():  # large actions, so the ±action_limit clip matters too
            model.policy.action_net.weight.mul_(2000)
        stage = save_stage(tmp_path / 'run', model, env)
    finally:
        env.close()

    result = export(stage, tmp_path / 'out')
    assert result['actor_inputs'] == 490 and len(result['warnings']) == 1 and 'phase_stop' in result['warnings'][0]
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    (given,), (taken,) = session.get_inputs(), session.get_outputs()
    assert (given.name, given.shape, given.type) == ('obs', [1, 490], 'tensor(float)')
    assert (taken.name, taken.shape) == ('actions', [1, 29])

    policy, norm = load_stage(stage)
    try:
        obs = np.random.default_rng(2).normal(0, 3, (20, 494)).astype(np.float32)
        expected = np.clip(policy.predict(norm.normalize_obs(obs), deterministic=True)[0], -10, 10)
        got = np.concatenate([session.run(['actions'], {'obs': row[None, :490]})[0] for row in obs])
        assert np.abs(expected).max() == 10  # the clip was exercised
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)
    finally:
        norm.close()
    verdict = check(stage, result['onnx'], samples=10, stride=2)
    assert verdict['ok'] and verdict['samples'] == 10

    deploy = yaml.safe_load(result['deploy'].read_text(encoding='utf-8'))
    assert deploy['joint_ids_map'] == list(range(29)) and deploy['step_dt'] == .02
    assert deploy['commands']['base_velocity']['ranges']['lin_vel_x'] == [
        config['task_config']['commands']['min_speed'], config['task_config']['commands']['max_speed']]
    action = deploy['actions']['JointPositionAction']
    assert action['offset'] == deploy['default_joint_pos'] and len(action['scale']) == 29
    assert len(action['clip']) == 29 and all(len(pair) == 2 and pair[0] < pair[1] for pair in action['clip'])
    terms = deploy['observations']
    assert tuple(terms) == TERMS
    assert all(term['history_length'] == 5 and term['clip'] is None and isinstance(term['params'], dict)
               for term in terms.values())
    assert [len(term['scale']) for term in terms.values()] == [3, 3, 3, 2, 29, 29, 29]
    assert terms['gait_phase']['params'] == {'period': .6}
    import onnx
    assert 'Where' not in {node.op_type for node in onnx.load(str(result['onnx'])).graph.node}  # no lock step


WRISTS = [20, 21, 27, 28]


def test_export_with_lock_wrists_puts_zeros_on_the_wrist_outputs(tmp_path):
    """predict knows nothing of the lock: the ONNX output equals the action the environment applies (the clipped
    mean with the wrist pitch and yaw at 0), and g1_ctrl's target for those joints is default_joint_pos."""
    import onnx
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config={**REAL, 'phase_stop': True, 'lock_wrists': True}))
    (tmp_path / 'run').mkdir()
    write_json(tmp_path / 'run' / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        with torch.no_grad():  # large actions: the policy's own wrist outputs are far from 0
            model.policy.action_net.weight.mul_(2000)
        stage = save_stage(tmp_path / 'run', model, env)
    finally:
        env.close()
    result = export(stage, tmp_path / 'out')
    assert result['warnings'] == []
    graph = onnx.load(str(result['onnx']))
    assert 'Where' in {node.op_type for node in graph.graph.node}
    assert {p.key: p.value for p in graph.metadata_props}['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock'
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    policy, norm = load_stage(stage)
    try:
        obs = np.random.default_rng(2).normal(0, 3, (20, 494)).astype(np.float32)
        predicted = np.clip(policy.predict(norm.normalize_obs(obs), deterministic=True)[0], -10, 10)
        got = np.concatenate([session.run(['actions'], {'obs': row[None, :490]})[0] for row in obs])
        assert np.abs(predicted[:, WRISTS]).min() > 0  # the lock has something to do
        assert not got[:, WRISTS].any()
        others = [i for i in range(29) if i not in WRISTS]
        np.testing.assert_allclose(got[:, others], predicted[:, others], atol=1e-4, rtol=0)
    finally:
        norm.close()
    verdict = check(stage, result['onnx'], samples=10, stride=2)
    assert verdict['ok'] and verdict['samples'] == 10
    deploy = yaml.safe_load(result['deploy'].read_text(encoding='utf-8'))
    offset = deploy['actions']['JointPositionAction']['offset']
    assert [offset[i] for i in WRISTS] == [deploy['default_joint_pos'][i] for i in WRISTS] == [0.] * 4


LOCKED = {**REAL, 'phase_stop': True, 'lock_wrists': True}


def make_stage(tmp_path, task_config, gain=1.):
    """A real-robot G1 stage with an untrained 32×32 actor; gain scales its action head (large: the clips act)."""
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=task_config))
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        with torch.no_grad():
            model.policy.action_net.weight.mul_(gain)
        return save_stage(run, model, env)
    finally:
        env.close()


def stage_env(stage, **overrides):
    from humanoid_sim.core import stage_task_config
    config = {**stage_task_config(stage), **overrides}
    return config, G1VelocityEnv(**config)


def test_stand_trim_acts_only_below_the_command_threshold_and_the_plain_graph_comes_back(tmp_path):
    """With a trim, policy.onnx returns clip(a − b) while the newest frame's command has norm < 0.1 (phase_stop's
    rule) and a otherwise; the export folder's adapter returns the same, check() compares with the same trim, and
    exporting again without it gives the plain graph (no trim step, no stand_trim.json)."""
    import onnx
    stage = make_stage(tmp_path, LOCKED)
    trim = np.full(29, .5)
    trim[WRISTS] = 0.
    trim[0] = 25.  # past the box: a − b is clipped to −10 again
    result = export(stage, tmp_path / 'out', trim, {'seeds': [1]})
    assert command_inputs(29, 5) == (42, 45)
    ops = [node.op_type for node in onnx.load(str(result['onnx'])).graph.node]
    assert 'Less' in ops and ops.count('Where') == 2  # lock_wrists, then the trim
    saved = read_json(result['trim'])
    assert (saved['b'], saved['threshold'], saved['command_inputs'], saved['seeds']) == (trim.tolist(), .1, [42, 45],
                                                                                         [1])
    header = result['deploy'].read_text(encoding='utf-8').splitlines()
    assert header[1].startswith('# stand trim') and yaml.safe_load('\n'.join(header))['step_dt'] == .02

    commands = np.array([[0., 0., 0.], [.09, 0., 0.], [0., -.05, .08], [.11, 0., 0.], [0., 0., -.12], [.5, 0., .2]] * 3)
    standing = np.linalg.norm(commands, axis=1) < .1
    obs = np.random.default_rng(3).normal(0, 1, (len(commands), 494)).astype(np.float32)
    obs[:, 42:45] = commands
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    exported = ExportedPolicy(tmp_path / 'out')
    assert exported.stage.resolve() == stage.resolve() and exported.trim.tolist() == trim.tolist()
    policy, norm = load_stage(stage)
    try:
        plain = np.clip(policy.predict(norm.normalize_obs(obs), deterministic=True)[0], -10, 10)
        plain[:, WRISTS] = 0.
        got = np.concatenate([session.run(['actions'], {'obs': row[None, :490]})[0] for row in obs])
        np.testing.assert_allclose(got[~standing], plain[~standing], atol=1e-4, rtol=0)
        np.testing.assert_allclose(got[standing], np.clip(plain[standing] - trim, -10, 10), atol=1e-4, rtol=0)
        assert (got[standing, 0] == -10).all() and not got[:, WRISTS].any()
        # The adapter: the raw observation in (critic extras dropped), the graph's action out, one row or a batch.
        assert exported.normalize_obs(obs) is obs
        np.testing.assert_allclose(exported.predict(obs)[0], got, atol=1e-6, rtol=0)
        np.testing.assert_allclose(exported.predict(obs[1])[0], got[1], atol=1e-6, rtol=0)
    finally:
        norm.close()
    assert check(stage, result['onnx'], samples=10, stride=2)['ok']

    again = export(stage, tmp_path / 'out')
    assert again['trim'] is None and not (tmp_path / 'out' / TRIM_FILE).exists()
    assert 'Less' not in [node.op_type for node in onnx.load(str(again['onnx'])).graph.node]
    assert 'stand_trim' not in onnx_properties(again['onnx']) and ExportedPolicy(tmp_path / 'out').trim is None
    assert not again['deploy'].read_text(encoding='utf-8').splitlines()[1].startswith('#')


def test_stand_trim_is_half_the_mean_standing_asymmetry_within_the_bound(tmp_path):
    """b: half the mean of π(o) − M_a·π(M_o·o) (π = predict clipped to the box) over the window of standing episodes
    driven by the policy itself, 0 on the locked wrists, clipped to ±0.052 rad of target per joint."""
    stage = make_stage(tmp_path, LOCKED, gain=2000.)  # bang-bang actions: an asymmetry far past the bound
    trim, info = stand_trim(stage, seeds=(5, 6), window=(1, 3))
    config, env = stage_env(stage, heading_hold=False)
    policy, norm = load_stage(stage)
    try:
        obs_index, obs_sign, action_index, action_sign = mirror_maps(env)
        act = lambda o: np.clip(policy.predict(norm.normalize_obs(o), deterministic=True)[0], -10, 10)
        deltas = []
        for seed in (5, 6):
            obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': 0})
            for step in range(3):
                if step >= 1:
                    deltas.append(act(obs) - act(obs[obs_index] * obs_sign)[action_index] * action_sign)
                obs = env.step(act(obs))[0]
        half = .5 * np.mean(deltas, axis=0)
        half[WRISTS] = 0.
        bound = .052 / env.scale
    finally:
        norm.close()
        env.close()
    np.testing.assert_allclose(info['unclipped'], half, atol=1e-6, rtol=0)
    np.testing.assert_allclose(trim, np.clip(half, -bound, bound), atol=1e-6, rtol=0)
    assert not trim[WRISTS].any() and info['clipped_joints'] and np.abs(info['b_rad']).max() <= .052 + 1e-9
    assert info['joints'][0] == 'left_hip_pitch_joint' and len(info['b_rad']) == 29


def test_exported_policy_acts_as_the_stage_with_its_stand_trim(tmp_path):
    """Along real trajectories (standing, walking): the export folder's action equals policy.predict clipped,
    locked and trimmed only while told to stand (reference_action), and the trim does change the standing action."""
    stage = make_stage(tmp_path, LOCKED, gain=10.)  # actions ~0.1: a trim well above the export's float error
    trim, info = stand_trim(stage, seeds=(5,), window=(1, 4))
    export(stage, tmp_path / 'out', trim, info)
    exported = ExportedPolicy(tmp_path / 'out')
    config, env = stage_env(stage)
    policy, norm = load_stage(stage)
    try:
        for command in ([0., 0., 0.], [.5, 0., 0.]):
            obs, _ = env.reset(seed=1, options={'command': command})
            for _ in range(5):
                expected = reference_action(policy, norm, obs, config, env, trim, (42, 45))
                untrimmed = reference_action(policy, norm, obs, config, env)
                got = exported.predict(obs)[0]
                np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)
                assert (np.abs(got - untrimmed).max() > 1e-3) == (command == [0., 0., 0.])
                obs = env.step(got)[0]
    finally:
        norm.close()
        env.close()


def test_symmetric_export_is_the_mirrored_mean_and_mirror_equivariant(tmp_path):
    stage = make_stage(tmp_path, LOCKED)
    result = export(stage, tmp_path / 'out', symmetric=True)
    assert onnx_properties(result['onnx'])['symmetric'] == 'true' and result['trim'] is None
    config, env = stage_env(stage)
    mirror = mirror_maps(env)
    obs_index, obs_sign, action_index, action_sign = mirror
    exported = ExportedPolicy(tmp_path / 'out')
    obs = np.random.default_rng(4).normal(0, 1, (8, 494)).astype(np.float32)
    policy, norm = load_stage(stage)
    try:
        got = exported.predict(obs)[0]
        expected = np.array([reference_action(policy, norm, row, config, env, mirror=mirror) for row in obs])
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)
        mirrored = exported.predict(obs[:, obs_index] * obs_sign)[0]
        np.testing.assert_allclose(mirrored, got[:, action_index] * action_sign, atol=1e-5, rtol=0)
    finally:
        norm.close()
        env.close()
    assert check(stage, result['onnx'], samples=10, stride=2)['ok']
    with pytest.raises(ValueError, match='Подстройка стоя и симметризация'):
        export(stage, tmp_path / 'both', np.zeros(29), symmetric=True)


KNEES = [3, 9]
FLOORED = {**LOCKED, 'knee_floor': .061}


def test_export_with_knee_floor_raises_the_knee_clip_and_leaves_the_graph_alone(tmp_path):
    """g1_ctrl clamps the knee target to deploy.yaml's clip, and its last_action is the ONNX output before that clamp,
    as the environment's previous action: the floor goes into the clip only. The same weights with and without it
    give the same graph and outputs; g1_ctrl's targets from deploy.yaml are the environment's PD targets (check), and
    an export whose clip still lets the knee target reach the stop fails the check."""
    import onnx
    stages = {}
    for name, task_config in (('plain', LOCKED), ('floored', FLOORED)):
        (tmp_path / name).mkdir()
        stages[name] = make_stage(tmp_path / name, task_config, gain=2000.)  # bang-bang: knees asked far below
    results = {name: export(stage, tmp_path / f'out-{name}') for name, stage in stages.items()}
    graphs = {name: onnx.load(str(result['onnx'])).graph for name, result in results.items()}
    assert [n.op_type for n in graphs['floored'].node] == [n.op_type for n in graphs['plain'].node]
    assert onnx_properties(results['floored']['onnx'])['action_schema'] == \
        'g1_velocity:action-v2-limit10-wristlock-knee0.061'
    obs = np.random.default_rng(5).normal(0, 3, (20, 494)).astype(np.float32)
    outputs = {}
    for name, result in results.items():
        session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
        outputs[name] = np.concatenate([session.run(['actions'], {'obs': row[None, :490]})[0] for row in obs])
    np.testing.assert_array_equal(outputs['floored'], outputs['plain'])
    assert (outputs['floored'][:, KNEES] < (.061 - .3) / .35).sum() >= 5  # the policy asks below the floor
    deploys = {name: yaml.safe_load(result['deploy'].read_text(encoding='utf-8')) for name, result in results.items()}
    clips = {name: deploy['actions']['JointPositionAction']['clip'] for name, deploy in deploys.items()}
    config, env = stage_env(stages['floored'])
    try:
        ranges = env.model.jnt_range[env.joints]
        assert clips['plain'] == [[float(low), float(high)] for low, high in ranges]
        assert [clips['floored'][i] for i in KNEES] == [[.061, float(ranges[i, 1])] for i in KNEES]
        others = [i for i in range(29) if i not in KNEES]
        assert [clips['floored'][i] for i in others] == [clips['plain'][i] for i in others]
        assert {k: v for k, v in deploys['floored'].items() if k != 'actions'} == \
            {k: v for k, v in deploys['plain'].items() if k != 'actions'}
        for action in outputs['floored']:  # g1_ctrl's target for the ONNX action = the environment's PD target
            target = deploy_target(action, deploys['floored'])
            np.testing.assert_allclose(target, env._target(env.applied_action(action)), rtol=0, atol=1e-9)
            assert (target[KNEES] >= .061).all()
    finally:
        env.close()
    verdict = check(stages['floored'], results['floored']['onnx'], samples=10, stride=2)
    assert verdict['ok'] and verdict['max_target_diff'] < 1e-4
    assert check(stages['plain'], results['plain']['onnx'], samples=10, stride=2)['ok']
    stale = {**deploys['floored'], 'actions': deploys['plain']['actions']}  # the knee clip down to the stop
    write_yaml(stale, results['floored']['deploy'], 'stale clip')
    verdict = check(stages['floored'], results['floored']['onnx'], samples=10, stride=2)
    assert not verdict['ok'] and verdict['max_abs_diff'] < 1e-4 and verdict['max_target_diff'] > .1


def test_stand_trim_works_with_knee_floor(tmp_path):
    """--stand-trim on a knee_floor stage: b comes from the stage's own environment (floor included), the graph is the
    trimmed one, deploy.yaml keeps the knee floor, and the check (actions and g1_ctrl targets) passes."""
    stage = make_stage(tmp_path, FLOORED, gain=10.)
    trim, info = stand_trim(stage, seeds=(5,), window=(1, 4))
    result = export(stage, tmp_path / 'out', trim, info)
    deploy = yaml.safe_load(result['deploy'].read_text(encoding='utf-8'))
    assert [deploy['actions']['JointPositionAction']['clip'][i][0] for i in KNEES] == [.061, .061]
    assert ExportedPolicy(tmp_path / 'out').trim.tolist() == trim.tolist() and result['trim'].is_file()
    verdict = check(stage, result['onnx'], samples=10, stride=2)
    assert verdict['ok'] and verdict['max_target_diff'] < 1e-4


def test_command_ranges_narrow_only_deploy_yaml_within_the_training_ranges(tmp_path):
    """A stage trained up to 2 m/s exported with the stage-1 clip: deploy.yaml's command ranges are the narrowed ones
    (noted in its header), everything else and the graph's outputs as the plain export; ranges wider than training or
    without 0 are refused before anything is written."""
    trained = {'min_speed': -1., 'max_speed': 2., 'lateral_speed': .5, 'yaw_rate': 1.}
    stage = make_stage(tmp_path, {**FLOORED, 'commands': trained})
    plain = export(stage, tmp_path / 'plain')
    narrow = export(stage, tmp_path / 'narrow', commands=(-.5, 1., .5, 1.))
    deploys = {name: yaml.safe_load(result['deploy'].read_text(encoding='utf-8'))
               for name, result in (('plain', plain), ('narrow', narrow))}
    ranges = lambda deploy: deploy['commands']['base_velocity']['ranges']
    assert ranges(deploys['plain']) == {'lin_vel_x': [-1., 2.], 'lin_vel_y': [-.5, .5], 'ang_vel_z': [-1., 1.],
                                        'heading': None}
    assert ranges(deploys['narrow']) == {'lin_vel_x': [-.5, 1.], 'lin_vel_y': [-.5, .5], 'ang_vel_z': [-1., 1.],
                                         'heading': None}
    assert narrow['commands'] == {'lin_vel_x': [-.5, 1.], 'lin_vel_y': [-.5, .5], 'ang_vel_z': [-1., 1.]}
    assert plain['commands'] == {'lin_vel_x': [-1., 2.], 'lin_vel_y': [-.5, .5], 'ang_vel_z': [-1., 1.]}
    assert {k: v for k, v in deploys['narrow'].items() if k != 'commands'} == \
        {k: v for k, v in deploys['plain'].items() if k != 'commands'}
    header = narrow['deploy'].read_text(encoding='utf-8').splitlines()
    assert header[1].startswith('# command ranges narrowed') and header[2] == '#   lin_vel_x [-1, 2] -> [-0.5, 1]'
    assert not plain['deploy'].read_text(encoding='utf-8').splitlines()[1].startswith('#')
    obs = np.random.default_rng(6).normal(0, 3, (10, 494)).astype(np.float32)
    outputs = []
    for result in (plain, narrow):
        session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
        outputs.append(np.concatenate([session.run(['actions'], {'obs': row[None, :490]})[0] for row in obs]))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert check(stage, narrow['onnx'], samples=10, stride=2)['ok']
    for wrong, message in (((-1.5, 1., .5, 1.), 'шире обучения'), ((-.5, 1., .7, 1.), 'шире обучения'),
                           ((.2, 1., .5, 1.), 'включать 0'), ((-.5, 1., .5, float('nan')), 'включать 0')):
        with pytest.raises(ValueError, match=message):
            export(stage, tmp_path / 'refused', commands=wrong)
    assert not (tmp_path / 'refused').exists()


def test_goal_check_and_supervise_run_an_export_folder(tmp_path):
    """--export: the folder's policy acts in its stage's environment; σ is still the stage's."""
    from humanoid_sim.goal_check import walks
    from humanoid_sim.supervise import check_stage
    stage = make_stage(tmp_path, LOCKED)
    trim = np.full(29, .3)
    trim[WRISTS] = 0.
    export(stage, tmp_path / 'out', trim)
    exported = ExportedPolicy(tmp_path / 'out')

    class Spy:
        calls = 0

        def normalize_obs(self, obs):
            return exported.normalize_obs(obs)

        def predict(self, obs, deterministic=True):
            Spy.calls += 1
            return exported.predict(obs, deterministic)

    spy = Spy()
    report = check_stage(exported.stage, seconds=.1, seeds=1, scenarios=('stand',), acting=(spy, spy))
    assert Spy.calls == 5 and report['policy_std'] is not None and report['scenarios']['stand']['falls'] == '0/1'
    walk = walks(exported.stage, seeds=1, seconds=.1, acting=(spy, spy))
    assert Spy.calls == 5 + 6 * 5 and set(walk['drift']) == {(1., 'left'), (1., 'right'), (.5, 'left'), (.5, 'right')}
