"""g1_getup task_config pd_bound (iteration 34, HoST's β): the hard PD error bound with its curriculum, CPU. The bound
keeps every motor target but the locked wrists' within c·y1/kp of the observed joint angle; c steps down per world at
resets by the assist's rule; its own action contract; stages keep c; evaluation and export at the lowest c."""
import json
import math

import mujoco
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.core import read_json, save_stage, write_json
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, g1_getup_config
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_getup import rows_of, small_bank  # noqa: F401 (fixture)

BOUND = [1., .6]  # the working value: start, lowest
OPTIONS = dict(action_delay_steps=0, motor_friction_scale=1., body_friction=.6, assist_force=0.)


def env_constants():
    env = G1GetupEnv()
    try:
        low, high = env.model.jnt_range[env.joints].T
        return dict(home=env.home.copy(), low=low, high=high, peak=env.peak.copy(), kp=env.kp.copy(),
                    scale=env.scale.copy(), free=env.bound_free.copy())
    finally:
        env.close()


def test_pd_bound_setting_and_its_own_action_contract():
    from humanoid_sim.tasks.catalog import resolve_task
    from humanoid_sim.warm_start import check_carry_over
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['pd_bound'] is None  # old stages and roots: off
    for given, kept in ((BOUND, BOUND), ((1, .6), BOUND), ([.8, .8], [.8, .8]), (None, None), ([1, .1], [1., .1])):
        assert validate_config({**base, 'task_config': {'pd_bound': given}})['task_config']['pd_bound'] == kept
    for bad in ([.6, 1.], [1.2, .6], [1., .05], [1.], [1., .6, .5], .6, 'min', [True, .6], [float('nan'), .6],
                [1., float('inf')], {'start': 1.}):
        with pytest.raises(ValueError):
            g1_getup_config({'pd_bound': bad})
    field = next(p for p in G1_GETUP_PARAMETERS if p['key'] == 'pd_bound')
    assert field['default'] is None and (field['min'], field['max']) == (.1, 1.) and 'пика' in field['label']
    saved = task_identity('g1_29dof', 'g1_getup')  # stages saved before the setting existed
    assert saved == task_identity('g1_29dof', 'g1_getup', {'pd_bound': None})
    bounded = task_identity('g1_29dof', 'g1_getup', {'pd_bound': BOUND})
    assert bounded == {**saved, 'action_schema': 'g1_getup:action-v2-limit10-wristlock-knee0.061-pdbound0.6'}
    assert task_identity('g1_29dof', 'g1_getup', {'pd_bound': [.9, .6]}) == bounded  # the start is the curriculum's
    with pytest.raises(ValueError):
        validate_resume(saved, bounded)  # an ordinary branch is refused
    check_carry_over(saved, bounded)  # carry_over may move the weights
    with pytest.raises(ValueError):  # another lowest bound is another contract
        validate_resume(bounded, task_identity('g1_29dof', 'g1_getup', {'pd_bound': [1., .5]}))
    walking = task_identity('g1_29dof', 'g1_velocity', dict(real_robot=True, critic_extras=True, history=5,
                                                            phase_stop=True, lock_wrists=True, knee_floor=.061,
                                                            action_limit=10.))
    assert walking['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock-knee0.061'  # walking untouched
    assert resolve_task('g1_29dof', 'g1_getup').evaluation_options == {'assist_force': 0., 'pd_bound': 'min'}


def test_the_bound_is_the_same_in_numpy_and_torch_and_keeps_targets_near_the_observed_angles():
    """a′ = min(max(a, (jp − e)/s), (jp + e)/s) on the free joints, e = c·y1/kp: the PD target home + s·a′ lies within
    e of the observed angle home + jp, an action already inside is untouched, the locked wrists are not bounded, and
    with the angles inside the joint ranges a′ stays inside the ±10 box (every free joint's range is)."""
    k = env_constants()
    rng = np.random.default_rng(0)
    n = 256
    action = rng.uniform(-10, 10, (n, 29))
    action[::4] = rng.normal(0, .3, (n // 4, 29))  # small actions: mostly inside the bound
    position = rng.uniform(k['low'], k['high'], (n, 29)) - k['home']
    fraction = rng.uniform(.1, 1., n)
    args = (k['peak'], k['kp'], k['scale'], k['free'])
    expected = task.pd_bound(action, position, fraction, *args)
    error = fraction[:, None] * k['peak'] / k['kp']
    offset = k['scale'] * expected - position  # PD target − observed angle
    free = k['free']
    assert (np.abs(offset[:, free]) <= error[:, free] * (1 + 1e-12) + 1e-12).all()
    inside = np.abs(k['scale'] * action - position) <= error
    np.testing.assert_array_equal(expected[inside], action[inside])
    np.testing.assert_array_equal(expected[:, ~free], action[:, ~free])
    assert (~inside[:, free]).mean() > .3 and inside[:, free].mean() > .1  # both cases in play
    assert np.abs(expected).max() <= 10
    torched = [torch.as_tensor(v) for v in (action, position, fraction, *args)]
    np.testing.assert_array_equal(task.pd_bound(*torched).numpy(), expected)
    single = [v.float() if v.dtype == torch.float64 else v for v in torched]
    np.testing.assert_allclose(task.pd_bound(*single).numpy(), expected, rtol=1e-5, atol=1e-5)


def test_the_bound_steps_down_by_the_assists_rule_to_the_lowest():
    fraction = np.array([1., 1., .61, .6, .62, .9, .5])
    head = np.array([1.05, .95, 1.2, 1.2, np.nan, 1.01, 1.3])
    stepped = task.pd_bound_update(fraction, head, .6)
    np.testing.assert_allclose(stepped, [.98, 1., .6, .6, .62, .88, .6], rtol=0, atol=1e-12)
    assert stepped[2] == stepped[3] == .6  # the lowest itself, not 0.59 or 0.6000000000000001
    np.testing.assert_array_equal(task.pd_bound_update(torch.as_tensor(fraction), torch.as_tensor(head), .6).numpy(),
                                  stepped)
    for values, dtype in ((np.array([1.]), None), (torch.ones(1), torch.float32)):  # 20 episodes up: 1.0 -> 0.6
        trail = []
        for _ in range(25):
            values = task.pd_bound_update(values, np.array([1.1]) if dtype is None else torch.tensor([1.1]), .6)
            trail.append(float(values[0]))
        assert trail[18] == pytest.approx(.62, abs=1e-6) and trail[19:] == [float(np.float32(.6)) if dtype else .6] * 6
    assert task.PD_BOUND_STEP == .02


def test_the_newest_frame_columns_are_the_angles_the_actor_reads(small_bank):  # noqa: F811
    env = G1GetupEnv(bank=small_bank)
    try:
        assert env.position_columns.tolist() == list(range(171, 200))  # history 5: 4 · 29 into joint_pos_rel's 145
        assert env.position_columns.tolist() == task.joint_positions(29, 5, (), (4,), range(29))[-29:]
        obs, _ = env.reset(seed=0, options=dict(OPTIONS, start=int(rows_of(env.bank, 'back')[0])))
        rng = np.random.default_rng(1)
        for _ in range(4):
            angles = (env.data.qpos[env.qadr] - env.home).astype(np.float32)
            np.testing.assert_array_equal(obs[env.position_columns], angles)
            np.testing.assert_array_equal(env.observed_position, angles.astype(np.float64))
            obs, *_ = env.step(rng.uniform(-3, 3, 29))
    finally:
        env.close()


def test_the_cpu_task_applies_the_bound_in_everything_after_it(small_bank):  # noqa: F811
    """With c = 0.5 the applied action (bounded, then the wrists at 0) goes to the PD (its target within e of the
    angles observed before the step), the reward's action change and the observed last action; without a bound the
    same actions drive another trajectory."""
    k = env_constants()
    env = G1GetupEnv(bank=small_bank, pd_bound=BOUND)
    plain = G1GetupEnv(bank=small_bank)
    try:
        row = int(rows_of(env.bank, 'back')[0])
        env.reset(seed=0, options=dict(OPTIONS, start=row, pd_bound=.5))
        plain.reset(seed=0, options=dict(OPTIONS, start=row))
        assert env.bound == .5 and plain.bound is None
        rng = np.random.default_rng(2)
        last = np.zeros(29)
        weight = env.getup_weights['action_change']
        for _ in range(5):
            action = rng.uniform(-10, 10, 29)
            observed = env.observed_position.copy()
            expected = task.lock_joints(torch.as_tensor(task.pd_bound(
                action[None], observed[None], np.array([.5]), k['peak'], k['kp'], k['scale'], k['free'])),
                env.wrist_mask)[0].numpy()
            np.testing.assert_array_equal(env.applied_action(action), expected)
            target = env._target(expected)
            offset = np.abs(target - (env.home + observed))[k['free']]
            limit = .5 * k['peak'][k['free']] / k['kp'][k['free']]
            within = offset <= limit + 1e-9  # the PD target clips to the joint ranges and the knee floor after it
            assert within.mean() > .9
            obs, _, _, _, info = env.step(action)
            plain.step(action)
            np.testing.assert_array_equal(env.previous_action, expected)
            np.testing.assert_array_equal(obs[task.newest_frame_columns(29, 5, (), 6)], expected.astype(np.float32))
            assert info['reward_components']['action_change'] == pytest.approx(-weight * ((expected - last) ** 2).sum())
            assert info['pd_bound'] == .5
            last = expected
        assert not np.allclose(env.data.qpos, plain.data.qpos, atol=1e-3)
    finally:
        env.close(), plain.close()


def test_actions_in_flight_at_an_episode_start_are_the_bounded_zero(small_bank, monkeypatch):  # noqa: F811
    """Under a 10 ms delay the first substeps follow the actions still in flight: with the bound the zero action
    bounded around the start pose as observed (unbounded: home, a jerk from lying), only at an episode's first step."""
    row = int(rows_of(bank_module.load(small_bank), 'front')[0])
    seen = {}
    for bounded in (False, True):
        env = G1GetupEnv(bank=small_bank, pd_bound=BOUND if bounded else None)
        targets = []
        original = env._target
        monkeypatch.setattr(env, '_target', lambda action: targets.append(np.array(action)) or original(action))
        try:
            env.reset(seed=0, options=dict(OPTIONS, start=row, action_delay_steps=5, pd_bound=.4 if bounded else None))
            assert len(env.delay_actions) == 1
            # A start far from home (as a probe sets a fall's state after the reset): the joints 1 rad away.
            low, high = env.model.jnt_range[env.joints].T
            env.data.qpos[env.qadr] = np.clip(env.home + 1., low, high)
            mujoco.mj_forward(env.model, env.data)
            env._observation('fill')
            zero = env.applied_action(np.zeros(29))
            env.step(np.zeros(29))
            seen[bounded] = targets[1]  # the first call is this step's action, the second the one in flight
            np.testing.assert_array_equal(targets[1], zero)
            targets.clear()
            action = env.applied_action(np.full(29, 3.))
            env.step(np.full(29, 3.))
            np.testing.assert_array_equal(targets[1], zero)  # in flight now: the first step's own (bounded) action
            assert not np.array_equal(targets[0], zero) and np.array_equal(targets[0], action)
        finally:
            env.close()
    assert not seen[False].any() and np.abs(seen[True]).max() > .1


def test_the_curriculum_and_the_training_state_on_the_cpu_task(small_bank):  # noqa: F811
    from humanoid_sim.train import vec_training_state
    env = G1GetupEnv(bank=small_bank, pd_bound=BOUND)
    try:
        assert env.reset(seed=0)[1]['pd_bound'] == 1.
        env.head_window.extend([1.05] * env.head_window.maxlen)  # the head above 1 m for the last 2 s
        info = env.reset(seed=1)[1]
        assert (info['pd_bound'], info['assist_force']) == (.98, 180.)
        env.head_window.extend([1.2] * (env.head_window.maxlen - 1))  # shorter than 2 s: no step
        assert env.reset(seed=2)[1]['pd_bound'] == .98
        env.bound = .61
        env.head_window.extend([1.2] * env.head_window.maxlen)
        assert env.reset(seed=3)[1]['pd_bound'] == .6
        env.head_window.extend([1.2] * env.head_window.maxlen)
        assert env.reset(seed=4)[1]['pd_bound'] == .6  # not below the lowest
        assert env.training_state() == {'assist_force': [140.], 'pd_bound': [.6]}
        for option, bound in (('min', .6), (.8, .8), (None, None)):
            assert env.reset(seed=5, options={'pd_bound': option})[1]['pd_bound'] == bound
        for bad in (0., -.5, True, 'max', float('nan')):
            with pytest.raises(ValueError):
                env.reset(seed=5, options={'pd_bound': bad})
        state = {'assist_force': [0., 0., 0., 0.], 'pd_bound': [1., .8, .6, .7]}
        env.restore_training_state(state, 1, 4)
        assert (env.assist, env.bound) == (0., .8)
        env.restore_training_state(state, 0, 2)  # quantiles of the sorted bounds, as the assist forces
        assert env.bound == .7
        env.restore_training_state({'assist_force': [0.], 'pd_bound': [.3]})  # within the setting's [lowest, start]
        assert env.bound == .6
        env.bound = .9
        env.restore_training_state({'assist_force': [0.]})  # a stage without bounds (carried over): kept
        assert env.bound == .9
        with pytest.raises(ValueError):
            env.restore_training_state({'assist_force': [0.], 'pd_bound': [-1.]})
    finally:
        env.close()
    plain = G1GetupEnv(bank=small_bank)
    try:
        plain.reset(seed=0)
        plain.restore_training_state({'assist_force': [50.], 'pd_bound': [.7]})
        assert plain.training_state() == {'assist_force': [50.]} and plain.bound is None
    finally:
        plain.close()
    vec = DummyVecEnv([lambda: G1GetupEnv(bank=small_bank, pd_bound=[.9, .5])] * 2)
    try:
        assert vec_training_state(vec) == {'assist_force': [200., 200.], 'pd_bound': [.9, .9]}
    finally:
        vec.close()


def test_without_the_setting_the_step_is_the_old_one(small_bank, monkeypatch):  # noqa: F811
    """The bound's functions never run without the setting (getting up and walking), and a task with the setting but
    the bound switched off steps exactly as one without it."""
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

    def refuse(*args, **kwargs):
        raise AssertionError('pd_bound used without the setting')
    rng = np.random.default_rng(3)
    actions = rng.uniform(-10, 10, (12, 29))
    trails = []
    for settings, options in (({}, {}), ({'pd_bound': BOUND}, {'pd_bound': None})):
        with monkeypatch.context() as patch:
            patch.setattr(task, 'pd_bound', refuse)
            patch.setattr(task, 'pd_bound_update', refuse)
            env = G1GetupEnv(bank=small_bank, **settings)
            try:
                trail = [env.reset(seed=7, options=dict(OPTIONS, action_delay_steps=5, **options))[0]]
                env.head_window.extend([1.2] * env.head_window.maxlen)
                trail += [env.reset(seed=8, options=dict(OPTIONS, action_delay_steps=5))[0]]
                trail += [env.step(action)[0] for action in actions]
                trails.append(np.array(trail))
            finally:
                env.close()
    np.testing.assert_array_equal(trails[0], trails[1])
    with monkeypatch.context() as patch:
        patch.setattr(task, 'pd_bound', refuse)
        walking = G1VelocityEnv(real_robot=True, history=5, phase_stop=True, lock_wrists=True, action_limit=10.,
                                action_delay=.02)
        try:
            walking.reset(seed=0)
            for action in actions[:3]:
                walking.step(action)
        finally:
            walking.close()


def test_evaluation_resets_at_the_lowest_bound(small_bank):  # noqa: F811
    from humanoid_sim.tasks.catalog import resolve_task
    options = dict(resolve_task('g1_29dof', 'g1_getup').evaluation_options)
    for settings, bound in (({'pd_bound': BOUND}, .6), ({}, None)):
        env = G1GetupEnv(bank=small_bank, **settings)
        try:
            env.reset(seed=0)
            assert env.reset(seed=1, options=options)[1]['pd_bound'] == bound
        finally:
            env.close()


def getup_run(tmp_path, name, **task_config):
    run = tmp_path / name
    run.mkdir()
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', n_steps=8, batch_size=8,
                                  task_config=task_config))
    write_json(run / 'run_config.json', config)
    return run, config


def getup_stage(tmp_path, small_bank, name='old', training_state=None, gain=None, **task_config):  # noqa: F811
    """An untrained g1_getup stage (32×32 actor; gain: its action head multiplied, so the box and the bound act)."""
    run, config = getup_run(tmp_path, name, **task_config)
    env = VecNormalize(DummyVecEnv([lambda: G1GetupEnv(bank=small_bank, **config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        columns, mean, variance = env.venv.envs[0].fixed_normalization
        env.obs_rms.mean[columns], env.obs_rms.var[columns] = mean, variance
        if gain is not None:
            with torch.no_grad():
                model.policy.action_net.weight.mul_(gain)
        return save_stage(run, model, env, training_state)
    finally:
        env.close()


def test_carry_over_moves_a_getup_stage_to_pd_bound_with_its_training_state(tmp_path, small_bank,  # noqa: F811
                                                                             monkeypatch):
    """contract_branch.py's path for root 2: check_carry_over and the environment accept pd_bound, carry_over keeps
    the weights, the normalization and the stage's training_state (its worlds' assist forces: a stage without them
    would start a branch at 200 N again); the bound itself starts fresh (no pd_bound in the source)."""
    from humanoid_sim.core import load_stage
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.warm_start import carry_over, check_carry_over
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    state = {'assist_force': [0., 20., 0., 40.]}
    stage = getup_stage(tmp_path, small_bank, training_state=state)
    run, config = getup_run(tmp_path, 'bounded', pd_bound=BOUND)
    saved = read_json(stage / 'metadata.json')
    check_carry_over(saved, task_identity('g1_29dof', 'g1_getup', config['task_config']))
    make_task('g1_getup', 'g1_29dof', config['task_config']).close()
    moved = carry_over(stage, run)
    metadata = read_json(moved / 'metadata.json')
    assert metadata['action_schema'] == 'g1_getup:action-v2-limit10-wristlock-knee0.061-pdbound0.6'
    assert metadata['training_state'] == state and metadata['task_config']['pd_bound'] == BOUND
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        raw = new_norm.venv.envs[0].unwrapped
        assert raw.pd_bound == BOUND and old_norm.venv.envs[0].unwrapped.pd_bound is None
        raw.restore_training_state(metadata['training_state'], 3, 4)
        assert (raw.assist, raw.bound) == (40., 1.)
    finally:
        old_norm.close(), new_norm.close()
    with pytest.raises(ValueError):  # carried over already: the same contract
        carry_over(moved, run)


def test_a_pd_bound_stage_exports_the_bound_into_the_graph(tmp_path, small_bank, monkeypatch):  # noqa: F811
    """export_g1: after the ±10 clip the graph bounds the action around the newest joint_pos_rel of its raw input at
    the lowest c (Max / Min), before the wrist zeros, and onnxruntime gives what the environment applies at that c on
    observations lying; properties and the deploy.yaml comment say so. Refused before anything is written while a
    world of the stage is above the lowest c (or the stage keeps none); the plain getting-up graph has no Max / Min."""
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from humanoid_sim.export_g1 import ExportedPolicy, check, export, onnx_properties
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    k = env_constants()
    down = getup_stage(tmp_path, small_bank, 'down', {'assist_force': [0., 0.], 'pd_bound': [.6, .6000000238418579]},
                       gain=2000., pd_bound=BOUND)
    result = export(down, tmp_path / 'out')
    properties = onnx_properties(result['onnx'])
    assert properties['action_schema'].endswith('-pdbound0.6') and properties['pd_bound'] == '0.6'
    assert properties['pd_bound_inputs'] == '[171, 200]'
    error = json.loads(properties['pd_bound_rad'])
    assert [e is None for e in error] == (~k['free']).tolist()  # the locked wrists are not bounded
    np.testing.assert_allclose([e for e in error if e is not None], (.6 * k['peak'] / k['kp'])[k['free']], rtol=1e-12)
    ops = {node.op_type for node in onnx.load(str(result['onnx'])).graph.node}
    assert {'Max', 'Min', 'Where'} <= ops
    text = result['deploy'].read_text(encoding='utf-8')
    assert '# pd_bound: policy.onnx bounds each action' in text and 'obs[171:200]' in text
    verdict = check(down, result['onnx'], samples=20, stride=2)
    assert verdict['ok'] and verdict['pd_bound'] == .6 and verdict['max_abs_diff'] < 1e-4
    exported = ExportedPolicy(result['onnx'].parent)
    env = G1GetupEnv(pd_bound=BOUND)
    try:
        obs, _ = env.reset(seed=0, options=dict(OPTIONS, start=0, pd_bound='min'))
        action = exported.predict(obs)[0].astype(np.float64)
        offset = np.abs(k['scale'] * action - obs[171:200])[k['free']]
        assert (offset <= (.6 * k['peak'] / k['kp'])[k['free']] + 1e-5).all() and not action[~k['free']].any()
        # The untrained ×2000 head saturates the box; the bound pulls most joints in.
        assert (np.abs(action[k['free']]) < 9.99).mean() > .5
        np.testing.assert_allclose(env.applied_action(action), action, atol=1e-5)  # the environment changes nothing
    finally:
        env.close()
    for name, state in (('up', {'assist_force': [0., 0.], 'pd_bound': [.6, .62]}), ('none', None)):
        stage = getup_stage(tmp_path, small_bank, name, state, pd_bound=BOUND)
        with pytest.raises(ValueError, match='не дошла до минимума'):
            export(stage, tmp_path / f'out-{name}')
        assert not (tmp_path / f'out-{name}').exists()
    plain = getup_stage(tmp_path, small_bank, 'plain', gain=2000.)
    result = export(plain, tmp_path / 'out-plain')
    assert not {'Max', 'Min'} & {node.op_type for node in onnx.load(str(result['onnx'])).graph.node}
    assert 'pd_bound' not in onnx_properties(result['onnx']) and '# pd_bound:' not in result['deploy'].read_text(
        encoding='utf-8')
    assert math.isfinite(check(plain, result['onnx'], samples=5, stride=1)['max_abs_diff'])
