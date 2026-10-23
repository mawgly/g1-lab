from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

PROFILES = ({}, dict(real_robot=True, action_limit=10., phase_stop=True, history=5))
KNEES = [3, 9]  # left/right knee, on both G1 models
OTHERS = [i for i in range(29) if i not in KNEES]
FLOOR = .061  # iteration 18: Unitree's soft knee limit (0.9 of the range)
# Candidate 3 (exp-6986f1d9544b4d00 @ 5481.4 M, lock_wrists): a real stage the knee_floor contract may start from.
CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-6986f1d9544b4d00/stage_005481431040'


def knee_mask(nu=29):
    mask = torch.zeros(nu, dtype=torch.bool)
    mask[KNEES] = True
    return mask


def below_floor(env, size=1.):
    """An action asking both knees `size` of the box below home (the target under the floor), small elsewhere."""
    action = .3 * np.sin(np.arange(env.model.nu) + 1.)
    action[KNEES] = -size * env.action_limit
    return action


def observed_previous_actions(env, obs):
    """Every observed previous-action frame, oldest first (the deploy history's last_action term, or the single frame
    before the gait phase in the simulation profile)."""
    nu = env.model.nu
    if not env.real_robot:
        return obs[None, -nu - 2:-2]
    start, end = task.deploy_terms(nu)[-1]
    return obs[env.history * start:env.history * end].reshape(env.history, nu)


def test_target_low_raises_only_the_knees_and_none_is_the_range_itself():
    mask = knee_mask()
    for dtype in (torch.float32, torch.float64):
        low = torch.linspace(-3., -.05, 29, dtype=dtype)
        assert task.target_low(low, mask) is low and task.target_low(low, mask, None) is low
        raised = task.target_low(low, mask, FLOOR)
        assert raised.dtype == dtype and torch.equal(raised[OTHERS], low[OTHERS])
        assert bool((raised[KNEES] == torch.tensor(FLOOR, dtype=dtype)).all())
        low[3] = .2  # a range already above the floor keeps its own bound
        assert float(task.target_low(low, mask, FLOOR)[3]) == pytest.approx(.2)
        # pd_target within it: the knee target is max(target, floor), whatever the action; the rest unchanged.
        home, scale = torch.full((29,), .3, dtype=dtype), torch.full((29,), .35, dtype=dtype)
        high = torch.full((29,), 2.88, dtype=dtype)
        action = torch.rand((64, 29), dtype=dtype) * 20 - 10
        floored = task.pd_target(action, home, scale, task.target_low(low, mask, FLOOR), high, 10.)
        plain = task.pd_target(action, home, scale, low, high, 10.)
        torch.testing.assert_close(floored[:, OTHERS], plain[:, OTHERS], rtol=0, atol=0)
        torch.testing.assert_close(floored[:, 9], plain[:, 9].clamp(min=FLOOR), rtol=0, atol=0)


def test_the_knees_are_picked_by_name_on_both_models_and_the_floor_is_unitrees_soft_limit():
    for legs_only in (False, True):
        env = G1VelocityEnv(legs_only=legs_only, knee_floor=FLOOR)
        try:
            names = [env.model.joint(j).name for j in env.joints]
            assert env.knee_joints == tuple(KNEES) and env.knee_mask.nonzero().flatten().tolist() == KNEES
            assert [names[i] for i in KNEES] == ['left_knee_joint', 'right_knee_joint']
            low, high = (env.model.jnt_range[env.joints, k] for k in (0, 1))
            soft = (low + high) / 2 - .9 * (high - low) / 2  # unitree_rl_mjlab joint_pos_limits, soft_ratio 0.9
            assert np.abs(soft[KNEES] - FLOOR).max() < 1e-3
            others = [i for i in range(env.model.nu) if i not in KNEES]
            assert (env.target_low[KNEES] == FLOOR).all() and np.array_equal(env.target_low[others], low[others])
            assert np.array_equal(env.target_high, high)
        finally:
            env.close()
    for bad in (3., -.1, float('nan')):  # outside the knee range [-0.087, 2.88], or not a number
        with pytest.raises(ValueError):
            G1VelocityEnv(knee_floor=bad)


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_the_target_and_the_step_are_bitwise_the_original(profile):
    """No floor: the PD target bounds are the joint ranges themselves and the step does not change."""
    plain, off = G1VelocityEnv(**profile), G1VelocityEnv(knee_floor=None, **profile)
    try:
        assert off.knee_floor is None and off.knee_joints == tuple(KNEES)
        low, high = (off.model.jnt_range[off.joints, k] for k in (0, 1))
        assert np.array_equal(off.target_low, low) and np.array_equal(off.target_high, high)
        np.testing.assert_array_equal(plain.reset(seed=5)[0], off.reset(seed=5)[0])
        limit = plain.action_limit
        for action in np.random.default_rng(2).uniform(-2 * limit, 2 * limit, (10, plain.model.nu)):
            clipped = np.clip(action, -limit, limit)
            old = task.pd_target(torch.from_numpy(clipped), torch.from_numpy(off.home), torch.from_numpy(off.scale),
                                 torch.from_numpy(low), torch.from_numpy(high), limit).numpy()
            np.testing.assert_array_equal(off._target(clipped), old)  # the target as written before the setting
            expected, result = plain.step(action), off.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1] and result[4]['reward_components'] == expected[4]['reward_components']
            np.testing.assert_array_equal(off.data.qpos, plain.data.qpos)
    finally:
        plain.close(), off.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_the_knee_target_is_never_below_the_floor_whatever_the_action(profile):
    floored, plain = G1VelocityEnv(knee_floor=FLOOR, **profile), G1VelocityEnv(**profile)
    try:
        limit = floored.action_limit
        actions = np.random.default_rng(3).uniform(-2 * limit, 2 * limit, (200, floored.model.nu))
        asked_lower = 0
        for action in actions:
            applied = floored.applied_action(action)
            np.testing.assert_array_equal(applied, plain.applied_action(action))  # the action is not raised
            target, unfloored = floored._target(applied), plain._target(applied)
            assert (target[KNEES] >= FLOOR).all()
            np.testing.assert_array_equal(target[KNEES], np.maximum(unfloored[KNEES], FLOOR))
            np.testing.assert_array_equal(target[OTHERS], unfloored[OTHERS])
            asked_lower += int((unfloored[KNEES] < FLOOR).sum())
        assert asked_lower > 50
    finally:
        floored.close(), plain.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_the_floor_reaches_the_pd_the_delay_queue_and_arm_press_but_not_the_observed_previous_action(profile,
                                                                                                    monkeypatch):
    """Knees asked below the floor, motors 15 substeps late: every target step computes (this step's and the delayed
    ones) and the one arm_press reads has its knees at the floor, while the observed previous action, history
    included, and the delay queue keep the policy's knee action, as g1_ctrl's last_action (the raw ONNX output)."""
    weights = {'arm_press': .1, 'action_change': .05}
    env = G1VelocityEnv(reward_weights=weights, action_delay=.03, knee_floor=FLOOR, **profile)
    pressed, original = [], task.arm_press

    def arm_press(target, position, arm_joints):
        pressed.append(target.clone())
        return original(target, position, arm_joints)

    monkeypatch.setattr(task, 'arm_press', arm_press)
    targets, own = [], env._target

    def recorded(action):
        targets.append(own(action))
        return targets[-1]

    monkeypatch.setattr(env, '_target', recorded)
    try:
        env.reset(seed=6, options={'command': [.5, 0., 0.], 'action_delay_steps': 15})
        limit, bound = env.action_limit, (FLOOR - env.home[KNEES]) / env.scale[KNEES]
        for step in range(6):
            action = below_floor(env)
            targets.clear()
            obs = env.step(action)[0]
            assert len(targets) == 1 + len(env.delay_actions) == 3  # this step's target and two older ones
            assert (targets[0][KNEES] == FLOOR).all() and all((t[KNEES] >= FLOOR).all() for t in targets)
            # The older actions in flight asked below the floor too, once there are any (at the start: zeros, home).
            assert sum(bool((t[KNEES] == FLOOR).all()) for t in targets[1:]) == min(step, 2)
            np.testing.assert_array_equal(pressed[-1][0].numpy(), own(np.clip(action, -limit, limit)))
            np.testing.assert_array_equal(env.previous_action, np.clip(action, -limit, limit))
            assert (env.previous_action[KNEES] < bound).all()  # the policy's knee action, not raised
            assert (observed_previous_actions(env, obs)[-1][KNEES] < bound).all()
            assert all((past[KNEES] < bound).all() for past in env.delay_actions[:min(step + 1, 2)])
    finally:
        env.close()


def test_floored_knees_stay_out_of_the_stop_whatever_the_policy_asks():
    """Standing, both knees asked for the bottom of the box (target −3.2 rad, clipped to the stop −0.087): without
    the floor the knees go into the stop, with it they stay at about the floor."""
    lowest = {}
    for floor in (None, FLOOR):
        env = G1VelocityEnv(real_robot=True, action_limit=10., history=5, phase_stop=True, knee_floor=floor)
        try:
            env.reset(seed=3, options={'command': [0., 0., 0.]})
            angles = []
            for _ in range(40):
                action = np.zeros(29)
                action[KNEES] = -10.
                _, _, fell, _, _ = env.step(action)
                angles.append(env.data.qpos[env.qadr][KNEES].min())
                if fell:
                    break
            lowest[floor] = min(angles[10:])
        finally:
            env.close()
    assert lowest[None] < -.05 and lowest[FLOOR] > .03


def test_knee_floor_config_and_its_own_action_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_PARAMETERS
    from humanoid_sim.warm_start import check_carry_over
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['knee_floor'] is None  # old branches keep theirs
    for given, kept in ((FLOOR, FLOOR), (0, 0.), (None, None), (1, 1.)):
        assert validate_config({**base, 'task_config': {'knee_floor': given}})['task_config']['knee_floor'] == kept
    for bad in (True, '0.061', -.01, 1.5, float('nan'), float('inf'), [FLOOR]):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'knee_floor': bad}})
    field = next(p for p in G1_PARAMETERS if p['key'] == 'knee_floor')
    assert field['default'] is None and (field['min'], field['max']) == (0., 1.) and 'колен' in field['label']
    for robot, config, schema in (
            ('g1_29dof', {}, 'g1_velocity:action-v1'),
            ('g1_12dof', {}, 'g1_velocity:action-v1'),
            ('g1_29dof', dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.),
             'g1_velocity:action-v2-limit10'),
            ('g1_29dof', dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.,
                              lock_wrists=True), 'g1_velocity:action-v2-limit10-wristlock')):
        saved = task_identity(robot, 'g1_velocity', config)  # stages saved before the setting existed
        assert saved == task_identity(robot, 'g1_velocity', {**config, 'knee_floor': None})
        assert saved['action_schema'] == schema
        floored = task_identity(robot, 'g1_velocity', {**config, 'knee_floor': FLOOR})
        assert floored == {**saved, 'action_schema': f'{schema}-knee0.061'}  # only the action contract changes
        with pytest.raises(ValueError):
            validate_resume(saved, floored)  # an ordinary branch is refused
        check_carry_over(saved, floored)  # carry_over may move the weights
        other = task_identity(robot, 'g1_velocity', {**config, 'knee_floor': .1})
        with pytest.raises(ValueError):
            validate_resume(floored, other)  # another floor is another contract


def knee_floor_run(tmp_path, name, **task_config):
    """A run dir with a validated real-robot config (lock_wrists, as candidate 3) plus task_config."""
    from humanoid_sim.core import write_json
    from humanoid_sim.lab import validate_config
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config={**dict(real_robot=True, critic_extras=True, action_limit=10.,
                                                      history=5, phase_stop=True, lock_wrists=True), **task_config}))
    run = tmp_path / name
    run.mkdir()
    write_json(run / 'run_config.json', config)
    return run, config


def stub_stage(tmp_path, name='old', bias=None, **task_config):
    """An untrained stage of knee_floor_run's contract; bias: the knee outputs' bias (below the floor: −3)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import save_stage
    from humanoid_sim.policies import CriticExtrasPolicy
    run, config = knee_floor_run(tmp_path, name, **task_config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        if bias is not None:
            with torch.no_grad():
                model.policy.action_net.bias[KNEES] = bias
        model.learn(16)  # optimizer moments and normalization statistics that are not the defaults
        return save_stage(run, model, env)
    finally:
        env.close()


def assert_carried_over(source, moved):
    """moved: source under the knee_floor contract, weights and normalization unchanged, loading with the floor."""
    from humanoid_sim.core import load_stage, read_json
    saved, metadata = read_json(source / 'metadata.json'), read_json(moved / 'metadata.json')
    assert metadata['action_schema'] == f"{saved['action_schema']}-knee0.061"
    assert {k: v for k, v in metadata.items() if k not in ('action_schema', 'task_config', 'created_at')} == \
        {k: v for k, v in saved.items() if k not in ('action_schema', 'task_config', 'created_at')}
    warm = read_json(moved / 'warm_start.json')
    assert (warm['from_action_schema'], warm['to_action_schema']) == (saved['action_schema'], metadata['action_schema'])
    (before, old_norm), (after, new_norm) = load_stage(source), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        assert new_norm.venv.envs[0].unwrapped.knee_floor == FLOOR
        assert old_norm.venv.envs[0].unwrapped.knee_floor is None
    finally:
        old_norm.close(), new_norm.close()


def test_carry_over_moves_a_stage_to_the_knee_floor_contract_as_contract_branch_does(tmp_path):
    """contract_branch.py's checks (check_carry_over, make_task) pass for knee_floor, then carry_over keeps the weights;
    the moved stage carried again, or carried to another box, is refused."""
    from humanoid_sim.core import read_json
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.warm_start import carry_over, check_carry_over
    stage = stub_stage(tmp_path)
    run, config = knee_floor_run(tmp_path, 'floored', knee_floor=FLOOR)
    check_carry_over(read_json(stage / 'metadata.json'), task_identity('g1_29dof', 'g1_velocity', config['task_config']))
    make_task('g1_velocity', 'g1_29dof', config['task_config']).close()
    moved = carry_over(stage, run)
    assert read_json(moved / 'metadata.json')['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock-knee0.061'
    assert_carried_over(stage, moved)
    with pytest.raises(ValueError):  # carried over already: the same contract, nothing to carry
        carry_over(moved, run)
    box, _ = knee_floor_run(tmp_path, 'box', knee_floor=FLOOR, action_limit=4.)
    with pytest.raises(ValueError):
        carry_over(stage, box)
    assert not any(box.glob('stage_*'))


def test_candidate_3_carries_over_to_knee_floor(tmp_path):
    """The real stage the iteration-18 branch may start from: carried over with its weights unchanged."""
    from humanoid_sim.core import read_json, write_json
    from humanoid_sim.warm_start import carry_over
    if not (CANDIDATE / 'metadata.json').is_file():
        pytest.skip('candidate 3 is not on this machine')
    saved = read_json(CANDIDATE / 'metadata.json')
    parent = read_json(CANDIDATE.parent / 'run_config.json')
    write_json(tmp_path / 'run_config.json', {**parent, 'source_id': None, 'source_stage': None,
                                              'task_config': {**saved['task_config'], 'knee_floor': FLOOR}})
    assert_carried_over(CANDIDATE, carry_over(CANDIDATE, tmp_path))


def test_probes_with_their_own_pd_floor_the_knee_target_as_env_step(tmp_path, monkeypatch):
    """um_probe.Stepper (self_contact.py, bridge_kick.py run on it) and robust_probe.Stepper redo the step's PD
    themselves within the environment's target bounds: on a knee_floor stage whose policy asks the knees below the
    floor they give env.step's PD targets (knees at the floor), observed last action (the policy's knee action),
    state and observation, step for step."""
    from humanoid_sim.core import load_stage, stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    from test_g1_lock_wrists import lab_probe
    stage = stub_stage(tmp_path, bias=-3., knee_floor=FLOOR)
    um_probe, robust_probe = lab_probe('um_probe', monkeypatch), lab_probe('robust_probe', monkeypatch)

    class Recorded(um_probe.Stepper):
        def __init__(self, raw):
            super().__init__(raw)
            self.targets = []

        def substep(self, target):
            self.targets.append(target.copy())
            super().substep(target)

    model, norm = load_stage(stage)
    runs = {}
    try:
        config = {**stage_task_config(stage), 'heading_hold': False}
        for name in ('env', 'um_probe', 'robust_probe'):
            env = make_task('g1_velocity', 'g1_29dof', config)
            raw = env.unwrapped
            assert raw.knee_floor == FLOOR
            obs, _ = env.reset(seed=73000, options={'command': [.5, 0., 0.], 'phase_steps': 0})
            stepper = Recorded(raw) if name == 'um_probe' else robust_probe.Stepper(raw, 0)
            rows = []
            for _ in range(20):
                action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
                if name == 'env':
                    obs = env.step(action)[0]
                    target = raw._target(raw.previous_action)
                elif name == 'um_probe':
                    stepper.policy_step(action)
                    obs = raw._observation('push')
                    target = stepper.targets[-1]
                else:
                    stepper.step(action)
                    obs = raw._observation('push')
                    target = stepper.sub[-1]
                rows.append(dict(asked=np.clip(action, -10, 10), target=np.array(target),
                                 previous=raw.previous_action.copy(), qpos=raw.data.qpos.copy(),
                                 qvel=raw.data.qvel.copy(), obs=obs.copy()))
            bound = (FLOOR - raw.home[KNEES]) / raw.scale[KNEES]
            env.close()
            runs[name] = rows
    finally:
        norm.close()
    expected = runs['env']
    assert all((row['asked'][KNEES] < bound).all() for row in expected)  # the policy asks below the floor
    for row in expected:
        np.testing.assert_array_equal(row['previous'][KNEES], row['asked'][KNEES])
        assert (row['target'][KNEES] == FLOOR).all()
    for name in ('um_probe', 'robust_probe'):
        for step, (got, want) in enumerate(zip(runs[name], expected)):
            for key in ('target', 'previous', 'qpos', 'qvel', 'obs'):
                np.testing.assert_array_equal(got[key], want[key], err_msg=f'{name}, step {step}: {key}')
