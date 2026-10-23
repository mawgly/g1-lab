import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

PROFILES = ({}, dict(real_robot=True, action_limit=10., phase_stop=True, history=5))
WRISTS = [20, 21, 27, 28]  # left/right wrist pitch and yaw
OTHERS = [i for i in range(29) if i not in WRISTS]
ARMS = list(range(15, 29))
# Iteration 14 (arm_perturb 0.1 from candidate 2): the stage the first lock_wrists branch starts from.
PARENT = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-14bae200a11f4856'
PROBES = Path(__file__).resolve().parents[1] / 'scripts' / 'lab' / 'probes'


def actions(count, nu, size, seed=0):
    return np.random.default_rng(seed).uniform(-size, size, (count, nu))


def observed_previous_actions(env, obs):
    """Every observed previous-action frame, oldest first: the deploy history's last_action term (the last term, h
    frames of nu), or the single frame before the gait phase in the simulation profile."""
    nu = env.model.nu
    if not env.real_robot:
        return obs[None, -nu - 2:-2]
    start, end = task.deploy_terms(nu)[-1]
    return obs[env.history * start:env.history * end].reshape(env.history, nu)


def test_lock_joints_zeroes_only_the_locked_entries():
    locked = torch.zeros(29, dtype=torch.bool)
    locked[WRISTS] = True
    for dtype in (torch.float32, torch.float64):
        action = torch.rand((3, 29), dtype=dtype) * 20 - 10
        kept = action.clone()
        applied = task.lock_joints(action, locked)
        assert applied.dtype == dtype and applied.shape == (3, 29)
        assert bool((applied[:, WRISTS] == 0).all()) and torch.equal(applied[:, OTHERS], kept[:, OTHERS])
        assert torch.equal(action, kept)  # not in place: the policy's action stays for PPO


def test_the_locked_joints_are_the_wrists_whose_box_lies_inside_the_joint_range():
    from humanoid_sim.symmetry import Mirror
    env = G1VelocityEnv(real_robot=True, action_limit=10., lock_wrists=True)
    try:
        names = [env.model.joint(j).name for j in env.joints]
        assert env.wrist_joints == tuple(WRISTS) and env.wrist_mask.nonzero().flatten().tolist() == WRISTS
        assert [names[i] for i in WRISTS] == ['left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                                              'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        low, high = (env.model.jnt_range[env.joints, k] - env.home for k in (0, 1))
        box = env.scale * env.action_limit
        assert np.flatnonzero((box < high) & (-box > low)).tolist() == WRISTS  # ±0.70 inside ±1.61; elsewhere wider
        assert np.allclose(env.home[WRISTS], 0)  # default_joint_pos: g1_ctrl's target for a 0 action
        # The mirror loss swaps the locked joints among themselves: a locked action mirrors to a locked action.
        mirror = Mirror(env)
        assert sorted(mirror.action_index[WRISTS].tolist()) == WRISTS
        action = torch.rand((4, 29)) * 20 - 10
        torch.testing.assert_close(mirror.action(task.lock_joints(action, env.wrist_mask)),
                                   task.lock_joints(mirror.action(action), env.wrist_mask), rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_the_step_is_bitwise_the_original(profile):
    plain, off = G1VelocityEnv(**profile), G1VelocityEnv(lock_wrists=False, **profile)
    try:
        assert not off.lock_wrists and off.wrist_joints == tuple(WRISTS)
        np.testing.assert_array_equal(plain.reset(seed=5)[0], off.reset(seed=5)[0])
        limit = plain.action_limit
        for action in actions(10, plain.model.nu, 2 * limit, seed=2):
            expected, result = plain.step(action), off.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1] and result[4]['reward_components'] == expected[4]['reward_components']
            np.testing.assert_array_equal(off.data.qpos, plain.data.qpos)
            np.testing.assert_array_equal(off.previous_action, np.clip(action, -limit, limit))  # wrists included
    finally:
        plain.close(), off.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_locked_wrists_get_action_zero_in_physics_reward_delay_and_observation(profile):
    """A locked robot given any action moves, observes and is rewarded exactly as an unlocked one given the applied
    action (the policy's, wrist pitch and yaw at 0), also under a 15-substep delay and inside an arm perturbation
    segment whose held action bends the wrists: the lock comes after the perturbation."""
    weights = {'arm_press': .1, 'action_change': .05}
    env = G1VelocityEnv(reward_weights=weights, action_delay=.03, arm_perturb=.1, lock_wrists=True, **profile)
    twin = G1VelocityEnv(reward_weights=weights, action_delay=.03, **profile)
    try:
        nu, limit = env.model.nu, env.action_limit
        options = {'command': [.5, 0., 0.], 'action_delay_steps': 15}
        np.testing.assert_array_equal(env.reset(seed=6, options=options)[0], twin.reset(seed=6, options=options)[0])
        held = np.zeros(nu)
        held[ARMS] = np.random.default_rng(9).uniform(-.8, .8, len(ARMS))  # wrists bent too
        env.set_arm_perturb(3, held)
        for step, action in enumerate(actions(8, nu, 2 * limit, seed=4)):  # beyond the box: clipped, then locked
            applied = np.clip(action, -limit, limit)
            if step < 3:
                applied[ARMS] = held[ARMS]
            applied[WRISTS] = 0.
            obs, reward, _, _, info = env.step(action)
            expected, expected_reward, _, _, expected_info = twin.step(applied)
            np.testing.assert_array_equal(env.data.qpos, twin.data.qpos)
            np.testing.assert_array_equal(env.data.qvel, twin.data.qvel)
            np.testing.assert_array_equal(obs, expected)
            assert reward == expected_reward and info['reward_components'] == expected_info['reward_components']
            np.testing.assert_array_equal(env.previous_action, applied)
            assert not env.previous_action[WRISTS].any()
            np.testing.assert_array_equal(env.previous_action[OTHERS], applied[OTHERS])
            assert not observed_previous_actions(env, obs)[:, WRISTS].any()  # every history frame
            np.testing.assert_array_equal(env._target(env.previous_action)[WRISTS], env.home[WRISTS])
            assert all(not past[WRISTS].any() for past in env.delay_actions)  # the queue the motors follow
        assert np.abs(np.clip(action, -limit, limit)[WRISTS]).min() > 0  # the policy did ask for a bend
    finally:
        env.close(), twin.close()


def test_locked_wrists_hold_the_home_pose_whatever_the_policy_asks():
    """Standing for 1 s, wrist actions uniform in the whole ±10 box (targets up to ±40° off home): locked, the wrist
    angles stay within ~1° of home (0.016 rad measured), unlocked they follow the targets (0.37 rad)."""
    worst = {}
    for lock in (False, True):
        env = G1VelocityEnv(real_robot=True, action_limit=10., history=5, phase_stop=True, lock_wrists=lock)
        try:
            env.reset(seed=3, options={'command': [0., 0., 0.]})
            rng = np.random.default_rng(0)
            angles = []
            for _ in range(50):
                action = rng.uniform(-.3, .3, 29)
                action[WRISTS] = rng.uniform(-10, 10, 4)
                _, _, fell, _, _ = env.step(action)
                assert not fell
                angles.append(np.abs(env.data.qpos[env.qadr][WRISTS] - env.home[WRISTS]).max())
            worst[lock] = max(angles)
        finally:
            env.close()
    assert worst[True] < .05 and worst[False] > .2


def test_training_episodes_lock_the_wrists_under_arm_perturbation():
    env = G1VelocityEnv(max_episode_steps=10 ** 6, real_robot=True, action_limit=10., arm_perturb=.5,
                        lock_wrists=True)
    try:
        env.reset(seed=1)
        assert env.perturb_live
        rng = np.random.default_rng(0)
        segments = 0
        for _ in range(600):  # the robot falls; the lock does not depend on it
            action = rng.uniform(-10, 10, env.model.nu)
            env.step(action)
            assert not env.previous_action[WRISTS].any()
            segments += env.perturb_steps > 0
        assert segments > 0
    finally:
        env.close()


def test_the_legs_only_g1_rejects_lock_wrists():
    """No wrists there: the setting would only rename the action contract, so it is refused."""
    with pytest.raises(ValueError):
        G1VelocityEnv(legs_only=True, lock_wrists=True)
    env = G1VelocityEnv(legs_only=True)
    try:
        assert env.wrist_joints == () and not env.wrist_mask.any()
    finally:
        env.close()


def test_lock_wrists_config_and_its_own_action_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_PARAMETERS
    from humanoid_sim.warm_start import check_carry_over
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['lock_wrists'] is False  # old branches keep theirs
    assert validate_config({**base, 'task_config': {'lock_wrists': True}})['task_config']['lock_wrists'] is True
    for bad in (1, 'true', None, 0.):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'lock_wrists': bad}})
    field = next(p for p in G1_PARAMETERS if p['key'] == 'lock_wrists')
    assert field['type'] == 'bool' and field['default'] is False and 'запяст' in field['label']
    for config, schema in (({}, 'g1_velocity:action-v1'),
                           (dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.),
                            'g1_velocity:action-v2-limit10')):
        saved = task_identity('g1_29dof', 'g1_velocity', config)  # stages saved before the setting existed
        assert saved == task_identity('g1_29dof', 'g1_velocity', {**config, 'lock_wrists': False})
        assert saved['action_schema'] == schema
        locked = task_identity('g1_29dof', 'g1_velocity', {**config, 'lock_wrists': True})
        assert locked == {**saved, 'action_schema': f'{schema}-wristlock'}  # only the action contract changes
        with pytest.raises(ValueError):
            validate_resume(saved, locked)  # an ordinary branch is refused
        check_carry_over(saved, locked)  # carry_over may move the weights


def test_the_iteration_14_stage_carries_over_to_lock_wrists(tmp_path):
    """The real parent of the first lock_wrists branch: carried over with its weights and normalization unchanged,
    the stage gets the new action contract and loads with the lock on."""
    from humanoid_sim.core import list_stages, load_stage, read_json, write_json
    from humanoid_sim.warm_start import carry_over
    stages = list_stages(PARENT) if PARENT.is_dir() else []
    if not stages:
        pytest.skip('iteration 14 stages are not on this machine')
    source = stages[-1]
    saved = read_json(source / 'metadata.json')
    if saved['task_config'].get('lock_wrists'):
        pytest.skip('the stage already has lock_wrists')
    parent = read_json(PARENT / 'run_config.json')
    write_json(tmp_path / 'run_config.json', {**parent, 'source_id': None, 'source_stage': None,
                                              'task_config': {**parent['task_config'], 'lock_wrists': True}})
    moved = carry_over(source, tmp_path)
    metadata = read_json(moved / 'metadata.json')
    assert metadata['action_schema'] == f"{saved['action_schema']}-wristlock"
    assert {k: v for k, v in metadata.items() if k not in ('action_schema', 'task_config', 'created_at')} == \
        {k: v for k, v in saved.items() if k not in ('action_schema', 'task_config', 'created_at')}
    assert read_json(moved / 'warm_start.json')['to_action_schema'] == metadata['action_schema']
    (before, old_norm), (after, new_norm) = load_stage(source), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        assert new_norm.venv.envs[0].unwrapped.lock_wrists and not old_norm.venv.envs[0].unwrapped.lock_wrists
    finally:
        old_norm.close(), new_norm.close()
    with pytest.raises(ValueError):  # carried over already: the same contract, nothing to carry
        carry_over(moved, tmp_path)


def lock_wrists_stage(tmp_path):
    """A stub real-robot stage with lock_wrists: an untrained policy whose outputs are scaled up, so its wrist actions
    ask for bends the lock has to take away."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import save_stage, write_json
    from humanoid_sim.lab import validate_config
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=dict(real_robot=True, action_limit=10., phase_stop=True,
                                                   lock_wrists=True)))
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32]})
        with torch.no_grad():
            model.policy.action_net.weight.mul_(100)
            model.policy.action_net.weight[WRISTS] *= 10  # wrist outputs at several units, ~10-40° off home
        return save_stage(run, model, env)
    finally:
        env.close()


def lab_probe(name, monkeypatch):
    """A module of scripts/lab/probes. um_probe hides CUDA from its process on import: the variable is put back after
    the test, so a GPU suite run in the same process keeps its device."""
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    return importlib.import_module(name)


def test_probes_with_their_own_pd_apply_what_env_step_applies_on_a_lock_wrists_stage(tmp_path, monkeypatch):
    """um_probe.Stepper (self_contact.py runs on it) and robust_probe.Stepper redo the step's PD themselves. On a
    lock_wrists stage they once drove the wrists with the policy's raw output (candidate 3: wrist pitch +11…+28°,
    um_probe selftest qpos 0.62 off). Through G1VelocityEnv.applied_action they give env.step's PD targets (wrist pitch
    and yaw at home), observed last action, state and observation, step for step."""
    from humanoid_sim.core import load_stage, stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    stage = lock_wrists_stage(tmp_path)
    um_probe, robust_probe = lab_probe('um_probe', monkeypatch), lab_probe('robust_probe', monkeypatch)

    class Recorded(um_probe.Stepper):
        """um_probe's stepper keeping every substep's PD target."""

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
            assert raw.lock_wrists
            obs, _ = env.reset(seed=73000, options={'command': [.5, 0., 0.], 'phase_steps': 0})
            stepper = Recorded(raw) if name == 'um_probe' else robust_probe.Stepper(raw, 0)
            rows = []
            for _ in range(30):
                action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
                applied = raw.applied_action(action)
                if name == 'env':
                    obs = env.step(action)[0]
                    target = raw._target(raw.previous_action)  # step's own target is that of what it applied
                elif name == 'um_probe':
                    stepper.policy_step(action)
                    obs = raw._observation('push')
                    assert all(np.array_equal(t, stepper.targets[-1]) for t in stepper.targets[-raw.frame_skip:])
                    target = stepper.targets[-1]
                else:
                    stepper.step(action)
                    obs = raw._observation('push')
                    target = stepper.sub[-1]  # delay 0: every substep's target
                rows.append(dict(asked=np.clip(action, -raw.action_limit, raw.action_limit), applied=applied,
                                 target=np.array(target), previous=raw.previous_action.copy(),
                                 qpos=raw.data.qpos.copy(), qvel=raw.data.qvel.copy(), obs=obs.copy()))
            home = raw.home.copy()
            env.close()
            runs[name] = rows
    finally:
        norm.close()
    expected = runs['env']
    assert np.abs(np.array([row['asked'] for row in expected])[:, WRISTS]).mean() > 1  # the policy asks for bends
    for row in expected:
        np.testing.assert_array_equal(row['previous'], row['applied'])  # applied_action is what step applies
        assert not row['applied'][WRISTS].any()
        np.testing.assert_array_equal(row['target'][WRISTS], home[WRISTS])
    for name in ('um_probe', 'robust_probe'):
        for step, (got, want) in enumerate(zip(runs[name], expected)):
            for key in ('target', 'previous', 'qpos', 'qvel', 'obs'):
                np.testing.assert_array_equal(got[key], want[key], err_msg=f'{name}, step {step}: {key}')
