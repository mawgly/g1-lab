"""task_config arm_targets (iteration 40), part B: the export for g1_ctrl (policy.onnx adds the newest slot offsets
over the scale to the arm outputs, deploy.yaml's arm_command term with the named poses), its check under arm targets,
ExportedPolicy's residual, and the sim2sim tools: gamepad.py's arm keys, analyze.py's replay of the arm_command term of
scripts/sim2sim/patches/g1_ctrl-arms.patch and hand_load.py's scene with loads in the hands."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.core import load_stage, save_stage, write_json
from humanoid_sim.export_g1 import (ARM_COMMAND_TERM, ARM_RELEASE, NAMED_ARM_POSES, ExportedPolicy, arm_command_inputs,
                                    check, export, named_arm_poses, onnx_properties, reference_action)
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

onnxruntime = pytest.importorskip('onnxruntime')
yaml = pytest.importorskip('yaml')
ROOT = Path(__file__).resolve().parents[1]
SIM2SIM = ROOT / 'scripts' / 'sim2sim'
ARMS = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
            lock_wrists=True, height_command=True, upper_body_slot=True, arm_targets=True,
            reward_weights={'arm_track': 2.})
WRISTS = [20, 21, 27, 28]


def arm_stage(tmp_path, gain=1.):
    """An arm_targets stage with an untrained actor whose weights on the slot are not 0 (so the slot reaches the
    actions inside the network too) and statistics off the defaults outside the fixed columns."""
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=ARMS))
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        size = env.observation_space.shape[0]
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': size - 4})
        positions = env.envs[0].unwrapped.fixed_normalization[0]
        rng = np.random.default_rng(1)
        free = np.setdiff1d(np.arange(size), positions)
        env.obs_rms.mean[free] = rng.normal(0, .3, len(free))
        with torch.no_grad():
            model.policy.action_net.weight.mul_(gain)
        return save_stage(run, model, env)
    finally:
        env.close()


def test_export_adds_the_slot_targets_to_the_arm_outputs_and_names_the_poses(tmp_path):
    """policy.onnx [1, 585]: the clipped, wrist-locked action plus the newest slot offsets d / scale on joints 12-28
    (nothing elsewhere; with the slot zeros exactly the plain output); deploy.yaml names the slot arm_command with the
    named poses (17 offsets each, the waist and the locked wrists 0, mirror images left and right), the release keys,
    the training's quintic; ONNX properties and header say so; check() passes with arm targets in its second half;
    ExportedPolicy hands the stage's environment the residual back."""
    stage = arm_stage(tmp_path, gain=2000.)
    result = export(stage, tmp_path / 'out')
    assert result['actor_inputs'] == 585 and any('g1_ctrl-arms.patch' in note for note in result['warnings'])
    properties = onnx_properties(result['onnx'])
    assert properties['arm_targets'] == 'true' and properties['arm_command_inputs'] == '[568, 585]'
    assert properties['action_schema'].endswith('-wristlock-armres')
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    policy, norm = load_stage(stage)
    try:
        raw = norm.venv.envs[0].unwrapped
        start, end = arm_command_inputs(raw)
        assert (start, end) == (568, 585)
        rng = np.random.default_rng(2)
        obs = rng.normal(0, 1, (20, 589)).astype(np.float32)
        obs[:, start:end] = rng.uniform(-1.5, 1.5, (20, 17))
        obs[:, start - 1] = 1.  # the flag
        obs[:, start:start + 3] = 0.  # the waist's d, as the environment shows it
        for j in WRISTS:
            obs[:, start + j - 12] = 0.
        config = {'action_limit': 10., 'lock_wrists': True}
        expected = np.stack([reference_action(policy, norm, row, config, raw) for row in obs])
        assert np.abs(expected).max() == 10  # the clip acts
        added = expected.copy()
        added[:, 12:] += obs[:, start:end] / raw.scale[12:]
        got = np.concatenate([session.run(['actions'], {'obs': row[None, :585]})[0] for row in obs])
        np.testing.assert_allclose(got, added, atol=1e-4, rtol=0)
        assert not got[:, WRISTS].any() and np.abs(got[:, 12:] - expected[:, 12:]).max() > .5
        zero = obs.copy()
        zero[:, start - 1:end] = 0.
        plain = np.stack([reference_action(policy, norm, row, config, raw) for row in zero])
        np.testing.assert_allclose(np.concatenate([session.run(['actions'], {'obs': row[None, :585]})[0]
                                                   for row in zero]), plain, atol=1e-4, rtol=0)
        exported = ExportedPolicy(result['onnx'].parent)
        np.testing.assert_allclose(exported.predict(obs)[0], expected, atol=2e-4, rtol=0)
    finally:
        norm.close()
    verdict = check(stage, result['onnx'], samples=20, stride=2)
    assert verdict['ok'] and verdict['arm_targets'] == 10, verdict
    text = result['deploy'].read_text(encoding='utf-8')
    deploy = yaml.safe_load(text)
    terms = list(deploy['observations'])
    assert terms[-2:] == ['height_command', ARM_COMMAND_TERM] and 'upper_body_command' not in terms
    params = deploy['observations'][ARM_COMMAND_TERM]['params']
    assert (params['size'], params['first_joint'], params['speed'], params['min_seconds'], params['release']) == (
        18, 12, task.ARM_MOVE_SPEED, task.ARM_MOVE_SECONDS, ARM_RELEASE)
    assert [(p['name'], p['keys']) for p in params['poses']] == [(name, keys) for name, keys, _ in NAMED_ARM_POSES]
    for pose in params['poses']:
        d = np.array(pose['offsets'])
        assert len(d) == 17 and not d[:3].any() and not d[[j - 12 for j in WRISTS]].any()
        assert not any(str(v) == '-0.0' for v in pose['offsets'])
    assert 'g1_ctrl-arms.patch' in text.split('joint_ids_map')[0]


def test_named_poses_lie_within_the_joints_and_the_carry_ranges_are_mirrored_and_touch_nothing():
    """The named poses: within the joint ranges, the carry ones within task.ARM_CARRY, left and right mirror images,
    none touching the body, the other arm or its own arm, in the pose or halfway from home (the libraries' rule,
    scripts/lab/arm_pose_library.py, with Unitree's hand capsules)."""
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
    from arm_pose_library import Touch
    from humanoid_sim.symmetry import model_joint_mirror
    env = G1VelocityEnv(**ARMS)
    touch = Touch()
    try:
        poses = named_arm_poses(env)
        assert [name for name, _, _ in poses] == [name for name, _, _ in NAMED_ARM_POSES]
        permutation, signs = model_joint_mirror(env.model, env.joints, env.qadr, env.home)
        names = [env.model.joint(j).name for j in env.joints]
        carry = {part: (low, high) for part, low, high in task.ARM_CARRY}
        for name, keys, q in poses:
            np.testing.assert_allclose(q[permutation] * signs, q, atol=1e-12)  # its own mirror image
            np.testing.assert_array_equal(q[:15], env.home[:15])
            assert not (q - env.home)[WRISTS].any()
            if name.startswith('carry') or name == 'extended':
                for part, (low, high) in carry.items():
                    assert low - 1e-9 <= q[names.index(f'left_{part}_joint')] <= high + 1e-9, (name, part)
            assert touch.keeps(q[touch.free]), name
    finally:
        env.close(), touch.close()


def test_gamepad_reads_the_poses_keys_of_an_export(tmp_path):
    sys.path.insert(0, str(SIM2SIM))
    import gamepad
    stage = arm_stage(tmp_path)
    result = export(stage, tmp_path / 'out')
    poses, release = gamepad.arm_poses(result['deploy'])
    assert [name for name, _ in poses] == [name for name, _, _ in NAMED_ARM_POSES] and release == ARM_RELEASE
    assert gamepad.arm_keys('LB + X') == (gamepad.BUTTON_LB, ('button', gamepad.BUTTON_X, 1))
    assert gamepad.arm_keys('LB + up') == (gamepad.BUTTON_LB, ('axis', gamepad.AXIS_DY, -1))
    for _, keys in poses + [('release', release)]:
        modifier, _ = gamepad.arm_keys(keys)
        assert modifier == gamepad.BUTTON_LB  # LB is free in Velocity (LT + B, RB + A are its transitions)
    assert len({keys for _, keys in poses} | {release}) == len(poses) + 1
    for bad in ('LT + up', 'LB', 'LB + Z'):
        with pytest.raises(ValueError):
            gamepad.arm_keys(bad)


def test_analyze_replays_the_arm_command_as_the_environment_moves_its_targets(tmp_path):
    """analyze.ArmReplay on g1_ctrl.log's warn lines: zeros before the first press, flag 1 and the training's quintic
    (task.arm_target_offset) from the press, the release back home over its steps, then zeros; a later Velocity entry's
    lines are not taken."""
    sys.path.insert(0, str(SIM2SIM))
    import analyze
    d = np.linspace(-.6, .6, 17).astype(np.float32)
    params = {'size': 18, 'poses': [{'name': 'carry', 'offsets': d.tolist()}]}
    log = tmp_path / 'g1_ctrl.log'
    log.write_text('\n'.join([
        '[x] [warning] arm_command: carry at policy step 10, 50 steps (sim2sim arms patch)',
        '[x] [warning] arm_command: release at policy step 100, 64 steps (sim2sim arms patch)',
        '[x] [warning] arm_command: free at policy step 164 (sim2sim arms patch)',
        '[x] [warning] arm_command: carry at policy step 3, 50 steps (sim2sim arms patch)']), encoding='utf-8')
    replay = analyze.ArmReplay(params, log)
    assert not replay.slot(9).any() and not replay.slot(0).any()
    for step in range(10, 100):
        expected = task.arm_target_offset(torch.zeros((1, 17), dtype=torch.float64), torch.from_numpy(d).double()[None],
                                          torch.tensor([step - 10]), torch.tensor([50]))[0].numpy()
        slot = replay.slot(step)
        assert slot[0] == 1
        np.testing.assert_allclose(slot[1:], expected, atol=1e-6)
    held = replay.slot(99)[1:]
    for step in range(100, 164):
        expected = task.arm_target_offset(torch.from_numpy(held).double()[None], torch.zeros((1, 17), dtype=torch.float64),
                                          torch.tensor([step - 100]), torch.tensor([64]))[0].numpy()
        np.testing.assert_allclose(replay.slot(step)[1:], expected, atol=1e-6)
        assert replay.slot(step)[0] == 1
    assert not replay.slot(164).any() and not replay.slot(300).any()


def test_hand_load_scene_carries_the_training_loads_at_the_palm(tmp_path):
    """hand_load.py on the G1 model: each wrist_yaw_link mass + kg and its COM toward the palm exactly as the training's
    task.hand_load_mass (and so as G1VelocityEnv.set_hand_load), the inertia kept; 0 kg leaves the text alone."""
    sys.path.insert(0, str(SIM2SIM))
    import hand_load
    assert hand_load.PALM == task.HAND_LOAD_POSITION and hand_load.BODIES == task.HAND_LOAD_BODIES
    assert hand_load.loads_of('1.5') == (1.5, 1.5) and hand_load.loads_of('1,2') == (1., 2.)
    for bad in ('-1', 'nan', '1,2,3'):
        with pytest.raises(ValueError):
            hand_load.loads_of(bad)
    source = ROOT / 'assets' / 'robots' / 'g1' / 'g1_29dof.xml'
    text = source.read_text(encoding='utf-8')
    assert hand_load.loaded(text, (0., 0.)) == text
    meshes = (source.parent / 'meshes').as_posix()
    loaded = hand_load.loaded(text, (1., 2.)).replace('meshdir="meshes"', f'meshdir="{meshes}"')
    model = mujoco.MjModel.from_xml_string(loaded)
    env = G1VelocityEnv(real_robot=True)
    try:
        env.reset(seed=0)
        env.set_hand_load([1., 2.])
        for name in task.HAND_LOAD_BODIES:
            ours, theirs = env.model.body(name).id, model.body(name).id
            assert model.body_mass[theirs] == pytest.approx(env.model.body_mass[ours], rel=1e-8)
            np.testing.assert_allclose(model.body_ipos[theirs], env.model.body_ipos[ours], atol=1e-8)
            np.testing.assert_allclose(model.body_inertia[theirs], env.model.body_inertia[ours], rtol=1e-6)
    finally:
        env.close()
