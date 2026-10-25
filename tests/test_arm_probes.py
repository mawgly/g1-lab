"""The arm probes of iteration 42 (scripts/lab/probes): train_falls.py's gravity oracle conditions (home_ff / target_ff,
the critic's lab-critic/it42/tf_ff.py) and arm_counterfactual.py (the critic's cf.py / cf_quick.py / cf_report.py)."""
import importlib
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from test_g1_arm_targets import ARMS, _stage

PROBES = Path(__file__).resolve().parents[1] / 'scripts' / 'lab' / 'probes'


def lab_probe(name, monkeypatch):
    """A module of scripts/lab/probes, the CUDA variable put back after the test (the probes hide CUDA on import)."""
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    return importlib.import_module(name)


def test_train_falls_conditions_and_the_gravity_oracle_moves_the_pd_target_by_the_static_torque(monkeypatch):
    """'<mode>_ff:<kg>' is the mode's episode with the oracle; the oracle's action moves the tracked joints' PD target
    by beta qfrc_bias / kp (the static torque the PD lacks), nothing else."""
    probe = lab_probe('train_falls', monkeypatch)
    assert probe.condition('free') == ('free', 0., False)
    assert probe.condition('target:0.5') == ('target', .5, False) and probe.condition('home:0') == ('home', 0., False)
    assert probe.condition('target_ff:1') == ('target', 1., True) and probe.condition('home_ff:2') == ('home', 2., True)
    for bad in ('free:1', 'free_ff', 'home', 'target_ff', 'target_ff:', 'walk:1', 'target_xx:1'):
        with pytest.raises(ValueError):
            probe.condition(bad)
    env = G1VelocityEnv(**ARMS)
    try:
        env.reset(seed=3, options={'command': [0., 0., 0.]})
        names = [env.model.joint(j).name for j in env.joints]
        pose = env.home.copy()
        for side in ('left', 'right'):
            pose[names.index(f'{side}_shoulder_pitch_joint')] = -1.2
            pose[names.index(f'{side}_elbow_joint')] = 1.2
        env.set_arm_targets(pose, 1)
        env.set_hand_load(1.)
        for _ in range(10):
            env.step(np.zeros(29))
        tracked = list(env.track_joints)
        bias = env.data.qfrc_bias[np.asarray(env.vadr)[tracked]]
        assert np.abs(bias).max() > 1.  # the loaded arms' gravity torque
        plain = env._target(np.zeros(29), env.arm_offset)
        for beta in (1., .5):
            action = np.zeros(29)
            action[tracked] += probe.gravity_action(env, beta)
            moved = env._target(action, env.arm_offset)
            np.testing.assert_allclose(env.kp[tracked] * (moved - plain)[tracked], beta * bias, rtol=1e-9, atol=1e-9)
            others = [j for j in range(29) if j not in tracked]
            np.testing.assert_array_equal(moved[others], plain[others])
    finally:
        env.close()


def test_arm_counterfactual_poses_and_break_even(monkeypatch):
    """The nine poses of the critic's probe; w* = -dO/dT on the unweighted arm_track, NaN without a gain, no row for
    a base that fell or a run without a window."""
    probe = lab_probe('arm_counterfactual', monkeypatch)
    poses = probe.poses()
    assert [name for name, _, _ in poses] == ['carry-1.5', 'carry-1.1', 'carry-0.7', 'carry-0.4', 'free0', 'free1',
                                              'free2', 'free3', 'extended']
    names, angles = poses[-1][1], dict(zip(poses[-1][1], poses[-1][2]))
    assert len(names) == 10 and angles['left_shoulder_pitch_joint'] == angles['right_shoulder_pitch_joint'] == -1.57
    assert angles['left_elbow_joint'] == 1.57 and angles['right_wrist_roll_joint'] == 0.
    base = dict(fell=None, track=.5, comps={'arm_track': 1., 'torso': -.1, 'alive': .15})
    oracle = dict(fell=None, track=.8, comps={'arm_track': 1.6, 'torso': -.3, 'alive': .15})
    gain, cost, wstar = probe.break_even(base, oracle)
    assert (gain, cost) == pytest.approx((.3, -.2)) and wstar == pytest.approx(2 / 3)
    assert math.isnan(probe.break_even(base, {**oracle, 'track': .5 + 1e-5})[2])
    assert probe.break_even({**base, 'fell': 300}, oracle) is None
    assert probe.break_even(base, {**oracle, 'fell': 120, 'comps': {}, 'track': None}) is None


def test_arm_counterfactual_runs_base_and_oracle_on_a_stage(tmp_path, monkeypatch, capsys):
    """A short run of arm_counterfactual on a small arm_targets stage: base and oracle, the oracle's c at work, the
    report's table and summary."""
    probe = lab_probe('arm_counterfactual', monkeypatch)
    for key, value in dict(STEPS=14, MOVE_AT=2, CORRECT_AT=4, WINDOW_AT=8).items():
        monkeypatch.setattr(probe, key, value)
    _, stage = _stage(tmp_path, ARMS)
    threads = torch.get_num_threads()
    try:
        name, names, pose = probe.poses()[-1]
        job = dict(stage=str(stage), pose_name=name, names=names, pose=pose, command=(0., 0., 0.), load=1.)
        result = probe.run(job)
    finally:
        torch.set_num_threads(threads)
    base, oracle = result['rows']
    assert (base['variant'], oracle['variant']) == ('base', 'oracle') and 'stage' not in result
    assert base['fell'] is None and len(base['signed']) == 10 and 0. <= base['track'] <= 1.
    assert not np.any(base['c']) and np.all(np.abs(oracle['c']) > 0)
    assert set(base['comps']) == set(oracle['comps']) and 'arm_track' in base['comps']
    summary = probe.report([result], [2., 4.])
    assert len(summary) == 1 and summary[0]['pose'] == 'extended'
    assert 'cmd 0.0 load 1.0: n 1' in capsys.readouterr().out


def test_arm_joints_errors_per_joint_and_the_palms(tmp_path, monkeypatch, capsys):
    """arm_joints on a small arm_targets stage, a short run: the mean signed error of each of the ten tracked joints by
    name, the palms against the target pose's (forward kinematics: home gives the palms of the home pose standing),
    the table's row."""
    probe = lab_probe('arm_joints', monkeypatch)
    for key, value in dict(STEPS=14, MOVE_AT=2, WINDOW_AT=8).items():
        monkeypatch.setattr(probe, key, value)
    _, stage = _stage(tmp_path, ARMS)
    import mujoco
    env = G1VelocityEnv(**ARMS)
    try:  # the palms of a pose do not depend on where the base stands
        env.reset(seed=3, options={'command': [0., 0., 0.]})
        data = mujoco.MjData(env.model)
        data.qpos[:] = env.data.qpos
        data.qpos[3:7] = [np.cos(.2), 0., np.sin(.1), np.sin(.2)] / np.linalg.norm([np.cos(.2), 0., np.sin(.1),
                                                                                     np.sin(.2)])
        data.qpos[env.qadr] = env.home
        mujoco.mj_kinematics(env.model, data)
        assert probe.target_palms(env, env.home) == pytest.approx(probe.palm_distance(env, data), abs=1e-9)
    finally:
        env.close()
    threads = torch.get_num_threads()
    try:
        carry = [p for p in probe.poses() if p[0].startswith('carry')]
        assert len(carry) == 4
        name, names, pose = carry[0]
        result = probe.run(dict(stage=str(stage), pose_name=name, names=names, pose=pose, load=1.))
    finally:
        torch.set_num_threads(threads)
    assert result['fell'] is None and len(result['error']) == 10
    assert set(result['error']) == {f'{side}_{part}_joint' for side in ('left', 'right') for part in probe.PARTS}
    assert result['palm'] > 0 and result['palm_target'] > 0
    summary = probe.report([result], [str(stage)], [1.])
    assert len(summary) == 1 and summary[0]['falls'] == 0 and len(summary[0]['shoulder_roll']) == 2
    assert summary[0]['palms'] == pytest.approx((result['palm'] - result['palm_target']) * 100)
    assert probe.label(stage) in capsys.readouterr().out
