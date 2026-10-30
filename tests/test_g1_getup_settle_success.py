"""g1_getup task_config settle_success and settle_starts (iteration 85), CPU: the settling success by joints (the
pelvis, every free arm joint, the mean over the free joints, the still time) and the starts from the target pose with noise
(a reverse curriculum), the rest from the bank rows of chosen classes. The settings' checks and labels, the unchanged
contract; the rule against a hand calculation, numpy and torch alike; the task's success, starts and their own stream;
scripts/lab/getup_track_clip.py --from-frame. Off (missing or null) the settling task is bit for bit the old one.
CPU–GPU: tests/test_g1_gpu.py -k settle_success."""
import hashlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_track
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OPTIONS, drop_bank
from test_g1_getup_settle import SETTING, lying_clip

SUCCESS = {'pelvis_degrees': 10, 'arm_joint': .35, 'mean': .15, 'still': .3}
STARTS = {'target': .5, 'noise': 1, 'classes': ['front']}


def settle_env(max_episode_steps=80, **settings):
    return G1GetupEnv(max_episode_steps, bank=drop_bank(), settle_clip=SETTING,
                      settle_bank=[lying_clip(0), lying_clip(1)], reward_weights={'impact': 2., 'sole_impact': 1.},
                      **settings)


def settle_digest(seed=66, steps=240, **settings):
    """sha256 of every observation, reward, end flag, qpos, success clock and reward term of the settling task: training
    resets with random actions over the ±10 box, then from each bank lying row with the motors holding its target."""
    env = settle_env(**settings)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()

    def eat(obs, reward, terminated, truncated, info):
        digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated])
                      + env.data.qpos.tobytes() + np.int64(info['settle_count']).tobytes()
                      + bytes([info['settle_done']]))
        for key in sorted(info['reward_components']):
            digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
    try:
        obs, _ = env.reset(seed=seed)
        digest.update(obs.tobytes())
        for k in range(steps):
            result = env.step(rng.uniform(-10., 10., 29) * (.3 if k % 7 else 1.))
            eat(*result)
            if result[2] or result[3]:
                obs, _ = env.reset(seed=seed + k + 1)
                digest.update(obs.tobytes())
        for row in (0, 1):
            obs, _ = env.reset(seed=seed, options=dict(OPTIONS, start=row))
            digest.update(obs.tobytes())
            action = (env.settle['position'][row] - env.home) / env.scale
            for _ in range(80):
                eat(*env.step(action))
    finally:
        env.close()
    return digest.hexdigest()


# settle_digest of the code before these settings (commit 49583b2), with drop_bank() and the lying clips.
OLD_SETTLE_DIGEST = 'f101f979267d90f0e3801117fd5cc1b186dcddd67227ad5264ee9ebf57a6d477'


def test_the_settings_are_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, SETTLE_START_CLASSES
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    plain = validate_config(base)['task_config']
    assert plain['settle_success'] is None and plain['settle_starts'] is None  # old stages: off
    on = validate_config({**base, 'task_config': {'settle_clip': SETTING, 'settle_success': SUCCESS,
                                                  'settle_starts': STARTS}})['task_config']
    assert on['settle_success'] == {key: float(value) for key, value in SUCCESS.items()}
    assert on['settle_starts'] == {'target': .5, 'noise': 1., 'classes': ['front']}
    short = validate_config({**base, 'task_config': {'settle_clip': SETTING, 'settle_starts': {'target': 1}}})
    assert short['task_config']['settle_starts'] == {'target': 1., 'noise': 1., 'classes': list(task.SETTLE_CLASSES)}
    assert SETTLE_START_CLASSES == task.SETTLE_CLASSES
    for value in ({**SUCCESS, 'still': 0}, {**SUCCESS, 'mean': 2.}, {**SUCCESS, 'x': 1}, {'pelvis_degrees': 10},
                  {**SUCCESS, 'arm_joint': '0.3'}, {**SUCCESS, 'still': float('nan')}, [10, .35, .15, .3], True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'settle_clip': SETTING, 'settle_success': value}})
    for value in ({'target': 1.5}, {'noise': 1}, {'target': .5, 'noise': 6}, {'target': .5, 'classes': []},
                  {'target': .5, 'classes': ['stand']}, {'target': .5, 'classes': ['front', 'front']},
                  {'target': .5, 'x': 1}, {'target': True}, .5):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'settle_clip': SETTING, 'settle_starts': value}})
    for other in ({'settle_success': SUCCESS}, {'settle_starts': STARTS}):  # both need settle_clip
        with pytest.raises(ValueError, match='needs settle_clip'):
            validate_config({**base, 'task_config': other})
    assert task_identity('g1_29dof', 'g1_getup', {'settle_clip': SETTING, 'settle_success': SUCCESS,
                                                 'settle_starts': STARTS}) == task_identity('g1_29dof', 'g1_getup', {})
    fields = {p['key']: p for p in G1_GETUP_PARAMETERS}
    assert fields['settle_success']['default'] is None and 'Успех укладки' in fields['settle_success']['label']
    assert fields['settle_starts']['default'] is None and 'Старты укладки' in fields['settle_starts']['label']


def test_off_the_settling_task_is_bit_for_bit_the_old_one():
    assert settle_digest() == OLD_SETTLE_DIGEST
    assert settle_digest(settle_success=None, settle_starts=None) == OLD_SETTLE_DIGEST
    env = settle_env()
    try:
        assert env.settle_hold == 15 and env.settle_classes == task.SETTLE_CLASSES
    finally:
        env.close()


def test_the_rule_by_hand_in_numpy_and_torch():
    rng = np.random.default_rng(85)
    n, k, nu = 64, 2, 29
    joints = tuple(j for j in range(nu) if j not in (20, 21, 27, 28))
    arms = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)
    target_position = rng.uniform(-1, 1, (k, nu))
    position = target_position[rng.integers(k, size=n)] + rng.uniform(-.4, .4, (n, nu)) * rng.uniform(0, 1, (n, 1))
    target_gravity = np.array([[-1., 0., 0.], [1., 0., 0.]])
    tilt = np.radians(rng.uniform(0, 30, n))
    gravity = np.stack([np.cos(tilt), np.sin(tilt), np.zeros(n)], 1) * rng.choice([-1, 1], (n, 1))
    worst = task.settle_arm_worst(position, target_position, arms)
    assert worst.shape == (n, k)
    for w in range(n):
        for t in range(k):
            assert worst[w, t] == np.abs(position[w, list(arms)] - target_position[t, list(arms)]).max()
    angle, pose, deviation = task.settle_errors(gravity, position, target_gravity, target_position, joints)
    rows = np.arange(n)
    success = {key: float(value) for key, value in SUCCESS.items()}
    near = task.settle_near(angle[rows, 0], deviation[rows, 0], worst[rows, 0], success)
    expected = ((angle[:, 0] <= math.radians(10)) & (worst[:, 0] <= .35) & (deviation[:, 0] <= .15))
    np.testing.assert_array_equal(near, expected)
    assert 0 < expected.sum() < n  # both kinds present
    # Without the setting: the old rule, bit for bit.
    np.testing.assert_array_equal(task.settle_near(angle[:, 0], deviation[:, 0]),
                                  (angle[:, 0] <= math.radians(25)) & (deviation[:, 0] <= .3))
    # One arm joint off by 0.5 rad, the rest exact: the mean passes the old rule and the new mean, the arm does not.
    one = target_position[:1].copy()
    one[0, 16] += .5
    one_angle, _, one_deviation = task.settle_errors(target_gravity[:1], one, target_gravity, target_position, joints)
    one_worst = task.settle_arm_worst(one, target_position, arms)
    assert task.settle_near(one_angle[:, 0], one_deviation[:, 0])[0]
    assert one_deviation[0, 0] == pytest.approx(.5 / 25) and one_worst[0, 0] == pytest.approx(.5)
    assert not task.settle_near(one_angle[:, 0], one_deviation[:, 0], one_worst[:, 0], success)[0]
    # The reward's near (the stillness gate) follows the rule; the terms are the old ones.
    velocity, joint_velocity = rng.normal(size=(n, 6)), rng.normal(size=(n, nu))
    action, previous = rng.uniform(-3, 3, (n, nu)), rng.uniform(-3, 3, (n, nu))
    low, high = -np.ones(nu), np.ones(nu)
    old_terms, nearest, old_near = task.settle_reward_components(
        angle, pose, deviation, joint_velocity, velocity, action, previous, position, low, high, joints)
    terms, chosen, new_near = task.settle_reward_components(
        angle, pose, deviation, joint_velocity, velocity, action, previous, position, low, high, joints,
        arm_worst=worst, success=success)
    np.testing.assert_array_equal(chosen, nearest)
    np.testing.assert_array_equal(new_near, task.settle_near(angle[rows, nearest], deviation[rows, nearest],
                                                             worst[rows, nearest], success))
    still = (joint_velocity[:, list(joints)] ** 2).mean(1) + (velocity ** 2).sum(1)
    np.testing.assert_allclose(terms['settle_still'], -(new_near * still))
    for key in terms:
        if key != 'settle_still':
            np.testing.assert_array_equal(terms[key], old_terms[key])
    assert task.settle_hold_steps(.02) == 15 and task.settle_hold_steps(.02, {**success, 'still': .5}) == 25
    # torch gives the same.
    t = torch.from_numpy
    np.testing.assert_array_equal(task.settle_arm_worst(t(position), t(target_position), arms).numpy(), worst)
    torch_terms, torch_nearest, torch_near = task.settle_reward_components(
        *(t(x) for x in (angle, pose, deviation, joint_velocity, velocity, action, previous, position, low, high)),
        joints, arm_worst=t(worst), success=success)
    np.testing.assert_array_equal(torch_nearest.numpy(), nearest)
    np.testing.assert_array_equal(torch_near.numpy(), new_near)
    for key, value in torch_terms.items():
        np.testing.assert_allclose(value.numpy(), terms[key], rtol=1e-12, atol=1e-12, err_msg=key)


def _hold(env, row, offset=None, steps=80):
    """Episode from bank row `row` with the motors holding its target (plus offset (29,) rad); the last info."""
    env.reset(seed=1, options=dict(OPTIONS, start=row))
    target = env.settle['position'][row] + (0. if offset is None else offset)
    action = (target - env.home) / env.scale
    info = {}
    for _ in range(steps):
        *_, info = env.step(action)
    return info


def test_the_task_settles_by_the_rule_and_an_arm_off_target_is_no_success():
    """Dropped onto its back with the motors holding the back target, the robot settles under the new rule too (the
    hold `still` long); with the left shoulder roll held 0.5 rad off, the old rule settles (the mean 0.02 rad) and the
    new one does not (that arm joint 0.5 > 0.35)."""
    offset = np.zeros(29)
    offset[16] = .5  # left shoulder roll
    old, new = settle_env(), settle_env(settle_success={**SUCCESS, 'still': .5})
    try:
        assert new.settle_hold == 25 and new.settle_arms == (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)
        assert all(new.model.joint(new.joints[j]).name.split('_')[1] in ('shoulder', 'elbow', 'wrist')
                   for j in new.settle_arms)
        info = _hold(new, 0)
        assert info['settle_done'] and info['settle_target'] == 'test-back' and info['settle_arm_worst'] < .35
        assert info['settle_angle'] < 10 and info['settle_deviation'] < .15
        assert _hold(old, 0, offset)['settle_done']
        info = _hold(new, 0, offset)
        assert not info['settle_done'] and not info['settle_near'] and info['settle_arm_worst'] > .35
        assert 'settle_arm_worst' not in _hold(old, 0)
    finally:
        old.close()
        new.close()


def test_target_starts_their_noise_classes_and_own_stream():
    """settle_starts: target 1 with noise 0 starts every training reset from a target's first frame at rest, out of the
    floor; with noise 1 the joints stay within ±0.1 rad of it; the bank starts come from `classes` only; the bank rows
    and the domain randomization are drawn as without the setting (the target's draws have their own stream); an
    evaluation reset (assist_force) never starts from the target; the option settle_target forces one."""
    old = settle_env()
    exact = settle_env(settle_starts={'target': 1, 'noise': 0})
    noisy = settle_env(settle_starts={'target': .5, 'noise': 1})
    front = settle_env(settle_starts={'target': .5, 'classes': ['front']})
    try:
        for seed in range(6):
            _, info = exact.reset(seed=seed)
            assert info['start_class'] == 'target' and info['start'] == -1
            target = int(np.argmin([np.abs(exact.data.qpos[7:] - p).max() for p in exact.settle['position']]))
            np.testing.assert_allclose(exact.data.qpos[7:], exact.settle['position'][target], atol=1e-12)
            np.testing.assert_allclose(exact.data.qvel, 0, atol=1e-12)
            assert g1_getup_track.floor_depth(exact.model, exact.data, exact.robot_geoms, exact.floor) > -1e-6
            assert info['settle_angle'] < 1e-6 and info['settle_deviation'] < 1e-12
        kinds, targets = [], 0
        for seed in range(60):
            _, reference = old.reset(seed=seed)
            _, info = noisy.reset(seed=seed)
            kinds.append(info['start_class'])
            assert info['body_friction'] == reference['body_friction']  # the task's own draws unchanged
            if info['start_class'] == 'target':
                targets += 1
                nearest = np.abs(noisy.data.qpos[7:][None] - noisy.settle['position']).max(1).min()
                assert nearest <= .1 + 1e-9 and np.abs(noisy.data.qvel[:6]).max() <= .1 + 1e-9
            else:
                assert info['start'] == reference['start']  # the same bank row
            _, info = front.reset(seed=seed)
            assert info['start_class'] in ('front', 'target')
            _, info = noisy.reset(seed=seed, options=dict(OPTIONS))  # evaluation: never a target start
            assert info['start_class'] != 'target'
        assert 15 < targets < 45 and {'back', 'front'} <= set(kinds)
        _, info = old.reset(seed=3, options=dict(OPTIONS, settle_target=1))
        assert info['start_class'] == 'target' and info['settle_deviation'] < .1  # the front target, noise 1
        with pytest.raises(ValueError):
            old.reset(seed=3, options=dict(settle_target=2))
        with pytest.raises(ValueError):
            old.reset(seed=3, options=dict(settle_target=0, start=0))
    finally:
        for env in (old, exact, noisy, front):
            env.close()


def test_from_frame_cuts_every_array_and_keeps_the_hold(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'lab'))
    import getup_track_clip
    clip = lying_clip(0)
    clip = {key: np.concatenate([value, value + 1e-3, value + 2e-3]) for key, value in clip.items()}  # 6 frames
    out, record = getup_track_clip.cut_clip(clip, 2)
    assert record == dict(from_frame=2, frames=4, source_frames=6)
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(out[key], clip[key][2:])
    hold, cut_hold = g1_getup_track.hold_frames(clip), g1_getup_track.hold_frames(out)
    for key in hold:
        np.testing.assert_array_equal(cut_hold[key], hold[key][2:])
    for start in (0, 5, 6):
        with pytest.raises(SystemExit):
            getup_track_clip.cut_clip(clip, start)


A493_V3 = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3'
A493_F21 = A493_V3 + '-f21'


@pytest.mark.skipif(not all(g1_getup_track.clip_path(name).is_file() for name in (A493_V3, A493_F21)),
                    reason='the lab clips runs/humanoid_lab/getup_track/<name>.npz are made locally')
def test_the_lab_cut_clip_is_a493_v3_from_frame_21():
    base, cut = g1_getup_track.load(A493_V3), g1_getup_track.load(A493_F21)
    assert cut['header']['patch']['from_frame'] == 21 and cut['header']['patch']['base_md5'] == base['md5']
    assert cut['header']['patch']['base_patch'] == base['header']['patch']
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(cut[key], base[key][21:])
    # Its first frame has no self-penetration deeper than 2 mm (frames 0-18 of the base: the right hand in the torso).
    env = G1GetupEnv(bank=drop_bank())
    try:
        for qpos, deepest in ((base['qpos'][0], -.05), (cut['qpos'][0], -.002)):
            env.data.qpos[:], env.data.qvel[:] = qpos, 0
            mujoco.mj_forward(env.model, env.data)
            depth = min([env.data.contact.dist[i] for i in range(env.data.ncon)
                         if env.robot_geoms[env.data.contact.geom1[i]] and env.robot_geoms[env.data.contact.geom2[i]]]
                        or [0.])
            assert (depth < deepest) if deepest < -.01 else (depth >= deepest)
    finally:
        env.close()
