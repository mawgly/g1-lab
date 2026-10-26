"""g1_getup task_config settle_clip (iteration 66), CPU: settling from any fall pose into the nearest of the clips'
first frames (the starts of the human trackers). The setting's checks, its label and the unchanged contract; the
errors, nearest target, terms and success clock (g1_tensor_math) against a hand calculation, numpy and torch alike; the
task's targets, starts, episode end, success and logs; the evaluation's metrics; scripts/robot/package.py checks each
clip's licence. Off (missing or null) the task is bit for bit the old one. CPU–GPU: tests/test_g1_gpu.py -k settle."""
import importlib
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_getup_track
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_SETTLE_WEIGHTS, GETUP_CRITIC_EXTRAS
from humanoid_sim.tasks.g1_getup import SETTLE_STEPS, G1GetupEnv
from test_g1_getup_impact import DROP, OLD_DIGESTS, OPTIONS, drop_bank, getup_digest

BACK, FRONT = {'file': 'test-back', 'md5': '1' * 32}, {'file': 'test-front', 'md5': '2' * 32}
SETTING = [BACK, FRONT]


def lying_clip(row):
    """A two-frame clip (g1_getup_track.build) of drop_bank's row `row` laid on the floor: its first frame is the
    home pose lying there (row 0 on the back, 1 on the front)."""
    env = bank_module.getup_model_env()
    start = drop_bank()['qpos'][row].copy()
    start[2] -= DROP
    return g1_getup_track.build(env, np.repeat(start[None], 2, 0))


def settle_env(**settings):
    return G1GetupEnv(bank=drop_bank(), settle_clip=SETTING, settle_bank=[lying_clip(0), lying_clip(1)], **settings)


def test_the_setting_is_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['settle_clip'] is None  # old stages: off
    on = validate_config({**base, 'task_config': {'settle_clip': SETTING}})['task_config']
    assert on['settle_clip'] == SETTING and on['track_clip'] is None
    assert validate_config({**base, 'task_config': {'settle_clip': (BACK,)}})['task_config']['settle_clip'] == [BACK]
    for value in ([], [BACK, BACK], [BACK] + [{'file': f'c{k}', 'md5': f'{k}' * 32} for k in range(3, 7)],
                  [{'file': 'x'}], [{'file': '../x', 'md5': '3' * 32}], [{'file': 'x', 'md5': 'abc'}],
                  [{'file': 'x', 'md5': '3' * 32, 'w': 1}], BACK, 'x', 5, True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'settle_clip': value}})
    for other in ({'track_clip': BACK}, {'pd_bound': [1., .6]}, {'human_starts': {'share': .3, 'md5': '3' * 32}},
                  {'style_judge': True}, {'reward_weights': {'style': .5}}, {'reward_weights': {'amp': 1.}}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'settle_clip': SETTING, **other}})
    plain = task_identity('g1_29dof', 'g1_getup', {})
    assert task_identity('g1_29dof', 'g1_getup', {'settle_clip': SETTING}) == plain  # actor 490, critic six extras
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['settle_clip']
    assert field['group'] is None and field['default'] is None and 'Укладка' in field['label']
    assert G1_GETUP_SETTLE_WEIGHTS == dict(settle_ori=1., settle_pose=1., settle_still=1., action_rate=.01,
                                           joint_limit=1., self_contact=1.)
    assert task.SETTLE_EPISODE_SECONDS == 7. and SETTLE_STEPS == 350 and 'stand' not in task.SETTLE_CLASSES


def test_off_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('a settle clip read with the setting off')
    monkeypatch.setattr(g1_getup_track, 'load', refuse)
    bank = drop_bank()
    for config, name in (({'settle_clip': None}, 'plain'), ({'pd_bound': [1., .6], 'settle_clip': None}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    env = G1GetupEnv(bank=bank)
    try:
        assert env.settle is None and env.max_episode_steps == 500
    finally:
        env.close()


def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_the_errors_nearest_terms_and_clock_by_hand_in_numpy_and_torch():
    rng = np.random.default_rng(0)
    n, k, nu = 32, 2, 29
    joints = tuple(j for j in range(nu) if j not in (20, 21, 27, 28))
    gravity, target_gravity = _unit(rng, n), _unit(rng, k)
    position, target_position = rng.uniform(-1, 1, (n, nu)), rng.uniform(-1, 1, (k, nu))
    position[0], gravity[0] = target_position[1] + .01, target_gravity[1]  # world 0 right at the front target
    angle, pose, deviation = task.settle_errors(gravity, position, target_gravity, target_position, joints)
    assert angle.shape == pose.shape == deviation.shape == (n, k)
    for w in range(n):
        for t in range(k):
            cosine = np.clip(gravity[w] @ target_gravity[t], -1, 1)
            assert angle[w, t] == pytest.approx(math.acos(cosine), abs=1e-7)
            difference = position[w, list(joints)] - target_position[t, list(joints)]
            assert pose[w, t] == pytest.approx((difference ** 2).mean())
            assert deviation[w, t] == pytest.approx(np.abs(difference).mean())
    score = np.exp(-angle ** 2 / .25) + np.exp(-pose / .25)
    nearest = task.settle_nearest(angle, pose)
    np.testing.assert_array_equal(nearest, score.argmax(1))
    assert nearest[0] == 1
    # The angle is exact near zero in float32 (no arccos floor).
    tilt = np.array([[0., math.sin(1e-4), -math.cos(1e-4)]], dtype=np.float32)
    small, _, _ = task.settle_errors(torch.tensor(tilt), torch.zeros((1, nu)), torch.tensor([[0., 0., -1.]]),
                                     torch.zeros((1, nu)), joints)
    assert float(small[0, 0]) == pytest.approx(1e-4, rel=.01)
    # The terms of the nearest target.
    velocity, joint_velocity = rng.normal(size=(n, 6)), rng.normal(size=(n, nu))
    action, previous = rng.uniform(-3, 3, (n, nu)), rng.uniform(-3, 3, (n, nu))
    low, high = -np.ones(nu), np.ones(nu)
    share, impact, sole = rng.random(n), rng.random(n), rng.random(n)
    weights = {**G1_GETUP_SETTLE_WEIGHTS, 'impact': 2., 'sole_impact': 1.}
    terms, chosen, near = task.settle_reward_components(
        angle, pose, deviation, joint_velocity, velocity, action, previous, position, low, high, joints, weights,
        self_contact=share, impact=impact, sole_impact=sole)
    np.testing.assert_array_equal(chosen, nearest)
    rows = np.arange(n)
    a, p, d = angle[rows, nearest], pose[rows, nearest], deviation[rows, nearest]
    expected_near = (a <= math.radians(25)) & (d <= .3)
    np.testing.assert_array_equal(near, expected_near)
    assert near[0] and not near[1:].all()
    still = (joint_velocity[:, list(joints)] ** 2).mean(1) + (velocity ** 2).sum(1)
    np.testing.assert_allclose(terms['settle_ori'], np.exp(-a ** 2 / .25))
    np.testing.assert_allclose(terms['settle_pose'], np.exp(-p / .25))
    np.testing.assert_allclose(terms['settle_still'], -(expected_near * still))
    np.testing.assert_allclose(terms['action_rate'], -.01 * ((action - previous) ** 2).sum(1))
    np.testing.assert_allclose(terms['joint_limit'], -np.clip(np.abs(position) - 1, 0, None).sum(1))
    np.testing.assert_allclose(terms['self_contact'], -share)
    np.testing.assert_allclose(terms['impact'], -2. * impact)
    np.testing.assert_allclose(terms['sole_impact'], -sole)
    off, _, _ = task.settle_reward_components(angle, pose, deviation, joint_velocity, velocity, action, previous,
                                              position, low, high, joints, weights)
    assert not any(off[key].any() for key in ('self_contact', 'impact', 'sole_impact'))
    # The success clock: on while near and every free joint slower than 0.5 rad/s, back to 0 otherwise.
    slow = np.zeros((3, nu))
    slow[1, 5] = .6
    slow[2, 20] = 5.  # a locked wrist does not count
    count = task.settle_update(np.array([4, 4, 4]), np.array([True, True, True]), slow, joints)
    np.testing.assert_array_equal(count, [5, 0, 5])
    np.testing.assert_array_equal(task.settle_update(np.array([4]), np.array([False]), slow[:1], joints), [0])
    # torch gives the same numbers.
    tensors = [torch.from_numpy(x) for x in (gravity, position, target_gravity, target_position)]
    torch_errors = task.settle_errors(*tensors, joints)
    for got, want in zip(torch_errors, (angle, pose, deviation)):
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-10, atol=1e-12)
    torch_terms, torch_nearest, torch_near = task.settle_reward_components(
        *torch_errors, *(torch.from_numpy(x) for x in (joint_velocity, velocity, action, previous, position, low,
                                                        high)), joints, weights, self_contact=torch.from_numpy(share),
        impact=torch.from_numpy(impact), sole_impact=torch.from_numpy(sole))
    np.testing.assert_array_equal(torch_nearest.numpy(), nearest)
    np.testing.assert_array_equal(torch_near.numpy(), near)
    for key, value in torch_terms.items():
        np.testing.assert_allclose(value.numpy(), terms[key], rtol=1e-10, atol=1e-12, err_msg=key)
    np.testing.assert_array_equal(task.settle_update(torch.tensor([4, 4, 4]), torch.ones(3, dtype=torch.bool),
                                                     torch.from_numpy(slow), joints).numpy(), [5, 0, 5])


def test_targets_starts_success_and_episode_end():
    """The targets are the clips' first frames (the pelvis gravity in its frame, the joints); training resets draw the
    bank's rows of the settling classes only (not the standing one), without the assist; an episode is 7 s; a robot
    dropped from 0.3 m onto its back with its motors holding the back target settles into it once still for 0.3 s,
    and the episode goes on after it; on the front into the front one; the observation is the getting-up one."""
    env = settle_env(reward_weights={'impact': 2., 'sole_impact': 1.})
    try:
        assert env.max_episode_steps == SETTLE_STEPS and env.assist == 0.
        assert env.observation_space.shape == (env.actor_inputs + GETUP_CRITIC_EXTRAS,)
        for k, clip in enumerate((lying_clip(0), lying_clip(1))):
            env.data.qpos[:], env.data.qvel[:] = clip['qpos'][0], 0
            mujoco.mj_forward(env.model, env.data)
            np.testing.assert_allclose(env.settle['gravity'][k], env.data.xmat[env.pelvis].reshape(3, 3).T @ [0, 0, -1],
                                       atol=1e-12)
            np.testing.assert_array_equal(env.settle['position'][k], clip['qpos'][0, 7:])
        assert env.settle['gravity'][0, 0] < -.99 and env.settle['gravity'][1, 0] > .99  # face up, face down
        classes = set()
        for seed in range(200):
            _, info = env.reset(seed=seed)
            classes.add((info['start'], info['start_class']))
            assert env.assist == 0. and info['assist_force'] == 0.
        assert classes == {(0, 'back'), (1, 'front'), (2, 'front')}
        for row, name in ((0, 'test-back'), (1, 'test-front')):
            obs, info = env.reset(seed=1, options=dict(OPTIONS, start=row))
            assert obs.shape == env.observation_space.shape and obs[-1] == 0.  # the critic's assist input
            action = (env.settle['position'][row] - env.home) / env.scale
            steps, first, info = 0, None, {}
            while True:
                *_, terminated, truncated, info = env.step(action)
                steps += 1
                assert set(info['reward_components']) == {'settle_ori', 'settle_pose', 'settle_still', 'action_rate',
                                                          'joint_limit', 'self_contact', 'impact', 'sole_impact'}
                assert info['settle_nearest'] == row and info['hand_force'] >= 0
                if first is None and info['settle_done']:
                    first = steps
                    assert info['settle_count'] == env.settle_hold == 15 and info['settle_near']
                if terminated or truncated:
                    break
            assert truncated and not terminated and steps == SETTLE_STEPS
            assert first is not None and info['settle_done'] and info['settle_target'] == name
            assert info['settle_seconds'] == pytest.approx(first * env.dt) and first * env.dt > .5  # it fell first
            assert info['settle_angle'] < 15 and info['settle_deviation'] < .1  # rolled a little on landing
    finally:
        env.close()


LAB_CLIPS = ('cmu-140_09-61-571', 'cmu-140_01-151-571')


@pytest.mark.skipif(not all(g1_getup_track.clip_path(name).is_file() for name in LAB_CLIPS),
                    reason='the lab clips runs/humanoid_lab/getup_track/<name>.npz are made locally')
def test_the_lab_clips_start_on_the_back_and_on_the_front():
    """cmu-140_09 starts lying face up, cmu-140_01 (CMU «get up face down») face down: the pelvis gravity along its −x
    and +x; their targets are far apart, so a pose near one is far from the other."""
    clips = [g1_getup_track.load(name) for name in LAB_CLIPS]
    assert clips[1]['header']['source'] == '140_01.amc' and clips[1]['header']['frames'] == [151, 571]
    env = G1GetupEnv(bank=drop_bank(), settle_clip=[{'file': n, 'md5': c['md5']} for n, c in zip(LAB_CLIPS, clips)])
    try:
        assert env.settle['gravity'][0, 0] < -.8 and env.settle['gravity'][1, 0] > .8
        env.data.qpos[:], env.data.qvel[:] = clips[1]['qpos'][0], 0
        mujoco.mj_forward(env.model, env.data)
        angle, pose, _ = env.settle_errors()
        assert angle[0, 1] < 1e-6 and pose[0, 1] < 1e-12 and math.degrees(angle[0, 0]) > 150
        assert task.settle_nearest(angle, pose)[0] == 1
    finally:
        env.close()


def test_the_evaluation_metrics_and_a_stage_restore_keep_no_assist():
    from humanoid_sim.tasks.evaluation import getup_metrics
    metrics = getup_metrics({'start_class': 'side'}, dict(getup_held=None, settle_done=True, settle_seconds=1.2,
                                                         settle_target='test-back'), False)
    assert metrics['success'] and metrics['settle_done'] and metrics['settle_s'] == 1.2
    assert metrics['settle_target'] == 'test-back' and metrics['start_class'] == 'side'
    assert getup_metrics({}, dict(getup_held=True), False) == dict(success=True, getup_s=None, head_m=None,
                                                                   start_class=None)
    env = settle_env()
    try:
        env.restore_training_state({'assist_force': [200., 140.]})
        assert env.assist == 0.
        assert env.training_state() == {'assist_force': [0.]}
    finally:
        env.close()


def test_package_checks_the_licence_of_every_settle_clip(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'robot'))
    package = importlib.import_module('package')
    monkeypatch.setattr(package, 'TRACK_DIR', tmp_path / 'clips')
    (tmp_path / 'clips').mkdir()
    settings = {}
    for name, dataset in (('cmu-a', 'cmu'), ('cmu-b', 'cmu'), ('lafan-a', 'lafan1')):
        path = tmp_path / 'clips' / f'{name}.npz'
        np.savez(path, header=np.array(json.dumps(dict(dataset=dataset, source=f'{name}.src', frames=[1, 2]))))
        settings[name] = {'file': name, 'md5': package.md5(path)}
    for clips, refused in ((['cmu-a', 'cmu-b'], False), (['cmu-a', 'lafan-a'], True)):
        stage = tmp_path / f'stage-{len(list(tmp_path.iterdir()))}'
        stage.mkdir()
        config = {'settle_clip': [settings[name] for name in clips]}
        (stage / 'metadata.json').write_text(json.dumps({'task_config': config}), encoding='utf-8')
        folder = tmp_path / f'export-{stage.name}'
        folder.mkdir()
        (folder / 'deploy.yaml').write_text(f'# humanoid_sim.export_g1: {stage} (100 steps)\n', encoding='utf-8')
        if refused:
            with pytest.raises(SystemExit, match='LAFAN1'):
                package.check_licence(folder)
        else:
            assert package.check_licence(folder)['file'] == 'cmu-a'
