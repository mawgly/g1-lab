"""g1_getup task_config track_clip (iteration 64), CPU: motion tracking of one human clip getting up (BeyondMimic /
mjlab). The clip (tasks/g1_getup_track) against the model's kinematics; the tracking errors, terms, terminations and
critic inputs (g1_tensor_math) against a hand calculation, numpy and torch alike; the task's starts, clock, episode end
and logs; its own critic contract (…-critic6-track11), the actor's unchanged; scripts/robot/package.py refuses such a
stage. Off (missing or null) the task is bit for bit the old one. CPU–GPU: tests/test_g1_gpu.py."""
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
from humanoid_sim.tasks.config import G1_GETUP_TRACK_WEIGHTS, GETUP_CRITIC_EXTRAS, TRACK_CRITIC_EXTRAS
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import DROP, OLD_DIGESTS, OPTIONS, drop_bank, getup_digest

MD5 = '0' * 32
SETTING = {'file': 'test-clip', 'md5': MD5}
FRAMES = 40
REAL = 'lafan1-fallAndGetUp1_subject1-2104-2209'


def synthetic_clip(frames=FRAMES):
    """A clip made by g1_getup_track.build: drop_bank's back row pressed 3 cm into the floor, moving 0.3 m forward
    while every joint swings by up to 0.3 rad and the pelvis rolls 0.4 rad (smooth, so central differences hold)."""
    env = bank_module.getup_model_env()
    start = drop_bank()['qpos'][0].copy()
    start[2] -= DROP + .03
    t = np.linspace(0., 1., frames)
    qpos = np.repeat(start[None], frames, 0)
    qpos[:, 0] += .3 * t
    swing = np.random.default_rng(3).uniform(-.3, .3, 29)
    qpos[:, 7:] += np.sin(np.pi * t)[:, None] * swing
    for k, angle in enumerate(.4 * t):
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([1., 0., 0.]), angle)
        mujoco.mju_mulQuat(qpos[k, 3:7], turn, start[3:7])
    return g1_getup_track.build(env, qpos)


def track_env(clip=None, **settings):
    return G1GetupEnv(bank=drop_bank(), track_clip=SETTING, track_bank=synthetic_clip() if clip is None else clip,
                      **settings)


def test_the_setting_is_validated_labelled_and_its_own_critic_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_clip'] is None  # old stages: off
    on = validate_config({**base, 'task_config': {'track_clip': SETTING}})['task_config']
    assert on['track_clip'] == SETTING
    for value in ({'file': 'x'}, {'file': 'x', 'md5': 'abc'}, {'file': '../x', 'md5': MD5}, {'file': 'a/b', 'md5': MD5},
                  {'file': 'x.npz', 'md5': MD5}, {'file': 'x', 'md5': MD5, 'w': 1}, ['x', MD5], 'x', 5, True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'track_clip': value}})
    for other in ({'pd_bound': [1., .6]}, {'human_starts': {'share': .3, 'md5': MD5}}, {'style_judge': True},
                  {'reward_weights': {'style': .5}}, {'reward_weights': {'amp': 1.}}):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'track_clip': SETTING, **other}})
    plain = task_identity('g1_29dof', 'g1_getup', {})
    assert task_identity('g1_29dof', 'g1_getup', {'track_clip': None}) == plain
    tracking = task_identity('g1_29dof', 'g1_getup', {'track_clip': SETTING})
    assert tracking['observation_schema'] == plain['observation_schema'] + '-track11'
    assert plain['observation_schema'].endswith('-critic6')
    assert {k: v for k, v in tracking.items() if k != 'observation_schema'} == {
        k: v for k, v in plain.items() if k != 'observation_schema'}
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['track_clip']
    assert field['group'] is None and field['default'] is None and 'записи человека' in field['label']
    assert TRACK_CRITIC_EXTRAS == 11 and G1_GETUP_TRACK_WEIGHTS == dict(
        anchor_pos=.5, anchor_ori=.5, body_pos=1., body_ori=1., body_lin_vel=1., body_ang_vel=1., action_rate=.1,
        joint_limit=10., self_contact=10.)
    assert task.TRACK_SIGMA == dict(anchor_pos=.3, anchor_ori=.4, body_pos=.3, body_ori=.4, body_lin_vel=1.,
                                    body_ang_vel=3.14)


def test_off_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('a track clip read with the setting off')
    monkeypatch.setattr(g1_getup_track, 'load', refuse)
    bank = drop_bank()
    for config, name in (({'track_clip': None}, 'plain'), ({'pd_bound': [1., .6], 'track_clip': None}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config
    env = G1GetupEnv(bank=bank)
    try:
        assert env.clip is None and env.observation_space.shape == (env.actor_inputs + GETUP_CRITIC_EXTRAS,)
    finally:
        env.close()


def _check_kinematics(clip, frames):
    """The clip's bodies at frames are mj_forward's of its qpos / qvel: frame origins, rotations and their world
    velocities from cvel (point velocity about the root's subtree_com, as mj_objectVelocity of the body frame, XBODY)."""
    env = bank_module.getup_model_env()
    m, d = env.model, env.data
    bodies = [m.body(name).id for name in task.TRACK_BODIES]
    velocity = np.zeros(6)
    for k in frames:
        d.qpos[:], d.qvel[:] = clip['qpos'][k], clip['qvel'][k]
        mujoco.mj_forward(m, d)
        for b, body in enumerate(bodies):
            np.testing.assert_allclose(clip['body_pos'][k, b], d.xpos[body], atol=1e-12)
            np.testing.assert_allclose(clip['body_rot'][k, b], d.xmat[body].reshape(3, 3), atol=1e-12)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)  # world axes, frame origin
            np.testing.assert_allclose(clip['body_ang'][k, b], velocity[:3], atol=1e-9)
            np.testing.assert_allclose(clip['body_lin'][k, b], velocity[3:], atol=1e-9)


def test_the_clip_is_the_models_kinematics_lifted_out_of_the_floor_with_the_wrists_at_home():
    """build: the locked wrists at home; every frame raised by at least its own floor depth (a running maximum, then a
    running mean), so nothing is left in the floor; qvel central differences; the bodies' positions, rotations and
    velocities those of mj_forward (and their velocities the central differences of the positions within the clip's
    curvature)."""
    env = bank_module.getup_model_env()
    clip = synthetic_clip()
    assert clip['raw_lift'].max() > .025 and (clip['lift'] >= clip['raw_lift'] - 1e-12).all()
    np.testing.assert_array_equal(clip['qpos'][:, 7:][:, env.wrist_mask.numpy()], 0.)
    m, d = env.model, env.data
    floor = m.geom('floor').id
    for row in clip['qpos']:
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        assert g1_getup_track.floor_depth(m, d, env.robot_geoms, floor) > -1e-4
    np.testing.assert_allclose(clip['qvel'][1:-1, :3], (clip['qpos'][2:, :3] - clip['qpos'][:-2, :3]) / (2 * env.dt))
    _check_kinematics(clip, range(FRAMES))
    dt = env.dt
    central = (clip['body_pos'][2:] - clip['body_pos'][:-2]) / (2 * dt)
    np.testing.assert_allclose(clip['body_lin'][1:-1], central, atol=.03)
    spin = clip['body_rot'][2:] @ np.swapaxes(clip['body_rot'][:-2], -1, -2)
    vee = np.stack([spin[..., 2, 1] - spin[..., 1, 2], spin[..., 0, 2] - spin[..., 2, 0],
                    spin[..., 1, 0] - spin[..., 0, 1]], -1) / 2 / (2 * dt)
    np.testing.assert_allclose(clip['body_ang'][1:-1], vee, atol=.05)
    frames = g1_getup_track.hold_frames(clip)
    assert frames['pos'].shape == (FRAMES + 1, 14, 3)
    np.testing.assert_array_equal(frames['pos'][-1], clip['body_pos'][-1])
    np.testing.assert_array_equal(frames['lin'][-1], 0.)
    np.testing.assert_array_equal(frames['ang'][-1], 0.)


@pytest.mark.skipif(not g1_getup_track.clip_path(REAL).is_file(),
                    reason='the lab clip runs/humanoid_lab/getup_track/<name>.npz is made locally')
def test_the_lab_clip_is_the_models_kinematics():
    clip = g1_getup_track.load(REAL)
    assert clip['header']['source'] == 'fallAndGetUp1_subject1.csv' and clip['header']['frames'] == [2104, 2209]
    assert len(clip['qpos']) == 176 and len(clip['header']['source_md5']) == 32
    _check_kinematics(clip, range(0, 176, 7))


def _rotations(rng, shape):
    quat = rng.normal(size=(*shape, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    out = np.zeros((*shape, 9))
    for index in np.ndindex(*shape):
        mujoco.mju_quat2Mat(out[index], quat[index])
    return out.reshape(*shape, 3, 3)


def _angle(a, b):
    """The rotation angle of aᵀb by the trace (float64, away from π)."""
    m = np.swapaxes(a, -1, -2) @ b
    return np.arccos(np.clip((np.trace(m, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def test_the_tracking_errors_terms_and_critic_inputs_by_hand_in_numpy_and_torch():
    rng = np.random.default_rng(0)
    n, bodies, anchor = 16, 14, 7
    position, ref_position = rng.normal(size=(n, bodies, 3)), rng.normal(size=(n, bodies, 3))
    rotation, ref_rotation = _rotations(rng, (n, bodies)), _rotations(rng, (n, bodies))
    linear, angular, ref_linear, ref_angular = (rng.normal(size=(n, bodies, 3)) for _ in range(4))
    errors = task.track_errors(position, rotation, linear, angular, ref_position, ref_rotation, ref_linear,
                               ref_angular, anchor)
    for w in range(n):
        between = rotation[w, anchor] @ ref_rotation[w, anchor].T
        yaw = math.atan2(between[1, 0], between[0, 0])
        turn = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        base = np.array([position[w, anchor, 0], position[w, anchor, 1], ref_position[w, anchor, 2]])
        relative = base + (ref_position[w] - ref_position[w, anchor]) @ turn.T
        assert errors['anchor_pos'][w] == pytest.approx(((ref_position[w, anchor] - position[w, anchor]) ** 2).sum())
        assert errors['anchor_ori'][w] == pytest.approx(_angle(ref_rotation[w, anchor], rotation[w, anchor]) ** 2,
                                                        abs=1e-9)
        assert errors['body_pos'][w] == pytest.approx(((relative - position[w]) ** 2).sum(-1).mean())
        assert errors['body_ori'][w] == pytest.approx((_angle(turn @ ref_rotation[w], rotation[w]) ** 2).mean(),
                                                      abs=1e-9)
        assert errors['body_lin_vel'][w] == pytest.approx(((ref_linear[w] - linear[w]) ** 2).sum(-1).mean())
        assert errors['body_ang_vel'][w] == pytest.approx(((ref_angular[w] - angular[w]) ** 2).sum(-1).mean())
    # The angle is exact near zero (float32: no arccos floor of ~5e-4 rad).
    small = _rotations(rng, (1,))[0]
    tilt = np.zeros(9)
    mujoco.mju_quat2Mat(tilt, np.array([math.cos(5e-5), math.sin(5e-5), 0., 0.]))
    near = torch.tensor(small @ tilt.reshape(3, 3), dtype=torch.float32)
    assert float(task.rotation_angle(torch.tensor(small, dtype=torch.float32)[None], near[None])[0]) \
        == pytest.approx(1e-4, rel=.05)
    # The terms.
    nu = 29
    action, previous, q = rng.uniform(-3, 3, (n, nu)), rng.uniform(-3, 3, (n, nu)), rng.uniform(-3, 3, (n, nu))
    low, high = -np.ones(nu), np.ones(nu)
    share, impact, sole = rng.random(n), rng.random(n), rng.random(n)
    weights = {**G1_GETUP_TRACK_WEIGHTS, 'impact': 2., 'sole_impact': 1.}
    terms = task.track_reward_components(errors, action, previous, q, low, high, weights, self_contact=share,
                                         impact=impact, sole_impact=sole)
    for key, sigma in task.TRACK_SIGMA.items():
        np.testing.assert_allclose(terms[key], weights[key] * np.exp(-errors[key] / sigma ** 2))
    np.testing.assert_allclose(terms['action_rate'], -.1 * ((action - previous) ** 2).sum(-1))
    np.testing.assert_allclose(terms['joint_limit'], -10. * (np.clip(np.abs(q) - 1, 0, None)).sum(-1))
    np.testing.assert_allclose(terms['self_contact'], -10. * share)
    np.testing.assert_allclose(terms['impact'], -2. * impact)
    np.testing.assert_allclose(terms['sole_impact'], -1. * sole)
    off = task.track_reward_components(errors, action, previous, q, low, high, weights)
    assert not any(off[key].any() for key in ('self_contact', 'impact', 'sole_impact'))
    # Critic inputs: phase, the reference anchor in the robot anchor's frame, its rotation's first two columns there.
    clock = np.arange(n) * 20
    extras = task.track_critic_extras(clock, 175, position[:, anchor], rotation[:, anchor], ref_position[:, anchor],
                                      ref_rotation[:, anchor])
    assert extras.shape == (n, TRACK_CRITIC_EXTRAS)
    for w in range(n):
        phase = math.pi * min(clock[w] / 175, 1)
        np.testing.assert_allclose(extras[w, :2], [math.sin(phase), math.cos(phase)], atol=1e-12)
        np.testing.assert_allclose(extras[w, 2:5], rotation[w, anchor].T @ (ref_position[w, anchor] - position[w, anchor]))
        relative = rotation[w, anchor].T @ ref_rotation[w, anchor]
        np.testing.assert_allclose(extras[w, 5:], relative[:, :2].ravel())
    # torch gives the same numbers.
    tensors = [torch.from_numpy(a) for a in (position, rotation, linear, angular, ref_position, ref_rotation,
                                             ref_linear, ref_angular)]
    for key, value in task.track_errors(*tensors, anchor).items():
        np.testing.assert_allclose(value.numpy(), errors[key], rtol=1e-10, atol=1e-12, err_msg=key)
    torch_terms = task.track_reward_components(
        task.track_errors(*tensors, anchor), *(torch.from_numpy(a) for a in (action, previous, q, low, high)), weights,
        self_contact=torch.from_numpy(share), impact=torch.from_numpy(impact), sole_impact=torch.from_numpy(sole))
    for key, value in torch_terms.items():
        np.testing.assert_allclose(value.numpy(), terms[key], rtol=1e-10, atol=1e-12, err_msg=key)
    np.testing.assert_allclose(task.track_critic_extras(torch.from_numpy(clock), 175, *(tensors[i][:, anchor] for i in (
        0, 1, 4, 5))).numpy(), extras, atol=1e-12)


def test_the_terminations_by_hand_and_the_bank_starts_grace():
    """Each of mjlab's three rules alone ends an episode just past its threshold and not just below it; the ankles' and
    hands' rule waits where ends_active is false."""
    n, bodies, anchor = 1, 14, 7
    ends = tuple(task.TRACK_BODIES.index(name) for name in task.TRACK_END_BODIES)
    base = np.zeros((n, bodies, 3))
    rotation = np.repeat(np.eye(3)[None, None], bodies, 1)

    def ended(position=base, turned=rotation, active=None):
        return bool(task.track_terminated(position, turned, base, rotation, anchor, ends, active)[0])
    assert not ended()
    for delta, expected in ((.24, False), (.26, True), (-.26, True)):
        moved = base.copy()
        moved[0, anchor, 2] += delta
        assert ended(moved) is expected
        limb = base.copy()
        limb[0, ends[3], 2] += delta
        assert ended(limb) is expected and not ended(limb, active=np.array([False]))
        other = base.copy()
        other[0, 1, 2] += 1.  # not an end body: only the anchor and the ends count
        assert not ended(other)
    for angle, expected in ((math.acos(.21), False), (math.acos(.19), True)):  # 1 − cos: 0.79 / 0.81
        tilted = rotation.copy()
        tilted[0, anchor] = [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
        assert ended(turned=tilted) is expected


def test_reset_starts_clock_and_episode_end():
    """A clip start without noise is the clip frame exactly (its velocities with track_frame, zero for the first frame
    of an evaluation reset), the observation's 17 critic extras end with the phase and a zero anchor error; following
    the clip's next frames the clock counts on, every step carries the track info, the reference holds the last frame
    (zero velocities) after the clip and the episode is truncated TRACK_HOLD_SECONDS after it; training resets start
    0.6 / 0.3 / 0.1 from the first frame, a uniform frame and the bank's back rows, evaluation ones from the first
    frame with noise; a bank start aligns the reference's first pelvis pose with the robot's."""
    env = track_env()
    try:
        assert env.observation_space.shape == (env.actor_inputs + GETUP_CRITIC_EXTRAS + TRACK_CRITIC_EXTRAS,)
        assert env.clip_last == FRAMES - 1 and env.clip_end == FRAMES - 1 + 150 and env.assist == 0.
        obs, info = env.reset(seed=1, options=dict(OPTIONS, track_frame=10, track_noise=False))
        assert info['track_start'] == 'frame' and info['track_frame'] == 10 and env.clip_clock == 10
        np.testing.assert_allclose(env.data.qpos, env.clip['qpos'][10], atol=1e-4)  # raised by what is left in the floor
        np.testing.assert_allclose(env.data.qvel, env.clip['qvel'][10], atol=1e-12)
        assert obs.shape == env.observation_space.shape
        extras = obs[-TRACK_CRITIC_EXTRAS:]
        phase = math.pi * 10 / env.clip_last
        np.testing.assert_allclose(extras[:2], [math.sin(phase), math.cos(phase)], atol=1e-6)
        np.testing.assert_allclose(extras[2:5], 0., atol=1e-4)
        np.testing.assert_allclose(extras[5:], [1, 0, 0, 1, 0, 0], atol=1e-6)
        assert obs[env.actor_inputs + GETUP_CRITIC_EXTRAS - 1] == 0.  # the assist force
        steps, info = 0, {}
        while True:
            frame = min(env.clip_clock + 1, env.clip_last)
            *_, terminated, truncated, info = env.step((env.clip['qpos'][frame, 7:] - env.home) / env.scale)
            steps += 1
            assert info['track_clock'] == 10 + steps and set(info['track_errors']) == set(task.TRACK_SIGMA)
            assert set(info['reward_components']) == {*task.TRACK_SIGMA, 'action_rate', 'joint_limit', 'self_contact',
                                                      'impact', 'sole_impact'}
            assert info['hand_force'] >= 0
            if terminated or truncated:
                break
        assert not info['track_failed']  # lying, the PD follows the slow synthetic clip
        assert truncated and env.clip_clock == env.clip_end and steps == env.clip_end - 10 and info['track_done']
        assert info['track_standing'] is False  # the clip's end lying: done, but not up
        _, _, _, angular = env._clip_reference()
        np.testing.assert_array_equal(angular, 0.)  # the hold frame
        obs, info = env.reset(seed=2, options=dict(assist_force=0.))
        assert info['track_start'] == 'clip' and env.clip_clock == 0 and info['start_class'] == 'clip'
        assert np.abs(env.data.qpos[:2] - env.clip['qpos'][0, :2]).max() <= task.TRACK_START_NOISE['xy'] + 1e-12
        assert 0 < np.abs(env.data.qpos[7:] - env.clip['qpos'][0, 7:]).max() <= task.TRACK_START_NOISE['joints'] + 1e-9
        counts = dict.fromkeys(task.TRACK_START_KINDS, 0)
        frames = set()
        for _ in range(3000):
            _, info = env.reset()
            counts[info['track_start']] += 1
            if info['track_start'] == 'frame':
                frames.add(info['track_frame'])
            elif info['track_start'] == 'bank':
                assert info['start'] == 0 and info['start_class'] == 'back'
        for kind, share in zip(task.TRACK_START_KINDS, task.TRACK_START_SHARES):
            assert abs(counts[kind] / 3000 - share) < .03, counts
        assert frames == set(range(FRAMES))
        env.reset(seed=3, options=dict(OPTIONS, start=0))
        position, rotation, _, _ = env._clip_reference()
        np.testing.assert_allclose(position[0, 0, :2], env.data.xpos[env.pelvis, :2], atol=1e-9)
        robot = task.track_heading(env.data.xmat[env.pelvis].reshape(1, 3, 3))
        turn = float(task.track_heading(rotation[:, 0])[0] - robot[0])
        assert math.cos(turn) == pytest.approx(1., abs=1e-12) and abs(math.sin(turn)) < 1e-9
        assert env.clip_start == 'bank' and env.clip_clock == 0
    finally:
        env.close()


def test_track_standing_is_the_getting_up_standing_state_of_the_step():
    """info track_standing is task.getup_standing of the step (head top above 1.15 m, tilt below 15°, pelvis above
    0.70 m, both soles down), which the probe reads at the end of the hold: a clip standing at home starts standing;
    the same state with the head 0.2 m lower (one knee, ~1.05 m) is not; the lying synthetic clip ends its hold
    track_done but not standing (test_reset_starts_clock_and_episode_end)."""
    env0 = bank_module.getup_model_env()
    row = np.zeros(36)
    row[2], row[3] = .5, 1.  # pressed into the floor: build lifts it onto the soles
    row[np.asarray(env0.qadr)] = env0.home
    standing_clip = g1_getup_track.build(env0, np.repeat(row[None], 20, 0))
    env = track_env(standing_clip)
    try:
        env.reset(seed=0, options=dict(OPTIONS, track_frame=0, track_noise=False))
        *_, terminated, truncated, info = env.step(np.zeros(29))
        assert info['track_standing'] is True and not terminated
        head = env._head()
        height, rotation = env.data.qpos[2:3], env.data.xmat[env.torso].reshape(1, 3, 3)
        soles = env._sole_height().numpy()
        assert bool(task.getup_standing(head, height, rotation, soles)[0])
        assert not task.getup_standing(head - .2, height, rotation, soles)[0]
    finally:
        env.close()


def test_a_noisy_start_is_raised_out_of_the_floor_and_a_tilt_ends_the_episode():
    """A clip start tilted by its noise is lifted by its deepest floor contact (nothing left in the floor); a robot
    standing while the reference lies (the anchor's height and gravity) ends with track_failed on its first step."""
    env = track_env()
    try:
        lifts = []
        for seed in range(20):
            _, info = env.reset(seed=seed, options=dict(assist_force=0.))
            lifts.append(info['track_lift'])
            assert g1_getup_track.floor_depth(env.model, env.data, env.robot_geoms, env.floor) > -1e-4
        assert max(lifts) > .01
        env.reset(seed=0, options=dict(OPTIONS, track_frame=0, track_noise=False))
        env.data.qpos[2:7] = [.8, 1., 0., 0., 0.]  # standing upright while the clip lies on its back
        env.data.qpos[env.qadr] = env.home
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        *_, terminated, _, info = env.step(np.zeros(29))
        assert terminated and info['track_failed'] and not info['track_done']
    finally:
        env.close()


def test_package_refuses_a_stage_that_followed_a_clip(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'robot'))
    package = importlib.import_module('package')
    for config, refused in (({'track_clip': SETTING}, True), ({'track_clip': None}, False), ({}, False),
                            ({'reward_weights': {'style': .5}}, True)):
        stage = tmp_path / f'stage-{len(list(tmp_path.iterdir()))}'
        stage.mkdir()
        (stage / 'metadata.json').write_text(json.dumps({'task_config': config}), encoding='utf-8')
        folder = tmp_path / f'export-{stage.name}'
        folder.mkdir()
        (folder / 'deploy.yaml').write_text(f'# humanoid_sim.export_g1: {stage} (100 steps)\n', encoding='utf-8')
        if refused:
            with pytest.raises(SystemExit, match='LAFAN1'):
                package.check_licence(folder)
        else:
            package.check_licence(folder)
