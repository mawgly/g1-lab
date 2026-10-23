"""task_config arm_perturb_radians and arm_perturb_contact_mask (iteration 30): stronger arm perturbations, and their
arm contacts left out of the self_contact share while a segment runs."""
import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_foot_collision import crossed
from test_g1_self_contact import PROFILE, arms_pressed

ARMS = list(range(15, 29))
WEIGHTS = {'self_contact': 4.}


def draw(radians=None, limit=100., share=.5, n=4096, seed=11):
    """One arm_perturb_draw from outside every segment, with the uniform stream's state after it."""
    env = G1VelocityEnv(real_robot=True, action_limit=10.)
    scale = torch.from_numpy(env.scale)
    env.close()
    generator = torch.Generator().manual_seed(seed)
    uniform = lambda k: torch.rand(k, generator=generator, dtype=torch.float64)
    extra = () if radians is None else (radians,)
    steps, held = task.arm_perturb_draw(torch.zeros(n, dtype=torch.long), torch.zeros((n, 29), dtype=torch.float64),
                                        uniform, share, .02, scale, limit, *extra)
    return steps, held, generator.get_state(), scale


def test_the_default_amplitude_is_the_original_draw_bitwise():
    assert task.ARM_PERTURB_RADIANS == .4
    steps, held, state, _ = draw()
    for radians in (task.ARM_PERTURB_RADIANS, .4):
        other = draw(radians)
        assert torch.equal(other[0], steps) and torch.equal(other[1], held) and torch.equal(other[2], state)


@pytest.mark.parametrize('radians', (.1, 1., 1.6))
def test_the_amplitude_scales_the_pose_and_draws_the_same_segments(radians):
    steps, held, state, scale = draw()
    other_steps, other, other_state, _ = draw(radians)
    # The same random stream: the same segments start, last as long and hold home in the same worlds.
    assert torch.equal(other_steps, steps) and torch.equal(other_state, state)
    assert torch.equal((other == 0).all(dim=1), (held == 0).all(dim=1))
    # Only the pose is scaled: the same uniform values times radians / scale (the box ±100 clips nothing here).
    torch.testing.assert_close(other * scale / radians, held * scale / task.ARM_PERTURB_RADIANS, rtol=1e-12, atol=1e-15)
    started = steps > 0
    random = other[started & ~(other == 0).all(dim=1)] * scale
    assert float(random.abs().max()) <= radians * (1 + 1e-12) and float(random.abs().max()) > .99 * radians


def test_a_wide_amplitude_is_clipped_to_the_action_box():
    _, held, _, scale = draw(1., limit=10., n=20000)
    wide = (1. / scale > 10.)[ARMS]
    assert int(wide.sum()) == 4  # the wrist pitch and yaw joints (scale 0.07): ±14.3 against the box ±10
    held = held[:, ARMS]
    assert float(held[:, wide].abs().max()) == 10. and float(held[:, ~wide].abs().max()) < 10.
    assert .98 < float((held[:, ~wide] * scale[ARMS][~wide]).abs().max()) <= 1.


def test_training_segments_with_a_wider_amplitude_start_at_the_same_steps():
    """Two training episodes of one seed, amplitude 0.4 and 1.0: the same segments at the same steps (the draws do not
    depend on the amplitude or on the physics), each holding the same pose scaled by 2.5 (within the box ±10)."""
    envs = [G1VelocityEnv(max_episode_steps=10 ** 6, arm_perturb=.5, action_limit=10., arm_perturb_radians=radians)
            for radians in (.4, 1.)]
    try:
        for env in envs:
            env.reset(seed=1)
            assert env.perturb_live
        rng = np.random.default_rng(0)
        segments = 0
        for _ in range(600):
            action = rng.uniform(-.05, .05, 29)
            for env in envs:
                env.step(action)
            narrow, wide = envs
            assert narrow.perturb_steps == wide.perturb_steps
            np.testing.assert_allclose(wide.perturb_action[ARMS], np.clip(2.5 * narrow.perturb_action[ARMS], -10, 10),
                                       rtol=1e-12, atol=0)
            on = not np.array_equal(wide.previous_action[ARMS], action[ARMS])
            if on:
                np.testing.assert_array_equal(wide.previous_action[ARMS], wide.perturb_action[ARMS])
                segments += 1
        assert segments > 100
    finally:
        for env in envs:
            env.close()


def test_self_contact_pairs_is_the_original_rule_without_the_mask_and_the_same_on_numpy_and_torch():
    env = G1VelocityEnv(foot_collision=True, **PROFILE)
    try:
        model = env.model
        rng = np.random.default_rng(5)
        geom = rng.integers(0, model.ngeom, (5000, 2)).astype(np.int32)  # as data.contact.geom
        masked = rng.random(5000) < .5
        plain = task.self_contact_pairs(geom, env.robot_geoms)
        np.testing.assert_array_equal(plain, env.robot_geoms[geom].all(axis=1))  # the rule before the setting
        expected = np.array([env.robot_geoms[a] and env.robot_geoms[b] and not (m and (env.arm_geoms[a] or env.arm_geoms[b]))
                             for (a, b), m in zip(geom, masked)])
        for flags in (masked, torch.from_numpy(masked)):  # numpy (the CPU task) and torch (the CUDA task's slots)
            arrays = (geom, env.robot_geoms, env.arm_geoms) if isinstance(flags, np.ndarray) else (
                torch.from_numpy(geom).long(), torch.from_numpy(env.robot_geoms), torch.from_numpy(env.arm_geoms))
            result = task.self_contact_pairs(*arrays, flags)
            np.testing.assert_array_equal(np.asarray(result), expected)
        # One world on the CPU task: a bool for all its contacts.
        np.testing.assert_array_equal(task.self_contact_pairs(geom, env.robot_geoms, env.arm_geoms, False), plain)
        np.testing.assert_array_equal(task.self_contact_pairs(geom, env.robot_geoms, env.arm_geoms, True),
                                      plain & ~env.arm_geoms[geom].any(axis=1))
        # By body, in a segment: an arm link with anything is left out; legs, feet, torso and pelvis count; the floor
        # never.
        first = lambda body: next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(body).id
                                  and (model.geom_contype[g] or model.geom_conaffinity[g]))
        left, right = (int(foot[0]) for foot in env.foot_capsules)
        names = {'torso': first('torso_link'), 'pelvis': first('pelvis'), 'thigh': first('left_hip_pitch_link'),
                 'knee': first('right_knee_link'), 'shoulder': first('left_shoulder_yaw_link'),
                 'elbow': first('right_elbow_link'), 'hand': first('left_wrist_yaw_link'), 'left_foot': left,
                 'right_foot': right, 'floor': model.geom('floor').id}
        counted = {('torso', 'thigh'): True, ('pelvis', 'knee'): True, ('left_foot', 'right_foot'): True,
                   ('torso', 'knee'): True, ('shoulder', 'torso'): False, ('hand', 'thigh'): False,
                   ('elbow', 'pelvis'): False, ('shoulder', 'hand'): False, ('left_foot', 'hand'): False,
                   ('floor', 'left_foot'): False}
        pairs = np.array([[names[a], names[b]] for a, b in counted], dtype=np.int32)
        np.testing.assert_array_equal(task.self_contact_pairs(pairs, env.robot_geoms, env.arm_geoms, True),
                                      list(counted.values()))
        outside = task.self_contact_pairs(pairs, env.robot_geoms, env.arm_geoms, False)  # outside a segment: all
        np.testing.assert_array_equal(outside, [a != 'floor' for a, _ in counted])
        arm_bodies = {model.body(int(b)).name for b in model.geom_bodyid[env.arm_geoms]}
        assert arm_bodies == {f'{side}_{link}_link' for side in ('left', 'right') for link in (
            'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')}
    finally:
        env.close()


def test_the_legs_only_g1_has_no_arm_links_to_mask():
    env = G1VelocityEnv(legs_only=True, arm_perturb=.5, arm_perturb_contact_mask=True)
    try:
        assert not env.arm_geoms.any()  # its arm links are welded to the torso: no arm joints, nothing perturbed
    finally:
        env.close()


def pair(**settings):
    """A masked environment and its unmasked twin with the same settings (a reward change only: the same physics)."""
    common = dict(reward_weights=WEIGHTS, arm_perturb=.1, foot_collision=True, **PROFILE, **settings)
    return G1VelocityEnv(arm_perturb_contact_mask=True, **common), G1VelocityEnv(**common)


def steps(masked, plain, action, count):
    """count steps of both under one action: (masked self_contact, plain self_contact) per step; everything else equal."""
    rows = []
    for _ in range(count):
        (obs, reward, *_, info), (obs_p, reward_p, *_, info_p) = masked.step(action), plain.step(action)
        np.testing.assert_array_equal(obs, obs_p)
        np.testing.assert_array_equal(masked.data.qpos, plain.data.qpos)
        mine, theirs = info['reward_components'], info_p['reward_components']
        assert {k: v for k, v in mine.items() if k != 'self_contact'} == {
            k: v for k, v in theirs.items() if k != 'self_contact'}
        assert reward == pytest.approx(reward_p - theirs['self_contact'] + mine['self_contact'], rel=1e-12, abs=1e-12)
        rows.append((mine['self_contact'], theirs['self_contact']))
    return rows


def test_arm_contacts_in_a_segment_are_not_charged_outside_one_they_are():
    """The upper arms pressed on the torso: chosen by the policy, charged as before; held by a perturbation segment
    (the policy's arms at home), left out of the share; the segment over, charged again."""
    masked, plain = pair()
    try:
        for env in (masked, plain):
            env.reset(seed=3, options={'command': [0., 0., 0.]})
        rows = steps(masked, plain, arms_pressed(), 8)  # the policy presses: no segment, the same charge
        assert [m for m, _ in rows] == [p for _, p in rows] and rows[-1] == (-4., -4.)
        assert not masked.arm_contacts_masked
        for env in (masked, plain):
            env.set_arm_perturb(6, arms_pressed())
        rows = steps(masked, plain, np.zeros(29), 6)  # the segment holds the same press
        assert [p for _, p in rows] == [-4.] * 6 and [m for m, _ in rows] == [0.] * 6
        rows = steps(masked, plain, arms_pressed(), 3)  # over: the policy presses again
        assert not masked.arm_contacts_masked and rows == [(-4., -4.)] * 3
        assert float(masked.data.qpos[2]) > .7  # still standing
    finally:
        masked.close(), plain.close()


def test_foot_contacts_in_a_segment_are_charged_as_before():
    """The feet driven into each other in the air (test_g1_foot_collision.cross_in_the_air) while a segment holds the
    arms at home: foot capsule against foot capsule is not an arm contact, so it is charged on both."""
    masked, plain = pair()
    try:
        for env in (masked, plain):
            env.reset(seed=4, options={'command': [0., 0., 0.]})
            env.data.qpos[env.qadr] = env.home + crossed(env, .12)
            env.data.qpos[2] = .9
            mujoco.mj_forward(env.model, env.data)
            env.set_arm_perturb(6, np.zeros(29))
        flags = []
        rows = []
        for _ in range(6):
            rows += steps(masked, plain, crossed(masked, .25) / masked.scale, 1)
            flags.append(masked.arm_contacts_masked)
        assert flags == [True] * 6
        assert [m for m, _ in rows] == [p for _, p in rows] and rows[0] == (0., 0.) and rows[-1] == (-4., -4.)
    finally:
        masked.close(), plain.close()


def test_the_mask_reads_the_segment_of_each_step_and_training_resets_too():
    """In a training episode (segments drawn) the mask is on exactly in the perturbed steps; evaluation resets (a
    command given) and set_command run none."""
    env = G1VelocityEnv(max_episode_steps=10 ** 6, arm_perturb=.5, arm_perturb_contact_mask=True,
                        reward_weights=WEIGHTS)
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(300):
            before = env.perturb_steps
            action = rng.uniform(-.05, .05, 29)
            env.step(action)
            perturbed = not np.array_equal(env.previous_action[ARMS], action[ARMS])
            assert env.arm_contacts_masked == perturbed == (before > 0 or env.perturb_steps > 0)
            seen.add(perturbed)
        assert seen == {True, False}
        env.reset(seed=2, options={'command': [.5, 0., 0.]})
        for _ in range(50):
            env.step(rng.uniform(-.05, .05, 29))
            assert not env.arm_contacts_masked
    finally:
        env.close()


def test_without_the_settings_the_step_is_the_original_bitwise():
    """The defaults given explicitly change nothing: the same step, observation and reward, segments included."""
    profile = dict(reward_weights={**WEIGHTS, 'arm_press': .1}, arm_perturb=.5, foot_collision=True, **PROFILE)
    plain = G1VelocityEnv(**profile)
    given = G1VelocityEnv(arm_perturb_radians=task.ARM_PERTURB_RADIANS, arm_perturb_contact_mask=False, **profile)
    try:
        np.testing.assert_array_equal(plain.reset(seed=5)[0], given.reset(seed=5)[0])
        rng = np.random.default_rng(2)
        for _ in range(40):
            action = arms_pressed() + rng.normal(0, .3, 29)
            expected, result = plain.step(action), given.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1] and result[4]['reward_components'] == expected[4]['reward_components']
            np.testing.assert_array_equal(given.data.qpos, plain.data.qpos)
            assert given.perturb_steps == plain.perturb_steps
    finally:
        plain.close(), given.close()


def test_the_settings_are_bounded_labelled_need_arm_perturb_and_keep_the_identity():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import ARM_PERTURB_RADIANS_RANGE, G1_PARAMETERS
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    config = validate_config(base)['task_config']
    assert config['arm_perturb_radians'] == .4 and config['arm_perturb_contact_mask'] is False  # old branches
    assert validate_config({**base, 'task_config': config})['task_config'] == config  # a saved config validates again
    asked = {'arm_perturb': .25, 'arm_perturb_radians': 1., 'arm_perturb_contact_mask': True}
    accepted = validate_config({**base, 'task_config': asked})['task_config']
    assert {key: accepted[key] for key in asked} == asked
    assert ARM_PERTURB_RADIANS_RANGE == (.1, 1.6)
    for value in (.1, 1.6, 1):
        assert validate_config({**base, 'task_config': {'arm_perturb': .1, 'arm_perturb_radians': value}})[
            'task_config']['arm_perturb_radians'] == value
    for bad in (.09, 1.61, float('nan'), True, '1.0', None):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'arm_perturb': .1, 'arm_perturb_radians': bad}})
    for bad in (1, 'true', None):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'arm_perturb': .1, 'arm_perturb_contact_mask': bad}})
    for alone in ({'arm_perturb_radians': 1.}, {'arm_perturb_contact_mask': True}):  # nothing to act on
        with pytest.raises(ValueError, match='requires arm_perturb'):
            validate_config({**base, 'task_config': alone})
    for bad in (.09, 1.7, float('nan')):
        with pytest.raises(ValueError):
            G1VelocityEnv(arm_perturb_radians=bad)
    fields = {p['key']: p for p in G1_PARAMETERS}
    radians, mask = fields['arm_perturb_radians'], fields['arm_perturb_contact_mask']
    assert (radians['default'], radians['min'], radians['max']) == (.4, .1, 1.6) and 'рук' in radians['label']
    assert mask['type'] == 'bool' and mask['default'] is False and 'касания рук' in mask['label']
    keys = [p['key'] for p in G1_PARAMETERS]
    assert keys.index('arm_perturb_radians') == keys.index('arm_perturb') + 1
    assert keys.index('arm_perturb_contact_mask') == keys.index('arm_perturb') + 2
    # Reward and training draws only: the checkpoint identity is the parent's, so a branch resumes.
    parent = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., lock_wrists=True,
                  knee_floor=.061, foot_collision=True, height_command=True, upper_body_slot=True, arm_perturb=.1)
    saved = task_identity('g1_29dof', 'g1_velocity', parent)
    branched = task_identity('g1_29dof', 'g1_velocity', {**parent, **asked})
    assert branched == saved and saved['observation_schema'].endswith('-hcmd-ubslot-critic4')
    validate_resume(saved, branched)
