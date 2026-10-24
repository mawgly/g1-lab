"""task_config hand_collision (iteration 33): Unitree's hand capsules, colliding with the floor and the robot."""
import mujoco
import numpy as np
import pytest

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks.g1_velocity import (ASSETS, HAND_CAPSULE_RADIUS, HAND_CAPSULES, G1VelocityEnv, load_model)
from humanoid_sim.tasks.identity import task_identity
from test_g1_foot_collision import REAL, mjb

PROFILES = ({}, REAL)


def hand_contacts(env, floor=None):
    """(geom1, geom2, force norm N, contact dimension) of this state's contacts that touch a hand capsule; floor: only
    those with the floor (True) or only those with the robot (False)."""
    hands, force, found, ground = set(env.hand_capsules.tolist()), np.zeros(6), [], env.model.geom('floor').id
    for k in range(env.data.ncon):
        pair = tuple(int(g) for g in env.data.contact.geom[k])
        if hands & set(pair) and (floor is None or (ground in pair) == floor):
            mujoco.mj_contactForce(env.model, env.data, k, force)
            found.append((*pair, float(np.linalg.norm(force[:3])), int(env.data.contact.dim[k])))
    return found


def capsule_bottom(env, g):
    """World z of the lowest point of capsule geom g."""
    half = env.data.geom_xmat[g].reshape(3, 3)[:, 2] * env.model.geom_size[g, 1]
    return min(env.data.geom_xpos[g, 2] - half[2], env.data.geom_xpos[g, 2] + half[2]) - env.model.geom_size[g, 0]


def test_load_model_without_the_setting_is_the_files_as_they_are():
    files = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene_29dof.xml'))
    model, kept = load_model(hand_collision=False)
    assert np.array_equal(mjb(model), mjb(files)) and kept == list(range(29))
    hands, kept = load_model(hand_collision=True)
    assert kept == list(range(29)) and hands.ngeom == files.ngeom + 2 and hands.njnt == files.njnt
    both, _ = load_model(foot_collision=True, hand_collision=True)
    assert both.ngeom == files.ngeom + 14 + 2
    with pytest.raises(ValueError):  # the legs-only G1 welds its arms to the torso
        load_model(legs_only=True, hand_collision=True)
    with pytest.raises(ValueError):
        G1VelocityEnv(legs_only=True, hand_collision=True)


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_the_step_is_bitwise_the_original(profile):
    plain, off = G1VelocityEnv(**profile), G1VelocityEnv(hand_collision=False, **profile)
    try:
        assert not off.hand_collision and off.hand_capsules.shape == (0,)
        assert np.array_equal(mjb(off.model), mjb(plain.model))
        np.testing.assert_array_equal(plain.reset(seed=5)[0], off.reset(seed=5)[0])
        limit = plain.action_limit
        for action in np.random.default_rng(2).uniform(-limit, limit, (10, plain.model.nu)):
            expected, result = plain.step(action), off.step(action)
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1] and result[4]['reward_components'] == expected[4]['reward_components']
            np.testing.assert_array_equal(off.data.qpos, plain.data.qpos)
    finally:
        plain.close(), off.close()


def test_the_capsules_are_unitrees_and_collide_with_the_floor_and_the_robot():
    profile = dict(REAL, foot_collision=True)
    plain, env = G1VelocityEnv(**profile), G1VelocityEnv(hand_collision=True, **profile)
    try:
        model, old = env.model, plain.model
        assert model.ngeom == old.ngeom + 2
        assert (model.nbody, model.njnt, model.nq, model.nv, model.nu) == (old.nbody, old.njnt, old.nq, old.nv, old.nu)
        for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'dof_armature', 'jnt_range'):
            np.testing.assert_array_equal(getattr(model, name), getattr(old, name), err_msg=name)
        assert env.hand_capsules.shape == (2,)
        for g, (body, fromto) in zip(env.hand_capsules, HAND_CAPSULES.items()):
            assert model.geom(int(g)).name == f'{body.split("_")[0]}_hand_collision'
            assert model.body(model.geom_bodyid[g]).name == body
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            assert model.geom_size[g, 0] == HAND_CAPSULE_RADIUS == .035
            # Unitree's contact (g1_constants.py FULL_COLLISION for .*_collision): every bit 1, condim 1, priority 0,
            # MuJoCo's default friction; group 3 as their class collision.
            assert (model.geom_contype[g], model.geom_conaffinity[g]) == (1, 1)
            assert (model.geom_condim[g], model.geom_group[g], model.geom_priority[g]) == (1, 3, 0)
            np.testing.assert_array_equal(model.geom_friction[g], [1., .005, .0001])
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            half = rotation.reshape(3, 3)[:, 2] * model.geom_size[g, 1]
            ends = np.sort([model.geom_pos[g] - half, model.geom_pos[g] + half], axis=0)
            np.testing.assert_allclose(ends, np.sort(np.reshape(fromto, (2, 3)), axis=0), rtol=0, atol=1e-12)
        # Unitree's file as it is: the left capsule ends 2 cm to the right of the hand's axis, the right one mirrored.
        assert HAND_CAPSULES == {'left_wrist_yaw_link': (.07, 0., 0., .15, -.02, 0.),
                                 'right_wrist_yaw_link': (.07, 0., 0., .15, .02, 0.)}
        # Robot and arm geoms: a hand pressing on the body counts in self_contact (and is masked in an arm segment).
        assert env.robot_geoms[env.hand_capsules].all() and env.robot_geoms.sum() == plain.robot_geoms.sum() + 2
        assert env.arm_geoms[env.hand_capsules].all() and env.arm_geoms.sum() == plain.arm_geoms.sum() + 2
        # The feet keep their sole spheres and capsules.
        assert env.soles.tolist() == plain.soles.tolist() and env.foot_capsules.tolist() == plain.foot_capsules.tolist()
        assert env.foot_geoms == plain.foot_geoms
    finally:
        plain.close(), env.close()


def test_a_hand_lowered_to_the_floor_touches_it_with_its_capsule_only_with_the_setting():
    """Standing in the home pose, the capsule reaches 13 cm below the wrist's own collision mesh (the palm, which has
    no collision in our files). Lowered 5 mm into the floor, the capsule touches it (condim 3: the floor's), while
    without the setting nothing of the hand does (the same pose: the same reset, the same drop)."""
    for flag in (True, False):
        env = G1VelocityEnv(hand_collision=flag, **REAL)
        try:
            env.reset(seed=4, options={'command': [0., 0., 0.]})
            mujoco.mj_forward(env.model, env.data)
            if flag:
                left = env.hand_capsules[0]
                wrist = [g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] == env.model.geom_bodyid[left]
                         and env.model.geom_contype[g] and g != left]
                vertices = []
                for g in wrist:
                    mesh = env.model.geom_dataid[g]
                    start, count = env.model.mesh_vertadr[mesh], env.model.mesh_vertnum[mesh]
                    vertices.append(env.data.geom_xpos[g] + env.model.mesh_vert[start:start + count]
                                    @ env.data.geom_xmat[g].reshape(3, 3).T)
                lowest = float(np.vstack(vertices)[:, 2].min())
                bottom = capsule_bottom(env, left)
                assert .12 < lowest - bottom < .14
                drop = bottom + .005
            env.data.qpos[2] -= drop
            mujoco.mj_forward(env.model, env.data)
            ground = env.model.geom('floor').id
            hands = {env.model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')}
            touching = {int(env.model.geom_bodyid[g]) for k in range(env.data.ncon)
                        for g in env.data.contact.geom[k] if ground in env.data.contact.geom[k]}
            if flag:  # both hands: the right one hangs 2 mm lower
                found = hand_contacts(env, floor=True)
                assert found and {dim for *_, dim in found} == {3}
                assert {g for a, b, *_ in found for g in (a, b)} - {ground} == set(env.hand_capsules.tolist())
            else:
                assert not touching & hands
        finally:
            env.close()


def test_standing_in_the_home_pose_the_hands_only_graze_the_hips():
    """In the home pose the capsules lie within a millimetre of the hip_roll meshes: while the robot stands, hand-hip
    contacts stay far below the self_contact threshold, and the term charges nothing."""
    from humanoid_sim.tasks import g1_tensor_math as task
    env = G1VelocityEnv(hand_collision=True, reward_weights={'self_contact': 4.}, foot_collision=True, **REAL)
    try:
        env.reset(seed=4, options={'command': [0., 0., 0.]})
        for _ in range(40):  # standing, before the unpowered home pose sags
            info = env.step(np.zeros(29))[-1]
            assert info['reward_components']['self_contact'] == 0
            body = hand_contacts(env, floor=False)
            assert all(force < task.SELF_CONTACT_FORCE for *_, force, _ in body)
            assert all(env.model.body(env.model.geom_bodyid[g]).name.endswith('hip_roll_link')
                       for a, b, *_ in body for g in (a, b) if g not in env.hand_capsules)
        assert env.data.qpos[2] > .7 and not hand_contacts(env, floor=True)
    finally:
        env.close()


def test_hand_collision_config_and_its_own_model_revision():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import FOOT_COLLISION_SUFFIX, G1_PARAMETERS, HAND_COLLISION_SUFFIX
    from humanoid_sim.warm_start import check_carry_over, model_suffixes
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['hand_collision'] is False  # old branches keep theirs
    assert validate_config({**base, 'task_config': {'hand_collision': True}})['task_config']['hand_collision'] is True
    for bad in (1, 'true', None, 0.):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'hand_collision': bad}})
    keys = [p['key'] for p in G1_PARAMETERS]
    field = G1_PARAMETERS[keys.index('hand_collision')]
    assert keys.index('hand_collision') == keys.index('foot_collision') + 1
    assert field['default'] is False and field['type'] == 'bool' and 'Кисти' in field['label']
    assert HAND_COLLISION_SUFFIX == '-handcol'
    real = dict(REAL, lock_wrists=True, knee_floor=.061)
    for config in ({}, real):
        saved = task_identity('g1_29dof', 'g1_velocity', config)  # stages saved before the setting existed
        assert saved == task_identity('g1_29dof', 'g1_velocity', {**config, 'hand_collision': False})
        hands = task_identity('g1_29dof', 'g1_velocity', {**config, 'hand_collision': True})
        feet = task_identity('g1_29dof', 'g1_velocity', {**config, 'foot_collision': True})
        both = task_identity('g1_29dof', 'g1_velocity', {**config, 'foot_collision': True, 'hand_collision': True})
        assert hands == {**saved, 'model_revision': saved['model_revision'] + '-handcol'}  # only the physics
        assert both == {**saved, 'model_revision': saved['model_revision'] + '-footcol-handcol'}
        assert model_suffixes(both['model_revision']) == (saved['model_revision'], [FOOT_COLLISION_SUFFIX,
                                                                                    HAND_COLLISION_SUFFIX])
        with pytest.raises(ValueError):
            validate_resume(saved, hands)  # an ordinary branch is refused
        # carry_over crosses one suffix at a time, either way, whichever came first
        for old, new in ((saved, hands), (feet, both), (hands, both)):
            check_carry_over(old, new)
            check_carry_over(new, old)
        for old, new in ((saved, both), (hands, feet), (hands, hands)):  # two at once, a swap, the same contract
            with pytest.raises(ValueError):
                check_carry_over(old, new)
        for other in ({**hands, 'model_revision': saved['model_revision'] + '-handcol-handcol'},
                      {**hands, 'model_revision': saved['model_revision'] + '-other-handcol'}):
            with pytest.raises(ValueError):
                check_carry_over(saved, other)


def test_carry_over_moves_a_foot_collision_stage_to_hand_collision(tmp_path):
    """contract_branch.py's path for the iteration-33 parents (foot_collision on): carry_over keeps the weights and
    the normalization and gives the stage the model revision with both suffixes."""
    import torch
    from humanoid_sim.core import load_stage, read_json
    from humanoid_sim.warm_start import carry_over
    from test_g1_knee_floor import knee_floor_run, stub_stage
    stage = stub_stage(tmp_path, 'feet', foot_collision=True)
    run, _ = knee_floor_run(tmp_path, 'hands', foot_collision=True, hand_collision=True)
    saved = read_json(stage / 'metadata.json')
    moved = carry_over(stage, run)
    metadata = read_json(moved / 'metadata.json')
    assert saved['model_revision'].endswith('-footcol')
    assert metadata['model_revision'] == saved['model_revision'] + '-handcol'
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        assert new_norm.venv.envs[0].unwrapped.hand_capsules.shape == (2,)
        assert old_norm.venv.envs[0].unwrapped.hand_capsules.shape == (0,)
    finally:
        old_norm.close(), new_norm.close()


def test_export_notes_hand_collision_and_leaves_the_graph_and_deploy_yaml_alone(tmp_path):
    pytest.importorskip('onnxruntime')
    yaml = pytest.importorskip('yaml')
    import onnx
    from humanoid_sim.export_g1 import export, onnx_properties
    from test_export_g1 import LOCKED, make_stage
    results = {}
    for name, task_config in (('plain', LOCKED), ('hands', {**LOCKED, 'hand_collision': True})):
        (tmp_path / name).mkdir()
        results[name] = export(make_stage(tmp_path / name, task_config), tmp_path / f'out-{name}')
    graphs = {name: onnx.load(str(result['onnx'])).graph for name, result in results.items()}
    assert [n.op_type for n in graphs['hands'].node] == [n.op_type for n in graphs['plain'].node]
    deploys = {name: result['deploy'].read_text(encoding='utf-8') for name, result in results.items()}
    assert yaml.safe_load(deploys['hands']) == yaml.safe_load(deploys['plain'])
    assert '\n# hand_collision: ' in deploys['hands'] and '# hand_collision' not in deploys['plain']
    properties = {name: onnx_properties(result['onnx']) for name, result in results.items()}
    assert properties['hands']['hand_collision'] == 'true'
    assert properties['hands']['model_revision'].endswith('-handcol')
    assert 'hand_collision' not in properties['plain'] and 'model_revision' not in properties['plain']


def test_the_browser_draws_the_same_geoms_with_and_without_the_capsules():
    from humanoid_sim.webapp import geom_transforms, model_geometry
    plain, _ = load_model()
    hands, _ = load_model(foot_collision=True, hand_collision=True)
    assert [g['name'] for g in model_geometry(plain)] == [g['name'] for g in model_geometry(hands)]
    datas = [mujoco.MjData(model) for model in (plain, hands)]
    for data in datas:
        data.qpos[:] = plain.qpos0
        data.qpos[7:] += np.linspace(-.2, .2, plain.nq - 7)
    assert np.allclose(geom_transforms(plain, datas[0]), geom_transforms(hands, datas[1]))
