"""task_config foot_collision (iteration 23): Unitree's foot capsules, colliding only foot with foot."""
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import (ASSETS, FOOT_CAPSULE_RADIUS, FOOT_CAPSULES, FOOT_COLLISION_BIT,
                                            G1VelocityEnv, load_model)
from humanoid_sim.tasks.identity import task_identity
from test_g1_lock_wrists import lab_probe

# The real-robot profile of the current recipe (as test_g1_self_contact), without its reward weights.
REAL = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, action_limit=10.,
            heading_hold=True, stand_heading_hold=False, gait=dict(period=.6, stance_fraction=.55))
PROFILES = ({}, REAL)
CAPSULES = 7  # per foot, unitree_rl_mjlab g1.xml left/right_foot1..7_collision
# Candidate 6 (exp-8ce0cbd70d0a4f1d @ 6039.1 M): the real stage a foot_collision branch may start from.
CANDIDATE = Path(__file__).resolve().parents[1] / 'runs/humanoid_lab/exp-8ce0cbd70d0a4f1d/stage_006039109632'


def mjb(model):
    buffer = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer


def old_soles(env):
    """The sole spheres as the environment picked them before foot_collision: the foot geoms with a contype."""
    return [[g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] == foot and env.model.geom_contype[g]]
            for foot in env.feet]


def capsule_contacts(env):
    """(geom1, geom2, force norm N, contact dimension) of this step's contacts that touch a capsule."""
    capsules, force, found = set(env.foot_capsules.ravel().tolist()), np.zeros(6), []
    for k in range(env.data.ncon):
        pair = tuple(int(g) for g in env.data.contact.geom[k])
        if capsules & set(pair):
            mujoco.mj_contactForce(env.model, env.data, k, force)
            found.append((*pair, float(np.linalg.norm(force[:3])), int(env.data.contact.dim[k])))
    return found


def crossed(env, roll, pitch=.08):
    """Hip roll inwards on both legs, the left leg back and the right one forward by `pitch` (so the heel and toe
    spheres of the two soles miss each other while the capsules overlap)."""
    names = [env.model.joint(j).name for j in env.joints]
    q = np.zeros(env.model.nu)
    q[names.index('left_hip_roll_joint')], q[names.index('right_hip_roll_joint')] = -roll, roll
    q[names.index('left_hip_pitch_joint')], q[names.index('right_hip_pitch_joint')] = -pitch, pitch
    return q


def cross_in_the_air(env, steps=6):
    """In the air (free fall: no floor, no weight on the legs), the feet start 2.4 cm apart and the PD drives them
    together and on (hip roll target 0.25 rad, contact at ~0.135): per step the self_contact term and the capsule
    contacts."""
    env.reset(seed=4, options={'command': [0., 0., 0.]})
    env.data.qpos[env.qadr] = env.home + crossed(env, .12)
    env.data.qpos[2] = .9
    mujoco.mj_forward(env.model, env.data)
    action = crossed(env, .25) / env.scale
    rows = []
    for _ in range(steps):
        info = env.step(action)[-1]
        rows.append((info['reward_components']['self_contact'], capsule_contacts(env)))
    return rows


def test_load_model_without_the_setting_is_the_files_as_they_are():
    files = mujoco.MjModel.from_xml_path(str(ASSETS / 'scene_29dof.xml'))
    for model, kept in (load_model(), load_model(foot_collision=False)):
        assert np.array_equal(mjb(model), mjb(files)) and kept == list(range(29))
    (legs, kept), (legs_feet, kept_feet) = load_model(legs_only=True), load_model(True, True)
    assert kept_feet == kept and legs_feet.ngeom == legs.ngeom + 2 * CAPSULES and legs_feet.njnt == legs.njnt
    feet, kept = load_model(foot_collision=True)
    assert kept == list(range(29)) and feet.ngeom == files.ngeom + 2 * CAPSULES


@pytest.mark.parametrize('profile', PROFILES)
def test_without_the_setting_the_soles_and_the_step_are_bitwise_the_original(profile):
    plain, off = G1VelocityEnv(**profile), G1VelocityEnv(foot_collision=False, **profile)
    try:
        assert not off.foot_collision and off.foot_capsules.shape == (2, 0)
        assert off.soles.tolist() == old_soles(off) and off.soles.shape == (2, 4)
        if off.real_robot:  # friction and priority on the sole spheres only, as before
            assert off.foot_geoms == [g for foot in old_soles(off) for g in foot]
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


def test_the_capsules_are_unitrees_and_collide_only_foot_with_foot():
    plain, env = G1VelocityEnv(**REAL), G1VelocityEnv(foot_collision=True, **REAL)
    try:
        model, old = env.model, plain.model
        assert model.ngeom == old.ngeom + 2 * CAPSULES
        assert (model.nbody, model.njnt, model.nq, model.nv, model.nu) == (old.nbody, old.njnt, old.nq, old.nv, old.nu)
        for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'dof_armature', 'jnt_range'):
            np.testing.assert_array_equal(getattr(model, name), getattr(old, name), err_msg=name)
        assert env.foot_capsules.shape == (2, CAPSULES)
        for foot, (body, capsules) in zip(env.foot_capsules, FOOT_CAPSULES.items()):
            assert body in task.FEET
            for number, (g, fromto) in enumerate(zip(foot, capsules), 1):
                assert model.geom(int(g)).name == f'{body.split("_")[0]}_foot{number}_collision'
                assert model.body(model.geom_bodyid[g]).name == body
                assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                assert model.geom_size[g, 0] == FOOT_CAPSULE_RADIUS == .01
                assert (model.geom_contype[g], model.geom_conaffinity[g]) == (FOOT_COLLISION_BIT, FOOT_COLLISION_BIT)
                # Unitree's contact (g1_constants.py FULL_COLLISION): condim 3, sliding friction 0.6, priority 1.
                assert (model.geom_condim[g], model.geom_group[g], model.geom_priority[g]) == (3, 3, 1)
                np.testing.assert_array_equal(model.geom_friction[g], [.6, .005, .0001])
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
                half = rotation.reshape(3, 3)[:, 2] * model.geom_size[g, 1]
                ends = np.sort([model.geom_pos[g] - half, model.geom_pos[g] + half], axis=0)
                np.testing.assert_allclose(ends, np.sort(np.reshape(fromto, (2, 3)), axis=0), rtol=0, atol=1e-12)
        # Unitree's file as it is: the left foot1 ends 1 mm wider than the right foot's mirror.
        assert FOOT_CAPSULES['left_ankle_roll_link'][0] == (.1, -.026, -.025, .05, -.027, -.025)
        assert FOOT_CAPSULES['right_ankle_roll_link'][0] == (.1, -.026, -.025, .05, -.026, -.025)
        capsules = set(env.foot_capsules.ravel().tolist())
        bit = lambda array: {g for g in range(model.ngeom) if array[g] & FOOT_COLLISION_BIT}
        assert bit(model.geom_contype) == bit(model.geom_conaffinity) == capsules  # no other geom has the bit
        passes = lambda a, b: bool(model.geom_contype[a] & model.geom_conaffinity[b]
                                   or model.geom_contype[b] & model.geom_conaffinity[a])
        for c in capsules:
            assert not any(passes(c, g) for g in range(model.ngeom) if g not in capsules)  # floor, spheres, meshes
        assert all(passes(a, b) for a in env.foot_capsules[0] for b in env.foot_capsules[1])
        # The floor contact stays the four sole spheres per foot, with the real-robot friction and priority.
        assert env.soles.shape == (2, 4) and not capsules & set(env.soles.ravel().tolist())
        for name in ('geom_size', 'geom_pos', 'geom_friction', 'geom_priority', 'geom_type'):
            np.testing.assert_array_equal(getattr(model, name)[env.soles], getattr(old, name)[plain.soles])
        assert env.foot_geoms == env.soles.ravel().tolist()
        # self_contact: the capsules are robot geoms, so a foot-foot contact counts.
        assert env.robot_geoms[env.foot_capsules].all() and env.robot_geoms.sum() == plain.robot_geoms.sum() + 14
    finally:
        plain.close(), env.close()
    legs = G1VelocityEnv(legs_only=True, foot_collision=True)
    try:
        assert legs.foot_capsules.shape == (2, CAPSULES) and legs.robot_geoms[legs.foot_capsules].all()
    finally:
        legs.close()


def test_crossed_feet_touch_and_are_charged_by_self_contact_only_with_the_capsules(monkeypatch):
    """The feet driven into each other in the air: without the capsules they pass through (Unitree's capsules overlap
    up to the axes, nothing touches, the term stays 0); with them they touch, foot capsule against foot capsule only,
    above 10 N, and self_contact charges the whole step."""
    probe = lab_probe('foot_overlap', monkeypatch)
    weights = {'self_contact': 4.}
    plain, env = (G1VelocityEnv(reward_weights=weights, foot_collision=flag, **REAL) for flag in (False, True))
    try:
        rows = cross_in_the_air(plain)
        assert [term for term, _ in rows] == [0.] * 6 and plain.data.ncon == 0
        assert probe.feet_distance(plain) < -.01  # Unitree's capsules would be 1 cm into each other
        rows = cross_in_the_air(env)
        terms = [term for term, _ in rows]
        assert terms[0] == 0 and terms[-3:] == [-4.] * 3  # every substep of the last steps above 10 N
        touching = [contact for _, contacts in rows for contact in contacts]
        assert touching and max(force for _, _, force, _ in touching) > task.SELF_CONTACT_FORCE
        left, right = (set(foot.tolist()) for foot in env.foot_capsules)
        assert all((a in left and b in right) or (a in right and b in left) for a, b, _, _ in touching)
        assert {dim for *_, dim in touching} == {3}  # with friction, as Unitree's foot capsules
        assert probe.feet_distance(env) > -.005  # held apart: millimetres of penetration
    finally:
        plain.close(), env.close()


def test_in_the_home_pose_nothing_touches_the_capsules_and_the_step_is_the_original():
    """Standing in the home pose the capsules touch nothing, so the floor contact and the whole step are bit for bit
    those without them."""
    weights = {'self_contact': 1.}
    plain, env = (G1VelocityEnv(reward_weights=weights, foot_collision=flag, **REAL) for flag in (False, True))
    # Each contact by its geoms' bodies and positions there (the right foot's geom numbers differ by the capsules).
    floor = lambda e: [tuple((int(e.model.geom_bodyid[g]), *e.model.geom_pos[g].tolist()) for g in pair)
                       for pair in e.data.contact.geom[:e.data.ncon]]
    try:
        for e in (plain, env):
            e.reset(seed=4, options={'command': [0., 0., 0.]})
        for _ in range(60):
            expected, result = plain.step(np.zeros(29)), env.step(np.zeros(29))
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1] == expected[1] and result[4]['reward_components'] == expected[4]['reward_components']
            np.testing.assert_array_equal(env.data.qpos, plain.data.qpos)
            assert not capsule_contacts(env) and result[4]['reward_components']['self_contact'] == 0
        assert floor(env) == floor(plain) and env.data.ncon == plain.data.ncon > 0
    finally:
        plain.close(), env.close()


def test_the_probe_measures_unitrees_capsules_exactly(monkeypatch):
    """foot_overlap's capsules are the model's (FOOT_CAPSULES from the ankle_roll frames) and its distance is MuJoCo's
    capsule-capsule distance, whatever the pose; on a model without the capsules it measures the same."""
    probe = lab_probe('foot_overlap', monkeypatch)
    plain, env = G1VelocityEnv(), G1VelocityEnv(foot_collision=True)
    rng = np.random.default_rng(0)
    try:
        overlapping = 0
        for k in range(60):
            q = env.home + rng.uniform(-.15, .15, 29) + (crossed(env, .13) if k % 2 else 0.)  # half of them crossed
            for e in (plain, env):
                e.reset(seed=k)
                e.data.qpos[e.qadr] = q
                mujoco.mj_forward(e.model, e.data)
            reference = min(mujoco.mj_geomDistance(env.model, env.data, int(a), int(b), 1., None)
                            for a in env.foot_capsules[0] for b in env.foot_capsules[1])
            assert probe.feet_distance(env) == pytest.approx(reference, abs=1e-12)
            assert probe.feet_distance(plain) == pytest.approx(reference, abs=1e-12)
            overlapping += reference < 0
        assert 5 < overlapping < 55
        start, end = np.array([0., 0., 0.]), np.array([1., 0., 0.])
        for other, want in ((((.5, .3, 0.), (.5, 1.3, 0.)), .3),  # perpendicular, its end nearest
                            (((2., 1., 0.), (3., 1., 0.)), 2 ** .5),  # beyond the end: end to end
                            (((.5, -1., 1.), (.5, 1., 1.)), 1.),  # crossing above
                            (((.8, .5, 0.), (.2, .5, 0.)), .5)):  # parallel
            got = probe.segment_distance(start, end, np.array(other[0]), np.array(other[1]))
            assert float(got) == pytest.approx(want, abs=1e-12)
    finally:
        plain.close(), env.close()


def test_foot_collision_config_and_its_own_model_revision():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import FOOT_COLLISION_SUFFIX, G1_PARAMETERS
    from humanoid_sim.warm_start import check_carry_over
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['foot_collision'] is False  # old branches keep theirs
    assert validate_config({**base, 'task_config': {'foot_collision': True}})['task_config']['foot_collision'] is True
    for bad in (1, 'true', None, 0.):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'foot_collision': bad}})
    field = next(p for p in G1_PARAMETERS if p['key'] == 'foot_collision')
    assert field['default'] is False and field['type'] == 'bool' and 'Стопы' in field['label']
    assert FOOT_COLLISION_SUFFIX == '-footcol'
    for robot, config in (('g1_29dof', {}), ('g1_12dof', {}),
                          ('g1_29dof', dict(REAL, lock_wrists=True, knee_floor=.061))):
        saved = task_identity(robot, 'g1_velocity', config)  # stages saved before the setting existed
        assert saved == task_identity(robot, 'g1_velocity', {**config, 'foot_collision': False})
        feet = task_identity(robot, 'g1_velocity', {**config, 'foot_collision': True})
        assert feet == {**saved, 'model_revision': saved['model_revision'] + '-footcol'}  # only the physics
        with pytest.raises(ValueError):
            validate_resume(saved, feet)  # an ordinary branch is refused
        check_carry_over(saved, feet)  # carry_over may move the weights, both ways
        check_carry_over(feet, saved)
        with pytest.raises(ValueError):  # the same contract: an ordinary branch
            check_carry_over(feet, feet)
        for other in ({**feet, 'model_revision': saved['model_revision'] + '-other'},  # an unknown physics change
                      {**feet, 'model_revision': saved['model_revision'] + '-footcol-footcol'},
                      {**feet, 'control_dt': .01}):
            with pytest.raises(ValueError):
                check_carry_over(saved, other)
    # With another contract change at once (lock_wrists: the action schema) the carry-over still goes.
    real = dict(REAL, critic_extras=True)
    check_carry_over(task_identity('g1_29dof', 'g1_velocity', real),
                     task_identity('g1_29dof', 'g1_velocity', {**real, 'lock_wrists': True, 'foot_collision': True}))


def test_carry_over_moves_a_stage_to_foot_collision_as_contract_branch_does(tmp_path):
    """contract_branch.py's checks (check_carry_over, make_task) pass for foot_collision, then carry_over keeps the
    weights and the normalization and gives the stage the new model revision; carried again, or to another action
    box, it is refused."""
    from humanoid_sim.core import load_stage, read_json
    from humanoid_sim.tasks.catalog import make_task
    from humanoid_sim.warm_start import carry_over, check_carry_over
    from test_g1_knee_floor import knee_floor_run, stub_stage
    stage = stub_stage(tmp_path)
    run, config = knee_floor_run(tmp_path, 'feet', foot_collision=True)
    saved = read_json(stage / 'metadata.json')
    check_carry_over(saved, task_identity('g1_29dof', 'g1_velocity', config['task_config']))
    make_task('g1_velocity', 'g1_29dof', config['task_config']).close()
    moved = carry_over(stage, run)
    metadata, warm = read_json(moved / 'metadata.json'), read_json(moved / 'warm_start.json')
    assert metadata['model_revision'] == saved['model_revision'] + '-footcol'
    assert {k: v for k, v in metadata.items() if k not in ('model_revision', 'task_config', 'created_at')} == \
        {k: v for k, v in saved.items() if k not in ('model_revision', 'task_config', 'created_at')}
    assert (warm['from_model_revision'], warm['to_model_revision']) == (saved['model_revision'],
                                                                        metadata['model_revision'])
    assert warm['from_action_schema'] == warm['to_action_schema'] == saved['action_schema']
    (before, old_norm), (after, new_norm) = load_stage(stage), load_stage(moved)
    try:
        for (name, a), (_, b) in zip(before.policy.state_dict().items(), after.policy.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        np.testing.assert_array_equal(old_norm.obs_rms.mean, new_norm.obs_rms.mean)
        np.testing.assert_array_equal(old_norm.obs_rms.var, new_norm.obs_rms.var)
        assert new_norm.venv.envs[0].unwrapped.foot_capsules.shape == (2, CAPSULES)
        assert old_norm.venv.envs[0].unwrapped.foot_capsules.shape == (2, 0)
    finally:
        old_norm.close(), new_norm.close()
    with pytest.raises(ValueError):  # carried over already: the same contract, nothing to carry
        carry_over(moved, run)
    box, _ = knee_floor_run(tmp_path, 'box', foot_collision=True, action_limit=4.)
    with pytest.raises(ValueError):
        carry_over(stage, box)
    assert not any(box.glob('stage_*'))


def test_candidate_6_carries_over_to_foot_collision(tmp_path):
    """The real stage a foot_collision branch may start from: carried over with its weights unchanged."""
    from humanoid_sim.core import read_json, write_json
    from humanoid_sim.warm_start import carry_over
    if not (CANDIDATE / 'metadata.json').is_file():
        pytest.skip('candidate 6 is not on this machine')
    saved = read_json(CANDIDATE / 'metadata.json')
    parent = read_json(CANDIDATE.parent / 'run_config.json')
    write_json(tmp_path / 'run_config.json', {**parent, 'source_id': None, 'source_stage': None,
                                              'task_config': {**saved['task_config'], 'foot_collision': True}})
    moved = carry_over(CANDIDATE, tmp_path)
    assert read_json(moved / 'metadata.json')['model_revision'] == saved['model_revision'] + '-footcol'


def test_goal_check_and_the_bridge_copy_keep_the_sole_spheres_and_switch_the_capsules_off_there(monkeypatch):
    """goal_check.sole_geoms (and the probes on it) still finds the four sole spheres per foot; um_probe's Unitree
    physics, a copy of unitree_mujoco whose feet pass through each other, switches the capsules off, and leaves a
    model without them as it was."""
    from humanoid_sim.goal_check import sole_geoms
    um = lab_probe('um_probe', monkeypatch)
    for flag in (False, True):
        env, files = G1VelocityEnv(foot_collision=flag, **REAL), load_model(foot_collision=flag)[0]
        try:
            assert sole_geoms(env) == env.soles.tolist() and [len(foot) for foot in sole_geoms(env)] == [4, 4]
            assert all(env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE for foot in sole_geoms(env) for g in foot)
            um.apply_physics(env, 'um_joint')  # joint parameters only
            np.testing.assert_array_equal(env.model.geom_contype, files.geom_contype)
            um.apply_physics(env, 'um')  # the geoms' friction and priority of the files, and no foot collision
            np.testing.assert_array_equal(env.model.geom_friction, files.geom_friction)
            capsules = env.foot_capsules.ravel()
            assert not env.model.geom_contype[capsules].any() and not env.model.geom_conaffinity[capsules].any()
            others = np.setdiff1d(np.arange(env.model.ngeom), capsules)
            np.testing.assert_array_equal(env.model.geom_contype[others], files.geom_contype[others])
            np.testing.assert_array_equal(env.model.geom_conaffinity[others], files.geom_conaffinity[others])
        finally:
            env.close()
    # um_caps / ours_caps: the feet as Unitree's training model has them, whatever the stage's setting.
    assert um.physics_config('um_caps') == um.physics_config('ours_caps') == {'foot_collision': True}
    assert um.physics_config('um') == um.physics_config('ours') == {}
    with pytest.raises(ValueError):  # without the capsules in the environment there is nothing to keep
        um.apply_physics(G1VelocityEnv(**REAL), 'um_caps')
    for physics in ('um_caps', 'ours_caps'):
        env = G1VelocityEnv(**{**REAL, **um.physics_config(physics)})
        try:
            files = load_model(foot_collision=True)[0]
            um.apply_physics(env, physics)
            capsules = env.foot_capsules.ravel()
            assert (env.model.geom_contype[capsules] == FOOT_COLLISION_BIT).all()
            assert (env.model.geom_conaffinity[capsules] == FOOT_COLLISION_BIT).all()
            assert (env.model.geom_condim[capsules] == 3).all() and (env.model.geom_priority[capsules] == 1).all()
            assert (env.model.geom_friction[capsules, 0] == .6).all()
            integrator = mujoco.mjtIntegrator.mjINT_EULER if physics == 'um_caps' else \
                mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            assert env.model.opt.integrator == integrator
            if physics == 'um_caps':  # the rest is Unitree's: the files' friction elsewhere
                np.testing.assert_array_equal(env.model.geom_friction, files.geom_friction)
        finally:
            env.close()


def test_foot_shock_runs_every_case_with_the_capsules_off_and_on(monkeypatch, tmp_path):
    """foot_shock.py (the critic's shock.py): A training-like episodes (noise only for an SB3 stage), B sideways pushes
    in our physics and Unitree's (um_caps when on), C side steps ±0.5 / ±1; each case off and on, the stage's own
    foot_collision overridden."""
    shock = lab_probe('foot_shock', monkeypatch)
    stage = tmp_path / 'exp-0000000000000000' / 'stage_000000000100'
    stage.mkdir(parents=True)
    (stage / 'policy.zip').touch()
    stages = {'stage': str(stage), 'v0': 'unitree_v0'}
    a = shock.jobs('A', stages, seeds=3)
    assert len(a) == (2 * 3 + 3) * 2 and {job['mode'] for job in a if job['name'] == 'v0'} == {'det'}
    b, c = shock.jobs('B', {'stage': str(stage)}), shock.jobs('C', {'stage': str(stage)})
    assert len(b) == 2 * 2 * 2 * 3 * 4 and {(job['phys'], job['dir'], job['mag']) for job in b} == {
        (p, d, m) for p in ('ours', 'um') for d in 'lr' for m in (1., 1.25, 1.5)}
    assert len(c) == 4 * 2 * 2 and {tuple(job['cmd']) for job in c} == {
        (0., .5, 0.), (0., -.5, 0.), (0., 1., 0.), (0., -1., 0.)}
    for jobs in (a, b, c):  # every case once off and once on
        case = lambda job: tuple(sorted((k, str(v)) for k, v in job.items() if k != 'caps'))
        off, on = ([case(job) for job in jobs if job['caps'] == caps] for caps in (False, True))
        assert sorted(off) == sorted(on) and len(set(off)) == len(off) == len(jobs) // 2
    with pytest.raises(ValueError):
        shock.jobs('D', stages)
    monkeypatch.setattr(shock, '_original_stage_data', lambda path: ('policy', 'norm', {'foot_collision': True}))
    for caps in (False, True):
        monkeypatch.setitem(shock.CAPS, 'on', caps)
        assert shock.stage_data('x') == ('policy', 'norm', {'foot_collision': caps})


def test_contact_step_marks_steps_with_any_substep_above_10_n():
    force = torch.tensor([[0.] * 10, [10.] * 10, [0.] * 9 + [10.01], [50.] * 10])
    assert task.contact_step(force).tolist() == [False, False, True, True]


def test_export_notes_foot_collision_and_leaves_the_graph_and_deploy_yaml_alone(tmp_path):
    """The robot does not change: the same weights with and without foot_collision give the same graph, outputs and
    deploy.yaml; the export says how it was trained (ONNX properties, a deploy.yaml comment)."""
    pytest.importorskip('onnxruntime')
    yaml = pytest.importorskip('yaml')
    import onnx
    import onnxruntime
    from humanoid_sim.export_g1 import check, export, onnx_properties
    from test_export_g1 import LOCKED, make_stage
    results = {}
    for name, task_config in (('plain', LOCKED), ('feet', {**LOCKED, 'foot_collision': True})):
        (tmp_path / name).mkdir()
        results[name] = export(make_stage(tmp_path / name, task_config), tmp_path / f'out-{name}')
    graphs = {name: onnx.load(str(result['onnx'])).graph for name, result in results.items()}
    assert [n.op_type for n in graphs['feet'].node] == [n.op_type for n in graphs['plain'].node]
    obs = np.random.default_rng(5).normal(0, 3, (10, 490)).astype(np.float32)
    outputs = {name: np.concatenate([onnxruntime.InferenceSession(str(result['onnx']), providers=[
        'CPUExecutionProvider']).run(['actions'], {'obs': row[None]})[0] for row in obs])
        for name, result in results.items()}
    np.testing.assert_array_equal(outputs['feet'], outputs['plain'])
    deploys = {name: result['deploy'].read_text(encoding='utf-8') for name, result in results.items()}
    assert yaml.safe_load(deploys['feet']) == yaml.safe_load(deploys['plain'])
    assert '\n# foot_collision: ' in deploys['feet'] and '# foot_collision' not in deploys['plain']
    properties = {name: onnx_properties(result['onnx']) for name, result in results.items()}
    assert properties['feet']['foot_collision'] == 'true' and properties['feet']['model_revision'].endswith('-footcol')
    assert 'foot_collision' not in properties['plain'] and 'model_revision' not in properties['plain']
    assert check(results['feet']['stage'], results['feet']['onnx'], samples=10, stride=2)['ok']


def test_the_browser_draws_the_same_geoms_with_and_without_capsules():
    # A race can put a foot_collision stage beside an older one; the browser takes the meshes from the first lane
    # and the transforms from each lane, so both models must yield the same shown geoms in the same order (the
    # capsules sit between the feet's other geoms and used to shift every later transform onto the wrong mesh).
    from humanoid_sim.webapp import geom_transforms, model_geometry
    plain, _ = load_model()
    feet, _ = load_model(foot_collision=True)
    assert [g['name'] for g in model_geometry(plain)] == [g['name'] for g in model_geometry(feet)]
    assert [g['id'] for g in model_geometry(feet)] == list(range(plain.ngeom))
    datas = [mujoco.MjData(model) for model in (plain, feet)]
    for data in datas:
        data.qpos[:] = plain.qpos0
        data.qpos[7:] += np.linspace(-.2, .2, plain.nq - 7)
    assert np.allclose(geom_transforms(plain, datas[0]), geom_transforms(feet, datas[1]))
