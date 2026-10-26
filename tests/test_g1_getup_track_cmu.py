"""The CMU source of g1_getup's track clip (iteration 64): scripts/lab/cmu_mocap.py reads ASF / AMC text and runs the
Acclaim forward kinematics (against a hand calculation on a small skeleton), retargets frames onto the G1 (within the
task's soft limits, nothing in the floor, the contacts on it) and settles the last pose to home; scripts/robot/package.py
lets a stage that followed a CMU clip into a package, with the database's acknowledgement in README.md, and still
refuses a LAFAN1 one. The subject-140 data is untracked (.tools/datasets/cmu_mocap_140): its tests skip without it."""
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
sys.path.insert(0, str(ROOT / 'scripts' / 'robot'))
cmu_mocap = importlib.import_module('cmu_mocap')
from humanoid_sim.tasks import g1_getup_track  # noqa: E402
from humanoid_sim.tasks import g1_tensor_math as task  # noqa: E402
from humanoid_sim.tasks.g1_getup_bank import getup_model_env  # noqa: E402

CMU = ROOT / '.tools' / 'datasets' / 'cmu_mocap_140'
LAB_CLIP = 'cmu-140_09-61-571'
ASF = """# a test skeleton
:version 1.10
:name TEST
:units
  mass 1.0
  length 0.45
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name thigh
     direction 0 -1 0
     length 9
     axis 0 0 30  XYZ
    dof rx rz
  end
  begin
     id 2
     name shin
     direction 0 -1 0
     length 4.5
     axis 0 0 0  XYZ
    dof rx
  end
:hierarchy
  begin
    root thigh
    thigh shin
  end
"""
AMC = """#!OML:ASF test
:FULLY-SPECIFIED
:DEGREES
1
root 4.5 9 0 0 90 0
thigh 0 0
shin 0
2
root 4.5 9 0 0 90 0
thigh 90 0
shin -90
"""


def _rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def test_asf_amc_are_read_as_text_and_the_acclaim_kinematics_hold(tmp_path):
    (tmp_path / 't.asf').write_text(ASF, encoding='latin-1')
    (tmp_path / 't.amc').write_text(AMC, encoding='latin-1')
    skeleton = cmu_mocap.read_asf(tmp_path / 't.asf')
    unit = 0.0254 / 0.45
    assert skeleton['length_scale'] == pytest.approx(unit) and skeleton['order'] == ['root', 'thigh', 'shin']
    frames = cmu_mocap.read_amc(tmp_path / 't.amc')
    assert [n for n, _ in frames] == [1, 2] and frames[1][1]['thigh'] == [90., 0.]
    for number, values in frames:
        pose = cmu_mocap.pose(skeleton, values)
        root = _ry(math.pi / 2)
        position = np.array([4.5, 9, 0]) * unit
        c = _rz(math.radians(30))
        thigh = root @ c @ _rx(math.radians(values['thigh'][0])) @ c.T
        knee = position + 9 * unit * thigh @ np.array([0, -1, 0])
        shin = thigh @ _rx(math.radians(values['shin'][0]))
        np.testing.assert_allclose(pose['root'][2], root, atol=1e-12)
        np.testing.assert_allclose(pose['thigh'][0], position, atol=1e-12)
        np.testing.assert_allclose(pose['thigh'][1], knee, atol=1e-12)
        np.testing.assert_allclose(pose['shin'][1], knee + 4.5 * unit * shin @ np.array([0, -1, 0]), atol=1e-12)
    # Z up: (x, y, z) = (z_cmu, x_cmu, y_cmu), a rotation.
    names, starts, ends, rotations = cmu_mocap.points(skeleton, frames[:1])
    np.testing.assert_allclose(starts[0, 1], [0, 4.5 * unit, 9 * unit], atol=1e-12)
    assert np.linalg.det(cmu_mocap.CMU_TO_ZUP) == pytest.approx(1.)
    np.testing.assert_allclose(rotations[0, 0] @ rotations[0, 0].T, np.eye(3), atol=1e-12)
    (tmp_path / 'bad.amc').write_text('root 1 2 3 4 5 6\n', encoding='latin-1')
    with pytest.raises(ValueError):
        cmu_mocap.read_amc(tmp_path / 'bad.amc')


def _settle_start(env):
    """A half crouch on the floor turned 0.8 rad about a tilted axis, the joints off home by up to 0.2 rad."""
    start = cmu_mocap.home_qpos(env)
    start[np.asarray(env.qadr)] += np.random.default_rng(0).uniform(-.2, .2, 29) * ~env.wrist_mask.numpy()
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.array([.1, .05, 1.]) / np.linalg.norm([.1, .05, 1.]), .8)
    start[3:7] = turn
    start[:2] = [.4, -.1]
    model, data, geoms = cmu_mocap._floor_model(env)
    start[2] -= cmu_mocap.lowest_point(model, data, start, geoms)
    return start


def test_settle_ends_standing_at_home_with_the_heading_and_the_feet_flat():
    """settle (by IK, iteration 67): the last frame is the home pose exactly, upright with the start's heading; every
    frame stands on the floor without a jump of the feet; once its weights are in, both feet are flat (all eight sole
    spheres on the floor) and the centre of mass inside the soles; the feet's midpoint stays near the start's."""
    env = getup_model_env()
    start = _settle_start(env)
    qpos = cmu_mocap.settle(env, start[None], 1.)
    assert len(qpos) == 1 + 50
    np.testing.assert_allclose(qpos[-1, np.asarray(env.qadr)], env.home, atol=1e-12)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, qpos[-1, 3:7])
    first = np.zeros(9)
    mujoco.mju_quat2Mat(first, start[3:7])
    assert rotation[8] == pytest.approx(1.)  # upright
    assert math.atan2(rotation[3], rotation[0]) == pytest.approx(math.atan2(first[3], first[0]))
    model, data, geoms = cmu_mocap._floor_model(env)
    middle = []
    ramp = int(cmu_mocap.SETTLE_RAMP * 50) + 1
    for k, row in enumerate(qpos):
        lowest = cmu_mocap.lowest_point(model, data, row, geoms)
        assert lowest == pytest.approx(0., abs=1e-9)  # on the floor
        middle.append(data.xpos[list(env.feet), :2].mean(0))
        soles = [float(data.geom_xpos[g, 2] - model.geom_size[g, 0]) for g in np.asarray(env.soles).ravel()]
        if k > ramp:
            assert max(soles) < .004, (k, soles)  # flat feet
            mujoco.mj_comPos(model, data)
            assert cmu_mocap.support_term(model, data, list(np.asarray(env.soles).ravel()), gradient=False)[0] > 0.
    middle = np.array(middle)
    assert np.linalg.norm(np.diff(middle, axis=0), axis=1).max() < .01  # no jump of the feet
    assert np.linalg.norm(middle - middle[0], axis=1).max() < .03
    assert cmu_mocap.settle(env, start[None], 0.) is not None and len(cmu_mocap.settle(env, start[None], 0.)) == 1


def test_geom_lowest_is_the_shapes_lowest_point():
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    qpos = cmu_mocap.home_qpos(env)
    rng = np.random.default_rng(1)
    for _ in range(5):
        qpos[3:7] = rng.normal(size=4)
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        qpos[2] = .2
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for g in geoms:
            lowest = cmu_mocap.geom_lowest(model, data, g)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                world = data.geom_xpos[g] + vertices @ data.geom_xmat[g].reshape(3, 3).T
                assert lowest[2] == pytest.approx(world[:, 2].min(), abs=1e-12)
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                assert lowest[2] == pytest.approx(data.geom_xpos[g, 2] - model.geom_size[g, 0], abs=1e-12)
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
                ends = data.geom_xpos[g] + np.outer([-1, 1], axis) * model.geom_size[g, 1]
                assert lowest[2] == pytest.approx(ends[:, 2].min() - model.geom_size[g, 0], abs=1e-12)
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:  # its two end rims, sampled
                rotation = data.geom_xmat[g].reshape(3, 3)
                radius, half = model.geom_size[g, :2]
                angle = np.linspace(0., 2 * np.pi, 3601)
                rim = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)], 1)
                points = np.concatenate([rim + [0, 0, half], rim - [0, 0, half]]) @ rotation.T + data.geom_xpos[g]
                assert lowest[2] == pytest.approx(points[:, 2].min(), abs=1e-6)
                assert lowest[2] <= points[:, 2].min() + 1e-12


@pytest.mark.skipif(not (CMU / '140_09.amc').is_file(), reason='CMU subject 140 is downloaded locally')
def test_a_retargeted_cmu_take_keeps_the_soft_limits_and_the_floor():
    """Frames of take 09 (lying, sitting up): every joint within the task's soft limits, the locked wrists at home,
    nothing below the floor, the legs' directions within a few degrees, the pelvis on the floor while the person lies."""
    env = getup_model_env()
    skeleton = cmu_mocap.read_asf(CMU / '140.asf')
    frames = cmu_mocap.read_amc(CMU / '140_09.amc')
    qpos, errors, ratios = cmu_mocap.retarget(env, skeleton, frames[60:200:10], frames[0], rate=12.)
    assert 0.75 < ratios['legs'] < 0.85 and 0.5 < ratios['arms'] < 0.65
    joints = np.asarray(env.joints)
    low, high = task.soft_limits(*env.model.jnt_range[joints].T)
    q = qpos[:, np.asarray(env.qadr)]
    free = ~env.wrist_mask.numpy()
    assert (q[:, free] >= low[free] - 1e-9).all() and (q[:, free] <= high[free] + 1e-9).all()
    assert (q[:, ~free] == env.home[~free]).all()
    model, data, geoms = cmu_mocap._floor_model(env)
    for row in qpos:
        assert cmu_mocap.lowest_point(model, data, row, geoms) >= -1e-9
    legs = [cmu_mocap.ERRORS.index(name) for name in ('left_thigh', 'left_shank', 'right_thigh', 'right_shank')]
    assert np.degrees(np.median(errors[:, legs])) < 5
    assert errors[0, cmu_mocap.ERRORS.index('contacts')] < .02  # lying on the floor


@pytest.mark.skipif(not (CMU / '140_01.amc').is_file(), reason='CMU subject 140 is downloaded locally')
def test_retarget_moves_no_joint_faster_than_its_limit_and_smooth_keeps_the_floor():
    """Take 01 frames 300-420 at 120 Hz (on hands and knees, the hands lifting; the iteration-64 IK turned a wrist roll
    there at up to 43 rad/s): no joint faster than JOINT_SPEED, the root's xy than ROOT_SPEED, frame to frame;
    smooth keeps the frame count, nothing below the floor, and slows the fastest joint further."""
    env = getup_model_env()
    skeleton = cmu_mocap.read_asf(CMU / '140.asf')
    frames = cmu_mocap.read_amc(CMU / '140_01.amc')
    qpos, _, _ = cmu_mocap.retarget(env, skeleton, frames[299:420], frames[0])
    rate = cmu_mocap.SOURCE_RATE
    joints = np.diff(qpos[:, np.asarray(env.qadr)], axis=0) * rate
    assert np.abs(joints).max() <= cmu_mocap.JOINT_SPEED + 1e-6
    # the root's xy (its height also takes the lift out of the floor after the step)
    assert np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).max() * rate <= cmu_mocap.ROOT_SPEED + 1e-6
    smooth = cmu_mocap.smooth(env, qpos, rate)
    assert smooth.shape == qpos.shape
    model, data, geoms = cmu_mocap._floor_model(env)
    assert min(cmu_mocap.lowest_point(model, data, row, geoms) for row in smooth) >= -1e-9
    assert np.abs(np.diff(smooth[:, np.asarray(env.qadr)], axis=0)).max() <= np.abs(np.diff(
        qpos[:, np.asarray(env.qadr)], axis=0)).max() + 1e-9
    np.testing.assert_allclose(np.linalg.norm(smooth[:, 3:7], axis=1), 1., atol=1e-12)


def test_support_term_is_the_signed_distance_to_the_hull_and_its_gradient():
    """support_term: the centre of mass's signed distance to the convex hull of the touching points (+ inside): standing
    at home with the arms forward, inside the hull of both feet's eight sole spheres and outside that of the left foot's
    four; in both, its gradient matches finite differences of the waist's and arms' coordinates (the feet's collinear
    sole spheres would change the hull's vertices themselves)."""
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    qpos = cmu_mocap.home_qpos(env)
    names = [model.joint(j).name for j in env.joints]
    for side in ('left', 'right'):
        qpos[env.qadr[names.index(f'{side}_shoulder_pitch_joint')]] = -1.2
    qpos[2] -= cmu_mocap.lowest_point(model, data, qpos, geoms)
    soles = [int(g) for g in np.asarray(env.soles).ravel()]
    upper = np.asarray(env.vadr)[12:]  # the waist and the arms
    rng = np.random.default_rng(2)
    for support, inside in ((soles, True), (soles[:4], False)):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        margin, gradient = cmu_mocap.support_term(model, data, support)
        assert (margin > 0) == inside
        for _ in range(3):
            v = np.zeros(model.nv)
            v[upper] = rng.normal(size=len(upper)) * 1e-6
            moved = qpos.copy()
            mujoco.mj_integratePos(model, moved, v, 1.)
            data.qpos[:] = moved
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            changed = cmu_mocap.support_term(model, data, support, gradient=False)[0] - margin
            # within 10 %: a sphere's lowest point is not fixed in its body, the Jacobian takes it as fixed
            assert changed == pytest.approx(gradient @ v, rel=.1, abs=1e-9)


@pytest.mark.skipif(not g1_getup_track.clip_path(LAB_CLIP).is_file(), reason='the lab clip is made locally')
def test_the_lab_cmu_clip_is_the_models_kinematics_and_names_its_data():
    from test_g1_getup_track import _check_kinematics
    clip = g1_getup_track.load(LAB_CLIP)
    header = clip['header']
    assert header['dataset'] == 'cmu' and header['source'] == '140_09.amc' and header['frames'] == [61, 571]
    assert header['source_rate'] == 120 and header['settle_seconds'] == 1. and 'mocap.cs.cmu.edu' in header['data']
    _check_kinematics(clip, range(0, len(clip['qpos']), 9))


def test_the_clip_script_finds_a_subjects_skeleton_and_takes(tmp_path, monkeypatch):
    """getup_track_clip.py --subject: a number is the folder cmu_mocap_<N> under the datasets (its <N>.asf), a path a
    folder with one .asf; a bare take gets the subject's number; without --take only subject 140 has a default."""
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'lab' / 'probes'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # the scripts hide the GPU on import; undone after the test
    clip_script = importlib.import_module('getup_track_clip')
    monkeypatch.setattr(clip_script, 'CMU_DATASETS', tmp_path)
    folder = tmp_path / 'cmu_mocap_111'
    folder.mkdir()
    (folder / '111.asf').write_text(ASF)
    assert clip_script.cmu_subject('111') == (folder, folder / '111.asf', '111')
    assert clip_script.cmu_subject(str(folder)) == (folder, folder / '111.asf', '111')
    assert clip_script.cmu_subject('140', data=folder) == (folder, folder / '111.asf', '111')  # the old --data
    (folder / 'other.asf').write_text(ASF)
    assert clip_script.cmu_subject('111')[1] == folder / '111.asf'
    with pytest.raises(SystemExit):
        clip_script.cmu_subject(str(folder))  # two skeletons and no number to choose by
    with pytest.raises(SystemExit):
        clip_script.cmu_subject('113')  # no folder
    assert clip_script.cmu_take('06', '111') == '111_06' and clip_script.cmu_take('111_07', '111') == '111_07'
    assert clip_script.cmu_take(None, '140') == '140_09'
    with pytest.raises(SystemExit):
        clip_script.cmu_take(None, '111')


def _clip_file(directory, name, header):
    path = directory / f'{name}.npz'
    with open(path, 'wb') as out:
        np.savez(out, qpos=np.zeros((2, 36)), header=np.array(json.dumps(header)))
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_package_takes_a_cmu_clip_with_the_acknowledgement_and_refuses_lafan1(tmp_path, monkeypatch):
    package = importlib.import_module('package')
    clips = tmp_path / 'clips'
    clips.mkdir()
    monkeypatch.setattr(package, 'TRACK_DIR', clips)
    cmu = _clip_file(clips, 'cmu-140_09-1-2', dict(dataset='cmu', source='140_09.amc', frames=[1, 2]))
    lafan = _clip_file(clips, 'lafan1-x-1-2', dict(source='fallAndGetUp1_subject1.csv', frames=[1, 2]))  # no dataset
    cases = (({'file': 'cmu-140_09-1-2', 'md5': cmu}, None), ({'file': 'lafan1-x-1-2', 'md5': lafan}, 'LAFAN1'),
             ({'file': 'cmu-140_09-1-2', 'md5': '0' * 32}, 'md5'), ({'file': 'missing', 'md5': cmu}, 'клипа'),
             ({'file': '../cmu-140_09-1-2', 'md5': cmu}, 'клипа'))
    for k, (setting, refused) in enumerate(cases):
        stage = tmp_path / f'stage-{k}'
        stage.mkdir()
        (stage / 'metadata.json').write_text(json.dumps({'task_config': {'track_clip': setting}}), encoding='utf-8')
        folder = tmp_path / f'export-{k}'
        folder.mkdir()
        (folder / 'deploy.yaml').write_text(f'# humanoid_sim.export_g1: {stage} (100 steps)\n', encoding='utf-8')
        if refused:
            with pytest.raises(SystemExit, match=refused):
                package.check_licence(folder)
        else:
            clip = package.check_licence(folder)
            assert clip == dict(file='cmu-140_09-1-2', dataset='cmu', source='140_09.amc', frames=[1, 2])
            note = package.track_clip_note(clip)
            assert package.CMU_ACKNOWLEDGEMENT in note and '140_09.amc' in note
    assert package.track_clip_note(None) == ''
    assert package.CMU_ACKNOWLEDGEMENT == ('The data used in this project was obtained from mocap.cs.cmu.edu. The '
                                           'database was created with funding from NSF EIA-0196217.')
