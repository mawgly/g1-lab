"""The BONES-SEED source of g1_getup's track clip (scripts/lab/getup_track_clip.py --source bones): the root's Euler
angles (degrees, about the fixed X, then Y, then Z), the CSV's units and header (our 29 joints in our order), the floor
fit (fit_floor: the trunk lifted, a limb in the floor raised by its own joints; drop_to_floor never lowers a frame into
the floor) and scripts/robot/package.py, which lets a stage that followed a BONES clip into a package with the
dataset's attribution in README.md. The data is untracked (.tools/datasets/bones_seed): its tests skip without it."""
import hashlib
import importlib
import json
import math
import os
import subprocess
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
from humanoid_sim.tasks.g1_getup_bank import getup_model_env, joint_names  # noqa: E402

BONES = ROOT / '.tools' / 'datasets' / 'bones_seed' / 'g1_getup'
TAKE = 'stand_up_lying_R_002__A472'


@pytest.fixture
def clip_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'lab' / 'probes'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # the scripts hide the GPU on import; undone after the test
    return importlib.import_module('getup_track_clip')


def _matrix(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def test_root_angles_turn_about_the_fixed_x_then_y_then_z(clip_script):
    angles = np.array([[-39.7, -83.1, -48.4], [10., 20., 30.], [0., 0., 0.], [170., -5., 95.]])
    quats = clip_script.euler_quat(angles)
    for row, q in zip(np.radians(angles), quats):
        expected = _matrix('z', row[2]) @ _matrix('y', row[1]) @ _matrix('x', row[0])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, q)
        np.testing.assert_allclose(rotation.reshape(3, 3), expected, atol=1e-12)
        assert np.linalg.norm(q) == pytest.approx(1.)


def test_a_bones_csv_is_read_in_metres_and_radians_and_another_header_refused(tmp_path, clip_script):
    env = getup_model_env()
    names = joint_names(env)
    header = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ', 'root_rotateX', 'root_rotateY',
              'root_rotateZ', *(f'{n}_dof' for n in names)]
    rows = np.zeros((3, 36))
    rows[:, 0] = [5, 6, 7]
    rows[:, 1:4] = [10., -20., 79.]
    rows[:, 7:] = np.arange(29)
    path = tmp_path / 'take.csv'
    path.write_text(','.join(header) + '\n' + '\n'.join(','.join(f'{v:g}' for v in r) for r in rows) + '\n',
                    encoding='utf-8')
    numbers, qpos = clip_script.read_bones(path, names)
    assert numbers.tolist() == [5, 6, 7]
    np.testing.assert_allclose(qpos[:, :3], [[.1, -.2, .79]] * 3)
    np.testing.assert_allclose(qpos[:, 3:7], [[1, 0, 0, 0]] * 3)
    np.testing.assert_allclose(qpos[0, 7:], np.radians(np.arange(29)))
    swapped = header.copy()
    swapped[7], swapped[8] = swapped[8], swapped[7]
    path.write_text(','.join(swapped) + '\n' + ','.join('0' for _ in header) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        clip_script.read_bones(path, names)


def _standing(env, below):
    """The home pose standing with its lowest robot geom `below` m under the floor."""
    model, data, geoms = cmu_mocap._floor_model(env)
    qpos = cmu_mocap.home_qpos(env)
    qpos[2] -= cmu_mocap.lowest_point(model, data, qpos, geoms) + below
    return qpos


def test_drop_to_floor_lowers_a_floating_body_onto_the_floor_and_never_into_it(clip_script):
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    heights = np.r_[np.full(10, .02), np.linspace(.02, -.01, 10), np.full(10, -.01)]
    qpos = np.stack([_standing(env, -h) for h in heights])
    dropped, drop = clip_script.drop_to_floor(env, qpos)
    lowest = np.array([cmu_mocap.lowest_point(model, data, row, geoms) for row in dropped])
    assert (drop >= 0).all() and (drop <= np.maximum(heights, 0) + 1e-9).all()
    assert (lowest >= np.minimum(heights, 0) - 1e-9).all()  # no frame lower than it was in the floor
    assert lowest[0] == pytest.approx(0., abs=1e-9)  # the floating start put on the floor
    assert (lowest[:5] < .005).all()  # the next nearly: the minimum and the mean over ±5 see the descent 10 ahead
    np.testing.assert_allclose(dropped[-5:], qpos[-5:])  # the frames in the floor left to build's lift


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_fit_floor_raises_a_limb_by_its_own_joints_and_lifts_only_a_trunk_in_the_floor(clip_script, version):
    """Standing 3 cm in the floor: the feet are a limb's, so the legs bend them out (no trunk lift, the root kept, the
    arms and waist untouched, every joint within the soft limits, the wrists at home); lying on the back 2 cm into
    it: the trunk is lifted by that. Both fit versions."""
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    qpos = _standing(env, .03)
    qpos[np.asarray(env.qadr)[env.wrist_mask.numpy()]] = .3
    fitted, stats = clip_script.fit_floor(env, qpos[None], version)
    assert stats['trunk_lift'][0] == 0. and stats['limb_before'][0, 2:].min() == pytest.approx(.03, abs=1e-9)
    assert stats['limb_after'].max() == 0.
    assert cmu_mocap.lowest_point(model, data, fitted[0], geoms) >= -1e-9
    np.testing.assert_allclose(fitted[0, :7], qpos[:7])
    names = joint_names(env)
    q = fitted[0, np.asarray(env.qadr)]
    legs = [k for k, n in enumerate(names) if n.startswith(('left_hip', 'left_knee', 'left_ankle', 'right_hip',
                                                            'right_knee', 'right_ankle'))]
    others = [k for k in range(29) if k not in legs and not env.wrist_mask.numpy()[k]]
    np.testing.assert_allclose(q[others], qpos[np.asarray(env.qadr)][others])
    np.testing.assert_allclose(q[env.wrist_mask.numpy()], env.home[env.wrist_mask.numpy()])
    low, high = task.soft_limits(*model.jnt_range[np.asarray(env.joints)].T)
    assert (q[legs] >= low[legs] - 1e-12).all() and (q[legs] <= high[legs] + 1e-12).all()
    lying = cmu_mocap.home_qpos(env)
    mujoco.mju_axisAngle2Quat(lying[3:7], np.array([0., 1., 0.]), -math.pi / 2)
    trunk = [g for g in geoms if model.body(model.geom_bodyid[g]).name in ('pelvis', 'torso_link')]
    lying[2] -= cmu_mocap.lowest_point(model, data, lying, trunk) + .02
    _, stats = clip_script.fit_floor(env, lying[None], version)
    assert stats['trunk_lift'][0] >= .02 - 1e-9


BRANCH71 = ('faint_stand_up_lying_stomach_R_001__A493', 749, 1237, '72837c64a1f4f923e64b8fa54848dcdf')  # its clip v1
BACK = ('faint_stand_up_lying_R_001__A495', 785, 1234)  # v1 threw the right hip pitch at 152 rad/s (data ≤ 8.2)


@pytest.mark.skipif(not (BONES / f'{BRANCH71[0]}.csv').is_file(), reason='BONES-SEED is downloaded locally')
def test_fit_v1_rebuilds_branch_71s_clip_byte_for_byte(tmp_path):
    """exp-f52f908587b3419d trains on bones-…A493-749-1237 by md5: --fit v1 makes that very file again (v1's header
    carries no bones_fit), so the fit v2 changes no clip a stage names."""
    take, first, last, md5 = BRANCH71
    out = tmp_path / 'clip.npz'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'lab' / 'getup_track_clip.py'), '--source', 'bones',
                    '--fit', 'v1', '--file', take, '--frames', str(first), str(last), '--out', str(out)],
                   check=True, capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    assert hashlib.md5(out.read_bytes()).hexdigest() == md5
    assert 'bones_fit' not in json.loads(str(np.load(out)['header']))


@pytest.mark.skipif(not (BONES / f'{BACK[0]}.csv').is_file(), reason='BONES-SEED is downloaded locally')
def test_fit_v2_bounds_the_joint_speeds_and_puts_a_palm_that_was_on_the_floor_on_ours(clip_script):
    """The back take that v1 broke: after v2 no joint is faster than 30 rad/s between 120 Hz frames (v1 152, the data
    8.2), and every frame where an arm was in or on the floor in the data has its lowest point at LIMB_CLEARANCE
    (v1 lifted the palm up to 25 cm off it); no limb left deeper than v1 left it."""
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    take, first, last = BACK
    numbers, qpos = clip_script.read_bones(BONES / f'{take}.csv', joint_names(env))
    qpos = qpos[first - numbers[0]:last - numbers[0] + 1]
    speeds = {}
    for version in ('v1', 'v2'):
        fitted, stats = clip_script.fit_floor(env, qpos, version)
        speeds[version] = float(np.abs(np.diff(fitted[:, 7:], axis=0)).max()) * clip_script.BONES_RATE
        if version == 'v1':
            v1_after = stats['limb_after'].max(0)
    assert speeds['v1'] > 100. and speeds['v2'] < 30.
    on_floor = stats['arm_on_floor']
    assert on_floor[:, :2].sum(0).min() > 20 and not on_floor[:, 2:].any()
    heights = stats['limb_height'][:, :2][on_floor[:, :2]]
    assert np.abs(heights - clip_script.LIMB_CLEARANCE).max() < clip_script.FIT_TOLERANCE
    assert (stats['limb_after'].max(0) <= v1_after + 1e-3).all()
    arms = [g for g in geoms if any(p in model.body(model.geom_bodyid[g]).name for p in ('shoulder', 'elbow', 'wrist'))]
    for k in np.flatnonzero(on_floor[:, :2].any(1))[::15]:  # the fitted frames themselves: an arm at the floor
        assert abs(cmu_mocap.lowest_point(model, data, fitted[k], arms) - clip_script.LIMB_CLEARANCE) < .002


@pytest.mark.skipif(not (BONES / f'{TAKE}.csv').is_file(), reason='BONES-SEED is downloaded locally')
def test_bones_joints_are_ours_standing_level_and_upright(clip_script):
    """The take's last frame (standing): our joint names in our order (read_bones checks the header), the pelvis
    upright, both feet's eight sole spheres level within 1 cm and within 2.5 cm of the floor: the same zeros."""
    env = getup_model_env()
    numbers, qpos = clip_script.read_bones(BONES / f'{TAKE}.csv', joint_names(env))
    assert numbers[0] == 0 and len(numbers) == 742  # the metadata's move_duration_frames, 120 frames/s
    model, data = env.model, env.data
    data.qpos[:] = qpos[-1]
    mujoco.mj_forward(model, data)
    assert data.xmat[env.pelvis][8] > .99
    soles = [data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in np.asarray(env.soles).ravel()]
    assert max(soles) - min(soles) < .01 and abs(np.mean(soles)) < .025


def _clip_file(directory, name, header):
    path = directory / f'{name}.npz'
    with open(path, 'wb') as out:
        np.savez(out, qpos=np.zeros((2, 36)), header=np.array(json.dumps(header)))
    return hashlib.md5(path.read_bytes()).hexdigest()


def _export(tmp_path, k, config):
    stage = tmp_path / f'stage-{k}'
    stage.mkdir()
    (stage / 'metadata.json').write_text(json.dumps({'task_config': config}), encoding='utf-8')
    folder = tmp_path / f'export-{k}'
    folder.mkdir()
    (folder / 'deploy.yaml').write_text(f'# humanoid_sim.export_g1: {stage} (100 steps)\n', encoding='utf-8')
    return folder


def test_package_takes_a_bones_clip_with_its_attribution(tmp_path, monkeypatch):
    package = importlib.import_module('package')
    clips = tmp_path / 'clips'
    clips.mkdir()
    monkeypatch.setattr(package, 'TRACK_DIR', clips)
    bones = _clip_file(clips, 'bones-a', dict(dataset='bones_seed', source=f'{TAKE}.csv', frames=[227, 519]))
    cmu = _clip_file(clips, 'cmu-a', dict(dataset='cmu', source='140_01.amc', frames=[1, 2]))
    clip = package.check_licence(_export(tmp_path, 0, {'track_clip': {'file': 'bones-a', 'md5': bones}}))
    assert clip == dict(file='bones-a', dataset='bones_seed', source=f'{TAKE}.csv', frames=[227, 519],
                        datasets=['bones_seed'])
    note = package.track_clip_note(clip)
    assert package.BONES_ATTRIBUTION in note and TAKE in note and package.CMU_ACKNOWLEDGEMENT not in note
    assert package.BONES_ATTRIBUTION == ('Training data includes Motion Data by Bones Studio (https://bones.studio/). '
                                         'Use of the underlying dataset is subject to the BONES Motion Capture Dataset '
                                         'License Agreement.')
    # a settling stage with clips of both: each dataset credited
    mixed = package.check_licence(_export(tmp_path, 1, {'settle_clip': [{'file': 'cmu-a', 'md5': cmu},
                                                                         {'file': 'bones-a', 'md5': bones}]}))
    assert mixed['dataset'] == 'cmu' and mixed['datasets'] == ['cmu', 'bones_seed']
    note = package.track_clip_note(mixed)
    assert package.CMU_ACKNOWLEDGEMENT in note and package.BONES_ATTRIBUTION in note
    # the attribution the clip script writes into the header is the package's
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
    assert importlib.import_module('getup_track_clip').BONES_ATTRIBUTION == package.BONES_ATTRIBUTION


@pytest.mark.skipif(not g1_getup_track.clip_path(f'bones-{TAKE}-227-519').is_file(), reason='the clip is made locally')
def test_the_lab_bones_clip_is_the_models_kinematics_and_names_its_data():
    from test_g1_getup_track import _check_kinematics
    clip = g1_getup_track.load(f'bones-{TAKE}-227-519')
    header = clip['header']
    assert header['dataset'] == 'bones_seed' and header['source'] == f'{TAKE}.csv' and header['frames'] == [227, 519]
    assert header['source_rate'] == 120 and header['root_order'] == 'extrinsic xyz' and 'bones.studio' in header['data']
    assert max(header['limb_depth_after']) < .03 and clip['qpos'][0, 2] < .15  # lying on the floor, not lifted
    _check_kinematics(clip, range(0, len(clip['qpos']), 9))
