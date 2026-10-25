"""scripts/sim2sim getting-up tools (iteration 35): getup_report.py's torso and head from the lowstate (pelvis IMU,
waist angles) against MuJoCo's forward kinematics of the G1, what a pose lies on, and gamepad.py's --fall-push."""
import importlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import load_model

ROOT = Path(__file__).resolve().parents[1]
WAIST = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/sim2sim'))
    return importlib.import_module('getup_report'), importlib.import_module('gamepad')


def lowstate_of(model, data, columns=102):
    """One lowstate.csv row (sim2sim_probe.cpp's columns) and one sport.csv row of a MuJoCo state: the pelvis
    quaternion as the IMU, the waist angles at their SDK indices, the pelvis position as sportmodestate's."""
    import analyze
    ls = np.zeros((1, columns))
    ls[0, analyze.LS_QUAT] = data.qpos[3:7]
    q = np.zeros(29)
    for index, name in zip((12, 13, 14), WAIST):
        q[index] = data.qpos[model.joint(name).qposadr[0]]
    ls[0, analyze.LS_Q] = q
    sport = np.zeros((1, 7))
    sport[0, 1:4] = data.qpos[:3]
    return ls, sport


def test_torso_and_head_follow_mujoco(tools):
    """Random pelvis poses and waist angles: the torso rotation, the top of the head (g1_tensor_math.head_position) and
    the pelvis position getup_report.py rebuilds match MuJoCo's on the G1 files (unitree_mujoco's offsets)."""
    report, _ = tools
    model, _ = load_model()
    data = mujoco.MjData(model)
    torso = model.body('torso_link').id
    rng = np.random.default_rng(0)
    for _ in range(20):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = rng.uniform(-1, 1, 3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        for name in WAIST:
            low, high = model.jnt_range[model.joint(name).id]
            data.qpos[model.joint(name).qposadr[0]] = rng.uniform(low, high)
        mujoco.mj_forward(model, data)
        ls, sport = lowstate_of(model, data)
        rotation, head, pelvis = report.torso_and_head(ls, sport)
        expected = data.xmat[torso].reshape(1, 3, 3)
        np.testing.assert_allclose(rotation, expected, atol=1e-9)
        np.testing.assert_allclose(head, task.head_position(data.xpos[torso][None], expected), atol=1e-6)
        np.testing.assert_allclose(pelvis[0], data.qpos[:3], atol=1e-12)


def test_lying_on(tools):
    """Back: the torso's forward axis up; front: down; a side: that side down; upright: sitting; above FALL_Z: standing."""
    report, _ = tools
    pitch = lambda angle: report.axis_rotation(np.array([angle]), 1)[0]  # noqa: E731
    roll = lambda angle: report.axis_rotation(np.array([angle]), 0)[0]  # noqa: E731
    assert report.lying_on(pitch(-np.pi / 2), .08) == 'спина'
    assert report.lying_on(pitch(np.pi / 2), .08) == 'живот'
    assert report.lying_on(roll(-np.pi / 2), .15) == 'левый бок'
    assert report.lying_on(roll(np.pi / 2), .15) == 'правый бок'
    assert report.lying_on(np.eye(3), .3) == 'сидя / на коленях'
    assert report.lying_on(np.eye(3), .79) == 'не упал'


def test_handover(tools, tmp_path):
    """The hand-over: the pose over GetUp's last POSE_S against default_joint_pos mapped to SDK order through
    joint_ids_map, torso and pelvis tilt there, joint speeds only within Velocity's first HANDOVER_S, the arms
    apart."""
    report, _ = tools
    import analyze
    order = list(range(29))[::-1]  # policy joint i is SDK motor 28 - i
    home = np.linspace(-.5, .5, 29)  # SDK order
    (tmp_path / 'getup').mkdir()
    (tmp_path / 'getup' / 'deploy.yaml').write_text(
        f'joint_ids_map: {order}\ndefault_joint_pos: {home[order].tolist()}\n', encoding='utf-8')
    t = np.arange(-500, 1000) * .002  # s from RT + A, 0 exactly at index 500
    t_exit = 10e9
    ls = np.zeros((len(t), 102))
    ls[:, analyze.LS_T] = t_exit + t * 1e9
    ls[:, analyze.LS_Q] = home
    ls[:, analyze.LS_Q.start + 24] += .6  # R_sh_y 0.6 rad off home
    ls[:, analyze.LS_Q.start + 3] -= .2  # L_knee 0.2 rad off
    ls[(t >= 0) & (t < .1), analyze.LS_DQ.start + 24] = 6.  # the arm swings back in Velocity's first second
    ls[(t >= 0) & (t < .1), analyze.LS_DQ.start + 9] = 4.
    ls[t > 1.5, analyze.LS_DQ.start] = 9.  # later than HANDOVER_S: not the hand-over's
    ls[t < 0, analyze.LS_DQ.start + 1] = 7.  # still in GetUp
    pitch = lambda deg: report.axis_rotation(np.full(len(t), np.radians(deg)), 1)  # noqa: E731
    w, y = np.cos(np.radians(10.)), np.sin(np.radians(10.))
    ls[:, analyze.LS_QUAT] = (w, 0., y, 0.)  # pelvis 20 deg about y
    h = report.handover(tmp_path, ls, pitch(8.), t_exit)
    assert h['pose_sq'] == pytest.approx(.6 ** 2 + .2 ** 2)
    assert [name for name, _, _ in h['farthest'][:2]] == ['R_sh_y', 'L_knee']
    assert h['farthest'][0][1:] == pytest.approx((home[24] + .6, home[24]))
    assert h['torso_tilt_deg'] == pytest.approx(8.)
    assert h['pelvis_tilt_deg'] == pytest.approx(20.)
    assert h['speeds']['max'] == 6. and h['speeds']['max_joint'] == 'R_sh_y'
    assert h['speeds']['per_joint']['L_hip_p'] == 0. and h['speeds']['per_joint']['L_hip_r'] == 0.
    assert h['arm'] == dict(joint='R_sh_y', speed=6., q_before=pytest.approx(home[24] + .6), home=home[24])


def test_fall_push(tools):
    """--fall-push '<direction>[:dv]': dv 0.5 m/s without it, every dv the push keys give has its modifiers."""
    _, gamepad = tools
    assert gamepad.fall_push('back') == ('back', .5)
    assert gamepad.fall_push('forward:2.25') == ('forward', 2.25)
    assert gamepad.push_modifiers(gamepad.fall_push('left:1.5')[1]) == ('Alt',)
    assert gamepad.push_modifiers(2.) is not None and gamepad.push_modifiers(2.4) is None
