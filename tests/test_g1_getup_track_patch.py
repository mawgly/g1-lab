"""Patching a made track clip (scripts/lab/getup_track_clip.py --patch / --base, iteration 72): the slowed clock, and
the A493 front clip v3 (--patch a493-kneel) against its v2: the frames no change touches are v2's arrays exactly, the
length, the left hand on the floor while it is held, the waist below its ceiling, no joint faster than in v2. The clip
is made locally from untracked data: those tests skip without it. --clear-self (iteration 79) on synthetic frames: the
untouched frames bit for bit, the gap reached, a hand on the floor kept there, clip_feasibility's penetration warning."""
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
cmu_mocap = importlib.import_module('cmu_mocap')
from humanoid_sim.tasks import g1_getup_track  # noqa: E402
from humanoid_sim.tasks.g1_getup_bank import getup_model_env  # noqa: E402

V2 = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v2'
V3 = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3'
HAVE = g1_getup_track.clip_path(V2).is_file() and g1_getup_track.clip_path(V3).is_file()


@pytest.fixture
def clip_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'lab' / 'probes'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return importlib.import_module('getup_track_clip')


def test_slow_clock_stretches_the_span_and_keeps_the_other_frames_whole(clip_script):
    clock = clip_script.slow_clock(254, .02, 2.0, 2.8, 1.5)
    assert len(clock) == 274 and clock[0] == 0. and clock[-1] == 253.
    np.testing.assert_array_equal(clock[:101], np.arange(101))  # up to the span's start one to one
    np.testing.assert_array_equal(clock[160:], np.arange(140, 254))  # after it shifted by 20 frames
    step = np.diff(clock)
    assert (step > 0).all() and step.max() <= 1. + 1e-12  # never faster than the source
    assert step[100:160].min() == pytest.approx((40 - 10) / (60 - 10))  # the plateau c, ramps of 0.2 s
    with pytest.raises(SystemExit):
        clip_script.slow_clock(254, .02, 2.0, 2.81, 1.5)  # not whole frames


def test_window_is_zero_outside_its_ramps(clip_script):
    t = np.arange(274) * .02
    weight = clip_script._window(t, 2.0, 3.8)
    assert (weight[t <= 1.9 + 1e-9] == 0.).all() and (weight[t >= 3.9 - 1e-9] == 0.).all()
    assert (weight[(t >= 2.0 - 1e-9) & (t <= 3.8 + 1e-9)] == 1.).all()


@pytest.mark.skipif(not HAVE, reason='the A493 clips are made locally (BONES-SEED)')
def test_v3_keeps_v2s_frames_outside_its_changes_and_records_them():
    v2, v3 = g1_getup_track.load(V2), g1_getup_track.load(V3)
    assert len(v3['qpos']) == len(v2['qpos']) + 20 == 274
    patch = v3['header']['patch']
    assert patch['base'] == V2 and patch['base_md5'] == v2['md5'] and patch['name'] == 'a493-kneel'
    assert patch['unchanged_frames'] == [[0, 95], [215, 273]]
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(v3[key][:96], v2[key][:96])  # 0-1.9 s bit for bit
        np.testing.assert_array_equal(v3[key][215:], v2[key][195:])  # from v2's 3.9 s on, 20 frames later
    assert v3['header']['seconds'] == 5.46 and v3['header']['source_md5'] == v2['header']['source_md5']


@pytest.mark.skipif(not HAVE, reason='the A493 clips are made locally (BONES-SEED)')
def test_v3_holds_the_left_hand_on_the_floor_keeps_the_waist_down_and_no_joint_faster():
    v2, v3 = g1_getup_track.load(V2), g1_getup_track.load(V3)
    patch = v3['header']['patch']
    hold = patch['hold_hand']
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    hand = [g for g in geoms if model.body(model.geom_bodyid[g]).name.startswith('left_wrist')]
    t = np.arange(len(v3['qpos'])) * env.dt
    held = np.flatnonzero((t >= hold['last_touch_seconds'] - 1e-9) & (t <= hold['held_until'] + 1e-9))
    assert len(held) >= 5
    for k in held:  # on the floor at its last touch point
        data.qpos[:] = v3['qpos'][k]
        mujoco.mj_kinematics(model, data)
        lowest = min((cmu_mocap.geom_lowest(model, data, g) for g in hand), key=lambda p: p[2])
        assert np.linalg.norm(lowest - hold['point']) < 2e-3
    for k in np.flatnonzero((t > 1.9 + 1e-9) & (t <= hold['last_touch_seconds'] + 1e-9)):  # touching before it
        assert cmu_mocap.lowest_point(model, data, v3['qpos'][k], hand) < 2e-3
    waist = model.jnt_qposadr[model.joint('waist_pitch_joint').id]
    clock = np.interp(t, [0, 2.0, 3.2, 5.46], [0, 2.0, 2.8, 5.06])  # v2's seconds at the span's ends
    inside = (clock >= 2.0) & (clock <= 3.8)
    assert v3['qpos'][inside, waist].max() <= .4 + 1e-9 and v2['qpos'][100:191, waist].max() > .5
    speed = lambda q: np.abs(np.diff(q[:, 7:], axis=0)).max() / env.dt
    assert speed(v3['qpos']) <= speed(v2['qpos']) + 1e-9


@pytest.mark.skipif(not HAVE, reason='the A493 clips are made locally (BONES-SEED)')
def test_patch_rebuilds_v3_byte_for_byte(tmp_path):
    out = tmp_path / 'clip.npz'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'lab' / 'getup_track_clip.py'), '--patch', 'a493-kneel',
                    '--out', str(out)], check=True, capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    assert hashlib.md5(out.read_bytes()).hexdigest() == g1_getup_track.load(V3)['md5']
    assert json.loads(str(np.load(out)['header']))['patch']['options']['slow'] == [2.0, 2.8, 1.5]


def _home_on_floor(env):
    model, data, geoms = cmu_mocap._floor_model(env)
    row = cmu_mocap.home_qpos(env)
    row[2] -= cmu_mocap.lowest_point(model, data, row, geoms)
    return row


def _arm_gaps(clip_script, env, frames, gap=.01):
    """(F, 2) the smallest distance of each arm to the legs, the trunk and the other arm (at most gap), m."""
    model, data, arms, others = clip_script._clear_model(env, gap)
    jacp = np.zeros((3, model.nv))
    return np.array([[clip_script._arm_terms(model, data, np.array(row), arm, others[a], gap, None, -1., jacp)[1]
                      for a, arm in enumerate(arms)] for row in frames])


def test_clear_self_moves_the_arm_off_the_hip_and_keeps_the_other_frames_bit_for_bit(clip_script):
    """Standing at home, the right shoulder rolled 0.3 rad in over frames 4-14 and back by 24 puts the right hand 8 cm
    into hip_roll: after --clear-self every arm is the gap away from the legs, the trunk and the other arm, nothing
    deeper than 1 cm is left, only the arms' shoulder and elbow joints moved, and the frames the correction never
    reached are the clip's arrays exactly."""
    import clip_feasibility
    env = getup_model_env()
    model = env.model
    count = 90
    qpos = np.repeat(_home_on_floor(env)[None], count, 0)
    roll = model.jnt_qposadr[model.joint('right_shoulder_roll_joint').id]
    qpos[:, roll] += .3 * np.clip(np.minimum(np.arange(count) - 4, 24 - np.arange(count)) / 10, 0, 1)
    clip = g1_getup_track.build(env, qpos)
    before = clip_feasibility.check_clip(clip, env)['summary']
    assert before['self_deep_frames'] > 0 and before['self_depth_max'] > .05
    assert 'предупреждение: проникновение' in clip_feasibility.verdict_text(before)
    out, record = clip_script.patch_clip(env, clip, clear=.01)
    cleared = record['clear_self']
    assert cleared['deep_frames_before'] == before['self_deep_frames'] and cleared['deep_frames_after'] == 0
    assert cleared['unreached'] == [] and cleared['closer_frames_after'] == [0, 0]
    assert (_arm_gaps(clip_script, env, out['qpos']) >= .01 - clip_script.CLEAR_TOLERANCE).all()
    after = clip_feasibility.check_clip(out, env)['summary']
    assert after['self_deep_frames'] == 0 and after['warnings'] == []
    arm = [model.jnt_qposadr[model.joint(f'{side}_{j}_joint').id] for side in ('left', 'right')
           for j in clip_script.CLEAR_JOINTS]
    rest = np.setdiff1d(np.arange(model.nq), arm)
    np.testing.assert_array_equal(out['qpos'][:, rest], clip['qpos'][:, rest])
    kept = [f for first, last in record['unchanged_frames'] for f in range(first, last + 1)]
    assert len(kept) >= 20 and kept[-1] == count - 1 and len(kept) == count - cleared['frames_changed']
    for key in g1_getup_track.ARRAYS:
        np.testing.assert_array_equal(out[key][kept], clip[key][kept])
    assert record['joint_speed_max'][1] < 5.  # rad/s: the correction comes in smoothly


def test_clear_self_keeps_a_hand_on_the_floor_where_it_is(clip_script):
    """A right arm pose whose shoulder roll link is 1.9 cm into the torso, the body lowered so that the right hand's
    lowest point is 1 mm above the floor: the arm is cleared by its shoulder and elbow while that point stays put."""
    env = getup_model_env()
    model, data, geoms = cmu_mocap._floor_model(env)
    row = _home_on_floor(env)
    arm = model.jnt_qposadr[[model.joint(f'right_{j}_joint').id for j in clip_script.CLEAR_JOINTS]]
    row[arm] = [-2.205, .46, 2.014, 1.847]
    hand = [g for g in geoms if model.body(model.geom_bodyid[g]).name.startswith('right_wrist')]
    row[2] -= cmu_mocap.lowest_point(model, data, row, hand) - .001
    qpos = np.repeat(row[None], 12, 0)
    assert (_arm_gaps(clip_script, env, qpos)[:, 1] < -.015).all()

    def lowest(frame):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        return min((cmu_mocap.geom_lowest(model, data, g) for g in hand), key=lambda p: p[2])
    rows, changed, record = clip_script.clear_self(env, qpos, .01)
    assert changed.all() and record['hand_held_frames'][1] == 12 and record['unreached'] == []
    assert (_arm_gaps(clip_script, env, rows)[:, 1] >= .01 - clip_script.CLEAR_TOLERANCE).all()
    start = lowest(row)
    for frame in rows:
        assert np.linalg.norm(lowest(frame) - start) < 1e-3  # on the floor, where it was
