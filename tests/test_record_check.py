"""scripts/robot/record_check.py's check of g1_ctrl's output against the patches' info lines (iteration 38): the GetUp
state's own lines of g1_ctrl-getup.patch are required only when the log shows the state built, so a walking session
with GetUp off (records A-D) passes and a getting-up session without them stops."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'scripts/sim2sim/patches'
# g1_ctrl's output with every patch of scripts/sim2sim/patches and GetUp off (sim2sim run 20261024-221822, cut down)
WALKING = """[info] JointAction: targets are published clipped (sim2sim clip-race patch)
[info] Initializing State_Velocity ...
[info] Policy directory: /home/mawgly/unitree/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v9_lab
[info] Velocity: pelvis tilt past 57.3 deg goes to Passive (sim2sim fall-passive patch)
[info] JointAction: targets are published clipped (sim2sim clip-race patch)
[info] FSM: Start Passive
[info] FSM: Change state from FixStand to Velocity
[info] Velocity: kept the previous target for 0 FSM ticks until the first policy step
"""
# the same with GetUp on (install_policy.sh getup; sim2sim run 20261024-222419-getup)
GETUP_BUILT = """[info] Initializing State_GetUp ...
[info] Policy directory: /home/mawgly/unitree/unitree_rl_mjlab/deploy/robots/g1/config/policy/getup/v9_lab
[info] JointAction: targets are published clipped (sim2sim clip-race patch)
"""
GETUP_LINES = """[info] GetUp: no pelvis tilt check, fall_tilt 0 in config.yaml (sim2sim getup patch)
[info] GetUp: Passive after 10.0 s in the state (sim2sim getup patch)
[info] GetUp: Passive once a joint is faster than its max_joint_speed, 20.0-50.0 rad/s over 29 SDK motors (sim2sim getup patch)
"""


@pytest.fixture
def record_check(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/robot'))
    return importlib.import_module('record_check')


def check(record_check, tmp_path, text):
    log = tmp_path / 'g1_ctrl.log'
    log.write_text(text, encoding='utf-8')
    return record_check.log_check(log, PATCHES, 1)


def missing(result):
    return sorted(x['text'] for x in result['lines'] if x['required'] and not x['found'])


def test_walking_log_without_getup_passes(record_check, tmp_path):
    result = check(record_check, tmp_path, WALKING)
    assert result['ok'] and not result['getup'] and not result['chain']
    skipped = [x for x in result['lines'] if not x['required']]
    getup = [x for x in skipped if x['patch'] == 'g1_ctrl-getup.patch']
    assert len(getup) == 3 and all(not x['found'] for x in skipped)
    assert {x['patch'] for x in skipped} == {'g1_ctrl-getup.patch', 'g1_ctrl-getupchain.patch'}
    # the getup patch's tilt check line is Velocity's: still required
    assert any(x['patch'] == 'g1_ctrl-getup.patch' and x['required'] and x['found'] for x in result['lines'])


def test_walking_log_without_a_patch_line_stops(record_check, tmp_path):
    result = check(record_check, tmp_path, WALKING.replace('(sim2sim fall-passive patch)', ''))
    assert not result['ok']
    assert missing(result) == ['{}: pelvis tilt past {:.1f} deg goes to Passive (sim2sim fall-passive patch)'] * 2


def test_getup_built_without_its_lines_stops(record_check, tmp_path):
    result = check(record_check, tmp_path, GETUP_BUILT + WALKING)
    assert result['getup'] and not result['ok']
    assert len(missing(result)) == 3
    assert all(x['patch'] == 'g1_ctrl-getup.patch' for x in result['lines'] if x['required'] and not x['found'])


def test_getup_built_with_its_lines_passes(record_check, tmp_path):
    result = check(record_check, tmp_path, GETUP_BUILT + GETUP_LINES + WALKING)
    assert result['getup'] and result['ok']
    assert all(x['required'] for x in result['lines'] if x['patch'] != 'g1_ctrl-getupchain.patch')
    assert not any(x['required'] for x in result['lines'] if x['patch'] == 'g1_ctrl-getupchain.patch')


# the getting-up chain on (run.sh --getup-track, g1_ctrl-getupchain.patch): its two states built, its info lines
CHAIN_BUILT = """[info] Initializing State_GetUpBlend ...
[info] Initializing State_GetUpTrack ...
"""
CHAIN_LINES = """[info] GetUp: hands over to GetUpBlend toward the tracker's frame 18 within 150 policy steps (sim2sim getupchain patch)
[info] GetUpBlend: PD from the last target to frame 18 over 175 policy steps (arms 0.30, others 0.70 of them, the others 0.00 s later), kp x0.30 -> x1.00, kd x3.00 -> x1.00, then GetUpTrack (sim2sim getupchain patch)
"""


def test_chain_built_without_its_lines_stops_and_with_them_passes(record_check, tmp_path):
    result = check(record_check, tmp_path, CHAIN_BUILT + GETUP_BUILT + GETUP_LINES + WALKING)
    assert result['chain'] and not result['ok']
    assert {x['patch'] for x in result['lines'] if x['required'] and not x['found']} == {'g1_ctrl-getupchain.patch'}
    result = check(record_check, tmp_path, CHAIN_BUILT + CHAIN_LINES + GETUP_BUILT + GETUP_LINES + WALKING)
    assert result['chain'] and result['ok'] and all(x['required'] for x in result['lines'])
