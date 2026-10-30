"""The getting-up chain probe (scripts/lab/probes/getup_chain.py, iteration 82) runs end to end: one held-out back row,
physics ours, both hand-overs (direct and pd), B cut short; the records carry the hand-over and the seam numbers.
Needs the lab's stages, start bank and clip under runs/ (untracked): skipped without them."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'scripts' / 'lab' / 'probes'


@pytest.fixture(scope='module')
def chain():
    for path in (PROBES, PROBES.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module('getup_chain')
    from humanoid_sim.tasks import g1_getup_bank, g1_getup_track
    if not ((ROOT / module.FIRST).is_dir() and (ROOT / module.SECOND).is_dir()):
        pytest.skip('the chain stages are not on this machine')
    try:
        bank = g1_getup_bank.load()
        g1_getup_track.load('bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3')
    except (OSError, ValueError, SystemExit):
        pytest.skip('the start bank or the A493 v3 clip is not on this machine')
    return module, bank


def test_hand_over_rule_fires_after_the_least():
    sys.path.insert(0, str(PROBES))
    module = importlib.import_module('getup_chain')
    rule = module.HandOver(.02)
    face = np.radians(30.)
    assert rule(1, np.radians(80.), 2.) is None  # not face down yet
    assert rule(2, face, 1.) is None and rule(3, face, .9) is None
    assert rule(4, face, .9 + .3 * module.PASS_MARGIN / 2) is None  # within the margin: noise, not the least passed
    assert rule(5, face, .9 + .3 * module.PASS_MARGIN + .01) == 'least'  # deviation's score is |q − q0| / 0.3
    assert module.HandOver(.02)(1, np.radians(10.), .1) == 'good'
    left = module.HandOver(.02)
    assert left(1, face, 1.) is None and left(2, np.radians(60.), 1.) == 'left'


def test_torso_rule_waits_for_the_chest_down():
    """--rule torso: nothing before face down, nothing face down with the torso beyond TORSO_GAP, 'torso' at the
    first step face down with the torso within it; the score rules never fire."""
    sys.path.insert(0, str(PROBES))
    module = importlib.import_module('getup_chain')
    rule = module.HandOver(.02, torso=True)
    face = np.radians(30.)
    assert rule(1, np.radians(80.), 2., 0.) is None and rule.faced_at is None
    assert rule(2, face, 1., .7) is None and rule.faced_at == 2
    assert rule(3, face, 1.5, .7) is None  # the old 'least' would have fired
    assert rule(4, np.radians(60.), 1., .1) is None  # left face down: waits
    assert rule(5, face, .1, module.TORSO_GAP + .01) is None  # the old 'good' would have fired
    assert rule(6, face, 1., module.TORSO_GAP) == 'torso'


def test_torso_gravity_is_the_kinematics_of_the_pelvis_and_waist():
    """torso_gravity from the pelvis's projected gravity and the joints equals the torso's own in a posed model
    (random pelvis rotation and waist angles), and torso_gap compares its z with a goal's."""
    import mujoco
    from humanoid_sim.tasks import g1_getup_bank
    from humanoid_sim.tasks import g1_tensor_math as task
    sys.path.insert(0, str(PROBES))
    module = importlib.import_module('getup_chain')
    raw = g1_getup_bank.getup_model_env()
    rng = np.random.default_rng(0)
    data = mujoco.MjData(raw.model)
    for _ in range(5):
        quat = rng.normal(size=4)
        data.qpos[:] = 0
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[raw.qadr] = raw.home + rng.uniform(-.4, .4, raw.model.nu)
        mujoco.mj_kinematics(raw.model, data)
        pelvis = task.projected_gravity(data.xmat[raw.pelvis].reshape(1, 3, 3))[0]
        torso = task.projected_gravity(data.xmat[raw.torso].reshape(1, 3, 3))[0]
        np.testing.assert_allclose(module.torso_gravity(raw, pelvis, data.qpos[raw.qadr]), torso, atol=1e-12)
        goal = np.array([0., 0., -.5])
        assert module.torso_gap(raw, pelvis, data.qpos[raw.qadr], goal) == pytest.approx(abs(torso[2] + .5), abs=1e-12)


def test_pd_transition_shapes_parse_and_refuse():
    """--handover pd S key=value …: PD_DEFAULTS' keys as floats in a sorted tuple, named back the same way; plain
    pd S has the empty shape (the first version's transition); bad keys and values stop the probe."""
    sys.path.insert(0, str(PROBES))
    module = importlib.import_module('getup_chain')
    assert module.variants([['direct'], ['pd', '1.0']]) == [('direct', None, ()), ('pd', 1., ())]
    (chosen,) = module.variants([['pd', '3.5', 'kp=0.3', 'kd=3', 'kp_to=1', 'kd_to=1', 'arms=0.3']])
    assert chosen == ('pd', 3.5, (('arms', .3), ('kd', 3.), ('kd_to', 1.), ('kp', .3), ('kp_to', 1.)))
    assert module.name(*chosen) == 'pd 3.5 arms=0.3 kd=3 kd_to=1 kp=0.3 kp_to=1'
    assert set(module.PD_DEFAULTS) == {'ramp', 'arms', 'legs', 'kp', 'kd', 'kp_to', 'kd_to'}
    for bad in (['pd', '3', 'stiff=1'], ['pd', '3', 'kp=0'], ['pd', '3', 'ramp=1.5'], ['pd', '3', 'legs=2.5'],
                ['pd', '3', 'kd'], ['pd', '9']):
        with pytest.raises(SystemExit):
            module.variants([bad])


@pytest.fixture
def original_step():
    """getup_probe's monitor replaces mujoco.mj_step for the process (two arguments only); put it back for the other
    tests, which also call mj_step(model, data, nstep)."""
    import mujoco
    step = mujoco.mj_step
    yield
    mujoco.mj_step = step


def test_chain_runs_both_hand_overs(chain, original_step):
    module, bank = chain
    row = int(np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index('back')))[0])
    for handover, seconds in (('direct', None), ('pd', .5)):
        record = module.run(dict(first=str(ROOT / module.FIRST), second=str(ROOT / module.SECOND), handover=handover,
                                 pd_seconds=seconds, physics='ours', start='back', row=row, seed=module.SEED,
                                 first_seconds=module.FIRST_SECONDS, second_seconds=.2))
        assert record['rule'] is not None and record['chained'] and record['fell'] is None
        assert 0 < record['handover_s'] < module.FIRST_SECONDS
        assert len(record['seam_speed']) == (1 if handover == 'direct' else 2)
        assert record['frame'] == module.TARGET_FRAME if handover == 'pd' else 0 <= record['frame'] <= record['clip_last']
        assert np.isfinite(record['target_jump']) and record['hand_max'] >= 0 and record['feet_end'] > 0
        assert set(record['hand_phase']) == ({'A', 'B'} if handover == 'direct' else {'A', 'pd', 'B'})
