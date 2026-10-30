"""g1_getup task_config track_handover_starts (iteration 83), CPU: a share of track_clip's training resets starts from
the state another getting-up policy hands over in (the hand-over bank, tasks/g1_getup_handover, made by
scripts/lab/getup_handover_bank.py). The setting (bounds, needs track_clip, not part of the contract); off (missing or
null) the task is bit for bit the old one and no bank is read; a start from the bank is its row's state exactly, the
first policy's actions in flight, no noise, the clock at the bank's frame and the reference aligned under the robot;
the share over 4000 resets; its ends' rule waits TRACK_HANDOVER_GRACE_SECONDS; the bank's clip must be the task's; the
file round trip. getup_handover_bank.py --pd (iteration 86): without it the rows are the old ones, with it the state at
the PD transition's end is getup_chain's, and such a row starts the tracker. CPU–GPU: tests/test_g1_gpu.py -k
handover."""
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.tasks import g1_getup_handover
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest
from test_g1_getup_track import MD5, SETTING, synthetic_clip, track_env
from test_g1_getup_track_reach import _follow, _shift_ankle
from test_g1_getup_track_sampling import OLD_EPISODES_DIGEST, episodes_digest
from test_g1_getup_track_starts import OLD_STARTS_DIGEST, starts_digest

BANK_MD5 = '1' * 32
HANDOVER = {'share': .4, 'file': 'test-handover', 'md5': BANK_MD5}
FRAME = 5


def handover_bank(rows=6, frame=FRAME, clip_md5=MD5):
    """A hand-over bank made the way getup_handover_bank.py makes one, without its policy: the synthetic clip's task
    from drop_bank's back row under random actions, a row after every few steps (the state then, the last three applied
    actions newest first), the header naming the test clip and `frame`."""
    env = track_env()
    rng = np.random.default_rng(83)
    out = {key: [] for key in ('qpos', 'qvel', 'actions')}
    try:
        env.reset(seed=0, options=dict(OPTIONS, start=0))
        applied = [np.zeros(29)] * g1_getup_handover.HANDOVER_ACTIONS
        for step in range(rows * 3):
            action = env.applied_action(rng.uniform(-2., 2., 29))
            env.step(action)
            applied = [action, *applied[:-1]]
            if step % 3 == 2:
                out['qpos'].append(env.data.qpos.copy())
                out['qvel'].append(env.data.qvel.copy())
                out['actions'].append(np.array(applied))
    finally:
        env.close()
    n = len(out['qpos'])
    bank = {key: np.array(value) for key, value in out.items()}
    bank.update(row=np.arange(n), start_class=np.zeros(n, dtype=np.int64), physics=np.zeros(n, dtype=np.int64),
                seed=np.arange(n), seconds=np.full(n, .5), rule=np.zeros(n, dtype=np.int64), angle=np.zeros(n),
                deviation=np.zeros(n), joint_speed=np.zeros(n), nearest=np.zeros(n, dtype=np.int64))
    bank['header'] = dict(clip={'file': SETTING['file'], 'md5': clip_md5}, frame=frame, classes=['back'],
                          physics=['ours'], rules=['least'])
    return bank


def handover_env(bank=None, share=HANDOVER['share'], **settings):
    return track_env(track_handover_starts={**HANDOVER, 'share': share},
                     handover_bank=handover_bank() if bank is None else bank, **settings)


def test_the_setting_is_validated_labelled_and_not_part_of_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['track_handover_starts'] is None  # old stages: off
    tracking = {'track_clip': SETTING}
    on = validate_config({**base, 'task_config': {**tracking, 'track_handover_starts': {
        'share': 1, 'file': 'handover-1164902400-93a1a449-f18', 'md5': BANK_MD5}}})['task_config']
    assert on['track_handover_starts'] == {'share': 1., 'file': 'handover-1164902400-93a1a449-f18', 'md5': BANK_MD5}
    assert type(on['track_handover_starts']['share']) is float
    for value in ({'share': 0., 'file': 'x', 'md5': BANK_MD5}, {'share': 1.5, 'file': 'x', 'md5': BANK_MD5},
                  {'share': float('nan'), 'file': 'x', 'md5': BANK_MD5}, {'share': True, 'file': 'x', 'md5': BANK_MD5},
                  {'share': .3, 'file': '../x', 'md5': BANK_MD5}, {'share': .3, 'file': 'x.npz', 'md5': BANK_MD5},
                  {'share': .3, 'file': 'x', 'md5': 'abc'}, {'share': .3, 'file': 'x'},
                  {'share': .3, 'file': 'x', 'md5': BANK_MD5, 'frame': 18}, [.3, 'x', BANK_MD5], 'x', .3):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {**tracking, 'track_handover_starts': value}})
    with pytest.raises(ValueError, match='needs track_clip'):
        validate_config({**base, 'task_config': {'track_handover_starts': HANDOVER}})
    with pytest.raises(ValueError, match='needs track_clip'):  # one clip, the bank's: not with track_clips
        validate_config({**base, 'task_config': {'track_clips': [SETTING, {'file': 'other', 'md5': '2' * 32}],
                                                 'track_handover_starts': HANDOVER}})
    assert task_identity('g1_29dof', 'g1_getup', {**tracking, 'track_handover_starts': HANDOVER}) == task_identity(
        'g1_29dof', 'g1_getup', tracking)
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['track_handover_starts']
    assert field['group'] is None and field['default'] is None and field['type'] == 'json'
    assert 'передачи' in field['label']
    assert task.TRACK_START_KINDS == ('clip', 'frame', 'bank', 'handover')
    assert task.TRACK_HANDOVER_GRACE_SECONDS == 1.


def test_off_the_task_is_bit_for_bit_the_old_one_and_reads_no_bank(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('a hand-over bank read with the setting off')
    monkeypatch.setattr(g1_getup_handover, 'load', refuse)
    assert getup_digest({'track_handover_starts': None}, drop_bank()) == OLD_DIGESTS['plain']
    assert starts_digest(track_handover_starts=None) == OLD_STARTS_DIGEST
    assert episodes_digest(track_handover_starts=None) == OLD_EPISODES_DIGEST
    env = track_env()
    try:
        assert env.track_handover is None and env._handover is None
        _, info = env.reset(seed=0)
        assert 'track_handover' not in info and env.handover_row is None
    finally:
        env.close()


def test_a_start_from_the_bank_is_the_recorded_state():
    """The reset option track_handover k: qpos (its quaternion normalized) and qvel of row k, the actions in flight
    under the delay its newest ones (one at 10 substeps, three at 25), no noise and no lift, the clock and frame at the
    bank's frame, the reference's pelvis there put on the robot's (xy and heading), the last action 0 and the history
    filled with the first frame; the info names the row and its class; a probe's own state (a dict) is the same start."""
    bank = handover_bank()
    env = handover_env(bank)
    try:
        for row, delay, flying in ((2, 0, 0), (3, 10, 1), (4, 25, 3)):
            obs, info = env.reset(seed=row, options=dict(OPTIONS, action_delay_steps=delay, track_handover=row))
            assert info['track_start'] == 'handover' and info['start_class'] == 'handover' and info['start'] == -1
            assert info['track_handover'] == row and info['handover_class'] == 'back' and info['track_lift'] == 0.
            np.testing.assert_allclose(env.data.qpos, bank['qpos'][row], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(env.data.qvel, bank['qvel'][row])
            assert np.abs(bank['qvel'][row]).max() > .1  # in motion
            assert len(env.delay_actions) == flying
            for k in range(flying):
                np.testing.assert_array_equal(env.delay_actions[k], bank['actions'][row, k])
            assert env.clip_clock == env.clip_frame == FRAME and env.clip_start == 'handover'
            assert not env.previous_action.any()
            position, rotation, _, _ = env._clip_reference()
            np.testing.assert_allclose(position[0, 0, :2], env.data.xpos[env.pelvis, :2], atol=1e-9)
            turn = float(task.track_heading(rotation[:, 0])[0]
                         - task.track_heading(env.data.xmat[env.pelvis].reshape(1, 3, 3))[0])
            assert np.cos(turn) == pytest.approx(1., abs=1e-12)
            frames = obs[:env.actor_inputs]
            np.testing.assert_array_equal(frames, env._observation('fill')[:env.actor_inputs])
            same, _ = env.reset(seed=row, options=dict(OPTIONS, action_delay_steps=delay, track_handover=dict(
                qpos=bank['qpos'][row], qvel=bank['qvel'][row], actions=bank['actions'][row], frame=FRAME)))
            np.testing.assert_array_equal(same, obs)
            assert env.handover_row == -2
        for bad in (-1, 6, True, 1.):
            with pytest.raises(ValueError, match='hand-over bank'):
                env.reset(seed=0, options=dict(OPTIONS, track_handover=bad))
        with pytest.raises(ValueError, match='does not go with'):
            env.reset(seed=0, options=dict(OPTIONS, track_handover=0, start=0))
    finally:
        env.close()


def test_the_training_share_and_the_other_starts_within_the_rest():
    """share 0.4 over 4000 training resets (within 0.03, ~4σ): hand-over starts, every row drawn, the clip's starts
    0.6 / 0.3 / 0.1 of the rest; evaluation resets (assist force set) never start there; share 1 always does."""
    env = handover_env()
    try:
        env.reset(seed=0)
        counts, rows = dict.fromkeys(task.TRACK_START_KINDS, 0), set()
        for _ in range(4000):
            _, info = env.reset()
            counts[info['track_start']] += 1
            if info['track_start'] == 'handover':
                rows.add(info['track_handover'])
                assert env.clip_clock == FRAME
        assert abs(counts['handover'] / 4000 - .4) < .03, counts
        for kind, share in zip(task.TRACK_START_KINDS, task.TRACK_START_SHARES):
            assert abs(counts[kind] / 4000 - .6 * share) < .03, counts
        assert rows == set(range(6))
        assert {env.reset(seed=k, options=dict(assist_force=0.))[1]['track_start'] for k in range(40)} == {'clip'}
    finally:
        env.close()
    env = handover_env(share=1.)
    try:
        assert {env.reset(seed=k)[1]['track_start'] for k in range(20)} == {'handover'}
    finally:
        env.close()


def test_a_handover_start_waits_its_longer_grace():
    """A hand-over state in the clip's first pose (bank frame 0), the reference's left ankle 0.6 m away throughout,
    with track_end_reach: no end while the hand-over grace lasts (TRACK_HANDOVER_GRACE_SECONDS, 50 steps, twice a bank
    start's), the end on the step after it, by the reach."""
    clip = synthetic_clip()
    bank = handover_bank(frame=0)
    bank['qpos'] = bank['qpos'].copy()
    bank['qvel'] = np.zeros_like(bank['qvel'])
    bank['qpos'][0] = clip['qpos'][0]
    env = handover_env(bank, track_end_reach=.4)
    try:
        assert env.handover_grace == round(task.TRACK_HANDOVER_GRACE_SECONDS / env.dt) == 50 == 2 * env.clip_grace
        _, info = env.reset(seed=0, options=dict(OPTIONS, track_handover=0))
        assert info['track_start'] == 'handover' and env.clip_clock == 0
        _shift_ankle(env, .6)
        run = _follow(env, 70)
        assert [step for step, terminated, _ in run if terminated] == [env.handover_grace + 1]
        assert run[-1][2]['track_end_reach']
    finally:
        env.close()


def test_the_grace_key_is_validated():
    from humanoid_sim.lab import validate_config
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    tracking = {'track_clip': SETTING}
    plain = validate_config({**base, 'task_config': {**tracking, 'track_handover_starts': HANDOVER}})['task_config']
    assert 'grace' not in plain['track_handover_starts']  # missing stays missing: the old rule
    on = validate_config({**base, 'task_config': {**tracking, 'track_handover_starts': {**HANDOVER, 'grace': 2}}})
    assert on['task_config']['track_handover_starts']['grace'] == 2. and type(
        on['task_config']['track_handover_starts']['grace']) is float
    for grace in (-1, 5.5, float('nan'), True, '2', None):
        with pytest.raises(ValueError, match='grace'):
            validate_config({**base, 'task_config': {**tracking, 'track_handover_starts': {**HANDOVER, 'grace': grace}}})


def test_the_anchor_rules_wait_the_settings_grace():
    """The reference lifted 0.6 m above a robot lying in the clip's first pose (bank frame 0), so the anchor's height
    rule holds throughout: without grace (missing) the episode ends on the first step, as a bank start's; with grace
    2 s no rule ends it while the grace lasts (100 steps) and the anchor's rule ends it on the step after; the anchor
    gate of task.track_terminated by hand (None: the old rule)."""
    clip = synthetic_clip()
    bank = handover_bank(frame=0)
    bank['qpos'], bank['qvel'] = bank['qpos'].copy(), np.zeros_like(bank['qvel'])
    bank['qpos'][0] = clip['qpos'][0]
    for grace, ends_at in ((None, 1), (2., 101)):
        setting = {**HANDOVER, **({} if grace is None else {'grace': grace})}
        env = track_env(track_handover_starts=setting, handover_bank=bank)
        try:
            assert env.handover_body is (grace is not None)
            assert env.handover_grace == (50 if grace is None else 100)
            env.reset(seed=0, options=dict(OPTIONS, track_handover=0))
            pos = env.clip_frames['pos'].copy()
            pos[..., 2] += .6
            env.clip_frames = {**env.clip_frames, 'pos': pos}
            run = _follow(env, 120)
            assert [step for step, terminated, _ in run if terminated] == [ends_at], grace
            assert run[-1][2]['track_failed']
        finally:
            env.close()
    n, bodies, anchor = 2, 14, 7
    ends = tuple(task.TRACK_BODIES.index(name) for name in task.TRACK_END_BODIES)
    rotation = np.repeat(np.eye(3)[None, None], bodies, 1).repeat(n, 0)
    low, high = np.zeros((n, bodies, 3)), np.zeros((n, bodies, 3))
    high[:, anchor, 2] = .3
    assert task.track_terminated(low, rotation, high, rotation, anchor, ends).tolist() == [True, True]
    assert task.track_terminated(low, rotation, high, rotation, anchor, ends, None, np.array([False, True])).tolist() \
        == [False, True]


def test_the_banks_clip_must_be_the_tasks():
    env = handover_env(handover_bank(clip_md5='3' * 32))
    try:
        with pytest.raises(ValueError, match='hand-over bank of clip'):
            env.reset(seed=0, options=dict(OPTIONS, track_handover=0))
    finally:
        env.close()
    env = handover_env(handover_bank(frame=10_000))
    try:
        with pytest.raises(ValueError, match='not a frame of the clip'):
            env.reset(seed=0, options=dict(OPTIONS, track_handover=0))
    finally:
        env.close()


def test_the_file_round_trip(tmp_path):
    """save writes arrays and a JSON header, no pickle; load returns them read-only with the file's md5, checks the
    md5 the setting names, the version and the shapes, and names the script that makes a missing file."""
    bank = handover_bank(rows=3)
    header = bank.pop('header')
    path = tmp_path / 'handover-test.npz'
    md5 = g1_getup_handover.save(path, bank, header)
    loaded = g1_getup_handover.load('ignored', md5, path=path)
    assert loaded['md5'] == md5 and loaded['header']['frame'] == FRAME
    assert loaded['header']['version'] == g1_getup_handover.HANDOVER_VERSION
    for key in g1_getup_handover.ARRAYS:
        np.testing.assert_array_equal(loaded[key], bank[key])
        assert not loaded[key].flags.writeable
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {*g1_getup_handover.ARRAYS, 'header'}
    with pytest.raises(ValueError, match='md5'):
        g1_getup_handover.load('ignored', '4' * 32, path=path)
    with pytest.raises(FileNotFoundError, match='getup_handover_bank.py'):
        g1_getup_handover.load('ignored', path=tmp_path / 'missing.npz')
    broken = dict(bank, actions=bank['actions'][:, :2])
    with pytest.raises(ValueError, match='bad arrays'):
        g1_getup_handover.save(tmp_path / 'broken.npz', broken, header)
    assert g1_getup_handover.bank_path('handover-x').name == 'handover-x.npz'


# getup_handover_bank.py --pd (iteration 86): the bank's state at the end of the PD transition after the hand-over.
# Needs the lab's stages, start bank and clip under runs/ (untracked): skipped without them.
ROOT = Path(__file__).resolve().parents[1]
FIRST_STAGE = 'runs/humanoid_lab/exp-207eb498dc824027/stage_001164902400'
TRACKER_STAGE = 'runs/humanoid_lab/exp-84b0540b0092467d/stage_000570359808'  # getup_chain.SECOND
A493 = 'bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3'
PD = (1., (('arms', .3), ('kd', 3.), ('kp', .3), ('legs', .7), ('ramp', 1.)))  # the it86 shape, 1 s instead of 3.5
# The HEAD script's small bank (classes front,back, --runs 16, seed 83000, 1164 → A493 v3 frame 18), before --pd: jobs
# 0 and 9, md5 of qpos, qvel and actions (float64 bytes) and the hand-over time.
OLD_ROWS = {0: (.1, '227023e85b18652feccbb965cb5a9b82'), 9: (.66, '2b6ba5728af7eede36cdc1e3010b2be0')}


@pytest.fixture(scope='module')
def bank_script():
    for path in (ROOT / 'scripts' / 'lab', ROOT / 'scripts' / 'lab' / 'probes'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not ((ROOT / FIRST_STAGE).is_dir() and (ROOT / TRACKER_STAGE).is_dir()):
        pytest.skip('the chain stages are not on this machine')
    from humanoid_sim.tasks import g1_getup_bank, g1_getup_track
    try:
        g1_getup_bank.load()
        g1_getup_track.load(A493)
    except (OSError, ValueError, SystemExit):
        pytest.skip('the start bank or the A493 v3 clip is not on this machine')
    return importlib.import_module('getup_handover_bank')


@pytest.fixture
def original_step():
    """getup_probe's monitor (getup_chain.run) replaces mujoco.mj_step for the process; put it back."""
    import mujoco
    step = mujoco.mj_step
    yield
    mujoco.mj_step = step


def _jobs(module, pd=None):
    jobs, _ = module.jobs_of(str(ROOT / FIRST_STAGE), A493, 18, ['front', 'back'], list(module.PHYSICS), 16,
                             module.getup_chain.FIRST_SECONDS, module.SEED, pd=pd)
    return jobs


def _digest(record):
    digest = hashlib.md5()
    for key in ('qpos', 'qvel', 'actions'):
        digest.update(np.ascontiguousarray(record[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def test_without_pd_the_bank_rows_are_the_old_ones(bank_script):
    """Without --pd a run records the hand-over state the script recorded before (the HEAD build's rows, bit for
    bit), with no PD fields; save writes ARRAYS alone."""
    jobs = _jobs(bank_script)
    for k, (seconds, digest) in OLD_ROWS.items():
        record = bank_script.run(jobs[k])
        assert record['fired'] and record['seconds'] == pytest.approx(seconds, abs=1e-9)
        assert _digest(record) == digest, k
        assert not any(key.startswith('pd_') for key in record)


def test_with_pd_the_end_state_is_the_chains(bank_script, original_step):
    """--pd: the bank's state at the transition's end (qpos, qvel, the last applied actions newest first) is
    getup_chain.run's for the same row, seed and physics with the same transition, bit for bit; the actions in flight
    give back the transition's own targets; the PD fields describe that state."""
    chain = bank_script.getup_chain
    jobs = _jobs(bank_script, PD)
    for k in (0, 9):  # a front row in ours, a back row in another physics
        job = jobs[k]
        record = bank_script.run(job)
        assert record['fired'] and 'pd_ended' not in record
        theirs = chain.run(dict(first=job['first'], second=str(ROOT / TRACKER_STAGE), handover='pd',
                                pd_seconds=PD[0], pd=dict(PD[1]), physics=job['physics'], start=job['start'],
                                row=job['row'], seed=job['seed'], first_seconds=job['first_seconds'],
                                second_seconds=.02, pd_state=True))
        assert theirs['handover_s'] == pytest.approx(record['seconds'], abs=1e-9)
        state = theirs['pd_state']
        assert state['fell'] is None
        np.testing.assert_array_equal(record['qpos'], state['qpos'])
        np.testing.assert_array_equal(record['qvel'], state['qvel'])
        np.testing.assert_array_equal(record['actions'], state['actions'])
        assert record['actions'].shape == (g1_getup_handover.HANDOVER_ACTIONS, 29)
        assert not record['actions'][:, [20, 21, 27, 28]].any()  # locked wrists
        for key in g1_getup_handover.PD_ARRAYS:
            assert np.isfinite(record[key]), key
        assert 0 <= record['pd_arm_worst'] < 3 and 0 < record['pd_torso'] < 1.


def test_the_human_range_by_hand_and_on_two_bank_rows():
    """human_rom by hand (each limit just inside and just outside, hip roll outward + left / − right, joints of other
    kinds not judged), then on the start bank: back row 0 (lying straight) within, back row 1 (hip roll 99° / −88°,
    yaw 84°) outside; jobs_of(human=True) draws only rows within."""
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
    module = importlib.import_module('getup_handover_bank')
    names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_knee_joint',
             'waist_pitch_joint', 'left_shoulder_pitch_joint'] + [f'other_{k}' for k in range(23)]

    def inside(**degrees):
        qpos = np.zeros((1, 36))
        for name, value in degrees.items():
            qpos[0, 7 + names.index(f'{name}_joint')] = np.radians(value)
        return bool(module.human_rom(qpos, names)[0])
    assert inside()
    assert inside(left_hip_pitch=-124.9) and not inside(left_hip_pitch=-125.1) and not inside(left_hip_pitch=30.1)
    assert inside(left_hip_roll=49.9) and not inside(left_hip_roll=50.1)  # left: + outward
    assert inside(left_hip_roll=-34.9) and not inside(left_hip_roll=-35.1)
    assert inside(right_hip_roll=-49.9) and not inside(right_hip_roll=-50.1)  # right: − outward
    assert inside(right_hip_roll=34.9) and not inside(right_hip_roll=35.1)
    assert inside(left_knee=144.9) and not inside(left_knee=-5.1) and not inside(waist_pitch=80.1)
    assert inside(left_shoulder_pitch=170.)  # not judged
    from humanoid_sim.tasks import g1_getup_bank
    try:
        bank = g1_getup_bank.load()
    except (OSError, ValueError):
        pytest.skip('the start bank is not on this machine')
    env = g1_getup_bank.getup_model_env()
    try:
        assert list(np.asarray(env.qadr)) == list(range(7, 36))  # qpos[7:] in joint_names' order
        within = module.human_rom(bank['qpos'][:2], g1_getup_bank.joint_names(env))
    finally:
        env.close()
    assert bank['start_class'][0] == bank['start_class'][1] == bank['classes'].index('back')
    assert within.tolist() == [True, False]
    if not (ROOT / FIRST_STAGE).is_dir():
        return
    rows = module.human_rom_rows(bank)
    jobs = _jobs(module)
    human, _ = module.jobs_of(str(ROOT / FIRST_STAGE), A493, 18, ['front', 'back'], list(module.PHYSICS), 16,
                              module.getup_chain.FIRST_SECONDS, module.SEED, human=True)
    assert all(rows[job['row']] for job in human) and len(human) == len(jobs) == 16


def test_a_pd_bank_row_starts_the_tracker(bank_script, tmp_path, monkeypatch):
    """A --pd bank (saved and loaded as the task reads it) starts tracker 570 through its reset option track_handover:
    the row's state, the transition's actions in flight under a 20 ms delay, the clock at the bank's frame; the
    tracker's steps from there are finite."""
    from humanoid_sim.tasks.catalog import make_task
    sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'probes'))
    import getup_probe
    job = _jobs(bank_script, PD)[0]
    record = bank_script.run(job)
    arrays = {key: np.array([record[key]]) for key in ('qpos', 'qvel', 'actions')}
    arrays.update(row=np.array([job['row']]), start_class=np.zeros(1, dtype=np.int64),
                  physics=np.zeros(1, dtype=np.int64), seed=np.array([job['seed']]),
                  seconds=np.array([record['seconds']]), rule=np.zeros(1, dtype=np.int64),
                  angle=np.array([record['angle']]), deviation=np.array([record['deviation']]),
                  joint_speed=np.array([record['joint_speed']]), nearest=np.array([record['nearest']]))
    arrays.update({key: np.array([record[key]]) for key in g1_getup_handover.PD_ARRAYS})
    goal = bank_script._target(A493, 18)
    header = dict(clip={'file': A493, 'md5': goal['md5']}, frame=18, classes=['back', 'front'], physics=['ours'],
                  rules=list(bank_script.RULES), pd=dict(seconds=PD[0], shape=dict(PD[1])))
    path = tmp_path / 'handover-pd-test.npz'
    md5 = g1_getup_handover.save(path, arrays, header)
    loaded = g1_getup_handover.load('ignored', md5, path=path)
    assert set(g1_getup_handover.PD_ARRAYS) <= set(loaded) and loaded['header']['pd']['seconds'] == PD[0]
    monkeypatch.setattr(g1_getup_handover, 'load', lambda name, md5=None, path=None: loaded)
    policy, norm, config = getup_probe._policy(str(ROOT / TRACKER_STAGE))
    env = make_task('g1_getup', 'g1_29dof', {**config, 'track_handover_starts': {
        'share': .3, 'file': 'handover-pd-test', 'md5': md5}})
    raw = env.unwrapped
    try:
        obs, info = env.reset(seed=0, options=dict(assist_force=0., action_delay_steps=10, track_handover=0))
        assert info['track_start'] == 'handover' and info['track_handover'] == 0
        np.testing.assert_allclose(raw.data.qpos, record['qpos'], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw.data.qvel, record['qvel'])
        assert len(raw.delay_actions) == 1
        np.testing.assert_array_equal(raw.delay_actions[0], record['actions'][0])
        assert raw.clip_clock == raw.clip_frame == 18 and not raw.previous_action.any()
        for _ in range(10):
            obs, _, _, _, info = env.step(policy.predict(norm.normalize_obs(obs), deterministic=True)[0])
            assert np.isfinite(obs).all() and not info.get('bad_state')
    finally:
        env.close()
