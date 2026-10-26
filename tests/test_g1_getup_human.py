"""g1_getup task_config human_starts (iteration 56, reference state init), CPU: a share of training resets starts from a
human frame getting up (tasks/g1_getup_human), whose episodes step neither the assist nor the bound; evaluation and
probes (a start, an assist force or a bound set) never do. Off (missing or null) the task is bit for bit the old one.
The CUDA task: tests/test_g1_gpu.py."""
import json

import numpy as np
import pytest

from humanoid_sim.tasks import g1_getup_human
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup_impact import OLD_DIGESTS, OPTIONS, drop_bank, getup_digest

MD5 = '0' * 32


def human_bank(n=6):
    """Human starts made from drop_bank's lying rows, shifted per row (the rows tell apart by qpos[0])."""
    bank = drop_bank()
    qpos = np.concatenate([bank['qpos'][:3]] * 2)[:n].copy()
    qpos[:, 0] = np.arange(n) + 10.
    qvel = np.zeros((n, 35))
    qvel[:, 6] = np.linspace(-1, 1, n)
    return dict(qpos=qpos, qvel=qvel)


def test_the_setting_is_validated_labelled_and_leaves_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS
    from humanoid_sim.tasks.identity import task_identity
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    assert validate_config(base)['task_config']['human_starts'] is None  # old stages: off
    on = validate_config({**base, 'task_config': {'human_starts': {'share': .3, 'md5': MD5}}})['task_config']
    assert on['human_starts'] == {'share': .3, 'md5': MD5}
    for value in ({'share': 0., 'md5': MD5}, {'share': 1.1, 'md5': MD5}, {'share': True, 'md5': MD5},
                  {'share': .3}, {'share': .3, 'md5': 'abc'}, {'share': .3, 'md5': MD5.upper().replace('0', 'A')},
                  {'share': .3, 'md5': MD5, 'x': 1}, [.3, MD5], .3, 'x'):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'human_starts': value}})
    assert task_identity('g1_29dof', 'g1_getup', {'human_starts': {'share': .3, 'md5': MD5}}) \
        == task_identity('g1_29dof', 'g1_getup', {})
    field = {p['key']: p for p in G1_GETUP_PARAMETERS}['human_starts']
    assert field['group'] is None and field['default'] is None and 'человека' in field['label']


def test_off_the_task_is_bit_for_bit_the_old_one(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('human starts read with the setting off')
    monkeypatch.setattr(g1_getup_human, 'load', refuse)
    bank = drop_bank()
    for config, name in (({'human_starts': None}, 'plain'), ({'pd_bound': [1., .6], 'human_starts': None}, 'pd_bound')):
        assert getup_digest(config, bank) == OLD_DIGESTS[name], config


def test_a_share_of_training_resets_starts_from_a_human_row_and_evaluation_never_does():
    """10 000 training resets at share 0.3: 0.3 of them from the human rows (each row drawn), the rest from the bank's
    training rows; a reset with a start, an assist force or a bound set (evaluation, probes) never; human_start gives
    the row exactly."""
    human = human_bank()
    env = G1GetupEnv(bank=drop_bank(), human_starts={'share': .3, 'md5': MD5}, human_bank=human)
    try:
        env.reset(seed=1)
        counts = np.zeros(len(human['qpos']), dtype=int)
        bank = 0
        for k in range(10_000):
            _, info = env.reset()
            if 'human_start' in info:
                assert info['start'] == -1 and info['start_class'] == 'human'
                row = info['human_start']
                assert env.data.qpos[0] == human['qpos'][row, 0] and env.data.qvel[6] == human['qvel'][row, 6]
                counts[row] += 1
            else:
                bank += 1
                assert env.data.qpos[0] < 5
        share = counts.sum() / 10_000
        assert abs(share - .3) < .015 and (counts > 300).all(), (share, counts)
        for options in (dict(start=0), dict(assist_force=0.), dict(pd_bound='min'),
                        dict(assist_force=0., pd_bound='min')):  # the last: task.evaluation_options
            for _ in range(200):
                _, info = env.reset(options=options)
                assert 'human_start' not in info, options
        _, info = env.reset(options=dict(OPTIONS, human_start=4))
        assert info['human_start'] == 4
        np.testing.assert_array_equal(env.data.qpos[7:], human['qpos'][4, 7:])
        np.testing.assert_array_equal(env.data.qvel, human['qvel'][4])
        for row in (-1, 6, True, 1.):
            with pytest.raises(ValueError):
                env.reset(options=dict(human_start=row))
    finally:
        env.close()


def test_a_human_episode_steps_neither_the_assist_nor_the_bound():
    human = human_bank()
    env = G1GetupEnv(bank=drop_bank(), human_starts={'share': 1., 'md5': MD5}, human_bank=human, pd_bound=[1., .6])
    try:
        env.reset(seed=2)
        assert env.human_start is not None
        before = env.assist, env.bound
        env.head_window.extend([1.3] * env.head_window.maxlen)  # up for the whole window: a bank episode would step
        env.reset()
        assert (env.assist, env.bound) == before
        env.reset(options=dict(start=0))  # the episode before was human: still no step
        assert (env.assist, env.bound) == before
        env.head_window.extend([1.3] * env.head_window.maxlen)
        env.reset(options=dict(start=0))  # after a bank episode the curricula step as before
        assert env.assist == before[0] - 20. and env.bound == pytest.approx(before[1] - .02)
    finally:
        env.close()


def test_the_file_is_read_without_pickle_and_its_md5_checked(tmp_path):
    path = tmp_path / 'human_starts.npz'
    with pytest.raises(FileNotFoundError, match='getup_human_starts.py'):
        g1_getup_human.load(MD5, path)
    human = human_bank()
    header = np.array(json.dumps(dict(version=g1_getup_human.HUMAN_STARTS_VERSION)))
    np.savez(path, qpos=human['qpos'], qvel=human['qvel'], subject=np.ones(6, np.int64), header=header)
    starts = g1_getup_human.load(None, path)
    assert len(starts['md5']) == 32 and not starts['qpos'].flags.writeable
    assert g1_getup_human.load(starts['md5'], path) is starts
    with pytest.raises(ValueError, match='md5'):
        g1_getup_human.load(MD5, path)
    np.savez(path, qpos=human['qpos'][:, :30], qvel=human['qvel'], header=header)
    with pytest.raises(ValueError, match='qpos'):
        g1_getup_human.load(None, path)
    np.savez(path, qpos=human['qpos'], qvel=human['qvel'], header=header, extra=np.array([{'a': 1}], dtype=object))
    with pytest.raises(ValueError):
        g1_getup_human.load(None, path)


@pytest.mark.skipif(not g1_getup_human.HUMAN_STARTS.is_file(),
                    reason='runs/humanoid_lab/getup_bank/human_starts.npz is made locally')
def test_the_lab_file():
    """getup_human_starts.py's rows: subjects 1-4 only, the head in the band before the lift, lifted out of the floor
    by at most 15 cm, a frame index of its clip each; the task starts from them."""
    starts = g1_getup_human.load(None)
    n = len(starts['qpos'])
    assert 5_000 < n < 16_000 and set(np.unique(starts['subject'])) <= {1, 2, 3, 4}
    assert ((starts['head'] > .3) & (starts['head'] < 1.1)).all() and (0 <= starts['lift']).all()
    assert starts['lift'].max() < .15 and len(starts['frame']) == n
    assert starts['header']['filter']['kept'] == n
    env = G1GetupEnv(human_starts={'share': .3, 'md5': starts['md5']})
    try:
        _, info = env.reset(seed=0, options=dict(OPTIONS, human_start=n // 2))
        head = float(env._head()[0])
        assert .25 < head < 1.3 and info['human_start'] == n // 2
    finally:
        env.close()
