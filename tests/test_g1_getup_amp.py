"""g1_getup reward weight amp (iteration 55), CPU: the AMP discriminator of humanoid_sim/getup_amp on pairs of
neighbouring states (getup_amp.amp_state), paid by the full-GPU learner (TensorPPO), not by the task. Off (weight 0) the
task and the learner are bit for bit the old ones; the discriminator separates, its gradient penalty is the one
specified, amp.pt goes with every stage and a resumed run restores it. The CUDA task: tests/test_g1_gpu.py."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim import getup_amp
from humanoid_sim.tasks import g1_getup_style
from humanoid_sim.tasks import g1_tensor_math as task
from test_g1_getup_arm_press import BRANCH_51
from test_g1_getup_impact import OLD_DIGESTS, drop_bank, getup_digest
from test_g1_getup_stance_width import BEFORE_46

FREE = tuple(j for j in range(29) if j not in (20, 21, 27, 28))  # all but the locked wrists
WIDTH = 2 * getup_amp.state_size(len(FREE))


def write_amp_library(folder, name, human, heldout, joints=FREE, version=getup_amp.AMP_VERSION, **extra):
    """An AMP library file as getup_amp_library.py writes it."""
    arrays = dict(human=np.asarray(human, dtype=np.float64), heldout=np.asarray(heldout, dtype=np.float64),
                  joints=np.array(joints, dtype=np.int64), header=np.array(json.dumps(dict(version=version))), **extra)
    np.savez(folder / f'{name}-amp.npz', **arrays)


def synthetic(rng, n, centre):
    return centre + rng.normal(0., .3, (n, WIDTH))


@pytest.fixture
def library(tmp_path, monkeypatch):
    """An AMP library 'test' in a temporary getup_style folder: human pairs around +0.5, held-out ones likewise."""
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    rng = np.random.default_rng(0)
    write_amp_library(tmp_path, 'test', synthetic(rng, 2000, .5), synthetic(rng, 300, .5))
    return getup_amp.load_library('test')


def test_the_weight_and_the_judge_are_validated_labelled_and_leave_the_contract():
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tasks.config import G1_GETUP_PARAMETERS, G1_GETUP_REWARD_WEIGHTS, G1_REWARD_WEIGHTS
    from humanoid_sim.tasks.identity import task_identity
    assert G1_GETUP_REWARD_WEIGHTS['amp'] == 0 and 'amp' not in G1_REWARD_WEIGHTS
    base = dict(robot_id='g1_29dof', task_id='g1_getup')
    plain = validate_config(base)['task_config']
    assert plain['reward_weights']['amp'] == 0 and plain['style_judge'] is False  # old stages: off
    weighted = validate_config({**base, 'task_config': {'reward_weights': {'amp': 1., 'style': 0.},
                                                        'style_judge': True}})['task_config']
    assert weighted['reward_weights']['amp'] == 1. and weighted['style_judge'] is True
    for value in (-.1, 101., float('nan'), '1', True):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'amp': value}}})
    for value in (1, 0, 'true', None):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'style_judge': value}})
    for config in ({}, BRANCH_51):  # the contract (and so branching) does not change
        changed = {**config, 'style_judge': True, 'reward_weights': {**config.get('reward_weights', {}), 'amp': 1.}}
        assert task_identity('g1_29dof', 'g1_getup', changed) == task_identity('g1_29dof', 'g1_getup', config)
    fields = {p['key']: p for p in G1_GETUP_PARAMETERS}
    assert fields['amp']['group'] == 'reward_weights' and fields['amp']['default'] == 0.
    assert 'AMP' in fields['amp']['label'] and '0,3–1,1 м' in fields['amp']['label']
    assert fields['style_judge']['group'] is None and fields['style_judge']['type'] == 'bool'
    assert fields['style_judge']['default'] is False and 'судья' in fields['style_judge']['label']


def test_the_state_layout_numpy_and_torch_alike():
    """amp_state: the free joints, Rᵀ·(0, 0, −1), pelvis and head heights, the points relative to the pelvis turned by
    the heading yaw (stance_width's rule, the y axis where the x axis is near vertical); amp_points: xpos + R·offset."""
    rng = np.random.default_rng(1)
    n = 16
    position = rng.normal(0, 1, (n, 29))
    rotation = np.linalg.qr(rng.normal(0, 1, (n, 3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))[:, None, None]
    rotation[0] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]  # x axis straight up: the y axis gives the heading
    pelvis, head = rng.normal(0, 1, (n, 3)), rng.uniform(.1, 1.3, n)
    points = rng.normal(0, 1, (n, len(getup_amp.AMP_POINTS), 3))
    state = getup_amp.amp_state(position, pelvis, rotation, head, points, FREE)
    assert state.shape == (n, 54) and getup_amp.state_size(len(FREE)) == 54
    for k in range(n):
        forward = rotation[k][:2, 0]
        if np.hypot(*forward) < task.GETUP_STANCE_HEADING:
            forward = np.array([rotation[k][1, 1], -rotation[k][0, 1]])
        yaw = np.arctan2(forward[1], forward[0])
        turn = np.array([[np.cos(yaw), np.sin(yaw), 0.], [-np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        local = (turn @ (points[k] - pelvis[k]).T).T.ravel()
        expected = np.concatenate([position[k, list(FREE)], rotation[k].T @ [0., 0., -1.], [pelvis[k, 2], head[k]],
                                   local])
        np.testing.assert_allclose(state[k], expected, rtol=0, atol=1e-12)
    assert getup_amp.heading_yaw(rotation[:1])[0] == pytest.approx(0.)
    torched = getup_amp.amp_state(*(torch.as_tensor(v) for v in (position, pelvis, rotation, head, points)), FREE)
    np.testing.assert_allclose(torched.numpy(), state, rtol=0, atol=1e-12)
    xpos, xmat = rng.normal(0, 1, (n, 40, 3)), np.linalg.qr(rng.normal(0, 1, (n, 40, 3, 3)))[0].reshape(n, 40, 9)
    bodies, offsets = np.array([3, 7]), np.array([[.11, 0., 0.], [0., 0., 0.]])
    got = getup_amp.amp_points(xpos, xmat, bodies, offsets)
    np.testing.assert_allclose(got[:, 0], xpos[:, 3] + xmat[:, 3].reshape(n, 3, 3) @ offsets[0], atol=1e-12)
    np.testing.assert_array_equal(got[:, 1], xpos[:, 7])
    np.testing.assert_allclose(getup_amp.amp_points(torch.as_tensor(xpos), torch.as_tensor(xmat),
                                                    torch.as_tensor(bodies), offsets).numpy(), got, atol=1e-12)


def test_the_points_of_the_getting_up_model():
    from humanoid_sim.tasks import g1_getup_bank
    env = g1_getup_bank.getup_model_env()
    try:
        bodies, offsets = getup_amp.point_bodies(env.model)
        assert [env.model.body(int(b)).name for b in bodies] == [name for name, _ in getup_amp.AMP_POINTS]
        np.testing.assert_array_equal(offsets[:2], [task.HAND_LOAD_POSITION] * 2)
        assert not offsets[2:].any()
    finally:
        env.close()


def test_the_library_is_read_without_pickle_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(g1_getup_style, 'STYLE_DIR', tmp_path)
    assert getup_amp.library_path('lafan1') == tmp_path / 'lafan1-amp.npz'
    with pytest.raises(ValueError):
        getup_amp.library_path('../x')
    with pytest.raises(FileNotFoundError, match='getup_amp_library.py'):
        getup_amp.load_library('absent')
    rng = np.random.default_rng(2)
    human, heldout = synthetic(rng, 50, 0.), synthetic(rng, 10, 0.)
    write_amp_library(tmp_path, 'good', human, heldout)
    library = getup_amp.load_library('good')
    np.testing.assert_array_equal(library['human'], human.astype(np.float32))
    assert library['human'].dtype == np.float32 and library['joints'] == FREE and len(library['md5']) == 32
    write_amp_library(tmp_path, 'other', human + 1e-3, heldout)
    assert getup_amp.load_library('other')['md5'] != library['md5']
    write_amp_library(tmp_path, 'pickled', human, heldout, extra=np.array([{'a': 1}], dtype=object))
    with pytest.raises(ValueError):
        getup_amp.load_library('pickled')
    write_amp_library(tmp_path, 'old', human, heldout, version='getup-amp-v0')
    with pytest.raises(ValueError, match='version'):
        getup_amp.load_library('old')
    write_amp_library(tmp_path, 'narrow', human[:, :100], heldout[:, :100])
    with pytest.raises(ValueError, match='pair'):
        getup_amp.load_library('narrow')
    with pytest.raises(ValueError, match='joints'):
        getup_amp.GetupAMP(1., library, 'cpu', FREE[:-1])


@pytest.mark.skipif(not getup_amp.library_path('lafan1').is_file(),
                    reason='the lab library runs/humanoid_lab/getup_style/lafan1-amp.npz is made locally')
def test_the_lab_library():
    """getup_amp_library.py's pairs: subjects 1-4 as many as the style library's frames (the same band rule on the
    second frame), subject 5 held out, the home-like standing of the second frames' heads in the band."""
    library = getup_amp.load_library('lafan1')
    assert library['joints'] == FREE and library['human'].shape[1] == 108
    assert 10_000 < len(library['human']) < 25_000 and library['header']['holdout_subject'] == 5
    head = library['human'][:, 54 + 29]
    assert ((head > .3) & (head < 1.1)).all()


def fill(amp, pairs, band=None):
    """One rollout step of `pairs` through the reward path, its in-band pairs into the FIFO."""
    band = torch.ones(len(pairs), dtype=torch.bool) if band is None else band
    amp.begin_rollout(1, len(pairs))
    reward = amp.reward(0, torch.as_tensor(pairs, dtype=torch.float32), band)
    return reward, amp.end_rollout()


def test_the_discriminator_separates_and_its_gradient_penalty_is_the_specified_one(library):
    """Human pairs around +0.5, policy pairs around −0.5: after D's steps D(human) > 0.5, D(policy) < −0.5 and
    D(held out) > 0.5; the logged penalty is mean ‖∇ D(normalized human pair)‖², checked by central differences on a copy
    of D; the reward is w·[band]·max(0, 1 − 0.25·(D − 1)²), 0 while D warms up and outside the band (NaN pairs too)."""
    torch.manual_seed(0)
    amp = getup_amp.GetupAMP(2., library, 'cpu', FREE, buffer=5000, warmup=1000)
    rng = np.random.default_rng(3)
    policy = synthetic(rng, 4000, -.5)
    reward, metrics = fill(amp, policy)
    assert amp.filled == 4000 and not reward.any() and amp.weight_now == 0.  # warming up
    # The penalty: the same draws on a float64 copy, the gradient by central differences.
    before = copy.deepcopy(amp.disc).double()
    torch.manual_seed(7)
    amp.train_step(64)
    torch.manual_seed(7)
    torch.randint(amp.filled, (64,))
    human = amp.normalize(amp.human[torch.randint(len(amp.human), (64,))]).double()
    step, gradient = 1e-6, torch.zeros_like(human)
    with torch.no_grad():
        for i in range(WIDTH):
            shift = torch.zeros(WIDTH, dtype=torch.float64)
            shift[i] = step
            gradient[:, i] = (before(human + shift) - before(human - shift)).squeeze(-1) / (2 * step)
    logged = amp._stats[-1].tolist()
    assert logged[1] == pytest.approx(float(gradient.square().sum(-1).mean()), rel=1e-4)
    with torch.no_grad():
        d_human = before(human).squeeze(-1)
    assert logged[2] == pytest.approx(float(d_human.mean()), rel=1e-4, abs=1e-6)
    for _ in range(300):
        amp.train_step(256)
    update = amp.end_update(1200)
    assert set(update) == {'train/disc_loss', 'train/disc_gp', 'train/disc_human', 'train/disc_policy',
                           'train/disc_heldout'}
    assert update['train/disc_human'] > .5 and update['train/disc_policy'] < -.5 and update['train/disc_heldout'] > .5
    with torch.no_grad():
        assert float(amp.score(amp.human).mean()) > .5
        assert float(amp.score(torch.as_tensor(policy, dtype=torch.float32)).mean()) < -.5
    assert amp.trained_steps == 1200 and amp.weight_now == 2.
    # The reward after the warm-up, by hand; outside the band and for a non-finite pair 0.
    pairs = torch.as_tensor(np.concatenate([synthetic(rng, 6, .5), synthetic(rng, 6, -.5)]), dtype=torch.float32)
    pairs[11, 3] = float('nan')
    band = torch.tensor([True, True, True, False, True, True, True, True, True, False, True, False])
    reward, metrics = fill(amp, pairs, band)
    with torch.no_grad():
        d = amp.score(torch.nan_to_num(pairs))
    expected = 2. * torch.where(band, (1 - .25 * (d - 1) ** 2).clamp_min(0), 0.)
    torch.testing.assert_close(reward, expected)
    assert torch.isfinite(reward).all() and reward[[3, 9, 11]].eq(0).all()
    assert reward[:3].mean() > reward[6:9].mean()  # human-like pairs earn more
    assert metrics['rollout/amp_reward'] == pytest.approx(float(expected.sum() / 2. / band.sum()), rel=1e-5)


def test_the_policy_buffer_is_a_fifo(library):
    amp = getup_amp.GetupAMP(1., library, 'cpu', FREE, buffer=10)
    first, second = (torch.full((7, WIDTH), float(k)) for k in (1, 2))
    fill(amp, first)
    assert amp.filled == 7 and amp.cursor == 7
    fill(amp, second, torch.tensor([True] * 6 + [False]))  # the out-of-band pair stays out
    assert amp.filled == 10 and amp.cursor == 3
    assert sorted(amp.buffer[:, 0].tolist()) == [1.] * 4 + [2.] * 6
    fill(amp, torch.arange(25.)[:, None].expand(25, WIDTH).contiguous())
    assert amp.filled == 10 and amp.buffer[:, 0].tolist() == list(range(15, 25))
    amp.begin_rollout(1, 3)
    amp.reward(0, torch.zeros(3, WIDTH), torch.zeros(3, dtype=torch.bool))
    assert np.isnan(amp.end_rollout()['rollout/amp_reward'])
    assert np.isnan(amp.end_update(100)['train/disc_loss']) and amp.trained_steps == 0  # no D step, no steps


def test_amp_pt_is_saved_and_restored_and_another_library_starts_fresh(library, tmp_path, monkeypatch):
    torch.manual_seed(1)
    amp = getup_amp.GetupAMP(1., library, 'cpu', FREE, buffer=500, warmup=10)
    fill(amp, synthetic(np.random.default_rng(4), 400, -.5))
    for _ in range(5):
        amp.train_step(64)
    amp.end_update(24)
    stage = tmp_path / 'stage'
    stage.mkdir()
    amp.save(stage)
    saved = torch.load(stage / getup_amp.AMP_FILE, weights_only=True)
    assert saved['version'] == getup_amp.AMP_VERSION and saved['library_md5'] == library['md5']
    assert saved['trained_steps'] == 24
    torch.manual_seed(99)  # another initialisation, overwritten by the restore
    restored = getup_amp.GetupAMP(1., library, 'cpu', FREE, stage=stage, buffer=500, warmup=10)
    assert restored.restored and restored.trained_steps == 24 and restored.weight_now == 1.
    assert restored.filled == 0  # the buffer is not kept
    probe = torch.as_tensor(synthetic(np.random.default_rng(5), 20, 0.), dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(restored.score(probe), amp.score(probe), rtol=0, atol=0)
    expected, actual = amp.optimizer.state_dict(), restored.optimizer.state_dict()
    assert actual['param_groups'] == expected['param_groups']
    for key, state in expected['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(actual['state'][key][name], value)
    rng = np.random.default_rng(6)
    write_amp_library(g1_getup_style.STYLE_DIR, 'changed', synthetic(rng, 100, .5), synthetic(rng, 10, .5))
    fresh = getup_amp.GetupAMP(1., getup_amp.load_library('changed'), 'cpu', FREE, stage=stage, buffer=500)
    assert not fresh.restored and fresh.trained_steps == 0 and fresh.weight_now == 0. and 'warm-up' in fresh.status
    monkeypatch.setattr(getup_amp, 'AMP_VERSION', 'getup-amp-v2')
    other = getup_amp.GetupAMP(1., library, 'cpu', FREE, stage=stage, buffer=500)
    assert not other.restored and other.trained_steps == 0


def test_off_the_task_is_bit_for_bit_the_old_one():
    """The amp weight and style_judge change nothing in the CPU task (the learner pays amp; the judge is CUDA only):
    training episodes give the digests of the code before them."""
    bank = drop_bank()
    for config in ({'reward_weights': {'amp': 1.}}, {'style_judge': True}, {'reward_weights': {'amp': 0.}}):
        assert getup_digest(config, bank) == OLD_DIGESTS['plain'], config
    weights = {**BRANCH_51['reward_weights'], 'amp': 1.}
    assert getup_digest({**BRANCH_51, 'reward_weights': weights, 'style_judge': True}, bank) == BEFORE_46[0]


class FakeAMPEnv:
    """The CUDA task's interface on the CPU with the amp weight: zero observations, an episode end every 3 steps, the
    assist forces as tests/test_g1_getup_full_gpu.FakeTaskEnv, and each step's state pairs (around −0.5) with every
    other world in the band."""

    def __init__(self, n_envs, reference, weight):
        from test_g1_getup_full_gpu import FakeTaskEnv
        self.base = FakeTaskEnv(n_envs, reference, True)
        self.amp_weight, self.amp_library, self.amp_joints = weight, 'test', FREE
        self.rng = torch.Generator().manual_seed(5)
        self.amp_pair = self.amp_band = None
        for name in ('device', 'n_envs', 'observation_space', 'action_space', 'fixed_normalization',
                     'last_episode_lengths', 'last_episode_returns', 'training_state', 'restore_training_state',
                     'reset', 'check_health', 'close'):
            setattr(self, name, getattr(self.base, name))

    def step(self, actions):
        result = self.base.step(actions)
        self.amp_pair = -.5 + .3 * torch.randn((self.n_envs, WIDTH), generator=self.rng)
        self.amp_band = torch.arange(self.n_envs) % 2 == 0
        return result


@pytest.fixture
def amp_full_gpu(monkeypatch, library):
    """full_gpu_train with CPU stand-ins for CUDA (tests/test_g1_getup_full_gpu.py's) and FakeAMPEnv with the
    task_config's amp weight; returns the environments made."""
    from humanoid_sim import full_gpu_train
    from humanoid_sim.backends import registry
    from humanoid_sim.tasks.catalog import make_task
    from test_g1_getup_full_gpu import CPUPPO
    import humanoid_sim.train as cpu_train
    made = []

    class Fake:
        def make(self, robot_id, task_id, backend, n_envs, seed, **task_config):
            weight = task_config.get('reward_weights', {}).get('amp', 0.)
            env = FakeAMPEnv(n_envs, make_task(task_id, robot_id, task_config), weight)
            made.append(env)
            return env
    monkeypatch.setattr(registry, 'backends', lambda: Fake())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU test substitute')
    monkeypatch.setattr(full_gpu_train, 'PPO', CPUPPO)
    monkeypatch.setattr(cpu_train, 'evaluate_stage', lambda *_: {'mean_return': 0., 'mean_seconds': 0.,
                                                                 'mean_forward_m': None, 'episodes': []})
    yield made
    for env in made:
        env.close()


def amp_config(weight, **values):
    from humanoid_sim.lab import validate_config
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='full_gpu', envs=4, n_steps=8,
                                  batch_size=16, n_epochs=2, steps=1024, target_kl=None, task_config={
                                      'style_library': 'test', 'reward_weights': {'amp': weight}}, **values))
    return {**config, 'steps': 32}  # one rollout


def test_stages_keep_amp_pt_and_a_resumed_run_restores_the_discriminator(tmp_path, amp_full_gpu):
    """A full-GPU run with the amp weight: every stage it saves has amp.pt (the first one, before training, a fresh D),
    the log has the discriminator's numbers and rollout/amp_reward, and a resumed run restores D (its steps go on, no
    warm-up again); a branch's copy of the stage (lab.create) carries amp.pt. Without the weight no discriminator is
    built, no amp.pt written, nothing logged."""
    import csv
    from humanoid_sim import full_gpu_train
    from humanoid_sim.core import list_stages, read_json, write_json
    from humanoid_sim.lab import Laboratory
    run = tmp_path / 'run'
    run.mkdir()
    config = amp_config(1.)
    write_json(run / 'run_config.json', config)
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=run, parent_stop=None), config)
    stages = list_stages(run)
    assert read_json(run / 'status.json')['state'] == 'completed' and len(stages) == 2
    assert [(stage / 'amp.pt').is_file() for stage in stages] == [True, True]
    saved = torch.load(stages[-1] / 'amp.pt', weights_only=True)
    assert saved['trained_steps'] == 32 and saved['library_md5'] == getup_amp.load_library('test')['md5']
    (session,) = run.glob('session_*')
    assert read_json(session / 'config.json')['amp']['status'].startswith('fresh')
    with open(session / 'progress.csv', newline='') as file:
        row = next(csv.DictReader(file))
    for key in ('rollout/amp_reward', 'train/disc_loss', 'train/disc_gp', 'train/disc_human', 'train/disc_policy',
                'train/disc_heldout'):
        assert np.isfinite(float(row[key])), key
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=run, parent_stop=None), config)
    newest = list_stages(run)[-1]
    assert torch.load(newest / 'amp.pt', weights_only=True)['trained_steps'] == 64  # restored, then trained on
    sessions = sorted(run.glob('session_*'))
    assert read_json(sessions[-1] / 'config.json')['amp']['status'].startswith('discriminator restored')
    # A branch's copy of the stage carries amp.pt.
    lab = Laboratory(tmp_path / 'lab')
    source = lab.root / 'exp-0000000000000001'
    source.mkdir()
    import shutil
    shutil.copytree(newest, source / newest.name)
    write_json(source / 'run_config.json', config)
    write_json(source / 'experiment.json', {'id': source.name, 'name': 'источник', 'config': config,
                                            'read_only': False, 'created_at': 0})
    entry = lab.create({**config, 'steps': 1024, 'name': 'ветка', 'source_id': source.name,
                        'source_stage': int(newest.name[6:])})
    assert (lab.root / entry['id'] / newest.name / 'amp.pt').is_file()
    # Without the weight: no discriminator.
    plain = tmp_path / 'plain'
    plain.mkdir()
    config = amp_config(0.)
    write_json(plain / 'run_config.json', config)
    learner, export = full_gpu_train.make_learner(config)
    try:
        assert learner.amp is None
    finally:
        export.close()
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=plain, parent_stop=None), config)
    assert not any((stage / 'amp.pt').exists() for stage in list_stages(plain))
    with open(next(plain.glob('session_*')) / 'progress.csv', newline='') as file:
        assert not any('disc' in key or 'amp' in key for key in next(csv.reader(file)))


def test_off_the_learner_is_the_old_one(library):
    """TensorPPO without amp: the same weights, rewards and statistics as with an amp of weight 0 never warmed
    (D steps and its RNG draws aside, which a run without the weight never makes): the reward is untouched while
    warming up; without amp no state-pair attribute is read."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.lab import validate_config
    from humanoid_sim.tensor_ppo import TensorNormalize, TensorPPO
    from test_full_gpu_export import FakeTensorEnv

    class NoPairs(FakeTensorEnv):
        def __getattr__(self, name):
            if name.startswith('amp'):
                raise AssertionError(f'{name} read without amp')
            raise AttributeError(name)

    torch.set_num_threads(2)
    export = VecNormalize(DummyVecEnv([lambda: gym.make('Humanoid-v5')]))
    config = validate_config(dict(envs=2, n_steps=8, batch_size=8, n_epochs=2, steps=1024, target_kl=None,
                                  learning_rate=.001))
    rewards = []
    try:
        for amp in (None, getup_amp.GetupAMP(1., library, 'cpu', FREE, buffer=100, warmup=10 ** 9)):
            env = NoPairs(2) if amp is None else FakeTensorEnv(2)
            if amp is not None:
                env.amp_pair = torch.zeros((2, WIDTH))
                env.amp_band = torch.ones(2, dtype=torch.bool)
            model = PPO('MlpPolicy', export, device='cpu', seed=23, n_steps=8, batch_size=8,
                        policy_kwargs={'net_arch': [32, 32]})
            learner = TensorPPO(model, env, config, TensorNormalize(env.observation_space.shape, 2, 'cpu'), amp=amp)
            torch.manual_seed(3)
            learner.collect()
            rewards.append(learner.rewards.clone())
            env.close()
        torch.testing.assert_close(rewards[0], rewards[1], rtol=0, atol=0)
    finally:
        export.close()
