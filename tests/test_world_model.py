"""The world model of SuperMimic stage 1 (humanoid_sim/world_model.py, scripts/lab/wm/collect.py and fit.py), CPU, small
sizes: the state layouts and their exact round trip, the kinematics against MuJoCo, the policy frame and the tracking
reward against the CPU task's own, the network's shapes, its loss falling on synthetic dynamics, save and load, the
collector on the CPU task (2 worlds × 20 steps, short episodes) and a fit of seconds on what it wrote; a clip that
rises out of the robot's reach, so the collected episodes end by the task's terminations and start from the bank too:
the data's step counts and start kinds against those terminations, held-out windows with terminations, the report's
termination numbers, checkpoints and resume; the collector's action modes and the refusal of rules TrackTerms lacks."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim import world_model as wm
from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_getup_track
from humanoid_sim.tasks import g1_tensor_math as task
from test_g1_getup_impact import DROP, drop_bank
from test_g1_getup_track import FRAMES, track_env

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f'wm_{name}', ROOT / 'scripts' / 'lab' / 'wm' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def env():
    return track_env()


def _random_raw(model, rng, count):
    """(count, RAW) random states of `model` (any orientation, joints within their ranges) and the MjData of each."""
    data = mujoco.MjData(model)
    raws, frames = [], []
    low, high = model.jnt_range[1:].T
    for _ in range(count):
        quat = rng.normal(size=4)
        data.qpos[:3] = rng.normal(size=3)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] = rng.uniform(low, high)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        raws.append(np.concatenate([data.qpos, data.qvel, rng.normal(size=model.nu)]))
        com = data.subtree_com[1]
        linear = data.cvel[:, 3:6] + np.cross(data.cvel[:, 0:3], data.xpos - com)
        frames.append((data.xpos.copy(), data.xmat.reshape(-1, 3, 3).copy(), linear, data.cvel[:, 0:3].copy()))
    return torch.tensor(np.array(raws)), frames


def test_the_state_layouts_round_trip_exactly(env):
    layout = wm.Layout(env.model.nu)
    assert (layout.raw, layout.canonical, layout.features, layout.deltas, layout.errors) == (100, 102, 97, 73, 70)
    raw, _ = _random_raw(env.model, np.random.default_rng(0), 6)
    state = layout.from_raw(raw)
    following = layout.from_raw(raw.roll(1, 0))
    rebuilt = layout.advance(state, layout.deltas_of(state, following), layout.part(following, 'previous'))
    assert torch.allclose(layout.rotation(rebuilt), layout.rotation(following), atol=1e-12)
    assert layout.error(rebuilt, following).abs().max() < 1e-12
    rotation = layout.rotation(state)  # the 6D gives the quaternion's rotation back, orthonormal
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3, dtype=rotation.dtype).expand(6, 3, 3),
                          atol=1e-12)


def test_the_kinematics_are_mujocos(env):
    layout = wm.Layout(env.model.nu)
    kinematics = wm.Kinematics(env.model, dtype=torch.float64)
    raw, frames = _random_raw(env.model, np.random.default_rng(1), 5)
    computed = kinematics(layout.from_raw(raw), layout)
    for k, frame in enumerate(frames):  # positions, rotations, frame-origin linear and angular velocities
        for ours, theirs in zip(computed, frame):
            assert np.abs(ours[k].numpy() - theirs).max() < 1e-10


def test_the_policy_frame_reward_and_terminations_are_the_tasks(env):
    terms = wm.TrackTerms(env, dtype=torch.float64)
    layout = terms.layout
    rng = np.random.default_rng(2)
    env.reset(seed=3)

    def raw():
        return layout.from_raw(torch.tensor(np.concatenate([env.data.qpos, env.data.qvel, env.previous_action]))[None])
    for _ in range(25):
        before = raw()
        _, reward, terminated, truncated, info = env.step(rng.normal(0., 1., env.model.nu))
        after = raw()
        frame = wm.policy_frame(after, layout, terms.kinematics, terms.home)
        assert (frame[0] - env.frames[0, -1]).abs().max() < 1e-10
        clock, yaw = torch.tensor([env.clip_clock]), torch.tensor([env.clip_yaw])
        offset = torch.tensor(env.clip_offset)[None]
        ours, _ = terms.reward(after, before, clock, yaw, offset)
        contact = sum(info['reward_components'][key] for key in ('self_contact', 'impact', 'sole_impact'))
        assert abs(float(ours[0]) - (reward - contact)) < 1e-10
        grace = torch.tensor([env.clip_start == 'bank' and env.steps <= env.clip_grace])
        assert bool(terms.terminated(after, clock, yaw, offset, grace)[0]) == bool(terminated)
        lifted = after.clone()
        lifted[:, 2] += 1.  # the anchor a metre above its reference ends the episode
        assert bool(terms.terminated(lifted, clock, yaw, offset, grace)[0])
        assert terms.margins(after, clock, yaw, offset, grace).shape == (1, wm.TRACK_GOAL)
        if terminated or truncated:
            env.reset()


def _margins_by_the_formula(terms, following, clock, yaw, offset, grace):
    """TrackTerms.margins as first written: every body by the full kinematics, the reference by clip_reference."""
    position, rotation, _, _ = terms.robot(following)
    reference, turned, _, _ = terms.reference(clock, yaw, offset)
    ends = list(terms.ends)
    speed = terms.layout.part(following, 'speeds').abs().amax(-1) / task.GETUP_MAX_JOINT_SPEED
    return torch.cat((position[:, terms.anchor, 2:3] - reference[:, terms.anchor, 2:3],
                      (rotation[:, terms.anchor, 2, 2] - turned[:, terms.anchor, 2, 2])[:, None],
                      position[:, ends, 2] - reference[:, ends, 2], speed[:, None],
                      grace.to(following.dtype)[:, None]), -1)


def test_the_poses_and_the_margins_are_the_full_kinematics(env):
    terms = wm.TrackTerms(env, dtype=torch.float64)
    layout = terms.layout
    raw, _ = _random_raw(env.model, np.random.default_rng(5), 7)
    state = layout.from_raw(raw)
    full = terms.kinematics(state, layout)
    bodies = [int(b) for b in terms.bodies[[3, 0, 9]]]
    position, rotation = terms.kinematics.poses(state, layout, bodies)
    assert (position - full[0][:, bodies]).abs().max() < 1e-12 and (rotation - full[1][:, bodies]).abs().max() < 1e-12
    clock = torch.tensor([0, 5, 17, 39, 40, 41, 500])  # the hold and past it too
    yaw = torch.linspace(-3., 3., 7, dtype=torch.float64)
    offset = torch.randn((7, 2), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    grace = torch.tensor([True, False] * 3 + [True])
    ours = terms.margins(state, clock, yaw, offset, grace)
    assert (ours - _margins_by_the_formula(terms, state, clock, yaw, offset, grace)).abs().max() < 1e-12


def test_track_terms_take_the_weights_and_refuse_rules_they_do_not_know():
    weighted = wm.TrackTerms(track_env(track_weights={'self_contact': 0., 'joint_limit': 2.}))
    assert weighted.weights['joint_limit'] == 2. and weighted.weights['self_contact'] == 0.
    with pytest.raises(ValueError, match='track_end_reach'):
        wm.TrackTerms(track_env(track_end_reach=.4))
    clip = {'file': 'x', 'md5': '0' * 32}
    wm.check_track_config({'track_clip': clip, 'track_clips': None, 'track_end_reach': None})
    for config in ({}, {'track_clip': None}, {'track_clip': clip, 'track_clips': [clip, clip]},
                   {'track_clip': clip, 'track_end_reach': .4},
                   {'track_clip': clip, 'track_handover_starts': {'share': .5, 'file': 'h', 'md5': '0' * 32}}):
        with pytest.raises(ValueError, match='модель мира'):
            wm.check_track_config(config)


class _Box:
    low, high = np.full(29, -10.), np.full(29, 10.)


class _Policy:
    """A stand-in for an SB3 policy: the mean 9.95 on joint 0 and 0 elsewhere; sampled adds 5."""
    action_space = _Box()

    def predict(self, observations, deterministic):
        mean = np.zeros((len(observations), 29))
        mean[:, 0] = 9.95
        return (mean if deterministic else mean + 5.), None


class _Identity:
    def normalize_obs(self, observations):
        return observations


def test_the_collectors_action_modes():
    collect = _script('collect')
    observations = np.zeros((4000, 3))
    noisy = collect.cpu_actor(_Policy(), _Identity(), 'noise', .1, seed=0)(observations)
    assert noisy[:, 0].max() <= 10. and (noisy[:, 0] == 10.).any()  # clipped to the box
    assert abs(noisy[:, 1:].std() - .1) < .005 and abs(noisy[:, 1:].mean()) < .005
    assert (collect.cpu_actor(_Policy(), _Identity(), 'deterministic')(observations)[:, 1:] == 0).all()
    assert (collect.cpu_actor(_Policy(), _Identity(), 'sampled')(observations)[:, 1:] == 5.).all()
    with pytest.raises(ValueError):
        collect.cpu_actor(_Policy(), _Identity(), 'other')
    # The CUDA collector's actor, run here on the CPU: the mean plus N(0, S²) (run_cuda clips to the box afterwards).
    seen = []

    def policy(observations, deterministic):
        seen.append(deterministic)
        return torch.full((len(observations), 29), 2.), None, None

    class Norm:
        def observation(self, observations, update):
            assert update is False
            return observations
    actions = collect.cuda_actor(policy, Norm(), 'noise', .1, seed=0, device='cpu')(torch.zeros((4000, 3)))
    assert abs(float((actions - 2.).std()) - .1) < .005 and seen == [True]
    assert torch.equal(collect.cuda_actor(policy, Norm(), 'deterministic', device='cpu')(torch.zeros((5, 3))),
                       torch.full((5, 29), 2.))
    collect.cuda_actor(policy, Norm(), 'sampled', device='cpu')(torch.zeros((5, 3)))
    assert seen[-2:] == [True, False]


def _small(goal=0):
    torch.manual_seed(0)
    return wm.WorldModel(29, len(wm.CONTEXT), goal, hidden=(64, 64), termination_hidden=(16,))


def _dynamics(state, action, context):
    """A made-up dynamics: joint speeds follow the action, damped by the first context number, joints the speeds, the
    root drifts with its velocity."""
    layout = wm.Layout(29)
    delta = torch.zeros((len(state), layout.deltas))
    delta[:, :3] = .02 * layout.part(state, 'linear')
    delta[:, 9:38] = .02 * layout.part(state, 'speeds')
    delta[:, 44:73] = .3 * action - .1 * layout.part(state, 'speeds') * context[:, :1]
    return layout.advance(state, delta, action)


def _synthetic(n, generator):
    """Random canonical states, actions and contexts and the states _dynamics gives after them."""
    layout = wm.Layout(29)
    state = layout.from_raw(torch.randn((n, layout.raw), generator=generator))
    action = torch.randn((n, 29), generator=generator)
    context = torch.rand((n, len(wm.CONTEXT)), generator=generator)
    return state, action, context, _dynamics(state, action, context)


def test_the_networks_shapes_and_its_loss_falls_on_synthetic_dynamics():
    model = _small()
    layout = model.layout
    generator = torch.Generator().manual_seed(0)
    state, action, context, following = _synthetic(512, generator)
    assert model.step(state, action, context).shape == (512, layout.canonical)
    assert model.termination_logit(state).shape == (512,)
    wm.fit_normalization(model, state, action, context, following)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        state, action, context, following = _synthetic(256, generator)
        # Two-step windows of the same dynamics: the second step starts from the first's true state.
        actions = torch.stack((action, torch.randn((256, 29), generator=generator)), 1)
        targets = torch.stack((following, _dynamics(following, actions[:, 1], context)), 1)
        alive = torch.ones((256, 2), dtype=torch.bool)
        terminated = torch.zeros((256, 2), dtype=torch.bool)
        loss, parts, states, logits = wm.window_loss(model, state, actions, context, targets, terminated, alive)
        assert states.shape == (256, 2, layout.canonical) and logits.shape == (256, 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), wm.GRAD_CLIP)
        optimizer.step()
        losses.append(float(parts['mse']))
    assert np.mean(losses[-10:]) < .3 * np.mean(losses[:10])


def test_masked_steps_give_no_nan_gradient():
    model = _small()
    generator = torch.Generator().manual_seed(1)
    state, action, context, following = _synthetic(8, generator)
    targets = torch.stack((following, torch.full_like(following, float('nan'))), 1)
    alive = torch.tensor([[True, False]] * 8)
    loss, _, _, _ = wm.window_loss(model, state, action[:, None].repeat(1, 2, 1), context, targets,
                                   torch.zeros((8, 2), dtype=torch.bool), alive)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_save_and_load_give_the_same_model(tmp_path):
    model = _small(goal=wm.TRACK_GOAL)
    generator = torch.Generator().manual_seed(2)
    state, action, context, following = _synthetic(64, generator)
    wm.fit_normalization(model, state, action, context, following, torch.randn((64, wm.TRACK_GOAL)))
    model.save(tmp_path / 'model.pt', note='test')
    loaded, extra = wm.WorldModel.load(tmp_path / 'model.pt')
    assert extra == {'note': 'test'}
    goal = torch.randn((64, wm.TRACK_GOAL))
    assert torch.equal(loaded.step(state, action, context), model.step(state, action, context))
    assert torch.equal(loaded.termination_logit(following, goal), model.termination_logit(following, goal))


@pytest.fixture(scope='module')
def collected(tmp_path_factory):
    collect = _script('collect')
    out = tmp_path_factory.mktemp('wm') / 'data'
    rng = np.random.default_rng(4)
    envs = [track_env(max_episode_steps=7) for _ in range(2)]  # short episodes: resets inside the chunks
    meta = collect.collect_cpu(envs, lambda observations: rng.normal(0., 2., (len(observations), 29)), 20, out,
                               chunk=8, meta=dict(task_config=None))
    return out, meta, envs[0]


def test_collect_on_the_cpu_task(collected):
    out, meta, env = collected
    fit = _script('fit')
    assert sorted(path.name for path in out.glob('chunk_*.npz')) == [f'chunk_0000{k}.npz' for k in range(3)]
    assert json.loads((out / 'meta.json').read_text(encoding='utf-8'))['worlds'] == 2
    _, arrays = fit.load(out)
    assert arrays['state'].shape == (20, 2, 100) and arrays['next'].shape == (20, 2, 100)
    assert arrays['state'].dtype == np.float32 and arrays['context'].shape == (20, 2, len(wm.CONTEXT))
    boundary = arrays['terminated'] | arrays['truncated']
    assert boundary.sum() >= 4  # 20 steps of 7-step episodes in two worlds
    inside = ~boundary[:-1]
    assert np.array_equal(arrays['next'][:-1][inside], arrays['state'][1:][inside])
    assert np.array_equal(arrays['next'][..., -29:], arrays['action'])  # the next state's previous action
    assert not np.array_equal(arrays['next'][:-1][boundary[:-1]], arrays['state'][1:][boundary[:-1]])
    assert (arrays['contact_next'] > 0).any() and (arrays['force_next'] > 0).any()
    # The tracking reward of the written (float32) states is the task's minus its contact terms.
    terms = wm.TrackTerms(env)
    layout = terms.layout
    state, following = layout.from_raw(torch.from_numpy(arrays['state'])), layout.from_raw(
        torch.from_numpy(arrays['next']))
    reward, _ = terms.reward(following.reshape(40, -1), state.reshape(40, -1),
                             torch.from_numpy(arrays['clock'].reshape(-1).astype(np.int64)),
                             torch.from_numpy(arrays['yaw'].reshape(-1)), torch.from_numpy(arrays['offset'].reshape(-1, 2)))
    expected = (arrays['reward'] - arrays['reward_contact']).reshape(-1)
    assert np.abs(reward.numpy() - expected).max() < 1e-3


def test_fit_runs_for_seconds_and_reports(collected, tmp_path):
    out, _, env = collected
    fit = _script('fit')
    lines = []
    report = fit.fit(out, tmp_path / 'fit', max_steps=20, device='cpu', hidden=(64, 64), batch=16, holdout=.5,
                     eval_stride=2, reference=env, log=lines.append)
    assert (tmp_path / 'fit' / 'model.pt').is_file() and (tmp_path / 'fit' / 'report.json').is_file()
    assert report['training']['steps'] == 20
    every = report['errors']['all']['all']
    assert np.isfinite(every['spread']['mean']) and set(every['spread']['groups']) == {
        'root', 'joints', 'velocities', 'bodies'}
    assert set(every['spread']['groups_persist']) == set(every['spread']['groups'])
    assert np.isfinite(every['spread']['persist']) and every['skill'] == pytest.approx(
        1 - every['spread']['mean'] / every['spread']['persist'])
    rise, hold = report['errors']['rise']['all'], report['errors']['hold']['all']
    assert rise['steps'] + hold.get('steps', 0) == every['steps']  # the rise and the hold split the steps
    assert report['reward']['formula_check']['max_abs'] < 1e-3
    assert {'r_persist', 'skill', 'anomaly_r'} <= set(report['reward']['rise']['all'])
    assert len(report['curve']) == 1 and report['curve'][0]['steps'] == 20  # the end's checkpoint
    assert (tmp_path / 'fit' / 'train_state.pt').is_file() and report['health'] is None
    assert any('Порог ступени 2 лестницы' in line for line in lines)
    again = fit.fit(out, tmp_path / 'fit', device='cpu', holdout=.5, eval_stride=2, reference=env,
                    evaluate_only=True, log=lines.append)
    assert again['errors']['all']['all']['spread']['mean'] == pytest.approx(every['spread']['mean'])


def rising_clip(frames=FRAMES, lift=.6):
    """test_g1_getup_track.synthetic_clip's motion lifted `lift` m over the clip: the anchor's reference leaves a robot
    on the floor by more than TRACK_ANCHOR_Z within half the clip, so its episodes end by the task's termination."""
    env = bank_module.getup_model_env()
    start = drop_bank()['qpos'][0].copy()
    start[2] -= DROP + .03
    t = np.linspace(0., 1., frames)
    qpos = np.repeat(start[None], frames, 0)
    qpos[:, 0] += .3 * t
    qpos[:, 2] += lift * t
    qpos[:, 7:] += np.sin(np.pi * t)[:, None] * np.random.default_rng(3).uniform(-.3, .3, 29)
    return g1_getup_track.build(env, qpos)


def ending_env(**settings):
    return track_env(rising_clip(), track_starts={'first': .5, 'rsi': 0., 'bank': .5},
                     track_bank_classes=['back', 'front'], max_episode_steps=30, **settings)


@pytest.fixture(scope='module')
def ended(tmp_path_factory):
    collect = _script('collect')
    out = tmp_path_factory.mktemp('wm-ended') / 'data'
    rng = np.random.default_rng(6)
    envs = [ending_env() for _ in range(3)]
    meta = collect.collect_cpu(envs, lambda observations: rng.normal(0., 1., (len(observations), 29)), 45, out,
                               chunk=16, meta=dict(task_config=None))
    return out, meta, envs[0]


def test_the_collected_steps_and_start_kinds_follow_the_tasks_terminations(ended):
    out, _, env = ended
    fit = _script('fit')
    _, arrays = fit.load(out)
    terminated, boundary = arrays['terminated'], arrays['terminated'] | arrays['truncated']
    assert terminated.sum() >= 6 and (arrays['kind'] == task.TRACK_START_KINDS.index('bank')).any()
    assert (arrays['kind'] == task.TRACK_START_KINDS.index('clip')).any() and not arrays['bad'].any()
    steps, kind = arrays['steps'], arrays['kind']
    assert (steps[0] == 1).all()
    after = boundary[:-1]  # the row after an end starts another episode, else the same one goes on
    assert (steps[1:][after] == 1).all() and (steps[1:][~after] == steps[:-1][~after] + 1).all()
    assert (kind[1:][~after] == kind[:-1][~after]).all()
    # TrackTerms.terminated of the written state after each step, its grace from the written kind and steps, is the
    # task's termination of that step.
    terms = wm.TrackTerms(env)
    following = terms.layout.from_raw(torch.from_numpy(arrays['next']).reshape(-1, 100))
    grace = terms.grace_of(torch.from_numpy(kind.astype(np.int64)), torch.from_numpy(steps.astype(np.int64)))
    said = terms.terminated(following, torch.from_numpy(arrays['clock'].astype(np.int64)).reshape(-1),
                            torch.from_numpy(arrays['yaw']).reshape(-1), torch.from_numpy(arrays['offset']).reshape(-1, 2),
                            grace.reshape(-1))
    assert np.array_equal(said.numpy(), terminated.reshape(-1))
    graced = grace.numpy() & (kind == task.TRACK_START_KINDS.index('bank'))
    assert graced.any()  # bank starts' first steps are in the data


def test_the_fit_sees_held_out_terminations_checkpoints_and_resumes(ended, tmp_path):
    out, _, env = ended
    fit = _script('fit')
    lines = []
    report = fit.fit(out, tmp_path / 'fit', max_steps=6, device='cpu', hidden=(32,), batch=8, holdout=.34,
                     eval_stride=1, reference=env, checkpoint_minutes=1e-9, curve_windows=64, log=lines.append)
    term = report['termination']
    assert term['positives'] > 0 and term['steps'] > term['positives']  # terminations in the held-out windows
    assert term['auroc'] is not None and term['ap'] is not None and term['bce_constant'] > 0
    assert term['rule_true']['agree'] == 1. and term['rule_true']['missed'] == 0 and term['rule_true']['extra'] == 0
    assert 'rule_predicted' in term
    assert [row['steps'] for row in report['curve']] == list(range(1, 7))  # a checkpoint after every step
    assert json.loads((tmp_path / 'fit' / 'curve.json').read_text(encoding='utf-8')) == report['curve']
    resumed = fit.fit(out, tmp_path / 'fit', max_steps=9, device='cpu', batch=8, holdout=.34, eval_stride=1,
                      reference=env,
                      resume=True, checkpoint_minutes=None, curve_windows=64, log=lines.append)
    assert resumed['training']['steps'] == 9 and resumed['curve'][-1]['steps'] == 9 and len(resumed['curve']) == 7
    assert any('продолжаю с шага 6' in line for line in lines)


def test_auroc_and_average_precision_by_hand():
    fit = _script('fit')
    score, label = np.array([.1, .4, .35, .8, .4]), np.array([False, False, True, True, True])
    # positive-negative pairs ordered right: (.35 > .1), (.8 > both), (.4 > .1), (.4 = .4: a half), (.35 < .4)
    assert fit._auroc(score, label) == pytest.approx((1 + 2 + 1 + .5) / 6)
    assert fit._average_precision(np.array([.9, .8, .7]), np.array([True, False, True])) == pytest.approx(
        (1 + 2 / 3) / 2)
    assert fit._auroc(score, np.zeros(5, dtype=bool)) is None


def test_the_fit_warns_about_an_unhealthy_collection():
    fit = _script('fit')
    assert fit.health_warning({'health': 'ok'}) is None and fit.health_warning({}) is None
    assert 'не здоров' in fit.health_warning({'health': 'nacon overflow'})
