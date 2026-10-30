"""The world model's collector on the CUDA task (scripts/lab/wm/collect.py: collecting_env, run_cuda). Run explicitly in
a free compute slot: RUN_G1_GPU_TESTS=1, .venv-fullgpu (as tests/test_g1_gpu.py). A synthetic clip and drop bank, short
episodes, a random policy: the hook leaves the trajectory as it is, keeps every world's state after the step before the
ended ones reset, the applied action is the next state's previous action, the context is what the task drew, and the
tracking reward of the written states by world_model.TrackTerms is the task's minus its contact terms; with a clip
rising out of reach and bank starts, the written step counts and start kinds follow the episodes and TrackTerms'
terminations of the written states are the task's. Walking (the g1_walk adapter, 27.10): the kicks and pushes the hook
records are the task's own, and the walking reward of the written states is the task's minus effort and self_contact;
fit's CUDA graph step gives the eager loop's numbers on the same windows, and torch.compile (Linux) the graph's."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get('RUN_G1_GPU_TESTS') != '1', reason='Requires exclusive GPU slot')


def _script(name):
    spec = importlib.util.spec_from_file_location(f'wm_{name}', ROOT / 'scripts' / 'lab' / 'wm' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings():
    from test_g1_getup_impact import drop_bank
    from test_g1_getup_track import SETTING, synthetic_clip
    return dict(bank=drop_bank(), track_clip=SETTING, track_bank=synthetic_clip(), max_episode_steps=6,
                reward_weights={'impact': 2., 'sole_impact': 1.})


def test_wm_collecting_env_keeps_the_trajectory_and_the_state_after_each_step():
    from humanoid_sim.g1_getup_tensor_env import TensorG1GetupEnv
    from humanoid_sim.g1_tensor_env import COM_OFFSET
    from humanoid_sim.tasks import g1_tensor_math as task
    collect = _script('collect')
    plain = TensorG1GetupEnv(8, seed=3, **_settings())
    hooked = collect.collecting_env(n_envs=8, seed=3, **_settings())
    try:
        generator = torch.Generator(device='cuda').manual_seed(0)
        first, second = plain.reset(), hooked.reset()
        torch.testing.assert_close(first, second)
        ended = 0
        for _ in range(15):
            context = hooked.context()
            friction, motor, delay, com, assist = context[:, 0], context[:, 1], context[:, 2], context[:, 3:6], \
                context[:, 6]
            assert ((friction >= .3) & (friction <= 1.)).all() and ((motor >= 0) & (motor <= 1.25)).all()
            assert ((delay >= 0) & (delay <= 1)).all() and (assist == 0).all()
            assert (com.abs() <= torch.tensor(COM_OFFSET, device='cuda') + 1e-6).all() and com.abs().max() > 0
            actions = 3 * torch.randn((8, 29), device='cuda', generator=generator)
            obs_plain, reward_plain, _, _, _ = plain.step(actions)
            hooked.post = {}
            obs, reward, terminated, truncated, _ = hooked.step(actions)
            post, hooked.post = hooked.post, None
            # Two Warp instances agree only to float32 contact noise: under 3σ random actions a few of the 4056 inputs
            # drift apart (seen 2.4e-4 on a 4080, 6e-3 on a 4090, 4.7e-2 once), so the hooks are judged by the share
            # of inputs that agree, not by the worst one.
            close = torch.isclose(obs, obs_plain, atol=1e-3, rtol=1e-3)
            assert float(close.float().mean()) >= .995, float(close.float().mean())
            torch.testing.assert_close(reward, reward_plain, atol=5e-2, rtol=1e-2)
            done = terminated | truncated
            ended += int(done.sum())
            now = torch.cat((hooked.qpos, hooked.qvel, hooked.previous_action), 1)
            torch.testing.assert_close(post['raw'][~done], now[~done])  # no reset: the state after the step
            applied = task.lock_joints(actions.clamp(-10, 10), hooked.wrist_mask)
            torch.testing.assert_close(post['raw'][:, -29:], applied)
            assert post['contact_part'].shape == (8,) and (post['contact_part'] <= 0).all()
        assert ended >= 8  # 6-step episodes: every world reset at least once
    finally:
        plain.close()
        hooked.close()


def test_wm_run_cuda_writes_transitions_whose_tracking_reward_is_the_tasks(tmp_path):
    from humanoid_sim import world_model as wm
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    collect, fit = _script('collect'), _script('fit')
    settings = _settings()
    env = collect.collecting_env(n_envs=4, seed=5, **settings)
    try:
        generator = torch.Generator(device='cuda').manual_seed(1)
        meta = collect.run_cuda(env, lambda observations: 2 * torch.randn((4, 29), device='cuda',
                                                                         generator=generator), 12, tmp_path / 'data',
                                chunk=5, meta=dict(task_config=None))
    finally:
        env.close()
    assert meta['health'] == 'ok' and meta['worlds'] == 4 and meta['chunks'] == 3
    _, arrays = fit.load(tmp_path / 'data')
    assert arrays['state'].shape == (12, 4, 100)
    boundary = arrays['terminated'] | arrays['truncated']
    inside = ~boundary[:-1]
    np.testing.assert_array_equal(arrays['next'][:-1][inside], arrays['state'][1:][inside])
    np.testing.assert_array_equal(arrays['next'][..., -29:], arrays['action'])
    assert boundary.sum() >= 4 and (arrays['force_next'] > 0).any()
    reference = G1GetupEnv(**{key: value for key, value in settings.items() if key != 'max_episode_steps'})
    terms = wm.TrackTerms(reference)
    layout = terms.layout
    state = layout.from_raw(torch.from_numpy(arrays['state']).reshape(48, -1))
    following = layout.from_raw(torch.from_numpy(arrays['next']).reshape(48, -1))
    reward, _ = terms.reward(following, state, torch.from_numpy(arrays['clock'].reshape(-1).astype(np.int64)),
                             torch.from_numpy(arrays['yaw'].reshape(-1)), torch.from_numpy(arrays['offset'].reshape(-1, 2)))
    expected = (arrays['reward'] - arrays['reward_contact']).reshape(-1)
    good = ~arrays['bad'].reshape(-1)
    np.testing.assert_allclose(reward.numpy()[good], expected[good], atol=5e-3)


def test_wm_run_cuda_steps_start_kinds_and_terminations_are_the_tasks(tmp_path):
    """A clip rising out of reach (test_world_model.rising_clip) with bank starts: the written step counts and start
    kinds follow the episodes, and TrackTerms.terminated of the written states, its grace from them, is the task's
    termination (up to float32 kinematics against Warp's on a margin's edge)."""
    from humanoid_sim import world_model as wm
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from test_world_model import rising_clip
    collect, fit = _script('collect'), _script('fit')
    settings = dict(_settings(), track_bank=rising_clip(), max_episode_steps=30,
                    track_starts={'first': .5, 'rsi': 0., 'bank': .5}, track_bank_classes=['back', 'front'])
    env = collect.collecting_env(n_envs=8, seed=7, **settings)
    try:
        generator = torch.Generator(device='cuda').manual_seed(2)
        meta = collect.run_cuda(env, lambda observations: torch.randn((8, 29), device='cuda', generator=generator),
                                45, tmp_path / 'data', chunk=16, meta=dict(task_config=None))
    finally:
        env.close()
    assert meta['health'] == 'ok'
    _, arrays = fit.load(tmp_path / 'data')
    terminated, boundary = arrays['terminated'], arrays['terminated'] | arrays['truncated']
    steps, kind = arrays['steps'], arrays['kind']
    assert terminated.sum() >= 16 and (kind == task.TRACK_START_KINDS.index('bank')).any()
    assert (steps[0] == 1).all()
    after = boundary[:-1]
    assert (steps[1:][after] == 1).all() and (steps[1:][~after] == steps[:-1][~after] + 1).all()
    assert (kind[1:][~after] == kind[:-1][~after]).all()
    reference = G1GetupEnv(**{key: value for key, value in settings.items() if key != 'max_episode_steps'})
    terms = wm.TrackTerms(reference)
    following = terms.layout.from_raw(torch.from_numpy(arrays['next']).reshape(-1, 100))
    grace = terms.grace_of(torch.from_numpy(kind.astype(np.int64)), torch.from_numpy(steps.astype(np.int64)))
    said = terms.terminated(following, torch.from_numpy(arrays['clock'].astype(np.int64)).reshape(-1),
                            torch.from_numpy(arrays['yaw']).reshape(-1),
                            torch.from_numpy(arrays['offset']).reshape(-1, 2), grace.reshape(-1)).numpy()
    good = ~arrays['bad'].reshape(-1)
    wrong = (said != terminated.reshape(-1))[good]
    assert wrong.sum() <= max(1, terminated.sum() // 100)


def test_wm_walking_records_the_tasks_kicks_and_pushes_and_its_reward(tmp_path):
    """g1_walk on TensorG1Env (candidate 8's task): the hook's kicks stay within REAL_PUSH and some happen, a forced
    push's per-step Δv inputs add up to the push in its heading frame, and the walking reward of written states by
    WalkTerms is the task's minus effort and self_contact."""
    from humanoid_sim.g1_tensor_env import REAL_PUSH
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.world_model_adapters.walk import G1Walk
    from test_world_model_adapters import walking_config, walking_env
    collect, fit = _script('collect'), _script('fit')
    config = walking_config()
    adapter = G1Walk(config)
    env = collect.collecting_env(adapter, n_envs=8, seed=3, max_episode_steps=40, **config)
    try:
        env.reset()
        generator = torch.Generator(device='cuda').manual_seed(0)
        bound = torch.tensor(REAL_PUSH, device='cuda') + 1e-5
        kicks, pushes, ended = 0, [], torch.zeros(8, dtype=torch.bool, device='cuda')
        for t in range(40):
            if t == 5:
                env.set_force_push(torch.tensor([1., .5], device='cuda'), .1)
                heading = task.heading(env.xmat[:, env.pelvis].reshape(-1, 3, 3))
            state = torch.cat((env.qpos, env.qvel, env.previous_action), 1).clone()
            env.post, env.exogenous = {}, None
            _, _, terminated, truncated, _ = env.step(.3 * torch.randn((8, 29), device='cuda', generator=generator))
            env.post = None
            inputs = adapter.cuda_inputs(env, state)
            assert (inputs[:, :6].abs() <= bound).all()
            kicks += int((inputs[:, :6].abs().sum(1) > 0).sum())
            if 5 <= t < 10:
                pushes.append(inputs[:, 6:])
                ended |= terminated | truncated
        assert kicks > 0
        total = torch.stack(pushes).sum(0)
        cos, sin = torch.cos(heading), torch.sin(heading)
        expected = torch.stack((cos - .5 * sin, sin + .5 * cos), -1)
        torch.testing.assert_close(total[~ended, :2], expected[~ended], atol=1e-4, rtol=1e-4)
    finally:
        env.close()
    env = collect.collecting_env(adapter, n_envs=8, seed=4, max_episode_steps=40, **config)
    try:
        generator = torch.Generator(device='cuda').manual_seed(1)
        meta = collect.run_cuda(env, lambda observations: .3 * torch.randn((8, 29), device='cuda', generator=generator),
                                30, tmp_path / 'data', chunk=10,
                                meta=dict(task_config=config, robot_id='g1_29dof', task_id='g1_velocity'))
    finally:
        env.close()
    assert meta['health'] == 'ok' and meta['adapter'] == 'g1_walk'
    _, arrays = fit.load(tmp_path / 'data')
    np.testing.assert_array_equal(arrays['next'][..., -29:], arrays['action'])
    terms = adapter.bind(walking_env(), 'cpu')
    layout = terms.layout
    flat = {field.name: torch.from_numpy(arrays[field.name].reshape(240, *arrays[field.name].shape[2:]).astype(
        np.int64 if arrays[field.name].dtype.kind == 'i' else np.float32)) for field in adapter.fields}
    flat.update(action=torch.from_numpy(arrays['action'].reshape(240, -1)),
                context=torch.from_numpy(arrays['context'].reshape(240, -1)))
    reward, _ = terms.reward(layout.from_raw(torch.from_numpy(arrays['next']).reshape(240, -1)),
                             layout.from_raw(torch.from_numpy(arrays['state']).reshape(240, -1)), flat)
    expected = (arrays['reward'] - arrays['reward_contact']).reshape(-1)
    good = ~arrays['bad'].reshape(-1)
    difference = np.abs(reward.numpy() - expected)[good]
    assert np.quantile(difference, .99) < 1e-3 and difference.max() < 5e-2  # a sole at the 1 cm threshold may flip


def _arm_on_cuda(tmp_path):
    from humanoid_sim.world_model_adapters import adapter_of_meta
    from test_world_model_adapters import _arm_collected
    fit = _script('fit')
    _arm_collected(tmp_path, worlds=4, steps=100)
    meta, arrays = fit.load(tmp_path / 'data')
    adapter = adapter_of_meta(meta)
    data = fit.Data(arrays, 'cuda', .25, adapter)
    return fit, meta, adapter, data, adapter.bind(adapter.reference(meta), 'cuda')


def _train(fit, meta, adapter, data, terms, speed, steps):
    from humanoid_sim import world_model as wm
    torch.manual_seed(0)
    model = wm.WorldModel(meta['nu'], len(meta['context']), adapter.goal, hidden=(64, 64),
                          **adapter.model_settings()).cuda()
    model.home.copy_(torch.as_tensor(meta['home'], dtype=torch.float32))
    fit.normalize(model, data, terms, model.layout, 'cuda', torch.Generator().manual_seed(1), adapter)
    info = {}
    _, _, last, _, generator = fit.train(model, data, terms, 'cuda', None, steps, 32, 2e-4, 0, lambda _: None,
                                         adapter=adapter, speed=speed, info=info)
    return last, [parameter.detach().clone() for parameter in model.parameters()], generator.get_state(), info


def test_wm_fit_graph_steps_are_the_eager_ones(tmp_path):
    """fit.train's CUDA graph over 20 steps against the eager loop: the same windows (the generator ends where the
    eager one does), the first step's losses and the parameters after the 20 steps within 1e-5."""
    fit, meta, adapter, data, terms = _arm_on_cuda(tmp_path)
    plain = _train(fit, meta, adapter, data, terms, 'plain', 20)
    graph = _train(fit, meta, adapter, data, terms, 'graph', 20)
    assert plain[3]['speed'] == 'plain' and graph[3]['speed'] == 'graph' and 'fallback' not in graph[3]
    assert graph[3]['warmup_steps'] == 3 and graph[3]['replay_ms'] > 0
    assert torch.equal(plain[2], graph[2])
    for key in ('mse', 'bce'):
        assert graph[0][key] == pytest.approx(plain[0][key], rel=1e-5, abs=1e-7)
    for ours, theirs in zip(graph[1], plain[1]):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='torch.compile needs Triton (Linux)')
def test_wm_fit_compiled_steps_are_the_graphs_without_recompiles(tmp_path):
    import torch._dynamo
    fit, meta, adapter, data, terms = _arm_on_cuda(tmp_path)
    graph = _train(fit, meta, adapter, data, terms, 'graph', 50)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = _train(fit, meta, adapter, data, terms, 'compile', 50)
    assert compiled[3]['speed'] == 'compile' and 'compile_error' not in compiled[3] and 'fallback' not in compiled[3]
    assert torch._dynamo.utils.counters['stats']['unique_graphs'] <= 1  # one compile, no recompile
    for ours, theirs in zip(compiled[1], graph[1]):
        torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)