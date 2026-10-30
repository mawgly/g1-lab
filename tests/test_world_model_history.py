"""The world model's history (28.10): with history H the dynamics also see the features of the H − 1 states before the
current one. A model without it is the old one (the other world model tests); with it, step needs the earlier states,
rollout slides them, the network reads them, files keep the setting; fit.Data gives each window the states before its
start, repeating its episode's first state before that (as a controller fills its history at a reset); and the arm
collects, fits and re-evaluates with history 2 end to end."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim import world_model as wm
from humanoid_sim.world_model_adapters.arm import ArmReach

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f'wm_{name}', ROOT / 'scripts' / 'lab' / 'wm' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(history, **settings):
    torch.manual_seed(0)
    model = wm.WorldModel(29, len(wm.CONTEXT), 0, hidden=(32, 32), termination_hidden=(8,), history=history,
                          **settings)
    generator = torch.Generator().manual_seed(3)
    for parameter in model.parameters():  # not the zero output layers' start
        parameter.data.normal_(0., .1, generator=generator)
    return model, generator


def test_a_history_model_reads_the_earlier_states_and_slides_them(tmp_path):
    model, generator = _model(3)
    layout = model.layout
    assert model.settings['history'] == 3
    assert model.dynamics[0].in_features == 3 * layout.features + 29 + len(wm.CONTEXT)
    state = layout.from_raw(torch.randn((8, layout.raw), generator=generator))
    past = layout.from_raw(torch.randn((8, 2, layout.raw), generator=generator))
    actions = torch.randn((8, 4, 29), generator=generator)
    context = torch.rand((8, len(wm.CONTEXT)), generator=generator)
    with pytest.raises(ValueError, match='earlier states'):
        model.step(state, actions[:, 0], context)
    first = model.step(state, actions[:, 0], context, past=past)
    other = past.clone()
    other[:, 0] = layout.from_raw(torch.randn((8, layout.raw), generator=generator))
    assert not torch.allclose(first, model.step(state, actions[:, 0], context, past=other))  # the oldest one counts
    states, _ = model.rollout(state, actions, context, past=past)
    current, earlier, expected = state, past, []
    for j in range(4):
        following = model.step(current, actions[:, j], context, past=earlier)
        earlier = torch.cat((earlier[:, 1:], current[:, None]), 1)
        current = following
        expected.append(current)
    assert torch.equal(states, torch.stack(expected, 1))
    model.save(tmp_path / 'model.pt')
    loaded, _ = wm.WorldModel.load(tmp_path / 'model.pt')
    assert loaded.history == 3 and torch.equal(loaded.step(state, actions[:, 0], context, past=past), first)
    alive = torch.ones((8, 4), dtype=torch.bool)
    loss, _, _, _ = wm.window_loss(model, state, actions, context, states + .01, torch.zeros_like(alive), alive,
                                   past=past)
    assert torch.isfinite(loss)


def test_without_history_a_model_file_has_no_new_setting():
    model, _ = _model(1)
    assert 'history' not in model.settings and model.history == 1


def test_history_with_kicks_and_pushes_and_the_normalization():
    model, generator = _model(2, kick=True, push=True)
    layout = model.layout
    state = layout.from_raw(torch.randn((64, layout.raw), generator=generator))
    past = layout.from_raw(torch.randn((64, 1, layout.raw), generator=generator))
    action, context = torch.randn((64, 29), generator=generator), torch.rand((64, len(wm.CONTEXT)), generator=generator)
    inputs = torch.randn((64, 9), generator=generator)
    following = layout.from_raw(torch.randn((64, layout.raw), generator=generator))
    wm.fit_normalization(model, state, action, context, following, inputs=inputs, past=past)
    size = 2 * layout.features + 29 + len(wm.CONTEXT) + 3
    assert model.input_mean.shape == (size,) and float(model.input_std.min()) > 0
    stepped = model.step(state, action, context, inputs, past=past)
    assert stepped.shape == state.shape and torch.isfinite(stepped).all()


def _arm_data(tmp_path, worlds=4, steps=120):
    from humanoid_sim.tasks.catalog import make_task
    collect = _script('collect')
    rng = np.random.default_rng(5)
    envs = [make_task('arm_reach', 'so101', {}).unwrapped for _ in range(worlds)]
    collect.collect_cpu(envs, lambda observations: rng.uniform(-1., 1., (len(observations), 5)), steps,
                        tmp_path / 'data', chunk=40, meta=dict(task_config={}, robot_id='so101', task_id='arm_reach'),
                        adapter=ArmReach({}))


def test_the_data_gives_each_window_the_states_before_it(tmp_path):
    fit = _script('fit')
    _arm_data(tmp_path, worlds=3, steps=120)
    _, arrays = fit.load(tmp_path / 'data')
    arrays['truncated'] = arrays['truncated'].copy()
    arrays['truncated'][50, 1] = True  # an episode boundary inside the data (after step 50 of world 1)
    data = fit.Data(arrays, 'cpu', 1 / 3, ArmReach({}))
    data.history = 4
    boundary = (arrays['terminated'] | arrays['truncated'] | arrays['bad'])
    starts = torch.tensor([0, 1, 2, 3, 10, 51, 52, 53, 54, 80])
    worlds = torch.tensor([0, 0, 0, 0, 2, 1, 1, 1, 1, 1])
    past = data.window(starts, worlds, 2, 'cpu')['past'].numpy()
    for row, (t, w) in enumerate(zip(starts.tolist(), worlds.tolist())):
        current, picked = t, []
        for j in range(1, 4):  # back from the start while still in its episode, else the earliest one repeats
            if t - j >= 0 and not boundary[t - j:t, w].any():
                current = t - j
            picked.append(current)
        expected = arrays['state'][picked[::-1], w]
        assert np.array_equal(past[row], expected), (t, w)
    assert np.array_equal(past[0], np.repeat(arrays['state'][0, 0][None], 3, 0))  # the data's first state
    assert np.array_equal(past[5], np.repeat(arrays['state'][51, 1][None], 3, 0))  # an episode's first state
    data.history = 1
    assert 'past' not in data.window(starts, worlds, 2, 'cpu')


def test_the_arm_fits_with_history_end_to_end(tmp_path):
    fit = _script('fit')
    _arm_data(tmp_path)
    report = fit.fit(tmp_path / 'data', tmp_path / 'fit', max_steps=12, device='cpu', hidden=(32,), batch=16,
                     holdout=.25, eval_stride=4, history=2, log=lambda _: None)
    assert np.isfinite(report['errors']['all']['all']['spread']['mean'])
    model, _ = wm.WorldModel.load(tmp_path / 'fit' / 'model.pt')
    assert model.history == 2 and model.settings['history'] == 2
    again = fit.fit(tmp_path / 'data', tmp_path / 'fit', device='cpu', holdout=.25, eval_stride=4,
                    evaluate_only=True, log=lambda _: None)
    assert again['errors']['all']['all']['spread']['mean'] == pytest.approx(
        report['errors']['all']['all']['spread']['mean'], rel=1e-5)


def test_an_ensemble_steps_as_its_members_mean():
    first, generator = _model(1)
    second, _ = _model(1)
    for parameter in second.parameters():
        parameter.data.normal_(0., .1, generator=generator)
    ensemble = wm.Ensemble([first, second])
    layout = first.layout
    state = layout.from_raw(torch.randn((8, layout.raw), generator=generator))
    actions = torch.randn((8, 3, 29), generator=generator)
    context = torch.rand((8, len(wm.CONTEXT)), generator=generator)
    mean = (first.step(state, actions[:, 0], context) + second.step(state, actions[:, 0], context)) / 2
    assert torch.allclose(ensemble.step(state, actions[:, 0], context), mean, atol=1e-6)
    states, logits = ensemble.rollout(state, actions, context)
    assert states.shape == (8, 3, layout.canonical) and logits.shape == (8, 3)
    spread = ensemble.disagreement(state, actions[:, 0], context)
    assert spread.shape == (8, layout.errors) and float(spread.detach().max()) > 0
    same = wm.Ensemble([first, first])
    assert float(same.disagreement(state, actions[:, 0], context).max()) == 0.
    with pytest.raises(ValueError, match='other layouts'):
        wm.Ensemble([first, wm.WorldModel(29, len(wm.CONTEXT), 0, hidden=(8,), kick=True)])
    longer, _ = _model(3)
    mixed = wm.Ensemble([first, longer])
    assert mixed.history == 3
    past = layout.from_raw(torch.randn((8, 2, layout.raw), generator=generator))
    expected = (first.step(state, actions[:, 0], context) + longer.step(state, actions[:, 0], context, past=past)) / 2
    assert torch.allclose(mixed.step(state, actions[:, 0], context, past=past), expected, atol=1e-6)


def test_the_arm_report_of_an_ensemble(tmp_path):
    fit = _script('fit')
    _arm_data(tmp_path)
    for seed in (0, 1):
        fit.fit(tmp_path / 'data', tmp_path / f'fit{seed}', max_steps=12, device='cpu', hidden=(32,), batch=16,
                holdout=.25, eval_stride=4, seed=seed, log=lambda _: None)
    lines = []
    report = fit.fit(tmp_path / 'data', tmp_path / 'both', device='cpu', holdout=.25, eval_stride=4,
                     evaluate_only=True, ensemble=[tmp_path / 'fit0', tmp_path / 'fit1'], log=lines.append)
    assert report['training']['ensemble'] == [str(tmp_path / 'fit0'), str(tmp_path / 'fit1')]
    assert np.isfinite(report['errors']['all']['all']['spread']['mean'])
    assert any('ансамбль из 2' in line for line in lines)
