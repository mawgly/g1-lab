"""The world model's adapters (humanoid_sim/world_model_adapters) and the ladder (27.10), CPU, small sizes: getting up
gives the golden digests of the code before the adapters (tests/data/world_model_golden.json, made from b36749f by the
same procedure), its model files have no new settings, a model without inputs steps as before; the fixed-base layout
and kinematics against MuJoCo on the SO-101; the arm's reward and success from states against ArmReachEnv.step; the
walking reward from states against G1VelocityEnv's components (candidate 8's task, pushes, motor friction, arm
perturbation), its velocity with a COM offset, the root kick, the push input and the physics hook; collection and a fit
of seconds for the arm and for walking; the registry's refusals; the window gather on a key subset; and the adapters'
formulas free of host syncs and boolean-mask indexing (what a CUDA graph cannot hold)."""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim import world_model as wm
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.world_model_adapters import (ADAPTERS, Terms, adapter_for, adapter_of_meta, arm as arm_adapter,
                                               getup as getup_adapter, walk as walk_adapter)
from humanoid_sim.world_model_adapters.arm import ArmReach
from humanoid_sim.world_model_adapters.getup import GetupTrack
from humanoid_sim.world_model_adapters.walk import G1Walk

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'data' / 'world_model_golden.json'
# Walking candidate 8's task_config (exp-28691193718e4d0e @ 7221510144).
CANDIDATE8 = dict(
    reward_weights=dict(velocity=2., forward=0., sideways=0., heading=0., yaw=2., yaw_drift=5., alive=.15, gait=.5,
                        swing=10., tilt=0., orientation=2., height=10., torso=1., base_motion=2., still=3.,
                        stand_pose=1., slip=1., feet_flat=2., clearance=0., self_contact=1., arm_press=.1, posture=.3,
                        hips=1., effort=1e-5, action_change=.05, fall=100.),
    heading_hold=True, commands=dict(min_speed=-1., max_speed=2., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                                     straight_probability=.5, side_share=.2, side_stop=1., side_speed=1.),
    gait=dict(period=.6, stance_fraction=.55), action_limit=10., real_robot=True, critic_extras=True,
    training_realism=1., history=5, stand_heading_hold=False, phase_stop=True, sole_contact=True, random_phase=True,
    action_delay=0., arm_perturb=.1, lock_wrists=True, knee_floor=.061, force_push=1.25, feet_flat_stance=False,
    foot_collision=True, side_heading_hold=False, motor_friction=1.25)


def _script(name):
    spec = importlib.util.spec_from_file_location(f'wm_{name}', ROOT / 'scripts' / 'lab' / 'wm' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walking_config(**changes):
    from humanoid_sim.tasks.config import g1_config
    return g1_config({**CANDIDATE8, **changes})


def walking_env(**changes):
    from humanoid_sim.tasks.catalog import make_task
    return make_task('g1_velocity', 'g1_29dof', walking_config(**changes)).unwrapped


def arm_env():
    from humanoid_sim.tasks.catalog import make_task
    return make_task('arm_reach', 'so101', {}).unwrapped


# --- step 2: getting up is the old code ----------------------------------------------------------------------------


def _close(old, new, path=''):
    """Recursively: every key of old in new, numbers within rel 1e-6 (abs 1e-9), the rest equal."""
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        for key, value in old.items():
            assert key in new, f'{path}/{key}'
            _close(value, new[key], f'{path}/{key}')
    elif isinstance(old, list):
        assert isinstance(new, list) and len(old) == len(new), path
        for index, (a, b) in enumerate(zip(old, new)):
            _close(a, b, f'{path}/{index}')
    elif isinstance(old, float) and not isinstance(old, bool):
        assert isinstance(new, (int, float)), path
        if math.isnan(old):
            assert math.isnan(new), path
        else:
            assert new == pytest.approx(old, rel=1e-6, abs=1e-9), path
    else:
        assert old == new, path


def test_getting_up_is_the_old_code(tmp_path):
    """b36749f's run (collect_cpu 4 × 80 of the ending task, a 40-step fit), rerun with adapter None: the same chunks
    byte for byte, the model's tensors and the report's old numbers (within rel 1e-6, so another CPU passes)."""
    from test_world_model import ending_env
    collect, fit = _script('collect'), _script('fit')
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(6)
        envs = [ending_env() for _ in range(4)]
        meta = collect.collect_cpu(envs, lambda observations: rng.normal(0., 1., (len(observations), 29)), 80,
                                   tmp_path / 'data', seed=0, chunk=50, meta=dict(task_config=None))
        fit.fit(tmp_path / 'data', tmp_path / 'fit', device='cpu', hidden=(64, 64), batch=64, max_steps=40,
                holdout=.25, eval_windows=2000, eval_stride=2, checkpoint_minutes=None, seed=0, reference=envs[0],
                log=lambda _: None)
    finally:
        torch.set_num_threads(threads)
    for name, digests in golden['chunks'].items():
        with np.load(tmp_path / 'data' / name) as chunk:
            assert list(chunk.files) == list(digests)  # the same keys in the same order
            for key, digest in digests.items():
                assert hashlib.sha256(np.ascontiguousarray(chunk[key]).tobytes()).hexdigest() == digest, (name, key)
    _close(golden['meta'], meta)
    saved = torch.load(tmp_path / 'fit' / 'model.pt', weights_only=False)
    assert saved['settings'] == golden['settings']
    for key, (total, size) in golden['state'].items():
        value = saved['state'][key].double()
        assert float(value.sum()) == pytest.approx(total, rel=1e-6, abs=1e-9), key
        assert float(value.abs().sum()) == pytest.approx(size, rel=1e-6, abs=1e-9), key
    report = json.loads((tmp_path / 'fit' / 'report.json').read_text(encoding='utf-8'))
    _close(golden['report'], report)
    assert report['adapter'] == 'g1_getup_track' and report['ladder']['passed'] == report['passed']
    assert 'horizon_32' in report and report['horizon_32']['windows'] > 0


def test_getting_up_model_files_keep_their_settings(tmp_path):
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    model = wm.WorldModel(29, len(wm.CONTEXT), wm.TRACK_GOAL, hidden=(64, 64), **GetupTrack().model_settings())
    assert set(model.settings) == set(golden['settings'])
    settings = dict(golden['settings'])  # today's settings dict loads as before
    settings.pop('version')
    wm.WorldModel(**settings).save(tmp_path / 'model.pt')
    loaded, _ = wm.WorldModel.load(tmp_path / 'model.pt')
    assert loaded.settings == golden['settings']
    walking = wm.WorldModel(29, 6, 2, hidden=(16,), **G1Walk().model_settings())
    assert walking.settings['kick'] and walking.settings['push'] and 'root' not in walking.settings


def _old_step(model, state, action, context):
    """WorldModel.step as written before the inputs."""
    inputs = torch.cat((model.layout.network_features(state, model.home), action, context), -1)
    delta = model.dynamics((inputs - model.input_mean) / model.input_std) * model.delta_std + model.delta_mean
    return model.layout.advance(state, delta, action)


def test_without_inputs_the_model_steps_as_before():
    torch.manual_seed(0)
    model = wm.WorldModel(29, len(wm.CONTEXT), wm.TRACK_GOAL, hidden=(32, 32), termination_hidden=(8,))
    layout = model.layout
    generator = torch.Generator().manual_seed(3)
    state = layout.from_raw(torch.randn((16, layout.raw), generator=generator))
    actions = torch.randn((16, 3, 29), generator=generator)
    context = torch.rand((16, len(wm.CONTEXT)), generator=generator)
    for parameter in model.parameters():  # not the zero output layers' start
        parameter.data.normal_(0., .1, generator=generator)
    assert torch.equal(model.step(state, actions[:, 0], context), _old_step(model, state, actions[:, 0], context))
    states, logits = model.rollout(state, actions, context)
    expected, current = [], state
    for j in range(3):
        current = _old_step(model, current, actions[:, j], context)
        expected.append(current)
    assert torch.equal(states, torch.stack(expected, 1))
    targets = torch.stack(expected, 1) + .01
    alive = torch.ones((16, 3), dtype=torch.bool)
    terminated = torch.zeros((16, 3), dtype=torch.bool)
    loss, _, _, _ = wm.window_loss(model, state, actions, context, targets, terminated, alive)
    again, _, _, _ = wm.window_loss(model, state, actions, context, targets, terminated, alive, inputs=None)
    assert torch.equal(loss, again) and torch.isfinite(loss)


def test_kicks_and_pushes_enter_the_model():
    torch.manual_seed(1)
    model = wm.WorldModel(29, 6, 2, hidden=(16,), kick=True, push=True)
    layout = model.layout
    generator = torch.Generator().manual_seed(4)
    state = layout.from_raw(torch.randn((8, layout.raw), generator=generator))
    action, context = torch.randn((8, 29), generator=generator), torch.rand((8, 6), generator=generator)
    assert model.dynamics[0].in_features == layout.features + 29 + 6 + 3
    zero = model.step(state, action, context, torch.zeros((8, 9)))
    assert torch.equal(zero, model.step(state, action, context))  # no inputs: zeros
    kick = torch.zeros((8, 9))
    kick[:, 0] = 1.  # +1 m/s along world x before the physics: the root's linear velocity in its frame moves by Rᵀ·x
    kicked = model.step(state, action, context, kick)
    rotation = layout.rotation(state)
    moved = layout.part(kicked, 'linear') - layout.part(zero, 'linear')
    assert torch.allclose(moved, rotation[:, 0, :], atol=1e-5)


# --- the fixed base (step 0) ---------------------------------------------------------------------------------------


def test_the_fixed_layout_round_trips():
    layout = wm.FixedLayout(6, 6)
    assert (layout.raw, layout.canonical, layout.features, layout.deltas, layout.errors) == (12, 12, 12, 12, 12)
    raw = torch.randn((5, 12), dtype=torch.float64)
    state, following = layout.from_raw(raw), layout.from_raw(raw.roll(1, 0))
    assert torch.equal(state, raw) and state.data_ptr() != raw.data_ptr()
    rebuilt = layout.advance(state, layout.deltas_of(state, following), torch.zeros((5, 6), dtype=torch.float64))
    assert layout.error(rebuilt, following).abs().max() < 1e-12
    kept = wm.FixedLayout(6, 6, previous=True)
    assert kept.raw == 18 and kept.features == 18 and kept.deltas == 12
    action = torch.randn((5, 6), dtype=torch.float64)
    assert torch.equal(kept.part(kept.advance(torch.randn((5, 18)), torch.zeros((5, 12)), action), 'previous'), action)
    with pytest.raises(ValueError):
        layout.rotation(state)
    assert torch.equal(layout.kicked(state, torch.ones((5, 6), dtype=torch.float64))[:, 6:], state[:, 6:] + 1)


def test_the_fixed_base_kinematics_are_mujocos():
    env = arm_env()
    model = env.model
    layout = wm.FixedLayout(6, 6)
    kinematics = wm.Kinematics(model, dtype=torch.float64)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    low, high = model.jnt_range.T
    for _ in range(5):
        data.qpos[:] = rng.uniform(low, high)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        state = layout.from_raw(torch.tensor(np.concatenate([data.qpos, data.qvel]))[None])
        position, rotation, linear, angular = kinematics(state, layout)
        bodies = [body for body in range(1, model.nbody) if model.body_mocapid[body] < 0]
        assert np.abs(position[0, bodies].numpy() - data.xpos[bodies]).max() < 1e-10
        assert np.abs(rotation[0, bodies].numpy() - data.xmat[bodies].reshape(-1, 3, 3)).max() < 1e-10
        com = data.subtree_com[model.body_rootid[bodies]]
        speed = data.cvel[bodies, 3:6] + np.cross(data.cvel[bodies, 0:3], data.xpos[bodies] - com)
        assert np.abs(linear[0, bodies].numpy() - speed).max() < 1e-10
        assert np.abs(angular[0, bodies].numpy() - data.cvel[bodies, 0:3]).max() < 1e-10
        tip = kinematics.points(state, layout, [model.site_bodyid[env.tip_id]], model.site_pos[env.tip_id][None])
        assert np.abs(tip[0, 0].numpy() - data.site_xpos[env.tip_id]).max() < 1e-10
        target = model.body('target').id
        assert torch.isnan(position[0, target]).all()  # the mocap target is not part of the state
    with pytest.raises(ValueError, match='mocap'):
        kinematics.poses(state, layout, [model.body('target').id])


def test_the_arm_reward_and_success_from_states_are_the_envs():
    env = arm_env()
    adapter = ArmReach({})
    terms = adapter.bind(env, 'cpu', torch.float64)
    layout = terms.layout
    env.reset(seed=2)
    rng = np.random.default_rng(1)
    successes = 0
    for t in range(150):
        # Toward the target (a P controller), with noise: the episode reaches it, holds and succeeds.
        action = np.clip((env.goal_qpos[:5] - env.data.qpos[:5]) / .08, -1, 1) + (rng.normal(0., .3, 5) if t < 20
                                                                                  else 0.)
        raw = adapter.cpu_raw(env)
        before = adapter.cpu_before(env)
        _, reward, terminated, truncated, info = env.step(action)
        fields, excluded, applied = adapter.cpu_after(env, info, raw, action)
        assert excluded == 0. and np.allclose(applied + raw[:6], env.data.ctrl, atol=1e-12, rtol=0)  # u = ctrl − q
        following = layout.from_raw(torch.tensor(adapter.cpu_raw(env))[None])
        flat = dict(target=torch.tensor(before['target'])[None], hold_before=torch.tensor([before['hold_before']]),
                    policy_action=torch.tensor(fields['policy_action'])[None])
        ours, components = terms.reward(following, layout.from_raw(torch.tensor(raw)[None]), flat)
        assert abs(float(ours[0]) - reward) < 1e-9
        assert bool(terms.terminated(following, flat)[0]) == terminated
        assert abs(float(terms.distance(following, flat['target'])[0]) - info['distance']) < 1e-12
        successes += terminated
        if terminated or truncated:
            env.reset()
    assert successes >= 1


def _arm_collected(tmp_path, worlds=4, steps=200):
    collect = _script('collect')
    rng = np.random.default_rng(5)
    envs = [arm_env() for _ in range(worlds)]
    meta = collect.collect_cpu(envs, lambda observations: rng.uniform(-1., 1., (len(observations), 5)), steps,
                               tmp_path / 'data', chunk=64, meta=dict(task_config={}, robot_id='so101',
                                                                      task_id='arm_reach'),
                               adapter=ArmReach({}))
    return meta


def test_the_arm_collects_and_fits_end_to_end(tmp_path):
    fit = _script('fit')
    meta = _arm_collected(tmp_path)
    assert meta['adapter'] == 'arm_reach' and meta['raw'] == 12 and meta['context'] == [] and meta['dt'] == .04
    meta_back, arrays = fit.load(tmp_path / 'data')
    assert arrays['state'].shape == (200, 4, 12) and arrays['force'].shape == (200, 4, 0)
    assert arrays['target'].shape == (200, 4, 3) and arrays['policy_action'].shape == (200, 4, 5)
    boundary = arrays['terminated'] | arrays['truncated']
    inside = ~boundary[:-1]
    assert np.array_equal(arrays['next'][:-1][inside], arrays['state'][1:][inside])
    lines = []
    report = fit.fit(tmp_path / 'data', tmp_path / 'fit', max_steps=30, device='cpu', hidden=(64,), batch=32,
                     holdout=.25, eval_stride=4, log=lines.append)
    assert report['adapter'] == 'arm_reach' and report['ladder']['step'] == 0
    assert report['reward']['formula_check']['max_abs'] < 1e-3
    assert set(report['errors']) == {'all'} and report['errors']['all']['contact_windows'] == dict(count=0)
    assert set(report['errors']['all']['all']['spread']['groups']) == {'joints', 'velocities', 'bodies'}
    assert report['horizon_32']['windows'] > 0 and 'spread' in report['horizon_32']['errors']['all']['all']
    assert report['horizon_32']['scales']['spread'] == report['scales']['spread']
    assert any('contact_windows: нет окон' in line for line in lines)
    names = [criterion['name'] for criterion in report['ladder']['criteria']]
    assert names == ['error', 'skill', 'reward', 'auroc']
    assert report['held_joints'] == [5] and 'deepest' in report['errors']['all']['all']
    assert report['training']['batch'] == 32 and report['training']['speed'] == 'plain'
    model, _ = wm.WorldModel.load(tmp_path / 'fit' / 'model.pt')
    assert model.settings['root'] == 'fixed' and model.settings['previous'] is False
    with pytest.raises(SystemExit, match='продолжение только'):  # a resume keeps the run's batch
        fit.fit(tmp_path / 'data', tmp_path / 'fit', max_steps=40, device='cpu', batch=16, holdout=.25,
                eval_stride=4, resume=True, log=lambda _: None)
    lines = []
    again = fit.fit(tmp_path / 'data', tmp_path / 'fit', max_steps=35, device='cpu', holdout=.25, eval_stride=4,
                    resume=True, log=lines.append)
    assert again['training']['batch'] == 32 and again['training']['steps'] == 35
    assert any('как до перерыва' in line for line in lines)


def test_the_report_leaves_held_joints_out_and_gives_the_deepest_step():
    fit = _script('fit')
    layout, arm = wm.FixedLayout(6, 6), ArmReach({})
    columns, groups, slices = fit.judged_columns(layout, arm)
    assert columns.tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert groups['joints'].tolist() == [0, 1, 2, 3, 4] and groups['velocities'].tolist() == [5, 6, 7, 8, 9]
    assert fit.judged_columns(wm.Layout(29), GetupTrack()) is None
    n, k = 4, 8
    alive = np.zeros((n, k), dtype=bool)
    alive[:, :3] = True  # every window's episode ends after its third step
    error = np.full((n, k, 12), .1, dtype=np.float32)
    error[..., 11] = 100.  # the held gripper's speed: huge, and left out
    stacked = dict(alive=alive, error=error, standing=np.ones((n, k, 12), dtype=np.float32))
    scales = dict(motion=np.ones(12, dtype=np.float32), spread=np.ones(12, dtype=np.float32))
    every = np.ones((n, k), dtype=bool)
    part = fit._errors(stacked, layout, scales, np.ones(12, dtype=np.float32), every, (columns, groups, slices))
    assert part['spread']['mean'] == pytest.approx(.01) and part['deepest'] == 3 and part['full_windows'] == 0
    assert part['physical']['speeds']['rms_step8'] is None
    whole = fit._errors(stacked, layout, scales, np.ones(12, dtype=np.float32), every)
    assert whole['spread']['mean'] > 100


def test_the_walking_report_leaves_the_locked_wrists_out():
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.world_model_adapters.walk import LOCKED_WRISTS
    env = walking_env()
    joints = [env.model.actuator(i).name for i in range(env.model.nu)]
    locked = tuple(i for i, name in enumerate(joints) if any(part in name for part in task.LOCKED_WRIST_PARTS))
    assert locked == LOCKED_WRISTS
    assert G1Walk(walking_config(lock_wrists=True)).held == LOCKED_WRISTS
    assert G1Walk(walking_config(lock_wrists=False)).held == ()


def test_the_walking_phases_and_pushed_windows():
    terms = G1Walk(walking_config()).bind(walking_env(), 'cpu')
    command = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 0., .5], [0., 0., .5], [0., .5, 0.]])[:, None]
    masks = terms.phase_masks(dict(command=command))
    names = [next(key for key, mask in masks.items() if mask[row, 0]) for row in range(5)]
    assert names == ['stand', 'walk', 'walk_turn', 'turn', 'side']
    assert sum(mask.int() for mask in masks.values()).eq(1).all()
    inputs = torch.zeros((2, 3, 9))
    inputs[:, 2, 0] = 1.  # a kick on the third step of both windows
    alive = torch.tensor([[True, True, True], [True, True, False]])  # the second window's episode ended before it
    assert terms.window_sets(dict(inputs=inputs, alive=alive))['pushed_windows'].tolist() == [True, False]


# --- step 1: walking -----------------------------------------------------------------------------------------------


def test_the_walking_reward_from_states_is_the_tasks():
    collect = _script('collect')
    config = walking_config()
    adapter = G1Walk(config)
    env = walking_env()
    terms = adapter.bind(env, 'cpu', torch.float64)
    layout = terms.layout
    calls = []
    collect._record_exogenous(env)
    record = env._before_physics

    def hook(pushes):
        calls.append(pushes)
        record(pushes)
    env._before_physics = hook
    env.reset(seed=1)
    rng = np.random.default_rng(2)
    pushed, heading, falls = [], None, 0
    for t in range(60):
        if t == 10:
            env.set_force_push((1., .5), .1)
            heading = float(task.heading(torch.tensor(env.data.xmat[env.pelvis].reshape(1, 3, 3)))[0])
        raw, before, context = adapter.cpu_raw(env), adapter.cpu_before(env), adapter.cpu_context(env)
        action = rng.normal(0., .3, 29)
        _, reward, terminated, truncated, info = env.step(action)
        assert len(calls) == t + 1 and (calls[-1] is not None) == (10 <= t < 15)  # once a step, with the push
        fields, excluded, applied = adapter.cpu_after(env, info, raw, action)
        inputs = adapter.cpu_inputs(env, raw)
        assert (inputs[:6] == 0).all()  # no kicks on the CPU task
        pushed.append(inputs[6:])
        flat = {key: torch.tensor(np.asarray(value))[None] for key, value in {**before, **fields}.items()}
        flat.update(action=torch.tensor(applied)[None], context=torch.tensor(context)[None],
                    inputs=torch.tensor(inputs)[None])
        state = layout.from_raw(torch.tensor(raw)[None])
        following = layout.from_raw(torch.tensor(adapter.cpu_raw(env))[None])
        _, components = terms.reward(following, state, flat)
        for key, value in info['reward_components'].items():
            if key not in adapter.excluded:
                assert abs(float(components[key][0]) - value) < 1e-6, key
        assert abs(sum(info['reward_components'][key] for key in adapter.excluded) - excluded) < 1e-12
        assert bool(terms.terminated(following, flat)[0]) == terminated
        falls += terminated
        if terminated or truncated:
            env.reset()
    turn = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    assert np.abs(np.sum(pushed, 0)[:2] - turn @ np.array([1., .5])).max() < 1e-9  # Σ Δv = the push's Δv
    assert np.abs(np.sum(pushed, 0)[2]) < 1e-12


def test_the_walking_velocity_takes_the_com_offset():
    env = walking_env()
    terms = G1Walk(walking_config()).bind(env, 'cpu', torch.float64)
    env.reset(seed=4)
    rng = np.random.default_rng(3)
    for _ in range(5):
        env.step(rng.normal(0., .3, 29))
    model, data = env.model, mujoco.MjData(env.model)
    data.qpos[:], data.qvel[:] = env.data.qpos, env.data.qvel
    data.qvel[:6] = rng.normal(size=6)
    offset = np.array([.02, -.015, .025])
    nominal = model.body_ipos[env.pelvis].copy()
    try:
        model.body_ipos[env.pelvis] = nominal + offset
        mujoco.mj_forward(model, data)
        expected = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, env.pelvis, expected, 1)
    finally:
        model.body_ipos[env.pelvis] = nominal
    state = terms.layout.from_raw(torch.tensor(np.concatenate([data.qpos, data.qvel, np.zeros(29)]))[None])
    ours = terms.velocity(state, torch.tensor(offset)[None])
    assert np.abs(ours[0].numpy() - expected).max() < 1e-10


def test_the_root_kick_is_applied_as_qvel():
    layout = wm.Layout(29)
    rng = np.random.default_rng(6)
    raw = torch.tensor(rng.normal(size=(4, layout.raw)))
    kick = torch.tensor(rng.normal(size=(4, 6)))
    by_hand = raw.clone()
    by_hand[:, layout.nq:layout.nq + 6] += kick
    assert (layout.kicked(layout.from_raw(raw), kick) - layout.from_raw(by_hand)).abs().max() < 1e-12


def test_walking_collects_and_fits_on_the_cpu(tmp_path):
    collect, fit = _script('collect'), _script('fit')
    config = walking_config()
    adapter = G1Walk(config)
    envs = [adapter.cpu_env(config) for _ in range(2)]
    rng = np.random.default_rng(7)
    meta = collect.collect_cpu(envs, lambda observations: rng.normal(0., .3, (len(observations), 29)), 60,
                               tmp_path / 'data', chunk=25,
                               meta=dict(task_config=config, robot_id='g1_29dof', task_id='g1_velocity'),
                               adapter=adapter)
    assert meta['adapter'] == 'g1_walk' and meta['context'] == list(adapter.context) and meta['inputs'][6] == 'push_x'
    _, arrays = fit.load(tmp_path / 'data')
    assert arrays['inputs'].shape == (60, 2, 9) and arrays['observed_command'].shape == (60, 2, 3)
    assert np.array_equal(arrays['next'][..., -29:], arrays['action'])
    report = fit.fit(tmp_path / 'data', tmp_path / 'fit', max_steps=10, device='cpu', hidden=(32,), batch=16,
                     holdout=.5, eval_stride=2, log=lambda _: None)
    check = report['reward']['formula_check']
    assert check['p99_abs'] < 1e-3 and check['max_abs'] < 1e-2
    assert set(report['errors']) == {'stand', 'walk', 'walk_turn', 'turn', 'side', 'all'}
    assert 'pushed_windows' in report['errors']['all'] and report['horizon_32']['windows'] > 0
    assert report['termination']['rule_true']['agree'] == 1.
    model, _ = wm.WorldModel.load(tmp_path / 'fit' / 'model.pt')
    assert model.kick and model.push


# --- the registry, the data, the formulas --------------------------------------------------------------------------


def test_the_registry_resolves_and_refuses_in_russian(tmp_path):
    collect = _script('collect')
    assert set(ADAPTERS) == {('g1_29dof', 'g1_getup'), ('g1_29dof', 'g1_velocity'), ('so101', 'arm_reach')}
    with pytest.raises(ValueError, match='адаптера пока нет'):
        adapter_for('humanoid', 'humanoid_v5', {})
    for changes in (dict(arm_targets=True, upper_body_slot=True), dict(height_command=True)):
        with pytest.raises(ValueError, match='пока не поддержана'):
            G1Walk({**CANDIDATE8, **changes})
    with pytest.raises(ValueError, match='реального робота'):
        G1Walk({**CANDIDATE8, 'real_robot': False})
    with pytest.raises(ValueError, match='задержка больше шага'):
        G1Walk({**CANDIDATE8, 'action_delay': .03})
    G1Walk({**CANDIDATE8, 'action_delay': .02}).check('cuda')
    weights = dict(CANDIDATE8['reward_weights'], self_contact=0.)
    with pytest.raises(ValueError, match='разбор касаний'):
        G1Walk({**CANDIDATE8, 'reward_weights': weights, 'foot_collision': False}).check('cuda')
    G1Walk({**CANDIDATE8, 'reward_weights': weights, 'foot_collision': False}).check('cpu')
    with pytest.raises(ValueError, match='нет сборщика CUDA'):
        ArmReach({}).check('cuda')
    with pytest.raises(ValueError, match='модель мира'):
        adapter_for('g1_29dof', 'g1_getup', {'track_clip': None})
    assert isinstance(adapter_of_meta({}), GetupTrack) and isinstance(adapter_of_meta({'task_config': None}),
                                                                       GetupTrack)
    empty = adapter_of_meta({'task_config': {}})  # as the old fit: an empty task_config runs without terms
    assert isinstance(empty, GetupTrack) and empty.reference({'task_config': {}}) is None
    assert isinstance(adapter_of_meta({'adapter': 'arm_reach', 'task_config': {}}), ArmReach)
    assert isinstance(adapter_of_meta({'adapter': 'g1_walk', 'task_config': CANDIDATE8}), G1Walk)
    run = tmp_path / 'so101'
    run.mkdir()
    revision = (ROOT / arm_adapter.REVISION).read_text(encoding='utf-8').strip()
    (run / 'training_config.json').write_text(json.dumps({'model_revision': revision}), encoding='utf-8')
    (run / 'best_model.zip').write_bytes(b'')
    source = collect.resolve(str(run))
    assert (source.robot_id, source.task_id, source.config, source.steps) == ('so101', 'arm_reach', {}, None)
    (run / 'training_config.json').write_text(json.dumps({'model_revision': 'other'}), encoding='utf-8')
    with pytest.raises(SystemExit, match='другой ревизии'):
        collect.resolve(str(run))


def test_a_window_of_a_key_subset_is_the_full_gather_on_those_keys(tmp_path):
    fit = _script('fit')
    _arm_collected(tmp_path, worlds=3, steps=20)
    meta, arrays = fit.load(tmp_path / 'data')
    data = fit.Data(arrays, 'cpu', .34, adapter_of_meta(meta))
    starts, worlds = data.sample(16, 8, torch.Generator().manual_seed(0))
    full = data.window(starts, worlds, 8, 'cpu')
    part = data.window(starts, worlds, 8, 'cpu', fit.training_keys(ArmReach({})))
    assert set(part) == {'state', 'context', 'next', 'action', 'terminated', 'bad', 'target', 'hold_before', 'alive'}
    for key, value in part.items():
        assert torch.equal(value, full[key]), key


class _Strict:
    """Within it a tensor refuses what a CUDA graph cannot hold: a value read on the host (item, tolist, bool, int,
    float, nonzero, numpy) and indexing by a boolean mask (a shape that depends on the data)."""
    NAMES = ('item', 'tolist', '__bool__', '__int__', '__float__', 'nonzero', 'numpy', '__getitem__')

    def __enter__(self):
        self.saved = {name: getattr(torch.Tensor, name) for name in self.NAMES}

        def refuse(name):
            def method(*args, **kwargs):
                raise AssertionError(f'host sync: Tensor.{name}')
            return method
        for name in self.NAMES[:-1]:
            setattr(torch.Tensor, name, refuse(name))
        getitem = self.saved['__getitem__']

        def item(tensor, index):
            for part in index if isinstance(index, tuple) else (index,):
                if isinstance(part, torch.Tensor) and part.dtype == torch.bool:
                    raise AssertionError('boolean-mask indexing')
            return getitem(tensor, index)
        torch.Tensor.__getitem__ = item
        return self

    def __exit__(self, *exc):
        for name, method in self.saved.items():
            setattr(torch.Tensor, name, method)


def _random_states(layout, n, generator):
    raw = torch.randn((n, layout.raw), generator=generator)
    if layout.free_root:
        raw[:, 2] = .5 + torch.rand(n, generator=generator)
    return layout.from_raw(raw)


def test_the_adapters_formulas_are_shape_static():
    """What a CUDA graph cannot hold (fit's graph step replays the goal closure; the report's formulas are meant to run
    inside imagination later): every Terms method of the three adapters on random states, under _Strict, gives
    outputs whose shapes depend on the inputs' alone; and no .item()/.tolist() in their source."""
    from test_g1_getup_track import track_env
    with _Strict():  # the guard itself
        with pytest.raises(AssertionError, match='boolean-mask'):
            torch.ones(2)[torch.tensor([True, False])]
        with pytest.raises(AssertionError, match='host sync'):
            float(torch.ones(()))
        with pytest.raises(AssertionError, match='host sync'):
            torch.ones(()).item()
    generator = torch.Generator().manual_seed(9)
    n, k = 6, 2
    cases = []
    getup = GetupTrack().bind(track_env(), 'cpu')
    cases.append((getup, dict(clock=torch.randint(0, 40, (n, k), generator=generator), yaw=torch.rand((n, k)),
                              offset=torch.rand((n, k, 2)), kind=torch.randint(0, 3, (n, k), generator=generator),
                              steps=torch.randint(0, 40, (n, k), generator=generator))))
    walk = G1Walk(walking_config()).bind(walking_env(), 'cpu')
    command = torch.randn((n, k, 3), generator=generator)
    command[0, :, :] = 0.
    command[1, :, 0], command[1, :, 2] = 0., 0.
    cases.append((walk, dict(observed_command=command, command=command, heading_target=torch.rand((n, k)),
                             yaw_drift_before=torch.rand((n, k)), phase_steps=torch.randint(0, 99, (n, k)),
                             steps=torch.randint(0, 99, (n, k)), inputs=torch.randn((n, k, 9), generator=generator))))
    arm = ArmReach({}).bind(arm_env(), 'cpu')
    cases.append((arm, dict(target=torch.rand((n, k, 3)), hold_before=torch.randint(0, 6, (n, k)),
                            policy_action=torch.rand((n, k, 5)))))
    for terms, fields in cases:
        layout = terms.layout
        batch = dict(fields, action=torch.randn((n, k, layout.nu), generator=generator),
                     context=torch.rand((n, 6 if terms is walk else 7), generator=generator),
                     alive=torch.rand((n, k), generator=generator) < .8)
        flat = {key: value.reshape(n * k, *value.shape[2:]) for key, value in fields.items()}
        flat.update(action=batch['action'].reshape(n * k, -1),
                    context=batch['context'][:, None].expand(n, k, -1).reshape(n * k, -1))
        state, following = _random_states(layout, n * k, generator), _random_states(layout, n * k, generator)
        with _Strict():
            reward, components = terms.reward(following, state, flat)
            terminated = terms.terminated(following, flat)
            goal = terms.goal(batch)
            rows = terms.goal_rows(following, {key: flat[key] for key in flat})
            first = goal(0, following[:n]) if goal is not None else None
            masks, sets = terms.phase_masks(batch), terms.window_sets(batch)
            bodies = terms.bodies(state)
        assert reward.shape == (n * k,) and all(value.shape == (n * k,) for value in components.values())
        assert terminated.shape == (n * k,) and terminated.dtype == torch.bool
        assert rows.shape[0] == n * k and (first is None or first.shape == (n, rows.shape[1]))
        assert all(mask.shape == (n, k) for mask in masks.values()) and all(s.shape == (n,) for s in sets.values())
        assert bodies[0].shape[:1] == (n * k,)
    for module in (getup_adapter, walk_adapter, arm_adapter):
        for _, kind in inspect.getmembers(module, inspect.isclass):
            if issubclass(kind, Terms) and kind is not Terms:
                source = inspect.getsource(kind)
                assert '.item(' not in source and '.tolist(' not in source, kind.__name__


def test_the_ladder_judges_each_step():
    fit = _script('fit')

    def report(error, skill, reward, positives, auroc, phase='all', reward_set='all'):
        return dict(errors={phase: {'all': {'spread': dict(mean=error, skill=skill)}}},
                    reward={phase: {reward_set: dict(anomaly_r=reward)}},
                    termination=dict(positives=positives, steps=1000, auroc=auroc))
    arm = ArmReach({})
    assert fit.ladder(report(.01, .95, .95, 30, .97), arm)['passed']
    assert not fit.ladder(report(.01, .85, .95, 30, .97), arm)['passed']  # skill below 0.9
    rare = fit.ladder(report(.01, .95, .95, 5, .2), arm)
    assert rare['passed'] and rare['criteria'][3]['passed'] is None and 'мало обрывов' in rare['criteria'][3]['reason']
    shown = fit.ladder(report(.01, .95, .95, 30, .2), arm)  # the arm's AUROC reads a logged input: it does not decide
    assert shown['passed'] and not shown['criteria'][3]['gating'] and 'счётчик' in shown['criteria'][3]['reason']
    walk = G1Walk()
    assert fit.ladder(report(.015, .85, .92, 0, None, reward_set='contact_steps'), walk)['passed']
    getup = GetupTrack()
    judged = fit.ladder(report(.01, .1, .95, 30, .3, phase='rise', reward_set='contact_steps'), getup)
    assert judged['passed']  # skill and AUROC shown, not judged
    assert fit.verdict(dict(report(.01, .1, .95, 30, .3, phase='rise', reward_set='contact_steps')), getup)[0]
