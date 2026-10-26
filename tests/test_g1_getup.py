"""Task g1_getup (iteration 33), CPU: contract, settings, resets from the start bank, assist force, stages."""
import json
from argparse import Namespace

import mujoco
import numpy as np
import pytest

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_DEFAULTS, G1_GETUP_REWARD_WEIGHTS, G1_GETUP_TASK_WEIGHTS, g1_getup_config
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from humanoid_sim.tasks.identity import task_identity

ROWS = 4  # per class in the small bank, the last one held out


@pytest.fixture(scope='module')
def small_bank(tmp_path_factory):
    """A bank file of ROWS states per class, made as g1_getup_bank makes them (falls: lying states, labelled so)."""
    env = bank_module.getup_model_env()
    qpos, qvel, start_class, holdout, seed = [], [], [], [], []
    for number, kind in enumerate(bank_module.CLASSES):
        made = 0
        while made < ROWS:
            rng = np.random.default_rng(1000 * number + made + 7)
            if kind == 'stand':
                _, ok = bank_module.standing(env, rng)
            elif kind == 'sit':
                _, ok = bank_module.sitting(env, rng)
            else:
                _, ok = bank_module.lying(env, rng, kind if kind in bank_module.LYING else 'back')
            if ok:
                qpos.append(env.data.qpos.copy())
                qvel.append(env.data.qvel.copy())
                start_class.append(number)
                holdout.append(made == ROWS - 1)
                seed.append(made)
                made += 1
    path = tmp_path_factory.mktemp('bank') / 'bank.npz'
    header = dict(generator='test', model_revision=task_identity('g1_29dof', 'g1_getup')['model_revision'])
    np.savez_compressed(path, qpos=np.array(qpos, np.float32), qvel=np.array(qvel, np.float32),
                        start_class=np.array(start_class, np.uint8), holdout=np.array(holdout),
                        seed=np.array(seed, np.int64), variant=np.zeros(len(seed), np.uint8),
                        classes=np.array(bank_module.CLASSES), header=np.array(json.dumps(header)))
    return path


def rows_of(bank, kind):
    return np.flatnonzero(bank['start_class'] == bank['classes'].index(kind))


def test_its_own_contract_on_the_walking_model_and_layout():
    getup = task_identity('g1_29dof', 'g1_getup')
    walking = task_identity('g1_29dof', 'g1_velocity', dict(real_robot=True, critic_extras=True, history=5,
                                                            phase_stop=True, lock_wrists=True, knee_floor=.061,
                                                            foot_collision=True, hand_collision=True, action_limit=10.))
    assert getup['task_id'] == 'g1_getup' and getup['control_dt'] == .02
    assert getup['model_revision'] == walking['model_revision'] and getup['model_revision'].endswith('-footcol-handcol')
    assert getup['observation_schema'] == 'g1_getup:obs-unitree-deploy-v0-h5-phasestop-critic6'
    assert getup['action_schema'] == 'g1_getup:action-v2-limit10-wristlock-knee0.061'
    assert task_identity('g1_29dof', 'g1_getup', {'assist_force': 50., 'reward_weights': {'stand': 2.}}) == getup
    with pytest.raises(ValueError):
        validate_resume(walking, getup)  # a walking stage is not a getting-up stage
    with pytest.raises(ValueError):  # the legs-only G1 has no hands to get up with
        task_identity('g1_12dof', 'g1_getup')


def test_settings_are_bounded_and_spelled_out():
    from humanoid_sim.lab import validate_config
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup'))
    assert config['task_config'] == {'reward_weights': G1_GETUP_REWARD_WEIGHTS, 'assist_force': 200.,
                                     'action_delay': .02, 'motor_friction': 1.25, 'body_friction': [.3, 1.],
                                     'pd_bound': None, 'stand_pose_std': .1, 'action_change_unbounded': False,
                                     'style_library': 'lafan1', 'style_judge': False, 'human_starts': None,
                                     'stance_pelvis': .5, 'hip_roll_out': None, 'track_clip': None,
                                     'settle_clip': None}
    assert G1_GETUP_REWARD_WEIGHTS == dict(head=1., upright=.5, stand=1., stand_pose=.5, stand_still=1., effort=2e-5,
                                           joint_speed=1e-4, action_change=.01, joint_limits=1., impact=0.,
                                           sole_impact=0., stance_width=0., arm_press=0., style=0., head_rise=0.,
                                           sole_slip=0., amp=0.)
    assert G1_GETUP_TASK_WEIGHTS == tuple(G1_GETUP_REWARD_WEIGHTS)[:-1]  # amp is paid by the learner (getup_amp)
    for bad in ({'invented': 1}, {'reward_weights': {'velocity': 1.}}, {'reward_weights': {'head': -1}},
                {'assist_force': -1.}, {'assist_force': 600.}, {'action_delay': .1}, {'motor_friction': 3.},
                {'body_friction': [1., .3]}, {'body_friction': [.3]}, {'body_friction': [0., 1.]},
                {'heading_hold': True}, {'assist_force': True}):
        with pytest.raises(ValueError):
            g1_getup_config(bad)
    with pytest.raises(ValueError):  # CPU or the CUDA task (g1_getup_tensor_env), not Warp physics with CPU PPO
        validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='gpu'))
    assert validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='full_gpu', envs=4096, n_steps=24,
                                batch_size=24576))['backend'] == 'full_gpu'
    from humanoid_sim.backends.registry import backends
    with pytest.raises(ValueError):  # the adapter is there (a missing one says so before any CUDA setup)
        backends().make('g1_29dof', 'g1_getup', 'gpu')
    assert ('g1_29dof', 'g1_getup', 'full_gpu') in backends()._factories
    from humanoid_sim.tasks.catalog import catalog
    g1 = next(r for r in catalog() if r['robot_id'] == 'g1_29dof')
    assert [t['task_id'] for t in g1['tasks']] == ['g1_velocity', 'g1_getup']
    getup = g1['tasks'][1]
    assert getup['label'] == 'Вставание после падения' and getup['jobs'] == ['report'] and getup['race'] is False
    assert [p['key'] for p in getup['parameters']] == [*G1_GETUP_DEFAULTS, *G1_GETUP_REWARD_WEIGHTS]
    assert not any(t['task_id'] == 'g1_getup' for r in catalog() if r['robot_id'] != 'g1_29dof' for t in r['tasks'])


def test_the_walking_connector_with_a_zero_command_and_its_critic_extras(small_bank):
    env = G1GetupEnv(bank=small_bank)
    try:
        assert env.observation_space.shape == (496,) and env.actor_inputs == 490
        assert env.action_space.shape == (29,) and env.action_limit == 10. and env.lock_wrists and env.knee_floor == .061
        assert env.hand_collision and env.foot_collision and env.real_robot and env.phase_stop and env.history == 5
        columns, mean, variance = env.fixed_normalization
        assert len(columns) == 25 and (mean == 0).all() and (variance == 1).all()  # command 3 + phase 2, 5 frames
        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=3)
        for step in range(20):
            np.testing.assert_array_equal(obs[columns], 0.)
            velocity = env._velocity()
            torso = env.data.xpos[env.torso] + env.data.xmat[env.torso].reshape(3, 3) @ np.array(task.HEAD_TOP)
            np.testing.assert_allclose(obs[490:], np.r_[velocity[3:6], env.data.qpos[2], torso[2], env.assist],
                                       rtol=1e-6, atol=1e-6)
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-2, 2, 29))
            assert set(info['reward_components']) == set(G1_GETUP_TASK_WEIGHTS)
            assert reward == pytest.approx(sum(info['reward_components'].values()))
            assert not terminated and not truncated
        np.testing.assert_array_equal(env.command, 0.)
        env.set_command([0., 0., 0.])
        with pytest.raises(ValueError):
            env.set_command([.5, 0., 0.])
    finally:
        env.close()


def test_resets_draw_training_rows_and_take_options(small_bank):
    env = G1GetupEnv(bank=small_bank)
    try:
        bank = env.bank
        seen = set()
        for seed in range(200):
            _, info = env.reset(seed=seed)
            seen.add(info['start'])
            assert not bank['holdout'][info['start']]
            np.testing.assert_allclose(env.data.qpos[7:], bank['qpos'][info['start'], 7:], rtol=0, atol=1e-6)
            low, high = env.body_friction
            assert low <= info['body_friction'] <= high
            assert (env.model.geom_friction[env.friction_geoms, 0] == info['body_friction']).all()
            assert 0 <= env.delay_steps <= 10 and 0 <= env.friction_scale <= 1.25
        assert seen == set(np.flatnonzero(~bank['holdout']).tolist())  # every training row, no held-out one
        floor = env.model.geom('floor').id
        assert floor in env.friction_geoms and set(env.hand_capsules.tolist()) <= set(env.friction_geoms.tolist())
        assert set(env.soles.ravel().tolist()) <= set(env.friction_geoms.tolist())
        held = int(np.flatnonzero(bank['holdout'])[0])
        _, info = env.reset(seed=0, options=dict(start=held, assist_force=0., action_delay_steps=0,
                                                 motor_friction_scale=1., body_friction=.8))
        assert (info['start'], info['assist_force'], info['body_friction']) == (held, 0., .8)
        assert env.delay_steps == 0 and env.friction_scale == 1.
        for bad in (dict(start=len(bank['qpos'])), dict(start=True), dict(assist_force=-1.), dict(body_friction=0.)):
            with pytest.raises(ValueError):
                env.reset(seed=0, options=bad)
        # The same seed, the same episode.
        first = [env.reset(seed=5)[0]] + [env.step(np.full(29, .5))[0] for _ in range(5)]
        second = [env.reset(seed=5)[0]] + [env.step(np.full(29, .5))[0] for _ in range(5)]
        np.testing.assert_array_equal(np.array(first), np.array(second))
    finally:
        env.close()


def test_the_assist_lifts_a_nearly_upright_pelvis_only(small_bank):
    bank = bank_module.load(small_bank)
    stand, lying = int(rows_of(bank, 'stand')[0]), int(rows_of(bank, 'back')[0])
    options = dict(action_delay_steps=0, motor_friction_scale=1., body_friction=.6)
    results = {}
    for force in (200., 0.):
        env = G1GetupEnv(bank=small_bank)
        try:
            for row in (stand, lying):
                env.reset(seed=0, options=dict(options, start=row, assist_force=force))
                info = env.step(np.zeros(29))[-1]
                results[force, row] = (info['lift'], env.data.qpos.copy())
                assert not env.data.xfrc_applied.any()  # only while the step runs
        finally:
            env.close()
    assert results[200., stand][0] == 200. and results[0., stand][0] == 0.
    assert results[200., lying][0] == results[0., lying][0] == 0.
    assert results[200., stand][1][2] > results[0., stand][1][2]  # lifted
    np.testing.assert_array_equal(results[200., lying][1], results[0., lying][1])  # untouched


def test_the_assist_steps_down_after_an_episode_up(small_bank):
    env = G1GetupEnv(bank=small_bank)
    try:
        env.reset(seed=0)
        assert env.assist == 200.
        env.head_window.extend([1.05] * env.head_window.maxlen)  # the head above 1 m for the last 2 s
        assert env.reset(seed=1)[1]['assist_force'] == 180.
        env.head_window.extend([.95] * env.head_window.maxlen)
        assert env.reset(seed=2)[1]['assist_force'] == 180.
        env.head_window.extend([1.2] * (env.head_window.maxlen - 1))  # shorter than 2 s: no step
        assert env.reset(seed=3)[1]['assist_force'] == 180.
        env.assist = 10.
        env.head_window.extend([1.2] * env.head_window.maxlen)
        assert env.reset(seed=4)[1]['assist_force'] == 0.  # not below 0
        assert env.training_state() == {'assist_force': [0.]}
        env.restore_training_state({'assist_force': [200., 100., 0., 50.]}, 1, 4)
        assert env.assist == 100.
        env.restore_training_state({'assist_force': [200., 100., 0., 50.]}, 0, 2)  # quantiles of the sorted forces
        assert env.assist == 50.
        with pytest.raises(ValueError):
            env.restore_training_state({'assist_force': [-1.]})
    finally:
        env.close()


def test_episodes_end_only_by_time_a_joint_too_fast_or_a_bad_state(small_bank, monkeypatch):
    env = G1GetupEnv(max_episode_steps=30, bank=small_bank)
    try:
        env.reset(seed=0)
        ends = [env.step(np.random.default_rng(k).uniform(-1, 1, 29))[2:4] for k in range(30)]
        assert ends[:-1] == [(False, False)] * 29 and ends[-1] == (False, True)  # lying all along: truncated at 30
        monkeypatch.setattr(task, 'GETUP_MAX_JOINT_SPEED', 0.)
        env.reset(seed=0)
        assert env.step(np.zeros(29))[2] is True
        monkeypatch.undo()
        # A state MuJoCo had to reset (a bad position, velocity or acceleration warning; it goes on from the reference
        # pose, which would read as standing up) and a non-finite one end the episode as bad states. Simulated here:
        # a real warning would write MUJOCO_LOG.TXT into the working directory.
        import humanoid_sim.tasks.g1_getup as getup_module
        counts = iter([0, 1])
        monkeypatch.setattr(getup_module, 'instabilities', lambda data: next(counts))
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(29))
        assert terminated and info['bad_state'] and reward == 0 and (obs == 0).all()
        monkeypatch.undo()
        forward = mujoco.mj_forward

        def poisoned(model, data):
            forward(model, data)
            data.qvel[0] = np.nan
        env.reset(seed=0)
        monkeypatch.setattr(getup_module.mujoco, 'mj_forward', poisoned)
        obs, reward, terminated, truncated, info = env.step(np.zeros(29))
        monkeypatch.undo()
        assert terminated and info['bad_state'] and np.isfinite(obs).all()
        env.reset(seed=0)
        assert not env.step(np.zeros(29))[2]
    finally:
        env.close()


def test_the_success_tracker_wants_up_within_6_s_then_3_s_held(small_bank):
    env = G1GetupEnv(bank=small_bank)
    try:
        def run(pattern):
            env.reset(seed=0)
            env.getup_at, env.getup_held, env.hold_steps = None, None, 0
            for env.steps, (up, holding) in enumerate(pattern, 1):
                env._track(up, holding)
            return env.getup_at, env.getup_held
        up_at = 100  # 2 s
        assert run([(False, False)] * (up_at - 1) + [(True, True)] + [(False, True)] * 150) == (2., True)
        assert run([(False, False)] * (up_at - 1) + [(True, True)] + [(False, True)] * 149) == (2., None)
        assert run([(False, False)] * (up_at - 1) + [(True, True)] + [(True, True)] * 80 + [(True, False)]) == (2.,
                                                                                                               False)
        assert run([(False, False)] * 300 + [(True, True)]) == (None, False)  # after 6 s it no longer counts
        assert run([(False, False)] * 299 + [(True, True)] + [(True, True)] * 150) == (6., True)
    finally:
        env.close()


def test_stages_keep_the_assist_forces_and_a_resumed_cpu_run_restores_them(tmp_path, small_bank, monkeypatch):
    """The CPU path: the actor sees only the 490 walking inputs (CriticExtrasPolicy), each stage saves the worlds'
    assist forces in metadata.json, a resumed run gives them back to its worlds, and evaluate_stage (here stubbed in
    training) runs without the assist."""
    import humanoid_sim.train as cpu_train
    from humanoid_sim.core import evaluate_stage, list_stages, load_stage, read_json
    from humanoid_sim.lab import Laboratory
    from humanoid_sim.policies import CriticExtrasPolicy
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    monkeypatch.setattr(cpu_train, 'evaluate_stage', lambda *_: {'mean_return': 0., 'mean_seconds': 0.,
                                                                 'mean_forward_m': None})
    restored = []
    original = G1GetupEnv.restore_training_state

    def spy(self, state, index=0, count=1):
        original(self, state, index, count)
        restored.append((index, count, self.assist))
    monkeypatch.setattr(G1GetupEnv, 'restore_training_state', spy)
    lab = Laboratory(tmp_path / 'original')
    # One environment (in this process, where the small bank is patched in); 1024 steps: one reset at 500, at most
    # one step down of the assist.
    entry = lab.create(dict(robot_id='g1_29dof', task_id='g1_getup', envs=1, steps=1024, n_steps=64, batch_size=64,
                            n_epochs=1, task_config={'assist_force': 150.}))
    run = lab.resolve(entry['id'])
    cpu_train.train(Namespace(run_dir=run, config=run / 'run_config.json', parent_stop=None))
    stages = list_stages(run)
    assert read_json(run / 'status.json')['state'] == 'completed' and len(stages) == 2
    for stage in stages:
        forces = read_json(stage / 'metadata.json')['training_state']['assist_force']
        assert len(forces) == 1 and forces[0] in (150., 130.)
    saved = read_json(stages[-1] / 'metadata.json')['training_state']['assist_force']
    policy, norm = load_stage(stages[-1])
    try:
        assert isinstance(policy.policy, CriticExtrasPolicy) and policy.policy.actor_inputs == 490
    finally:
        norm.close()
    cpu_train.train(Namespace(run_dir=run, config=run / 'run_config.json', parent_stop=None))
    assert [(index, count) for index, count, _ in restored] == [(0, 1)]
    assert [force for *_, force in restored] == saved
    result = evaluate_stage(stages[0], episodes=2, max_steps=20)
    assert result['task_id'] == 'g1_getup' and result['success_rate'] == 0.
    assert all(r['success'] is False and r['start_class'] in bank_module.CLASSES for r in result['episodes'])
    from humanoid_sim.tasks.catalog import resolve_task
    assert resolve_task('g1_29dof', 'g1_getup').evaluation_options == {'assist_force': 0., 'pd_bound': 'min'}


def test_the_vector_helpers_leave_other_tasks_alone(small_bank):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    from humanoid_sim.train import keeps_training_state, vec_training_state
    walking = DummyVecEnv([lambda: G1VelocityEnv()])
    getup = DummyVecEnv([lambda: G1GetupEnv(bank=small_bank, assist_force=120.)] * 2)
    try:
        assert not keeps_training_state(walking) and vec_training_state(walking) is None
        assert keeps_training_state(getup) and vec_training_state(getup) == {'assist_force': [120., 120.]}
    finally:
        walking.close(), getup.close()

    class Batched:  # the Warp Humanoid VecEnv answers None for any attribute
        def has_attr(self, name):
            return True

        def get_attr(self, name):
            return [None]
    assert not keeps_training_state(Batched())


def test_body_friction_reaches_every_colliding_robot_geom_and_the_floor(small_bank):
    env = G1GetupEnv(bank=small_bank)
    try:
        model = env.model
        colliding = [g for g in range(model.ngeom) if env.robot_geoms[g] and (model.geom_contype[g]
                                                                             or model.geom_conaffinity[g])]
        assert sorted(env.friction_geoms.tolist()) == sorted(colliding + [model.geom('floor').id])
        env.set_body_friction(.45)
        assert (model.geom_friction[env.friction_geoms, 0] == .45).all()
        others = np.setdiff1d(np.arange(model.ngeom), env.friction_geoms)
        np.testing.assert_array_equal(model.geom_friction[others, 0], env.nominal_friction[others])
        mujoco.mj_forward(model, env.data)
    finally:
        env.close()
