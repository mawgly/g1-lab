"""task_config arm_targets (iteration 40, stage 3 "arms to targets"), CPU: the arm command in the upper-body slot, the
residual arm action on home moved by the targets, the episode modes, the target generator and its pose libraries, the
hand loads and their curve, the reward terms, the contracts, warm_start.unblind_arms (scripts/lab/arm_branch.py).
Off, and in every flag-0 episode (resets with a command, a free-arms training episode), the task is bit for bit the old
one (the walking digests of the code before the setting)."""
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import ARM_POSES, G1VelocityEnv, arm_pose_library, saved_load
from humanoid_sim.tasks.identity import task_identity

ROOT = Path(__file__).resolve().parents[1]
# Candidate 10 (exp-fa2450e3a6af4bf4 @ 8784248832), the parent of iteration 40's branch.
STAGE_10 = ROOT / 'runs/humanoid_lab/exp-fa2450e3a6af4bf4/stage_008784248832'
# Candidate 10's task without arm_blind, with a 20 ms delay so that the queue runs.
RECIPE = dict(
    reward_weights=dict(velocity=2., yaw=2., yaw_drift=5., alive=.15, gait=.5, swing=10., tilt=0., orientation=2.,
                        height=10., height_ceiling=2., torso=1., base_motion=2., still=3., stand_pose=1., slip=1.,
                        feet_flat=2., self_contact=1., arm_press=.1, posture=.3, hips=1., effort=1e-5,
                        action_change=.05, fall=100.),
    heading_hold=True,
    commands=dict(min_speed=-1., max_speed=2., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                  straight_probability=.5, side_share=.2, side_stop=1., side_speed=1., crouch_speed=1.,
                  crouch_share=[.5, .33], crouch_min=[.64, .7]),
    gait=dict(period=.6, stance_fraction=.55), action_limit=10., real_robot=True, critic_extras=True, history=5,
    stand_heading_hold=False, phase_stop=True, sole_contact=True, random_phase=True, action_delay=.02,
    arm_perturb=.25, arm_perturb_radians=1., arm_perturb_contact_mask=True, lock_wrists=True, knee_floor=.061,
    force_push=1.25, foot_collision=True, side_heading_hold=False, motor_friction=1.25, height_command=True,
    upper_body_slot=True)
ARMS = {**RECIPE, 'arm_targets': True, 'reward_weights': {**RECIPE['reward_weights'], 'arm_track': 2.}}
COMMANDS = ([.5, 0., 0.], [0., 0., 0.], [1., 0., .3], [0., .6, 0.])
# walking_digest of the code before arm_targets (commit d89ace4) with RECIPE: training resets and resets with a
# command. Another MuJoCo, numpy or torch build changes them: then make them again with that commit's code.
OLD_DIGESTS = {'training': 'c93e0ebdcc3f44afb2ad0242c37ba33e37d975182fdb8844bff7cff0bc191c38',
               'command': '5a7011e1306a3a88755431a8bd1869c52a60e2ed129cccfd0ef3f6dbc81f954a'}
TRACKED = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)  # shoulders, elbows, wrist roll
WRISTS = (20, 21, 27, 28)
ARM_JOINTS = tuple(range(15, 29))


def walking_digest(config, commanded, seed=40, steps=300, free=False):
    """sha256 of every observation, reward, end flag, qpos, observed command and reward term but arm_track and
    arm_error (terms added later; arm_error 0 without its weight) of a few
    episodes of 60 steps under random actions over the ±10 box: training resets or resets with a command. free: after
    each reset the episode's arms are set free (set_arm_targets(None, 0)) and its loads off, the reset's observation
    read again (a flag-0 training episode), with a load ceiling of 1.5 kg so that flag-1 resets do load the hands."""
    env = G1VelocityEnv(max_episode_steps=60, **config)
    if free:
        env.load_max = 1.5
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    episode = 0

    def start(k):
        options = {'command': COMMANDS[episode % len(COMMANDS)]} if commanded else None
        obs, _ = env.reset(seed=seed + k, options=options)
        if free:
            env.set_arm_targets(None, 0)
            env.set_hand_load(0.)
            obs = env._observation('fill')
        digest.update(obs.tobytes())

    try:
        start(0)
        for k in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-10., 10., 29) * (.3 if k % 7 else 1.))
            digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated])
                          + env.data.qpos.tobytes() + np.asarray(info['command'], dtype=np.float64).tobytes())
            for key in sorted(info['reward_components']):
                if key not in ('arm_track', 'arm_error'):
                    digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
            if terminated or truncated:
                episode += 1
                start(k + 1)
    finally:
        env.close()
    return digest.hexdigest()


def columns(env):
    """(slot columns per frame (5, 18), newest last action (29,)) of the observation."""
    slot = np.array(task.term_positions(env.model.nu, env.history, env.extra_terms, task.UPPER_BODY_TERM))
    return slot.reshape(env.history, task.UPPER_BODY_SIZE), np.array(
        task.newest_frame_columns(env.model.nu, env.history, env.extra_sizes, term=6))


def test_the_setting_its_validation_contracts_and_labels():
    from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS, G1_REWARD_WEIGHTS, g1_config
    field = next(p for p in G1_PARAMETERS if p['key'] == 'arm_targets')
    assert field['default'] is False and field['type'] == 'bool' and 'Руки по целям' in field['label']
    assert G1_REWARD_WEIGHTS['arm_track'] == 0 and 'exp' in G1_LABELS['arm_track']
    assert next(p for p in G1_PARAMETERS if p['key'] == 'arm_track')['default'] == 0
    assert g1_config({'real_robot': True})['arm_targets'] is False
    base = {'real_robot': True, 'upper_body_slot': True, 'lock_wrists': True}
    assert g1_config({**base, 'arm_targets': True})['arm_targets'] is True
    for values in ({'arm_targets': True}, {**base, 'arm_targets': 1}, {**base, 'upper_body_slot': False,
                   'arm_targets': True}, {**base, 'lock_wrists': False, 'arm_targets': True},
                   {**base, 'arm_blind': True, 'arm_targets': True}):
        with pytest.raises(ValueError):
            g1_config(values)
    for values in ({'arm_targets': True}, {**base, 'lock_wrists': False, 'arm_targets': True}):
        with pytest.raises(ValueError):
            G1VelocityEnv(**values)
    plain = task_identity('g1_29dof', 'g1_velocity', RECIPE)
    assert task_identity('g1_29dof', 'g1_velocity', {**RECIPE, 'arm_targets': False}) == plain
    assert plain['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hcmd-ubslot-critic4'
    assert plain['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock-knee0.061'
    arms = task_identity('g1_29dof', 'g1_velocity', ARMS)
    assert arms['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-phasestop-hcmd-ubslot-armtgt-critic4'
    assert arms['action_schema'] == 'g1_velocity:action-v2-limit10-wristlock-knee0.061-armres'
    assert {k: v for k, v in arms.items() if k not in ('observation_schema', 'action_schema')} == \
        {k: v for k, v in plain.items() if k not in ('observation_schema', 'action_schema')}
    with pytest.raises(ValueError):
        validate_resume(plain, arms)


def test_off_and_free_arms_are_bit_for_bit_the_old_task():
    """No setting: training resets and resets with a command give the code's before the setting (OLD_DIGESTS). With the
    setting, resets with a command (flag 0) give the same, and so do training episodes set free after the reset: the
    arm process draws from its own stream, so the task's own draws are the old ones."""
    assert walking_digest(RECIPE, False) == OLD_DIGESTS['training']
    assert walking_digest(RECIPE, True) == OLD_DIGESTS['command']
    assert walking_digest(ARMS, True) == OLD_DIGESTS['command']
    assert walking_digest(ARMS, False, free=True) == OLD_DIGESTS['training']


def test_training_resets_draw_the_modes_and_resets_with_a_command_keep_free_arms():
    env = G1VelocityEnv(max_episode_steps=50, **ARMS)
    try:
        env.load_max = 1.
        counts, loads = np.zeros(3), []
        for seed in range(600):
            env.reset(seed=seed)
            counts[env.arm_mode] += 1
            assert env.arm_flag == (env.arm_mode != task.ARM_FREE)
            assert env.perturb_live == (env.arm_mode == task.ARM_FREE)  # no arm perturbation with flag 1
            if env.arm_flag:
                loads.append(env.hand_load.copy())
            else:
                assert not env.hand_load.any() and not env.arm_offset.any()
            if env.arm_mode == task.ARM_HOME:
                assert not env.arm_offset.any()
        np.testing.assert_allclose(counts / counts.sum(), task.ARM_MODES, atol=.05)
        loads = np.array(loads)
        assert loads.min() >= 0 and loads.max() <= 1. and loads.mean() == pytest.approx(.5, abs=.05)
        assert abs(np.corrcoef(loads.T)[0, 1]) < .15  # each hand its own share
        for command in COMMANDS:
            env.reset(seed=3, options={'command': command})
            assert env.arm_mode == task.ARM_FREE and not env.arm_flag and not env.hand_load.any()
    finally:
        env.close()


def test_the_slot_and_the_observed_last_action_carry_the_targets():
    """Flag 1: the slot of every frame is [1, waist 0, the arm offsets with the locked wrists 0], the newest frame shows
    the targets of the next step; the last action observed is clip(a) + d/scale of the step's own targets (what the ONNX
    graph outputs), the other joints' as applied; flag 0: the slot zeros, the last action the applied one."""
    env = G1VelocityEnv(**ARMS)
    try:
        slot, last = columns(env)
        obs, _ = env.reset(seed=1, options={'command': [0., 0., 0.]})
        assert not obs[slot].any()
        target = env.home.copy()
        target[list(TRACKED)] += np.linspace(-.6, .6, 10)
        target[list(WRISTS)] = .5  # locked: their offsets stay 0
        env.set_arm_targets(target, 1)
        obs = env._observation()
        d = env.arm_offset.copy()
        np.testing.assert_allclose(d[list(TRACKED)], np.linspace(-.6, .6, 10), atol=1e-12)
        assert not d[list(WRISTS)].any() and not d[:15].any()
        newest = obs[slot[-1]]
        assert newest[0] == 1. and not newest[1:4].any()
        np.testing.assert_allclose(newest[1:], d[task.UPPER_BODY_FIRST:].astype(np.float32), atol=1e-7)
        action = np.random.default_rng(2).uniform(-3., 3., 29)
        obs, *_ = env.step(action)
        applied = env.applied_action(action)
        shown = applied.copy()
        shown[list(ARM_JOINTS)] += d[list(ARM_JOINTS)] / env.scale[list(ARM_JOINTS)]
        np.testing.assert_allclose(obs[last], shown.astype(np.float32), rtol=1e-6, atol=1e-6)
        assert (obs[slot][:, 0] == 1).sum() == 2  # the reset frames were filled before the targets came
        env.set_arm_targets(None, 0)
        obs = env._observation()  # free again: the slot zeros, the last action still what the flag-1 step output
        assert not obs[slot[-1]].any()
        np.testing.assert_allclose(obs[last], shown.astype(np.float32), rtol=1e-6, atol=1e-6)
        obs, *_ = env.step(action)  # a flag-0 step: its last action is the applied one
        np.testing.assert_array_equal(obs[last], applied.astype(np.float32))
    finally:
        env.close()


def test_the_arm_action_is_a_residual_on_home_moved_by_the_targets_also_in_the_delay_queue():
    """Flag 1: the PD target of an arm joint is clip(home + d + 0.44·clip(a, ±10), range); with a delay the queue keeps
    each step's action with its own offsets, so the substeps still following the step before use that step's targets."""
    env = G1VelocityEnv(**ARMS)
    try:
        env.reset(seed=4, options={'command': [0., 0., 0.], 'action_delay_steps': 5})
        first = env.home.copy()
        first[list(TRACKED)] += .4
        env.set_arm_targets(first, 1)
        action = np.full(29, 2.5)
        action[15] = 30.  # beyond the box
        d1 = env.arm_offset.copy()
        expected = np.clip(env.home + d1 + env.scale * np.clip(action, -10, 10), env.target_low, env.target_high)
        target = env._target(env.applied_action(action), d1)
        np.testing.assert_allclose(target[list(TRACKED)], expected[list(TRACKED)], atol=1e-12)
        np.testing.assert_array_equal(target[list(WRISTS)], env.home[list(WRISTS)])  # locked: at home
        env.step(action)
        assert len(env.delay_actions) == 1
        np.testing.assert_array_equal(env.delay_offsets[0], d1)
        second = env.home.copy()
        second[list(TRACKED)] -= .3
        env.set_arm_targets(second, 1)
        calls = []
        original = env._target

        def spy(action, offset=None):
            calls.append((action.copy(), None if offset is None else offset.copy()))
            return original(action, offset)
        env._target = spy
        env.step(np.zeros(29))
        assert len(calls) == 2  # this step's target, then the delayed one
        np.testing.assert_array_equal(calls[0][1], env.previous_offset)
        np.testing.assert_allclose(calls[0][1][list(TRACKED)], -.3, atol=1e-12)
        np.testing.assert_array_equal(calls[1][0], env.applied_action(action))  # the step before's own action
        np.testing.assert_array_equal(calls[1][1], d1)  # with its own targets
    finally:
        env.close()


def test_targets_move_by_a_quintic_and_the_generator_holds_and_draws_segments():
    s = torch.linspace(0., 1., 11, dtype=torch.float64)
    q = task.quintic(s)
    assert float(q[0]) == 0 and float(q[-1]) == 1 and float(task.quintic(torch.tensor([.5]))[0]) == .5
    h = 1e-6
    slope = (task.quintic(torch.tensor([h, 1 - h], dtype=torch.float64)) - torch.tensor([0., 1.],
                                                                                         dtype=torch.float64)) / h
    assert float(slope.abs().max()) < 1e-4  # zero speed at both ends
    start, end = torch.zeros((2, 29), dtype=torch.float64), torch.zeros((2, 29), dtype=torch.float64)
    end[0, 15], end[1, 15] = .75, -3.
    assert task.arm_move_steps(start, end, TRACKED, .02).tolist() == [50, 100]  # 1 s at least, else |Δ|/1.5 rad/s
    env = G1VelocityEnv(**ARMS)
    try:
        env.reset(seed=5, options={'command': [0., 0., 0.]})
        target = env.home.copy()
        target[list(TRACKED)] += .9
        env.set_arm_targets(target, 1, seconds=1.)
        seen = [env.arm_offset[15]]
        for _ in range(60):
            env.step(np.zeros(29))
            seen.append(env.arm_offset[15])
        expected = .9 * task.quintic(torch.arange(61, dtype=torch.float64) / 50).numpy()
        np.testing.assert_allclose(seen, expected, atol=1e-12)
        # The training generator: segments from home, quintics of their own length, holds of 1-8 s.
        env.reset(seed=6)
        while env.arm_mode != task.ARM_TARGET:
            env.reset()
        segments = []
        for _ in range(1500):
            start, end, clock, move, hold = (value.clone() for value in env.arm_state)
            env._advance_arm_targets()
            if int(env.arm_state[2][0]) == 0:
                segments.append((start, end, int(move[0]), int(hold[0]), int(clock[0])))
                np.testing.assert_array_equal(env.arm_state[0], end)  # the next one starts where it ended
        assert len(segments) >= 5
        for start, end, move, hold, clock in segments:
            assert clock + 1 == move + hold and 50 <= hold <= 400
            delta = float((end - start)[0, list(TRACKED)].abs().max())
            assert move == max(50, round(delta / (1.5 * .02)))
    finally:
        env.close()


def test_pose_libraries_kept_poses_touch_nothing_and_carry_poses_are_mirrored():
    """The libraries (humanoid_sim/tasks/data/g1_arm_poses.npz, scripts/lab/arm_pose_library.py): the ten free arm
    joints in their ranges, the carry poses within task.ARM_CARRY and mirror images of themselves (symmetry's joint
    map), samples of both touch nothing in the pose and halfway from home; arm_pose_draw picks a carry pose 40 % of
    the time."""
    import sys
    sys.path.insert(0, str(ROOT / 'scripts/lab'))
    from arm_pose_library import Touch
    from humanoid_sim.symmetry import model_joint_mirror
    names, poses, carry, ranges = arm_pose_library(ARM_POSES)
    assert len(poses) == 16384 and len(carry) == 8192 and poses.shape[1] == carry.shape[1] == 10
    touch = Touch()
    try:
        assert list(names) == touch.names
        assert (poses >= touch.low - 1e-6).all() and (poses <= touch.high + 1e-6).all()
        np.testing.assert_array_equal(ranges, [row[1:] for row in task.ARM_CARRY])
        for k, (part, low, high) in enumerate(task.ARM_CARRY):
            left = carry[:, names.index(f'left_{part}_joint')]
            assert left.min() >= low - 1e-6 and left.max() <= high + 1e-6
        permutation, signs = model_joint_mirror(touch.model, touch.env.joints, touch.env.qadr, touch.env.home)
        full = np.tile(touch.home, (len(carry), 1))
        full[:, touch.free] = carry
        np.testing.assert_allclose(full[:, permutation] * signs, full, atol=1e-6)  # its own mirror image
        rng = np.random.default_rng(0)
        for rows in (poses, carry):
            for row in rows[rng.choice(len(rows), 60, replace=False)]:
                assert touch.keeps(row.astype(np.float64))
    finally:
        touch.close()
    library = torch.zeros((4, 29), dtype=torch.float64)
    library[:, 15] = 1.
    carried = torch.zeros((4, 29), dtype=torch.float64)
    carried[:, 15] = 2.
    rng = np.random.default_rng(1)
    drawn = task.arm_pose_draw(lambda n: torch.from_numpy(rng.random(n)), 20000, library, carried)
    assert float((drawn[:, 15] == 2.).double().mean()) == pytest.approx(task.ARM_CARRY_SHARE, abs=.01)


def test_reward_terms_with_flag_one_and_flag_zero():
    """arm_track = w·mean over the ten free arm joints of exp(−e²/0.3²) with flag 1, 0 with flag 0; posture leaves the
    arm joints out and arm_press is 0 with flag 1; flag 0 keeps both bit for bit; without arm_flag the old terms."""
    n = 2
    generator = torch.Generator().manual_seed(0)
    home = torch.zeros(29, dtype=torch.float64)
    position = torch.rand((n, 29), generator=generator, dtype=torch.float64) - .5
    target = home + torch.rand((n, 29), generator=generator, dtype=torch.float64) - .5
    args = (torch.zeros((n, 6), dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.zeros((n, 3), dtype=torch.float64),
            torch.zeros((n, 29), dtype=torch.float64), torch.zeros((n, 29), dtype=torch.float64),
            torch.zeros((n, 29), dtype=torch.float64), torch.full((n, 2), .034, dtype=torch.float64),
            torch.zeros(n, dtype=torch.float64), position, home, torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    weights = {**task.G1_REWARD_WEIGHTS, 'arm_track': 2., 'posture': .3, 'arm_press': .1}
    press = torch.tensor([.7, .9], dtype=torch.float64)
    posture = task.POSTURE_JOINTS
    without = tuple(j for j in posture if j not in ARM_JOINTS)
    tracked = task.arm_track(position, target, TRACKED)
    expected = torch.exp(-(position - target)[:, list(TRACKED)] ** 2 / .09).mean(dim=1)
    torch.testing.assert_close(tracked, expected, rtol=0, atol=1e-15)
    old, _ = task.reward_components(*args, weights=weights, arm_press=press)
    new, _ = task.reward_components(*args, weights=weights, arm_press=press, arm_flag=torch.tensor([True, False]),
                                    arm_track=tracked, arm_posture_joints=without)
    assert float(new['arm_track'][0]) == pytest.approx(2 * float(tracked[0]), abs=1e-15)
    assert float(new['arm_track'][1]) == 0 and float(old['arm_track'].abs().max()) == 0
    assert float(new['arm_press'][0]) == 0 and float(new['arm_press'][1]) == float(old['arm_press'][1])
    assert float(new['posture'][0]) == pytest.approx(-.3 * float((position[0, list(without)] ** 2).sum()), abs=1e-15)
    assert float(new['posture'][1]) == float(old['posture'][1])
    for key in set(old) - {'arm_track', 'arm_press', 'posture'}:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        task.reward_components(*args, weights=weights, arm_flag=torch.tensor([True, False]))
    env = G1VelocityEnv(**ARMS)
    try:  # the CPU task hands them over: its arm_track is the tracking of home + d after the step
        env.reset(seed=7, options={'command': [0., 0., 0.]})
        goal = env.home.copy()
        goal[15], goal[22] = -.5, -.5
        env.set_arm_targets(goal, 1)
        _, _, _, _, info = env.step(np.zeros(29))
        q = env.data.qpos[env.qadr]
        e = (q - goal)[list(TRACKED)]
        assert info['reward_components']['arm_track'] == pytest.approx(2 * np.exp(-e ** 2 / .09).mean(), rel=1e-12)
        assert info['reward_components']['arm_press'] == 0 and info['arm_error'] == pytest.approx(np.abs(e).mean())
        assert info['reward_components']['posture'] == pytest.approx(
            -.3 * ((q - env.home)[list(env.arm_posture_joints)] ** 2).sum(), rel=1e-12)
    finally:
        env.close()


def test_no_arm_perturbation_while_the_arms_follow_targets():
    env = G1VelocityEnv(max_episode_steps=200, **{**ARMS, 'arm_perturb': .5})
    try:
        episodes, perturbed = {True: 0, False: 0}, {True: 0, False: 0}
        for seed in range(60):
            env.reset(seed=seed)
            episodes[env.arm_flag] += 1
            for _ in range(120):
                env.step(np.zeros(29))
                if env.arm_flag:
                    assert env.perturb_steps == 0 and not env.arm_contacts_masked
                perturbed[env.arm_flag] += env.perturb_steps > 0
            if min(episodes.values()) >= 4:
                break
        assert min(episodes.values()) >= 4 and perturbed[True] == 0 and perturbed[False] > 0
    finally:
        env.close()


def test_hand_loads_move_the_mass_the_com_and_the_subtree_masses():
    env = G1VelocityEnv(**ARMS)
    try:
        model = env.model
        mass, ipos, subtree = model.body_mass.copy(), model.body_ipos.copy(), model.body_subtreemass.copy()
        hands = [model.body(name).id for name in task.HAND_LOAD_BODIES]
        env.reset(seed=1, options={'command': [0., 0., 0.]})
        env.set_hand_load([1., 2.])
        for hand, load in zip(hands, (1., 2.)):
            assert model.body_mass[hand] == pytest.approx(mass[hand] + load)
            np.testing.assert_allclose(model.body_ipos[hand], (mass[hand] * ipos[hand] + load * np.array(
                task.HAND_LOAD_POSITION)) / (mass[hand] + load), atol=1e-15)
        assert model.body_subtreemass[model.body('pelvis').id] == pytest.approx(subtree[model.body('pelvis').id] + 3)
        assert model.body_subtreemass[0] == pytest.approx(subtree[0] + 3)
        assert model.body_subtreemass[model.body('left_elbow_link').id] == pytest.approx(
            subtree[model.body('left_elbow_link').id] + 1)
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        mujoco.mj_forward(model, data)
        root = model.body('pelvis').id
        weights = model.body_mass[1:, None]
        np.testing.assert_allclose(data.subtree_com[root], (weights * data.xipos[1:]).sum(0) / weights.sum(), atol=1e-12)
        env.set_hand_load(0.)
        for field, saved in ((model.body_mass, mass), (model.body_ipos, ipos), (model.body_subtreemass, subtree)):
            np.testing.assert_array_equal(field, saved)
        for bad in (-1., np.nan, [1., 2., 3.]):
            with pytest.raises(ValueError):
                env.set_hand_load(bad)
    finally:
        env.close()
    plain = G1VelocityEnv(real_robot=True)  # the CPU task takes loads without arm_targets too; a reset takes them off
    try:
        plain.reset(seed=0)
        plain.set_hand_load(1.)
        assert plain.model.body_mass[hands[0]] == pytest.approx(mass[hands[0]] + 1)
        plain.reset(seed=0)
        assert plain.model.body_mass[hands[0]] == mass[hands[0]] and not plain.hand_load.any()
    finally:
        plain.close()


def test_the_load_curve_steps_within_its_bounds_and_the_ceiling_is_kept_and_restored():
    update = task.load_curve_update
    steps = 75000  # flag-1 world-steps of a window: 75 episodes of 1000 steps
    assert update(0., steps, 0, .7, 1000) == .1  # no fall, tracking at 0.7 of the weight: up
    assert update(.5, steps, 0, .59, 1000) == .5  # tracking too poor: stays
    assert update(.5, steps, 1, .9, 1000) == .5  # 1.3 % of the episodes fall: neither
    assert update(.5, steps, 3, .9, 1000) == .4  # 4 %: down
    assert update(2., steps, 0, .9, 1000) == 2. and update(0., steps, 5, .9, 1000) == 0.
    assert update(1.3, 0, 0, float('nan'), 1000) == 1.3  # no flag-1 step in the window
    level = 0.
    for _ in range(25):
        level = update(level, steps, 0, .9, 1000)
    assert level == 2.
    env = G1VelocityEnv(**ARMS)
    try:
        assert env.keeps_training_state and env.training_state() == {'load_max': [0.]}
        env.restore_training_state({'load_max': [1.2, 1.4]})
        assert env.load_max == pytest.approx(1.3) and env.training_state() == {'load_max': [env.load_max]}
        for seed in range(30):
            env.reset(seed=seed)
            assert (env.hand_load <= 1.3 + 1e-12).all()
            np.testing.assert_allclose(env.hand_load, env.load_share * 1.3)
        for bad in ({'load_max': [-1.]}, {'load_max': [float('nan')]}, {'load_max': []}):
            with pytest.raises(ValueError):
                saved_load(bad)
        assert saved_load({'load_max': [5.]}) == task.LOAD_LIMIT
    finally:
        env.close()
    plain = G1VelocityEnv(**RECIPE)
    try:
        assert not plain.keeps_training_state
    finally:
        plain.close()


def test_fixed_normalization_of_the_arm_columns_and_the_mirror_of_the_slot():
    from humanoid_sim.symmetry import Mirror
    env = G1VelocityEnv(**ARMS)
    try:
        positions, mean, variance = env.fixed_normalization
        arms = env.arm_columns
        assert len(arms) == 210 and positions[-210:].tolist() == arms and not mean[-210:].any()
        assert variance[-210:].tolist() == [task.ARM_TARGET_VARIANCE[term] for term in (4, 5, 6) for _ in range(70)]
        assert [task.ARM_TARGET_VARIANCE[term] for term in (4, 5, 6)] == pytest.approx([.36, 4., 2.25])
        plain = G1VelocityEnv(**RECIPE)
        try:
            np.testing.assert_array_equal(positions[:-210], plain.fixed_normalization[0])
        finally:
            plain.close()
        mirror = Mirror(env)
        env.reset(seed=8, options={'command': [0., 0., 0.]})
        env.set_arm_targets(env.home + env.arm_carry[3].numpy(), 1)  # a symmetric carry pose
        obs = torch.from_numpy(env._observation()).double()
        slot, _ = columns(env)
        torch.testing.assert_close(mirror.observation(obs)[slot[-1]], obs[slot[-1]], rtol=0, atol=1e-6)
    finally:
        env.close()


def _stage(path, task_config):
    """A small real-robot stage under run dir `path` (trained 16 steps, statistics off the defaults)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import save_stage, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', n_steps=8, batch_size=8,
                                  task_config=task_config))
    write_json(path / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1VelocityEnv(**config['task_config'])]))
    try:
        size = env.observation_space.shape[0]
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': size - 4})
        model.learn(16)
        env.obs_rms.mean[:] = np.random.default_rng(1).normal(0, .3, size)
        with torch.no_grad():
            model.policy.action_net.weight.mul_(10)
        return config, save_stage(path, model, env)
    finally:
        env.close()


def _run(path, config, **changes):
    from humanoid_sim.core import write_json
    path.mkdir()
    task_config = {**config['task_config'], **changes}
    write_json(path / 'run_config.json', {**config, 'task_config': task_config})
    return path


SMALL = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
             lock_wrists=True, height_command=True, upper_body_slot=True)


def test_unblind_arms_folds_the_blind_constant_into_the_biases_and_frees_the_arm_columns(tmp_path):
    """A blind stage moved by unblind_arms: validate_resume takes it under arm_targets only; the first layers' arm
    columns are 0 and their biases b + W·z, so it acts as its source (start_difference ≤ 1e-5) and random arm inputs
    change nothing; the arm columns' statistics are the fixed ones, the rest and the count the source's; Adam's moments
    of those columns m 0, v the row mean of the others; steps kept. Refused: a source that is not blind, a task without
    arm_targets, any other contract change; carry_over and insert_terms do not switch arm_targets."""
    from humanoid_sim.checkpoint_schema import checkpoint_identity
    from humanoid_sim.core import load_stage, read_json
    from humanoid_sim.warm_start import (blind_arms, carry_over, check_insert_terms, first_layers, start_difference,
                                         unblind_arms)
    (tmp_path / 'old').mkdir()
    config, plain = _stage(tmp_path / 'old', SMALL)
    z = np.random.default_rng(7).uniform(-1, 1, 210)
    blind = blind_arms(plain, _run(tmp_path / 'blind', config, arm_blind=True), z)
    blind_config = read_json(tmp_path / 'blind' / 'run_config.json')
    run = _run(tmp_path / 'arms', blind_config, arm_blind=False, arm_targets=True)
    moved = unblind_arms(blind, run)
    metadata = read_json(moved / 'metadata.json')
    arms = task_identity('g1_29dof', 'g1_velocity', {**config['task_config'], 'arm_targets': True})
    validate_resume(checkpoint_identity(metadata), arms)
    assert metadata['observation_schema'].endswith('-ubslot-armtgt-critic4')
    assert metadata['action_schema'].endswith('-wristlock-armres') and metadata['steps'] == read_json(
        blind / 'metadata.json')['steps']
    warm = read_json(moved / 'warm_start.json')
    np.testing.assert_allclose(warm['z'], z, rtol=0, atol=1e-12)
    assert start_difference(blind, moved) < 1e-5
    (before, before_norm), (after, after_norm) = load_stage(blind), load_stage(moved)
    try:
        columns = np.asarray(after_norm.venv.envs[0].unwrapped.arm_columns)
        positions, mean, variance = after_norm.venv.envs[0].unwrapped.fixed_normalization
        np.testing.assert_array_equal(positions[-210:], columns)
        np.testing.assert_array_equal(after_norm.obs_rms.mean[columns], mean[-210:])
        np.testing.assert_array_equal(after_norm.obs_rms.var[columns], variance[-210:])
        others = np.setdiff1d(np.arange(len(after_norm.obs_rms.mean)), columns)  # the extra terms' too: the source's
        np.testing.assert_array_equal(after_norm.obs_rms.mean[others], before_norm.obs_rms.mean[others])
        assert after_norm.obs_rms.count == before_norm.obs_rms.count
        old, new = before.policy.state_dict(), after.policy.state_dict()
        layers = first_layers(after.policy, len(after_norm.obs_rms.mean))
        assert len(layers) == 2
        for name in layers:
            assert not new[name][:, columns].any()
            bias = name[:-len('weight')] + 'bias'
            expected = old[bias].double() + old[name][:, columns].double() @ torch.from_numpy(z)
            torch.testing.assert_close(new[bias].double(), expected, rtol=0, atol=1e-6)
        for name in set(old) - set(layers) - {name[:-len('weight')] + 'bias' for name in layers}:
            torch.testing.assert_close(new[name], old[name], rtol=0, atol=0, msg=name)
        state, names = after.policy.optimizer.state_dict()['state'], [n for n, _ in after.policy.named_parameters()]
        for number, moments in state.items():
            if names[number] in layers:
                assert not moments['exp_avg'][:, columns].any()
                rest = np.setdiff1d(np.arange(moments['exp_avg_sq'].shape[1]), columns)
                torch.testing.assert_close(moments['exp_avg_sq'][:, columns[0]],
                                           moments['exp_avg_sq'][:, rest].mean(dim=1))
        raw = after_norm.venv.envs[0].unwrapped
        obs, _ = raw.reset(seed=3, options={'command': [.5, 0., 0.]})
        reference = after.predict(after_norm.normalize_obs(obs), deterministic=True)[0]
        rng = np.random.default_rng(4)
        for _ in range(5):
            other = obs.copy()
            other[columns] = rng.uniform(-5, 5, len(columns))
            assert np.abs(after.predict(after_norm.normalize_obs(other), deterministic=True)[0] - reference).max() <= 1e-5
    finally:
        before_norm.close(), after_norm.close()
    refused = [(plain, config, dict(arm_targets=True)),  # not blind
               (blind, blind_config, dict(arm_blind=False)),  # no arm_targets
               (blind, blind_config, dict(arm_blind=False, arm_targets=True, knee_floor=.061))]  # another change
    for number, (source, base, changes) in enumerate(refused):
        target = _run(tmp_path / f'refused{number}', base, **changes)
        with pytest.raises(ValueError):
            unblind_arms(source, target)
        assert not any(target.glob('stage_*'))
    with pytest.raises(ValueError, match='arm_branch'):
        carry_over(plain, _run(tmp_path / 'carried', config, arm_targets=True))
    with pytest.raises(ValueError, match='arm_branch'):
        check_insert_terms(task_identity('g1_29dof', 'g1_velocity', {**SMALL, 'height_command': False,
                                                                     'upper_body_slot': True}),
                           task_identity('g1_29dof', 'g1_velocity', {**SMALL, 'arm_targets': True}))


@pytest.mark.skipif(not (STAGE_10 / 'metadata.json').exists(), reason='candidate 10 is not on this machine')
def test_unblinding_candidate_10_starts_as_candidate_10(tmp_path):
    """Candidate 10 moved by unblind_arms (as arm_branch.py does): on its reference runs (standing, 0.5 and 1 m/s) the
    actions are its own within 1e-5."""
    from humanoid_sim.core import read_json, write_json
    from humanoid_sim.lab import validate_config
    from humanoid_sim.warm_start import start_difference, unblind_arms
    parent = read_json(STAGE_10.parent / 'run_config.json')
    task_config = {**parent['task_config'], 'arm_blind': False, 'arm_targets': True,
                   'reward_weights': {**parent['task_config']['reward_weights'], 'arm_track': 2.}}
    write_json(tmp_path / 'run_config.json', validate_config({**{k: v for k, v in parent.items()
                                                                 if k != 'task_config'}, 'task_config': task_config}))
    moved = unblind_arms(STAGE_10, tmp_path)
    assert start_difference(STAGE_10, moved) < 1e-5
