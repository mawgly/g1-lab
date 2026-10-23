import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.catalog import make_task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity


def test_legs_only_robot_has_twelve_actuators_and_its_own_contract():
    env = make_task('g1_velocity', 'g1_12dof', {})
    try:
        raw = env.unwrapped
        obs, _ = env.reset(seed=1)
        assert env.action_space.shape == (12,) and obs.shape == (14 + 3 * 12,)
        assert [raw.model.joint(j).name for j in raw.joints][:3] == ['left_hip_pitch_joint', 'left_hip_roll_joint',
                                                                       'left_hip_yaw_joint']
        assert raw.posture_joints == (1, 2, 7, 8) == raw.hip_joints == task.HIP_JOINTS
        assert raw.model.body('torso_link').id > 0  # welded upper body is still there for torso metrics
        full = G1VelocityEnv()
        try:
            assert full.posture_joints == task.POSTURE_JOINTS and full.hip_joints == task.HIP_JOINTS
            assert raw.model.body_mass.sum() == pytest.approx(full.model.body_mass.sum())
            np.testing.assert_array_equal(raw.home, full.home[:12])
        finally:
            full.close()
        for _ in range(3):
            env.step(np.zeros(12))
    finally:
        env.close()
    legs, full = task_identity('g1_12dof', 'g1_velocity'), task_identity('g1_29dof', 'g1_velocity')
    assert legs['model_revision'] != full['model_revision']
    with pytest.raises(ValueError):
        validate_resume(full, legs)


def test_orientation_and_height_terms_are_absent_unless_weighted():
    tilt = .3
    c, s = np.cos(tilt), np.sin(tilt)
    rotation = torch.tensor([[[c, 0., s], [0., 1., 0.], [-s, 0., c]]] * 2, dtype=torch.float64)
    n = 2

    def rewards(weights):
        return task.reward_components(
            torch.zeros(n, 6, dtype=torch.float64), rotation, torch.tensor([.78, .68], dtype=torch.float64),
            torch.tensor([[1., 0, 0]] * n, dtype=torch.float64), torch.zeros(n, 29), torch.zeros(n, 29),
            torch.zeros(n, 29), torch.tensor([[.034, .034]] * n), torch.tensor([.25, .25]),
            torch.zeros(n, 29), torch.zeros(29), rotation, weights=weights)[0]

    default = rewards(None)
    assert default['orientation'].abs().max() == 0 and default['height'].abs().max() == 0
    weighted = rewards({**task.G1_REWARD_WEIGHTS, 'orientation': 2., 'height': 20.})
    torch.testing.assert_close(weighted['orientation'], torch.full((2,), -2 * s * s, dtype=torch.float64))
    torch.testing.assert_close(weighted['height'], torch.tensor([0., -20 * .1 ** 2], dtype=torch.float64))


def test_forward_speed_reward_grows_with_speed_and_is_clipped():
    n = 4
    velocity = torch.zeros(n, 6, dtype=torch.float64)
    velocity[:, 3] = torch.tensor([.5, 2., 9., -3.])
    args = (velocity, torch.eye(3, dtype=torch.float64).repeat(n, 1, 1), torch.full((n,), .78, dtype=torch.float64),
            torch.tensor([[1., 0, 0]] * n, dtype=torch.float64), torch.zeros(n, 29), torch.zeros(n, 29),
            torch.zeros(n, 29), torch.tensor([[.034, .034]] * n), torch.full((n,), .25), torch.zeros(n, 29),
            torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    assert task.reward_components(*args)[0]['forward'].abs().max() == 0  # absent unless weighted
    sprint = task.reward_components(*args, weights={**task.G1_REWARD_WEIGHTS, 'forward': 3.})[0]['forward']
    torch.testing.assert_close(sprint, torch.tensor([1.5, 6., 3 * task.FORWARD_SPEED_CAP, -3.], dtype=torch.float64))


def test_forward_reward_counts_only_speed_along_the_target_course():
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    turned = torch.tensor([[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]], dtype=torch.float64)  # facing world +y
    velocity = torch.zeros(1, 6, dtype=torch.float64)
    velocity[0, 3] = 2.  # 2 m/s along the pelvis axis
    on_course = task.course_speed(velocity, turned, torch.tensor([np.pi / 2], dtype=torch.float64))
    off_course = task.course_speed(velocity, turned, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(on_course, torch.tensor([2.], dtype=torch.float64))
    assert abs(float(off_course)) < 1e-12  # sideways to the course: running in circles earns nothing


def test_running_gait_clock_and_sideways_penalty():
    phase = torch.tensor([.45, .2])
    command = torch.tensor([[1., 0, 0]] * 2)
    walking, _ = task.gait_targets(phase, command)
    running, _ = task.gait_targets(phase, command, stance_fraction=.4)
    assert walking[0].tolist() == [True, False] and running[0].tolist() == [False, False]  # flight phase
    assert running[1].tolist() == [True, False]
    env = G1VelocityEnv(gait={'period': .7, 'stance_fraction': .4},
                        reward_weights={'sideways': 3.})
    try:
        env.reset(seed=1, options={'command': [1., 0, 0]})
        env.steps = 35  # one full 0.7 s cycle at 50 Hz (the 0.8 s walking clock would be at 7/8)
        np.testing.assert_allclose(env._observation()[-2:], [0., 1.], atol=1e-6)
        _, _, _, _, info = env.step(np.zeros(29))
        assert info['reward_components']['sideways'] <= 0
    finally:
        env.close()
    velocity = torch.zeros(1, 6, dtype=torch.float64)
    velocity[0, 4] = .5
    sideways = task.reward_components(
        velocity, torch.eye(3, dtype=torch.float64)[None], torch.tensor([.78], dtype=torch.float64),
        torch.tensor([[1., 0, 0]], dtype=torch.float64), torch.zeros(1, 29), torch.zeros(1, 29), torch.zeros(1, 29),
        torch.tensor([[.034, .034]]), torch.tensor([.25]), torch.zeros(1, 29), torch.zeros(29),
        torch.eye(3, dtype=torch.float64)[None], weights={**task.G1_REWARD_WEIGHTS, 'sideways': 3.})[0]['sideways']
    torch.testing.assert_close(sideways, torch.tensor([-.75], dtype=torch.float64))


def test_heading_term_penalises_body_turned_off_course():
    rotations = torch.stack([torch.tensor([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]],
                                          dtype=torch.float64) for a in (0., .5, -3.)])
    n = 3
    args = (torch.zeros(n, 6, dtype=torch.float64), rotations, torch.full((n,), .78, dtype=torch.float64),
            torch.tensor([[1., 0, 0]] * n, dtype=torch.float64), torch.zeros(n, 29), torch.zeros(n, 29),
            torch.zeros(n, 29), torch.tensor([[.034, .034]] * n), torch.full((n,), .25), torch.zeros(n, 29),
            torch.zeros(29), rotations)
    assert task.reward_components(*args)[0]['heading'].abs().max() == 0  # absent unless weighted
    weighted = task.reward_components(*args, weights={**task.G1_REWARD_WEIGHTS, 'heading': 2.},
                                      heading_target=torch.full((n,), .5, dtype=torch.float64))[0]['heading']
    wrapped = (-3. - .5 + np.pi) % (2 * np.pi) - np.pi  # the short way round
    torch.testing.assert_close(weighted, torch.tensor([-2 * .25, 0., -2 * wrapped ** 2], dtype=torch.float64))


def test_hip_term_leaves_arms_free():
    position = torch.zeros(2, 29)
    position[0, 1] = .3   # hip roll: legs apart
    position[1, 15] = .3  # shoulder: arm swing
    args = (torch.zeros(2, 6), torch.eye(3).repeat(2, 1, 1), torch.full((2,), .78), torch.tensor([[1., 0, 0]] * 2),
            torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 29), torch.tensor([[.034, .034]] * 2),
            torch.full((2,), .25), position, torch.zeros(29), torch.eye(3).repeat(2, 1, 1))
    components = task.reward_components(*args, weights={**task.G1_REWARD_WEIGHTS, 'hips': 1.})[0]
    torch.testing.assert_close(components['hips'], torch.tensor([-.09, 0.]))
    assert task.reward_components(*args)[0]['hips'].abs().max() == 0  # absent unless weighted


def test_still_term_charges_slow_creep_only_when_told_to_stand():
    velocity = torch.zeros(3, 6, dtype=torch.float64)
    velocity[:, 3], velocity[:, 2] = .03, .05  # creeping 3 cm/s and turning 0.05 rad/s
    command = torch.tensor([[0., 0, 0], [1., 0, 0], [0., 0, .05]], dtype=torch.float64)  # stand, walk, stand + hold
    args = (velocity, torch.eye(3, dtype=torch.float64).repeat(3, 1, 1), torch.full((3,), .78, dtype=torch.float64),
            command, torch.zeros(3, 29), torch.zeros(3, 29), torch.zeros(3, 29), torch.tensor([[.034, .034]] * 3),
            torch.full((3,), .25), torch.zeros(3, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(3, 1, 1))
    assert task.reward_components(*args)[0]['still'].abs().max() == 0  # absent unless weighted
    components = task.reward_components(*args, weights={**task.G1_REWARD_WEIGHTS, 'still': 2.})[0]
    # Standing: 2·(0.03 + 0.05); walking: no charge; standing with a small heading-hold turn: only the creep.
    torch.testing.assert_close(components['still'], torch.tensor([-.16, 0., -.06], dtype=torch.float64))
    assert float(components['velocity'][0]) > 1.99  # the tracking term barely notices the same creep


def test_stand_pose_holds_the_legs_only_while_standing():
    position = torch.zeros(2, 29)
    position[:, 5] = .1   # left ankle roll off the default pose
    position[:, 20] = .5  # an arm: not part of this term
    args = (torch.zeros(2, 6), torch.eye(3).repeat(2, 1, 1), torch.full((2,), .78),
            torch.tensor([[0., 0, 0], [1., 0, 0]]), torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 29),
            torch.tensor([[.034, .034]] * 2), torch.full((2,), .25), position, torch.zeros(29),
            torch.eye(3).repeat(2, 1, 1))
    assert task.reward_components(*args)[0]['stand_pose'].abs().max() == 0  # absent unless weighted
    weighted = task.reward_components(*args, weights={**task.G1_REWARD_WEIGHTS, 'stand_pose': 2.})[0]['stand_pose']
    torch.testing.assert_close(weighted, torch.tensor([-2 * .1 ** 2, 0.]))  # standing: charged; walking: free


def test_slip_charges_feet_moving_on_the_ground_only():
    feet = torch.tensor([[.034, .2]])  # left on the ground, right in the air
    args = (torch.zeros(1, 6), torch.eye(3)[None], torch.full((1,), .78), torch.tensor([[1., 0, 0]]),
            torch.zeros(1, 29), torch.zeros(1, 29), torch.zeros(1, 29), feet, torch.full((1,), .25),
            torch.zeros(1, 29), torch.zeros(29), torch.eye(3)[None])
    speed = torch.tensor([[.3, 2.]])
    assert task.reward_components(*args, foot_speed=speed)[0]['slip'].abs().max() == 0  # absent unless weighted
    weights = {**task.G1_REWARD_WEIGHTS, 'slip': .2}
    torch.testing.assert_close(task.reward_components(*args, weights=weights, foot_speed=speed)[0]['slip'],
                               torch.tensor([-.2 * .3]))  # the swinging foot may move
    assert task.reward_components(*args, weights=weights)[0]['slip'].abs().max() == 0  # no speeds given


def test_cpu_env_reports_foot_slip():
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    env = G1VelocityEnv(reward_weights={'slip': 1.}, real_robot=True, action_limit=10.)
    try:
        env.reset(seed=1, options={'command': [0., 0., 0.]})
        rng = np.random.default_rng(0)
        slips = [env.step(rng.normal(0, 2., 29))[-1]['reward_components']['slip'] for _ in range(20)]
        assert min(slips) < 0 and all(np.isfinite(slips))  # thrashing legs slide the feet
    finally:
        env.close()


def _foot_pitch(angle):
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], dtype=torch.float64)


def test_feet_flat_charges_tilted_soles_on_both_feet():
    n = 3
    args = (torch.zeros(n, 6, dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.tensor([[1., 0, 0], [1., 0, 0], [0., 0, 0]]),
            torch.zeros(n, 29), torch.zeros(n, 29), torch.zeros(n, 29), torch.tensor([[.034, .2]] * n),
            torch.full((n,), .25), torch.zeros(n, 29), torch.zeros(29), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    flat, heel = torch.eye(3, dtype=torch.float64), _foot_pitch(np.radians(-50))  # toe up 50°
    roll = torch.tensor([[1., 0, 0], [0, np.cos(.3), -np.sin(.3)], [0, np.sin(.3), np.cos(.3)]], dtype=torch.float64)
    # Flat feet; left foot on its heel while walking (lifted or not, every step counts); both tilted while standing.
    feet = torch.stack([torch.stack([flat, flat]), torch.stack([heel, flat]), torch.stack([heel, roll])])
    weights = {**task.G1_REWARD_WEIGHTS, 'feet_flat': .5}
    assert task.G1_REWARD_WEIGHTS['feet_flat'] == 0
    assert task.reward_components(*args, foot_rotation=feet)[0]['feet_flat'].abs().max() == 0  # absent unless weighted
    torch.testing.assert_close(task.reward_components(*args, weights=weights, foot_rotation=feet)[0]['feet_flat'],
                               torch.tensor([0., -.5 * np.sin(np.radians(50)) ** 2,
                                             -.5 * (np.sin(np.radians(50)) ** 2 + np.sin(.3) ** 2)],
                                            dtype=torch.float64))
    assert task.reward_components(*args, weights=weights)[0]['feet_flat'].abs().max() == 0  # no foot frames given


def test_cpu_env_feet_flat_uses_foot_frames_and_home_pose_is_flat():
    import mujoco
    env = G1VelocityEnv(reward_weights={'feet_flat': 1.}, real_robot=True, action_limit=10.)
    try:
        env.reset(seed=0, options={'command': [0., 0., 0.]})
        env.data.qpos[env.qadr] = env.home  # the exact home pose, without reset noise
        mujoco.mj_forward(env.model, env.data)

        def tilt():  # sin^2 of each sole's tilt, straight from the simulator (xmat rows: R[2, 0], R[2, 1])
            return sum(float(env.data.xmat[foot][6] ** 2 + env.data.xmat[foot][7] ** 2) for foot in env.feet)

        assert tilt() < 1e-12  # the ankle_roll frames are level in the home pose
        for _ in range(25):  # standing in the home pose on the floor: soles stay flat (measured < 1°)
            flat = env.step(np.zeros(29))[-1]['reward_components']['feet_flat']
            assert flat == pytest.approx(-tilt()) and flat > -2 * np.sin(np.radians(2)) ** 2
        assert env.data.xpos[env.feet, 2].max() < task.CONTACT_HEIGHT  # it really stands on the floor
        rng = np.random.default_rng(0)
        thrashing = []
        for _ in range(20):
            thrashing.append(env.step(rng.normal(0, 2., 29))[-1]['reward_components']['feet_flat'])
            assert thrashing[-1] == pytest.approx(-tilt())
        assert min(thrashing) < -.01  # thrashing ankles tilt the soles
    finally:
        env.close()


def test_feet_flat_weight_is_validated_and_keeps_the_checkpoint_contract():
    from humanoid_sim.lab import validate_config
    base = dict(robot_id='g1_29dof', task_id='g1_velocity')
    assert validate_config(base)['task_config']['reward_weights']['feet_flat'] == 0  # old branches keep their reward
    accepted = validate_config({**base, 'task_config': {'reward_weights': {'feet_flat': .5}}})
    assert accepted['task_config']['reward_weights']['feet_flat'] == .5
    for value in (-.1, 101., float('nan')):
        with pytest.raises(ValueError):
            validate_config({**base, 'task_config': {'reward_weights': {'feet_flat': value}}})
    # The real-robot recipe of the current branches (exp-e53e68fb27224208): only the reward changes, so a stage
    # saved without the term can be branched into a run with it.
    recipe = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10.,
                  heading_hold=True, stand_heading_hold=False,
                  reward_weights=dict(velocity=2., yaw=2., alive=.15, slip=1., stand_pose=1., still=3., fall=100.))
    with_term = {**recipe, 'reward_weights': {**recipe['reward_weights'], 'feet_flat': .5}}
    saved = task_identity('g1_29dof', 'g1_velocity', recipe)
    assert task_identity('g1_29dof', 'g1_velocity', with_term) == saved
    validate_resume(saved, task_identity('g1_29dof', 'g1_velocity', with_term))


def test_wider_action_limit_opens_joint_range_and_is_its_own_contract():
    from humanoid_sim.core import save_stage, load_stage, write_json, read_json
    from humanoid_sim.lab import validate_config
    base = task_identity('g1_29dof', 'g1_velocity')
    assert task_identity('g1_29dof', 'g1_velocity', {}) == base  # old stages: original contract
    wide = task_identity('g1_29dof', 'g1_velocity', {'action_limit': 4})
    assert wide['action_schema'] == 'g1_velocity:action-v2-limit4' and base['action_schema'] == 'g1_velocity:action-v1'
    with pytest.raises(ValueError):
        validate_resume(base, wide)
    with pytest.raises(ValueError):
        validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', task_config={'action_limit': 20}))
    env = G1VelocityEnv(action_limit=4.)
    narrow = G1VelocityEnv()
    try:
        assert env.action_space.high[0] == 4 and narrow.action_space.high[0] == 1
        knee = 3
        for e in (env, narrow):
            e.reset(seed=0, options={'command': [1., 0, 0]})
            action = np.zeros(29)
            action[knee] = 4.
            for _ in range(25):
                e.step(action)
        bend = [np.degrees(e.data.qpos[e.qadr[knee]]) for e in (env, narrow)]
        assert bend[0] > bend[1] + 30  # the knee goes far beyond the old ±20° window
        torque = task.pd_torque(torch.tensor([[4.]]), torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1),
                                torch.ones(1), torch.ones(1), torch.zeros(1), torch.full((1,), -9.),
                                torch.full((1,), 9.), torch.full((1,), -100.), torch.full((1,), 100.), limit=4.)
        assert float(torque) == 4.
    finally:
        env.close(); narrow.close()


def test_command_range_setting_bounds_training_commands():
    generator = torch.Generator().manual_seed(2)
    commands = task.sample_commands(5000, lambda n: torch.rand(n, generator=generator), .3, 1., 0.)
    assert commands[:, 0].min() >= .3 and commands[:, 0].max() <= 1. and commands[:, 1:].abs().max() == 0
    env = G1VelocityEnv(commands={'min_speed': .3, 'max_speed': 1., 'stand_probability': 0.}, legs_only=True)
    try:
        for seed in range(5):
            env.reset(seed=seed)
            assert .3 <= env.command[0] <= 1.
    finally:
        env.close()
