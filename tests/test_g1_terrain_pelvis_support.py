"""task_config terrain pelvis_support (critic, it94): the pelvis height the fall, the height and height_ceiling terms
and the critic see is measured above the mean ground under the two feet (their ankle_roll origins), not above the
ground under the pelvis, which on stairs up passes over the next step before the feet do. Missing or false is the
old task bit for bit (a digest of the code before the setting)."""
import hashlib

import mujoco
import numpy as np
import pytest
import torch
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import TERRAIN_DEFAULTS, g1_config, g1_config_identity, terrain_setting
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv

# Candidate 8's real-robot profile (as tests/test_g1_terrain.py), with height_command for the ceiling and the
# crouched fall height.
PROFILE = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, sole_contact=True, lock_wrists=True,
               knee_floor=.061, foot_collision=True, action_limit=10., heading_hold=True, stand_heading_hold=False,
               side_heading_hold=False, gait=dict(period=.6, stance_fraction=.55),
               reward_weights=dict(feet_flat=2., height=10., fall=100.))

# md5 of walking_digest(terrain {}) before task_config terrain had pelvis_support (6765190)
OLD_DIGEST = 'e04a4f07e878b14c8317930912184ce0'


def walking_digest(terrain):
    """md5 over observations (critic extras included), rewards, their components, falls and the info's pelvis height of
    an episode on the top stairs up tile and two training-reset episodes, with seeded random actions."""
    raw = G1VelocityEnv(terrain=terrain, **PROFILE)
    digest = hashlib.md5()
    rng = np.random.default_rng(94)
    try:
        for seed, options in ((0, {'command': [.5, 0., 0.], 'terrain_type': 'stairs_up', 'terrain_level': 9}),
                              (1, {}), (2, {})):
            observation, _ = raw.reset(seed=seed, options=options)
            if options:  # off the spawn platform, onto the steps, where the pelvis and the feet stand over other ground
                raw.data.qpos[0] += 1.6
                raw.data.qpos[2] += float(raw._ground(raw.data.qpos[:2])) + .02
                mujoco.mj_forward(raw.model, raw.data)
                observation = raw._observation('fill')
            digest.update(observation.tobytes())
            for _ in range(60):
                observation, reward, terminated, truncated, info = raw.step(rng.normal(0., .5, raw.model.nu))
                components = np.array([info['reward_components'][key] for key in sorted(info['reward_components'])])
                digest.update(observation.tobytes())
                digest.update(np.array([reward, terminated, truncated, info['pelvis_height']], np.float64).tobytes())
                digest.update(components.tobytes())
                if terminated or truncated:
                    break
    finally:
        raw.close()
    return digest.hexdigest()


def test_without_the_setting_the_task_is_the_old_one_bit_for_bit():
    assert terrain_setting({}) == TERRAIN_DEFAULTS and terrain_setting({'pelvis_support': False}) == TERRAIN_DEFAULTS
    assert terrain_setting({'pelvis_support': True}) == {**TERRAIN_DEFAULTS, 'pelvis_support': True}
    for bad in (1, 0, 'true', None):
        with pytest.raises(ValueError):
            terrain_setting({'pelvis_support': bad})
    identity = {'task_id': 'g1_velocity', 'model_revision': 'm', 'action_schema': 'a', 'observation_schema': 'o'}
    assert (g1_config_identity(identity, g1_config({'terrain': {'pelvis_support': True}}))
            == g1_config_identity(identity, g1_config({'terrain': {}})))  # not part of the contract
    assert walking_digest({}) == walking_digest({'pelvis_support': False}) == OLD_DIGEST
    assert walking_digest({'pelvis_support': True}) != OLD_DIGEST  # the steps: the pelvis over other ground


def test_the_support_ground_is_the_mean_under_the_feet():
    """pelvis_ground on a hand-made heightfield: a 0.2 m step under the pelvis, the feet on the lower floor."""
    heights = torch.zeros((21, 21), dtype=torch.float64)
    heights[:, 11:] = .2  # a riser between x = 0.5 and 0.55 (cell 0.05, corner 0)
    pelvis = torch.tensor([[.6, .5]], dtype=torch.float64)
    feet = torch.tensor([[[.4, .38], [.42, .62]]], dtype=torch.float64)
    point = task.pelvis_ground(pelvis, feet, heights, (0., 0.), .05, .002)
    support = task.pelvis_ground(pelvis, feet, heights, (0., 0.), .05, .002, support=True)
    assert point.tolist() == pytest.approx([.202]) and support.tolist() == pytest.approx([.002])
    one_up = torch.tensor([[[.4, .38], [.7, .62]]], dtype=torch.float64)  # one foot on the step: the mean
    assert task.pelvis_ground(pelvis, one_up, heights, (0., 0.), .05, .002, True).tolist() == pytest.approx([.102])


@pytest.fixture(scope='module')
def stairs():
    """The task with pelvis_support on and off, the robot in home on a flat tile; the reward's ground (terrain_heights,
    not the physics' heightfield) raised by 0.25 m under the pelvis and ahead of it, between the feet, as a stair the
    pelvis passes over before the feet (stairs94b: on 20 cm steps the pelvis stood 0.68-0.74 m above the feet,
    0.52-0.55 above the step)."""
    envs = {}
    for support in (False, True):
        raw = G1VelocityEnv(terrain={'pelvis_support': support}, **PROFILE)
        raw.reset(seed=0, options={'command': [0., 0., 0.], 'terrain_type': 'flat', 'terrain_level': 0})
        raw.data.qpos[raw.qadr] = raw.home
        mujoco.mj_kinematics(raw.model, raw.data)
        raw.data.qpos[2] += -.0005 - float(raw._sole_height().min())
        mujoco.mj_forward(raw.model, raw.data)
        # The step's nodes: from just behind the pelvis on, and within 0.06 m of it sideways, so the feet (0.12 m to
        # either side, 0.03 m behind) stand beyond the triangles that reach up to one cell past them.
        cell, (left, bottom) = raw.terrain_cell, raw.terrain_corner
        rows, columns = raw.terrain_heights.shape
        x = left + cell * torch.arange(columns, dtype=torch.float64)
        y = bottom + cell * torch.arange(rows, dtype=torch.float64)
        step = (x[None, :] >= raw.data.qpos[0] - cell) & ((y[:, None] - raw.data.qpos[1]).abs() <= .06)
        feet = raw.data.xpos[raw.feet, :2]
        assert (np.abs(feet[:, 1] - raw.data.qpos[1]) > .06 + cell).all(), feet
        raw.terrain_heights = raw.terrain_heights + .25 * step.to(torch.float64)
        envs[support] = raw
    yield envs
    for raw in envs.values():
        raw.close()


def test_a_pelvis_over_a_riser_is_not_a_fall(stairs):
    """The robot standing in home, its pelvis 0.78 m above the feet: measured above the step under it (0.53 m, below
    the fall height 0.55) it falls; above the feet's ground it stands, and the height term, the critic's height column
    and the info agree."""
    for support, raw in stairs.items():
        ground = raw._ground_terms()
        observation = raw._observation()
        assert float(ground['height'][0]) == pytest.approx(raw.pelvis_height())
        assert float(observation[-1]) == pytest.approx(raw.pelvis_height(), abs=1e-6)  # the critic's last column
        expected = .78 if support else .53
        assert raw.pelvis_height() == pytest.approx(expected, abs=.02)
        _, _, terminated, _, info = raw.step(np.zeros(raw.model.nu))
        assert info['pelvis_height'] == pytest.approx(expected, abs=.02)
        assert info['fell'] == (not support) and terminated == (not support)
        if support:
            assert info['reward_components']['height'] > -.01  # (h − 0.78)² at weight 10
