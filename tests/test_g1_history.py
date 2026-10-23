import numpy as np
import pytest
import torch

from humanoid_sim.checkpoint_schema import validate_resume
from humanoid_sim.symmetry import Mirror
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

REAL = dict(real_robot=True, critic_extras=True, action_limit=10., heading_hold=True,
            gait={'period': .6, 'stance_fraction': .55})
FRAME, H, N = 98, 5, 29


def _terms(obs, history):
    """The frame-size view of an observation: (history, 98) from the term-major layout."""
    blocks, start = [], 0
    for a, b in task.deploy_terms(N):
        size = b - a
        blocks.append(obs[start:start + history * size].reshape(history, size))
        start += history * size
    return np.concatenate(blocks, axis=1), obs[start:]


def test_history_is_term_major_oldest_first_and_filled_at_reset():
    single, multi = G1VelocityEnv(**REAL), G1VelocityEnv(**REAL, history=H)
    try:
        assert multi.observation_space.shape == (H * FRAME + 4,) and multi.actor_inputs == H * FRAME
        first, _ = single.reset(seed=2, options={'command': [.5, 0., 0.]})
        reset, _ = multi.reset(seed=2, options={'command': [.5, 0., 0.]})
        frames, extras = _terms(reset, H)
        np.testing.assert_allclose(frames, np.tile(first[:FRAME], (H, 1)), atol=1e-6)  # filled with the first frame
        np.testing.assert_allclose(extras, first[FRAME:], atol=1e-6)  # critic extras: current only, at the end
        action = np.random.default_rng(0).normal(0, .5, N)
        after_single = single.step(action)[0]
        after = multi.step(action)[0]
        frames, _ = _terms(after, H)
        np.testing.assert_allclose(frames[-1], after_single[:FRAME], atol=1e-6)  # newest last
        np.testing.assert_allclose(frames[:-1], np.tile(first[:FRAME], (H - 1, 1)), atol=1e-6)
        again = multi._observation()  # re-reading (as evaluation does after set_command) does not advance it
        np.testing.assert_allclose(again, after, atol=1e-6)
    finally:
        single.close()
        multi.close()


def test_history_is_its_own_contract():
    plain = task_identity('g1_29dof', 'g1_velocity', {**REAL})
    deep = task_identity('g1_29dof', 'g1_velocity', {**REAL, 'history': H})
    assert deep['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-h5-critic4'
    assert plain['observation_schema'] == 'g1_velocity:obs-unitree-deploy-v0-critic4'
    with pytest.raises(ValueError):
        validate_resume(plain, deep)
    with pytest.raises(ValueError):
        G1VelocityEnv(history=H)  # simulation profile has no deploy layout


def test_mirror_with_history_mirrors_every_frame():
    single, multi = G1VelocityEnv(**REAL), G1VelocityEnv(**REAL, history=H)
    try:
        one, many = Mirror(single), Mirror(multi)
        frames = torch.randn(3, H, FRAME, dtype=torch.float64)
        extras = torch.randn(3, 4, dtype=torch.float64)
        obs = torch.cat((task.history_observation(frames, N), extras), dim=1)
        mirrored_frames = torch.stack([one.observation(torch.cat((frames[:, h], extras), dim=1))[:, :FRAME]
                                       for h in range(H)], dim=1)
        expected = torch.cat((task.history_observation(mirrored_frames, N), one.observation(
            torch.cat((frames[:, -1], extras), dim=1))[:, FRAME:]), dim=1)
        torch.testing.assert_close(many.observation(obs), expected)
        torch.testing.assert_close(many.observation(many.observation(obs)), obs)
    finally:
        single.close()
        multi.close()
