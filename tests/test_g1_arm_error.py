"""Reward weight arm_error (iteration 43, the critic's lab-critic/it42b): −w·mean |q − (home + d)| over the ten free arm
joints after the step with flag 1, only while the targets hold (g1_tensor_math.arm_holding); weight 0, the default, is
the task bit for bit; the contract and branching are unchanged."""
import hashlib

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_LABELS, G1_PARAMETERS, G1_REWARD_WEIGHTS
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_arm_targets import ARMS, TRACKED

WEIGHTED = {**ARMS, 'reward_weights': {**ARMS['reward_weights'], 'arm_error': 6.}}
# flag_one_digest(ARMS) of the code before arm_error (commit 9e6a1e7), with its count of flag-1 steps. Another MuJoCo,
# numpy or torch build changes it: then make it again with that commit's code.
OLD_FLAG_ONE = ('97645a345148fd386db0d0050f268e97e067c58f400fe6eec2fe6a543a96aaea', 172)


def flag_one_digest(config, seed=50, steps=300):
    """(sha256, flag-1 steps) of every observation, reward, end flag, qpos, the arms' error and every reward term but
    arm_error of training episodes of 60 steps (the modes drawn, the targets moving and holding, loads up to 1.5 kg)
    under random actions over the ±10 box."""
    env = G1VelocityEnv(max_episode_steps=60, **config)
    env.load_max = 1.5
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    flags = 0
    try:
        obs, _ = env.reset(seed=seed)
        digest.update(obs.tobytes())
        for k in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-10., 10., 29) * (.3 if k % 7 else 1.))
            flags += bool(info['arm_flag'])
            error = np.nan if info['arm_error'] is None else info['arm_error']
            digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated])
                          + env.data.qpos.tobytes() + np.float64(error).tobytes())
            for key in sorted(info['reward_components']):
                if key != 'arm_error':
                    digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + k + 1)
                digest.update(obs.tobytes())
    finally:
        env.close()
    return digest.hexdigest(), flags


def test_the_weight_its_label_and_the_contract():
    assert G1_REWARD_WEIGHTS['arm_error'] == 0 and 'удержании' in G1_LABELS['arm_error']
    field = next(p for p in G1_PARAMETERS if p['key'] == 'arm_error')
    assert field['group'] == 'reward_weights' and field['default'] == 0
    assert task_identity('g1_29dof', 'g1_velocity', WEIGHTED) == task_identity('g1_29dof', 'g1_velocity', ARMS)


def test_without_the_weight_flag_one_training_is_the_old_task():
    digest, flags = flag_one_digest(ARMS)
    assert flags > 100  # the arms followed targets in most steps
    assert (digest, flags) == OLD_FLAG_ONE
    assert flag_one_digest(WEIGHTED)[0] != digest  # the weight does charge those episodes


def test_holds_and_the_term():
    """A hold: the quintic over (clock ≥ move) or no move (start = end); the term −w·error with flag 1 on a hold, 0
    elsewhere and without the error given; it needs the flags and the holds."""
    start = torch.zeros((4, 29), dtype=torch.float64)
    end = start.clone()
    end[:2, 15] = -.8
    clock, move = torch.tensor([3, 50, 0, 7]), torch.tensor([50, 50, 1, 50])
    assert task.arm_holding(start, end, clock, move).tolist() == [False, True, True, True]
    n = 4
    generator = torch.Generator().manual_seed(1)
    home = torch.zeros(29, dtype=torch.float64)
    position = torch.rand((n, 29), generator=generator, dtype=torch.float64) - .5
    target = home + torch.rand((n, 29), generator=generator, dtype=torch.float64) - .5
    args = (torch.zeros((n, 6), dtype=torch.float64), torch.eye(3, dtype=torch.float64).repeat(n, 1, 1),
            torch.full((n,), .78, dtype=torch.float64), torch.zeros((n, 3), dtype=torch.float64),
            torch.zeros((n, 29), dtype=torch.float64), torch.zeros((n, 29), dtype=torch.float64),
            torch.zeros((n, 29), dtype=torch.float64), torch.full((n, 2), .034, dtype=torch.float64),
            torch.zeros(n, dtype=torch.float64), position, home, torch.eye(3, dtype=torch.float64).repeat(n, 1, 1))
    error = task.arm_error(position, target, TRACKED)
    torch.testing.assert_close(error, (position - target)[:, list(TRACKED)].abs().mean(dim=1), rtol=0, atol=1e-15)
    flag, hold = torch.tensor([True, True, False, False]), torch.tensor([True, False, True, False])
    arms = dict(arm_flag=flag, arm_track=task.arm_track(position, target, TRACKED),
                arm_posture_joints=task.POSTURE_JOINTS)
    weights = {**task.G1_REWARD_WEIGHTS, 'arm_error': 6.}
    plain, _ = task.reward_components(*args, weights=weights, **arms)
    charged, _ = task.reward_components(*args, weights=weights, arm_error=error, arm_hold=hold, **arms)
    assert float(charged['arm_error'][0]) == pytest.approx(-6 * float(error[0]), abs=1e-15)
    assert charged['arm_error'][1:].abs().max() == 0 and plain['arm_error'].abs().max() == 0
    for key in set(plain) - {'arm_error'}:
        torch.testing.assert_close(charged[key], plain[key], rtol=0, atol=0)
    off, _ = task.reward_components(*args, arm_error=error, arm_hold=hold, **arms)  # the default weight 0
    assert off['arm_error'].abs().max() == 0
    for missing in (dict(arm_error=error), dict(arm_error=error, arm_hold=hold, arm_flag=None)):
        with pytest.raises(ValueError):
            task.reward_components(*args, weights=weights, **{**arms, **missing})


def test_the_cpu_task_charges_the_error_on_holds_only():
    """Two CPU tasks on the same trajectory, weight 0 and 6: a target reached in a 0.2 s quintic (10 steps) is not
    charged while it moves, −6·arm_error from its 11th step on; the reward differs by exactly that term. Arms at home
    (flag 1, no move) hold from the first step; free arms (flag 0) are never charged."""
    envs = [G1VelocityEnv(**ARMS), G1VelocityEnv(**WEIGHTED)]
    try:
        for env in envs:
            env.reset(seed=5, options={'command': [0., 0., 0.]})
            pose = env.home.copy()
            pose[list(TRACKED)] += np.abs(np.linspace(-.5, .5, 10)) * np.array([1, 1, 1, 1, 1, 1, -1, -1, 1, -1])
            env.set_arm_targets(pose, 1, seconds=.2)
            env.set_hand_load(1.)
        holds = []
        for step in range(16):
            action = .3 * np.sin(np.arange(29) + 1.3 * step)
            (_, plain, _, _, first), (_, charged, _, _, info) = (env.step(action) for env in envs)
            assert first['arm_error'] == info['arm_error'] and first['reward_components']['arm_error'] == 0
            holds.append(info['arm_hold'])
            term = info['reward_components']['arm_error']
            assert term == (pytest.approx(-6 * info['arm_error'], rel=1e-12) if info['arm_hold'] else 0)
            assert charged - plain == pytest.approx(term, abs=1e-12)
        assert holds == [False] * 10 + [True] * 6
        env = envs[1]
        env.reset(seed=6, options={'command': [0., 0., 0.]})
        _, _, _, _, info = env.step(np.zeros(29))
        assert info['arm_flag'] is False and info['arm_hold'] is False and info['reward_components']['arm_error'] == 0
        env.set_arm_targets(None, 1)  # home, at once
        _, _, _, _, info = env.step(np.zeros(29))
        assert info['arm_hold'] and info['reward_components']['arm_error'] == pytest.approx(-6 * info['arm_error'])
        assert info['arm_error'] > 0
    finally:
        for env in envs:
            env.close()
