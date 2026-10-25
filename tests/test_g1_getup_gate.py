"""Gate A of g1_getup (iteration 33), CPU only: the start bank, every start class over 1000 steps, the reward order, the
walking task unchanged bit for bit, and MJWarp's contact demand on the CPU device. The bank is runs/ data (untracked):
the tests that read it skip without it (python -m humanoid_sim.tasks.g1_getup_bank makes it)."""
import hashlib
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_GETUP_REWARD_WEIGHTS
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity

needs_bank = pytest.mark.skipif(not bank_module.BANK.is_file(), reason='the g1_getup start bank is not on this machine')
LYING_CLASSES = ('back', 'front', 'side', 'fall')


@pytest.fixture(scope='module')
def bank():
    return bank_module.load()


@needs_bank
def test_the_bank_has_every_class_in_its_share_and_a_holdout(bank):
    header = bank['header']
    assert bank['classes'] == bank_module.CLASSES == ('back', 'front', 'side', 'fall', 'sit', 'stand')
    assert header['generator'] == bank_module.GENERATOR
    assert header['model_revision'] == task_identity('g1_29dof', 'g1_getup')['model_revision']  # made for this model
    total = header['total']
    assert total >= 40000 and len(bank['qpos']) == total + bank_module.HOLDOUT * len(bank['classes'])
    for number, kind in enumerate(bank['classes']):
        rows = bank['start_class'] == number
        training = int((rows & ~bank['holdout']).sum())
        assert training == round(bank_module.SHARES[kind] * total), kind
        assert int((rows & bank['holdout']).sum()) == bank_module.HOLDOUT, kind
    assert dict(zip(bank['classes'], np.bincount(bank['start_class'][~bank['holdout']]) / total)) == pytest.approx(
        dict(back=.30, front=.30, side=.10, fall=.15, sit=.05, stand=.10), abs=1e-4)
    assert len(set(zip(bank['start_class'].tolist(), bank['seed'].tolist()))) == len(bank['seed'])  # no repeats
    assert np.isfinite(bank['qpos']).all() and np.isfinite(bank['qvel']).all()
    assert np.abs(bank['qvel'][:, 6:]).max() <= task.GETUP_MAX_JOINT_SPEED
    np.testing.assert_allclose(np.linalg.norm(bank['qpos'][:, 3:7], axis=1), 1., atol=1e-6)


@needs_bank
def test_the_bank_classes_lie_sit_and_stand_as_named(bank):
    """Per class, from its own states: lying classes on the floor (pelvis below 0.5 m, head below 0.7 m; never the
    model's reference pose that a MuJoCo reset would leave), standing ones up (pelvis above 0.7, tilt below 10°, head
    above 1.15), the lying ones on their back / front as named (the pelvis x axis up / down)."""
    stats = bank_module.statistics(bank)
    for kind in LYING_CLASSES:
        assert stats[kind]['pelvis_z'][2] < .5 and stats[kind]['head_z'][2] < .7, (kind, stats[kind])
    assert stats['sit']['pelvis_z'][2] < .6
    assert stats['stand']['pelvis_z'][0] > .7 and stats['stand']['tilt_deg'][2] < 10 and stats['stand']['head_z'][0] > 1.15
    env = bank_module.getup_model_env()
    import mujoco
    for kind, sign in (('back', 1), ('front', -1)):
        rows = np.flatnonzero(bank['start_class'] == bank['classes'].index(kind))
        up = []
        for row in rows[::50]:
            env.data.qpos[:] = bank['qpos'][row]
            mujoco.mj_kinematics(env.model, env.data)
            up.append(env.data.xmat[env.pelvis].reshape(3, 3)[2, 0])  # world z of the pelvis forward axis
        assert np.mean(np.array(up) * sign > .5) > .8, kind


@needs_bank
@pytest.mark.parametrize('kind', bank_module.CLASSES)
def test_1000_steps_from_every_start_class_stay_finite(bank, kind):
    """Random bang-bang targets over the whole action box (held 0.1-0.3 s), the training randomization on, from this
    class's training and held-out starts: 1000 steps without a non-finite or MuJoCo-reset state. Such flailing throws a
    joint past 50 rad/s (the episode's end) in about 2 % of the steps (shoulders, hip yaw, even the locked wrists by the
    impacts); σ 1 noise around home never did (at most 27 rad/s)."""
    env = G1GetupEnv(max_episode_steps=1000)
    rng = np.random.default_rng(bank_module.CLASSES.index(kind))
    rows = np.flatnonzero(bank['start_class'] == bank['classes'].index(kind))
    try:
        steps = fast = 0
        for start in (int(rows[0]), int(rows[-1])):  # a training start and the last held-out one
            obs, _ = env.reset(seed=steps, options=dict(start=start))
            hold = 0
            for _ in range(500):
                if hold <= 0:
                    action, hold = rng.uniform(-10, 10, 29), int(rng.integers(5, 16))
                hold -= 1
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                assert not info.get('bad_state') and np.isfinite(obs).all() and math.isfinite(reward), (kind, steps)
                if terminated:  # a joint faster than 50 rad/s: allowed, then on from the same start
                    fast += 1
                    obs, _ = env.reset(seed=steps, options=dict(start=start))
        print(f'{kind}: {fast} ends by a joint over 50 rad/s in 1000 steps')
        assert steps == 1000 and fast <= 50, fast
    finally:
        env.close()


def _best_still(bank, classes):
    """The best per-step reward a bank state of `classes` could earn staying as it is: its task terms with no cost
    (still, no torque, the same action)."""
    env = bank_module.getup_model_env()
    import mujoco
    best = -math.inf
    for kind in classes:
        for row in np.flatnonzero(bank['start_class'] == bank['classes'].index(kind)):
            env.data.qpos[:] = bank['qpos'][row]
            mujoco.mj_kinematics(env.model, env.data)
            head = task.head_height(env.data.xpos[env.torso][None], env.data.xmat[env.torso].reshape(1, 3, 3))
            torso = env.data.xmat[env.torso].reshape(1, 3, 3)
            sole = (env.data.geom_xpos[env.soles, 2] - env.model.geom_size[env.soles, 0]).min(-1)[None]
            zeros = np.zeros((1, 29))
            terms, _ = task.getup_reward_components(
                head, env.data.qpos[2:3], torso, np.zeros((1, 6)), env.data.qpos[env.qadr][None], env.home,
                *task.soft_limits(*env.model.jnt_range[env.joints].T), range(task.LEGS), sole, zeros, zeros, zeros,
                zeros)
            best = max(best, sum(float(v[0]) for k, v in terms.items() if k != 'joint_limits'))
    return best


@needs_bank
def test_standing_in_home_with_noise_beats_any_lying(bank):
    """The order of the rewards: standing in the home pose under σ 1 action noise (the policy's exploration, around
    action 0; no assist), averaged over its first 0.4 s, earns more per step than any lying bank state (back, front,
    side, fall) earns lying still (its task terms alone, every cost 0). The sitting class's best is printed too."""
    env = G1GetupEnv()
    rng = np.random.default_rng(0)
    rows = np.flatnonzero((bank['start_class'] == bank['classes'].index('stand')) & ~bank['holdout'])
    standing = []
    try:
        for start in rows[:10]:
            env.reset(seed=int(start), options=dict(start=int(start), assist_force=0., action_delay_steps=0))
            for _ in range(20):
                standing.append(env.step(rng.normal(0., 1., 29))[1])
    finally:
        env.close()
    lying, sitting = _best_still(bank, LYING_CLASSES), _best_still(bank, ('sit',))
    print(f'standing with σ 1: {np.mean(standing):.3f} per step (min {np.min(standing):.3f}); best lying still: '
          f'{lying:.3f}, best sitting still: {sitting:.3f}')
    assert np.mean(standing) > lying


# The walking task before iteration 33 (commit 4ad43f2): sha256 of every observation, reward, end flag, reward term and
# qpos of these rollouts, made by walking_digest with that code. Any change of g1_velocity's numbers fails here (and so
# would another MuJoCo, numpy or torch build: then make the digests again with the code of 4ad43f2). The recipe is
# candidate 8's task_config (exp-28691193718e4d0e @ 7221.5 M) with action_delay 0.02, so the delay path runs too.
CANDIDATE_8 = dict(
    reward_weights=dict(velocity=2., forward=0., sideways=0., heading=0., yaw=2., yaw_drift=5., alive=.15, gait=.5,
                        swing=10., tilt=0., orientation=2., height=10., torso=1., base_motion=2., still=3.,
                        stand_pose=1., slip=1., feet_flat=2., clearance=0., self_contact=1., arm_press=.1, posture=.3,
                        hips=1., effort=1e-5, action_change=.05, fall=100.),
    heading_hold=True, commands=dict(min_speed=-1., max_speed=2., stand_probability=.1, lateral_speed=.5, yaw_rate=1.,
                                     straight_probability=.5, side_share=.2, side_stop=1., side_speed=1.),
    gait=dict(period=.6, stance_fraction=.55), action_limit=10., real_robot=True, critic_extras=True,
    training_realism=1., history=5, stand_heading_hold=False, phase_stop=True, sole_contact=True, random_phase=True,
    action_delay=.02, arm_perturb=.1, lock_wrists=True, knee_floor=.061, force_push=1.25, feet_flat_stance=False,
    foot_collision=True, side_heading_hold=False, motor_friction=1.25)
WALKING = {'simulation': dict(), 'candidate_8': CANDIDATE_8, 'legs_only': dict(legs_only=True)}
WALKING_DIGESTS = {'simulation': '4219f1693915df10396a7b081c8e3253642c3becdc8cd44d8897353011a95bcd',
                   'candidate_8': 'edc47937c977f2b5acfd94fc425307cb5c5a3d5cf5cc73d6d4b0c3995f5f8583',
                   'legs_only': '80132e71bffdaee08981b209a42b9c406ac8940c333fdf9c254d86729949c94f'}


def walking_digest(config, seed=33, steps=400):
    env = G1VelocityEnv(**config)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    try:
        obs, _ = env.reset(seed=seed)  # a training reset: random command, pushes, perturbation, delay, friction
        digest.update(obs.tobytes())
        for k in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-env.action_limit, env.action_limit,
                                                                            env.model.nu) * (.3 if k % 7 else 1.))
            digest.update(obs.tobytes() + np.float64(reward).tobytes() + bytes([terminated, truncated])
                          + env.data.qpos.tobytes())
            for key in sorted(info['reward_components']):
                if key not in LATER_TERMS:
                    digest.update(key.encode() + np.float64(info['reward_components'][key]).tobytes())
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + k + 1)
                digest.update(obs.tobytes())
    finally:
        env.close()
    return digest.hexdigest()


# Reward terms added after 4ad43f2, always among the components (0 without their setting): left out of the digest.
LATER_TERMS = ('arm_track', 'arm_error')


@pytest.mark.parametrize('name', WALKING)
def test_walking_is_bit_for_bit_the_one_before_iteration_33(name):
    assert walking_digest(WALKING[name]) == WALKING_DIGESTS[name]


WARP_PROBE = textwrap.dedent('''
    import json, sys
    import numpy as np, torch
    import warp as wp
    import mujoco_warp as mjw
    from humanoid_sim.tasks import g1_getup_bank as bank_module, g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    worlds, steps = int(sys.argv[1]), int(sys.argv[2])
    wp.init()
    assert not wp.is_cuda_available()
    wp.set_device('cpu')
    env = G1GetupEnv()
    bank = bank_module.load()
    rng = np.random.default_rng(33)
    rows = np.sort(rng.choice(np.flatnonzero(~bank['holdout']), worlds, replace=False))
    model, low, high = env.model, *env.model.jnt_range[env.joints].T
    env.set_body_friction(1.)
    m = mjw.put_model(model)
    d = mjw.make_data(model, nworld=worlds, nconmax=256, njmax=1024)
    d.qpos.assign(bank['qpos'][rows].astype(np.float32))
    d.qvel.assign(bank['qvel'][rows].astype(np.float32))
    mjw.forward(m, d)
    motor = tuple(torch.from_numpy(r.copy()) for r in env.motors)
    contacts, rows_max, overflow = np.zeros(worlds, int), np.zeros(worlds, int), 0
    targets, hold = np.zeros((worlds, model.nu)), np.zeros(worlds, int)
    for step in range(steps):
        new = hold <= 0
        draw = np.clip(env.home + rng.uniform(-1, 1, (worlds, model.nu)) * (high - low) * .5, env.target_low, high)
        draw[:, list(env.wrist_joints)] = env.home[list(env.wrist_joints)]
        targets[new], hold[new] = draw[new], rng.integers(5, 16, int(new.sum()))
        hold -= 1
        for _ in range(env.frame_skip):
            qpos, qvel = d.qpos.numpy(), d.qvel.numpy().astype(np.float64)
            v = qvel[:, env.vadr]
            tau = env.kp * (targets - qpos[:, env.qadr]) - env.kd * v
            tau = task.motor_torque(torch.from_numpy(tau), torch.from_numpy(v), *motor).numpy()
            d.ctrl.assign(np.clip(tau, *model.actuator_ctrlrange.T).astype(np.float32))
            mjw.step(m, d)
            count = int(d.nacon.numpy()[0])
            contacts = np.maximum(contacts, np.bincount(d.contact.worldid.numpy()[:min(count, d.naconmax)],
                                                        minlength=worlds))
            rows_max = np.maximum(rows_max, d.nefc.numpy())
    print(json.dumps(dict(worlds=worlds, steps=steps, contacts=int(contacts.max()), nefc=int(rows_max.max()),
                          finite=bool(np.isfinite(d.qpos.numpy()).all()),
                          classes=sorted({bank['classes'][c] for c in bank['start_class'][rows]}))))
''')


@needs_bank
@pytest.mark.skipif(os.environ.get('RUN_G1_WARP_CPU_TESTS') != '1',
                    reason='MJWarp on the CPU device, minutes: RUN_G1_WARP_CPU_TESTS=1 in .venv-fullgpu')
def test_mjwarp_contacts_from_the_bank_fit_the_buffers():
    """256 worlds from the bank's training starts in MJWarp on the CPU device (CUDA hidden, as the critic's
    warp_cpu_contacts.py), the hand and foot capsules in, flailing with random targets over half the joint ranges:
    every world stays below 128 contacts and 400 constraint rows (Warp's defaults are 128 / 512 per world)."""
    pytest.importorskip('mujoco_warp')
    import json
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([sys.executable, '-c', WARP_PROBE, '256', '60'], cwd=root, env=environment,
                          capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    print(result)
    assert result['finite'] and len(result['classes']) == 6
    assert result['contacts'] < 128 and result['nefc'] < 400
