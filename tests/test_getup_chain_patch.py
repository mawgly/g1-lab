"""The getting-up chain's g1_ctrl patch (iteration 82, scripts/sim2sim/patches/g1_ctrl-getupchain.patch) through its
Python model scripts/sim2sim/chain_model.py (the C++ number for number, float32): the hand-over rule against
getup_chain.HandOver, the PD transition's targets and gains against getup_chain.pd_schedule and the tracker stage's own
PD targets, the config.yaml the patch adds against getup_chain's chosen variant, and the patch itself: `git apply` in a
clean copy of unitree_rl_mjlab (setup.sh's pinned commit, from the WSL tree or $UNITREE_RL_MJLAB) after setup.sh's
other patches in their order, with and without the height patch."""
import importlib
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'scripts/sim2sim/patches'
SETUP = ROOT / 'scripts/sim2sim/setup.sh'
CHAIN_PATCH = 'g1_ctrl-getupchain.patch'
GIT = ROOT / '.tools/git/cmd/git.exe'
WSL_TREE = Path(r'\\wsl.localhost\Ubuntu\home\mawgly\unitree\unitree_rl_mjlab')
TRACKER = ROOT / 'runs/humanoid_lab/exp-84b0540b0092467d/stage_000570359808'
TRACKER_EXPORT = ROOT / 'runs/humanoid_lab/export/exp-84b0540b0092467d-570359808'
# getup_chain.py's chosen transition (the lead's decision of 27.10): pd 3.5 arms=0.3 kp=0.3 kd=3 kp_to=1 kd_to=1
CHOSEN = dict(seconds=3.5, arms=.3, kp=.3, kd=3., kp_to=1., kd_to=1.)
ARMS = list(range(15, 29))  # the G1 29-DoF arm joints (ARM_JOINT_PARTS: shoulder, elbow, wrist)
LOCKED = [20, 21, 27, 28]
DT = .02
FIRST = ROOT / 'runs/humanoid_lab/exp-207eb498dc824027/stage_001164902400'
FIRST_EXPORT = ROOT / 'runs/humanoid_lab/export/exp-207eb498dc824027-1164902400'
BANK = ROOT / 'runs/humanoid_lab/getup_bank/bank.npz'
# Iteration 91 (the critic's it91 bridgepose.py): lowstate at GetUp's entry in the Unitree bridge run
# 20261029-065629-getup-forward1-caps-chain, where 1164 got up alone: on the back after a forward push, both hips flexed
# +2.69 rad, knees 0.39. Pelvis height (sportmodestate), IMU quaternion (w, x, y, z), the 29 motors in SDK order.
BRIDGE_POSE = dict(z=0.159363091, quat=[0.79196, 0.001993, -0.610565, 0.002564], q=[
    2.69139, 0.06663, 0.0136, 0.39101, -0.87385, -0.06085, 2.69164, -0.05615, -0.01818, 0.39099, -0.87386, 0.06125,
    -0.00212, -0.00819, -0.3042, 0.27039, 0.28531, 0.19091, 1.88692, -0.06643, -0.37578, -0.23129, 0.2658, -0.27416,
    -0.18709, 1.90011, 0.06547, -0.38296, 0.23869])


@pytest.fixture(scope='module')
def modules():
    """(getup_chain, chain_model); getup_chain sets CUDA_VISIBLE_DEVICES at import, put back here."""
    saved = os.environ.get('CUDA_VISIBLE_DEVICES')
    for path in ('scripts/lab/probes', 'scripts/lab', 'scripts/sim2sim'):
        sys.path.insert(0, str(ROOT / path))
    try:
        getup_chain = importlib.import_module('getup_chain')
        chain_model = importlib.import_module('chain_model')
    finally:
        if saved is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = saved
    return getup_chain, chain_model


def synthetic(rng, nu=29):
    """(deploy.yaml, handover.yaml) of a made-up tracker: identity joint map, ranges ±U(1, 2.5), the locked wrists."""
    half = rng.uniform(1., 2.5, nu)
    home = rng.uniform(-.3, .3, nu)
    home[LOCKED] = 0.
    deploy = dict(joint_ids_map=list(range(nu)), step_dt=DT, stiffness=list(rng.uniform(10, 100, nu)),
                  damping=list(rng.uniform(.5, 6, nu)), default_joint_pos=list(home),
                  actions=dict(JointPositionAction=dict(clip=[[-h, h] for h in half])))
    joints = rng.uniform(-3., 3., nu)  # some beyond their range: the clip must act
    joints[LOCKED] = 0.
    frame = dict(frame=18, seconds=8.1, gravity=[1., 0., 0.], joints=list(joints), locked=LOCKED,
                 free=[j for j in range(nu) if j not in LOCKED])
    return deploy, frame


def reference_targets(getup_chain, shape, start, goal, low, high, home, steps):
    """getup_chain.run's PD targets and gain multipliers: start + smoothstep(progress)·(goal − start) by pd_schedule,
    the locked joints at home (applied_action's lock), the rest clipped to the joint range (_target)."""
    import getup_roll_clip
    shape = {**getup_chain.PD_DEFAULTS, **shape}
    share, late = getup_chain.pd_shares(len(start), ARMS, shape, DT)
    out = []
    for k in range(1, steps + 1):
        progress, kp, kd = getup_chain.pd_schedule(k, steps, share, late, shape)
        q = np.clip(start + getup_roll_clip.smoothstep(progress) * (goal - start), low, high)
        q[LOCKED] = home[LOCKED]
        out.append((q, kp, kd))
    return out


@pytest.mark.parametrize('shape', [
    {},  # the first version (no arms share, gains constant)
    dict(arms=.3, kp=.3, kd=3., kp_to=1., kd_to=1.),  # the chosen one
    dict(ramp=.8, arms=.5, legs=.5, kp=.5, kd=2., kp_to=1., kd_to=1.5),
    dict(ramp=.6, legs=.3, kp=.4, kd=2.),
])
def test_the_transition_is_getup_chains_schedule(modules, shape):
    """Every step's targets and gains of chain_model.Blend (the C++'s) are getup_chain.run's within float32."""
    getup_chain, chain_model = modules
    rng = np.random.default_rng(82)
    deploy, frame = synthetic(rng)
    seconds = 3.5
    cfg = dict(seconds=seconds, ramp=shape.get('ramp', .7), arms=shape.get('arms'), legs=shape.get('legs', 0.),
               arm_joints=ARMS, kp=shape.get('kp', 1.), kd=shape.get('kd', 1.), kp_to=shape.get('kp_to'),
               kd_to=shape.get('kd_to'))
    blend = chain_model.Blend(cfg, deploy, frame)
    steps = int(round(seconds / DT))
    assert blend.steps == steps == 175
    clip = np.array(deploy['actions']['JointPositionAction']['clip'])
    home = np.array(deploy['default_joint_pos'])
    start = np.clip(rng.uniform(-2.5, 2.5, 29), clip[:, 0], clip[:, 1])
    start[LOCKED] = 0.
    blend.enter(start)
    reference = reference_targets(getup_chain, {key: value for key, value in shape.items()}, start,
                                  np.array(frame['joints']), clip[:, 0], clip[:, 1], home, steps)
    stiffness, damping = np.array(deploy['stiffness']), np.array(deploy['damping'])
    for k, (q, kp, kd) in enumerate(reference, 1):
        got_q, got_kp, got_kd = blend.at(k)
        assert np.abs(got_q - q).max() < 1e-5, k
        assert np.abs(got_kp - stiffness * kp).max() < 1e-4 and np.abs(got_kd - damping * kd).max() < 1e-5, k
    # the last step is the frame's targets, the gains the tracker's own (with kp_to = kd_to = 1)
    final = np.clip(np.array(frame['joints']), clip[:, 0], clip[:, 1])
    final[LOCKED] = home[LOCKED]
    assert np.abs(blend.at(steps)[0] - final).max() < 1e-5
    # run()'s step for the time since entry: 1 during the first policy step, steps from the last on
    assert blend.step_at(0.) == 1 and blend.step_at(.0199) == 1 and blend.step_at(.0201) == 2
    assert blend.step_at(10.) == steps


def test_the_hand_over_rule_is_getup_chains(modules):
    """chain_model.Rule (the C++'s) fires on the same step with the same rule as getup_chain.HandOver over 4000 random
    runs of observed angles and deviations; every rule fires somewhere; the settings are getup_chain's constants."""
    getup_chain, chain_model = modules
    from humanoid_sim.tasks import g1_tensor_math as task
    defaults = chain_model.RULE_DEFAULTS
    assert abs(defaults['face_angle'] - getup_chain.FACE_ANGLE) < 1e-6
    assert abs(defaults['angle_scale'] - task.SETTLE_DONE_ANGLE) < 1e-6
    assert defaults['pose_scale'] == task.SETTLE_DONE_POSE and defaults['pass_margin'] == getup_chain.PASS_MARGIN
    assert defaults['good_score'] == getup_chain.GOOD_SCORE and defaults['face_seconds'] == getup_chain.FACE_SECONDS
    assert defaults['first_seconds'] == getup_chain.FIRST_SECONDS
    rng = np.random.default_rng(83)
    fired = {}
    for run in range(4000):
        python, model = getup_chain.HandOver(DT), chain_model.Rule(DT)
        assert model.steps == round(getup_chain.FIRST_SECONDS / DT)
        angle = rng.uniform(.5, 2.5)
        speed = rng.uniform(-.08, .02)
        deviation = rng.uniform(.3, 1.5)
        for step in range(1, model.steps + 1):
            angle = float(np.clip(angle + speed + rng.normal(0, .03), 0., np.pi))
            deviation = float(np.clip(deviation + rng.normal(-.005, .03), 0., 3.))
            a, b = python(step, angle, deviation), model.step(step, angle, deviation)
            assert (a or '') == b, (run, step)
            if a:
                fired[a] = fired.get(a, 0) + 1
                break
    assert set(fired) == {'good', 'left', 'least', 'timeout'}, fired


def test_left_reset_starts_the_rule_afresh_as_getup_chains(modules):
    """With left 'reset' (iteration 89) chain_model.Rule and getup_chain.HandOver fire on the same step with the same
    rule over 4000 random runs, never 'left'; leaving face down clears faced and least, and a run that comes back
    face down is followed afresh. Without the key the chain's Rule is the old one (left fires)."""
    getup_chain, chain_model = modules
    rng = np.random.default_rng(89)
    fired, refaced = {}, 0
    for run in range(4000):
        python, model = getup_chain.HandOver(DT, left='reset'), chain_model.Rule(DT, left='reset')
        angle = rng.uniform(.5, 2.5)
        speed = rng.uniform(-.08, .02)
        deviation = rng.uniform(.3, 1.5)
        faced = []
        for step in range(1, model.steps + 1):
            angle = float(np.clip(angle + speed + rng.normal(0, .03), 0., np.pi))
            deviation = float(np.clip(deviation + rng.normal(-.005, .03), 0., 3.))
            a, b = python(step, angle, deviation), model.step(step, angle, deviation)
            assert (a or '') == b, (run, step)
            assert (python.faced_at is None) == (model.faced_at < 0)
            assert python.least == model.least or abs(python.least - model.least) < 1e-5  # float32 against float64
            if model.faced_at >= 0 and (not faced or faced[-1] != model.faced_at):
                faced.append(model.faced_at)
            if a:
                fired[a] = fired.get(a, 0) + 1
                break
        refaced += len(faced) > 1
    assert 'left' not in fired and {'good', 'least', 'timeout'} <= set(fired), fired
    assert refaced > 0  # the rule did start afresh somewhere
    # one face-down step, then out: the old rule hands over ('left'), reset clears it
    old, new = chain_model.Rule(DT), chain_model.Rule(DT, left='reset')
    for rule in (old, new):
        assert rule.step(1, .5, 1.5) == ''
    assert old.step(2, 1., 1.5) == 'left'
    assert new.step(2, 1., 1.5) == '' and new.faced_at == -1 and new.least == np.inf
    with pytest.raises(ValueError):
        chain_model.Rule(DT, left='other')


def test_the_way_back_is_getup_chains(modules):
    """GetUpBlend's way back to GetUp (iteration 89): chain_model.Retry (State_GetUpBlend::back_due with
    getup_chain::returns and State_RLBase::enter) and getup_chain.Retry (the probe) agree over random falls: each GetUp
    entry may hand over or not, each transition step goes back for the same reason; every reason and the block past
    max_retries occur; abort_angle fires at any step, retry_angle only at the last; a fall's next GetUp entry from
    another state starts the count afresh."""
    getup_chain, chain_model = modules
    rng = np.random.default_rng(890)
    seen = set()
    steps = 30
    for run in range(800):
        retry_angle = None if rng.random() < .2 else float(rng.uniform(.5, 2.))
        abort_angle = None if retry_angle is None or rng.random() < .2 else float(rng.uniform(2., 3.))
        if retry_angle is None and abort_angle is None:
            abort_angle = float(rng.uniform(2., 3.))
        max_retries = int(rng.integers(0, 4))
        handover = dict(max_retries=max_retries)
        if retry_angle is not None:
            handover['retry_angle'] = retry_angle
        if abort_angle is not None:
            handover['abort_angle'] = abort_angle
        model = chain_model.Retry(handover)
        assert model.on and model.max_retries == max_retries
        for fall in range(2):  # a second fall: GetUp entered from Passive, the count afresh
            probe = getup_chain.Retry(retry_angle, abort_angle, max_retries)
            for attempt in range(max_retries + 3):
                allowed = model.enter()
                assert allowed == probe.allowed(), (run, fall, attempt)
                if not allowed:
                    seen.add('blocked')
                    break
                angle = rng.uniform(0., 1.5)
                drift = rng.uniform(-.02, .12)
                why = ''
                for k in range(1, steps + 1):
                    angle = float(np.clip(angle + drift + rng.normal(0, .05), 0., np.pi))
                    why = model.blend(angle, k == steps)
                    assert why == (probe(angle, k == steps) or ''), (run, fall, attempt, k)
                    if why:
                        assert why == 'abort_angle' or k == steps
                        seen.add(why)
                        break
                assert model.count == probe.count
                if not why:  # on to the tracker: this fall is over
                    seen.add('track')
                    break
            model.pending = False  # the chain left for Passive or the tracker: the next GetUp entry is not a return
    assert seen == {'abort_angle', 'retry_angle', 'blocked', 'track'}, seen


def test_without_the_new_keys_nothing_goes_back(modules):
    """No retry_angle / abort_angle: chain_model.Retry is off (GetUp always may hand over, GetUpBlend never goes
    back, max_retries -1 as State_RLBase's handover_retries) and back_reason never fires; the rule's left defaults
    to handing over."""
    _, chain_model = modules
    retry = chain_model.Retry(dict(to='GetUpBlend', max_retries=0))
    assert not retry.on and retry.max_retries == -1
    for k in range(10):
        assert retry.enter() and retry.blend(np.pi, True) == '' and retry.count == 0
    assert chain_model.back_reason(np.pi, True, None, None) == ''
    assert chain_model.back_reason(np.pi, True, -1., -1.) == ''
    assert chain_model.back_reason(2., False, 1.5, None) == '' and chain_model.back_reason(2., True, 1.5, None) \
        == 'retry_angle'
    assert chain_model.back_reason(2.7, False, 1.5, 2.6) == 'abort_angle'
    assert not chain_model.Rule(DT).left_reset


def test_the_imu_gravity_is_the_tasks_projected_gravity(modules):
    """chain_model.imu_gravity (getup_chain::imu_gravity, R^T (0, 0, -1) of the IMU quaternion) is
    g1_tensor_math.projected_gravity of the same rotation, so the way back's angle is the hand-over rule's."""
    import mujoco
    from humanoid_sim.tasks import g1_tensor_math as task
    _, chain_model = modules
    rng = np.random.default_rng(891)
    for _ in range(200):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, quat)
        wanted = np.asarray(task.projected_gravity(matrix.reshape(1, 3, 3)))[0]
        assert np.abs(chain_model.imu_gravity(quat) - wanted).max() < 1e-6


def test_without_the_legs_home_keys_there_is_no_phase(modules):
    """No legs_home_hip / legs_home_seconds (iteration 91): chain_model.LegsHome is off (State_RLBase's legs_home.on
    false) and never due, the bridge pose included; either key alone turns it on with the other at its default."""
    _, chain_model = modules
    deploy, frame = synthetic(np.random.default_rng(91))
    old = dict(to='GetUpBlend', left='reset', retry_angle=1.57, abort_angle=2.62, max_retries=2)
    phase = chain_model.LegsHome(old, deploy, frame)
    assert not phase.on and not phase.due(BRIDGE_POSE['q'])
    for key, value in (('legs_home_hip', 2.5), ('legs_home_seconds', 0.5)):
        phase = chain_model.LegsHome({**old, key: value}, deploy, frame)
        assert phase.on
        assert phase.hip == np.float32(value if key == 'legs_home_hip' else 2.)
        assert phase.steps == (25 if key == 'legs_home_seconds' else 50)
    assert chain_model.LEGS_HOME_DEFAULTS == dict(legs_home_hip=2.0, legs_home_seconds=1.0)


def test_the_legs_home_phase_is_getup_chains(modules):
    """chain_model.LegsHome (getup_chain::LegsHome) against getup_chain.legs_home_phase's schedule: every step's targets
    are pd_transition's quintic (PD_DEFAULTS: ramp 0.7, gains x1) from the angles on entry to them with the legs at
    home, the locked motors at home, the rest clipped; the gains GetUp's own. It is due on the bridge pose, not on a
    return's entry nor with one hip below the limit; the policy steps by the time since its start."""
    getup_chain, chain_model = modules
    import getup_roll_clip
    assert getup_chain.LEGS_HOME_HIP == chain_model.LEGS_HOME_DEFAULTS['legs_home_hip']
    assert getup_chain.LEGS_HOME_SECONDS == chain_model.LEGS_HOME_DEFAULTS['legs_home_seconds']
    assert getup_chain.PD_DEFAULTS['ramp'] == chain_model.LEGS_HOME_RAMP
    rng = np.random.default_rng(910)
    deploy, frame = synthetic(rng)
    phase = chain_model.LegsHome(dict(legs_home_hip=2., legs_home_seconds=1.), deploy, frame)
    q0 = np.array(BRIDGE_POSE['q'])
    assert phase.due(q0) and not phase.due(q0, returned=True)
    for hip in chain_model.HIPS:
        one = q0.copy()
        one[hip] = 1.9
        assert not phase.due(one)
    clip = np.array(deploy['actions']['JointPositionAction']['clip'])
    home = np.array(deploy['default_joint_pos'])
    start = rng.uniform(-3., 3., 29)  # some beyond their clip: held, but clipped
    goal = start.copy()
    goal[list(chain_model.LEGS)] = home[list(chain_model.LEGS)]
    phase.enter(start)
    steps = phase.steps
    assert steps == 50
    shape = dict(getup_chain.PD_DEFAULTS)
    share, late = getup_chain.pd_shares(29, ARMS, shape, DT)
    for k in range(1, steps + 1):
        progress, kp, kd = getup_chain.pd_schedule(k, steps, share, late, shape)
        wanted = np.clip(start + getup_roll_clip.smoothstep(progress) * (goal - start), clip[:, 0], clip[:, 1])
        wanted[LOCKED] = home[LOCKED]
        q, gain_p, gain_d = phase.at(k)
        assert np.abs(q - wanted).max() < 1e-5, k
        assert kp == kd == 1. and np.allclose(gain_p, deploy['stiffness']) and np.allclose(gain_d, deploy['damping'])
    held = [j for j in range(29) if j not in chain_model.LEGS and j not in LOCKED]
    final = phase.at(steps)[0]
    assert np.allclose(final[held], np.clip(start, clip[:, 0], clip[:, 1])[held], atol=1e-6)
    assert np.allclose(final[list(chain_model.LEGS)], np.clip(home, clip[:, 0], clip[:, 1])[list(chain_model.LEGS)],
                       atol=1e-6)
    assert phase.at(35)[0][0] == pytest.approx(final[0], abs=1e-6)  # the quintic done at 0.7 of the steps
    assert phase.step_at(0.) == 1 and phase.step_at(.0201) == 2 and phase.step_at(.999) == steps
    assert phase.step_at(1.) is None


@pytest.mark.skipif(not ((FIRST / 'metadata.json').is_file() and BANK.is_file() and (TRACKER / 'metadata.json').is_file()
                         and (FIRST_EXPORT / 'deploy.yaml').is_file()),
                    reason='stage 1164, its export, tracker 73 or the start bank is not here')
def test_the_legs_home_phase_in_the_probe(modules, monkeypatch):
    """getup_chain's --legs-home in 1164's CPU task: the legs are SDK motors 0-11, the hips 0 and 6; no held-out bank
    row is due (and the phase does not run from one, its run is the run without the key, record for record); from the
    bridge pose it runs, its last target is chain_model.LegsHome's from the same observed angles with 1164's export
    deploy.yaml, the hips come home, no joint above 25 rad/s, then A starts with last action 0 and pd_bound lowest."""
    import mujoco
    # getup_chain.run installs getup_probe's mj_step hook for the whole process: put the original back afterwards
    monkeypatch.setattr(mujoco, 'mj_step', mujoco.mj_step)
    getup_chain, chain_model = modules
    from humanoid_sim.tasks import g1_getup_bank
    env, *_ = getup_chain._setup(str(FIRST), False)
    raw = env.unwrapped
    legs, hips = getup_chain.leg_joints(raw)
    assert legs == list(chain_model.LEGS) and hips == list(chain_model.HIPS)
    deploy = yaml.safe_load((FIRST_EXPORT / 'deploy.yaml').read_text(encoding='utf-8'))
    phase = chain_model.LegsHome(dict(legs_home_hip=2., legs_home_seconds=1.), deploy, dict(locked=LOCKED))
    bank = g1_getup_bank.load()
    held = np.flatnonzero(bank['holdout'])
    assert not any(phase.due(bank['qpos'][row][raw.qadr]) for row in held)
    back = int(held[bank['start_class'][held] == list(bank['classes']).index('back')][0])
    options = dict(assist_force=0., start=back, action_delay_steps=0, motor_friction_scale=1., pd_bound='min')
    env.reset(seed=1, options=options)
    assert getup_chain.legs_home_phase(env, 2., 1., [np.zeros(29)] * 3) is None
    job = dict(first=str(FIRST), second=str(TRACKER), handover='pd', pd_seconds=3.5, pd=dict(CHOSEN), physics='ours',
               start='back', row=back, seed=getup_chain.SEED, first_seconds=3., rule='pelvis', second_seconds=.2,
               left='reset', retry=None)
    plain, keyed = getup_chain.run(job), getup_chain.run(dict(job, legs_home=dict(hip=2., seconds=1.)))
    assert keyed.pop('legs_home') == dict(hip=2., seconds=1., ran=False) and keyed == plain
    # the bridge pose, lifted out of the floor as the critic's bridgepose.py
    env.reset(seed=1, options=options)
    qpos = raw.data.qpos.copy()
    qpos[2], qpos[3:7], qpos[raw.qadr] = BRIDGE_POSE['z'], BRIDGE_POSE['quat'], BRIDGE_POSE['q']
    raw.data.qpos[:], raw.data.qvel[:] = qpos, 0
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qpos[2] += max([-raw.data.contact[k].dist for k in range(raw.data.ncon)] + [0.]) + .002
    mujoco.mj_forward(raw.model, raw.data)
    raw.previous_action[:] = 0
    raw._observation('fill')
    _, observed = getup_chain.reading(raw)
    assert phase.due(observed)
    speeds = []
    out = getup_chain.legs_home_phase(env, 2., 1., [np.zeros(29)] * 3,
                                      lambda bad: speeds.append(float(np.abs(raw.data.qvel[raw.vadr]).max())))
    assert out is not None and out['transition']['fell'] is None and len(speeds) == phase.steps
    assert out['hips'] == pytest.approx(list(observed[hips]))
    phase.enter(observed)
    assert np.abs(phase.at(phase.steps)[0] - out['transition']['target']).max() < 1e-5
    assert np.abs(raw.data.qpos[raw.qadr][hips] - raw.home[hips]).max() < .3 and max(speeds) < 25.
    assert raw.bound == pytest.approx(raw.pd_bound[1]) and not raw.previous_action.any()


def test_the_return_lines_are_getup_reports():
    """The patch's warn lines of a return (GetUpBlend) and of GetUp past max_retries, formatted from their fmt strings
    in the patch, are what getup_report.py reads."""
    sys.path.insert(0, str(ROOT / 'scripts/sim2sim'))
    import getup_report
    text = (PATCHES / CHAIN_PATCH).read_text(encoding='utf-8')
    back = re.search(r'"\{\}: pelvis \{:\.1f\} deg from frame \{\}\'s at policy step \{\} of \{\} > \{\} \{:\.1f\} deg, '
                     r'back to \{\} "\n\+\s+"\(return \{\}, lowstate tick \{\}\) \(sim2sim getupchain patch\)"', text)
    assert back, 'the return line is not in the patch'
    line = ("{}: pelvis {:.1f} deg from frame {}'s at policy step {} of {} > {} {:.1f} deg, back to {} "
            "(return {}, lowstate tick {}) (sim2sim getupchain patch)").format(
        'GetUpBlend', 177.3, 18, 175, 175, 'retry_angle', 90., 'GetUp', 2, 123456)
    m = getup_report.RETURNED.match(line)
    assert m and m.group(1) == '177.3' and m.group(3) == '175' and m.group(5) == 'retry_angle' and m.group(8) == '2'
    assert '"{}: return {} from {} > max_retries {}, no hand-over on this entry (sim2sim getupchain "' in text
    blocked = "{}: return {} from {} > max_retries {}, no hand-over on this entry (sim2sim getupchain patch)".format(
        'GetUp', 3, 'GetUpBlend', 2)
    assert getup_report.BLOCKED.match(blocked)


def g(*args, cwd=None):
    return subprocess.run([str(GIT), '-c', 'safe.directory=*', *args], cwd=cwd, capture_output=True, text=True)


def setup_order():
    """setup.sh's G1CTRL_PATCHES, in order."""
    text = SETUP.read_text(encoding='utf-8')
    names = re.search(r'G1CTRL_PATCHES=\(([^)]*)\)', text).group(1).split()
    commit = re.search(r'^RL_MJLAB_SHA=(\w+)', text, re.M).group(1)
    return names, commit


@pytest.fixture(scope='module')
def clean_tree(tmp_path_factory):
    """deploy/include and deploy/robots/g1 of unitree_rl_mjlab at setup.sh's commit, untouched by any patch (git
    archive of $UNITREE_RL_MJLAB or the WSL tree); skipped without git or the tree."""
    repo = Path(os.environ.get('UNITREE_RL_MJLAB', WSL_TREE))
    if not GIT.exists():
        pytest.skip('no portable git in .tools/git')
    try:
        reachable = (repo / '.git').exists()
    except OSError:
        reachable = False
    if not reachable:
        pytest.skip(f'no unitree_rl_mjlab tree at {repo} (set UNITREE_RL_MJLAB)')
    _, commit = setup_order()
    archive = tmp_path_factory.mktemp('rl_mjlab') / 'deploy.tar'
    result = g('-C', str(repo), 'archive', '--format=tar', '-o', str(archive), commit, 'deploy/include',
               'deploy/robots/g1/src', 'deploy/robots/g1/include', 'deploy/robots/g1/main.cpp',
               'deploy/robots/g1/config/config.yaml')
    if result.returncode:
        pytest.skip(f'git archive {commit}: {result.stderr.strip()}')
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent, filter='data')
    archive.unlink()
    return archive.parent


def copy_tree(source, target):
    import shutil
    shutil.copytree(source, target)
    return target


@pytest.mark.parametrize('height', [True, False])
def test_the_patch_applies_after_setups_others_and_reverses(modules, clean_tree, tmp_path, height):
    """In a clean copy, setup.sh's patches in its order (without the height patch: SIM2SIM_HEIGHT=off) apply, the chain
    patch last; it then checks out reversed (setup.sh's apply_patch finds it on); its config.yaml parses with the chain's
    states commented out among the enabled ones and GetUp's rule and GetUpBlend's shape getup_chain's."""
    names, _ = setup_order()
    assert names[-1] == CHAIN_PATCH and names.index('g1_ctrl-getup.patch') < names.index(CHAIN_PATCH)
    tree = copy_tree(clean_tree, tmp_path / 'tree')
    for name in names:
        if name == 'g1_ctrl-height.patch' and not height:
            continue
        patch = str(PATCHES / name)
        check = g('apply', '--check', patch, cwd=tree)
        assert check.returncode == 0, f'{name}: {check.stderr}'
        assert g('apply', patch, cwd=tree).returncode == 0
    assert g('apply', '--reverse', '--check', str(PATCHES / CHAIN_PATCH), cwd=tree).returncode == 0
    for path in ('deploy/include/FSM/getup_chain.h', 'deploy/include/FSM/State_GetUpBlend.h'):
        assert (tree / path).is_file()
    assert '#include "FSM/State_GetUpBlend.h"' in (tree / 'deploy/robots/g1/main.cpp').read_text(encoding='utf-8')
    text = (tree / 'deploy/robots/g1/config/config.yaml').read_text(encoding='utf-8')
    assert '    # GetUpBlend: {id: 6, type: GetUpBlend}' in text and '    # GetUpTrack: {id: 7, type: RLBase}' in text
    config = yaml.safe_load(text)
    fsm = config['FSM']
    assert set(fsm['_']) == {'Passive', 'FixStand', 'Velocity', 'Mimic_Dance1_subject2'}  # the chain off
    getup_chain, chain_model = modules
    RULE_DEFAULTS = chain_model.RULE_DEFAULTS
    handover = fsm['GetUp']['handover']
    assert handover['to'] == 'GetUpBlend' and handover['policy_dir'] == fsm['GetUpTrack']['policy_dir']
    assert {key: handover[key] for key in RULE_DEFAULTS} == pytest.approx(RULE_DEFAULTS, abs=1e-7)
    # iteration 89: left reset and the way back at its values (90°, 150°, 2), the probe's defaults the same
    defaults = chain_model.RETRY_DEFAULTS
    assert handover['left'] == 'reset' and handover['max_retries'] == defaults['max_retries'] == getup_chain.MAX_RETRIES
    assert handover['retry_angle'] == pytest.approx(defaults['retry_angle'], abs=1e-7)
    assert defaults['retry_angle'] == pytest.approx(np.radians(90.), abs=1e-7)
    assert handover['abort_angle'] == pytest.approx(defaults['abort_angle'], abs=1e-7)
    assert defaults['abort_angle'] == pytest.approx(getup_chain.ABORT_ANGLE, abs=1e-7)
    assert chain_model.retry(config).on
    # iteration 91: the legs-home phase at its defaults, the probe's
    assert {key: handover[key] for key in chain_model.LEGS_HOME_DEFAULTS} == chain_model.LEGS_HOME_DEFAULTS
    assert handover['legs_home_hip'] == getup_chain.LEGS_HOME_HIP
    assert handover['legs_home_seconds'] == getup_chain.LEGS_HOME_SECONDS
    header = (tree / 'deploy/include/FSM/getup_chain.h').read_text(encoding='utf-8')
    assert 'inline const char * back_reason(' in header and 'inline Returns returns;' in header
    assert 'struct LegsHome' in header and 'static constexpr float kRamp = 0.7f;' in header
    assert 'static constexpr int kLegs = 12;' in header and 'static constexpr int kHips[2] = {0, 6};' in header
    source = (tree / 'deploy/robots/g1/src/State_RLBase.cpp').read_text(encoding='utf-8')
    assert 'h["legs_home_hip"].as<float>(legs_home.hip)' in source and 'void State_RLBase::legs_run()' in source
    blend = fsm['GetUpBlend']
    assert blend['next'] == 'GetUpTrack' and blend['policy_dir'] == fsm['GetUpTrack']['policy_dir']
    assert {key: blend[key] for key in CHOSEN} == CHOSEN and blend['ramp'] == .7 and blend['legs'] == 0.
    assert blend['arm_joints'] == ARMS
    assert blend['transitions'] == fsm['GetUpTrack']['transitions'] == {'Velocity': 'RT + A.on_pressed',
                                                                       'Passive': 'LT + B.on_pressed'}
    track = fsm['GetUpTrack']
    assert track['fall_tilt'] == 0 and track['timeout'] == 'handover'
    # the GetUp state's own settings stay the getup patch's
    assert fsm['GetUp']['timeout'] == 10. and fsm['GetUp']['fall_tilt'] == 0


@pytest.mark.skipif(not (TRACKER / 'metadata.json').is_file(), reason='the tracker stage of branch 73 is not here')
def test_the_tracker_exports_frame_and_the_models_targets_are_the_environments(modules):
    """export_g1.handover_params of the tracker (exp-84b0540b @ 570, A493 v3) at frame 18: its joints and gravity are
    getup_chain's goal (clip_joints, clip_gravity), seconds the clip's rest plus the hold; chain_model.Blend's targets
    from a random A target are the environment's PD targets (_target of the applied action, as getup_chain.run takes
    them), the last one handover.yaml's target; the arm joints are the environment's."""
    getup_chain, chain_model = modules
    from humanoid_sim import export_g1
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_getup', 'g1_29dof', stage_task_config(TRACKER))
    try:
        raw = env.unwrapped
        frame = export_g1.handover_params(raw, 18)
        assert list(raw.arm_joints) == ARMS and frame['locked'] == LOCKED
        assert np.allclose(frame['joints'], raw.clip_joints[18]) and np.allclose(frame['gravity'], raw.clip_gravity[18])
        assert frame['seconds'] == pytest.approx((raw.clip_end - 18) * raw.dt)
        assert frame['clip_seconds'] == pytest.approx((raw.clip_last - 18) * raw.dt)
        with pytest.raises(ValueError):
            export_g1.handover_params(raw, raw.clip_last + 1)
        from humanoid_sim.tasks.config import G1_GETUP_PROFILE, g1_config
        config = {**g1_config(G1_GETUP_PROFILE), 'getup': True, 'pd_bound': None}
        deploy = export_g1.deploy_config(raw, config)
        cfg = dict(arm_joints=ARMS, ramp=.7, legs=0., **CHOSEN)
        blend = chain_model.Blend(cfg, deploy, frame)
        rng = np.random.default_rng(18)
        start = raw._target(raw.applied_action(rng.uniform(-3, 3, raw.model.nu)))  # an A target
        blend.enter(start)
        share, late = getup_chain.pd_shares(raw.model.nu, raw.arm_joints, {**getup_chain.PD_DEFAULTS, **CHOSEN}, DT)
        goal = raw.clip_joints[18]
        for k in range(1, blend.steps + 1):
            progress, kp, kd = getup_chain.pd_schedule(k, blend.steps, share, late,
                                                       {**getup_chain.PD_DEFAULTS, **CHOSEN})
            import getup_roll_clip
            wanted = start + getup_roll_clip.smoothstep(progress) * (goal - start)
            target = raw._target(raw.applied_action((wanted - raw.home) / raw.scale))
            q, gain_p, gain_d = blend.at(k)
            assert np.abs(q - target).max() < 1e-5, k
            assert np.abs(gain_p - raw.kp * kp).max() < 1e-4 and np.abs(gain_d - raw.kd * kd).max() < 1e-5
        assert np.abs(blend.at(blend.steps)[0] - np.array(frame['target'])).max() < 1e-5
    finally:
        env.close()


@pytest.mark.skipif(not (TRACKER_EXPORT / 'handover.yaml').is_file(), reason='the tracker export is not here')
def test_the_tracker_export_carries_its_clip_and_frame():
    """The export folder of the tracker: deploy.yaml's track_clip comment (run.sh's check), handover.yaml at frame 18,
    the ONNX properties naming the clip and the frame."""
    from humanoid_sim import export_g1
    text = (TRACKER_EXPORT / 'deploy.yaml').read_text(encoding='utf-8')
    assert any(line.startswith('# track_clip: ') for line in text.splitlines())
    assert any(line.startswith('# g1_getup: ') for line in text.splitlines())
    frame = yaml.safe_load((TRACKER_EXPORT / 'handover.yaml').read_text(encoding='utf-8'))
    assert frame['frame'] == 18 and len(frame['joints']) == 29 and frame['free'] == [j for j in range(29)
                                                                                     if j not in LOCKED]
    properties = export_g1.onnx_properties(TRACKER_EXPORT / 'policy.onnx')
    assert properties['track_clip'] == frame['clip'] and properties['handover_frame'] == '18'
