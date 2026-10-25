"""scripts/lab/probes/chatter.py, torque chattering (iteration 42): the split by frequency on synthetic signals, the
sensor noise on the observation, the summary, the worst joint and the gate against the reference on made-up runs, and a
short run on a small arm_targets stage."""
import math

import numpy as np
import pytest
import torch

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from test_g1_arm_targets import ARMS, _stage
from test_g1_lock_wrists import lab_probe

NAMES = ['joint%02d' % k for k in range(29)]
WRISTS = (20, 21, 27, 28)  # locked: their PD target never moves


@pytest.fixture
def ch(monkeypatch):
    return lab_probe('chatter', monkeypatch)


def made_up_run(scenario, seed, torque=4., target=.1, share=.4, fell=None, wrist_torque=.2):
    """A chatter.run result: joint 3 has the largest high-frequency torque, joint 4 the largest target jitter, the
    locked wrists a target that never moves (no shares) and a torque of their own."""
    out = dict(stage='x', scenario=scenario, seed=seed, delay=0., delay_steps=0, fell=fell, names=NAMES,
               peak=[100.] * 29)
    if fell is not None:
        return out
    hf, rad, shares, torque_shares = [1.] * 29, [.01] * 29, [share] * 29, [.2] * 29
    hf[3], rad[4] = torque, target
    for w in WRISTS:
        hf[w], rad[w], shares[w], torque_shares[w] = wrist_torque, None, None, .99
    return dict(out, action_hf_share=shares, action_hf_rms_rad=rad, torque_hf_share=torque_shares,
                torque_hf_rms_nm=hf, torque_rms_nm=[5.] * 29, mean_abs_delta_action=[.2] * 29)


def chatter_runs(jobs, **values):
    """Made-up results of chatter jobs, every run with the same values."""
    return [made_up_run(job['scenario'], job['seed'], **values) for job in jobs]


def test_band_splits_a_synthetic_signal_at_the_cutoff(ch):
    """A 2 Hz sine of amplitude 3 and a 12 Hz one of amplitude 1 on a line, 9 s at 500 Hz: a tenth of the variance is
    above 10 Hz, its RMS 1/√2; the line goes; a constant column has no share; 30 Hz alone is all above. At 50 Hz with a
    5 Hz cutoff the same."""
    t = np.arange(4500) / 500.
    x = np.stack([3 * np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 12 * t) + .5 + .2 * t, np.full_like(t, .7),
                  .3 * np.sin(2 * np.pi * 30 * t)], axis=1)
    share, rms = ch.band(x, 500., 10.)
    assert share[0] == pytest.approx(.1, rel=1e-2) and rms[0] == pytest.approx(1 / math.sqrt(2), rel=1e-2)
    assert math.isnan(share[1]) and math.isnan(rms[1])
    assert share[2] == pytest.approx(1., abs=1e-5) and rms[2] == pytest.approx(.3 / math.sqrt(2), rel=1e-3)
    one, _ = ch.band(x[:, 0], 500., 10.)  # one series
    assert one.shape == (1,) and one[0] == pytest.approx(share[0], rel=1e-12)
    t = np.arange(450) / 50.
    share, rms = ch.band(np.sin(2 * np.pi * t) + .5 * np.sin(2 * np.pi * 8 * t), 50., 5.)
    assert share[0] == pytest.approx(.2, rel=1e-2) and rms[0] == pytest.approx(.5 / math.sqrt(2), rel=1e-2)
    assert ch.band(np.sin(2 * np.pi * t), 50., 5.)[0][0] < 1e-6  # all below the cutoff


def test_the_noise_falls_on_the_sensors_of_the_newest_frame(ch):
    """chatter.noisy: angular velocity, gravity, joint angles (plus the encoder bias) and speeds of the newest frame
    get the CUDA task's uniform noise; the command, phase, last action, the extra terms, the older frames and the
    critic extras stay exact."""
    from humanoid_sim.g1_tensor_env import NOISE
    env = G1VelocityEnv(**ARMS)
    try:
        env.reset(seed=3, options={'command': [.5, 0., 0.]})
        for _ in range(3):
            env.step(np.zeros(29))
        exact = env._observation('replace')
        bias = np.full(29, .005)
        noisy = ch.noisy(env, env._observation, np.random.default_rng(1), bias)('replace')
        nu, history, extra = env.model.nu, env.history, env.extra_sizes
        index = task.history_index(nu, history, extra)
        spans = task.deploy_terms(nu, extra)
        newest = np.asarray(index[-1])
        diff = (noisy - exact)[newest]
        limits = {0: NOISE['angular_velocity'], 1: NOISE['gravity'], 4: NOISE['joint_position'],
                  5: NOISE['joint_velocity']}
        for k, (start, end) in enumerate(spans):
            part = diff[start:end] - (bias if k == 4 else 0.)
            if k in limits:
                assert np.abs(part).max() <= limits[k] + 1e-5 and np.abs(part).max() > limits[k] / 10
            else:
                np.testing.assert_allclose(part, 0., atol=1e-6)
        older = np.setdiff1d(np.arange(env.actor_inputs), newest)
        np.testing.assert_array_equal(noisy[older], exact[older])
        np.testing.assert_array_equal(noisy[env.actor_inputs:], exact[env.actor_inputs:])
    finally:
        env.close()


def test_the_summary_combines_runs_and_leaves_the_locked_wrists_out_of_the_worst(ch):
    """Per joint: per scenario over the seeds, then over the scenarios with equal weights, RMS values as the root of
    the mean square; the worst joint among the joints whose target moves; the carry scenario apart; the peaks divide
    the torque RMS."""
    jobs = ch.jobs('x', arms=True)
    assert len(jobs) == 21 and {job['scenario'] for job in jobs[18:]} == {'carry'}
    torque = {74000: 3., 74001: 4., 74002: 5.}
    runs = [made_up_run(job['scenario'], job['seed'], torque=torque[job['seed']], wrist_torque=50.,
                        share=.5 if job['scenario'] == 'stand' else .3) for job in jobs[:18]]
    runs += [made_up_run('carry', 74000 + k, torque=9.) for k in range(3)]
    summary = ch.summary_of(runs)
    assert summary['torque_hf_rms_nm'] == pytest.approx(math.sqrt(50 / 3))
    assert summary['worst_joint'] == summary['worst_joints']['torque_hf_rms_nm'] == 'joint03'  # not the wrists' 50
    assert summary['torque_hf_share'] == pytest.approx(.2) and summary['worst_joints']['action_hf_rms_rad'] == 'joint04'
    assert summary['action_hf_share'] == pytest.approx((.5 + 5 * .3) / 6)
    assert summary['torque_rms_peak_share'] == pytest.approx(.05)
    assert summary['torque_hf_peak_share'] == pytest.approx(math.sqrt(50 / 3) / 100)
    assert (summary['runs'], summary['falls']) == (18, [])
    values = ch.joint_values(runs, ch.BASE)
    assert values['torque_hf_rms_nm'][20] == pytest.approx(50.) and math.isnan(values['action_hf_share'][20])
    stand = [made_up_run(job['scenario'], job['seed'], torque=2. if job['scenario'] == 'stand' else 4.)
             for job in jobs[:18]]
    assert ch.summary_of(stand)['torque_hf_rms_nm'] == pytest.approx(math.sqrt(14.))  # (2² + 5·4²) / 6
    result = ch.summarize(runs, chatter_runs(ch.reference_jobs(0.)))
    assert result['carry']['torque_hf_rms_nm'] == pytest.approx(9.) and result['carry']['runs'] == 3
    assert result['scenarios']['stand']['action_hf_share'] == pytest.approx(.5)
    assert result['joints']['names'] == NAMES and result['joints']['action_hf_share'][20] is None
    assert result['summary']['ok'] and result['reasons'] == []


def test_the_gate_against_the_reference(ch):
    """ok while the worst joint's torque_hf_rms_nm and action_hf_rms_rad stay within 1.5 x the reference's and no run
    fell; the reasons name both joints."""
    jobs, reference = ch.jobs('x'), chatter_runs(ch.reference_jobs(0.))

    def verdict(**values):
        result = ch.summarize(chatter_runs(jobs, **values), reference)
        return result['summary']['ok'], result['reasons']

    assert verdict(torque=6., target=.15) == (True, [])
    assert verdict(torque=6.01) == (False, ['torque_hf_rms_nm 6.01 (joint03) > 1.5 x v0 4 (joint03)'])
    assert verdict(target=.151) == (False, ['action_hf_rms_rad 0.151 (joint04) > 1.5 x v0 0.1 (joint04)'])
    assert verdict(share=.99) == (True, [])  # the shares are printed, not judged
    runs = chatter_runs(jobs)
    runs[4] = made_up_run(jobs[4]['scenario'], jobs[4]['seed'], fell=210)
    result = ch.summarize(runs, reference)
    assert not result['summary']['ok'] and result['reasons'] == ['падений 1 (ходьба 0.5 м/с зерно 74001)']
    assert result['summary']['falls'] == [dict(scenario='walk05', seed=74001, step=210)]
    summary = result['summary']
    assert summary['reference']['torque_hf_rms_nm'] == 4. and summary['gate'] == ch.gate_text()
    assert set(ch.KEYS) <= set(summary) and {'worst_joint', 'reference', 'ok', 'gate'} <= set(summary)
    assert 'дрожание моментов' in ch.line(result) and ch.line(result).endswith('— не ok')


def test_a_short_run_on_a_small_stage(ch, tmp_path, monkeypatch):
    """chatter.run on a small arm_targets stage (0.6 s; standing, stop-start, the carry pose): every field per joint,
    none of the shares for the locked wrists, the training delay drawn, the motor model put back, the same numbers for
    the same seed, others for another."""
    for key, value in dict(WARMUP=.2, CARRY_MOVE_AT=.1, CARRY_FROM=.4).items():
        monkeypatch.setattr(ch, key, value)
    _, stage = _stage(tmp_path, ARMS)
    assert ch.has_arm_targets(str(stage)) and ch.stage_delay(str(stage)) == .02
    threads, original = torch.get_num_threads(), task.motor_torque
    try:
        runs = [ch.run(dict(job, seconds=.6)) for job in ch.jobs(str(stage), arms=True, seeds=1)]
        again = ch.run(dict(stage=str(stage), scenario='stand', seed=ch.SEED, seconds=.6, delay=None))
        other = ch.run(dict(stage=str(stage), scenario='stand', seed=ch.SEED + 1, seconds=.6, delay=None))
    finally:
        torch.set_num_threads(threads)
    assert task.motor_torque is original
    assert [r['scenario'] for r in runs] == list(ch.BASE) + ['carry']
    for r in runs:
        assert r['fell'] is None and r['delay'] == .02 and 0 <= r['delay_steps'] <= 10
        assert all(len(r[key]) == 29 for key in ch.FIELDS)
        assert all(r['action_hf_share'][w] is None and r['mean_abs_delta_action'][w] == 0. for w in WRISTS)
        assert all(v is not None and v > 0 for v in r['torque_rms_nm'] + r['torque_hf_rms_nm'])
        assert all(0 <= v <= 1 for v in r['torque_hf_share'])
    assert again == runs[0] and other['torque_hf_rms_nm'] != again['torque_hf_rms_nm']
    result = ch.summarize(runs, runs)  # the stage as its own reference
    assert result['summary']['ok'] and result['carry']['runs'] == 1
