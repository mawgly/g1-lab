import numpy as np
import pytest
from humanoid_sim.motion.metrics import compute_metrics


def trajectory(n=101, dt=.02):
    return dict(time=np.arange(n)*dt, contacts=np.ones((n,2),bool),
                foot_positions=np.zeros((n,2,3)), contact_speed=np.zeros((n,2)),
                joint_velocity=np.zeros((n,3)), torque=np.zeros((n,3)),
                actions=np.zeros((n,3)), com=np.tile([0.,0.,1.],(n,1)),
                tilt=np.zeros(n), velocity=np.zeros((n,2)), commands=np.zeros((n,3)), fell=False)


def test_standing_has_no_invented_steps_or_human_score():
    result=compute_metrics(trajectory())
    assert result['step_count']==0
    assert result['mean_step_length_m'] is None
    assert result['cadence_steps_per_min']==0
    assert result['contact_slip_speed_m_s']==0
    assert result['human_likeness_score'] is None


def test_contact_slip_excludes_swing_and_power_is_not_battery_energy():
    t=trajectory();t['contact_speed'][:]=3;t['contacts'][:,0]=False;t['contact_speed'][:,1]=.2
    t['joint_velocity'][:]=2;t['torque'][:]=4
    result=compute_metrics(t)
    assert result['contact_slip_speed_m_s']==pytest.approx(.2)
    assert result['absolute_mechanical_work_j']==pytest.approx(48)


def test_alternating_touchdowns_and_contact_chatter():
    t=trajectory();t['contacts'][:]=False
    for start,side in [(10,0),(30,1),(50,0),(70,1)]:
        t['contacts'][start:start+10,side]=True
        t['foot_positions'][start:,side,0]=start*.01
    t['contacts'][90,0]=True  # one-frame noise must not count
    result=compute_metrics(t)
    assert result['step_count']==4
    assert result['alternation_fraction']==1
    assert result['mean_step_length_m']==pytest.approx(.2)


def test_short_fall_and_bad_time_are_explicit():
    t=trajectory(n=1);t['fell']=True
    result=compute_metrics(t)
    assert result['fell'] and result['duration_s']==0
    assert result['action_rate_rms_s_inv'] is None
    t=trajectory();t['time'][5]=t['time'][4]
    with pytest.raises(ValueError):compute_metrics(t)
