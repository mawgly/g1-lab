"""Interpretable metrics; no unsupported human-likeness scalar.

Contact events require at least 40ms sustained support to reject chatter.
Step length is planar distance between consecutive opposite-foot touchdowns;
on turns it is not forward stride length. Mechanical work is not battery use.
"""
import numpy as np


def compute_metrics(trajectory: dict, robot_spec=None) -> dict:
    t=np.asarray(trajectory['time'],dtype=float)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Trajectory requires finite strictly increasing time')
    n=len(t)
    def array(key, shape=None):
        a=np.asarray(trajectory[key],dtype=float)
        if a.shape[0]!=n or not np.isfinite(a).all() or (shape and a.shape[1:]!=shape):
            raise ValueError(f'Invalid trajectory {key}')
        return a
    contacts=array('contacts',(2,)).astype(bool)
    feet=array('foot_positions',(2,3));slip=array('contact_speed',(2,))
    qvel=array('joint_velocity');torque=array('torque');actions=array('actions')
    if qvel.shape!=torque.shape or qvel.ndim!=2 or actions.ndim!=2:
        raise ValueError('Invalid joint/action trajectory dimensions')
    com=array('com',(3,));tilt=array('tilt');velocity=array('velocity',(2,));commands=array('commands',(3,))
    duration=float(t[-1]-t[0]); dt=np.diff(t)
    events=[]
    for side in range(2):
        starts=np.flatnonzero(np.diff(contacts[:,side].astype(int),prepend=0)==1)
        for start in starts:
            if start==0:continue  # initial support is not a new step
            end=start
            while end+1<n and contacts[end+1,side]:end+=1
            if t[end]-t[start]+1e-9>=.04:
                events.append((int(start),side))
    events.sort()
    opposite=[(a,b) for a,b in zip(events,events[1:]) if a[1]!=b[1]]
    lengths=[float(np.linalg.norm(feet[b[0],b[1],:2]-feet[a[0],a[1],:2])) for a,b in opposite]
    def mean(value):return float(np.mean(value)) if np.size(value) else None
    def integral(value):
        return float(np.sum(.5*(value[1:]+value[:-1])*dt)) if n>1 else 0.
    power=np.abs(torque*qvel).sum(axis=1)
    rates=np.diff(actions,axis=0)/dt[:,None] if n>1 else np.empty(0)
    return dict(schema_version=1,duration_s=duration,fell=bool(trajectory['fell']),
                step_count=len(events),cadence_steps_per_min=60*len(events)/duration if duration else None,
                mean_step_length_m=mean(lengths),
                alternation_fraction=len(opposite)/(len(events)-1) if len(events)>1 else None,
                left_support_fraction=mean(contacts[:,0]),right_support_fraction=mean(contacts[:,1]),
                contact_slip_speed_m_s=mean(slip[contacts]),
                com_vertical_std_m=float(np.std(com[:,2])),tilt_rms_rad=float(np.sqrt(np.mean(tilt**2))),
                velocity_error_rmse_m_s=float(np.sqrt(np.mean(np.sum((velocity-commands[:,:2])**2,axis=1)))) if trajectory.get('commanded',True) else None,
                action_rate_rms_s_inv=float(np.sqrt(np.mean(rates**2))) if rates.size else None,
                absolute_mechanical_work_j=integral(power),human_likeness_score=None)
