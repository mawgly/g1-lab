"""Read-only MuJoCo trajectory sampling for supported locomotion adapters."""
import mujoco
import numpy as np

ROLES={
    'humanoid':('torso',('left_foot','right_foot')),
    'g1_29dof':('pelvis',('left_ankle_roll_link','right_ankle_roll_link')),
    'g1_12dof':('pelvis',('left_ankle_roll_link','right_ankle_roll_link')),
}


class MotionRecorder:
    def __init__(self,env,robot_id):
        self.env=env.unwrapped
        self.model,self.data=self.env.model,self.env.data
        if robot_id not in ROLES:raise ValueError('No locomotion body roles for this robot')
        torso,feet=ROLES[robot_id]
        self.torso=self.model.body(torso).id
        self.feet=[self.model.body(name).id for name in feet]
        self.commanded=hasattr(self.env,'command')
        self.rows=[]
        self.dofs=np.concatenate([np.arange(self.model.jnt_dofadr[i],self.model.jnt_dofadr[i]+1)
                                 for i in range(self.model.njnt)
                                 if self.model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)])

    def capture(self,action):
        m,d=self.model,self.data
        contacts=np.zeros(2,dtype=bool);speeds=[[],[]]
        force=np.zeros(6);jac=np.zeros((3,m.nv))
        for cid,contact in enumerate(d.contact):
            a,b=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])
            # Ground only: another limb touching a foot is not ground support.
            if 0 not in (a,b):continue
            foot=b if a==0 else a
            if foot not in self.feet:continue
            index=self.feet.index(foot)
            # Contact constraint address can be -1 for an inactive contact.
            if contact.efc_address<0:continue
            mujoco.mj_contactForce(m,d,cid,force)
            if force[0]<5.:continue
            contacts[index]=True
            mujoco.mj_jac(m,d,jac,None,contact.pos,foot)
            speeds[index].append(float(np.linalg.norm((jac@d.qvel)[:2])))
        rotation=d.xmat[self.torso].reshape(3,3)
        vel=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.torso,vel,1)
        self.rows.append(dict(time=float(d.time),contacts=contacts,
                             foot_positions=d.xpos[self.feet].copy(),
                             contact_speed=np.array([np.mean(s) if s else 0. for s in speeds]),
                             joint_velocity=d.qvel[self.dofs].copy(),torque=d.qfrc_actuator[self.dofs].copy(),
                             actions=np.array(action,copy=True),com=(d.xipos*m.body_mass[:,None]).sum(axis=0)/m.body_mass.sum(),
                             tilt=float(np.arccos(np.clip(rotation[2,2],-1,1))),velocity=vel[3:5].copy(),
                             commands=np.array(getattr(self.env,'command',[0,0,0]),copy=True)))

    def finish(self,fell):
        if not self.rows:raise ValueError('No recorded frames')
        return {**{key:np.asarray([row[key] for row in self.rows]) for key in self.rows[0]},
                'fell':bool(fell),'commanded':self.commanded}
