import numpy as np
from humanoid_sim.tasks.catalog import make_task
from humanoid_sim.motion.trajectory import MotionRecorder
from humanoid_sim.motion.metrics import compute_metrics


def test_recorder_captures_real_g1_state_without_changing_physics():
    env=make_task('g1_velocity','g1_29dof',{})
    try:
        env.reset(seed=2)
        recorder=MotionRecorder(env,'g1_29dof')
        for _ in range(10):
            env.step(np.zeros(29))
            before=env.data.qpos.copy()
            recorder.capture(np.zeros(29))
            np.testing.assert_array_equal(before,env.data.qpos)
        t=recorder.finish(False)
        assert t['foot_positions'].shape==(10,2,3)
        result=compute_metrics(t)
        assert np.isfinite(result['absolute_mechanical_work_j'])
        assert result['duration_s']>0
    finally:env.close()
