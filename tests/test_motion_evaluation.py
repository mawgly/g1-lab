from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_sim.tasks.catalog import make_task
from humanoid_sim.core import save_stage
from humanoid_sim.evaluation.runner import evaluate_motion


def test_checkpoint_motion_evaluation_uses_frozen_policy_and_records_seed(tmp_path):
    env=VecNormalize(DummyVecEnv([lambda:make_task('humanoid_v5','humanoid',{})]))
    try:
        model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu')
        stage=save_stage(tmp_path,model,env)
    finally:env.close()
    result=evaluate_motion(stage,episodes=1,seconds=.15,seed_start=82000)
    assert result['episodes'][0]['seed']==82000
    assert result['episodes'][0]['velocity_error_rmse_m_s'] is None
    assert result['episodes'][0]['human_likeness_score'] is None
    assert result['robot_id']=='humanoid'
    assert not (stage/'motion.json').exists()  # source checkpoint remains unchanged


def test_scenario_changes_policy_observations_and_recorded_commands(tmp_path,monkeypatch):
    import numpy as np
    import humanoid_sim.evaluation.scenarios as scenarios
    from humanoid_sim.core import write_json
    write_json(tmp_path/'run_config.json',{'robot_id':'g1_29dof','task_id':'g1_velocity'})
    env=VecNormalize(DummyVecEnv([lambda:make_task('g1_velocity','g1_29dof',{})]))
    try:
        model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu')
        stage=save_stage(tmp_path,model,env)
    finally:env.close()
    scenario=scenarios.Scenario('test',((0.,(0.,0.,0.)),(.04,(1.,0.,0.))))
    monkeypatch.setattr(scenarios,'get_scenario',lambda _:scenario)
    evaluate_motion(stage,episodes=1,seconds=.12,scenario_id='test',trajectory_dir=tmp_path/'traces')
    trace=np.load(next((tmp_path/'traces').glob('*.npz')))
    assert trace['commands'][0,0]==0
    assert trace['commands'][-1,0]==1
