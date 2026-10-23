from copy import deepcopy
from humanoid_sim.core import write_json
from humanoid_sim.lab import Laboratory
from humanoid_sim.evolution.manager import CampaignManager


def evaluated(tmp_path,budget=8192,error=.2):
    manager=CampaignManager(Laboratory(tmp_path/'original'))
    campaign=manager.create(dict(robot_id='g1_29dof',task_id='g1_velocity',envs=1,n_steps=128,batch_size=64,steps=1024),
        [{'name':'control','changes':{}},{'name':'candidate','changes':{'learning_rate':.0001}}],
        budget_steps=budget,max_rounds=4,minimum_improvement=.02)
    finish(campaign,error)
    write_json(manager._path(campaign['id']),campaign)
    return manager,campaign


def finish(campaign,error):
    for index,branch in enumerate(campaign['branches']):
        branch.update(experiment_id='exp-'+str(index+campaign.get('round',0)*10).zfill(16),
                      evaluation_stage='stage_000000001024')
    ids=[b['experiment_id'] for b in campaign['branches']]
    campaign.update(state='evaluated',selection=dict(selected=[ids[-1]],decisions={
        ids[0]:dict(metrics={'velocity_error_rmse_m_s':.4,'fall_rate':0.}),
        ids[-1]:dict(metrics={'velocity_error_rmse_m_s':error,'fall_rate':0.})}))


def test_round_preserves_control_and_parent_and_reserves_once(tmp_path):
    manager,campaign=evaluated(tmp_path)
    planned=manager.next_round(campaign['id'])
    assert planned['round']==1 and planned['state']=='planned'
    assert len(planned['history'])==1
    assert planned['reserved_steps']==4096
    assert planned['branches'][0]['config']['source_id']==campaign['branches'][0]['experiment_id']
    assert planned['branches'][1]['config']['source_stage']==1024
    assert planned['branches'][1]['config']['learning_rate']==.0001
    assert manager.next_round(campaign['id'])==planned


def test_budget_exhaustion_and_two_round_plateau(tmp_path):
    manager,campaign=evaluated(tmp_path/'budget',budget=2048)
    assert manager.next_round(campaign['id'])['stop_reason']=='budget'
    manager,campaign=evaluated(tmp_path/'plateau',error=.4)
    next_one=manager.next_round(campaign['id'])
    assert next_one['plateau_rounds']==1
    finish(next_one,.4)
    write_json(manager._path(campaign['id']),next_one)
    stopped=manager.next_round(campaign['id'])
    assert stopped['state']=='completed' and stopped['stop_reason']=='plateau'
    assert stopped['reserved_steps']==4096


def test_more_falls_are_not_counted_as_progress(tmp_path):
    manager,campaign=evaluated(tmp_path,error=.1)
    candidate=campaign['branches'][1]['experiment_id']
    campaign['selection']['decisions'][candidate]['metrics']['fall_rate']=.2
    write_json(manager._path(campaign['id']),campaign)
    assert manager.next_round(campaign['id'])['plateau_rounds']==1
