import pytest
from humanoid_sim.lab import Laboratory
from humanoid_sim.evolution.manager import CampaignManager


def setup_campaign(tmp_path,budget=8192):
    lab=Laboratory(tmp_path/'original')
    manager=CampaignManager(lab)
    config=dict(robot_id='g1_29dof',task_id='g1_velocity',envs=1,n_steps=128,batch_size=64,steps=1025)
    branches=[{'name':'Контроль','changes':{}}, {'name':'Меньше LR','changes':{'learning_rate':.0001}},
              {'name':'Плавность','changes':{'task_config':{'reward_weights':{'action_change':.02}}}},
              {'name':'Наклон','changes':{'task_config':{'reward_weights':{'tilt':2.}}}}]
    return lab,manager,config,branches


def test_campaign_reserves_rollout_rounded_budget_and_enqueues_once(tmp_path):
    lab,manager,config,branches=setup_campaign(tmp_path)
    campaign=manager.create(config,branches,budget_steps=8192)
    assert campaign['reserved_steps']==4*1152
    first=manager.advance(campaign['id'])
    second=CampaignManager(lab).advance(campaign['id'])
    assert first['branches']==second['branches']
    assert len(lab.queue()['items'])==4
    assert len(lab.entries())==5
    assert all(b['state']=='pending' for b in second['branches'])


def test_invalid_budget_or_model_mutation_creates_nothing(tmp_path):
    lab,manager,config,branches=setup_campaign(tmp_path)
    with pytest.raises(ValueError):manager.create(config,branches,budget_steps=4096)
    branches[1]['changes']={'robot_id':'humanoid'}
    with pytest.raises(ValueError):manager.create(config,branches,budget_steps=8192)
    assert len(lab.entries())==1


def test_stop_campaign_cancels_pending_without_stopping_other_experiments(tmp_path):
    lab,manager,config,branches=setup_campaign(tmp_path)
    campaign=manager.create(config,branches,budget_steps=8192)
    manager.advance(campaign['id'])
    unrelated=lab.create({'name':'Unrelated'})
    lab.enqueue([unrelated['id']])
    manager.stop(campaign['id'])
    assert next(i for i in lab.queue()['items'] if i['id']==unrelated['id'])['state']=='pending'
    assert not (lab.root/'stop.request').exists()
    assert manager.advance(campaign['id'])['state']=='stopped'


def test_recover_after_created_branch_before_campaign_save(tmp_path,monkeypatch):
    import humanoid_sim.evolution.manager as module
    lab,manager,config,branches=setup_campaign(tmp_path)
    campaign=manager.create(config,branches,budget_steps=8192)
    original=module.write_json
    def crash(path,value):
        if path.name==campaign['id']+'.json':raise OSError('simulated lost process')
        return original(path,value)
    monkeypatch.setattr(module,'write_json',crash)
    with pytest.raises(OSError):manager.advance(campaign['id'])
    assert len(lab.entries())==2
    monkeypatch.setattr(module,'write_json',original)
    restored=CampaignManager(lab).advance(campaign['id'])
    assert len(lab.entries())==5
    assert len({b['experiment_id'] for b in restored['branches']})==4


def test_failed_branch_is_not_silently_restarted(tmp_path):
    from humanoid_sim.core import write_json
    lab,manager,config,branches=setup_campaign(tmp_path)
    campaign=manager.create(config,branches,budget_steps=8192)
    manager.advance(campaign['id'])
    queue=lab.queue()
    queue['items'][0].update(state='failed',error='bad state')
    write_json(lab.root/'queue.json',queue)
    updated=manager.advance(campaign['id'])
    assert updated['state']=='needs_attention'
    assert updated['branches'][0]['error']=='bad state'
    assert len(lab.queue()['items'])==4
