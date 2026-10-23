import pytest
from humanoid_sim.core import write_json, read_json, RunLock
from humanoid_sim.lab import Laboratory
from humanoid_sim.evolution.manager import CampaignManager


def prepared(tmp_path):
    lab=Laboratory(tmp_path/'original');manager=CampaignManager(lab)
    campaign=manager.create(dict(robot_id='g1_29dof',task_id='g1_velocity',envs=1,n_steps=128,batch_size=64,steps=1024),
        [{'name':'control','changes':{}},{'name':'low-lr','changes':{'learning_rate':.0001}}],budget_steps=2048,
        selection_rules=dict(max_fall_rate=.2,max_velocity_error=.6,min_duration_fraction=.8))
    campaign=manager.advance(campaign['id'])
    for branch in campaign['branches']:
        stage=lab.resolve(branch['experiment_id'])/'stage_000000001024'
        stage.mkdir()
        (stage/'policy.zip').write_bytes(b'policy')
        (stage/'normalize.pkl').write_bytes(b'normalizer')
        write_json(stage/'metadata.json',{'steps':1024})
    queue=lab.queue()
    for item in queue['items']:item['state']='completed'
    write_json(lab.root/'queue.json',queue)
    return lab,manager,campaign


def test_evaluate_reuses_saved_results_after_partial_failure(tmp_path,monkeypatch):
    import humanoid_sim.evaluation.runner as runner
    import humanoid_sim.evolution.selection as selection
    lab,manager,campaign=prepared(tmp_path)
    calls=[]
    def evaluate(stage,split):
        calls.append(str(stage))
        if len(calls)==2:raise RuntimeError('simulated evaluator interruption')
        return {'checkpoint':str(stage),'split':split}
    monkeypatch.setattr(runner,'evaluate_suite',evaluate)
    monkeypatch.setattr(selection,'select_candidates',lambda results,rules:dict(selected=sorted(results),decisions={}))
    with pytest.raises(RuntimeError):manager.evaluate(campaign['id'])
    completed=CampaignManager(lab).evaluate(campaign['id'])
    assert completed['state']=='evaluated'
    assert len(calls)==3
    assert len(completed['selection']['selected'])==2
    manager.advance(campaign['id'])
    manager.evaluate(campaign['id'])
    assert len(calls)==3


def test_evaluation_respects_compute_lock(tmp_path):
    lab,manager,campaign=prepared(tmp_path)
    with RunLock(lab.root/'compute'):
        with pytest.raises(RuntimeError):manager.evaluate(campaign['id'])


def test_stop_between_evaluations_prevents_remaining_work(tmp_path,monkeypatch):
    import humanoid_sim.evaluation.runner as runner
    lab,manager,campaign=prepared(tmp_path)
    calls=[]
    def evaluate(stage,split):
        calls.append(stage)
        manager.stop(campaign['id'])
        return {'checkpoint':str(stage)}
    monkeypatch.setattr(runner,'evaluate_suite',evaluate)
    result=manager.evaluate(campaign['id'])
    assert result['state']=='stopped'
    assert len(calls)==1
    assert 'selection' not in result


def test_checkpoint_changed_after_interruption_is_rejected(tmp_path,monkeypatch):
    import humanoid_sim.evaluation.runner as runner
    lab,manager,campaign=prepared(tmp_path)
    def fail(*args):raise RuntimeError('interrupted')
    monkeypatch.setattr(runner,'evaluate_suite',fail)
    with pytest.raises(RuntimeError):manager.evaluate(campaign['id'])
    branch=read_json(manager._path(campaign['id']))['branches'][0]
    (lab.resolve(branch['experiment_id'])/branch['evaluation_stage']/'policy.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):manager.evaluate(campaign['id'])
