from argparse import Namespace
import pytest
from humanoid_sim.lab import Laboratory, validate_config
from humanoid_sim.core import write_json, read_json, list_stages, load_stage, evaluate_stage
from humanoid_sim.train import train


def test_robot_task_backend_validation():
    assert validate_config({'robot_id':'g1_29dof','task_id':'g1_velocity'})['robot_id'] == 'g1_29dof'
    for options in ({'robot_id':'so101','task_id':'humanoid_v5'},
                    {'robot_id':'so101','task_id':'arm_reach','backend':'gpu'}):
        with pytest.raises(ValueError):
            validate_config(options)


@pytest.mark.parametrize('robot,task', [('g1_29dof','g1_velocity'),('so101','arm_reach')])
def test_train_save_evaluate_and_branch(tmp_path, robot, task):
    lab = Laboratory(tmp_path/'original')
    entry = lab.create(dict(robot_id=robot,task_id=task,envs=1,steps=1024,n_steps=128,batch_size=128,n_epochs=1))
    run = lab.resolve(entry['id'])
    train(Namespace(run_dir=run, config=run/'run_config.json', parent_stop=None))
    assert read_json(run/'status.json')['state'] == 'completed'
    stage = list_stages(run)[-1]
    assert read_json(stage/'metadata.json')['steps'] >= 1024
    result = evaluate_stage(stage, episodes=1, max_steps=10)
    assert result['robot_id'] == robot
    if robot == 'so101':
        assert result['mean_forward_m'] is None
        assert result['episodes'][0]['fell'] is None
        assert result['mean_distance_m'] >= 0
    child = lab.create(dict(robot_id=robot,task_id=task,source_id=entry['id']))
    assert list_stages(lab.resolve(child['id']))
    with pytest.raises(ValueError):
        lab.create(dict(source_id=entry['id']))
