import numpy as np
import pytest
from humanoid_sim.lab import validate_config
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv


def test_catalog_describes_task_fields_and_defaults():
    from humanoid_sim.tasks.catalog import catalog
    robots=catalog()
    g1=next(r for r in robots if r['robot_id']=='g1_29dof')['tasks'][0]
    fields=g1['parameters']
    from humanoid_sim.tasks.config import G1_COMMANDS, G1_GAIT, G1_REWARD_WEIGHTS
    assert [f['key'] for f in fields]==['real_robot', 'critic_extras', 'history', 'training_realism', 'heading_hold',
                                         'stand_heading_hold', 'side_heading_hold', 'phase_stop', 'sole_contact',
                                         'feet_flat_stance', 'random_phase', 'action_delay', 'arm_perturb',
                                         'arm_perturb_radians', 'arm_perturb_contact_mask', 'force_push',
                                         'motor_friction', 'lock_wrists', 'knee_floor', 'foot_collision',
                                         'hand_collision', 'height_command', 'upper_body_slot', 'arm_blind',
                                         'action_limit', *G1_COMMANDS, *G1_GAIT, *G1_REWARD_WEIGHTS]
    assert fields[0]['type']=='bool' and fields[0]['default'] is False  # simulation profile by default
    assert fields[1]['type']=='bool' and fields[1]['default'] is False
    assert fields[2]['default']==1 and fields[2]['min']==1 and fields[2]['max']==10  # no history by default
    assert fields[3]['default']==1 and fields[3]['min']==0 and fields[3]['max']==1
    assert fields[4]['type']=='bool' and fields[4]['default'] is True and fields[4]['group'] is None
    assert fields[5]['type']=='bool' and fields[5]['default'] is False  # new runs: a stand command stays zero
    assert fields[6]['type']=='bool' and fields[6]['default'] is False  # new runs: a side step is not steered
    assert fields[7]['type']=='bool' and fields[7]['default'] is True  # new runs: the phase stops as on the robot
    assert fields[8]['type']=='bool' and fields[8]['default'] is False  # reward contact by the ankle height unless asked
    assert fields[9]['type']=='bool' and fields[9]['default'] is False  # feet_flat on both feet unless asked
    assert fields[10]['type']=='bool' and fields[10]['default'] is True  # new runs: episodes start on either foot
    assert fields[11]['default']==0 and fields[11]['min']==0 and fields[11]['max']==.05  # no action delay unless asked
    assert fields[12]['default']==0 and fields[12]['min']==0 and fields[12]['max']==.5  # no arm perturbation unless asked
    # The original arm poses (±0.4 rad) and every arm contact charged unless asked.
    assert fields[13]['default']==.4 and fields[13]['min']==.1 and fields[13]['max']==1.6
    assert fields[14]['type']=='bool' and fields[14]['default'] is False
    assert fields[15]['default']==0 and fields[15]['min']==0 and fields[15]['max']==2  # no force pushes unless asked
    assert fields[16]['default']==0 and fields[16]['min']==0 and fields[16]['max']==2  # measured motor friction unless asked
    assert fields[17]['type']=='bool' and fields[17]['default'] is False  # the policy drives the wrists unless asked
    # No knee floor unless asked: a number field the browser may leave empty (null).
    assert 'type' not in fields[18] and fields[18]['default'] is None and (fields[18]['min'],fields[18]['max'])==(0,1)
    assert fields[19]['type']=='bool' and fields[19]['default'] is False  # the feet pass through each other unless asked
    assert fields[20]['type']=='bool' and fields[20]['default'] is False  # the hands touch nothing unless asked
    # No height command and no upper-body slot unless asked (the original observation layout).
    assert all(fields[k]['type']=='bool' and fields[k]['default'] is False for k in (21, 22))
    assert fields[23]['type']=='bool' and fields[23]['default'] is False  # the policy sees its arms unless asked
    assert fields[24]['group'] is None and fields[24]['default']==1  # the original action contract by default
    assert next(f for f in fields if f['key']=='action_change')['default']==.01
    assert next(f for f in fields if f['key']=='torso')['default']==2  # recommended for new experiments
    weights=fields[25+len(G1_COMMANDS)+len(G1_GAIT):]
    assert all(f['group']=='reward_weights' and f['min']==0 and f['max']==100 for f in weights)
    assert [(f['group'],f['default']) for f in fields[25:25+len(G1_COMMANDS)]]==[('commands',v) for v in G1_COMMANDS.values()]
    assert next(r for r in robots if r['robot_id']=='so101')['tasks'][0]['parameters']==[]


def test_g1_reward_ablation_changes_only_selected_component():
    first=G1VelocityEnv()
    second=G1VelocityEnv(reward_weights={'action_change':0.})
    try:
        first.reset(seed=6);second.reset(seed=6)
        a=first.step(np.full(29,.2));b=second.step(np.full(29,.2))
        np.testing.assert_array_equal(a[0],b[0])
        assert b[4]['reward_components']['action_change']==0
        assert b[1]-a[1]==pytest.approx(-a[4]['reward_components']['action_change'])
    finally:
        first.close();second.close()


def test_task_config_rejects_unknown_nonfinite_and_wrong_task_parameters():
    base=dict(robot_id='g1_29dof',task_id='g1_velocity')
    config=validate_config({**base,'task_config':{'reward_weights':{'action_change':0}}})
    assert config['task_config']['reward_weights']['action_change']==0
    for value in ({'reward_weights':{'invented':1}}, {'reward_weights':{'tilt':float('nan')}},
                  {'reward_weights':{'fall':-1}}, {'max_episode_steps':1}):
        with pytest.raises(ValueError):
            validate_config({**base,'task_config':value})
    with pytest.raises(ValueError):
        validate_config({'task_config':{'reward_weights':{'tilt':1}}})
    from humanoid_sim.tasks.config import G1_COMMANDS, G1_GAIT, G1_REWARD_WEIGHTS
    assert validate_config(base)['task_config']=={'reward_weights':G1_REWARD_WEIGHTS,'heading_hold':False,
                                                  'commands':G1_COMMANDS,'gait':G1_GAIT,'action_limit':1.,
                                                  'real_robot':False,'critic_extras':False,'training_realism':1.,
                                                  'history':1,'stand_heading_hold':True,'phase_stop':False,
                                                  'sole_contact':False,'random_phase':False,'action_delay':0.,
                                                  'arm_perturb':0.,'arm_perturb_radians':.4,
                                                  'arm_perturb_contact_mask':False,'lock_wrists':False,'knee_floor':None,
                                                  'force_push':0.,'feet_flat_stance':False,'foot_collision':False,
                                                  'side_heading_hold':True,'motor_friction':0.,
                                                  'height_command':False,'upper_body_slot':False,'arm_blind':False,
                                                  'hand_collision':False}
    for history in (0, 2.5, 11):  # whole frames in [1, 10], and only for the real-robot profile
        with pytest.raises(ValueError):
            validate_config({**base,'task_config':{'real_robot':True,'history':history}})
    with pytest.raises(ValueError):
        validate_config({**base,'task_config':{'history':5}})
    assert validate_config({**base,'task_config':{'real_robot':True,'history':5.}})['task_config']['history']==5
    for gait in ({'period':.2}, {'stance_fraction':.9}, {'cadence':1}):
        with pytest.raises(ValueError):
            validate_config({**base,'task_config':{'gait':gait}})
    assert validate_config({**base,'task_config':{'heading_hold':True}})['task_config']['heading_hold'] is True
    with pytest.raises(ValueError):
        validate_config({**base,'task_config':{'heading_hold':1}})
    narrowed=validate_config({**base,'task_config':{'commands':{'min_speed':.3,'max_speed':1}}})['task_config']
    assert narrowed['commands']=={'min_speed':.3,'max_speed':1.,'stand_probability':.2,'lateral_speed':0.,'yaw_rate':0.,
                                  'straight_probability':0.,'side_share':0.,'side_stop':0.,'side_speed':0.,
                                  'crouch_share':[0.,0.],'crouch_min':[.64,.70],'crouch_speed':1.}
    for commands in ({'min_speed':1.2,'max_speed':1.}, {'max_speed':2.5}, {'stand_probability':-.1}, {'min_speed':-1.5},
                     {'max_speed':float('inf')}, {'turn':1}):
        with pytest.raises(ValueError):
            validate_config({**base,'task_config':{'commands':commands}})


def test_checkpoint_preserves_reward_config_after_run_settings_change(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.core import write_json,save_stage,load_stage,read_json
    config=dict(robot_id='g1_29dof',task_id='g1_velocity',task_config={'reward_weights':{'action_change':0}})
    write_json(tmp_path/'run_config.json',config)
    env=VecNormalize(DummyVecEnv([lambda:G1VelocityEnv(**config['task_config'])]))
    try:
        policy=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu')
        stage=save_stage(tmp_path,policy,env)
        write_json(tmp_path/'run_config.json',{**config,'task_config':{'reward_weights':{'action_change':1}}})
        _,norm=load_stage(stage)
        try:
            assert norm.venv.envs[0].unwrapped.reward_weights['action_change']==0
            assert read_json(stage/'metadata.json')['task_config']['reward_weights']['action_change']==0
        finally:
            norm.close()
    finally:
        env.close()
