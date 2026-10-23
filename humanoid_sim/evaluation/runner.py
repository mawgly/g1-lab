"""CPU motion evaluation. Results are returned separately from checkpoints."""
import math
import time
from pathlib import Path
import numpy as np
from ..core import load_stage,read_json,stage_task_config
from ..checkpoint_schema import checkpoint_identity
from ..tasks.catalog import make_task
from ..motion.trajectory import MotionRecorder
from ..motion.metrics import compute_metrics
from .statistics import summarize


def run_motion_diagnostics(run_dir,episodes=5,seconds=15.):
    from ..core import list_stages,write_json
    import torch
    torch.set_num_threads(2)
    run=Path(run_dir);stages=list_stages(run)
    if not stages:raise ValueError('No complete checkpoints')
    scored=[(p,read_json(p/'evaluation.json',{}).get('mean_return')) for p in stages]
    scored=[(p,s) for p,s in scored if s is not None and math.isfinite(s)]
    best=max(scored,key=lambda x:x[1])[0] if scored else stages[-1]
    cache={};checkpoints={}
    for role,path in [('baseline',stages[0]),('best',best),('last',stages[-1])]:
        if path not in cache:
            cache[path]=evaluate_motion(path,episodes=episodes,seconds=seconds,seed_start=82000)
        checkpoints[role]=cache[path]
    result=dict(schema_version=1,protocol='motion-v1',created_at=time.time(),
                selection='best selected by original reward, not by gait quality',checkpoints=checkpoints)
    write_json(run/'motion.json',result)
    return result


def evaluate_motion(stage,episodes=5,seconds=15.,seed_start=80000,command=None,trajectory_dir=None,scenario_id=None):
    from .scenarios import get_scenario
    scenario=get_scenario(scenario_id) if scenario_id is not None else None
    if scenario is not None:
        if command is not None:raise ValueError('Choose a fixed command or a scenario, not both')
        command=scenario.command_at(0.)
    if type(episodes) is not int or not 1<=episodes<=1000:
        raise ValueError('episodes must be 1..1000')
    if not math.isfinite(seconds) or not 0<seconds<=600:
        raise ValueError('seconds must be finite and in (0,600]')
    identity=checkpoint_identity(read_json(Path(stage)/'metadata.json'))
    if identity['task_id'] not in ('humanoid_v5','g1_velocity'):
        raise ValueError('This task has no locomotion metrics')
    if command is not None and identity['robot_id']=='humanoid':
        raise ValueError('Legacy Humanoid policy has no velocity command')
    policy,norm=load_stage(stage);env=None
    records=[]
    try:
        limit=math.ceil(seconds/identity['control_dt'])
        env=make_task(identity['task_id'],identity['robot_id'],{**stage_task_config(stage),'max_episode_steps':limit})
        for episode in range(episodes):
            seed=seed_start+episode
            obs,_=env.reset(seed=seed,options={'command':command} if command is not None else None)
            recorder=MotionRecorder(env,identity['robot_id'])
            recorder.capture(np.zeros(env.action_space.shape))
            total=0.
            for step in range(limit):
                if scenario is not None:
                    env.unwrapped.set_command(scenario.command_at(step*identity['control_dt']))
                    obs=env.unwrapped._observation()
                action=policy.predict(norm.normalize_obs(obs),deterministic=True)[0]
                obs,reward,terminated,truncated,info=env.step(action)
                recorder.capture(action);total+=float(reward)
                if terminated or truncated:break
            trajectory=recorder.finish(terminated)
            record=compute_metrics(trajectory)
            record.update(seed=seed,return_raw=total,termination='fall' if terminated else 'time_limit')
            records.append(record)
            if trajectory_dir is not None:
                folder=Path(trajectory_dir);folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{Path(stage).name}-seed-{seed}.npz',**trajectory)
    finally:
        if env is not None:env.close()
        norm.close()
    numeric={key:float(np.mean([r[key] for r in records if r[key] is not None]))
             for key in records[0] if key not in ('seed','schema_version','termination')
             and any(isinstance(r[key],(int,float,bool)) for r in records)}
    return dict(schema_version=1,robot_id=identity['robot_id'],task_id=identity['task_id'],
                checkpoint=str(Path(stage).resolve()),protocol='motion-v1',seconds=seconds,
                seed_start=seed_start,command=command,scenario_id=scenario_id,episodes=records,means=numeric,
                statistics=summarize(records),
                limits=['No calibrated human-likeness score','Mechanical work is not battery energy',
                        'Step length is planar touchdown spacing','Contact threshold 5 N; support debounce 40 ms'])


def evaluate_suite(stage,split='screen'):
    from .scenarios import get_suite
    suite=get_suite(split)
    identity=checkpoint_identity(read_json(Path(stage)/'metadata.json'))
    if identity['task_id']!='g1_velocity':raise ValueError('Command suite requires a G1 command policy')
    results={name:evaluate_motion(stage,episodes=len(suite.seeds),seconds=suite.seconds,
                                  seed_start=suite.seeds[0],scenario_id=name) for name in suite.scenarios}
    return dict(protocol=suite.version,split=split,seeds=list(suite.seeds),seconds=suite.seconds,
                policy_contract=identity,
                checkpoint=str(Path(stage).resolve()),scenarios=results)


def run_suite_diagnostics(run_dir):
    from ..core import list_stages, write_json
    import torch
    torch.set_num_threads(2)
    run = Path(run_dir)
    stages = list_stages(run)
    if not stages:
        raise ValueError('No complete checkpoints')
    stage = stages[-1]
    result = evaluate_suite(stage, 'screen')
    result.update(created_at=time.time(), checkpoint=str(stage.resolve()),
                  selection='latest complete checkpoint at job start')
    write_json(run/'command_suite.json', result)
    return result
