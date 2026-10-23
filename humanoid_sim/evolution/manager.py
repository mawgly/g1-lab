"""Persistent campaign queue bridge. Training remains owned by lab_worker."""
from copy import deepcopy
import hashlib
import math
import re
import time
import uuid
from ..core import read_json, write_json, RunLock, list_stages
from ..lab import validate_config, queue_lock


class CampaignManager:
    def __init__(self, lab):
        self.lab = lab
        self.root = lab.root/'campaigns'
        self.root.mkdir(parents=True,exist_ok=True)

    def _path(self, identity):
        if not isinstance(identity,str) or not re.fullmatch(r'campaign-[0-9a-f]{16}',identity):
            raise ValueError('Unknown campaign')
        return self.root/(identity+'.json')

    def create(self, base, branches, *, budget_steps, selection_rules=None,
               max_rounds=3, minimum_improvement=.02):
        base=validate_config(base)
        if type(max_rounds) is not int or not 1<=max_rounds<=20:
            raise ValueError('max_rounds must be 1..20')
        if type(minimum_improvement) not in (int,float) or not math.isfinite(minimum_improvement) or minimum_improvement<=0:
            raise ValueError('minimum_improvement must be finite and positive')
        if selection_rules is not None:
            from .selection import validate_rules
            selection_rules=validate_rules(selection_rules)
            if base['task_id']!='g1_velocity':
                raise ValueError('No campaign evaluation adapter for this task')
        if base['source_id'] and base['source_stage'] is None:
            raise ValueError('Campaign parent checkpoint must be pinned')
        if type(budget_steps) is not int or budget_steps<1 or not isinstance(branches,list) or not 2<=len(branches)<=4:
            raise ValueError('Campaign needs a positive budget and 2..4 branches')
        if not all(isinstance(branch,dict) for branch in branches):
            raise ValueError('Campaign branches must be objects')
        if branches[0].get('changes')!={}:
            raise ValueError('First branch must be the unchanged control')
        planned=[]
        for branch in branches:
            if set(branch)!={'name','changes'} or not isinstance(branch['changes'],dict):
                raise ValueError('Expected branch name and changes')
            changes=branch['changes']
            if set(changes)-{'learning_rate','task_config'} or len(changes)>1:
                raise ValueError('Change one allowed factor per branch')
            if 'task_config' in changes:
                supplied=changes['task_config']
                if (not isinstance(supplied,dict) or set(supplied)!={'reward_weights'}
                        or not isinstance(supplied['reward_weights'],dict) or len(supplied['reward_weights'])!=1):
                    raise ValueError('Change exactly one reward weight')
                merged=deepcopy(base['task_config'])
                merged.setdefault('reward_weights',{}).update(supplied['reward_weights'])
                changes={**changes,'task_config':merged}
            config=validate_config({**base,**changes,'name':branch['name']})
            rollout=config['envs']*config['n_steps']
            cost=((config['steps']+rollout-1)//rollout)*rollout
            planned.append(dict(name=config['name'],config=config,changes=deepcopy(branch['changes']),
                                reserved_steps=cost,state='planned',experiment_id=None))
        reserve=sum(b['reserved_steps'] for b in planned)
        if reserve>budget_steps:
            raise ValueError('Campaign budget cannot cover complete rollouts')
        identity='campaign-'+uuid.uuid4().hex[:16]
        campaign=dict(schema_version=1,id=identity,state='planned',created_at=time.time(),
                      budget_steps=budget_steps,reserved_steps=reserve,branches=planned,
                      selection_rules=selection_rules, round=0, history=[], plateau_rounds=0,
                      max_rounds=max_rounds, minimum_improvement=minimum_improvement)
        write_json(self._path(identity),campaign)
        return campaign

    def advance(self, identity):
        path=self._path(identity)
        with RunLock(self.root/identity):
            campaign=read_json(path)
            if campaign is None:raise ValueError('Unknown campaign')
            if campaign['state'] in ('evaluated','completed'):return campaign
            if campaign['state']=='stopped':
                self._cancel_pending(campaign)
                return campaign
            for index,branch in enumerate(campaign['branches']):
                if branch['experiment_id'] is None:
                    entry=self.lab.create(branch['config'],request_key=f'{identity}:{campaign.get("round",0)}:{index}')
                    branch['experiment_id']=entry['id']
                    write_json(path,campaign)
            with queue_lock(self.lab.root):
                queue=self.lab.queue()
                by_id={i['id']:i for i in queue['items']}
                for branch in campaign['branches']:
                    exp=branch['experiment_id']
                    if exp not in by_id:
                        if (self.lab.resolve(exp)/'status.json').exists():
                            raise ValueError('Campaign experiment ran outside its queue')
                        item=dict(id=exp,state='pending')
                        queue['items'].append(item);by_id[exp]=item
                    branch['state']=by_id[exp]['state']
                    branch['error']=by_id[exp].get('error')
                write_json(self.lab.root/'queue.json',queue)
            states={b['state'] for b in campaign['branches']}
            campaign['state']=('awaiting_evaluation' if states=={'completed'} else
                               'needs_attention' if states & {'failed','interrupted','stopped'} else 'queued')
            write_json(path,campaign)
            return campaign

    def evaluate(self, identity):
        from ..evaluation.runner import evaluate_suite
        from .selection import select_candidates
        path=self._path(identity)
        campaign=self.advance(identity)
        if campaign['state'] in ('stopped','evaluated','completed'):return campaign
        if campaign['state']!='awaiting_evaluation' or campaign.get('selection_rules') is None:
            raise ValueError('Campaign needs completed branches and predeclared selection rules')
        with RunLock(self.lab.root/'compute'):
            if self.lab.active_trainers():
                raise ValueError('A trainer still owns a run; wait for it to finish')
            results={}
            for index in range(len(campaign['branches'])):
                with RunLock(self.root/identity):
                    campaign=read_json(path)
                    if campaign['state']=='stopped':return campaign
                    branch=campaign['branches'][index]
                    run=self.lab.resolve(branch['experiment_id'])
                    if 'evaluation_stage' not in branch:
                        stages=list_stages(run)
                        if not stages:raise ValueError('Completed branch has no checkpoint')
                        branch['evaluation_stage']=stages[-1].name
                    stage=run/branch['evaluation_stage']
                    digest=hashlib.sha256()
                    for filename in ('policy.zip','normalize.pkl','metadata.json'):
                        digest.update((stage/filename).read_bytes())
                    fingerprint=digest.hexdigest()
                    if branch.get('evaluation_fingerprint',fingerprint)!=fingerprint:
                        raise ValueError('Pinned evaluation checkpoint changed')
                    branch['evaluation_fingerprint']=fingerprint
                    campaign['state']='evaluating'
                    write_json(path,campaign)
                cache_path=self.root/identity/(branch['experiment_id']+'-screen.json')
                cached=read_json(cache_path)
                if cached is not None and cached.get('fingerprint')!=fingerprint:
                    raise ValueError('Evaluation cache does not match checkpoint')
                if cached is None:
                    result=evaluate_suite(stage,'screen')
                    cached=dict(fingerprint=fingerprint,result=result)
                    write_json(cache_path,cached)
                results[branch['experiment_id']]=cached['result']
            with RunLock(self.root/identity):
                campaign=read_json(path)
                if campaign['state']=='stopped':return campaign
                campaign['selection']=select_candidates(results,campaign['selection_rules'])
                campaign['state']='evaluated'
                campaign['evaluated_at']=time.time()
                write_json(path,campaign)
                return campaign

    def next_round(self, identity):
        """Reserve a continuation round atomically; advance enqueues it afterwards."""
        path=self._path(identity)
        with RunLock(self.root/identity):
            campaign=read_json(path)
            if campaign is None:raise ValueError('Unknown campaign')
            if campaign['state']!='evaluated':return campaign
            selection=campaign['selection']
            control=campaign['branches'][0]
            decisions=selection['decisions']
            reference=campaign.get('best_metrics',decisions[control['experiment_id']]['metrics'])
            metric='velocity_error_rmse_m_s'
            selected=set(selection['selected'])
            eligible=[decisions[i]['metrics'] for i in selected
                      if decisions[i]['metrics']['fall_rate']<=reference['fall_rate']]
            best=min(eligible,key=lambda m:m[metric]) if eligible else reference
            improved=reference[metric]-best[metric]>=campaign['minimum_improvement']
            campaign['plateau_rounds']=0 if improved else campaign['plateau_rounds']+1
            campaign['best_metrics']=deepcopy(best if improved else reference)
            campaign['history'].append(dict(round=campaign['round'],branches=deepcopy(campaign['branches']),
                selection=deepcopy(selection),improved=improved,recorded_at=time.time()))
            reason=None
            if not selected:reason='no_eligible_candidates'
            elif campaign['plateau_rounds']>=2:reason='plateau'
            elif campaign['round']+1>=campaign['max_rounds']:reason='round_limit'
            continuing=[b for b in campaign['branches'] if b is control or b['experiment_id'] in selected]
            cost=sum(b['reserved_steps'] for b in continuing)
            if reason is None and campaign['reserved_steps']+cost>campaign['budget_steps']:reason='budget'
            if reason:
                campaign.update(state='completed',stop_reason=reason)
            else:
                branches=[]
                for branch in continuing:
                    config=deepcopy(branch['config'])
                    config.update(source_id=branch['experiment_id'],source_stage=int(branch['evaluation_stage'][6:]))
                    branches.append(dict(name=branch['name'],config=config,changes=deepcopy(branch['changes']),
                                         reserved_steps=branch['reserved_steps'],state='planned',experiment_id=None))
                campaign.update(round=campaign['round']+1,branches=branches,state='planned',
                                reserved_steps=campaign['reserved_steps']+cost)
                campaign.pop('selection',None)
            write_json(path,campaign)
            return campaign

    def _cancel_pending(self, campaign):
        ids={b['experiment_id'] for b in campaign['branches'] if b['experiment_id']}
        with queue_lock(self.lab.root):
            queue=self.lab.queue()
            for item in queue['items']:
                if item['id'] not in ids:continue
                if item['state']=='pending':item['state']='stopped'
                elif item['state']=='running':(self.lab.resolve(item['id'])/'stop.request').touch()
            write_json(self.lab.root/'queue.json',queue)

    def stop(self, identity):
        path=self._path(identity)
        with RunLock(self.root/identity):
            campaign=read_json(path)
            if campaign is None:raise ValueError('Unknown campaign')
            campaign['state']='stopped'
            write_json(path,campaign)
            self._cancel_pending(campaign)
            return campaign
