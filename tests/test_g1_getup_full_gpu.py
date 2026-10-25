"""g1_getup on the full-GPU path (iteration 33, part B), on the CPU: make_learner's policy and normalization for the
task, and the worlds' assist forces kept by every stage and given back to a resumed run. The CUDA task itself is
stood in for by a fake with the same interface (its physics: tests/test_g1_gpu.py)."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from humanoid_sim.core import list_stages, load_stage, read_json, write_json
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks import g1_tensor_math as task


class FakeTaskEnv:
    """The CUDA task's interface on the CPU for n worlds: spaces from the task's CPU environment, zero observations,
    an episode end every 3 steps, and (getup) assist forces per world that training_state saves and
    restore_training_state takes back, recorded with the calls' order."""

    def __init__(self, n_envs, reference, getup):
        self.device = torch.device('cpu')
        self.n_envs = n_envs
        self.reference = reference
        self.observation_space, self.action_space = reference.observation_space, reference.action_space
        self.fixed_normalization = reference.fixed_normalization
        self.lengths = torch.zeros(n_envs, dtype=torch.long)
        self.last_episode_lengths = torch.zeros(n_envs)
        self.last_episode_returns = torch.zeros(n_envs)
        self.calls = []
        if getup:
            self.assist = torch.linspace(200., 140., n_envs)
            self.training_state = lambda: {'assist_force': self.assist.tolist()}
            self.restore_training_state = self._restore

    def _restore(self, state):
        self.calls.append(('restore', list(state['assist_force'])))
        self.assist = torch.as_tensor(task.assist_resample(np.asarray(state['assist_force']), self.n_envs),
                                      dtype=torch.float32)

    def _obs(self):
        obs = torch.zeros((self.n_envs, self.observation_space.shape[0]))
        obs[:, -1] = getattr(self, 'assist', torch.zeros(self.n_envs))
        return obs

    def reset(self):
        self.calls.append(('reset',))
        return self._obs()

    def step(self, actions):
        self.lengths += 1
        truncated = self.lengths >= 3
        self.last_episode_lengths.copy_(self.lengths.float())
        self.lengths[truncated] = 0
        if hasattr(self, 'assist'):  # one world gets up each episode: its assist steps down
            self.assist[0] -= 20. * float(truncated[0])
        obs = self._obs()
        return obs, torch.ones(self.n_envs), torch.zeros(self.n_envs, dtype=torch.bool), truncated, obs

    def check_health(self):
        pass

    def close(self):
        self.reference.close()


class CPUPPO(PPO):
    """PPO on the CPU wherever make_learner asks for CUDA."""

    def __init__(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return PPO.load(*args, **kwargs)


@pytest.fixture
def cpu_full_gpu(monkeypatch):
    """full_gpu_train with CPU stand-ins for CUDA: the backend adapter gives FakeTaskEnv over the task's CPU
    environment; returns the environments made."""
    from humanoid_sim import full_gpu_train
    from humanoid_sim.backends import registry
    from humanoid_sim.tasks.catalog import make_task
    made = []

    class Fake:
        def make(self, robot_id, task_id, backend, n_envs, seed, **task_config):
            assert backend == 'full_gpu'
            env = FakeTaskEnv(n_envs, make_task(task_id, robot_id, task_config), task_id == 'g1_getup')
            made.append(env)
            return env
    monkeypatch.setattr(registry, 'backends', lambda: Fake())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU test substitute')
    monkeypatch.setattr(full_gpu_train, 'PPO', CPUPPO)
    yield made
    for env in made:
        env.close()


def getup_config(**values):
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', backend='full_gpu', envs=4, n_steps=8,
                                  batch_size=32, n_epochs=1, steps=1024, target_kl=None, **values))
    return {**config, 'steps': 32}  # one rollout


def test_make_learner_gives_getup_the_actor_of_the_robot_inputs_and_its_fixed_columns(cpu_full_gpu):
    """g1_getup has no task_config critic_extras: the asymmetric critic follows the task's own observation (490 actor
    inputs of 496), and the command and gait phase columns (task.zero_term_columns, 25) stay at mean 0, variance 1
    through the updates. Walking is chosen as before: critic_extras gives the asymmetric critic, the simulation
    profile the plain one."""
    from humanoid_sim import full_gpu_train
    learner, export = full_gpu_train.make_learner(getup_config())
    try:
        policy = learner.model.policy
        assert isinstance(policy, CriticExtrasPolicy) and policy.actor_inputs == 490
        assert learner.env.observation_space.shape == (496,)
        columns = task.zero_term_columns(29, 5)
        assert len(columns) == 25 and learner.normalizer.fixed[0].tolist() == columns
        learner.collect()
        learner.update()
        moments = learner.normalizer.obs
        assert bool((moments.mean[columns] == 0).all()) and bool((moments.var[columns] == 1).all())
        assert float(moments.mean[-1]) > 100  # the assist column is estimated as any other
    finally:
        export.close()
    for task_config, expected in (({'real_robot': True, 'critic_extras': True, 'action_limit': 10.}, CriticExtrasPolicy),
                                  ({}, ActorCriticPolicy)):
        config = validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', backend='full_gpu', envs=4,
                                      n_steps=8, batch_size=32, n_epochs=1, steps=1024, task_config=task_config))
        learner, export = full_gpu_train.make_learner(config)
        try:
            assert type(learner.model.policy) is expected
            if expected is CriticExtrasPolicy:
                assert learner.model.policy.actor_inputs == 98 and learner.env.observation_space.shape == (102,)
        finally:
            export.close()


def test_stages_keep_the_worlds_assist_forces_and_a_resumed_run_gives_them_back(tmp_path, cpu_full_gpu, monkeypatch):
    """Every stage of a full-GPU g1_getup run keeps the worlds' assist forces (metadata.json training_state, as the
    CPU path); a resumed run restores them into its worlds before their first reset, spread over another world count
    by quantiles (task.assist_resample). A walking run's stages have no training_state."""
    from humanoid_sim import full_gpu_train
    import humanoid_sim.train as cpu_train
    monkeypatch.setattr(cpu_train, 'evaluate_stage', lambda *_: {'mean_return': 0., 'mean_seconds': 0.,
                                                                 'mean_forward_m': None, 'episodes': []})
    run = tmp_path / 'run'
    run.mkdir()
    config = getup_config()
    write_json(run / 'run_config.json', config)
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=run, parent_stop=None), config)
    stages = list_stages(run)
    assert read_json(run / 'status.json')['state'] == 'completed' and len(stages) == 2
    first, last = (read_json(stage / 'metadata.json')['training_state']['assist_force'] for stage in stages)
    assert first == pytest.approx([200., 180., 160., 140.]) and last[0] < first[0] and last[1:] == first[1:]
    policy, norm = load_stage(stages[-1])  # an ordinary stage on the CPU
    try:
        assert isinstance(policy.policy, CriticExtrasPolicy) and policy.policy.actor_inputs == 490
    finally:
        norm.close()
    resumed = {**config, 'envs': 8, 'batch_size': 64}
    write_json(run / 'run_config.json', resumed)
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=run, parent_stop=None), resumed)
    env = cpu_full_gpu[-1]
    assert env.calls[:2] == [('restore', last), ('reset',)]  # before the worlds' first reset
    # [160, 180, 160, 140] over 8 worlds: the sorted forces at evenly spread quantiles; world 0 then got up twice.
    assert task.assist_resample(np.asarray(last), 8).tolist() == [140., 140., 160., 160., 160., 160., 180., 180.]
    newest = read_json(list_stages(run)[-1] / 'metadata.json')
    assert newest['steps'] == 96 and newest['training_state']['assist_force'] == pytest.approx(
        [100., 140., 160., 160., 160., 160., 180., 180.])
    # Walking: no training state anywhere.
    walk = tmp_path / 'walk'
    walk.mkdir()
    config = {**validate_config(dict(robot_id='g1_29dof', task_id='g1_velocity', backend='full_gpu', envs=4, n_steps=8,
                                     batch_size=32, n_epochs=1, steps=1024, target_kl=None)), 'steps': 32}
    write_json(walk / 'run_config.json', config)
    full_gpu_train.train_full_gpu(SimpleNamespace(run_dir=walk, parent_stop=None), config)
    assert all('training_state' not in read_json(stage / 'metadata.json') for stage in list_stages(walk))
