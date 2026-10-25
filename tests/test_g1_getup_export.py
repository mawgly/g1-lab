"""Export of a g1_getup stage for g1_ctrl's GetUp state (iteration 33, part B): the walking connector 490 -> 29, no
velocity command in deploy.yaml, the task named in the ONNX properties and the deploy.yaml comment, onnxruntime against
policy.predict on observations lying."""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_sim.core import save_stage, write_json
from humanoid_sim.export_g1 import TERMS, ZERO_RANGES, ExportedPolicy, check, export, onnx_properties, write_yaml
from humanoid_sim.lab import validate_config
from humanoid_sim.policies import CriticExtrasPolicy
from humanoid_sim.tasks import g1_getup_bank as bank_module
from humanoid_sim.tasks.g1_getup import G1GetupEnv
from test_g1_getup import small_bank  # noqa: F401 (fixture)

onnxruntime = pytest.importorskip('onnxruntime')
yaml = pytest.importorskip('yaml')


@pytest.fixture
def getup_stage(tmp_path, small_bank, monkeypatch):  # noqa: F811
    """An untrained g1_getup stage (32×32 actor, its action head ×2000 so the ±10 clip and the wrist lock act) with
    normalization statistics far from the defaults, the command and phase columns at their fixed 0 and 1."""
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    config = validate_config(dict(robot_id='g1_29dof', task_id='g1_getup', n_steps=8, batch_size=8))
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: G1GetupEnv(bank=small_bank)]))
    try:
        model = PPO(CriticExtrasPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=1,
                    policy_kwargs={'net_arch': [32, 32], 'actor_inputs': 490})
        rng = np.random.default_rng(1)
        env.obs_rms.mean[:] = rng.normal(0, 1, 496)
        env.obs_rms.var[:] = rng.uniform(.01, 4, 496)
        columns, mean, variance = env.venv.envs[0].fixed_normalization
        env.obs_rms.mean[columns], env.obs_rms.var[columns] = mean, variance
        with torch.no_grad():
            model.policy.action_net.weight.mul_(2000)
        return save_stage(run, model, env)
    finally:
        env.close()


def test_a_getup_stage_goes_out_through_the_walking_connector_without_commands(tmp_path, getup_stage):
    result = export(getup_stage, tmp_path / 'out')
    assert result['actor_inputs'] == 490 and result['actions'] == 29 and result['history'] == 5
    assert result['warnings'] == [] and result['commands'] == ZERO_RANGES
    session = onnxruntime.InferenceSession(str(result['onnx']), providers=['CPUExecutionProvider'])
    (given,), (taken,) = session.get_inputs(), session.get_outputs()
    assert (given.name, given.shape, taken.name, taken.shape) == ('obs', [1, 490], 'actions', [1, 29])
    properties = onnx_properties(result['onnx'])
    assert properties['task_id'] == 'g1_getup' and properties['getup'] == 'true'
    assert properties['observation_schema'] == 'g1_getup:obs-unitree-deploy-v0-h5-phasestop-critic6'
    assert properties['action_schema'] == 'g1_getup:action-v2-limit10-wristlock-knee0.061'
    assert properties['foot_collision'] == properties['hand_collision'] == 'true'
    text = result['deploy'].read_text(encoding='utf-8')
    assert '# g1_getup: a getting-up policy' in text and 'g1_ctrl-getup.patch' in text
    deploy = yaml.safe_load(text)
    assert deploy['commands']['base_velocity']['ranges'] == {**ZERO_RANGES, 'heading': None}
    assert tuple(deploy['observations']) == TERMS
    assert all(term['history_length'] == 5 for term in deploy['observations'].values())
    assert deploy['observations']['gait_phase']['params'] == {'period': .6}
    clip = deploy['actions']['JointPositionAction']['clip']
    assert clip[3][0] == clip[9][0] == pytest.approx(.061)  # the knee floor, as the walking candidates
    verdict = check(getup_stage, result['onnx'], samples=20, stride=2)
    assert verdict['ok'] and verdict['samples'] == 20 and verdict['zero_commands'] and verdict['starts'] == 5
    assert verdict['max_abs_diff'] < 1e-4 and verdict['max_target_diff'] < 1e-4
    # The exported policy acts in the stage's own environment, lying: the wrists locked, the box clip exercised.
    exported = ExportedPolicy(result['onnx'].parent)
    env = G1GetupEnv()
    try:
        obs, _ = env.reset(seed=0, options=dict(start=0, assist_force=0.))
        action = exported.predict(obs)[0]
        assert action.shape == (29,) and not action[[20, 21, 27, 28]].any() and np.abs(action).max() == 10
    finally:
        env.close()
    # A deploy.yaml that would let the stick through fails the check.
    deploy['commands']['base_velocity']['ranges']['lin_vel_x'] = [-.5, 1.]
    write_yaml(deploy, result['deploy'], 'edited')
    assert not check(getup_stage, result['onnx'], samples=5, stride=1)['ok']


def test_walking_options_are_refused_for_a_getup_stage(tmp_path, getup_stage):
    for options in (dict(commands=(-.5, 1., .5, 1.)), dict(symmetric=True), dict(trim=np.zeros(29))):
        with pytest.raises(ValueError):
            export(getup_stage, tmp_path / 'out', **options)
    assert not (tmp_path / 'out').exists()  # refused before anything is written
