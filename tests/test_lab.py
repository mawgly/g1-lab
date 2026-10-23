import pytest

from humanoid_sim.lab import Laboratory, validate_config


def test_configuration_bounds_and_unknown_fields():
    assert validate_config({'name': 'Проба'})['every'] == 1000000
    assert validate_config({'name': 'Проба', 'every': 10000000})['every'] == 10000000  # G1 full GPU: ~5 min
    for every in (99999, 50000001):
        with pytest.raises(ValueError):
            validate_config({'every': every})
    for settings in ({'envs': 0}, {'learning_rate': float('nan')}, {'steps': -1}, {'backend': 'shell'}, {'command': 'x'}, {'seed': True}):
        with pytest.raises(ValueError):
            validate_config(settings)


def test_create_isolated_and_protect_original(tmp_path):
    original = tmp_path / 'humanoid'
    original.mkdir()
    lab = Laboratory(original)
    a = lab.create({'name': 'Тест'})
    assert lab.resolve(a['id']) != original
    assert lab.resolve('baseline') == original
    with pytest.raises(ValueError):
        lab.resolve('../humanoid')
    lab.enqueue([a['id']])
    with pytest.raises(ValueError):
        lab.enqueue(['baseline'])
    with pytest.raises(ValueError):
        lab.enqueue([a['id']])
    assert lab.queue()['items'][0]['id'] == a['id']


def test_series_preserves_seeds_and_changes(tmp_path):
    lab = Laboratory(tmp_path / 'original')
    rows = lab.create_series({'steps': 10000, 'backend': 'cpu', 'envs': 2})
    assert len(rows) == 9
    assert {r['config']['seed'] for r in rows} == {42, 43, 44}
    assert len({r['id'] for r in rows}) == 9


def test_training_uses_frozen_config_and_stops_from_parent(tmp_path):
    from argparse import Namespace
    from humanoid_sim.core import write_json, read_json, list_stages, load_stage
    from humanoid_sim.train import train
    run = tmp_path / 'run'
    run.mkdir()
    settings = validate_config({'envs': 1, 'steps': 1024, 'n_steps': 8, 'batch_size': 8,
                                'learning_rate': 0.00005, 'target_kl': 0.02, 'n_epochs': 1, 'gamma': .95})
    write_json(run / 'run_config.json', settings)
    stop = tmp_path / 'parent.stop'
    stop.touch()
    train(Namespace(run_dir=run, config=run/'run_config.json', parent_stop=stop))
    assert read_json(run/'status.json')['state'] == 'stopped'
    policy, norm = load_stage(list_stages(run)[-1])
    try:
        assert policy.target_kl == .02
        assert policy.learning_rate == .00005
        assert policy.num_timesteps <= 8
        assert norm.gamma == policy.gamma == .95
    finally:
        norm.close()


def test_orphan_trainer_blocks_restart_and_can_be_stopped(tmp_path):
    from humanoid_sim.core import RunLock, write_json
    lab = Laboratory(tmp_path / 'original')
    experiment = lab.create({'name': 'Survivor'})
    run = lab.resolve(experiment['id'])
    lab.enqueue([experiment['id']])
    write_json(run / 'status.json', {'state': 'training'})
    with RunLock(run):
        with pytest.raises(ValueError, match='работает'):
            lab.start()
        lab.stop()
        assert (run / 'stop.request').exists()


def test_queue_edit_waits_for_other_writer(tmp_path):
    import threading
    import time
    from humanoid_sim.lab import queue_lock
    from humanoid_sim.core import RunLock
    result = []
    def acquire():
        with queue_lock(tmp_path):
            result.append(True)
    with RunLock(tmp_path / 'queue-edit'):
        thread = threading.Thread(target=acquire)
        thread.start()
        time.sleep(.1)
        assert not result
    thread.join(timeout=2)
    assert result == [True]


def test_job_ownership_survives_server_object_recreation(tmp_path):
    from humanoid_sim.core import RunLock, write_json
    from humanoid_sim.lab_api import LabJobs
    lab = Laboratory(tmp_path / 'original')
    write_json(lab.root / 'job.json', {'kind': 'diagnostics', 'experiment': 'baseline'})
    with RunLock(lab.root / 'job'):
        restored = LabJobs(lab)
        assert restored.status()['active']
        assert restored.status()['kind'] == 'diagnostics'
        with pytest.raises(ValueError):
            restored.launch('benchmark')


def test_full_gpu_config_and_interpreter_are_isolated():
    from humanoid_sim.lab_worker import training_interpreter
    config = validate_config({'backend': 'full_gpu', 'envs': 256, 'n_steps': 32})
    assert config['backend'] == 'full_gpu'
    assert '.venv-fullgpu' in str(training_interpreter('full_gpu'))
    assert training_interpreter('gpu') == training_interpreter('full_gpu')
    with pytest.raises(ValueError):
        validate_config({'backend': 'full_gpu', 'envs': 1024, 'n_steps': 4096})


def test_full_gpu_dispatch_uses_validated_config_before_cpu_lock(tmp_path, monkeypatch):
    from argparse import Namespace
    import sys
    from types import ModuleType
    import humanoid_sim.train as trainer
    from humanoid_sim.core import write_json
    fake = ModuleType('humanoid_sim.full_gpu_train')
    calls = []
    fake.train_full_gpu = lambda args, config: calls.append((args, config)) or 'gpu-result'
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(trainer, 'RunLock', lambda *args: pytest.fail('CPU RunLock must not be acquired before GPU dispatch'))
    path = tmp_path / 'config.json'
    write_json(path, {'backend': 'full_gpu', 'envs': 256, 'n_steps': 32})
    args = Namespace(run_dir=tmp_path, config=path)
    assert trainer.train(args) == 'gpu-result'
    assert calls[0][1]['envs'] == args.envs == 256
    assert calls[0][1]['target_kl'] == .03
    write_json(path, {'backend': 'full_gpu', 'envs': -1})
    with pytest.raises(ValueError):
        trainer.train(args)
    assert len(calls) == 1


def test_full_gpu_benchmark_launch_uses_cuda_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import humanoid_sim.lab_api as module
    from humanoid_sim.core import read_json
    lab = Laboratory(tmp_path / 'original')
    python = tmp_path / '.venv-fullgpu' / 'Scripts' / 'python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    launched = []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda command, **kwargs: launched.append(command) or SimpleNamespace(pid=123, poll=lambda: None))
    jobs = module.LabJobs(lab)
    jobs.launch('full_gpu_benchmark')
    assert launched[0][0] == str(python)
    assert 'full_gpu_benchmark' in launched[0]
    assert read_json(lab.root/'job.json')['kind'] == 'full_gpu_benchmark'
    with pytest.raises(ValueError):
        jobs.launch('benchmark')


def test_manual_training_and_lab_share_compute_slot(tmp_path, monkeypatch):
    import sys
    import humanoid_sim.cli as cli
    import humanoid_sim.train as trainer
    from humanoid_sim.core import RunLock
    original = tmp_path / 'humanoid'
    monkeypatch.setattr(cli, 'DEFAULT_RUN', original)
    lab = Laboratory(original)
    seen = []

    def fake_train(args):
        seen.append(lab.compute_active())
        entry = lab.create({'name': 'Очередь'})
        lab.enqueue([entry['id']])
        with pytest.raises(ValueError, match='Вычисления заняты'):
            lab.start()

    monkeypatch.setattr(trainer, 'train', fake_train)
    monkeypatch.setattr(sys, 'argv', ['cli', 'train', '--run-dir', str(tmp_path / 'manual')])
    cli.main()
    assert seen == [True] and not lab.compute_active()
    with RunLock(lab.root / 'compute'):
        with pytest.raises(SystemExit):
            cli.main()
    assert seen == [True]


def test_lab_jobs_follow_task_catalog(tmp_path):
    from humanoid_sim.lab_api import LabJobs
    lab = Laboratory(tmp_path / 'original')
    arm = lab.create({'name': 'Рука', 'robot_id': 'so101', 'task_id': 'arm_reach'})
    with pytest.raises(ValueError, match='недоступна'):
        LabJobs(lab).launch('motion', arm['id'])
    with pytest.raises(ValueError, match='недоступна'):
        LabJobs(lab).launch('command_suite', 'baseline')
