import pytest
from humanoid_sim.core import write_json, read_json
from humanoid_sim.evaluation import runner


def test_suite_job_pins_latest_checkpoint_and_persists_protocol(tmp_path, monkeypatch):
    stage = tmp_path/'stage_000000000016'
    stage.mkdir()
    for name in ('policy.zip', 'normalize.pkl'):
        (stage/name).touch()
    write_json(stage/'metadata.json', {'steps':16})
    calls = []
    monkeypatch.setattr(runner, 'evaluate_suite', lambda path, split: calls.append((path, split)) or
                        dict(protocol='g1-command-v1', split=split, scenarios={}))
    runner.run_suite_diagnostics(tmp_path)
    assert calls == [(stage, 'screen')]
    result = read_json(tmp_path/'command_suite.json')
    assert result['selection'] == 'latest complete checkpoint at job start'
    assert result['checkpoint'] == str(stage.resolve())
    assert result['created_at'] > 0


def test_screen_includes_restart_period():
    from humanoid_sim.evaluation.scenarios import get_suite, get_scenario
    suite = get_suite('screen')
    assert suite.seconds > get_scenario('stop_start').schedule[-1][0]


def test_suite_report_shows_sample_size_and_uncertainty(tmp_path):
    from humanoid_sim.reports import build_report, render_report
    write_json(tmp_path/'command_suite.json', dict(protocol='g1-command-v2', checkpoint='stage_16',
        scenarios={'stand':dict(episodes=[{}]*5, means={'fell':0., 'duration_s':20.},
            statistics={'fell':{'wilson95':[0., .43]}})}))
    report = build_report(tmp_path)
    assert report['command_suite']['protocol']=='g1-command-v2'
    html = render_report(report)
    assert 'Проверка команд' in html and 'Стояние' in html and '43' in html
    assert 'Это измеренный признак больших изменений' not in html
