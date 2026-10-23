import json
from pathlib import Path
from humanoid_sim.reports import build_report, write_report


def make_stage(root, name, steps, reward=None):
    path = root / name
    path.mkdir()
    (path / 'metadata.json').write_text(json.dumps({'steps':steps,'created_at':1000+steps}))
    if reward is not None:
        (path / 'evaluation.json').write_text(json.dumps({'mean_return':reward,'mean_seconds':2,'mean_forward_m':1,'episodes':[]}))
    return path


def test_missing_evaluation_is_unknown_and_best_ignores_missing(tmp_path):
    make_stage(tmp_path,'stage_000',0,-2)
    make_stage(tmp_path,'stage_010',10)
    report = build_report(tmp_path)
    assert report['summary']['best']['mean_return'] == -2
    assert report['summary']['last']['mean_return'] is None
    assert report['audit']['historical_mujoco_version'] is None
    assert report['training_statistics']['train/approx_kl']['median'] is None


def test_report_preserves_source_bytes_and_evidence_cutoff(tmp_path):
    path = make_stage(tmp_path,'stage_000',0,1)
    before = {p:p.read_bytes() for p in path.iterdir()}
    report = write_report(tmp_path)
    assert {p:p.read_bytes() for p in path.iterdir()} == before
    assert report['asOf'] == '1970-01-01T00:16:40+00:00'
    assert (tmp_path/'report.html').exists()
    assert report['provenance'][0]['sha256']


def test_csv_missing_and_nan_not_zero(tmp_path):
    folder = tmp_path/'session_1'
    folder.mkdir()
    (folder/'progress.csv').write_text('time/total_timesteps,train/approx_kl\n10,\n20,nan\n30,0.4\n')
    report = build_report(tmp_path)
    assert report['training_statistics']['train/approx_kl']['count'] == 1
    assert report['training_statistics']['train/approx_kl']['median'] == 0.4
    assert report['training_logs'][0]['train/approx_kl'] is None


def test_empty_new_run_preserves_experiment_name(tmp_path):
    (tmp_path/'experiment.json').write_text(json.dumps({'name':'GPU experiment'}))
    report = write_report(tmp_path)
    assert report['title'] == 'GPU experiment'
    assert report['summary']['last_steps'] is None
    assert report['asOf'] is None
    assert 'GPU experiment' in (tmp_path/'report.html').read_text(encoding='utf-8')
