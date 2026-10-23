from humanoid_sim.core import write_json
from humanoid_sim.reports import build_report, render_report


def test_motion_report_includes_measured_values_and_limits(tmp_path):
    write_json(tmp_path/'motion.json',dict(created_at=1,protocol='motion-v1',checkpoints={
        'best':dict(checkpoint='stage_1',seed_start=82000,seconds=15,episodes=[{'fell':False}],
                    means={'contact_slip_speed_m_s':.2,'tilt_rms_rad':.3,'duration_s':15,'fell':0},
                    limits=['No calibrated human-likeness score'])}))
    report=build_report(tmp_path)
    assert report['motion']['protocol']=='motion-v1'
    assert any(s['path']=='motion.json' for s in report['provenance'])
    html=render_report(report)
    assert 'Контакты и движение' in html
    assert '0.20' in html
    assert '82000' in html
    assert 'No calibrated human-likeness score' in html


def test_motion_job_deduplicates_best_and_last_without_modifying_checkpoint(tmp_path,monkeypatch):
    import humanoid_sim.evaluation.runner as runner
    stage=tmp_path/'stage_000000000001';stage.mkdir()
    for name in ('policy.zip','normalize.pkl'):(stage/name).touch()
    write_json(stage/'metadata.json',{'environment':'Humanoid-v5','steps':1})
    write_json(stage/'evaluation.json',{'mean_return':10})
    calls=[]
    monkeypatch.setattr(runner,'evaluate_motion',lambda p,**kw:calls.append(p) or {'checkpoint':str(p)})
    result=runner.run_motion_diagnostics(tmp_path)
    assert len(calls)==1
    assert set(result['checkpoints'])=={'baseline','best','last'}
    assert (tmp_path/'motion.json').is_file()
    assert not (stage/'motion.json').exists()
