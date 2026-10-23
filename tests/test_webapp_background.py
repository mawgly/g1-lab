import subprocess

from humanoid_sim import webapp


def test_background_server_has_no_window_and_logs_to_the_lab(tmp_path, monkeypatch):
    started, answers = [], iter([False, True])
    monkeypatch.setattr(webapp.subprocess, 'Popen', lambda command, **options: started.append((command, options)))
    monkeypatch.setattr(webapp, 'existing_dashboard', lambda url, run_dir: next(answers))
    monkeypatch.setattr(webapp.time, 'sleep', lambda seconds: None)
    run = tmp_path / 'humanoid'
    assert webapp.start_hidden(8799, run) == 'http://127.0.0.1:8799'
    (command, options), = started
    assert command[1:] == ['-u', '-m', 'humanoid_sim.webapp', '--port', '8799', '--run-dir', str(run)]
    assert options['creationflags'] & getattr(subprocess, 'CREATE_NO_WINDOW', 0)  # no console window on Windows
    assert options['stdin'] is subprocess.DEVNULL and (tmp_path / 'humanoid_lab' / 'server.log').exists()
