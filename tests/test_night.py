from types import SimpleNamespace

from humanoid_sim.train import StageCallback


def test_deadline_stops_training_and_reports_reason(tmp_path):
    callback = StageCallback(tmp_path, 100, 1000, duration_seconds=1)
    callback.model = SimpleNamespace(num_timesteps=8)
    callback.start -= 2
    assert callback._on_step() is False
    assert callback.timed_out


def test_sleep_guard_releases_on_failure():
    from humanoid_sim.night import SleepGuard
    calls = []
    try:
        with SleepGuard(set_state=lambda flags: calls.append(flags) or 1):
            raise RuntimeError('test')
    except RuntimeError:
        pass
    assert calls == [0x80000001, 0x80000000]
