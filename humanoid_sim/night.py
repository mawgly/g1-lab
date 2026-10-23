"""Thread-scoped Windows sleep inhibition; never changes the power plan."""
import ctypes
import sys


class SleepGuard:
    def __init__(self, enabled=True, set_state=None):
        self.enabled = enabled
        self.set_state = set_state

    def __enter__(self):
        if self.enabled:
            if self.set_state is None:
                if sys.platform != 'win32':
                    raise RuntimeError('Night sleep protection requires Windows')
                self.set_state = ctypes.windll.kernel32.SetThreadExecutionState
            if not self.set_state(0x80000001):
                raise RuntimeError('Windows could not inhibit automatic sleep')
        return self

    def __exit__(self, *_):
        if self.enabled:
            self.set_state(0x80000000)
