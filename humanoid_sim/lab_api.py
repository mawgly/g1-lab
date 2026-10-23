"""HTTP-independent lab jobs and response data."""
import subprocess
import sys
import threading
from pathlib import Path

from .core import ROOT, GPU_ENVIRONMENT, read_json, RunLock, write_json


class LabJobs:
    def __init__(self, lab):
        self.lab = lab
        self.process = None
        self.kind = None
        self.identity = None
        self.lock = threading.Lock()
        self._hardware = {}

    def active(self):
        if self.process is not None and self.process.poll() is None:
            return True
        try:
            with RunLock(self.lab.root / 'job'):
                return False
        except RuntimeError:
            return True

    def status(self):
        saved = read_json(self.lab.root / 'job.json', {})
        return {**saved, 'active': self.active(), 'kind': self.kind or saved.get('kind'),
                'experiment': self.identity or saved.get('experiment'),
                'exit_code': self.process.poll() if self.process else saved.get('exit_code')}

    def launch(self, kind, identity='baseline'):
        with self.lock:
            if self.active() or self.lab.worker_active() or self.lab.active_trainers() or self.lab.compute_active():
                raise ValueError('Дождитесь завершения или остановите текущее задание')
            run = self.lab.resolve(identity)
            if kind in ('benchmark', 'full_gpu_benchmark'):
                python = ROOT / GPU_ENVIRONMENT / 'Scripts' / 'python.exe'
                if not python.exists():
                    raise ValueError('GPU-окружение ещё не установлено: запустите setup_fullgpu.cmd')
                command = [str(python), '-u', '-m', 'humanoid_sim.lab_tasks', kind, '--run-dir', str(run), '--lab-root', str(self.lab.root)]
            elif kind in ('diagnostics', 'report', 'motion', 'command_suite'):
                from .tasks.catalog import resolve_task
                config = read_json(run / 'run_config.json', {})
                task = resolve_task(config.get('robot_id') or 'humanoid', config.get('task_id') or 'humanoid_v5')
                if kind not in task.jobs:
                    raise ValueError(f'Эта проверка недоступна для задачи «{task.label or task.task_id}»')
                command = [sys.executable, '-u', '-m', 'humanoid_sim.lab_tasks', kind, '--run-dir', str(run), '--lab-root', str(self.lab.root)]
            else:
                raise ValueError('Неизвестное задание')
            with (self.lab.root / 'jobs.log').open('ab') as log:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.kind, self.identity = kind, identity
            write_json(self.lab.root / 'job.json', {'kind': kind, 'experiment': identity, 'pid': self.process.pid})
            return {'accepted': True}

    def hardware(self):
        import time
        if time.monotonic() - self._hardware.get('_time', -10) < 3:
            return self._hardware
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            values = result.stdout.strip().splitlines()[0].split(',')
            self._hardware = dict(name=values[0].strip(), utilization=float(values[1]), memory_used=float(values[2]),
                                 memory_total=float(values[3]), temperature=float(values[4]), _time=time.monotonic())
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            self._hardware = {'error': 'Метрики NVIDIA недоступны', '_time': time.monotonic()}
        return self._hardware

    def snapshot(self):
        benchmark = read_json(self.lab.root / 'gpu_benchmark.json')
        return {'experiments': self.lab.entries(), 'queue': self.lab.queue(), 'notes': self.lab.notes(),
                'worker_active': self.lab.worker_active(),
                'active_trainers': self.lab.active_trainers(),
                'job': self.status(), 'hardware': self.hardware(), 'benchmark': benchmark,
                'full_gpu_benchmark': read_json(self.lab.root / 'full_gpu_benchmark.json')}
