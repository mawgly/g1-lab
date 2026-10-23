"""Persistent queue runner; independent of browser and HTTP-server lifetime."""
import argparse
import subprocess
import sys
import time

from .core import ROOT, GPU_ENVIRONMENT, RunLock, read_json, write_json
from .lab import Laboratory, queue_lock


def training_interpreter(backend):
    if backend in ('gpu', 'full_gpu'):
        return ROOT / GPU_ENVIRONMENT / 'Scripts' / 'python.exe'
    return sys.executable


def run_queue(original):
    lab = Laboratory(original)
    with RunLock(lab.root / 'worker'), RunLock(lab.root / 'compute'):
        if lab.active_trainers():
            raise RuntimeError('A previous trainer is still active; stop it before restarting the queue')
        while not (lab.root / 'stop.request').exists():
            with queue_lock(lab.root):
                queue = lab.queue()
                # A terminated previous worker never silently resumes an old job.
                for item in queue['items']:
                    if item['state'] == 'running':
                        item.update(state='interrupted', error='Предыдущий процесс очереди был прерван')
                item = next((i for i in queue['items'] if i['state'] == 'pending'), None)
                if item is None:
                    queue['state'] = 'completed'
                    write_json(lab.root / 'queue.json', queue)
                    return
                identity = item['id']
                item.update(state='running', started_at=time.time())
                queue['state'] = 'running'
                write_json(lab.root / 'queue.json', queue)
            run = lab.resolve(identity)
            config = read_json(run / 'run_config.json')
            python = training_interpreter(config['backend'])
            result_state, error = 'failed', None
            try:
                with (run / 'training.log').open('ab') as log:
                    process = subprocess.Popen([str(python), '-u', '-m', 'humanoid_sim.cli', 'train',
                        '--run-dir', str(run), '--config', str(run / 'run_config.json'),
                        '--parent-stop', str(lab.root / 'stop.request')], cwd=ROOT,
                        stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    while process.poll() is None:
                        if (lab.root / 'stop.request').exists():
                            (run / 'stop.request').touch()
                        time.sleep(.25)
                status = read_json(run / 'status.json', {})
                result_state = status.get('state', 'failed') if process.returncode == 0 else 'failed'
                if result_state not in ('completed', 'stopped', 'failed'):
                    result_state = 'failed'
                if result_state == 'failed':
                    error = status.get('error') or 'Процесс завершился с ошибкой; см. журнал обучения'
                from .reports import write_report
                write_report(run)
            except Exception as exc:
                error = str(exc)
            with queue_lock(lab.root):
                queue = lab.queue()
                item = next(i for i in queue['items'] if i['id'] == identity and i['state'] == 'running')
                item.update(state=result_state, error=error, finished_at=time.time())
                write_json(lab.root / 'queue.json', queue)
        with queue_lock(lab.root):
            queue = lab.queue()
            queue['state'] = 'paused'
            write_json(lab.root / 'queue.json', queue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    run_queue(parser.parse_args().original)
