"""Local experiment registry. The original experiment is always read-only."""
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .core import ROOT, read_json, write_json, list_stages, RunLock

DEFAULTS = dict(name='Новый эксперимент', backend='cpu', envs=8, steps=2000000,
                every=1000000, seed=42, learning_rate=0.0003, n_steps=1024,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.0, clip_range=0.2, target_kl=0.03, hours=None,
                # Full-GPU PPO only: penalty on the action mean outside ±1.1, a cap on the policy std and the
                # left–right mirror loss (G1 real-robot profile, symmetry.py). range_bounds: the bounds loss uses
                # the action where the motor target reaches the joint's range instead of the action box.
                bounds_coef=0.0, max_std=None, mirror_coef=0.0, range_bounds=False,
                preset='custom', source_id=None, source_stage=None,
                robot_id='humanoid', task_id='humanoid_v5', task_config=None)
PRESETS = {
    'baseline': dict(learning_rate=0.0003, n_epochs=10, ent_coef=0.0, target_kl=None),
    'careful': dict(learning_rate=0.00005, n_epochs=5, ent_coef=0.0, target_kl=0.02),
    'explore': dict(learning_rate=0.0001, n_epochs=5, ent_coef=0.002, target_kl=0.03),
}


# Research-tree verdicts written by the supervisor (Claude) or the user; shown as chips in the tree.
VERDICTS = ('active', 'best', 'control', 'dead_end')


@contextmanager
def edit_lock(root, name='queue-edit'):
    deadline = time.monotonic() + 5
    while True:
        lock = RunLock(Path(root) / name)
        try:
            lock.__enter__()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Файл очереди занят; повторите действие')
            time.sleep(.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def queue_lock(root):
    return edit_lock(root, 'queue-edit')


def validate_config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError('Неизвестные настройки')
    config = {**DEFAULTS, **values}
    from .tasks.catalog import registries
    robots, tasks = registries()
    if not isinstance(config['robot_id'], str):
        raise ValueError('Некорректный робот')
    task = tasks.resolve(config['task_id'], robots.get(config['robot_id']))
    config['task_config'] = task.config_validator({} if config['task_config'] is None else config['task_config'])
    if config['backend'] not in task.training_backends:
        raise ValueError('Для этой задачи не поддерживается выбранный backend')
    if not isinstance(config['name'], str) or not 1 <= len(config['name'].strip()) <= 100:
        raise ValueError('Название: от 1 до 100 символов')
    if config['backend'] not in ('cpu', 'gpu', 'full_gpu'):
        raise ValueError('Движок: cpu, gpu или full_gpu')
    bounds = {'envs': (1, 4096), 'steps': (1024, 2000000000), 'every': (100000, 50000000),
              'seed': (0, 2147483647), 'n_steps': (8, 4096), 'batch_size': (8, 65536), 'n_epochs': (1, 30)}
    for key, (low, high) in bounds.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f'{key}: допустимо {low}–{high}')
    if config['backend'] == 'cpu' and config['envs'] > 24:
        raise ValueError('Для CPU максимум 24 среды')
    if config['envs'] * config['n_steps'] > 262144:
        raise ValueError('Слишком большой пакет опыта: уменьшите число сред или шагов на среду')
    if config['batch_size'] > config['envs'] * config['n_steps']:
        raise ValueError('Мини-пакет больше объёма опыта')
    for key, low, high in [('learning_rate', 1e-6, .01), ('gamma', .8, .9999), ('gae_lambda', .8, 1),
                           ('ent_coef', 0, .1), ('clip_range', .01, .4), ('target_kl', .001, .5), ('hours', .01, 24),
                           ('bounds_coef', 0, 1), ('max_std', .01, 10), ('mirror_coef', 0, 10)]:
        value = config[key]
        if value is None and key in ('hours', 'target_kl', 'max_std'):
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key}: допустимо {low}–{high}')
    if type(config['range_bounds']) is not bool:
        raise ValueError('range_bounds: true или false')
    if config['source_stage'] is not None and (type(config['source_stage']) is not int or config['source_stage'] < 0):
        raise ValueError('Некорректный этап-источник')
    if config['source_id'] is not None and not isinstance(config['source_id'], str):
        raise ValueError('Некорректный источник')
    if config['preset'] not in ('custom', *PRESETS):
        raise ValueError('Неизвестный пресет')
    return config


class Laboratory:
    def __init__(self, original):
        self.original = Path(original).resolve()
        self.root = self.original.parent / (self.original.name + '_lab')
        self.root.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.process = None

    def resolve(self, identity):
        if identity == 'baseline':
            return self.original
        if not isinstance(identity, str) or not identity.startswith('exp-') or len(identity) != 20 or any(c not in '0123456789abcdef' for c in identity[4:]):
            raise ValueError('Неизвестный эксперимент')
        path = self.root / identity
        if not (path / 'experiment.json').is_file():
            raise ValueError('Эксперимент не найден')
        return path

    def create(self, values, *, request_key=None):
        config = validate_config(values)
        with self.mutex:
            if request_key is not None:
                if not isinstance(request_key,str) or not 1 <= len(request_key) <= 160:
                    raise ValueError('Invalid creation request key')
                for manifest in self.root.glob('exp-*/experiment.json'):
                    previous = read_json(manifest)
                    if previous.get('request_key') == request_key:
                        if previous['config'] != config:
                            raise ValueError('Creation request key reused with different configuration')
                        return previous
            source = None
            if config['source_id']:
                source_run = self.resolve(config['source_id'])
                stages = list_stages(source_run)
                if not stages:
                    raise ValueError('У источника пока нет сохранений')
                source = stages[-1] if config['source_stage'] is None else source_run / f'stage_{config["source_stage"]:012d}'
                if source not in stages:
                    raise ValueError('Этап-источник не найден')
                from .checkpoint_schema import validate_resume
                from .tasks.identity import task_identity
                validate_resume(read_json(source / 'metadata.json'),
                                task_identity(config['robot_id'], config['task_id'], config.get('task_config') or {}))
                config['source_stage'] = int(source.name[6:])
            identity = 'exp-' + uuid.uuid4().hex[:16]
            path = self.root / identity
            path.mkdir()
            if source:
                shutil.copytree(source, path / source.name)
            entry = {'id': identity, 'name': config['name'].strip(), 'created_at': time.time(),
                     'config': config, 'read_only': False}
            if request_key is not None:
                entry['request_key'] = request_key
            write_json(path / 'run_config.json', config)
            # Publish the discoverable manifest only after the runnable config.
            write_json(path / 'experiment.json', entry)
            return entry

    def create_series(self, values):
        base = dict(values)
        validate_config(base)
        result = []
        for preset, overrides in PRESETS.items():
            for seed in (42, 43, 44):
                result.append(self.create({**base, **overrides, 'preset': preset, 'seed': seed,
                                          'name': f'{preset} · seed {seed}'}))
        return result

    def entries(self):
        # The original run has no run_config.json (legacy Humanoid); a test original may have one.
        rows = [{'id': 'baseline', 'name': 'Эксперимент 1 · исходный PPO', 'read_only': True, 'created_at': 0,
                 'config': {'backend': 'cpu', 'envs': 8, 'seed': 42, **read_json(self.original / 'run_config.json', {})}}]
        rows += [read_json(p) for p in sorted(self.root.glob('exp-*/experiment.json'))]
        for entry in rows:
            run = self.resolve(entry['id'])
            entry['status'] = read_json(run / 'status.json', {'state': 'ready', 'steps': 0})
            stages = list_stages(run)
            entry['stage_count'] = len(stages)
            entry['last_stage'] = int(stages[-1].name[6:]) if stages else None
            entry['result'] = read_json(stages[-1] / 'evaluation.json', {}) if stages else {}
            entry['has_report'] = (run / 'report.json').exists()
            if entry['status'].get('state') in ('training', 'starting', 'saving'):
                try:
                    with RunLock(run):
                        entry['status']['state'] = 'interrupted'
                except RuntimeError:
                    pass
        return sorted(rows, key=lambda r: r['created_at'])

    def queue(self):
        return read_json(self.root / 'queue.json', {'items': [], 'state': 'idle'})

    def notes(self):
        """{experiment id: {verdict, note, updated_at, history: [...]}} — the research tree's decision log."""
        return read_json(self.root / 'notes.json', {})

    def annotate(self, identity, verdict=None, note=None):
        self.resolve(identity)
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError('Неизвестная оценка ветки')
        if note is not None and (not isinstance(note, str) or len(note) > 2000):
            raise ValueError('Заметка: до 2000 символов')
        with self.mutex, edit_lock(self.root, 'notes-edit'):
            data = self.notes()
            entry = data.setdefault(identity, {'history': []})
            if verdict is not None:
                entry['verdict'] = verdict
            if note is not None:
                entry['note'] = note
            entry['updated_at'] = time.time()
            entry['history'].append({'at': entry['updated_at'], 'verdict': verdict, 'note': note})
            write_json(self.root / 'notes.json', data)
            return entry

    def enqueue(self, identities):
        if not isinstance(identities, list) or not identities or len(identities) > 20:
            raise ValueError('Выберите от 1 до 20 экспериментов')
        with self.mutex, queue_lock(self.root):
            data = self.queue()
            existing = {i['id'] for i in data['items'] if i['state'] in ('pending', 'running')}
            for identity in identities:
                run = self.resolve(identity)
                if identity == 'baseline':
                    raise ValueError('Первый эксперимент защищён: создайте продолжение')
                if identity in existing:
                    raise ValueError('Эксперимент уже в очереди')
                if (run / 'status.json').exists():
                    raise ValueError('Этот запуск уже выполнялся: создайте продолжение')
                existing.add(identity)
            data['items'] += [{'id': i, 'state': 'pending'} for i in identities]
            write_json(self.root / 'queue.json', data)
        return data

    def worker_active(self):
        return self._locked('worker')

    def compute_active(self):
        """Held by the queue worker, lab jobs and manual `humanoid_sim.cli train` runs."""
        return self._locked('compute')

    def _locked(self, name):
        try:
            with RunLock(self.root / name):
                return False
        except RuntimeError:
            return True

    def active_trainers(self):
        active = []
        for entry in self.entries():
            try:
                with RunLock(self.resolve(entry['id'])):
                    pass
            except RuntimeError:
                active.append(entry['id'])
        return active

    def start(self):
        with self.mutex:
            if self.worker_active() or self.process and self.process.poll() is None:
                raise ValueError('Очередь уже выполняется')
            if self.active_trainers():
                raise ValueError('Обучение ещё работает. Сначала остановите и дождитесь сохранения')
            if self.compute_active():
                raise ValueError('Вычисления заняты другим процессом (ручное обучение или проверка)')
            if not any(i['state'] == 'pending' for i in self.queue()['items']):
                raise ValueError('В очереди нет новых экспериментов')
            (self.root / 'stop.request').unlink(missing_ok=True)
            with (self.root / 'worker.log').open('ab') as log:
                self.process = subprocess.Popen([sys.executable, '-u', '-m', 'humanoid_sim.lab_worker', '--original', str(self.original)],
                    cwd=ROOT, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return {'accepted': True}

    def stop(self):
        (self.root / 'stop.request').touch()
        for identity in self.active_trainers():
            (self.resolve(identity) / 'stop.request').touch()
        return {'accepted': True}
