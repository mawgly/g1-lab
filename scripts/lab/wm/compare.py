"""compare.py <old dir> <new dir>: two world model runs of the same procedure (collect.py's data/ and fit.py's fit/ in
each folder) are the same, for the refactor into adapters (27.10): every chunk array byte for byte (dtype and shape
too), model.pt's tensors torch.equal and its settings equal, train_state.pt (Adam, the window generator) equal, and
every key of the old report.json equal in the new one, recursively; keys only the new code writes are ignored, and so
are the run's timings and paths (training seconds and windows per second, evaluation_seconds, the curve's seconds,
data). Prints each difference; exit code 1 when there is one."""
import json
import math
import sys
from pathlib import Path

import numpy as np

STATE_ADDED = {'batch_mode'}  # fit.py's train_state since 27.10 (a resume keeps an auto-chosen batch)
TIMING = {('training', 'seconds'),('training', 'windows_per_second'), ('evaluation_seconds',), ('data',)}


def _same(old, new, path, differences):
    if path in TIMING or (len(path) == 3 and path[0] == 'curve' and path[2] == 'seconds'):
        return
    if isinstance(old, dict):
        if not isinstance(new, dict):
            differences.append(f'{"/".join(map(str, path))}: словарь против {type(new).__name__}')
            return
        for key, value in old.items():
            if key not in new:
                differences.append(f'{"/".join(map(str, path + (key,)))}: нет в новом')
            else:
                _same(value, new[key], path + (key,), differences)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            differences.append(f'{"/".join(map(str, path))}: другой список')
            return
        for index, (a, b) in enumerate(zip(old, new)):
            _same(a, b, path + (index,), differences)
    elif isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new):
        return
    elif old != new:
        differences.append(f'{"/".join(map(str, path))}: {old!r} против {new!r}')


def _tensors(old, new, path, differences):
    import torch
    if torch.is_tensor(old):
        if not (torch.is_tensor(new) and old.dtype == new.dtype and old.shape == new.shape and torch.equal(old, new)):
            differences.append(f'{path}: тензоры различаются')
    elif isinstance(old, dict):
        added = STATE_ADDED if path == 'train_state' else set()  # train_state keys only the new code writes
        if not isinstance(new, dict) or set(old) != set(new) - (added - set(old)):
            differences.append(f'{path}: другие ключи')
            return
        for key in old:
            if path.endswith('train_state') and key in ('seconds', 'curve'):
                continue
            _tensors(old[key], new[key], f'{path}/{key}', differences)
    elif isinstance(old, (list, tuple)):
        if not isinstance(new, (list, tuple)) or len(old) != len(new):
            differences.append(f'{path}: другой список')
            return
        for index, (a, b) in enumerate(zip(old, new)):
            _tensors(a, b, f'{path}/{index}', differences)
    elif old != new:
        differences.append(f'{path}: {old!r} против {new!r}')


def compare(old, new):
    """The differences (a list of Russian lines) between the runs in folders old and new."""
    import torch
    old, new = Path(old), Path(new)
    differences = []
    names = sorted(path.name for path in (old / 'data').glob('chunk_*.npz'))
    if names != sorted(path.name for path in (new / 'data').glob('chunk_*.npz')):
        differences.append('другие файлы данных')
    for name in names:
        with np.load(old / 'data' / name) as a, np.load(new / 'data' / name) as b:
            for key in a.files:
                if key not in b.files:
                    differences.append(f'{name}/{key}: нет в новом')
                elif (a[key].dtype != b[key].dtype or a[key].shape != b[key].shape
                      or a[key].tobytes() != b[key].tobytes()):
                    differences.append(f'{name}/{key}: массивы различаются')
    meta_old = json.loads((old / 'data' / 'meta.json').read_text(encoding='utf-8'))
    meta_new = json.loads((new / 'data' / 'meta.json').read_text(encoding='utf-8'))
    meta_old.pop('seconds', None)
    _same(meta_old, meta_new, ('meta',), differences)
    model_old = torch.load(old / 'fit' / 'model.pt', map_location='cpu', weights_only=False)
    model_new = torch.load(new / 'fit' / 'model.pt', map_location='cpu', weights_only=False)
    if model_old['settings'] != model_new['settings']:
        differences.append(f'model.pt settings: {model_old["settings"]} против {model_new["settings"]}')
    _tensors(model_old['state'], model_new['state'], 'model.pt', differences)
    state_old = torch.load(old / 'fit' / 'train_state.pt', map_location='cpu', weights_only=False)
    state_new = torch.load(new / 'fit' / 'train_state.pt', map_location='cpu', weights_only=False)
    _tensors(state_old, state_new, 'train_state', differences)
    report_old = json.loads((old / 'fit' / 'report.json').read_text(encoding='utf-8'))
    report_new = json.loads((new / 'fit' / 'report.json').read_text(encoding='utf-8'))
    _same(report_old, report_new, (), differences)
    return differences


def main():
    if len(sys.argv) != 3:
        raise SystemExit('compare.py <старая папка> <новая папка>')
    differences = compare(sys.argv[1], sys.argv[2])
    for line in differences:
        print(line)
    print('совпадает' if not differences else f'различий: {len(differences)}')
    sys.exit(1 if differences else 0)


if __name__ == '__main__':
    main()
