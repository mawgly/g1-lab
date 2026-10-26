"""The style libraries of g1_getup's style reward weight (iteration 53): human frames getting up, as
g1_tensor_math.style_features, against which the term measures each frame (g1_tensor_math.style_distance).

A library is runs/humanoid_lab/getup_style/<name>.npz (task_config style_library names it), made by
scripts/lab/getup_style_library.py from LAFAN1's fallAndGetUp clips retargeted to the G1 by Unitree. LAFAN1 (Ubisoft La
Forge) is CC BY-NC-ND 4.0: neither the clips nor anything made from them (this library) are published, so the folder is
under runs/ (untracked), not in the repository or assets/, and a policy trained with the style term does not go into a
client's robot package. Arrays only, read without pickle:
- features (m, f) float64: the library's frames (subjects 1-4, the head in STYLE_HEAD_BAND);
- holdout (k, f) float64: the same of the held-out subject (5), a positive control for probes, never matched against;
- joints (j,) int64: the joints of the features (the free ones, all but the locked wrists);
- source (m,) int64 and time (m,) float64: each frame's clip (header files) and time in it (s);
- header: JSON text (version STYLE_VERSION, clips, subjects, rate, band, scales).
"""
import json
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from .config import GETUP_STYLE_NAME
from .g1_tensor_math import STYLE_VERSION, style_norms

ROOT = Path(__file__).resolve().parents[2]
STYLE_DIR = ROOT / 'runs' / 'humanoid_lab' / 'getup_style'


def library_path(name):
    """The file of library `name` (config.GETUP_STYLE_NAME: a file name, never a path)."""
    if not isinstance(name, str) or not re.fullmatch(GETUP_STYLE_NAME, name):
        raise ValueError('style_library must be a library name (letters, digits, _ and -)')
    return STYLE_DIR / f'{name}.npz'


def load(name):
    """Library `name` as a dict of read-only arrays (features, holdout, joints, source, time, norms: style_norms of
    features) plus header (dict), read once per file version in a process. Raises FileNotFoundError with the command
    that makes it, ValueError for another feature layout."""
    path = library_path(name)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup style library {path} is missing: '
                                f'python scripts/lab/getup_style_library.py --name {name}')
    status = path.stat()
    return _load(str(path.resolve()), status.st_mtime_ns, status.st_size)


@lru_cache(maxsize=2)
def _load(path, mtime, size):
    with np.load(path, allow_pickle=False) as data:
        library = {key: data[key] for key in data.files}
    header = json.loads(str(library.pop('header')))
    if header.get('version') != STYLE_VERSION:
        raise ValueError(f'style library {path}: version {header.get("version")}, expected {STYLE_VERSION}')
    features = library['features'].astype(np.float64)
    joints = library['joints'].astype(np.int64)
    if features.ndim != 2 or not len(features) or features.shape[1] != 2 * len(joints) + 5 \
            or library['holdout'].shape[1:] != features.shape[1:] or not np.isfinite(features).all():
        raise ValueError(f'style library {path}: bad feature arrays')
    library.update(features=features, holdout=library['holdout'].astype(np.float64), joints=joints,
                   norms=style_norms(features))
    for value in library.values():
        value.setflags(write=False)
    library['header'] = header
    return library
