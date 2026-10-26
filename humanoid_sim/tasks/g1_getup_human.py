"""Human starts of g1_getup (task_config human_starts, iteration 56; reference state init): poses and velocities of a
human getting up (LAFAN1 fallAndGetUp, subjects 1-4, retargeted to the G1 by Unitree, the head 0.3-1.1 m up, 50 Hz),
lifted out of the floor and kept only when the PD holds them (scripts/lab/getup_human_starts.py). A share of training
resets starts there instead of from the start bank, so the policy visits the states of a human rise (where the AMP
discriminator's human pairs lie).

The file is runs/humanoid_lab/getup_bank/human_starts.npz (untracked: LAFAN1 is CC BY-NC-ND 4.0, never published),
arrays only, read without pickle: qpos (n, 36), qvel (n, 35) float64; subject, clip, frame (n,) int64 (the 50 Hz frame
of the clip, the clip an index into the header's list); head, lift (n,) float64; header (JSON text: version, clips,
filter). The setting names the file's md5, so a run never starts from another file than the one it was branched with.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HUMAN_STARTS = ROOT / 'runs' / 'humanoid_lab' / 'getup_bank' / 'human_starts.npz'
HUMAN_STARTS_VERSION = 'g1-getup-human-starts-v1'


def load(md5, path=None):
    """The human starts of file `path` (default HUMAN_STARTS) as a dict of read-only arrays plus header (dict) and md5,
    read once per file version in a process. Raises FileNotFoundError with the command that makes it, ValueError when
    the file's md5 is not `md5` (None: not checked; tests) or its arrays do not fit."""
    path = Path(HUMAN_STARTS if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup human starts {path} are missing: python scripts/lab/getup_human_starts.py')
    status = path.stat()
    starts = _load(str(path.resolve()), status.st_mtime_ns, status.st_size)
    if md5 is not None and starts['md5'] != md5:
        raise ValueError(f'human starts {path}: md5 {starts["md5"]}, the setting names {md5} (another file)')
    return starts


@lru_cache(maxsize=2)
def _load(path, mtime, size):
    data = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        starts = {key: arrays[key] for key in arrays.files}
    header = json.loads(str(starts.pop('header')))
    if header.get('version') != HUMAN_STARTS_VERSION:
        raise ValueError(f'human starts {path}: version {header.get("version")}, expected {HUMAN_STARTS_VERSION}')
    qpos, qvel = starts['qpos'].astype(np.float64), starts['qvel'].astype(np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or qvel.shape != (len(qpos), 35) or not len(qpos) \
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError(f'human starts {path}: bad qpos / qvel arrays')
    starts.update(qpos=qpos, qvel=qvel)
    for value in starts.values():
        value.setflags(write=False)
    starts.update(header=header, md5=hashlib.md5(data).hexdigest())
    return starts
