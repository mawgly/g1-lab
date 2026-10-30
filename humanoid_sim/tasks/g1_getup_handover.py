"""The hand-over bank of g1_getup's task_config track_handover_starts (iteration 83): the states in which a getting-up
policy (by default stage 1164 of exp-207eb498dc824027) hands the robot over to a tracker of one human clip, so the
tracker learns to take a robot in motion (getup_chain.py: handed over directly, tracker 73 took 0 of 80).

The file is runs/humanoid_lab/getup_bank/<name>.npz (untracked; scripts/lab/getup_handover_bank.py makes it: the
first policy from the start bank's training rows in several physics, with the CUDA task's sensor noise and an encoder
bias, until getup_chain.HandOver fires on the robot's own sensors), arrays only, read without pickle:
  qpos (n, 36), qvel (n, 35) float64: the simulated state at the hand-over;
  actions (n, HANDOVER_ACTIONS, 29) float64: the first policy's last applied actions, newest first (the actions in
    flight under an action delay when the tracker takes over);
  row (n,) int64: the start bank's training row it came from; start_class (n,) int64: its class (an index into the
    header's classes); physics (n,) int64: an index into the header's physics; seed (n,) int64;
  seconds (n,) float64: the hand-over time after the start; rule (n,) int64: an index into the header's rules;
  angle, deviation (n,) float64: the observed pelvis gravity angle (rad) and mean |q − q0| (rad) against the target
    frame; joint_speed (n,) float64: the fastest free joint (rad/s); nearest (n,) int64: the clip frame nearest the
    observed pose (getup_chain's direct rule, for reference);
  header (JSON text): version, first (the stage), clip (file, md5: the tracker's track_clip), frame (where the tracker's
    clip clock starts), classes, physics, rules, the generator's settings and counts.
A bank made with getup_handover_bank.py --pd (iteration 86) holds the state at the end of the PD transition that follows
the hand-over (getup_chain.pd_transition, as g1_ctrl's GetUpBlend) instead: qpos, qvel then, actions the transition's
last applied ones (its targets over the action scale, the ones in flight when the tracker takes over); seconds, rule,
angle, deviation and nearest still describe the hand-over itself; the header's pd names the transition (seconds and
shape), and PD_ARRAYS (n,) float64 describe the state the tracker gets: pd_angle and pd_deviation (the true pelvis
gravity angle, rad, and mean |q − q0| over the free joints against the frame), pd_joint_speed (fastest free joint,
rad/s), pd_torso (the torso origin's height, m), pd_arm_worst (the largest |q − q0| of a free arm joint, rad),
pd_anchor_gap and pd_up_gap (the torso's height and z-axis z minus the frame's).
The setting names the file's md5, so a run never starts from another bank than the one it was branched with.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HANDOVER_DIR = ROOT / 'runs' / 'humanoid_lab' / 'getup_bank'
HANDOVER_VERSION = 'g1-getup-handover-v1'
HANDOVER_ACTIONS = 3  # the actions in flight under action_delay up to 0.05 s (25 substeps of 10 a step: 3 steps)
ARRAYS = ('qpos', 'qvel', 'actions', 'row', 'start_class', 'physics', 'seed', 'seconds', 'rule', 'angle', 'deviation',
          'joint_speed', 'nearest')
PD_ARRAYS = ('pd_angle', 'pd_deviation', 'pd_joint_speed', 'pd_torso', 'pd_arm_worst', 'pd_anchor_gap', 'pd_up_gap')
# Optional per-row arrays, written when given: PD_ARRAYS and human_rom (n,) bool, the source row of the start bank
# within a human's range of motion (getup_handover_bank.HUMAN_ROM, judged on the row before the run).
OPTIONAL_ARRAYS = (*PD_ARRAYS, 'human_rom')


def bank_path(name):
    """runs/humanoid_lab/getup_bank/<name>.npz (name a config.TRACK_HANDOVER_NAME, checked by the task's validator)."""
    return HANDOVER_DIR / f'{name}.npz'


def load(name, md5=None, path=None):
    """The hand-over bank `name` (or the file `path`) as a dict of read-only arrays plus header (dict) and md5, read
    once per file version in a process. Raises FileNotFoundError with the command that makes it, ValueError when the
    file's md5 is not `md5` (None: not checked; tests) or its arrays do not fit."""
    path = Path(bank_path(name) if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(f'g1_getup hand-over bank {path} is missing: python scripts/lab/getup_handover_bank.py')
    status = path.stat()
    bank = _load(str(path.resolve()), status.st_mtime_ns, status.st_size)
    if md5 is not None and bank['md5'] != md5:
        raise ValueError(f'hand-over bank {path}: md5 {bank["md5"]}, the setting names {md5} (another file)')
    return bank


@lru_cache(maxsize=2)
def _load(path, mtime, size):
    data = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        bank = {key: arrays[key] for key in arrays.files}
    header = json.loads(str(bank.pop('header')))
    if header.get('version') != HANDOVER_VERSION:
        raise ValueError(f'hand-over bank {path}: version {header.get("version")}, expected {HANDOVER_VERSION}')
    check(bank, header, f'hand-over bank {path}')
    for value in bank.values():
        value.setflags(write=False)
    bank.update(header=header, md5=hashlib.md5(data).hexdigest())
    return bank


def check(bank, header, what='hand-over bank'):
    """Raises ValueError unless bank's qpos, qvel and actions have their shapes (n ≥ 1 rows of the 29-DoF G1) and are
    finite, every other array of ARRAYS (and of OPTIONAL_ARRAYS present) has n rows, and header names the clip (file, md5)
    and a frame ≥ 0."""
    n = len(bank.get('qpos', ()))
    shapes = dict(qpos=(n, 36), qvel=(n, 35), actions=(n, HANDOVER_ACTIONS, 29))
    if n < 1 or any(key not in bank or np.shape(bank[key]) != shape or not np.isfinite(bank[key]).all()
                    for key, shape in shapes.items()) or any(
            key not in bank or np.shape(bank[key]) != (n,) for key in ARRAYS if key not in shapes) or any(
            np.shape(bank[key]) != (n,) for key in OPTIONAL_ARRAYS if key in bank):
        raise ValueError(f'{what}: bad arrays (expected {shapes} and (n,) for the others)')
    clip, frame = header.get('clip'), header.get('frame')
    if not isinstance(clip, dict) or not isinstance(clip.get('file'), str) or not isinstance(clip.get('md5'), str) \
            or isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f'{what}: the header must name the clip ({{file, md5}}) and the clock frame')


def save(path, arrays, header):
    """Writes the bank (ARRAYS, the OPTIONAL_ARRAYS given, and the header as JSON text) to path with numpy's savez, no
    pickle; returns its md5."""
    header = {**header, 'version': HANDOVER_VERSION}
    rows = {key: np.asarray(arrays[key]) for key in (*ARRAYS, *(k for k in OPTIONAL_ARRAYS if k in arrays))}
    check(rows, header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending.npz')
    np.savez(temporary, header=np.array(json.dumps(header, ensure_ascii=False)), **rows)
    temporary.replace(path)
    return hashlib.md5(path.read_bytes()).hexdigest()
