"""The versions behind the site's «Как это работает» page (web/how.html), read from the files that pin them, so the page
never states a version the lab does not run: requirements-gpu.txt (training on the GPU), requirements-lock.txt (the
CPU environment, export), scripts/sim2sim/setup.sh (Unitree's simulator build) and the vendored three.js. Served as
/api/lab/stack and exported by site_export as data/stack.<lang>.json (the same numbers in every language). counts()
gives the chapter on the AI team its numbers the same way: G1 branches, council iterations, journal entries, lessons."""
import json
import re
import sys

from .core import ROOT, read_json

# Page key -> (file, pinned name in it). The page shows each as <span data-version="key">.
PINNED = {
    'mujoco': ('requirements-lock.txt', 'mujoco'),
    'mujoco_warp': ('requirements-gpu.txt', 'mujoco-warp'),
    'warp': ('requirements-gpu.txt', 'warp-lang'),
    'torch': ('requirements-gpu.txt', 'torch'),
    'gymnasium': ('requirements-lock.txt', 'gymnasium'),
    'sb3': ('requirements-lock.txt', 'stable-baselines3'),
    'numpy': ('requirements-lock.txt', 'numpy'),
    'onnx': ('requirements-lock.txt', 'onnx'),
    'onnxruntime': ('requirements-lock.txt', 'onnxruntime'),
}


def _pins(name):
    path = ROOT / name
    if not path.is_file():
        return {}
    pins = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        package, _, version = line.strip().partition('==')
        if version:
            pins[package.lower()] = version
    return pins


def _search(path, pattern):
    path = ROOT / path
    match = re.search(pattern, path.read_text(encoding='utf-8')) if path.is_file() else None
    return match.group(1) if match else None


def versions():
    """{page key: version string}; a version the files do not give is left out (the page then shows none)."""
    files = {name: _pins(name) for name in {file for file, _ in PINNED.values()}}
    found = {key: files[file].get(package) for key, (file, package) in PINNED.items()}
    found['unitree_mujoco'] = _search('scripts/sim2sim/setup.sh', r'MUJOCO_VERSION=([0-9.]+)')
    found['three'] = _search('humanoid_sim/web/vendor/three.core.js', r"REVISION\s*=\s*'(\d+)")
    found['python'] = f'{sys.version_info.major}.{sys.version_info.minor}'
    return {key: value for key, value in found.items() if value}


def trained_steps(run, config):
    """Steps a branch trained itself: its newest step count (status.json or its last stage; pruned stages leave the
    status) less the stage it started from (a continuation's source_stage, a carried branch's warm_start)."""
    manifest = read_json(run / 'experiment.json') or {}
    start = config.get('source_stage') or (manifest.get('warm_start') or {}).get('source_stage') or 0
    newest = [int(path.name[6:]) for path in run.glob('stage_*') if path.name[6:].isdigit()]
    status = read_json(run / 'status.json') or {}
    return max(0, max(newest + [int(status.get('steps') or 0), int(start)]) - int(start))


def counts(lab_root=ROOT / 'runs' / 'humanoid_lab'):
    """{page key: number} of the lab as it is: G1 branches in the tree, iterations in the journal's «Совет
    лаборатории», journal entries, the lessons in docs/lab-norms.md; a source that is missing is left out."""
    found = {}
    branches, steps = 0, 0
    for config in lab_root.glob('exp-*/run_config.json'):
        try:
            values = json.loads(config.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not str(values.get('robot_id', '')).startswith('g1'):
            continue
        branches += 1
        steps += trained_steps(config.parent, values)
    if branches:
        found['branches'] = branches
        found['steps_bn'] = round(steps / 1e9, 1)
    saved = read_json(lab_root / 'journal' / 'journal.json') or {}
    for key in ('iterations', 'entries'):
        if saved.get(key):
            found[key] = len(saved[key])
    norms = ROOT / 'docs' / 'lab-norms.md'
    if norms.is_file():
        lessons = norms.read_text(encoding='utf-8').partition('## Уроки')[2].split('\n## ')[0]
        found['lessons'] = len(re.findall(r'(?m)^\d+\. ', lessons))
    return found
