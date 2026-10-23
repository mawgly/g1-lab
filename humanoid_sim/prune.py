"""Thin out the stages of finished branches: keep what the research and the journal need, recycle the rest.

    .venv\\Scripts\\python.exe -m humanoid_sim.prune exp-… [exp-…] [--keep 5] [--apply]
    .venv\\Scripts\\python.exe -m humanoid_sim.prune --finished [--apply]

Kept per branch: `keep` stages spread evenly from the first to the last (the progress a journal race shows),
the report's best and last stage, stages other branches start from, stages the journal's races show, and
stages the branch's notes mention («… 674,1 млн …»). Without --apply only the plan is printed. Removed stages
go to the Windows Recycle Bin, so they can be restored; a branch whose trainer runs and the original run
(baseline) are never touched. Decided with the user on 20.10.2026: they follow a few progress stages per
branch in the journal, not every saved checkpoint.
"""
import argparse
import re
from pathlib import Path

from .core import DEFAULT_RUN, RunLock, list_stages, read_json
from .lab import Laboratory

MENTION = re.compile(r'(\d+(?:[.,]\d+)?)\s*млн')


def spread(stages, keep):
    """`keep` stages evenly spaced over the sorted step counts, first and last included."""
    if len(stages) <= keep:
        return set(stages)
    return {stages[round(i * (len(stages) - 1) / (keep - 1))] for i in range(keep)}


def pinned(lab, branch, stages):
    """Stages of `branch` that something refers to: report, child branches, journal races, notes."""
    from . import journal
    found = set()
    summary = read_json(lab.resolve(branch) / 'report.json', {}).get('summary', {})
    found |= {summary.get('last_steps'), (summary.get('best') or {}).get('steps')}
    for entry in lab.entries():
        if entry.get('config', {}).get('source_id') == branch:
            found.add(entry['config'].get('source_stage'))
    data = journal.load(lab.root)
    races = [entry.get('race') for entry in data['entries']]
    for stage in data['roadmap']:  # the stage stories: «было → стало» of each path step and the comparisons
        story = stage.get('story') or {}
        races += [step.get('race') for step in story.get('path', [])] + list(story.get('races', []))
    for race in races:
        for lane in (race or {}).get('lanes', []):
            if lane['experiment'] == branch:
                found.add(lane['stage'])
    note = lab.notes().get(branch, {})
    for text in [h.get('note') or '' for h in note.get('history', [])]:
        for value in MENTION.findall(text):
            millions = float(value.replace(',', '.')) * 1e6
            nearest = min(stages, key=lambda s: abs(s - millions))
            if abs(nearest - millions) <= 1.5e6:  # «674,1 млн» names the stage saved at 674 070 528
                found.add(nearest)
    return found & set(stages)


def plan(lab, branch, keep=5):
    """(kept steps, removed stage dirs) for one branch; nothing for a running or original branch."""
    if branch == 'baseline':
        raise ValueError('Исходный эксперимент не чистится')
    run = lab.resolve(branch)
    try:
        with RunLock(run):
            pass
    except RuntimeError:
        raise ValueError(f'{branch}: идёт обучение, чистить нельзя')
    dirs = {int(p.name[6:]): p for p in list_stages(run)}
    stages = sorted(dirs)
    if not stages:
        return set(), []
    kept = spread(stages, keep) | pinned(lab, branch, stages)
    return kept, [dirs[s] for s in stages if s not in kept]


def recycle(path):
    """Moves a file or folder to the Windows Recycle Bin (restorable), never deleting it outright."""
    import ctypes
    from ctypes import wintypes

    class Operation(ctypes.Structure):
        _fields_ = [('hwnd', wintypes.HWND), ('wFunc', wintypes.UINT), ('pFrom', wintypes.LPCWSTR),
                    ('pTo', wintypes.LPCWSTR), ('fFlags', ctypes.c_uint16), ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', ctypes.c_void_p), ('lpszProgressTitle', wintypes.LPCWSTR)]
    # FO_DELETE with FOF_ALLOWUNDO recycles; FOF_WANTNUKEWARNING asks before anything too big for the bin.
    flags = 0x4 | 0x10 | 0x40 | 0x400 | 0x4000
    operation = Operation(None, 3, str(Path(path).resolve()) + '\0', None, flags, False, None, None)
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if code or operation.fAnyOperationsAborted or Path(path).exists():
        raise OSError(f'Не удалось переместить в Корзину: {path} (код {code})')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('branches', nargs='*', help='experiment ids')
    parser.add_argument('--finished', action='store_true', help='every lab branch that is not training')
    parser.add_argument('--keep', type=int, default=5, help='evenly spread progress stages to keep')
    parser.add_argument('--apply', action='store_true', help='move the removed stages to the Recycle Bin')
    args = parser.parse_args()
    lab = Laboratory(DEFAULT_RUN)
    branches = args.branches or ([e['id'] for e in lab.entries() if e['id'] != 'baseline'] if args.finished else [])
    total, freed = 0, 0
    for branch in branches:
        try:
            kept, removed = plan(lab, branch, max(2, args.keep))
        except ValueError as exc:
            print(exc, flush=True)
            continue
        size = sum(f.stat().st_size for d in removed for f in d.rglob('*') if f.is_file())
        print(f'{branch}: оставить {len(kept)} этапов ({", ".join(f"{s / 1e6:.0f}" for s in sorted(kept))} млн), '
              f'убрать {len(removed)} ({size / 2 ** 20:.0f} МБ)', flush=True)
        if args.apply:
            for directory in removed:
                recycle(directory)
        total, freed = total + len(removed), freed + size
    print(f'{"Перемещено в Корзину" if args.apply else "Будет перемещено в Корзину"}: {total} этапов, '
          f'{freed / 2 ** 30:.2f} ГБ', flush=True)


if __name__ == '__main__':
    main()
