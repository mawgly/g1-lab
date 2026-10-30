"""CPU only. getup_clip_set.py [--workers 8] [--waist 3] [--hand 250] [--only <pattern>] [--also <clip> …] [--json <file>]
[--patch-out <file>] [--dry-run]
The clip set of g1_getup's task_config track_clips (iteration 78; Pavel 27.10 after SuperMimic: one tracker of many
rises). Every BONES-SEED rise from the floor of .tools/datasets/bones_seed/g1_getup/selection.csv (the move names with
stand_up_lying: on the back, the front (_stomach), a side (_side), after a faint, with their _M mirrors) is cut to its
rise (rise_window), made a track clip (scripts/lab/getup_track_clip.py --source bones, fit v2, 1 s settle to home;
runs/humanoid_lab/getup_track/<name>.npz, an existing one of the same name reused) and judged by
scripts/lab/probes/clip_feasibility.py, no physics: kept when the waist's torque need over what it has (needed /
available of the waist joints, the median over WAIST_WINDOW, its largest) is at most --waist and one hand's floor load
at most --hand N (the critic's rule, it76: the learned A493 v3 front needs 3.1× in its worst frame, the unlearnable
A495 v3 back up to 14×; the lead's 250 N on one hand); the penetration between robot parts (clip_feasibility's frames
deeper than 1 cm and the deepest, iteration 79) is printed beside them, not judged. Prints a table by start class (the
clip's first frame, g1_tensor_math.track_start_class), the kept clips per class and their list for track_clips (JSON);
--json writes every
row. --also judges made clips too (patched ones, getup_track_clip.py --patch) and takes them into the set when they
pass. --patch-out writes the task patch for scripts/lab/insert_root.py @<file> (two clips at least). --dry-run only
cuts the windows. LAFAN1 and CMU are not here: BONES-SEED's licence lets a model trained on it go to
a robot (scripts/robot/package.py, BONES_ATTRIBUTION), the clips themselves never leave runs/.

Rise window (rise_window, 120 Hz frames of the CSV): the standing height is the pelvis' median over the take's last
second; the rise ends at the first frame after the last lying one (pelvis below LIE_HEIGHT) within STAND_GAP of it;
it starts at the last still frame (joint speeds summed below REST_SPEED, smoothed over ±0.1 s) before the last lying
one, at most LEAD seconds before it, and lying.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'probes'))

DATA = ROOT / '.tools' / 'datasets' / 'bones_seed' / 'g1_getup'
SELECTION = DATA / 'selection.csv'
PATTERN = 'stand_up_lying'  # the floor rises among the selection's moves
RATE = 120.
LIE_HEIGHT = .25  # m: the pelvis below it lies
STAND_GAP = .01  # m: the rise ends this close to the standing height
REST_SPEED = 3.  # rad/s, the joint speeds summed: still below it
REST_SMOOTH = 12  # frames (±0.1 s at 120 Hz)
LEAD = 2.  # s: at most this much lying before the last lying frame
WAIST_WINDOW = 5  # frames (0.1 s at 50 Hz): the waist need's running median
WAIST_LIMIT, HAND_LIMIT = 3., 250.
# Calibration (the critic's it76): the learned front and the unlearnable back, printed beside the set.
CALIBRATION = ('bones-faint_stand_up_lying_stomach_R_001__A493-749-1237-v3',
               'bones-faint_stand_up_lying_R_001__A495-785-1234-v3', 'bones-stand_up_lying_R_002__A473-150-600-v2')
NAME_LIMIT = 64  # config.GETUP_TRACK_NAME's length
CLASS_NAMES = {'back': 'спина', 'front': 'живот', 'side': 'бок', None: 'стоя/сидя'}


def takes(pattern=PATTERN, only=None):
    """The selection's moves whose name has `pattern` (and `only`, a regular expression, if given), their CSV on disk:
    (stem, path) sorted by name."""
    with open(SELECTION, encoding='utf-8') as f:
        names = sorted({row['filename'] for row in csv.DictReader(f) if pattern in row['filename']})
    found = [(name, DATA / f'{name}.csv') for name in names if only is None or re.search(only, name)]
    return [(name, path) for name, path in found if path.is_file()]


def rise_window(numbers, qpos):
    """(first, last) Frame numbers of the rise in a take (numbers (F,), qpos (F, 36) of read_bones), or a reason (str)
    why there is none (the module docstring's rule)."""
    z = qpos[:, 2]
    stand = float(np.median(z[-int(RATE):]))
    lying = np.flatnonzero(z < LIE_HEIGHT)
    if not len(lying):
        return 'нет кадров лёжа'
    if stand < LIE_HEIGHT * 2:
        return f'в конце не стоит (таз {stand:.2f} м)'
    up = np.flatnonzero(z >= stand - STAND_GAP)
    for low in lying[::-1]:  # the last lying frame followed by standing
        after = up[up > low]
        if len(after):
            break
    else:
        return 'после лёжа не встаёт'
    end = int(after[0])
    lying = lying[lying <= low]
    speed = np.r_[np.abs(np.diff(qpos[:, 7:], axis=0)).sum(1) * RATE, 0.]
    index = np.clip(np.arange(len(speed))[:, None] + np.arange(-REST_SMOOTH, REST_SMOOTH + 1)[None], 0, len(speed) - 1)
    still = np.flatnonzero(speed[index].mean(1) < REST_SPEED)
    earliest = max(0, int(low - LEAD * RATE))
    rest = still[(still >= earliest) & (still <= low)]
    start = int(rest[-1]) if len(rest) else earliest
    candidates = lying[lying >= start]
    start = int(candidates[0]) if len(candidates) else int(low)
    if end - start < RATE / 2:
        return f'подъём короче 0,5 с ({start}-{end})'
    return int(numbers[start]), int(numbers[end])


def clip_name(stem, first, last):
    """getup_track_clip.py's name of a fit v2 clip, stand_up_lying shortened to sul past NAME_LIMIT characters."""
    name = f'bones-{stem}-{first}-{last}-v2'
    return name if len(name) <= NAME_LIMIT else name.replace('stand_up_lying', 'sul')


def waist_need(result, window=WAIST_WINDOW):
    """(the largest running median over `window` frames, the largest frame, when the first is (s)) of the waist joints'
    need: |torque| over what each has in its direction plus clip_feasibility's slack, the most of the three per frame
    (check_clip's rows)."""
    import clip_feasibility
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    names = result['names']
    waist = [j for j, name in enumerate(names) if 'waist' in name]
    peak = clip_feasibility.FrameChecker(getup_model_env()).peak[waist]
    torque = np.abs(np.array([r['torque'] for r in result['rows']]))[:, waist]
    have = np.maximum(np.array([r['have'] for r in result['rows']])[:, waist], 0.)
    need = (torque / (have + clip_feasibility.TORQUE_SLACK * peak[None])).max(1)
    half = window // 2
    index = np.clip(np.arange(len(need))[:, None] + np.arange(-half, half + 1)[None], 0, len(need) - 1)
    window_need = np.median(need[index], axis=1)
    return float(window_need.max()), float(need.max()), float(np.argmax(window_need) * result['rows'][1]['t'])


def judge(name):
    """clip_feasibility of the clip `name` and the set's numbers: seconds, start class, the waist need (window, frame),
    one hand's largest load (N), the clip's md5."""
    import torch
    torch.set_num_threads(1)
    import clip_feasibility
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    clip = g1_getup_track.load(name)
    result = clip_feasibility.check_clip(clip)
    window, frame, at = waist_need(result)
    gravity = task.projected_gravity(g1_getup_track.hold_frames(clip)['rot'][:1, 0])
    kind = int(task.track_start_class(gravity)[0])
    summary = result['summary']
    return dict(clip=name, md5=clip['md5'], seconds=round(summary['seconds'], 2),
                start_class=task.TRACK_CLASSES[kind] if kind >= 0 else None, waist=round(window, 2),
                waist_frame=round(frame, 2), waist_at=round(at, 2), hand=round(summary['hand_max'], 1),
                feasible=summary['feasible'], feasible_waist_free=summary['feasible_waist_free'],
                self_deep=summary['self_deep_frames'], self_depth=round(summary['self_depth_max'], 4))


def make(job):
    """The take's rise window, its clip (made by getup_track_clip.py unless one of that name is on disk) and judge's
    numbers; or the reason it has none."""
    from getup_track_clip import read_bones
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env, joint_names
    stem, path = job['stem'], Path(job['path'])
    numbers, qpos = read_bones(path, joint_names(getup_model_env()))
    window = rise_window(numbers, qpos)
    row = dict(take=stem, mirrored=stem.endswith('_M'))
    if isinstance(window, str):
        return dict(row, reason=window)
    first, last = window
    name = clip_name(stem, first, last)
    row.update(frames=[first, last], clip=name)
    if job['dry_run']:
        return row
    if not g1_getup_track.clip_path(name).is_file():
        done = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'lab' / 'getup_track_clip.py'), '--source',
                               'bones', '--file', str(path), '--frames', str(first), str(last), '--name', name],
                              capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=ROOT,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if done.returncode or not g1_getup_track.clip_path(name).is_file():
            return dict(row, reason='клип не сделан: ' + (done.stderr or done.stdout).strip().splitlines()[-1][:200])
        row['made'] = True
    return {**row, **judge(name)}


def verdict(row, waist=WAIST_LIMIT, hand=HAND_LIMIT):
    """None when the clip is kept, else the reason."""
    if 'reason' in row:
        return row['reason']
    reasons = []
    if row['waist'] > waist:
        reasons.append(f'пояс {row["waist"]:.1f}× > {waist:g}×')
    if row['hand'] > hand:
        reasons.append(f'кисть {row["hand"]:.0f} Н > {hand:g} Н')
    return '; '.join(reasons) or None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--waist', type=float, default=WAIST_LIMIT, help='наибольшая потребность пояса, раз')
    parser.add_argument('--hand', type=float, default=HAND_LIMIT, help='наибольшая нагрузка на одну кисть, Н')
    parser.add_argument('--only', help='только записи, чьё имя подходит под это регулярное выражение')
    parser.add_argument('--json', type=Path, help='все строки таблицы, JSON')
    parser.add_argument('--patch-out', type=Path,
                        help='правка задачи для insert_root.py @<файл>: {"track_clip": null, "track_clips": [...]}')
    parser.add_argument('--dry-run', action='store_true', help='только окна подъёма, без клипов')
    parser.add_argument('--also', action='append', default=[],
                        help='ещё сделанный клип runs/humanoid_lab/getup_track/<имя>.npz (например, с правкой --patch): '
                             'судить так же и взять в набор, если проходит (можно несколько раз)')
    args = parser.parse_args()
    if not SELECTION.is_file():
        raise SystemExit(f'нет {SELECTION}')
    jobs = [dict(stem=stem, path=str(path), dry_run=args.dry_run) for stem, path in takes(only=args.only)]
    print(f'записей подъёма с пола в выборке: {len(jobs)}' + (f', ещё клипов: {len(args.also)}' if args.also else ''))
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        rows = pool.map(make, jobs, chunksize=1)
        if not args.dry_run:
            rows += [dict(take=name, mirrored=name.split('-')[1].endswith('_M'), also=True, **row)
                     for name, row in zip(args.also, pool.map(judge, args.also))]
        calibration = [] if args.dry_run else pool.map(judge, [name for name in CALIBRATION if (
            ROOT / 'runs' / 'humanoid_lab' / 'getup_track' / f'{name}.npz').is_file()])
    if args.dry_run:
        for row in rows:
            print(f'   {row["take"]}: ' + (row['reason'] if 'reason' in row else
                                          f'кадры {row["frames"][0]}-{row["frames"][1]} → {row["clip"]}'))
        return
    print(f'правило: пояс (медиана по {WAIST_WINDOW} кадрам, наибольшая) ≤ {args.waist:g}×, одна кисть ≤ {args.hand:g} Н')
    for row in calibration:
        print(f'   опора: {row["clip"]}: пояс {row["waist"]:.1f}× (кадр {row["waist_frame"]:.1f}×), кисть '
              f'{row["hand"]:.0f} Н, проникновение {row["self_deep"]} кадров до {row["self_depth"] * 100:.1f} см')
    kept = []
    for kind in ('back', 'front', 'side', None):
        mine = [r for r in rows if r.get('start_class', 'none') == kind]
        if not mine:
            continue
        print(f'{CLASS_NAMES[kind]}:')
        for row in mine:
            why = verdict(row, args.waist, args.hand)
            row['kept'] = why is None
            if row['kept']:
                kept.append(row)
            print(f'   {"+" if row["kept"] else "-"} {row["clip"]} {row["seconds"]:.2f} с, пояс {row["waist"]:.1f}× '
                  f'(кадр {row["waist_frame"]:.1f}×, {row["waist_at"]:.2f} с), кисть {row["hand"]:.0f} Н, '
                  f'проникновение {row["self_deep"]} кадров до {row["self_depth"] * 100:.1f} см'
                  + (f' — {why}' if why else ''))
    failed = [r for r in rows if 'reason' in r]
    for row in failed:
        row['kept'] = False
        print(f'   - {row["take"]}: {row["reason"]}')
    counts = {CLASS_NAMES[kind]: sum(r['start_class'] == kind for r in kept) for kind in ('back', 'front', 'side', None)}
    print(f'оставлено {len(kept)} из {len(rows)}: ' + ', '.join(f'{k} {v}' for k, v in counts.items() if v))
    setting = [{'file': r['clip'], 'md5': r['md5']} for r in kept]
    print('track_clips: ' + json.dumps(setting, ensure_ascii=False))
    if args.patch_out:
        if len(setting) < 2:
            print(f'правка не записана: в наборе {len(setting)} клип(ов), track_clips нужно от 2')
        else:
            args.patch_out.write_text(json.dumps({'track_clip': None, 'track_clips': setting}, ensure_ascii=False),
                                      encoding='utf-8')
    if args.json:
        args.json.write_text(json.dumps(dict(rule=dict(waist=args.waist, hand=args.hand, window=WAIST_WINDOW),
                                             rows=rows, calibration=calibration, track_clips=setting),
                                        ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
