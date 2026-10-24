"""CPU only. prefilter.py <stage dir>... [--workers 3] [--out x.json]

The quick screen before stage_select (iteration 32; the critic's side_ctrl.py, lab-critic/it28b, and his advice in
lab-critic/it31): only the cells of step 2 that failed the stages of the blind-arm line, 22 runs a stage (23 with the
height command), about a minute a stage on 3 processes:
  - the side steps in our physics as step 2 runs them: 'наша' at ±0.3 and ±0.5 m/s and the bridge group's 9 side
    runs of 'наша +6 мс' again at ±0.5 (20 runs);
  - the walk at 0.5 m/s with no motor model and with motor friction x0.25 (2 runs);
  - with the height command, the crouched walk at 0.70 in the bridge copy (1 run).
Same cases and seeds as stage_select, so the numbers are its numbers. Passes: no falls, the worst |yaw rate| < 0.05 of
the side steps and the walks, the stop right after a side step < 0.3 m, the crouched walk < 0.03 (stage_select's
limits). Printed also: the mean drift into the step (left +, right -, over the four side cells of both variants; the
critic's measure of the systematic part, which training on the blind arms kept from the parent) and the common part.
Only stages that pass go on to stage_select."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_select as ss  # noqa: E402

st = ss.st
SIDE_VARIANTS = ('ours', 'ours_6ms')
WALK_VARIANTS = ('nomotor', 'fric')
SIDE_KEYS = ('L', 'R', 'L05', 'R05')
SIDE_LABELS = dict(L='влево', R='вправо', L05='влево 0.5 м/с', R05='вправо 0.5 м/с')


def fields_of(name):
    return next(fields for key, _, fields in ss.VARIANTS if key == name)


def jobs(stage, height):
    """stand_turn.run cases of one stage, as stage_select builds them."""
    stage = str(stage)
    cases = [st.case(plan=plan, stage=stage, **fields_of('ours')) for plan in ('side', ss.SIDE05)]
    bridge = [c for c in st.grid(stage, 'bridge') if c['physics'] == 'ours' and c['plan'] == 'side']
    cases += bridge + [dict(c, plan=ss.SIDE05) for c in bridge]
    cases += [st.case(plan='walk', stage=stage, **fields_of(name)) for name in WALK_VARIANTS]
    if height:
        cases.append(ss.height_jobs(stage)[0])
    return cases


def summary(results, height):
    """Numbers and reasons ([] = passes) from one stage's results."""
    side = {key: [] for key in SIDE_KEYS}
    stops, walks, falls = [], {}, []
    for r in results:
        case = r['case']
        if case['plan'] == ss.CROUCH_WALK:
            continue
        if r['fell_at'] is not None:
            falls.append('%s: %s' % (ss.variant_label(ss.variant_of(case)), ss.fall_of(r)))
            continue
        got = ss.measures(r)
        if case['plan'] == 'walk':
            walks[ss.variant_of(case)] = max(got['w05'], key=abs) if got['w05'] else None
        else:
            for key in SIDE_KEYS:
                side[key] += got[key]
            stops += got['stop'] + got['stop05']
    worst = {key: (max(values, key=abs) if values else None) for key, values in side.items()}
    mean = {key: (float(np.mean(values)) if values else None) for key, values in side.items()}
    known = [mean[key] for key in SIDE_KEYS if mean[key] is not None]
    into = (float(np.mean([mean['L'], -mean['R'], mean['L05'], -mean['R05']]))
            if len(known) == len(SIDE_KEYS) else None)
    common = float(np.mean(known)) if known else None
    reasons = ['падений %d (%s)' % (len(falls), '; '.join(falls))] if falls else []
    for key in SIDE_KEYS:
        if worst[key] is None:
            reasons.append('увод %s не измерен' % SIDE_LABELS[key])
        elif abs(worst[key]) >= ss.DRIFT_LIMIT:
            reasons.append('увод %s %+.3f, |.| >= %.2f' % (SIDE_LABELS[key], worst[key], ss.DRIFT_LIMIT))
    if stops and max(stops) >= ss.STOP_LIMIT:
        reasons.append('сдвиг после стопа за шагом вбок %.2f м >= %.1f' % (max(stops), ss.STOP_LIMIT))
    for name in WALK_VARIANTS:
        value = walks.get(name)
        if value is None:
            if not any(ss.variant_label(name) in f for f in falls):
                reasons.append('ходьба 0.5 м/с (%s) не измерена' % ss.variant_label(name))
        elif abs(value) >= ss.DRIFT_LIMIT:
            reasons.append('увод на ходу 0.5 м/с %+.3f, |.| >= %.2f (%s)' % (value, ss.DRIFT_LIMIT,
                                                                           ss.variant_label(name)))
    crouch = None
    if height:
        got = ss.crouch_run(results, ss.CROUCH_WALK, 'walk')
        crouch = None if got['seg'] is None else got['seg']['yaw_rate']
        if got['fell']:
            reasons.append('падение на ходу в приседе (%s)' % got['fell'])
        elif crouch is None:
            reasons.append('ходьба в приседе не измерена')
        elif abs(crouch) >= ss.CROUCH_DRIFT_LIMIT:
            reasons.append('увод на ходу в приседе %+.3f, |.| >= %.2f' % (crouch, ss.CROUCH_DRIFT_LIMIT))
    return dict(worst=worst, mean=mean, into_step=into, common=common, stop=max(stops, default=None), walks=walks,
                height=height, crouch=crouch, falls=falls, reasons=reasons, passes=not reasons)


def number(value):
    return '—' if value is None else '%+.3f' % value


def report(label, s):
    print('%s  %s' % (label, 'ПРОХОДИТ' if s['passes'] else 'мимо'))
    print('  вбок худшие: %s | в сторону шага %s, общая %s | стоп %s м' % (
        ', '.join('%s %s' % (key, number(s['worst'][key])) for key in SIDE_KEYS), number(s['into_step']),
        number(s['common']), '—' if s['stop'] is None else '%.2f' % s['stop']))
    print('  ходьба 0.5 м/с: %s%s' % (
        ', '.join('%s %s' % (ss.variant_label(name), number(s['walks'].get(name))) for name in WALK_VARIANTS),
        ' | в приседе 0.70 %s' % number(s['crouch']) if s['height'] else ''))
    for reason in s['reasons']:
        print('  - ' + reason)


def init():
    import torch
    torch.set_num_threads(1)


def work(item):
    index, case = item
    return index, st.run(case)


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Быстрый пред-фильтр перед stage_select: шаг вбок в нашей физике, '
                                                 'ходьба 0.5 без моторной модели и с трением x0.25, ходьба в приседе')
    parser.add_argument('stages', nargs='+', type=Path, help='папка этапа')
    parser.add_argument('--workers', type=int, default=3, help='рабочих процессов (пока идёт обучение — немного)')
    parser.add_argument('--out', type=Path, help='JSON: итоги и все прогоны')
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.stages]
    for p in paths:
        if not ((p / 'metadata.json').is_file() and (p / 'policy.zip').is_file()):
            raise SystemExit(f'Не этап: {p}')
    if not st.LOGS.is_dir():
        raise SystemExit(f'Нет логов связки (окна времени): {st.LOGS} — запущена ли WSL Ubuntu?')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # one core per worker
        os.environ.setdefault(key, '1')
    heights = [st.has_height(p) for p in paths]
    items = [(i, case) for i, p in enumerate(paths) for case in jobs(p, heights[i])]
    order = sorted(range(len(items)), key=lambda k: -ss.cost('run', items[k][1]))
    results = [None] * len(items)
    t0 = time.time()
    with Pool(args.workers, initializer=init, maxtasksperchild=16) as pool:
        for index, result in pool.imap_unordered(work, [(k, items[k][1]) for k in order]):
            results[index] = result
    print('этапов: %d, прогонов: %d, %.0f с' % (len(paths), len(items), time.time() - t0))
    stages = []
    for i, p in enumerate(paths):
        runs = [results[k] for k, item in enumerate(items) if item[0] == i]
        s = summary(runs, heights[i])
        report(ss.label(p), s)
        stages.append(dict(label=ss.label(p), path=str(p), **s, results=runs))
    passed = [s['label'] for s in stages if s['passes']]
    print('\nпред-фильтр проходят %d из %d: %s' % (len(passed), len(stages), ', '.join(passed) or '-'))
    if args.out:
        args.out.write_text(json.dumps(stages, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
