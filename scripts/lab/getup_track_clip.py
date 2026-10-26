"""CPU only. getup_track_clip.py [--source fallAndGetUp1_subject1.csv | cmu] [--subject 140 | <folder>] [--take 140_09]
[--frames a b] [--neutral N] [--settle 1.0] [--rounds 1] [--name <name>] [--force] [--out <file.npz>] [--data <folder>]
The clip of g1_getup's task_config track_clip (iteration 64; the critic's lab-critic/it63, select63.py / track63.py):
runs/humanoid_lab/getup_track/<name>.npz (tasks/g1_getup_track describes the arrays).

Source LAFAN1 (the default): LAFAN1's fallAndGetUp clips retargeted to the G1 29 DoF by Unitree
(.tools/datasets/lafan1_g1_unitree/g1, 30 frames/s; per frame the root position, its quaternion x y z w and the 29
joints in our order). The default is subject 1, fallAndGetUp1, frames 2104-2209 (3.5 s: back -> roll over -> knees ->
standing); the spare A′ is 2147-2209. The frames are resampled to the 50 Hz control step (joints and the root position
linearly, the root quaternion by slerp: getup_style_library.resample). LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0: a
research prototype, never published, and scripts/robot/package.py refuses a stage trained on it.

Source CMU (--source cmu): the CMU motion capture database, subject --subject N (default 140 «getting up from the
ground»; .tools/datasets/cmu_mocap_<N>: <N>.asf and <N>_<take>.amc, 120 frames/s; or a folder), take --take (subject
140: 140_08, 140_09 from the back, 01, 02 from the front, 03, 04 from a side; others: 111_06-08, 113_08, 139_16-18; a
bare 06 is <N>_06; another subject's take and frames have no default), AMC frames --frames a b (the file's numbers,
inclusive), retargeted by our
own IK in the getting-up model (cmu_mocap.retarget: the legs' and upper arms' directions, the hands relative to the
shoulders, the pelvis, the torso less the spine's bend in frame --neutral (default the take's first, lying flat) and the
feet's forward axes), at 120 Hz, then resampled to 50 Hz (cmu_mocap.resample). The subject ends every take in a half
crouch (knees ~65°, trunk ~40° forward), so --settle seconds (default 1) are added that take the last pose to the home
pose standing (cmu_mocap.settle: joints by a smoothstep, pelvis upright with its heading, the feet's midpoint kept).
The CMU licence (.tools/datasets/cmu_mocap_<N>/LICENSE.txt) allows commercial products but not reselling the data, even
converted: the clip stays under runs/ (untracked) and a policy trained on it may go to a robot, with the database's
acknowledgement in the package's README.

Since iteration 67 (cmu_mocap.RETARGET_VERSION 'cmu-ik-v2') the CMU IK pulls each joint toward the frame before and
caps the joints' and the root's speed, counts a foot, knee or hand down only while still, smooths the joints before the
50 Hz resampling, and settles by IK with both feet flat; its other mechanisms (support, flat feet, pins, weight
transfer, torque rounds --rounds, project) are off by default (cmu_mocap's comment says why).

Both: the locked wrists set to home, each frame raised by the smoothed depth of its deepest floor contact in the
getting-up model (Unitree's foot and hand capsules included; LAFAN's data sinks 5-12 cm), and the kinematics of the 14
tracked bodies taken (g1_getup_track.build). The header keeps the source file's md5, the dataset (lafan1 / cmu), which
package.py reads, and the clip's physical check (scripts/lab/probes/clip_feasibility.py: the verdict, its reasons and
numbers). An existing clip is not overwritten (stages name it by md5) without --force; --out writes elsewhere (trials).
Prints the clip, the lift, the check and the file's md5, which the setting names.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'probes'))
import clip_feasibility  # noqa: E402
from getup_style_library import DATA, SOURCE_RATE, resample  # noqa: E402

SOURCE, FRAMES = 'fallAndGetUp1_subject1.csv', (2104, 2209)
CMU_DATASETS = ROOT / '.tools' / 'datasets'  # subject N in cmu_mocap_N: N.asf and N_<take>.amc
CMU_SUBJECT = '140'
CMU_RATE = 120.
CMU_TAKE, CMU_FRAMES, CMU_SETTLE = '140_09', (61, 571), 1.
LAFAN_NOTE = ('LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0), retargeted to the G1 by Unitree; research prototype, not for '
              'publication or a robot package')
CMU_NOTE = ('CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), subject {subject}, retargeted to the G1 by the '
            'lab (scripts/lab/cmu_mocap.py); the licence allows commercial products but not reselling the data, even in '
            'converted form; acknowledgement: The data used in this project was obtained from mocap.cs.cmu.edu. The '
            'database was created with funding from NSF EIA-0196217.')


def cmu_subject(subject, data=None):
    """(folder, skeleton file, subject number) of a CMU subject: --subject a number (the folder cmu_mocap_<N> under
    .tools/datasets) or a folder; --data, if given, is the folder (the old option). The skeleton is <N>.asf, else the
    folder's only .asf; the number is its stem."""
    folder = Path(data) if data else (CMU_DATASETS / f'cmu_mocap_{subject}' if str(subject).isdigit() else Path(subject))
    if not folder.is_dir():
        raise SystemExit(f'нет папки {folder}')
    named = folder / f'{subject}.asf'
    found = [named] if str(subject).isdigit() and named.is_file() else sorted(folder.glob('*.asf'))
    if len(found) != 1:
        raise SystemExit(f'{folder}: нужен один файл скелета .asf, найдено {len(found)}')
    return folder, found[0], found[0].stem


def cmu_take(take, number):
    """The take's file stem: <N>_<take> as given, or the subject's number added to a bare take (06 -> 111_06); no take
    is subject 140's CMU_TAKE, and refused for another subject."""
    if take is None:
        if number != CMU_SUBJECT:
            raise SystemExit(f'субъект {number}: укажите --take (например {number}_01)')
        return CMU_TAKE
    return take if '_' in take else f'{number}_{take}'


def feasibility_header(summary):
    """The header's record of clip_feasibility's summary: the verdict, its reasons and the numbers it was judged by."""
    keys = ('feasible', 'reasons', 'com_margin_min', 'com_margin_at', 'residual_force_max', 'residual_moment_max',
            'ratio_max', 'ratio_joint', 'ratio_at', 'hand_max', 'hand_over_seconds', 'noise', 'qd_max', 'float_max',
            'feasible_waist_free', 'reasons_waist_free', 'waist_free_ratio_max', 'waist_free_ratio_joint')
    out = {key: round(summary[key], 4) if isinstance(summary[key], float) else summary[key] for key in keys}
    out.update(check='scripts/lab/probes/clip_feasibility.py', touch=clip_feasibility.TOUCH,
               criteria=dict(com_outside=clip_feasibility.COM_OUTSIDE, residual_force=clip_feasibility.RESIDUAL_FORCE,
                             residual_arm=clip_feasibility.RESIDUAL_ARM, torque_slack=clip_feasibility.TORQUE_SLACK))
    return out


def lafan_frames(args, env):
    """(qpos at 50 Hz, header fields, name) of a LAFAN1 clip."""
    first, last = args.frames or FRAMES
    data = args.data or DATA
    source = data / args.source
    if not source.is_file():
        raise SystemExit(f'нет файла {source}')
    rows = np.loadtxt(source, delimiter=',')
    if rows.ndim != 2 or rows.shape[1] != 36 or not 0 <= first < last < len(rows):
        raise SystemExit(f'{source.name}: {rows.shape}, кадры {first}-{last} вне клипа')
    qpos, _, _ = resample(rows[first:last + 1].copy(), env.dt)
    header = dict(dataset='lafan1', source_rate=SOURCE_RATE, source=source.name,
                  source_md5=hashlib.md5(source.read_bytes()).hexdigest(), frames=[first, last], data=LAFAN_NOTE)
    return qpos, header, args.name or f'lafan1-{source.stem}-{first}-{last}'


def cmu_frames(args, env):
    """(qpos at 50 Hz, header fields, name) of a CMU take, retargeted (cmu_mocap.retarget), resampled and settled."""
    import cmu_mocap
    folder, skeleton_file, subject = cmu_subject(args.subject, args.data)
    take = cmu_take(args.take, subject)
    if args.frames is None and subject != CMU_SUBJECT:
        raise SystemExit(f'{take}: укажите --frames (окно подъёма, номера кадров AMC)')
    first, last = args.frames or CMU_FRAMES
    source = folder / f'{take}.amc'
    if not source.is_file():
        raise SystemExit(f'нет файла {source}')
    skeleton = cmu_mocap.read_asf(skeleton_file)
    frames = cmu_mocap.read_amc(source)
    numbers = [number for number, _ in frames]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))) or not numbers[0] <= first < last <= numbers[-1]:
        raise SystemExit(f'{source.name}: кадры {numbers[0]}-{numbers[-1]}, а просят {first}-{last}')
    neutral = numbers[0] if args.neutral is None else args.neutral
    if not numbers[0] <= neutral <= numbers[-1]:
        raise SystemExit(f'{source.name}: нейтрального кадра {neutral} нет')
    chosen = frames[first - numbers[0]:last - numbers[0] + 1]
    rounds = {}
    qpos, errors, ratios = cmu_mocap.retarget(env, skeleton, chosen, frames[neutral - numbers[0]],
                                              cmu_mocap.ROUNDS if args.rounds is None else args.rounds, rounds)
    qpos = cmu_mocap.smooth(env, qpos, CMU_RATE)
    projected = {}
    if cmu_mocap.PROJECT:
        qpos = cmu_mocap.project(env, qpos, CMU_RATE, projected)
        waited = sum(frames for _, frames in projected['pauses'])
        print(f'   опора: переносов веса {len(projected["pauses"])} ({waited / CMU_RATE:.2f} с ожидания, не удалось '
              f'{len(projected["gave_up"])}), нехватка запаса ЦМ до {projected["short"] * 100:.1f} см')
    qpos = cmu_mocap.settle(env, cmu_mocap.resample(qpos, CMU_RATE, env.dt), args.settle)
    report = {}
    for k, target in enumerate(cmu_mocap.ERRORS):
        scale = 100 if target in cmu_mocap.DISTANCES else 180 / np.pi
        report[target] = [round(float(np.median(errors[:, k])) * scale, 1), round(float(errors[:, k].max()) * scale, 1)]
    header = dict(dataset='cmu', subject=subject, source_rate=CMU_RATE, source=source.name,
                  source_md5=hashlib.md5(source.read_bytes()).hexdigest(), skeleton=skeleton_file.name,
                  skeleton_md5=hashlib.md5(skeleton_file.read_bytes()).hexdigest(), frames=[first, last],
                  neutral_frame=neutral, settle_seconds=args.settle,
                  ratios={key: round(float(value), 4) for key, value in ratios.items()},
                  retarget_error=report, retarget_units='cm for hands, contacts, floor and support, degrees otherwise: '
                                                        '[median, max]',
                  retarget=dict(version=cmu_mocap.RETARGET_VERSION, rounds=rounds.get('rounds', []),
                                smooth_seconds=cmu_mocap.SMOOTH_SECONDS, com_margin=cmu_mocap.COM_MARGIN,
                                project=projected or None),
                  data=CMU_NOTE.format(subject=subject))
    print(f'ИК {source.name} кадры {first}-{last}: длины ног × {ratios["legs"]:.3f}, рук × {ratios["arms"]:.3f}; '
          'ошибка цели медиана / max: ' + ', '.join(f'{k} {v[0]:g}/{v[1]:g}' for k, v in report.items()))
    for k, r in enumerate(rounds.get('rounds', [])):
        waited = sum(frames for _, frames in r['pauses'])
        print(f'   проход {k + 1}: переносов веса {len(r["pauses"])} ({waited} кадров ожидания, не удалось '
              f'{len(r["gave_up"])}), пределы сдвинуты в {r["moved_frames"]} кадрах ({", ".join(r["joints"]) or "—"}), '
              f'нехватка запаса ЦМ до {r["support_short"] * 100:.1f} см')
    return qpos, header, args.name or f'cmu-{take}-{first}-{last}'


def main():
    import mujoco
    from humanoid_sim.tasks import g1_getup_track
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.g1_getup_bank import getup_model_env
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument('--source', default=SOURCE, help='файл клипа LAFAN1 в папке --data или cmu (запись CMU)')
    parser.add_argument('--subject', default=CMU_SUBJECT,
                        help='CMU: номер субъекта (папка .tools/datasets/cmu_mocap_<N>) или путь к папке с <N>.asf и '
                             'записями <N>_<take>.amc (по умолчанию 140)')
    parser.add_argument('--take', help='запись CMU (файл <take>.amc; 06 — то же, что <N>_06), для --source cmu; '
                                       'по умолчанию 140_09 у субъекта 140')
    parser.add_argument('--frames', type=int, nargs=2, metavar=('FROM', 'TO'),
                        help='первый и последний кадр включительно: LAFAN1 30 Гц (по умолчанию 2104 2209), CMU — '
                             'номера кадров AMC, 120 Гц (по умолчанию 61 571)')
    parser.add_argument('--neutral', type=int, help='CMU: кадр AMC с прямой спиной (по умолчанию первый в записи)')
    parser.add_argument('--settle', type=float, default=CMU_SETTLE,
                        help='CMU: секунд доводки последней позы до стойки home (0 — без неё)')
    parser.add_argument('--rounds', type=int, default=None,
                        help='CMU: проходов ИК с поправкой пределов по моментам (по умолчанию cmu_mocap.ROUNDS)')
    parser.add_argument('--name', help='имя клипа (файл runs/humanoid_lab/getup_track/<имя>.npz); по умолчанию '
                                       'lafan1-<файл>-<от>-<до> или cmu-<запись>-<от>-<до>')
    parser.add_argument('--force', action='store_true', help='перезаписать клип с тем же именем')
    parser.add_argument('--out', type=Path, help='записать клип в этот файл, а не в runs/humanoid_lab/getup_track (пробы)')
    parser.add_argument('--data', type=Path, help='папка с данными (LAFAN1 для G1 или CMU; для CMU то же, что '
                                                  '--subject <папка>)')
    args = parser.parse_args()
    env = getup_model_env()
    if env.model.nq != 36:
        raise SystemExit(f'модель с {env.model.nq} координатами, нужен G1 29 DoF')
    qpos, fields, name = (cmu_frames if args.source == 'cmu' else lafan_frames)(args, env)
    clip = g1_getup_track.build(env, qpos)
    # What is left of the floor penetration after the lift, per frame (m, ≤ 0).
    floor = env.model.geom('floor').id
    left = []
    for row in clip['qpos']:
        env.data.qpos[:] = row
        mujoco.mj_forward(env.model, env.data)
        left.append(g1_getup_track.floor_depth(env.model, env.data, env.robot_geoms, floor))
    head = task.head_height(clip['body_pos'][:, task.TRACK_BODIES.index(task.TRACK_ANCHOR)],
                            clip['body_rot'][:, task.TRACK_BODIES.index(task.TRACK_ANCHOR)])
    feasibility = clip_feasibility.check_clip(clip, env)['summary']
    header = dict(version=g1_getup_track.TRACK_VERSION, bodies=list(task.TRACK_BODIES), anchor=task.TRACK_ANCHOR,
                  rate=1 / env.dt, seconds=round((len(clip['qpos']) - 1) * env.dt, 3),
                  lift_frames=g1_getup_track.LIFT_FRAMES, locked_wrists='home', **fields,
                  feasibility=feasibility_header(feasibility))
    path = g1_getup_track.clip_path(name) if args.out is None else args.out
    if path.exists() and not args.force:
        raise SystemExit(f'{path} уже есть: этапы ссылаются на клип по md5, новый — под другим --name (или --force)')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.pending-{path.name}')
    with open(temporary, 'wb') as out:
        np.savez(out, **clip, header=np.array(json.dumps(header)))
    os.replace(temporary, path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    g1_getup_track.load(name, digest, path=path)
    first, last = fields['frames']
    print(f'{path}: {fields["source"]} кадры {first}-{last} → {len(clip["qpos"])} кадров 50 Гц ({header["seconds"]} с); '
          f'таз {clip["qpos"][0, 2]:.2f} → {clip["qpos"][-1, 2]:.2f} м, макушка {head[0]:.2f} → {head[-1]:.2f} м')
    print(f'подъём из пола: проникание до {clip["raw_lift"].max() * 100:.1f} см (медиана '
          f'{np.median(clip["raw_lift"]) * 100:.1f}), подъём после сглаживания до {clip["lift"].max() * 100:.1f} см; '
          f'осталось проникания до {abs(min(left)) * 100:.1f} см')
    print(f'выполнимость (clip_feasibility): {clip_feasibility.verdict_text(feasibility)}')
    print(f'без правила моментов пояса: {clip_feasibility.verdict_text(feasibility, waist_free=True)}')
    print(f'md5 {digest}')
    print(json.dumps({'track_clip': {'file': name, 'md5': digest}}))


if __name__ == '__main__':
    main()
