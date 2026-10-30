"""Training a lab branch on a rented Linux GPU server over SSH (27.10, the first test: immers.cloud RTX 2080 Ti).

    .venv\\Scripts\\python.exe scripts\\remote\\remote.py up <experiment> [--host gpu1] [--steps N] [--cuda cu128]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py status [<experiment>] [--host gpu1]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py stop <experiment> [--host gpu1]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py fetch <experiment> [--host gpu1]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py clean [--host gpu1] [--all]

The world model (SuperMimic stage 1, scripts/lab/wm), without training a branch:
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py wm-up <stage dir | experiment:steps> [--host] [--cuda cu128]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py wm-start [--host] [--noise 0.1] [--minutes 60] [--worlds 4096]
        [--steps 1000] [--det-worlds 512]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py wm-log [--host]
    .venv\\Scripts\\python.exe scripts\\remote\\remote.py wm-fetch [--host]
`wm-up` sends HEAD, builds .venv-fullgpu, puts the Warp cache of the card's architecture, sends exactly that stage
folder and the data its task_config reads, and names it in runs/humanoid_lab/world_model/current on the server.
`wm-start` writes and starts under nohup (log wm.log) a script that collects the main set (--action-noise), a
deterministic one (--det-worlds worlds, seed 1), fits for --minutes (checkpoints every 10 min; started again it skips
finished sets and resumes the fit), evaluates the model on the deterministic set and packs the Warp cache. `wm-fetch`
brings fit/, fit-det/, wm.log and both sets' meta.json into runs/humanoid_lab/world_model/<tag>/ and the Warp cache
into .tools/remote/warp-cache-sm<NN>.tar (if there is none yet). `clean --all` also removes the world model's data
(the reference clip may be BONES-SEED).

The host is an ~/.ssh/config entry with key login (user ubuntu, ~/sim on the server). `up` sends the lab's HEAD (git
archive, so commit first), builds .venv-fullgpu once (setup_linux.sh), sends the branch (run_config, manifest, its
newest stage) and the data files its task_config reads (pose bank, track / settle clips, human starts, style
libraries), then starts `humanoid_sim.cli train` under nohup; `--steps` caps this run (run_config `steps` on the
server's copy). `fetch` brings back the stages the local branch lacks plus status, training.log and SB3 logs, so the
tree and the probes see them. `up` leaves remote.json (host, commit, stage sent) and a status.json in the local
branch, so the local queue refuses it while it trains there; continuing elsewhere is a new branch from a fetched
stage, as for any finished run. `clean` removes the licensed data (BONES-SEED / CMU / LAFAN1 clips and libraries) from
the server; `--all` also the runs. The first `fetch` from a GPU architecture keeps the server's compiled Warp kernels in
.tools/remote/warp-cache-sm<NN>.tar, which `up` puts on the next server of that architecture. Create the branch locally first without queueing it (it is a normal lab branch)."""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid_sim.core import list_stages, read_json, write_json  # noqa: E402

LAB = ROOT / 'runs' / 'humanoid_lab'
GIT = ROOT / '.tools' / 'git' / 'cmd' / 'git.exe'
CACHE = ROOT / '.tools' / 'remote'  # warp-cache-sm<NN>.tar: the server's compiled Warp kernels, by GPU architecture
LICENSED = ('runs/humanoid_lab/getup_track', 'runs/humanoid_lab/getup_style',
            'runs/humanoid_lab/getup_bank/human_starts.npz')


def ssh(host, command, data=None, capture=False):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command], input=data, capture_output=capture,
                            check=True)
    return result.stdout if capture else None


def data_files(task_config):
    """The lab files under runs/ a branch's task reads (paths relative to the repository)."""
    files = []
    if ('track_clip' in task_config or 'track_clips' in task_config or 'settle_clip' in task_config
            or 'assist_force' in task_config):
        files += ['runs/humanoid_lab/getup_bank/bank.npz', 'runs/humanoid_lab/getup_bank/bank.json']
    clips = ([task_config.get('track_clip')] + list(task_config.get('settle_clip') or [])
             + list(task_config.get('track_clips') or []))
    files += [f'runs/humanoid_lab/getup_track/{clip["file"]}.npz' for clip in clips if clip]
    if task_config.get('human_starts'):
        files.append('runs/humanoid_lab/getup_bank/human_starts.npz')
    weights = task_config.get('reward_weights') or {}
    if weights.get('style') or weights.get('amp') or task_config.get('style_judge'):
        name = task_config.get('style_library') or 'lafan1'
        files += [f'runs/humanoid_lab/getup_style/{name}.npz', f'runs/humanoid_lab/getup_style/{name}-amp.npz']
    return [path for path in files if (ROOT / path).is_file()]


def architecture(host):
    cap = ssh(host, 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1', capture=True)
    return 'sm' + cap.decode().strip().replace('.', '')


def pack(paths):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for path in paths:
            tar.add(ROOT / path, arcname=path)
    return buffer.getvalue()


def send_code(host, cuda):
    """HEAD by git archive into ~/sim, .venv-fullgpu (setup_linux.sh), the Warp cache of the card's architecture."""
    dirty = subprocess.run([str(GIT), 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        print('внимание: на сервер уходит последний коммит, незакоммиченные правки не попадут:\n' + dirty)
    code = subprocess.run([str(GIT), 'archive', '--format=tar', 'HEAD'], cwd=ROOT, capture_output=True,
                          check=True).stdout
    ssh(host, 'mkdir -p ~/sim && tar -xf - -C ~/sim', data=code)
    print('код отправлен')
    ssh(host, f'bash ~/sim/scripts/remote/setup_linux.sh {cuda}')
    cache = CACHE / f'warp-cache-{architecture(host)}.tar'
    if cache.is_file():  # kernels compiled on a card of this architecture before: no compile of several minutes
        ssh(host, 'mkdir -p ~/.cache && tar -xf - -C ~/.cache', data=cache.read_bytes())
        print('ядра Warp из кеша:', cache.name)


def up(args):
    run = LAB / args.experiment
    config = read_json(run / 'run_config.json')
    if config is None:
        raise SystemExit(f'нет ветки {args.experiment}')
    if config['backend'] != 'full_gpu':
        raise SystemExit('на сервер отправляются только ветки full_gpu')
    stages = list_stages(run)
    if not stages:
        raise SystemExit(f'у {args.experiment} нет этапов')
    send_code(args.host, args.cuda)
    branch = f'runs/humanoid_lab/{args.experiment}'
    paths = [f'{branch}/run_config.json', f'{branch}/experiment.json', f'{branch}/{stages[-1].name}']
    paths += data_files(config['task_config'])
    ssh(args.host, 'cd ~/sim && tar -xf -', data=pack(paths))
    print('отправлено:', ', '.join(Path(path).name for path in paths))
    steps = ''
    if args.steps:  # this run only: the server's copy of run_config
        steps = (f"&& .venv-fullgpu/bin/python -c \"import json,sys; p=sys.argv[1]; c=json.load(open(p)); "
                 f"c['steps']={int(args.steps)}; json.dump(c, open(p, 'w'), ensure_ascii=False, indent=2)\" "
                 f"{branch}/run_config.json ")
    ssh(args.host, f'cd ~/sim {steps}&& rm -f {branch}/stop.request && (nohup .venv-fullgpu/bin/python -u -m '
                   f'humanoid_sim.cli train --run-dir {branch} --config {branch}/run_config.json '
                   f'>> {branch}/training.log 2>&1 < /dev/null &)')
    commit = subprocess.run([str(GIT), 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    # The branch now trains there: remote.json says where and on which code; a local status.json makes lab.enqueue
    # refuse it here ("уже выполнялся"), so one branch never trains in two places; fetch replaces it with the server's.
    write_json(run / 'remote.json', {'host': args.host, 'commit': commit, 'stage': stages[-1].name,
                                     'started': time.time()})
    if not (run / 'status.json').exists():
        write_json(run / 'status.json', {'state': 'remote', 'steps': int(stages[-1].name[6:]), 'host': args.host})
    print(f'обучение запущено на {args.host} (код {commit[:7]}); первый запуск компилирует ядра Warp несколько минут')


def status(args):
    branch = f'~/sim/runs/humanoid_lab/{args.experiment}' if args.experiment else None
    command = 'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; '
    command += 'pgrep -af "humanoid_sim.cli train" | cut -c1-160; '
    if branch:
        command += (f'grep -E \'"(state|steps|steps_per_second|error)"\' {branch}/status.json; '
                    f'ls {branch} | grep stage_ | tail -3')
    ssh(args.host, command)


def stop(args):
    ssh(args.host, f'touch ~/sim/runs/humanoid_lab/{args.experiment}/stop.request')
    print('стоп запрошен: тренер сохранит этап и выйдет')


def fetch(args):
    run = LAB / args.experiment
    if not run.is_dir():
        raise SystemExit(f'нет локальной ветки {args.experiment}')
    branch = f'runs/humanoid_lab/{args.experiment}'
    listing = ssh(args.host, f'cd ~/sim/{branch} && ls', capture=True).decode().split()
    have = {path.name for path in list_stages(run)}
    stages = [name for name in listing if name.startswith('stage_') and name not in have
              and not name.startswith('.pending')]
    logs = [name for name in listing if name in ('status.json', 'training.log', 'report.json')
            or name.startswith('session_')]
    names = ' '.join(stages + logs)
    data = ssh(args.host, f'cd ~/sim/{branch} && tar -cf - {names}', capture=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(run, filter='data')
    print('получено:', ', '.join(stages) or 'новых этапов нет', '+', ', '.join(logs))
    cache = CACHE / f'warp-cache-{architecture(args.host)}.tar'
    if not cache.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(ssh(args.host, 'tar -cf - -C ~/.cache warp', capture=True))
        print('кеш ядер Warp сохранён:', cache.name, f'{cache.stat().st_size / 2**20:.0f} МБ')


def clean(args):
    targets = ' '.join(LICENSED)
    if args.all:  # the runs, the world model's data among them (states of a BONES-SEED clip's tracking)
        targets += f' {WM} runs'
    ssh(args.host, f'cd ~/sim && rm -rf {targets} && echo удалено: {targets}')


# --- the world model (SuperMimic stage 1): collect.py and fit.py on the server, no branch trained -------------------

WM = 'runs/humanoid_lab/world_model'
# The server's script: the main set, the deterministic one, the fit (resumed if started again), the fit's model on the
# deterministic set, the Warp cache. Each finished set is kept (its meta.json is written last), so a restart goes on.
WM_RUN = '''set -e
cd ~/sim
P=.venv-fullgpu/bin/python
W={wm}/{tag}
S={stage}
echo "== старт $(date -Is), этап $S, код {commit}"
if [ ! -f $W/data/meta.json ]; then
  rm -rf $W/data
  $P -u scripts/lab/wm/collect.py $S --out $W/data --worlds {worlds} --steps {steps} --action-noise {noise}
fi
if [ ! -f $W/data-det/meta.json ]; then
  rm -rf $W/data-det
  $P -u scripts/lab/wm/collect.py $S --out $W/data-det --deterministic --worlds {det_worlds} --steps {steps} --seed 1
fi
R=
if [ -f $W/fit/train_state.pt ]; then R=--resume; fi
$P -u scripts/lab/wm/fit.py $W/data --out $W/fit --minutes {minutes} $R
mkdir -p $W/fit-det
cp $W/fit/model.pt $W/fit-det/model.pt
$P -u scripts/lab/wm/fit.py $W/data-det --out $W/fit-det --evaluate-only --holdout 0.5
tar -cf $W/warp-cache.tar -C ~/.cache warp
echo "== DONE $(date -Is)"
'''


def wm_stage(spec):
    """(stage folder, its path relative to the repository, its task_config, tag <experiment>-<M steps>) of a stage
    folder or <experiment>:<steps>, named exactly, whose task has a world model adapter with a CUDA collector
    (world_model_adapters.adapter_for: getting up by one clip, walking) and whose task_config it knows."""
    path = Path(spec)
    path = path if path.is_absolute() else ROOT / path
    if (path / 'metadata.json').is_file():
        stage = path.resolve()
    else:
        name, _, steps = spec.partition(':')
        if not steps:
            raise SystemExit('wm-up: нужен этап явно: папка этапа или exp-…:<шаги>')
        stage = LAB / name / f'stage_{int(steps):012d}'
    metadata = read_json(stage / 'metadata.json')
    if metadata is None:
        raise SystemExit(f'нет этапа {spec}')
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.world_model_adapters import adapter_for
    config = stage_task_config(stage)
    try:
        adapter = adapter_for(metadata.get('robot_id'), metadata.get('task_id'), config)
    except ValueError as error:
        raise SystemExit(str(error))
    if 'cuda' not in adapter.backends:
        raise SystemExit(f'wm-up: у адаптера {adapter.name} нет сборщика на cuda')
    try:
        adapter.check('cuda')
    except ValueError as error:
        raise SystemExit(str(error))
    if adapter.licence:
        print(adapter.licence)
    try:
        relative = stage.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f'этап должен лежать в лаборатории: {stage}')
    return stage, relative, config, f'{stage.parent.name}-{int(stage.name[6:]) // 1_000_000}'


def wm_current(host):
    """(tag, stage, commit) wm-up named on the server."""
    lines = ssh(host, f'cat ~/sim/{WM}/current', capture=True).decode().split()
    if len(lines) != 3:
        raise SystemExit('на сервере нет этапа модели мира: сначала wm-up')
    return lines


def wm_up(args):
    stage, relative, config, tag = wm_stage(args.experiment)
    send_code(args.host, args.cuda)
    paths = [relative] + data_files(config)
    ssh(args.host, 'cd ~/sim && tar -xf -', data=pack(paths))
    commit = subprocess.run([str(GIT), 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    ssh(args.host, f'mkdir -p ~/sim/{WM}/{tag} && cat > ~/sim/{WM}/current',
        data=f'{tag}\n{relative}\n{commit[:12]}\n'.encode())
    print('отправлено:', ', '.join(paths))
    print(f'этап модели мира {tag} на {args.host} (код {commit[:7]}); дальше wm-start')


def wm_start(args):
    tag, stage, commit = wm_current(args.host)
    if not args.noise > 0:
        raise SystemExit('--noise должен быть положительным')
    script = WM_RUN.format(wm=WM, tag=tag, stage=stage, commit=commit, worlds=int(args.worlds),
                           steps=int(args.steps or 1000), noise=args.noise, det_worlds=int(args.det_worlds),
                           minutes=args.minutes)
    folder = f'{WM}/{tag}'
    # [w]: the pattern does not match the remote shell running this pgrep
    running = ssh(args.host, 'pgrep -f "bash [^ ]*[w]m_run.sh" || true', capture=True).decode().strip()
    if running:
        raise SystemExit(f'модель мира уже считается на {args.host} (pid {running}); wm-log покажет ход')
    ssh(args.host, f'mkdir -p ~/sim/{folder} && cat > ~/sim/{folder}/wm_run.sh', data=script.encode())
    ssh(args.host, f'cd ~/sim && (nohup bash {folder}/wm_run.sh >> {folder}/wm.log 2>&1 < /dev/null &)')
    print(f'запущено на {args.host}: {folder}/wm.log; сбор 4096 миров и сборка ядер Warp — первые 10–20 мин')


def wm_log(args):
    tag, _, _ = wm_current(args.host)
    ssh(args.host, f'tail -n {int(args.lines)} ~/sim/{WM}/{tag}/wm.log; '
                   'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; '
                   'pgrep -af "[w]m_run.sh|scripts/lab/[w]m/" | cut -c1-160 || true')


def wm_fetch(args):
    tag, _, _ = wm_current(args.host)
    folder = f'~/sim/{WM}/{tag}'
    names = ('wm.log', 'fit', 'fit-det', 'data/meta.json', 'data-det/meta.json')
    data = ssh(args.host, f'cd {folder} && tar -cf - $(ls -d {" ".join(names)} 2>/dev/null)', capture=True)
    out = LAB / 'world_model' / tag
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        got = sorted({member.name.split('/')[0] for member in tar.getmembers()})
        tar.extractall(out, filter='data')
    print('получено в', out.relative_to(ROOT), ':', ', '.join(got) or 'ничего')
    cache = CACHE / f'warp-cache-{architecture(args.host)}.tar'
    if not cache.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(ssh(args.host, f'cat {folder}/warp-cache.tar 2>/dev/null || tar -cf - -C ~/.cache warp',
                              capture=True))
        print('кеш ядер Warp сохранён:', cache.name, f'{cache.stat().st_size / 2**20:.0f} МБ')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('up', 'status', 'stop', 'fetch', 'clean', 'wm-up', 'wm-start', 'wm-log',
                                           'wm-fetch'))
    parser.add_argument('experiment', nargs='?', help='ветка exp-…; для wm-up — папка этапа или exp-…:<шаги>')
    parser.add_argument('--host', default='gpu1')
    parser.add_argument('--steps', type=float, help='up: шагов обучения; wm-start: шагов управления на мир (1000)')
    parser.add_argument('--cuda', default='cu130', choices=('cu130', 'cu128'))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--noise', type=float, default=.1, help='wm-start: шум действий основного набора')
    parser.add_argument('--minutes', type=float, default=60., help='wm-start: минут обучения модели мира')
    parser.add_argument('--worlds', type=int, default=4096, help='wm-start: миров основного набора')
    parser.add_argument('--det-worlds', type=int, default=512, help='wm-start: миров детерминированного набора')
    parser.add_argument('--lines', type=int, default=25, help='wm-log: строк журнала')
    args = parser.parse_args()
    if args.action in ('up', 'stop', 'fetch', 'wm-up') and not args.experiment:
        raise SystemExit(f'{args.action}: нужна ветка exp-…' if args.action != 'wm-up'
                         else 'wm-up: нужен этап: папка этапа или exp-…:<шаги>')
    sys.stdout.reconfigure(encoding='utf-8')
    {'up': up, 'status': status, 'stop': stop, 'fetch': fetch, 'clean': clean, 'wm-up': wm_up, 'wm-start': wm_start,
     'wm-log': wm_log, 'wm-fetch': wm_fetch}[args.action](args)


if __name__ == '__main__':
    main()
