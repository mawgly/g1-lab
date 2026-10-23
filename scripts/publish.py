"""Publishes the lab's public face under the user's standing permission (23.10.2026: the site every evening and after
important events, GitHub once a week and a release for each new robot candidate, without asking, if the checks pass):

  publish.py site [--dry-run]       site_export -> scan -> upload to ihc:www/g1lab.ru (leftovers removed) -> check
  publish.py github [--dry-run]     the lab's HEAD (git archive) -> the clean copy ../g1-lab-public -> scan -> CPU
                                    tests there -> one commit (GitHub noreply author) -> push to mawgly/g1-lab
  publish.py release <export dir> --tag candidate-N --title "..." --notes <english notes.md> [--dry-run]
                                    github first, then a release with policy.onnx + deploy.yaml zipped and the
                                    safety warning around the notes (numbers in the notes come from the lab)

Any failed check stops before anything is published. The scan reads the personal tokens from
.tools/publish/forbidden.txt (untracked, never published; the script refuses to run without it), allows only the
e-mail addresses in EMAILS and refuses private keys and GitHub tokens. Every run appends to .tools/publish/log.jsonl."""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.tools' / 'publish'
FORBIDDEN = STATE / 'forbidden.txt'
LOG = STATE / 'log.jsonl'
DIST = ROOT / '.tools' / 'site' / 'dist'
ZIP = ROOT / '.tools' / 'site' / 'g1-lab-site.zip'
PUBLIC = ROOT.parent / 'g1-lab-public'
LAB_GIT = ROOT / '.tools' / 'git' / 'cmd' / 'git.exe'
# Git for Windows runs the `!gh auth git-credential` helper (MinGit has no shell for it).
FULL_GIT = Path(r'C:\Program Files\Git\cmd\git.exe')
GH = Path(r'C:\Program Files\GitHub CLI\gh.exe')
SSH = Path(r'C:\Windows\System32\OpenSSH\ssh.exe')
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
HOST, REMOTE_DIR = 'ihc', 'www/g1lab.ru'
SITE_URLS = ('https://www.g1lab.ru/', 'http://www.g1lab.ru/')
REPO = 'mawgly/g1-lab'
AUTHOR_NAME, AUTHOR_EMAIL = 'Pavel Maltsev', '150599366+mawgly@users.noreply.github.com'
EMAILS = {AUTHOR_EMAIL, 'noreply@anthropic.com', 'jac@d.qvel'}  # the last is code (jac @ d.qvel)
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}\b')
SECRETS = (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'), re.compile(r'\bgh[opsu]_[A-Za-z0-9]{30,}'),
           re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}'), re.compile(r'\bsk-[A-Za-z0-9_-]{30,}'))
TEXT_LIMIT = 20_000_000
WARNING = """> [!WARNING]
> **Not tested on a real robot.** This policy has run only in simulation: our MuJoCo setup and Unitree's own simulator \
(`unitree_mujoco` with Unitree's `g1_ctrl` deploy controller). It has never been run on a physical G1. A humanoid robot \
is heavy, and its motors can injure people and damage the robot. If you run it on hardware, you do so entirely at your \
own risk: use a hoist, keep people clear, keep the emergency stop in hand and follow Unitree's safety instructions. \
Provided "as is", without warranty of any kind (Apache License 2.0, see `LICENSE`).
"""
FOOTER = '\n\nThe lab journal and the comparisons: https://www.g1lab.ru\n\n' \
         'This is an independent project, not affiliated with or endorsed by Unitree Robotics.\n'


class Refused(Exception):
    pass


def run(command, cwd=None, check=True, **kwargs):
    result = subprocess.run([str(part) for part in command], cwd=cwd, capture_output=True, **kwargs)
    if check and result.returncode:
        error = result.stderr.decode('utf-8', 'replace') if isinstance(result.stderr, bytes) else result.stderr
        raise Refused(f'{Path(str(command[0])).name} {command[1] if len(command) > 1 else ""}: {error.strip()[-800:]}')
    return result


def text(result):
    return result.stdout.decode('utf-8', 'replace')


def log(action, ok, **details):
    STATE.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as file:
        file.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), action=action, ok=ok, **details),
                              ensure_ascii=False) + '\n')


def forbidden_tokens():
    if not FORBIDDEN.exists():
        raise Refused(f'нет {FORBIDDEN}: без списка личных данных публиковать нельзя')
    tokens = [line.strip().lower() for line in FORBIDDEN.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not tokens:
        raise Refused(f'{FORBIDDEN} пуст')
    return tokens


def masked(token):
    return token[:2] + '…' + hashlib.sha256(token.encode()).hexdigest()[:6]


def scan(files, base):
    """Hits as (relative path, what) for personal tokens, unknown e-mail addresses and secrets in the text files."""
    tokens, hits = forbidden_tokens(), []
    for path in files:
        try:
            data = path.read_bytes()
        except OSError as error:
            hits.append((str(path), f'не читается: {error}'))
            continue
        if len(data) > TEXT_LIMIT or b'\0' in data[:8192]:
            continue  # binary (clips, pictures, policies): no text to leak
        body = data.decode('utf-8', 'ignore')
        lower, name = body.lower(), path.relative_to(base).as_posix()
        hits += [(name, f'личный токен {masked(token)}') for token in tokens if token in lower]
        hits += [(name, f'адрес {masked(address.lower())}') for address in set(EMAIL.findall(body))
                 if address.lower() not in EMAILS]
        hits += [(name, f'секрет ({pattern.pattern[:16]}…)') for pattern in SECRETS if pattern.search(body)]
    return hits


def refuse_hits(hits, where):
    if hits:
        listed = '\n'.join(f'  {name}: {what}' for name, what in hits[:30])
        raise Refused(f'{where}: найдено {len(hits)} совпадений, публикация остановлена\n{listed}')


# --- site ---------------------------------------------------------------------------------------------------------

def site_files():
    return sorted(path for path in DIST.rglob('*') if path.is_file())


def tar_filter(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mode = 0o755 if info.isdir() else 0o644  # Windows reports 0o666: never world-writable on the server
    return info


def ssh(command, **kwargs):
    return run([SSH, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST, command], **kwargs)


def fetch(url):
    request = urllib.request.Request(url, headers={'Cache-Control': 'no-cache', 'User-Agent': 'g1lab-publish'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


def site(dry_run):
    run([PYTHON, '-m', 'humanoid_sim.site_export', '--out', DIST, '--zip', ZIP], cwd=ROOT,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    summary = json.loads((DIST / 'site-export.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary.get('missing'):
        raise Refused(f'выгрузка неполная: {summary.get("missing")}')
    files = site_files()
    refuse_hits(scan(files, DIST), 'сайт')
    size = sum(path.stat().st_size for path in files)
    if dry_run:
        return dict(files=len(files), mb=round(size / 1e6, 1), dry_run=True)
    upload = subprocess.Popen([str(SSH), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST,
                               f'tar xf - -C {REMOTE_DIR}'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=upload.stdin, mode='w|') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(DIST).as_posix(), filter=tar_filter)
    upload.stdin.close()
    if upload.wait():
        raise Refused(f'загрузка: {upload.stderr.read().decode("utf-8", "replace").strip()[-800:]}')
    local = {path.relative_to(DIST).as_posix() for path in files}
    remote = set(text(ssh(f'cd {REMOTE_DIR} && find . -type f -print0')).split('\0')) - {''}
    remote = {name[2:] if name.startswith('./') else name for name in remote}
    extra = sorted(name for name in remote - local if '..' not in name.split('/'))
    if extra:  # files of an older export that the new one no longer has; only inside the site folder
        ssh(f'cd {REMOTE_DIR} && xargs -0 rm -f -- && find . -mindepth 1 -type d -empty -delete',
            input=('\0'.join(extra) + '\0').encode())
    for url in SITE_URLS:
        try:
            status, body = fetch(url + 'site-export.json')
            served = json.loads(body).get('exported_at')
        except (OSError, ValueError):
            continue
        if status == 200 and served == summary['exported_at']:
            return dict(files=len(files), mb=round(size / 1e6, 1), removed=len(extra), checked=url)
    raise Refused('сайт залит, но ни по одному адресу не отдаёт новую site-export.json')


# --- github -------------------------------------------------------------------------------------------------------

def public_git(*args, **kwargs):
    return run([FULL_GIT if FULL_GIT.exists() else LAB_GIT, '-C', PUBLIC, *args], **kwargs)


def lab_git(*args, **kwargs):
    return run([LAB_GIT, '-C', ROOT, *args], **kwargs)


def read_state():
    path = STATE / 'github.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def github(dry_run):
    if not (PUBLIC / '.git').exists():
        raise Refused(f'нет чистой копии {PUBLIC}')
    if text(public_git('status', '--porcelain')).strip():
        raise Refused(f'в {PUBLIC} есть незакоммиченные изменения: разберите их вручную')
    if text(public_git('branch', '--show-current')).strip() != 'main':
        raise Refused(f'{PUBLIC} не на ветке main')
    if REPO not in text(public_git('remote', 'get-url', 'origin')):
        raise Refused(f'origin в {PUBLIC} — не {REPO}')
    head = text(lab_git('rev-parse', 'HEAD')).strip()
    state = read_state()
    since = state.get('lab_commit')  # the message first: a failure here leaves the copy untouched
    subjects = text(lab_git('log', '--format=%s', f'{since}..HEAD' if since else '-20')).splitlines()
    message = f'Lab update {time.strftime("%Y-%m-%d")}\n\n' + '\n'.join(f'- {line}' for line in subjects[:60])
    base = text(public_git('rev-parse', 'HEAD')).strip()
    tracked = text(public_git('ls-files', '-z')).split('\0')
    for name in filter(None, tracked):  # the copy mirrors the lab: drop what it had, then lay out the lab's HEAD
        path = PUBLIC / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    archive = lab_git('archive', '--format=tar', 'HEAD')
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(PUBLIC, filter='data')
    public_git('add', '-A')
    changed = text(public_git('diff', '--cached', '--name-only')).split()
    if not changed:
        public_git('reset', '-q', '--hard', 'HEAD')
        return dict(changed=0, lab=head[:7])
    try:
        files = [PUBLIC / name for name in filter(None, text(public_git('ls-files', '-z')).split('\0'))]
        refuse_hits(scan([path for path in files if path.is_file()], PUBLIC), 'GitHub')
        tests = run([PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd=PUBLIC, check=False,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'})
        tail = text(tests).strip().splitlines()[-1:] or ['']
        if tests.returncode:
            raise Refused(f'тесты на чистой копии не прошли: {tail[0]}')
        if dry_run:
            public_git('reset', '-q', '--hard', base)
            public_git('clean', '-fdq')
            return dict(changed=len(changed), lab=head[:7], tests=tail[0], dry_run=True)
        identity = dict(GIT_AUTHOR_NAME=AUTHOR_NAME, GIT_AUTHOR_EMAIL=AUTHOR_EMAIL, GIT_COMMITTER_NAME=AUTHOR_NAME,
                        GIT_COMMITTER_EMAIL=AUTHOR_EMAIL)
        public_git('commit', '-q', '-F', '-', input=message.encode(), env={**os.environ, **identity})
        helper = f'!"{GH.as_posix()}" auth git-credential'
        public_git('-c', 'credential.helper=', '-c', f'credential.helper={helper}', 'push', '-q', 'origin', 'main')
    except Refused:  # anything after the copy was laid out: back to the last pushed commit
        public_git('reset', '-q', '--hard', base)
        public_git('clean', '-fdq')
        raise
    commit = text(public_git('rev-parse', 'HEAD')).strip()
    (STATE / 'github.json').write_text(json.dumps(dict(lab_commit=head, public_commit=commit,
                                                       at=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1),
                                       encoding='utf-8')
    return dict(changed=len(changed), lab=head[:7], public=commit[:7], tests=tail[0])


# --- release ------------------------------------------------------------------------------------------------------

def release(export, tag, title, notes, dry_run):
    export, notes = Path(export), Path(notes)
    if not re.fullmatch(r'candidate-\d+', tag):
        raise Refused('тег — candidate-<номер>')
    for name in ('policy.onnx', 'deploy.yaml'):
        if not (export / name).is_file():
            raise Refused(f'в {export} нет {name}')
    body = notes.read_text(encoding='utf-8').strip()
    if re.search('[А-Яа-яЁё]', re.sub(r'«[^»]*»', '', body)):
        raise Refused('заметки к релизу — на английском (по-русски только названия разделов docs в «…»)')
    if run([GH, 'release', 'view', tag, '--repo', REPO], check=False).returncode == 0:
        raise Refused(f'релиз {tag} уже есть')
    folder = f'g1-{tag.replace("-", "")}-policy'
    files = {name: (export / name).read_bytes() for name in ('policy.onnx', 'deploy.yaml', 'stand_trim.json')
             if (export / name).is_file()}
    experiment = re.match(r'(exp-[0-9a-f]{16})-\d+$', export.name)
    config = ROOT / 'runs' / 'humanoid_lab' / experiment.group(1) / 'run_config.json' if experiment else None
    if config and config.is_file():
        files['run_config.json'] = config.read_bytes()
    files['MD5SUMS'] = ''.join(f'{hashlib.md5(files[name]).hexdigest()}  {name}\n'
                               for name in ('policy.onnx', 'deploy.yaml')).encode()
    notes_text = WARNING + '\n' + body + FOOTER
    files['README_POLICY.md'] = notes_text.encode('utf-8')
    out = STATE / 'releases'
    shutil.rmtree(out / folder, ignore_errors=True)
    for name, data in files.items():
        (out / folder).mkdir(parents=True, exist_ok=True)
        (out / folder / name).write_bytes(data)
    refuse_hits(scan(sorted((out / folder).iterdir()), out), 'релиз')
    archive = out / f'{folder}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in files:
            zipped.write(out / folder / name, f'{folder}/{name}')
    (out / f'{tag}.md').write_text(notes_text, encoding='utf-8')
    synced = github(dry_run)
    if dry_run:
        return dict(zip=archive.name, github=synced, dry_run=True)
    run([GH, 'release', 'create', tag, archive, '--repo', REPO, '--target', 'main', '--title', title,
         '--notes-file', out / f'{tag}.md'])
    return dict(tag=tag, zip=archive.name, github=synced, url=f'https://github.com/{REPO}/releases/tag/{tag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('site', 'github', 'release'))
    parser.add_argument('export', nargs='?')
    parser.add_argument('--tag')
    parser.add_argument('--title')
    parser.add_argument('--notes')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        if args.action == 'site':
            result = site(args.dry_run)
        elif args.action == 'github':
            result = github(args.dry_run)
        else:
            if not (args.export and args.tag and args.title and args.notes):
                raise Refused('release: нужны <export dir> --tag --title --notes')
            result = release(args.export, args.tag, args.title, args.notes, args.dry_run)
    except Refused as error:
        log(args.action, False, error=str(error)[:2000])
        print('ОТКАЗ:', error)
        return 1
    log(args.action, True, **result)
    print('ok', json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
