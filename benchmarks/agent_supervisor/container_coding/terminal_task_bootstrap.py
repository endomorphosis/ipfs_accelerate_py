"""Create an explicit bounded Git baseline in a disposable /app task container.

Only declared public input bytes enter the baseline. This is setup bookkeeping,
not a task solution, correctness check or proof of the instruction's meaning.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex


def bootstrap_script(profile: dict) -> str:
    from .terminal_task_profile import validate_task_profile
    selected = validate_task_profile(profile)
    return "profile = " + repr(selected) + "\n" + r'''
import hashlib, json, os, pathlib, stat, subprocess
root = pathlib.Path('/app')
if root.resolve(strict=True) != root or not root.is_dir():
    raise ValueError('canonical existing /app workspace required')
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), '-c', 'safe.directory=/app',
        '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', *args], stderr=subprocess.PIPE)
expected = set(profile['input_paths'])
observed = {}
total = 0
for base, dirs, files in os.walk(root, followlinks=False):
    parent = pathlib.Path(base)
    for name in list(dirs):
        path = parent/name
        if path.is_symlink():
            raise ValueError('linked task directories are unsupported')
        if parent == root and name == '.git':
            dirs.remove(name)
    for name in files:
        path = parent/name
        rel = path.relative_to(root).as_posix()
        if rel not in expected or path.is_symlink():
            raise ValueError('task workspace contains undeclared public input')
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 2*1024*1024:
            raise ValueError('bounded independent public input required')
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('task source changed while capturing baseline')
        total += len(raw)
        if total > 16*1024*1024 or len(observed) >= 252:
            raise ValueError('public input population exceeds bound')
        observed[rel] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),
            'mode':stat.S_IMODE(after.st_mode)}
if set(observed) != expected:
    raise ValueError('declared public inputs are missing')
for output in profile['outputs']:
    if output['effect'] == 'create' and (root/output['path']).exists():
        raise ValueError('declared output already exists')
created = not (root/'.git').exists()
if created:
    git('init', '-q')
    if expected:
        git('add', '--', *sorted(expected))
    git('-c', 'user.name=Isolated benchmark', '-c', 'user.email=benchmark@example.invalid',
        'commit', '--allow-empty', '-qm', 'Record exact original public task input bytes')
else:
    if not (root/'.git').is_dir() or (root/'.git').is_symlink():
        raise ValueError('independent canonical Git directory required')
    if git('rev-parse', '--show-toplevel').decode().strip() != str(root):
        raise ValueError('task is not its exact Git root')
    names = {name.decode() for name in git('ls-files', '-z').split(b'\0') if name}
    if names != expected or git('ls-files', '--others', '--exclude-standard', '-z'):
        raise ValueError('Git source inventory differs from declared public inputs')
for name, info in observed.items():
    path = root/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256'] or stat.S_IMODE(path.stat().st_mode) != info['mode']:
        raise ValueError('baseline creation changed original task bytes or modes')
print(json.dumps({'schema':'terminal-public-task-bootstrap@1', 'git_initialized':created,
    'source_files':observed, 'original_source_bytes_preserved':True,
    'head':git('rev-parse', 'HEAD').decode().strip(), 'provider_calls':0}))
'''


async def bootstrap_task_repository(environment, *, profile: dict, output: Path) -> dict:
    """Install harness prerequisites, then capture the task's explicit inputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    command = ('apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y '
               '--no-install-recommends git python3 python3-pip python3-venv')
    result = await environment.exec(command=command, cwd='/app', user='root', timeout_sec=600)
    (output/'prerequisites.log').write_text((result.stdout or '')+(result.stderr or ''))
    if result.return_code:
        raise RuntimeError('generic task harness prerequisites failed')
    result = await environment.exec(command=shlex.join(['python3', '-I', '-c', bootstrap_script(profile)]),
                                    cwd='/app', user='root', timeout_sec=60)
    if result.return_code:
        (output/'bootstrap-error.log').write_text((result.stdout or '')+(result.stderr or ''))
        raise RuntimeError('generic task baseline failed; see retained diagnostic')
    if len(result.stdout.encode()) > 131072:
        raise ValueError('public baseline receipt exceeds bound')
    receipt = json.loads(result.stdout)
    (output/'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    return receipt
