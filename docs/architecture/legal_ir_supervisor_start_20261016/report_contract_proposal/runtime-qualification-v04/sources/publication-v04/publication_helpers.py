"""Exact unchanged frozen V06 publication reader/Git/GitHub helpers.

Copied source only. Runtime mutations occur solely from the finite owner entry.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
MAXIMUM = 8 * 1024 * 1024
ACCELERATOR = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
PR_ENDPOINT = 'repos/endomorphosis/ipfs_accelerate_py/pulls/292'


def need(value, reason):
    if not value:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def canonical(path):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical unaliased absolute path required')
    return path

def read(path, maximum=MAXIMUM):
    path = canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= maximum, 'bounded owned regular source required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                          s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
    need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
         'source or engineering observation changed during read')
    return raw

def safe_relative(spelling):
    need(type(spelling) is str and re.fullmatch(r'[A-Za-z0-9_./-]+', spelling), 'bounded source relative path required')
    path = PurePosixPath(spelling)
    need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == spelling,
         'canonical source relative path required')
    return Path(spelling)

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def write_new(path, raw, mode=0o644):
    path = canonical(path)
    need(path.parent.is_dir() and type(raw) is bytes and len(raw) <= MAXIMUM, 'existing charged publication parent required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)

def environment():
    result = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    result.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    return result

def git(*args, raw=False):
    result = subprocess.run(['/usr/bin/git', '-c', 'core.hooksPath=/dev/null', '-c', 'credential.helper=',
        '-c', 'credential.https://github.com.helper=!/usr/bin/gh auth git-credential', *args],
        cwd=ACCELERATOR, env=environment(), stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    need(result.returncode == 0 and len(result.stdout) <= 16 * MAXIMUM,
         'bounded owned Git operation failed: ' + ' '.join(args[:2]) + ' (exit ' + str(result.returncode) + ')')
    return result.stdout if raw else result.stdout.decode().strip()

def gh(*args):
    result = subprocess.run(['/usr/bin/gh', *args], cwd=ACCELERATOR, env=environment(),
        stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    need(result.returncode == 0 and len(result.stdout) <= 1024 * 1024, 'bounded official GitHub API operation failed')
    return result.stdout

def read_pr():
    result = strict_json(gh('api', PR_ENDPOINT))
    need(type(result) is dict and result.get('number') == 292 and result.get('state') == 'open'
         and result.get('merged') is False and result.get('draft') is True
         and result.get('head', {}).get('ref') == BRANCH
         and result.get('head', {}).get('repo', {}).get('full_name') == 'endomorphosis/ipfs_accelerate_py'
         and result.get('base', {}).get('repo', {}).get('full_name') == 'endomorphosis/ipfs_accelerate_py'
         and type(result.get('body')) is str and len(result['body'].encode()) <= 65536,
         'exact open draft owned PR292 required')
    return result

def verify_bundle(root, digest):
    root = canonical(root)
    manifest_raw = read(root / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) and sha(manifest_raw) == digest,
         'frozen source manifest differs')
    manifest = strict_json(manifest_raw)
    need(type(manifest) is dict and type(manifest.get('files')) is list and 1 <= len(manifest['files']) <= 64,
         'finite frozen source bundle required')
    files = {}
    for row in manifest['files']:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed frozen source row required')
        spelling = row['path']
        if type(spelling) is str and spelling.startswith('/'):
            absolute = canonical(spelling)
            need(root in absolute.parents, 'absolute source row must remain within its selected frozen bundle')
            relative = absolute.relative_to(root)
        else:
            relative = safe_relative(spelling)
        need(relative.name != 'SOURCE_MANIFEST.json' and relative.as_posix() not in files, 'unique frozen source row required')
        raw = read(root / relative)
        need(type(row['bytes']) is int and len(raw) == row['bytes'] and sha(raw) == row['sha256'],
             'frozen source bundle bytes differ')
        files[relative.as_posix()] = raw
    need(sum(len(raw) for raw in files.values()) <= 4 * MAXIMUM, 'bounded frozen source bundle required')
    files['SOURCE_MANIFEST.json'] = manifest_raw
    return files
