"""Off-tree bounded manifest-only page-cache advice; never changes file bodies."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import time

MAX_FILES = 32768
MAX_BYTES = 8 * 1024**3
MAX_MANIFEST = 8 * 1024**2
ROOT = '/opt/ipfs-supervisor'
MANIFEST_SHA = '4d13cfeb3ad2daf67a50c135babbf58c6a197a71f70376bc730051508ce8fb27'
ARCHIVE_SHA = 'a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b'
PREFIXES = ('source/', 'datasets/', 'kit/', 'toolchains/lean/', 'extensions/',
            'models/source384/', 'runtime-wheels/torch-cpu/')

def checked_rows(rows):
    if type(rows) is not list or not 1 <= len(rows) <= MAX_FILES:
        raise ValueError('bounded nonempty inventory required')
    seen = set(); total = 0
    for row in rows:
        if type(row) is not dict or set(row) != {'path','bytes','mode','sha256'}:
            raise ValueError('closed inventory row required')
        name = row['path']
        if (type(name) is not str or not name or len(name.encode()) > 4096
                or not name.startswith(PREFIXES) or '\\' in name or '\x00' in name
                or any(part in ('', '.', '..') for part in name.split('/'))
                or PurePosixPath(name).is_absolute() or name in seen):
            raise ValueError('unsafe or duplicate inventory path')
        if (type(row['bytes']) is not int or not 0 <= row['bytes'] <= MAX_BYTES
                or type(row['mode']) is not int or row['mode'] not in (0o644,0o755)
                or type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):
            raise ValueError('invalid expected file metadata')
        total += row['bytes'];seen.add(name)
        if total > MAX_BYTES:raise ValueError('inventory byte bound exceeded')
    return total

def root_fd(root):
    root = Path(root)
    if not root.is_absolute() or str(root.resolve(strict=True)) != str(root):
        raise ValueError('canonical absolute runtime root required')
    fd = os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        return fd
    except BaseException:
        os.close(fd);raise

def open_file(root, name):
    parent = os.dup(root)
    try:
        parts = name.split('/')
        for part in parts[:-1]:
            child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            os.close(parent);parent=child
        return os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
    finally:
        os.close(parent)

def identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_nlink,info.st_uid,info.st_gid,
            info.st_size,info.st_atime_ns,info.st_mtime_ns,info.st_ctime_ns)

def require_file(info, row):
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_size != row['bytes'] or stat.S_IMODE(info.st_mode) != row['mode']):
        raise ValueError('manifest metadata or regular-file identity differs')

def advise(root, rows):
    total = checked_rows(rows);started=time.monotonic();fd=root_fd(root)
    original={};advised_files=advised_bytes=0;errors=[]
    try:
        # Check the entire population before issuing the first hint.
        for row in rows:
            child=open_file(fd,row['path'])
            try:
                info=os.fstat(child);require_file(info,row)
                original[row['path']]=identity(info)
            finally:os.close(child)
        for row in rows:
            child=open_file(fd,row['path'])
            try:
                info=os.fstat(child);require_file(info,row)
                if identity(info)!=original[row['path']]:raise ValueError('file changed after preflight')
                try:
                    os.fdatasync(child)
                    os.posix_fadvise(child,0,0,os.POSIX_FADV_DONTNEED)
                    advised_files+=1;advised_bytes+=row['bytes']
                except OSError as exc:
                    # Advice is best effort; a failed syscall is never counted as successful.
                    errors.append(dict(path=row['path'],error_type=type(exc).__name__,errno=exc.errno))
                    if len(errors)>32:raise ValueError('too many cache advice errors')
                if identity(os.fstat(child))!=original[row['path']]:raise ValueError('file metadata changed during advice')
            finally:os.close(child)
        # Detect path replacement after a descriptor was closed as well.
        for row in rows:
            child=open_file(fd,row['path'])
            try:
                if identity(os.fstat(child))!=original[row['path']]:raise ValueError('file path changed after advice')
            finally:os.close(child)
    finally:os.close(fd)
    return dict(schema='manifest-cache-advice@1',selected_files=len(rows),selected_bytes=total,
        advised_files=advised_files,advised_bytes=advised_bytes,errors=errors,
        metadata_unchanged=True,body_reads=0,body_writes=0,advice_is_best_effort=True,
        freed_bytes_claimed=False,global_drop_caches=False,cgroup_writes=False,
        seconds=time.monotonic()-started)

def manifest_bytes(path):
    path=Path(path);parent=root_fd(path.parent)
    try:fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
    finally:os.close(parent)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=MAX_MANIFEST:
            raise ValueError('bounded regular manifest required')
        raw=stream.read(MAX_MANIFEST+1)
        after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('manifest changed during read')
    return raw

def main():
    def expired(*args):raise TimeoutError('bounded cache diagnostic expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    if len(sys.argv)!=2:raise ValueError('one pinned manifest path required')
    path=Path(sys.argv[1])
    if str(path) != ROOT+'/cache-diagnostic-manifest.json':
        raise ValueError('fixed diagnostic manifest path required')
    raw=manifest_bytes(path)
    if len(raw)>MAX_MANIFEST or hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:
        raise ValueError('exact archive manifest required')
    manifest=json.loads(raw)
    if manifest['archive_sha256']!=ARCHIVE_SHA:raise ValueError('archive selection differs')
    try: result=advise(ROOT,manifest['files'])
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    result.update(manifest_sha256=MANIFEST_SHA,archive_sha256=ARCHIVE_SHA)
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
