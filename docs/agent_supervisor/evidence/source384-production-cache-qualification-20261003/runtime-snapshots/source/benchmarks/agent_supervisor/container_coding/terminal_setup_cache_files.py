"""Bounded archive page-cache advice; never changes file bodies."""
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
PREFIXES = ('source/', 'datasets/', 'kit/', 'toolchains/lean/', 'extensions/',
            'models/source384/', 'runtime-wheels/torch-cpu/')

class PathRefusal(OSError):
    def __init__(self, details):
        self.details=details
        super().__init__(json.dumps(details,sort_keys=True))

def stat_metadata(info):
    kind=('symlink' if stat.S_ISLNK(info.st_mode) else 'directory' if stat.S_ISDIR(info.st_mode)
          else 'regular' if stat.S_ISREG(info.st_mode) else 'other')
    return dict(kind=kind,mode=info.st_mode,device=info.st_dev,inode=info.st_ino,
                links=info.st_nlink,uid=info.st_uid,gid=info.st_gid,bytes=info.st_size)

def descriptor_metadata(fd):
    value=stat_metadata(os.fstat(fd))
    try:value['descriptor_target']=os.readlink('/proc/self/fd/'+str(fd))[:4096]
    except OSError as exc:value['descriptor_target_error']=type(exc).__name__
    return value

def path_refusal(exc, root, parent, part, name, position):
    try:entry=stat_metadata(os.stat(part,dir_fd=parent,follow_symlinks=False))
    except OSError as nested:entry=dict(error_type=type(nested).__name__,errno=nested.errno)
    return PathRefusal(dict(schema='manifest-cache-path-refusal@1',error_type=type(exc).__name__,
        errno=exc.errno,path=name,component=part,component_position=position,
        entry_nofollow=entry,parent_descriptor=descriptor_metadata(parent),
        root_descriptor=descriptor_metadata(root),body_read=False))

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
        for position,part in enumerate(parts[:-1]):
            try:child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            except OSError as exc:raise path_refusal(exc,root,parent,part,name,position) from exc
            os.close(parent);parent=child
        try:return os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        except OSError as exc:raise path_refusal(exc,root,parent,parts[-1],name,len(parts)-1) from exc
    finally:
        os.close(parent)

def identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_nlink,info.st_uid,info.st_gid,
            info.st_size,info.st_atime_ns,info.st_mtime_ns,info.st_ctime_ns)

def require_file(info, row):
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != 0 or info.st_gid != 0
            or info.st_size != row['bytes'] or stat.S_IMODE(info.st_mode) != row['mode']):
        raise ValueError('manifest metadata or regular-file identity differs')

def advise(root, rows):
    total = checked_rows(rows);started=time.monotonic();fd=root_fd(root)
    original={};advised_files=advised_bytes=0;errors=[]
    try:
        # Check the entire population before issuing the first hint.
        for index,row in enumerate(rows):
            try:child=open_file(fd,row['path'])
            except PathRefusal as exc:
                raise PathRefusal({**exc.details,'row_index':index,'manifest_row':row,
                    'phase':'complete_inventory_preflight','advice_issued':False}) from exc
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
                except TimeoutError:
                    raise
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
    if os.geteuid() != 0:
        raise ValueError('root permission required for selected archive advice')
    if len(sys.argv) != 4 or sys.argv[1] != ROOT + '/setup-cache-manifest.json':
        raise ValueError('fixed manifest path and independent manifest/archive digests required')
    expected_manifest, expected_archive = sys.argv[2:]
    if not all(re.fullmatch('[0-9a-f]{64}', value) for value in (expected_manifest, expected_archive)):
        raise ValueError('canonical SHA256 pins required')
    def expired(*args):
        raise TimeoutError('bounded archive advice expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        raw = manifest_bytes(Path(sys.argv[1]))
        if len(raw) > MAX_MANIFEST or hashlib.sha256(raw).hexdigest() != expected_manifest:
            raise ValueError('exact archive manifest required')
        manifest = json.loads(raw)
        if manifest['archive_sha256'] != expected_archive:
            raise ValueError('archive selection differs')
        if manifest.get('setup_cache', {}).get('policy') != 'source384-native-aarch64-dontneed@1':
            raise ValueError('explicit supported setup cache policy required')
        result = advise(ROOT, manifest['files'])
        result.update(manifest_sha256=expected_manifest, archive_sha256=expected_archive)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

if __name__ == '__main__':
    main()
