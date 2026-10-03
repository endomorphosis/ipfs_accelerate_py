"""Advice for exactly four freshly hashed public native Codex binaries."""
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from cache_candidate import root_fd, identity

ROOT='/opt/ipfs-supervisor'
SOURCE_UID=1000
TARGET_UID=0
RECEIPT_UID=0
MAX_FILE=512*1024**2
MAX_TOTAL=2*1024**3
PINS={
 'codex':'0059c73b149a1433b634e26ad8a715e02717a9920665a9cc5676941db03c45cd',
 'codex-code-mode-host':'68237b34d0bc182e99c43ca2197a25120339c154dd3fc29f908c2a1b022efa5b',
}
VENDOR_PATTERN='versions/node/*/lib/node_modules/@openai/codex/node_modules/@openai/codex-*/vendor/*/bin/codex'

def require_receipt(value):
    if (type(value) is not dict or set(value)!={'schema','codex_version','files','executable_checks','provider_calls'}
            or value['schema']!='native-codex-runtime-bundle@1' or value['codex_version']!='0.158.0'
            or type(value['provider_calls']) is not int or value['provider_calls']!=0
            or type(value['files']) is not list or len(value['files'])!=2):
        raise ValueError('exact post-boundary native Codex receipt required')
    found=set()
    for row in value['files']:
        if (type(row) is not dict or set(row)!={'name','source_sha256','sha256','uid','mode'}
                or row['name'] not in PINS or row['name'] in found
                or row['sha256']!=PINS[row['name']] or row['source_sha256']!=PINS[row['name']]
                or type(row['uid']) is not int or row['uid']!=0
                or type(row['mode']) is not int or row['mode']!=0o755):
            raise ValueError('native bundle rows differ from independent pins')
        found.add(row['name'])
    checks=value['executable_checks']
    if type(checks) is not dict or set(checks)!=set(PINS):raise ValueError('both native checks required')
    for row in checks.values():
        if (type(row) is not dict or set(row)!={'returncode','stdout_sha256'}
                or type(row['returncode']) is not int or row['returncode']!=0
                or type(row['stdout_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',row['stdout_sha256'])):
            raise ValueError('successful bounded native executable checks required')

def selection(root):
    # This is the exact unique-vendor selector used by native_codex_exposure_script.
    matches=list(islice((root/'home/.nvm').glob(VENDOR_PATTERN),2))
    if len(matches)!=1:raise ValueError('one exact installed Codex vendor bin required')
    vendor=matches[0].parent
    prefix=vendor.relative_to(root).as_posix()
    if (not prefix.startswith('home/.nvm/versions/node/') or any(p in ('','..','.') for p in prefix.split('/'))):
        raise ValueError('vendor location outside approved native package')
    return [dict(path=prefix+'/'+name,name=name,role='vendor',uid=SOURCE_UID) for name in sorted(PINS)]+[
        dict(path='provider-bin/'+name,name=name,role='exposed',uid=TARGET_UID) for name in sorted(PINS)]

def open_binary(root, name):
    if not hasattr(os,'O_NOATIME'):raise RuntimeError('O_NOATIME required; no fallback')
    parent=os.dup(root)
    try:
        for part in name.split('/')[:-1]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            os.close(parent);parent=child
        return os.open(name.split('/')[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_NOATIME,dir_fd=parent)
    finally:os.close(parent)

def require_binary(info,row):
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or not 0<info.st_size<=MAX_FILE
            or stat.S_IMODE(info.st_mode)!=0o755 or info.st_uid!=row['uid']):
        raise ValueError('bounded protected singleton native binary required')

def advise(root, receipt):
    require_receipt(receipt);started=time.monotonic();root=Path(root);owner=root_fd(root)
    descriptors=[];records=[];total=read_bytes=0;seen=set()
    try:
        rows=selection(root)
        if len(rows)!=4:raise ValueError('exact four binary paths required')
        # Open and verify the complete population before hashing or advice.
        for row in rows:
            fd=open_binary(owner,row['path']);descriptors.append(fd)
            info=os.fstat(fd);require_binary(info,row)
            key=(info.st_dev,info.st_ino)
            if key in seen:raise ValueError('four distinct native copies required')
            seen.add(key);total+=info.st_size
            if total>MAX_TOTAL:raise ValueError('native selected byte bound exceeded')
            records.append({**row,'bytes':info.st_size,'sha256':PINS[row['name']],
                'identity':identity(info)})
        for fd,row in zip(descriptors,records):
            digest=hashlib.sha256();count=0
            while True:
                raw=os.read(fd,min(1024**2,row['bytes']-count+1))
                if not raw:break
                count+=len(raw);read_bytes+=len(raw)
                if count>row['bytes']:raise ValueError('native body grew during hash')
                digest.update(raw)
            if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                raise ValueError('native body differs from independent version pin')
            if identity(os.fstat(fd))!=row['identity']:raise ValueError('native metadata changed during hash')
        # All bodies must be accepted before issuing even one hint.
        for row in records:
            probe=open_binary(owner,row['path'])
            try:
                if identity(os.fstat(probe))!=row['identity']:raise ValueError('native path changed after hash')
            finally:os.close(probe)
        advised=[];errors=[]
        for fd,row in zip(descriptors,records):
            if identity(os.fstat(fd))!=row['identity']:raise ValueError('native file changed before advice')
            try:
                os.fdatasync(fd);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                advised.append(row)
            except OSError as exc:errors.append(dict(path=row['path'],error_type=type(exc).__name__,errno=exc.errno))
            if identity(os.fstat(fd))!=row['identity']:raise ValueError('native metadata changed during advice')
        for row in records:
            probe=open_binary(owner,row['path'])
            try:
                if identity(os.fstat(probe))!=row['identity']:raise ValueError('native path changed after advice')
            finally:os.close(probe)
    finally:
        for fd in descriptors:os.close(fd)
        os.close(owner)
    return dict(schema='pinned-native-codex-cache-advice@1',codex_version='0.158.0',
        files=[{k:v for k,v in row.items() if k!='identity'} for row in records],
        selected_files=4,selected_bytes=total,hashed_files=4,body_read_bytes=read_bytes,
        body_reads_after_advice=0,body_writes=0,advised_files=len(advised),
        advised_bytes=sum(row['bytes'] for row in advised),errors=errors,
        metadata_unchanged=True,atime_preserved_with_noatime=True,
        advice_is_best_effort=True,freed_bytes_claimed=False,global_drop_caches=False,
        credential_contents_read=False,seconds=time.monotonic()-started)

def read_receipt(root, expected_sha256):
    parent=root_fd(root)
    try:fd=open_binary(parent,'codex-cache-exposure.json')
    finally:os.close(parent)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=RECEIPT_UID
                or stat.S_IMODE(info.st_mode)!=0o644 or not 0<info.st_size<=32768):
            raise ValueError('protected singleton bounded public receipt required')
        raw=stream.read(32769)
        if identity(os.fstat(stream.fileno()))!=identity(info):raise ValueError('receipt changed during read')
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('post-boundary receipt changed')
    return json.loads(raw)

def main():
    if os.geteuid()!=0:raise ValueError('root permission required for exact noatime advice')
    if len(sys.argv)!=3 or sys.argv[1]!=ROOT+'/codex-cache-exposure.json':
        raise ValueError('fixed receipt path and independent digest required')
    def expired(*args):raise TimeoutError('bounded native cache advice expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        receipt=read_receipt(ROOT,sys.argv[2])
        result=advise(ROOT,receipt);result['post_boundary_receipt_sha256']=sys.argv[2]
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
