"""Explicit opt-in leases for the dispatcher's own new-dispatch-only queue hold.

Unknown/manual markers are never adopted, removed or inferred from age. Paths
are fixed; registration precedes the live hardlink, and expiry preserves exact
registered inodes in a private token-named archive without overwriting history.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

QUEUE_HOLD_LEASE_VERSION = 1
SCHEMA = 'ipfs_accelerate_py.agent_supervisor.queue-hold-lease@1'
MAX_SECONDS = 1800
_KEYS = {'schema','scope','token','created_at','expires_at','device','inode','body_sha256','mtime_ns'}


def _finite(value):
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except (OverflowError,TypeError,ValueError):
        return False


def _decode_metadata(raw):
    try:
        return json.loads(raw)
    except RecursionError as exc:
        raise ValueError('queue metadata JSON nesting exceeds decoder limit') from exc


def _paths(config, token=''):
    directory = Path(config.state_dir).absolute()
    archive = directory / 'expired-queue-holds'
    return dict(directory=directory, hold=directory/'hold', lease=directory/'hold.lease.json',
        lock=directory/'queue-hold-lease.lock', archive=archive,
        reserve=directory/('.queue-hold.'+token+'.reserve'),
        staging_lease=directory/('.queue-hold.'+token+'.lease'),
        archived_hold=archive/(token+'.hold'), archived_lease=archive/(token+'.lease.json'))


def _exists(path):
    return path.exists() or path.is_symlink()


def _read(path, maximum=16384, private=True):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_size > maximum or (private and before.st_mode & 0o077)):
            raise ValueError('owned bounded ordinary lease member required')
        raw = os.read(fd, maximum+1)
        after = os.fstat(fd)
        current = path.lstat()
        keys = ('st_dev','st_ino','st_size','st_mtime_ns')
        if len(raw)!=before.st_size or any(getattr(before,k)!=getattr(after,k) or getattr(before,k)!=getattr(current,k) for k in keys):
            raise ValueError('lease member changed during read')
        return raw, current
    finally:
        os.close(fd)


def _directory(path, create=False):
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('owned ordinary lease directory required')


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def _lock(config):
    p = _paths(config); _directory(p['directory'], create=True)
    fd = os.open(p['lock'], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_mode & 0o077:
            raise ValueError('owned private queue lease lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def review_queue_hold(config, now):
    p = _paths(config)
    result = dict(status='absent', authenticated_expired=False, hold_path=str(p['hold']))
    try:
        raw, info = _read(p['lease'], 4096)
    except FileNotFoundError:
        return result
    except (OSError, ValueError) as exc:
        return dict(result, status='invalid', reason=type(exc).__name__)
    try:
        _directory(p['directory'])
        if config.repair_lock_path is None: raise ValueError('canonical repair marker must be configured')
        lease = _decode_metadata(raw)
        if type(lease) is not dict or set(lease)!=_KEYS or lease['schema']!=SCHEMA or lease['scope']!='new_dispatch_only':
            raise ValueError('exact queue lease schema required')
        token = lease['token']
        if type(token) is not str or not re.fullmatch('[0-9a-f]{32}', token):
            raise ValueError('queue lease token invalid')
        created, expires = lease['created_at'], lease['expires_at']
        if (not all(_finite(v) for v in (created,expires,now))
                or not 0<created<expires<=created+MAX_SECONDS or now<created):
            raise ValueError('finite bounded unchanged queue lease required')
        if (any(type(lease[k]) is not int or lease[k]<0 for k in ('device','inode','mtime_ns'))
                or lease['inode']==0 or type(lease['body_sha256']) is not str
                or not re.fullmatch('[0-9a-f]{64}',lease['body_sha256'])):
            raise ValueError('queue hold custody invalid')
        p = _paths(config, token)
        if _exists(p['archive']): _directory(p['archive'])
        # Every hardlink must be one exact protocol companion, including crash windows.
        lease_members = [p[k] for k in ('lease','staging_lease','archived_lease') if _exists(p[k])]
        for member in lease_members:
            data, item = _read(member,4096)
            if (item.st_dev,item.st_ino,data)!=(info.st_dev,info.st_ino,raw) or item.st_nlink!=len(lease_members):
                raise ValueError('unregistered lease hardlink or archive conflict')
        hold_members = [p[k] for k in ('hold','reserve','archived_hold') if _exists(p[k])]
        if not hold_members:
            raise ValueError('registered hold inode unavailable')
        for member in hold_members:
            data, item = _read(member)
            body = _decode_metadata(data)
            if (item.st_dev,item.st_ino,item.st_mtime_ns)!=(lease['device'],lease['inode'],lease['mtime_ns']) or hashlib.sha256(data).hexdigest()!=lease['body_sha256']:
                raise ValueError('queue hold replaced or mutated')
            if item.st_nlink!=len(hold_members) or type(body) is not dict or body.get('schema')!=SCHEMA or body.get('token')!=token or body.get('scope')!='new_dispatch_only':
                raise ValueError('unregistered hold hardlink or token')
        return dict(result,status='expired' if now>=expires else 'active',authenticated_expired=now>=expires,
            token=token,created_at=created,expires_at=expires,lease_sha256=hashlib.sha256(raw).hexdigest(),
            lease_device=info.st_dev,lease_inode=info.st_ino,live_hold_present=_exists(p['hold']),
            reserve_present=_exists(p['reserve']),archived_hold_present=_exists(p['archived_hold']),
            device=lease['device'],inode=lease['inode'],body_sha256=lease['body_sha256'],mtime_ns=lease['mtime_ns'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(result,status='invalid',reason=type(exc).__name__)


def register_queue_hold(config, reason, ttl_seconds, *, clock=None):
    """Create the durable lease BEFORE linking a new live queue hold.

    This pauses only new dispatch; it does not require or disturb worker presence.
    The returned exact custody can be supplied to a maintenance publisher.
    """
    clock = clock or time.time
    capability_check = require_live_queue_hold_capability
    if type(reason) is not str or len(reason.encode())>4096:
        raise ValueError('bounded queue hold reason required')
    if not _finite(ttl_seconds) or not 0<ttl_seconds<=MAX_SECONDS:
        raise ValueError('queue hold lease must be bounded to1800seconds')
    if config.repair_lock_path is None:
        raise ValueError('canonical repair marker must be configured')
    created = clock()
    if not _finite(created) or created<=0 or not _finite(created+ttl_seconds) or not created<created+ttl_seconds<=created+MAX_SECONDS:
        raise ValueError('finite creation clock required')
    _confirm_capability(capability_check(config, created))
    with _lock(config):
        prior = review_queue_hold(config, created)
        if _released(prior): _retire_released_descriptor(config, prior)
        token = uuid.uuid4().hex; p = _paths(config,token)
        if any(_exists(p[k]) for k in ('hold','lease','archived_hold','archived_lease')):
            raise FileExistsError('existing queue hold or lease is preserved')
        raw = (json.dumps(dict(schema=SCHEMA,scope='new_dispatch_only',token=token,reason=reason),sort_keys=True)+'\n').encode()
        if len(raw)>16384: raise ValueError('bounded encoded queue hold body required')
        _directory(p['archive'],create=True)
        _write_new(p['reserve'],raw); info = p['reserve'].stat()
        lease = dict(schema=SCHEMA,scope='new_dispatch_only',token=token,created_at=created,expires_at=created+ttl_seconds,
            device=info.st_dev,inode=info.st_ino,body_sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=info.st_mtime_ns)
        data = (json.dumps(lease,sort_keys=True)+'\n').encode()
        _write_new(p['staging_lease'],data); os.link(p['staging_lease'],p['lease']); _sync(p['directory'])
        fresh = clock()
        if not _finite(fresh) or not created<=fresh<lease['expires_at']:
            raise ValueError('queue lease expired or clock changed before live link')
        _confirm_capability(capability_check(config, fresh))
        before_link = clock()
        if not _finite(before_link) or not created<=before_link<lease['expires_at']:
            raise ValueError('queue lease expired during live capability verification')
        os.link(p['reserve'],p['hold']); _sync(p['directory'])
        p['reserve'].unlink(); p['staging_lease'].unlink(); _sync(p['directory'])
        return dict(lease,path=str(p['hold']),lease_path=str(p['lease']),lease_sha256=hashlib.sha256(data).hexdigest(),
            archive_hold_path=str(p['archived_hold']),archive_lease_path=str(p['archived_lease']))


def expire_queue_hold(config, state, now, *, apply=False):
    """Retire only an authenticated expired queue lease under native repair ownership."""
    review = review_queue_hold(config,now)
    if not (review['authenticated_expired'] or _released(review)) or not apply:
        return review
    if 'active_worker' in state or 'spawn_intent' in state or config.repair_lock_path is None:
        return dict(review,retirement='waiting_for_quiescence')
    from .repair_dispatcher import repair_lock, AlreadyRunning, SCHEMA as DISPATCHER_SCHEMA
    try:
        with repair_lock(config.repair_lock_path), _lock(config):
            raw, info = _read(_paths(config)['directory']/'state.json',1024*1024,private=False)
            actual = _decode_metadata(raw)
            if (info.st_nlink!=1 or type(actual) is not dict or actual.get('schema')!=DISPATCHER_SCHEMA
                    or 'active_worker' in actual or 'spawn_intent' in actual):
                return dict(review,retirement='waiting_for_quiescence')
            current = review_queue_hold(config,now)
            if not (current['authenticated_expired'] or _released(current)) or current.get('token')!=review['token'] or current.get('lease_sha256')!=review['lease_sha256']:
                return dict(current,retirement='custody_changed')
            p = _paths(config,current['token']); _directory(p['archive'],create=True)
            source_hold = next(p[k] for k in ('hold','reserve','archived_hold') if _exists(p[k]))
            for source,target in ((source_hold,p['archived_hold']),
                                  (p['lease'],p['archived_lease'])):
                if not _exists(target):os.link(source,target)
                elif source.stat().st_ino!=target.stat().st_ino or source.stat().st_dev!=target.stat().st_dev:
                    raise ValueError('existing queue archive is preserved')
            _sync(p['archive'])
            # The lock excludes protocol writers; revalidate exact registered aliases before unlinking.
            checked = review_queue_hold(config,now)
            if not (checked['authenticated_expired'] or _released(checked)) or checked.get('token')!=current['token']:
                raise ValueError('queue lease changed during archival')
            for key in ('hold','reserve','staging_lease','lease'):
                if _exists(p[key]): _unlink_registered(p[key], current, key in ('lease','staging_lease'))
            _sync(p['directory'])
            return dict(current,retirement='released_descriptor_retired' if _released(current) else 'archived',live_hold_present=False,
                        archive_hold_path=str(p['archived_hold']),archive_lease_path=str(p['archived_lease']))
    except AlreadyRunning:
        return dict(review,retirement='waiting_for_repair_lock')
    except (OSError,ValueError,TypeError) as exc:
        return dict(review,retirement='refused',retirement_error=type(exc).__name__)


def _released(review):
    return (review['status'] in ('active','expired') and not review.get('live_hold_present')
            and review.get('archived_hold_present') and not review.get('reserve_present'))


def _retire_released_descriptor(config, review):
    """An already archived marker needs only metadata cleanup under the lease lock.

    No live hold or worker is changed; this also avoids blocking the next registration.
    """
    if not _released(review): raise ValueError('exact completed-release archive required')
    p = _paths(config,review['token'])
    if not _exists(p['archived_lease']): os.link(p['lease'],p['archived_lease'])
    checked = review_queue_hold(config,review['created_at'])
    if not _released(checked) or checked.get('token')!=review['token']:
        raise ValueError('completed queue release custody changed')
    _sync(p['archive'])
    for key in ('staging_lease','lease'):
        if _exists(p[key]): _unlink_registered(p[key], checked, True)
    _sync(p['directory'])


def _confirm_capability(value):
    if (type(value) is not dict or type(value.get('version')) is not int
            or value['version']!=QUEUE_HOLD_LEASE_VERSION or value.get('identity_verified') is not True):
        raise ValueError('live queue lease capability is not verified')


def require_live_queue_hold_capability(config, now):
    """Disk source support cannot stand in for this fresh, kernel-bound live status."""
    from .repair_dispatcher import probe_process, SCHEMA as DISPATCHER_SCHEMA
    from .watchdog_guardian import _python_entrypoint
    raw, info = _read(_paths(config)['directory']/'status.json',1024*1024,private=False)
    status = _decode_metadata(raw)
    if type(status) is not dict or status.get('schema')!=DISPATCHER_SCHEMA or info.st_nlink!=1:
        raise ValueError('canonical live dispatcher status required')
    version, pid, ticks, stamp = (status.get(k) for k in
        ('queue_hold_lease_version','control_pid','control_start_ticks','control_heartbeat_at'))
    if (type(version) is not int or version!=QUEUE_HOLD_LEASE_VERSION or type(pid) is not int or pid<=0
            or type(ticks) is not int or ticks<=0 or type(stamp) not in (int,float)
            or not _finite(stamp) or not _finite(now) or not 0<=now-stamp<=120):
        raise ValueError('fresh live queue lease capability required')
    before = probe_process(pid)
    directory = Path('/proc')/str(pid)
    if directory.stat().st_uid!=os.getuid(): raise ValueError('dispatcher UID differs')
    argv = before.get('argv',[])
    mode,entrypoint,index = _python_entrypoint(argv)
    if (mode!='module' or entrypoint!='ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher'
            or index<1 or argv[index]!='-m'
            or any(flag not in ('-B','-u','-E','-s','-S','-I','-O','-OO') for flag in argv[1:index])
            or not re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?',Path(argv[0]).name)
            or Path(argv[0]).resolve()!=(directory/'exe').resolve()):
        raise ValueError('actual canonical Python module entrypoint required')
    if argv.count('--apply')!=1 or '--once' in argv or status.get('apply') is not True:
        raise ValueError('persistent applying dispatcher capability required')
    def option(name):
        values = [argv[i+1] for i,item in enumerate(argv[:-1]) if item==name]
        values += [item.split('=',1)[1] for item in argv if item.startswith(name+'=')]
        if len(values)!=1: raise ValueError('exact dispatcher command binding required')
        return values[0]
    cwd = Path(before.get('cwd',''))
    def absolute(value):
        path = Path(value); return (path if path.is_absolute() else cwd/path).resolve()
    if (not before.get('alive') or before.get('start_ticks')!=ticks or cwd.resolve()!=Path(config.repo_root).resolve()
            or option('-m')!='ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher'
            or absolute(option('--repo-root'))!=Path(config.repo_root).resolve()
            or absolute(option('--manifest-path'))!=Path(config.manifest_path).resolve()
            or absolute(option('--state-dir'))!=Path(config.state_dir).resolve()
            or absolute(option('--repair-lock-path'))!=Path(config.repair_lock_path).resolve()):
        raise ValueError('live dispatcher process identity differs')
    after = probe_process(pid)
    if (not after.get('alive') or any(after.get(k)!=before.get(k) for k in ('pid','start_ticks','argv','cwd'))
            or directory.stat().st_uid!=os.getuid()):
        raise ValueError('live dispatcher birth changed during capability check')
    return dict(version=version,identity_verified=True,pid=pid,start_ticks=ticks)


def publisher_queue_hold_proof(config, registration, *, now=None):
    """Project exact new-lease custody to the unchanged publisher's seven fields."""
    review = review_queue_hold(config,time.time() if now is None else now)
    if review['status']!='active' or not review.get('live_hold_present') or review.get('token')!=registration.get('token'):
        raise ValueError('current unexpired registered queue hold required')
    root = Path(config.repo_root).resolve(); p = _paths(config,review['token'])
    return dict(schema='jevops.repair-queue-hold-lease.v1',path=str(p['hold'].relative_to(root)),
        device=review['device'],inode=review['inode'],sha256=review['body_sha256'],token=review['token'],
        release_archive=str(p['archived_hold'].relative_to(root)))


def _unlink_registered(path, review, descriptor=False):
    raw,info = _read(path,4096 if descriptor else 16384)
    expected = (review['lease_device'],review['lease_inode'],review['lease_sha256']) if descriptor else (review['device'],review['inode'],review['body_sha256'])
    if (info.st_dev,info.st_ino,hashlib.sha256(raw).hexdigest())!=expected:
        raise ValueError('registered queue member was replaced before retirement')
    if not descriptor and info.st_mtime_ns!=review['mtime_ns']:
        raise ValueError('registered queue hold timestamp changed before retirement')
    path.unlink()


def expired_queue_hold_matches(config, path, now, review):
    if not review['authenticated_expired'] or Path(path).absolute()!=_paths(config)['hold']:
        return False
    current = review_queue_hold(config,now)
    return (current['authenticated_expired'] and current.get('live_hold_present')
            and all(current.get(k)==review.get(k) for k in ('token','device','inode','body_sha256','lease_sha256')))
