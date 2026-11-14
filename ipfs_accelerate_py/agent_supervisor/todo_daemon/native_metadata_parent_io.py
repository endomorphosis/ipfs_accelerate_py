"""Exact original datasets IO inside an authenticated metadata parent operation.

This is an audit admission helper, never a resource grant.  Source authentication
runs after the ordinary before-body guard and before an operation starts.  The
original active-parent inspector and original secret-bearing nested acquisition
still own storage binding and compute admission.  No disk-ledger write, root
allocation, unregistered heartbeat thread, arbitrary callback, or socket is
admitted here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
import types


_D = Path('/home/barberb/lift_coding/external/ipfs_datasets')
_PREFIX = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
_RELATIVE = Path('ipfs_datasets_py/optimizers/logic_theorem_optimizer')
_SOURCE_PINS = {
    'autoencoder_parented_resource_inspection': (19896, '0118bb08213eb537e417daf9bd8ab1a8f34df4c2c3a7c4be7b58396e9a6c9bf4'),
    'autoencoder_resource_recovery': (50610, 'e132d4db0cd5cfce0c8f9fb0e047decd45c7370e98d4a77a0a2635014e5bb84e'),
    'autoencoder_daemon_resources': (41756, '4c351874ac6725bfdaf43b9e16b1ceba70d283044e9d59a4588c35f83f1142c0'),
    'existing_resource_scheduler': (24518, 'd654ab5ed2fec314ea5ac07a662784ab20843b0b2feb56f13989ce1504607776'),
    'resource_scheduler': (144291, '2c00683202b4a783b5050329caa24a411f83788010baba86d9380123bae7d083'),
}
_AUTHENTICATED = None
PARENT_CAPACITY_IO_ADMISSION_FIELDS = frozenset({
    'schema', 'event', 'operation', 'paths', 'source', 'original_frame',
    'disk_reservation_id', 'native_parent_lease_id',
})


def _need(condition, message):
    if not condition:
        raise RuntimeError('native parent operational IO: ' + message)


def _receiver():
    module = sys.modules.get(__package__ + '.native_metadata_parent_resources')
    expected = str(Path(__file__).with_name('native_metadata_parent_resources.py'))
    if (type(module) is not types.ModuleType
            or module.__dict__.get('__file__') != expected
            or getattr(module.__dict__.get('__spec__'), 'origin', None) != expected):
        return None
    return module


def _plain_path(value):
    """Lexical exactness only: this helper performs no audited file opening."""
    try:
        text = os.fsdecode(os.fspath(value))
    except (TypeError, ValueError):
        return None
    if (not text or len(text) > 4096 or '\0' in text or not os.path.isabs(text)
            or text != os.path.normpath(text)):
        return None
    return text


def _codes(function):
    """Include contextmanager bodies and their literal comprehension code."""
    result = set()
    while isinstance(function, types.FunctionType):
        pending = [function.__code__]
        while pending:
            code = pending.pop()
            result.add(code)
            pending.extend(value for value in code.co_consts
                           if isinstance(value, types.CodeType))
        function = getattr(function, '__wrapped__', None)
    return frozenset(result)


def authenticate_parent_capacity_io_sources():
    """Capture exact already-imported original sources; no broker/ledger IO.

    Call after the ordinary raw-source guard before each bounded operation.
    Fixed source bytes and module identities are rechecked; the existing facade
    also retains its original producer/policy checks.  No caller-selected source
    root or digest, or sustained parent datasets lease, is assumed.
    This function intentionally runs outside the audit operation to avoid
    recursive source reads from inside an audit event.
    """
    global _AUTHENTICATED
    receiver = _receiver()
    _need(receiver is not None and getattr(receiver._OPERATION, 'value', None) is None,
          'source authentication must precede an operation')
    _need(_AUTHENTICATED is None or (_AUTHENTICATED[0] == os.getpid()
          and _AUTHENTICATED[2] is receiver), 'authenticated receiver object changed')
    modules = {}
    for name, (size, digest) in _SOURCE_PINS.items():
        path = _D / _RELATIVE / (name + '.py')
        module = sys.modules.get(_PREFIX + name)
        _need(module is not None and type(module) is types.ModuleType
              and module.__dict__.get('__file__') == str(path)
              and getattr(module.__dict__.get('__spec__'), 'origin', None) == str(path),
              'ordinary original source origin differs')
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_uid == os.geteuid() and info.st_size == size
              and not any(parent.is_symlink() for parent in (path, *path.parents)),
              'original source identity differs')
        raw = path.read_bytes()
        _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
              'original source bytes differ')
        modules[name] = (module, {'path': str(path), 'bytes': size, 'sha256': digest})
    _AUTHENTICATED = (os.getpid(), modules, receiver)
    return {name: dict(value[1]) for name, value in modules.items()}


def _function(modules, name, qualified):
    module, _ = modules[name]
    value = module
    for part in qualified.split('.'):
        value = getattr(value, part)
    return value


def _original_frame(frames, modules, name, qualified):
    module, pin = modules[name]
    # The ordinary before-body source loader owns byte authenticity.  These
    # checks additionally prevent a same-name private alias from borrowing IO.
    if (sys.modules.get(_PREFIX + name) is not module
            or module.__dict__.get('__file__') != pin['path']):
        return None
    allowed = _codes(_function(modules, name, qualified))
    for frame in frames:
        if (frame.f_globals is module.__dict__ and frame.f_code in allowed
                and frame.f_code.co_filename == pin['path']):
            return frame
    return None


def _receiver_frame(frames, receiver, envelope, name):
    function = getattr(receiver._ParentEnvelope, name)
    for frame in frames:
        if (frame.f_globals is receiver.__dict__ and frame.f_code is function.__code__
                and frame.f_code.co_filename == receiver.__file__
                and frame.f_locals.get('self') is envelope):
            return frame
    return None


def _failed_original_acquire_cleanup(frames, modules, caller, value, deadline):
    """Only original retirement of this unreturned nested admission may write."""
    if (not deadline.cleanup_only
            or deadline.native_deadline_monotonic > value['sealed_native_deadline_monotonic']
            or time.monotonic() >= deadline.native_deadline_monotonic):
        return False
    native = modules['resource_scheduler'][0]
    acquisition = _original_frame(frames, modules, 'resource_scheduler', 'GlobalResourceScheduler.acquire')
    retirement = _original_frame(frames, modules, 'resource_scheduler',
                                  'GlobalResourceScheduler._retire_failed_scoped_admission')
    if acquisition is None or retirement is None:
        return False
    scheduler = caller.f_locals.get('scheduler')
    token = acquisition.f_locals.get('parent_lease')
    sealed = value['native_lease_token']
    if (type(scheduler) is not modules['existing_resource_scheduler'][0]._PinnedGlobalResourceScheduler
            or acquisition.f_locals.get('self') is not scheduler
            or retirement.f_locals.get('self') is not scheduler
            or str(scheduler.state_path) != sealed['state_path']
            or type(token) is not native.ResourceLeaseToken
            or (token.lease_id, token.lease_key, token.state_path)
               != (sealed['lease_id'], sealed['lease_key'], sealed['state_path'])
            or acquisition.f_locals.get('request_id') != caller.f_locals.get('request_id')
            or retirement.f_locals.get('deadline') is not acquisition.f_locals.get('operation_deadline')
            or retirement.f_locals.get('waiter_id') != acquisition.f_locals.get('waiter_id')
            or retirement.f_locals.get('granted_record') is not acquisition.f_locals.get('granted_record')):
        return False
    granted = retirement.f_locals.get('granted_record')
    return (granted is None or (type(granted) is dict
        and granted.get('parent_lease_id') == sealed['lease_id']
        and granted.get('owner_pid') == os.getpid()
        and granted.get('request_id') == caller.f_locals.get('request_id')))


def _context(context, frames):
    receiver = _receiver()
    if (receiver is None or type(context) is not tuple or len(context) != 2
            or getattr(receiver._OPERATION, 'value', None) is not context):
        return None
    envelope, kind = context
    if (type(envelope) is not receiver._ParentEnvelope or envelope.pid != os.getpid()
            or kind not in ('inspect', 'allocate', 'renew', 'release')):
        return None
    if (_AUTHENTICATED is None or _AUTHENTICATED[0] != os.getpid()
            or _AUTHENTICATED[2] is not receiver
            or set(_AUTHENTICATED[1]) != set(_SOURCE_PINS)):
        return None
    modules = _AUTHENTICATED[1]
    value = envelope._value
    state = _plain_path(value['native_lease_token']['state_path'])
    ledger = _plain_path(value['disk_ledger_path'])
    if state is None or ledger is None or state == ledger:
        return None
    native = modules['resource_scheduler'][0]
    deadline = native.resource_operation_deadline()
    if deadline is None:
        return None
    if kind in ('renew', 'release'):
        caller = _receiver_frame(frames, receiver, envelope,
                                 '_heartbeat' if kind == 'renew' else 'release')
        grant = None if caller is None else caller.f_locals.get('grant')
        if (type(grant) is not receiver._DGrant or grant.envelope is not envelope
                or type(grant.lease) is not native.ResourceLease
                or envelope._grants.get(grant.lease.lease_id) is not grant
                or grant.lease._scheduler is not grant.scheduler
                or type(grant.scheduler) is not modules['existing_resource_scheduler'][0]._PinnedGlobalResourceScheduler
                or grant.lease.owner_pid != os.getpid()
                or grant.lease.parent_lease_id != value['native_lease_token']['lease_id']
                or str(grant.scheduler.state_path) != state):
            return None
        if kind == 'release':
            if (not deadline.cleanup_only
                    or deadline.native_deadline_monotonic > value['sealed_native_deadline_monotonic']
                    or time.monotonic() >= deadline.native_deadline_monotonic):
                return None
        else:
            loop = _original_frame(frames, modules, 'resource_scheduler', 'ResourceLease._heartbeat_loop')
            renew = _original_frame(frames, modules, 'resource_scheduler', 'GlobalResourceScheduler.renew')
            if (deadline.cleanup_only
                    or deadline.work_deadline_monotonic > value['sealed_work_deadline_monotonic']
                    or deadline.native_deadline_monotonic > value['sealed_native_deadline_monotonic']
                    or time.monotonic() >= min(deadline.work_deadline_monotonic, grant.deadline)
                    or grant.released or grant.heartbeat_thread is not threading.current_thread()
                    or grant.heartbeat_ident != threading.get_ident()
                    or grant.scheduler.config.auto_renew_leases is not False
                    or grant.lease._heartbeat_thread is not None
                    or loop is None or loop.f_locals.get('self') is not grant.lease
                    or renew is None or renew.f_locals.get('self') is not grant.scheduler
                    or renew.f_locals.get('lease_id') != grant.lease.lease_id
                    or renew.f_locals.get('lease_key') != grant.lease.lease_key):
                return None
    else:
        name = 'inspect' if kind == 'inspect' else 'acquire'
        caller = _receiver_frame(frames, receiver, envelope, name)
        if kind == 'inspect' and caller is None:
            caller = _receiver_frame(frames, receiver, envelope, 'check')
        if caller is None:
            return None
        if kind == 'allocate' and deadline.cleanup_only:
            if not _failed_original_acquire_cleanup(frames, modules, caller, value, deadline):
                return None
        elif (deadline.cleanup_only
                or deadline.work_deadline_monotonic > value['sealed_work_deadline_monotonic']
                or deadline.native_deadline_monotonic > value['sealed_native_deadline_monotonic']
                or time.monotonic() >= deadline.work_deadline_monotonic):
            return None
        if kind == 'allocate' and getattr(envelope, '_authenticated', False) is not True:
            return None
    return receiver, envelope, kind, value, modules, state, ledger


def _read_only(mode, flags):
    return (type(flags) is int
            and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            and not (type(mode) is str and any(c in mode for c in 'wax+')))


def _temporary(path, state, frame):
    if path is None or frame is None:
        return False
    candidate = frame.f_locals.get('temporary')
    expected = Path(state)
    return (_plain_path(candidate) == path and Path(path).parent == expected.parent
            and re.fullmatch(re.escape('.' + expected.name + '.' + str(os.getpid()) + '.')
                             + r'[0-9a-f]{32}\.tmp', Path(path).name) is not None)


def _regular_if_present(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid()


def authorize_parent_capacity_io(event, args, context):
    """Return a closed, credential-free admission record or None.

    The caller must retain its ordinary deny path on None.  In particular a
    forged context, source alias, disk-ledger write, arbitrary broker temp,
    dirfd operation, or unregistered heartbeat receives no exception.
    """
    if event not in ('open', 'os.rename', 'os.remove'):
        return None
    frames = []
    frame = sys._getframe(1)
    while frame is not None and len(frames) < 96:
        frames.append(frame)
        frame = frame.f_back
    selected = _context(context, frames)
    if selected is None:
        return None
    receiver, envelope, kind, value, modules, state, ledger = selected
    locked = _original_frame(frames, modules, 'existing_resource_scheduler',
                             '_PinnedGlobalResourceScheduler._locked_state')
    persist = _original_frame(frames, modules, 'existing_resource_scheduler',
                              '_PinnedGlobalResourceScheduler._persist_locked_state')
    reader = _original_frame(frames, modules, 'existing_resource_scheduler', '_read')
    module_name = 'existing_resource_scheduler'
    original = None
    paths = None
    if event == 'open' and len(args) == 3:
        path, mode, flags = args
        path = _plain_path(path)
        if path is None:
            return None
        if _read_only(mode, flags):
            if path == state and reader is not None and _plain_path(reader.f_locals.get('path')) == state:
                original, paths = reader, [path]
            elif path == state + '.lock' and locked is not None:
                scheduler = locked.f_locals.get('self')
                if str(scheduler.state_path) == state and str(scheduler.lock_path) == path:
                    original, paths = locked, [path]
            elif path == str(Path(state).parent) and persist is not None:
                original, paths = persist, [path]
            elif kind == 'inspect':
                inspect_frame = _original_frame(frames, modules,
                    'autoencoder_parented_resource_inspection', 'inspect_active_parented_reservation')
                if inspect_frame is None or inspect_frame.f_locals.get('reservation_id') != value['disk_reservation_id']:
                    return None
                if path == ledger:
                    candidate = _original_frame(frames, modules, 'autoencoder_daemon_resources',
                                               'DaemonResourceReservation._read')
                    helper = None if candidate is None else candidate.f_locals.get('self')
                    if (helper is not None and str(helper.ledger_path) == ledger
                            and helper._require_existing_ledger is True):
                        original, paths, module_name = candidate, [path], 'autoencoder_daemon_resources'
                elif path == ledger + '.lock':
                    candidate = _original_frame(frames, modules, 'autoencoder_resource_recovery',
                                               '_validation_inspection_lock')
                    helper = None if candidate is None else candidate.f_locals.get('helper')
                    if helper is not None and str(helper.ledger_path) == ledger and str(helper.lock_path) == path:
                        original, paths, module_name = candidate, [path], 'autoencoder_resource_recovery'
                elif path in (value['request_pin']['path'], value['execution_request_pin']['path']):
                    candidate = _original_frame(frames, modules, 'autoencoder_parented_resource_inspection', '_read_pin')
                    if candidate is not None and candidate.f_locals.get('pin') in (value['request_pin'], value['execution_request_pin']):
                        original, paths, module_name = candidate, [path], 'autoencoder_parented_resource_inspection'
        elif kind in ('allocate', 'renew', 'release'):
            # All writers must be the original existing-policy facade, not the
            # root scheduler's allocator or the disk reservation writer.
            if path == state + '.lock' and locked is not None:
                scheduler = locked.f_locals.get('self')
                allowed = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
                if (flags & ~getattr(os, 'O_CLOEXEC', 0) == allowed
                        and str(scheduler.state_path) == state and str(scheduler.lock_path) == path):
                    original, paths = locked, [path]
            elif _temporary(path, state, persist):
                allowed = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if (flags & ~getattr(os, 'O_CLOEXEC', 0) == allowed
                        and _regular_if_present(path)):
                    original, paths = persist, [path]
    elif kind in ('allocate', 'renew', 'release') and persist is not None:
        if event == 'os.rename' and len(args) == 4:
            source, destination, source_fd, destination_fd = args
            source, destination = _plain_path(source), _plain_path(destination)
            if (source_fd in (None, -1) and destination_fd in (None, -1)
                    and destination == state and _temporary(source, state, persist)
                    and _regular_if_present(source)):
                original, paths = persist, [source, destination]
        elif event == 'os.remove' and len(args) == 2:
            path, directory_fd = args
            path = _plain_path(path)
            if directory_fd in (None, -1) and _temporary(path, state, persist) and _regular_if_present(path):
                original, paths = persist, [path]
    if original is None or paths is None:
        return None
    record = {'schema': 'native-parent-capacity-io-admission/v1', 'event': event,
        'operation': kind, 'paths': paths, 'source': dict(modules[module_name][1]),
        'original_frame': original.f_code.co_qualname,
        'disk_reservation_id': value['disk_reservation_id'],
        'native_parent_lease_id': value['native_lease_token']['lease_id']}
    _need(set(record) == PARENT_CAPACITY_IO_ADMISSION_FIELDS, 'closed audit record differs')
    return record


__all__ = ['PARENT_CAPACITY_IO_ADMISSION_FIELDS', 'authenticate_parent_capacity_io_sources',
           'authorize_parent_capacity_io']
