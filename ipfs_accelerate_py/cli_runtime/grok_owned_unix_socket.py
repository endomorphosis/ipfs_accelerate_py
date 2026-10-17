"""Exact source-bound Unix listener consumer outside a strict owned attempt.

The existing attempt sampler is unchanged. One private sibling beneath an
already admitted named root is precharged through the existing reservation and
physically sampled here. No arbitrary socket allowance or root lease exists.
Authority remains in the existing genuine P API; this class grants none.
P producer and strict inventory sources remain unchanged.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import socket
import stat
import sys
import threading

P_RESOURCE_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer')


class OwnedUnixSocketError(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise OwnedUnixSocketError(reason)


def signature(row):
    return (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid, row.st_nlink)


def directory(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'owned_socket_directory_alias')
    for parent in (path, *path.parents):
        need(stat.S_ISDIR(parent.lstat().st_mode), 'owned_socket_directory_alias')
    return path


def _sample_created_listener(listener, endpoint, parent_identity, path_identity, fd_identity, maximum_bytes):
    """Physical sampler only; it grants no native/storage authority."""
    need(type(listener) is socket.socket and listener.family == socket.AF_UNIX and
         listener.type == socket.SOCK_STREAM and listener.getsockname() == str(endpoint),
         'owned_socket_actual_listener_required')
    parent = directory(endpoint.parent)
    metadata = parent.lstat()
    need(signature(metadata) == parent_identity and stat.S_IMODE(metadata.st_mode) == 0o700 and
         metadata.st_uid == os.getuid(), 'owned_socket_private_parent_changed')
    need(signature(os.fstat(listener.fileno())) == fd_identity, 'owned_socket_descriptor_changed')
    entries = list(os.scandir(parent))
    need(len(entries) == 1 and entries[0].name == endpoint.name, 'owned_socket_only_exact_entry_allowed')
    row = entries[0].stat(follow_symlinks=False)
    need(signature(row) == path_identity and stat.S_ISSOCK(row.st_mode) and
         stat.S_IMODE(row.st_mode) == 0o600 and row.st_uid == os.getuid() and row.st_nlink == 1,
         'owned_socket_path_identity_changed')
    need(row.st_size <= maximum_bytes, 'owned_socket_physical_byte_bound')
    need(signature(parent.lstat()) == parent_identity and signature(endpoint.lstat()) == path_identity,
         'owned_socket_path_changed_during_sample')
    return {'scope': 'one_exact_owned_listener_directory', 'entry_count': 1,
            'apparent_bytes': row.st_size, 'exact_owned_socket_count': 1,
            'maximum_precharged_bytes': maximum_bytes, 'filesystem_quota': False}


class OwnedUnixListener:
    """Actual owner/P phase only; never serializable or a sampler callback."""
    def __init__(self, *, disk, parent, maximum_bytes=65536):
        module = sys.modules.get('ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources')
        base = P_RESOURCE_ROOT
        need(module is not None and module.__file__ == str(base/'parented_phase_resources.py') and
             module.__spec__.origin == module.__file__ and type(disk) is module.ParentedPhaseResourceReservation,
             'owned_socket_actual_parented_reservation_required')
        need(type(maximum_bytes) is int and 1 <= maximum_bytes <= 1024**2,
             'owned_socket_bounded_precharge_required')
        need(disk.native_lease.owner_pid == os.getpid(), 'owned_socket_existing_owner_PID_required')
        parent = Path(parent)
        need(parent.is_absolute() and not os.path.lexists(parent), 'owned_socket_fresh_parent_required')
        directory(parent.parent)
        need(any(root in parent.parents for root in disk.roots) and
             not (parent == disk.directory or disk.directory in parent.parents or parent in disk.directory.parents),
             'owned_socket_named_root_sibling_required')
        endpoint = parent/'s'
        need(len(os.fsencode(endpoint)) < 108, 'owned_socket_existing_root_path_too_long')
        self.disk, self.parent, self.endpoint = disk, parent, endpoint
        self.maximum_bytes = maximum_bytes
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._host = sys.modules.get('ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_daemon_resources')
        need(self._host is not None and self._host.__file__ == str(base/'autoencoder_daemon_resources.py') and
             self._host.__spec__.origin == self._host.__file__, 'owned_socket_actual_host_module_required')
        self._birth = self._host._process(self._pid)['birth']
        self.closed = False
        self.listener = None
        self.charge_key = 'exact-unix-'+hashlib.sha256(os.fsencode(endpoint)).hexdigest()
        disk.account_external_bytes(self.charge_key, maximum_bytes)
        # No host/storage root is added and no estimate is silently raised.
        parent.mkdir(mode=0o700)
        self._parent_identity = signature(parent.lstat())
        try:
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.listener.bind(str(endpoint)); endpoint.chmod(0o600)
            self.listener.listen(1); self.listener.setblocking(False)
            self._path_identity = signature(endpoint.lstat())
            self._fd_identity = signature(os.fstat(self.listener.fileno()))
            self.check_usage()
        except BaseException:
            # Construction has no worker. Only an identity-confirmed own
            # endpoint is removable; the precharge remains in the real claim.
            if self.listener is not None:
                self.listener.close()
            if hasattr(self, '_path_identity') and os.path.lexists(endpoint) and signature(endpoint.lstat()) == self._path_identity:
                endpoint.unlink()
            if not list(parent.iterdir()):
                parent.rmdir()
            raise

    def __reduce__(self):
        raise TypeError('owned Unix listener cannot be serialized')

    def _owner(self):
        need(not self.closed and os.getpid() == self._pid and threading.get_ident() == self._thread and
             self._host._process(self._pid)['birth'] == self._birth, 'owned_socket_owner_lifetime_changed')
        self.disk._verify_owner_roots()

    def witness(self):
        self._owner()
        row = self.endpoint.lstat()
        need(signature(row) == self._path_identity, 'owned_socket_path_identity_changed')
        return {'device': row.st_dev, 'inode': row.st_ino, 'mode': row.st_mode,
                'uid': row.st_uid, 'gid': row.st_gid, 'nlink': row.st_nlink}

    def check_usage(self):
        self._owner()
        need(self.disk.native_lease.owner_pid == self._pid, 'owned_socket_native_owner_changed')
        record = self.disk.to_dict()['record']
        need(record.get('status') == 'active' and record.get('external_charges', {}).get(self.charge_key) == self.maximum_bytes,
             'owned_socket_existing_precharge_changed')
        self.disk.check_usage(self.disk.directory)
        return _sample_created_listener(self.listener, self.endpoint, self._parent_identity,
            self._path_identity, self._fd_identity, self.maximum_bytes)

    def close_after_group_drain(self):
        self._owner()
        need(not self.disk._native_consumers() and not self._host._group_usage(self.disk._child)['live_processes'],
             'owned_socket_actual_native_consumers_or_group_live')
        _sample_created_listener(self.listener, self.endpoint, self._parent_identity,
            self._path_identity, self._fd_identity, self.maximum_bytes)
        self.endpoint.unlink()
        self.listener.close()
        need(not os.path.lexists(self.endpoint) and not list(self.parent.iterdir()) and
             signature(self.parent.lstat()) == self._parent_identity, 'owned_socket_close_not_observed')
        self.parent.rmdir()
        descriptor = os.open(self.parent.parent, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.closed = True


def open_owned_unix_listener(*, disk, parent, maximum_bytes=65536):
    return OwnedUnixListener(disk=disk, parent=parent, maximum_bytes=maximum_bytes)
