"""Same-process existing-capability guard candidate, never an admission API.

This is not a complete server or adoption of the old separate-owner launch.
No project import, resource acquisition, token reconstruction or operation on
import. A new reviewed owner profile must supply already-live exact objects.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import types

P_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets')
CLAIM_FIELDS = ('database_task_cid', 'database_attempt_id', 'database_claim_id',
    'database_attempt_number', 'coordination_lease_id', 'owner_session_id', 'fencing_token', 'fence_epoch')
RESOURCE_FIELDS = ('root_lease_id', 'phase_lease_id', 'disk_native_lease_id', 'disk_reservation_id')


class ExistingOwnerRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise ExistingOwnerRefusal(reason)


def module(name, root, relative, pins):
    value = sys.modules.get(name)
    expected = root/relative
    need(value is not None and Path(value.__file__) == expected and
         value.__spec__.origin == str(expected) and expected.resolve(strict=True) == expected and
         str(expected) in pins, 'existing_owner_genuine_module_origin_required')
    return value


def source_digest(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path and not str(path).endswith(
        ('.state.json', '.safetensors', '.ckpt', '.pt', '.pth', '.duckdb', '.sqlite', '.db')),
        'existing_owner_regular_source_path_required')
    descriptor = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= 128*1024**2,
             'existing_owner_source_bound')
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        result, count = hashlib.sha256(), 0
        while True:
            body = os.read(descriptor, 128*1024)
            if not body:
                break
            count += len(body)
            need(count <= 128*1024**2, 'existing_owner_source_bound')
            result.update(body)
        need(count == before.st_size and identity(before) == identity(os.fstat(descriptor)) == identity(path.lstat()),
             'existing_owner_source_changed_during_read')
        return result.hexdigest()
    finally:
        os.close(descriptor)


def _require_source_profile_binding(source_pins, source_profile_pin, launch_binding):
    """Join immutable metadata only; this supplies no source/native authority."""
    need(type(source_pins) is dict and type(source_profile_pin) is dict and
         set(source_profile_pin) == {'path', 'sha256'} and
         source_pins.get(source_profile_pin['path']) == source_profile_pin['sha256'],
         'existing_owner_source_profile_pin_required')
    need(type(launch_binding) is dict and
         launch_binding.get('source_bindings_sha256') == source_profile_pin['sha256'],
         'existing_owner_source_profile_binding_mismatch')


class ExistingProductionOwner:
    """Guard actual objects in their owning PID/thread; cannot be serialized."""
    def __init__(self, *, core, database_owner, attempt, root, phase, disk, source_roots,
                 source_pins, source_locks, launch_binding, native_deadline_monotonic, source_profile_pin):
        need(type(source_roots) is tuple and len(source_roots) == 2 and source_roots[1] == P_ROOT,
             'existing_owner_exact_source_roots_required')
        accelerator = source_roots[0]
        need(type(accelerator) is type(Path()) and accelerator.is_absolute() and
             accelerator.resolve(strict=True) == accelerator, 'existing_owner_accelerator_root')
        need(type(source_pins) is dict and 1 <= len(source_pins) <= 10000,
             'existing_owner_complete_pinned_sources_required')
        need(type(core) is types.ModuleType and sys.modules.get(core.__name__) is core and
             getattr(core, '__spec__', None) is not None and
             core.__spec__.origin == core.__file__ and Path(core.__file__).resolve(strict=True) == Path(core.__file__) and
             source_pins.get(core.__file__) == source_digest(core.__file__),
             'existing_owner_genuine_client_core_required')
        self._pins = tuple(sorted(source_pins.items()))
        _require_source_profile_binding(source_pins, source_profile_pin, launch_binding)
        need(source_pins.get(str(Path(__file__).absolute())) == source_digest(Path(__file__).absolute()),
             'existing_owner_guard_source_pin_required')
        for path, digest in self._pins:
            need(type(path) is str and type(digest) is str and len(digest) == 64 and
                 all(c in '0123456789abcdef' for c in digest), 'existing_owner_source_pin_shape')
        sdk = module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler', P_ROOT,
                     'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py', source_pins)
        disk_module = module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources', P_ROOT,
                     'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py', source_pins)
        self._host = module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_daemon_resources', P_ROOT,
                     'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py', source_pins)
        daemon = module('ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon', accelerator,
                        'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py', source_pins)
        need(type(root) is sdk.ResourceLease and type(phase) is sdk.ResourceLease and
             type(disk) is disk_module.ParentedPhaseResourceReservation and
             type(database_owner) is daemon.DatabaseImplementationDaemon and type(attempt) is daemon.DatabaseTaskAttempt,
             'existing_owner_exact_capability_types_required')
        need(root.owner_pid == phase.owner_pid == os.getpid() and
             root.parent_lease_id is None and phase.parent_lease_id == root.lease_id and
             root._scheduler is phase._scheduler and disk._parent is phase and
             disk.native_lease._scheduler.state_path == root._scheduler.state_path and
             disk.native_lease.owner_pid == os.getpid() and disk.native_lease.parent_lease_id == phase.lease_id,
             'existing_owner_same_PID_actual_native_chain_required')
        need(type(launch_binding) is dict and launch_binding.get('schema') == core.CONTEXT_SCHEMA and
             set(launch_binding) == {'schema', *core.PRELAUNCH_FIELDS}, 'existing_owner_exact_launch_binding_required')
        need(type(native_deadline_monotonic) in (int, float) and math.isfinite(native_deadline_monotonic) and
             launch_binding['native_deadline_monotonic'] == native_deadline_monotonic and
             launch_binding['work_deadline_monotonic'] <= native_deadline_monotonic <= disk._deadline,
             'existing_owner_original_deadline_required')
        self._core = core
        self._binding_bytes = core.canonical(launch_binding)
        self._request_sha256 = core.digest(launch_binding)
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._birth = core.process(self._pid)['birth']
        self._owner, self._attempt = database_owner, attempt
        self._root, self._phase, self._disk = root, phase, disk
        need(type(source_locks) is tuple and len(source_locks) == len(source_roots) and
             {row.get('root') for row in source_locks if type(row) is dict} == set(map(str, source_roots)),
             'existing_owner_exact_source_read_lock_roots_required')
        for row in source_locks:
            need(type(row) is dict and set(row) == {'root', 'path', 'device', 'inode', 'kernel_key'} and
                 row['path'] == str(Path(row['root'])/'workspace/control-plane/source-leases/source.lock') and
                 type(row['device']) is int and type(row['inode']) is int and
                 row['kernel_key'] == f"{os.major(row['device']):02x}:{os.minor(row['device']):02x}:{row['inode']}",
                 'existing_owner_genuine_source_read_lock_identity_required')
        self._source_locks = tuple(core.canonical(row) for row in source_locks)
        self._child = None
        self._failed = False
        self.current()

    def __reduce__(self):
        raise TypeError('existing production owner cannot be serialized')

    def bind_child(self, child):
        need(self._child is None and type(child) is subprocess.Popen, 'existing_owner_single_actual_Popen_required')
        observed = self._core.process(child.pid)
        need(observed['parent_pid'] == self._pid and observed['group_pid'] == child.pid,
             'existing_owner_actual_direct_worker_group_required')
        self._disk.check_usage(self._disk.directory, child_pid=child.pid)
        self._child = observed
        return dict(observed)

    def current(self):
        need(not self._failed, 'existing_owner_failure_latched')
        try:
            need(os.getpid() == self._pid and threading.get_ident() == self._thread and
                 self._core.process(self._pid)['birth'] == self._birth, 'existing_owner_process_or_thread_changed')
            binding = self._core.decode(self._binding_bytes)
            need(time.monotonic() < min(binding['work_deadline_monotonic'], binding['native_deadline_monotonic']),
                 'existing_owner_original_deadline_expired')
            need(all(source_digest(path) == digest for path, digest in self._pins), 'existing_owner_producer_source_changed')
            need(time.monotonic() < min(binding['work_deadline_monotonic'], binding['native_deadline_monotonic']),
                 'existing_owner_original_deadline_expired')
            claim = self._owner._attempt_claim(self._attempt)
            self._owner._protect_attempt_claim(self._attempt, claim)
            observed_claim = dict(database_task_cid=str(claim.task_cid), database_attempt_id=str(claim.attempt_id),
                database_claim_id=str(claim.claim_id), database_attempt_number=int(claim.attempt_number),
                coordination_lease_id=str(claim.lease_id), owner_session_id=str(claim.owner_session_id),
                fencing_token=int(claim.fencing_token), fence_epoch=int(claim.fence_epoch))
            need(observed_claim == {key: binding[key] for key in CLAIM_FIELDS} and
                 binding['native_attempt_id'] == observed_claim['database_attempt_id'],
                 'existing_owner_actual_claim_fence_changed')
            need(time.monotonic() < min(binding['work_deadline_monotonic'], binding['native_deadline_monotonic']),
                 'existing_owner_original_deadline_expired')
            need(self._root.renew() and self._phase.renew() and self._disk.heartbeat(),
                 'existing_owner_native_ancestry_revoked')
            native = self._disk.native_lease
            resources = dict(root_lease_id=self._root.lease_id, phase_lease_id=self._phase.lease_id,
                disk_native_lease_id=native.lease_id, disk_reservation_id=self._disk.reservation_id)
            need(resources == {key: binding[key] for key in RESOURCE_FIELDS} and
                 self._root.owner_pid == self._phase.owner_pid == native.owner_pid == self._pid and
                 self._root._scheduler is self._phase._scheduler and self._disk._parent is self._phase and
                 native._scheduler.state_path == self._root._scheduler.state_path and
                 self._phase.parent_lease_id == self._root.lease_id and native.parent_lease_id == self._phase.lease_id,
                 'existing_owner_actual_native_chain_changed')
            usage = self._disk.check_usage(self._disk.directory)
            owner = self._host._process(self._pid)
            need(owner is not None and owner['rss_bytes']+usage['group_rss']['rss_bytes'] <= self._disk.memory_mb*1024**2,
                 'existing_owner_actual_aggregate_memory_exceeded')
            locks = [line.split() for line in Path('/proc/locks').read_text().splitlines()]
            for raw in self._source_locks:
                lock = self._core.decode(raw)
                metadata = Path(lock['path']).lstat()
                need((metadata.st_dev, metadata.st_ino) == (lock['device'], lock['inode']) and
                     any(len(row) >= 6 and row[1:4] == ['FLOCK', 'ADVISORY', 'READ'] and
                         row[4] == str(self._pid) and row[5] == lock['kernel_key'] for row in locks),
                     'existing_owner_actual_source_read_lock_lost')
            if self._child is not None:
                need(self._core.process(self._child['pid']) == self._child,
                     'existing_owner_actual_worker_identity_changed')
            need(time.monotonic() < min(binding['work_deadline_monotonic'], binding['native_deadline_monotonic']),
                 'existing_owner_original_deadline_expired')
            return {'claim_binding': observed_claim, 'resource_binding': resources,
                'source_bindings_sha256': binding['source_bindings_sha256'],
                'observed_monotonic': time.monotonic(), 'observation_is_not_native_authority': True}
        except BaseException:
            self._failed = True
            raise


def bind_existing_production_owner(*, core, database_owner, attempt, root, phase, disk, source_roots,
        source_pins, source_locks, launch_binding, native_deadline_monotonic, source_profile_pin):
    return ExistingProductionOwner(core=core, database_owner=database_owner, attempt=attempt,
        root=root, phase=phase, disk=disk, source_roots=source_roots, source_pins=source_pins,
        source_locks=source_locks, launch_binding=launch_binding,
        native_deadline_monotonic=native_deadline_monotonic, source_profile_pin=source_profile_pin)
