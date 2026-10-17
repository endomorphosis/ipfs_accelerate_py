"""Opt-in provider-stream binding for a NEW same-PID production owner profile.

Source candidate only. The profile must construct a genuine owner and server
at the Portal provider seam, after the final worktree exists. No resource,
claim, source lock, credential slot or session is acquired by this module.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

MODULE_NAME = 'ipfs_accelerate_py.cli_runtime.grok_production_handoff'
POLL_SECONDS = 0.05
OWNER_FLAGS = ('--owner-context', '--owner-request-sha256', '--owner-allocation-session-id')
CAPTURE_FIELDS = ('allocation_session_id', 'provider', 'native_session_id', 'workspace', 'resume_style')


class ProductionHandoffRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise ProductionHandoffRefusal(reason)


class ProductionProviderHandoff:
    """One actual stream, on the existing owner thread; never serializable.

    The caller supplies a server that already holds genuine same-PID root,
    phase, disk and database-attempt objects, plus an already initialized
    allocation store. Existing store reads may open a connection; they occur
    only after the complete worker process group has drained.
    """

    def __init__(self, *, server_module, server, allocation_store, capsule_pin,
                 source_bindings_pin, worker_path):
        need(type(server_module) is types.ModuleType and
             sys.modules.get(server_module.__name__) is server_module and
             server_module.__name__ == 'ipfs_accelerate_py.cli_runtime.grok_production_owner_server' and
             type(server) is server_module.ProductionOwnerServer,
             'handoff_exact_source_bound_server_required')
        self.server_module, self.server = server_module, server
        self.owner, self.core = server.owner, server.core
        self._pins = dict(self.owner._pins)
        factory = sys.modules.get(type(self.owner).__module__)
        need(type(factory) is types.ModuleType and hasattr(factory, 'source_digest'),
             'handoff_genuine_owner_factory_required')
        self.factory = factory
        daemon = sys.modules.get('ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon')
        allocation = sys.modules.get('ipfs_accelerate_py.llm_allocation.duckdb_store')
        need(type(daemon) is types.ModuleType and type(allocation) is types.ModuleType and
             type(self.owner._owner) is daemon.DatabaseImplementationDaemon and
             type(allocation_store) is allocation.AllocationStore and allocation_store._ready is True,
             'handoff_existing_database_owner_and_initialized_store_required')
        accelerator = Path(daemon.__file__).parents[3]
        self._daemon_module = daemon
        self._allocation_module = allocation
        self._cleanup_module = sys.modules.get('ipfs_accelerate_py.agent_supervisor.todo_daemon.core')
        self.allocation_store = allocation_store
        expected = {
            MODULE_NAME: accelerator/'ipfs_accelerate_py/cli_runtime/grok_production_handoff.py',
            server_module.__name__: accelerator/'ipfs_accelerate_py/cli_runtime/grok_production_owner_server.py',
            allocation.__name__: accelerator/'ipfs_accelerate_py/llm_allocation/duckdb_store.py',
            daemon.__name__: accelerator/'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
            'ipfs_accelerate_py.agent_supervisor.todo_daemon.core': accelerator/'ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py',
        }
        for name, path in expected.items():
            module = sys.modules.get(name)
            need(type(module) is types.ModuleType and module.__file__ == str(path) and
                 module.__spec__.origin == str(path) and
                 self._pins.get(str(path)) == factory.source_digest(path),
                 'handoff_normal_package_complete_source_pins_required')
        self._binding_bytes = bytes(self.owner._binding_bytes)
        need(type(allocation_store.path) is type(Path()) and
             str(allocation_store.path) == self.binding['allocation_store_path'] and
             allocation_store.path.resolve(strict=True) == allocation_store.path,
             'handoff_exact_existing_allocation_store_required')
        for pin, digest in ((capsule_pin, self.binding['capsule_sha256']),
                            (source_bindings_pin, self.binding['source_bindings_sha256'])):
            need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'} and
                 type(pin['bytes']) is int and pin['bytes'] > 0 and pin['sha256'] == digest and
                 Path(pin['path']).stat().st_size == pin['bytes'] and
                 factory.source_digest(pin['path']) == digest,
                 'handoff_exact_existing_capsule_and_profile_pins_required')
        self.capsule_pin, self.source_bindings_pin = dict(capsule_pin), dict(source_bindings_pin)
        need(capsule_pin['bytes'] <= 65_536, 'handoff_bounded_capsule_required')
        descriptor = os.open(capsule_pin['path'], os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            need(before.st_size == capsule_pin['bytes'], 'handoff_capsule_size_changed')
            capsule_body = os.read(descriptor, 65_536+1)
            after = os.fstat(descriptor)
            identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                    row.st_size, row.st_mtime_ns, row.st_ctime_ns)
            need(len(capsule_body) == capsule_pin['bytes'] and identity(before) == identity(after),
                 'handoff_capsule_changed_during_cleanup_budget_read')
        finally:
            os.close(descriptor)
        need(self.factory.source_digest(capsule_pin['path']) == capsule_pin['sha256'],
             'handoff_capsule_changed_during_cleanup_budget_read')
        def unique_pairs(items):
            result = {}
            for key, value in items:
                need(key not in result, 'handoff_capsule_duplicate_JSON_field')
                result[key] = value
            return result
        def bad_constant(_value):
            raise ProductionHandoffRefusal('handoff_capsule_nonfinite_JSON')
        # Capsule bound matches V4's existing 65,536-byte file contract.
        # RPC decode remains separately restricted to its original 8 KiB.
        capsule = json.loads(capsule_body.decode('utf-8'), object_pairs_hook=unique_pairs,
                             parse_constant=bad_constant)
        need(type(capsule) is dict and self.core.canonical(capsule) == capsule_body and
             capsule.get('schema') == 'legal-ir-grok-planning-transport-capsule/v4',
             'handoff_capsule_schema_or_canonical_encoding_differs')
        reserve = capsule['named_session']['cleanup_reserve_seconds']
        need(type(reserve) in (int, float) and math.isfinite(reserve) and 1 <= reserve <= 10,
             'handoff_original_capsule_cleanup_reserve_required')
        self._cleanup_seconds = float(reserve)
        self.worker_path = Path(worker_path)
        need(str(self.worker_path) in self._pins and
             self._pins[str(self.worker_path)] == self.binding['worker_source_sha256'] and
             factory.source_digest(self.worker_path) == self.binding['worker_source_sha256'],
             'handoff_exact_worker_source_required')
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._birth = self.core.process(self._pid)['birth']
        self._prepared = False
        self._process = None
        self._child_identity = None
        self._completed = False
        self._closed = False
        self._capture = None
        self._observations = None
        self.owner.current()

    def __reduce__(self):
        raise TypeError('production provider handoff cannot be serialized')

    def _lifetime(self):
        need(os.getpid() == self._pid and threading.get_ident() == self._thread and
             self.core.process(self._pid)['birth'] == self._birth,
             'handoff_existing_owner_process_or_thread_changed')

    @property
    def binding(self):
        # Every use decodes the original immutable owner bytes; a caller only
        # receives a copy and cannot extend the original work/native deadline.
        return self.core.decode(self._binding_bytes)

    def prepare_for_stream(self, *, portal_daemon, task, portal_attempt,
                           workspace, command, maximum_timeout_seconds):
        """Finish argv before Popen; publish context only from actual birth."""
        self._lifetime()
        need(not self._prepared and not self._closed and
             type(portal_daemon) is self._daemon_module.PortalImplementationDaemon,
             'handoff_single_exact_portal_provider_required')
        authority = portal_daemon._database_attempt_authority
        need(type(authority) is dict and all(type(authority.get(key)) is type(self.binding[key]) and
             authority[key] == self.binding[key] for key in
             ('database_task_cid', 'database_attempt_id', 'database_claim_id', 'database_attempt_number')) and
             type(task) is self._daemon_module.PortalTask and
             task.task_id == self.binding['portal_task_id'] and
             portal_daemon._canonical_ref(task) == self.binding['portal_task_revision_cid'] and
             type(portal_attempt) is int and portal_attempt == self.binding['portal_attempt'] and
             str(workspace) == self.binding['workspace'] and
             not portal_daemon.production_provider_policy,
             'handoff_actual_portal_attempt_workspace_or_route_differs')
        need(type(command) in (tuple, list) and len(command) >= 3 and
             tuple(command[:3]) == ('/usr/bin/python3', '-B', str(self.worker_path)) and
             all(type(part) is str and part for part in command) and
             not any(part == flag or part.startswith(flag+'=') for part in command for flag in OWNER_FLAGS),
             'handoff_exact_worker_argv_without_owner_overrides_required')
        for flag, expected in (('--transport-capsule', self.capsule_pin['path']),
                               ('--transport-capsule-sha256', self.capsule_pin['sha256'])):
            indices = [index for index, part in enumerate(command) if part == flag]
            need(len(indices) == 1 and indices[0]+1 < len(command) and command[indices[0]+1] == expected and
                 not any(part.startswith(flag+'=') for part in command),
                 'handoff_actual_worker_capsule_argv_differs')
        now = time.monotonic()
        deadline = self.binding['work_deadline_monotonic']
        need(type(maximum_timeout_seconds) in (int, float) and math.isfinite(maximum_timeout_seconds) and
             0 < deadline-now <= maximum_timeout_seconds and
             deadline <= self.binding['native_deadline_monotonic'],
             'handoff_original_absolute_deadline_required')
        self.owner.current()
        self.command = tuple(command) + (
            '--owner-context', str(self.server.context_path),
            '--owner-request-sha256', self.owner._request_sha256,
            '--owner-allocation-session-id', self.binding['allocation_session_id'],
        )
        self._prepared = True
        return list(self.command)

    @property
    def absolute_deadline(self):
        self._lifetime()
        need(self._prepared, 'handoff_provider_not_prepared')
        return self.binding['work_deadline_monotonic']

    def on_started(self, process):
        """Existing runner calls this before any original stdin is delivered."""
        self._lifetime()
        need(self._prepared and self._process is None and type(process) is subprocess.Popen,
             'handoff_one_actual_direct_Popen_required')
        self._process = process
        self._child_identity = self.core.process(process.pid)
        need(self._child_identity['parent_pid'] == self._pid and
             self._child_identity['group_pid'] == process.pid, 'handoff_actual_direct_worker_birth_required')
        self.server.bind_child(process)

    def poll_or_cancel(self):
        try:
            self._lifetime()
            need(self._prepared and self._process is not None and not self._closed,
                 'handoff_provider_lifetime_changed')
            return self.server.poll_or_cancel()
        except BaseException:
            return 'production_provider_handoff_currentness_refused'

    def close_after_stream_failure(self):
        """No speculative deletion: genuine close refuses any live consumer."""
        self._lifetime()
        if not self._closed:
            cleanup_deadline = time.monotonic()+self._cleanup_seconds
            def remaining_cleanup():
                remaining = cleanup_deadline-time.monotonic()
                need(remaining > 0, 'handoff_single_cleanup_budget_expired')
                return remaining
            if self._process is not None:
                need(self._child_identity is not None and
                     self._child_identity['pid'] == self._process.pid,
                     'handoff_cleanup_actual_birth_custody_unverified')
                child = {'pid': self._child_identity['pid'], 'birth': str(self._child_identity['birth'])}
                usage = self.owner._host._group_usage(child)
                need(usage['available'] is True, 'handoff_cleanup_group_inventory_unverified')
                if usage['live_processes']:
                    # The genuine strict SDK checks the ORIGINAL captured birth
                    # before any signal, including leader-exit and PID reuse.
                    # No fresh PID sample grants replacement authority.
                    fenced = self._cleanup_module.terminate_pid_tree(self._process.pid,
                        grace_seconds=min(0.1, remaining_cleanup()), freeze_first=True,
                        require_gone=True, owned_process_group_id=self._process.pid,
                        expected_root_start_time_ticks=int(self._child_identity['birth']),
                        strict_timeout_seconds=remaining_cleanup())
                    need(fenced is True, 'handoff_cleanup_strict_native_fence_unverified')
                self._process.wait(timeout=remaining_cleanup())
                usage = self.owner._host._group_usage(child)
                need(usage['available'] is True and usage['live_processes'] == 0,
                     'handoff_cleanup_group_drain_unverified')
            # SDK native-consumer and group checks still run independently.
            # Any false/unknown path above retains the endpoint and precharge.
            remaining_cleanup()
            self.server.close_after_group_drain()
            self._closed = True

    def after_stream_return(self, completed):
        """Collect independent evidence after the actual runner drains group."""
        self._lifetime()
        need(not self._completed and self._process is not None and
             type(completed) is subprocess.CompletedProcess and type(completed.returncode) is int and
             completed.returncode == self._process.returncode and tuple(completed.args) == self.command,
             'handoff_actual_stream_result_differs')
        # The genuine consumer independently requires native/group absence.
        # No post-exit current() is attempted: the old child identity is gone.
        self.server.close_after_group_drain()
        self._closed = True
        if completed.returncode == 0:
            self.server.assert_result_eligible()
            self._observations = self.server.observations()
            row = self.allocation_store.get_cli_session(self.binding['allocation_session_id'], 'grok_cli')
            need(type(row) is dict and all(type(row.get(key)) is str and row[key] for key in CAPTURE_FIELDS) and
                 row['allocation_session_id'] == self.binding['allocation_session_id'] and
                 row['provider'] == 'grok_cli' and row['workspace'] == self.binding['workspace'] and
                 row['resume_style'] == 'resume', 'handoff_independent_store_capture_readback_missing')
            observed = time.monotonic()
            need(observed < self.binding['native_deadline_monotonic'],
                 'handoff_independent_capture_after_native_deadline')
            self._capture = {'allocation_store_path': self.binding['allocation_store_path'],
                             'row': {key: row[key] for key in CAPTURE_FIELDS},
                             'observed_monotonic': observed}
        self._completed = True

    def freeze_owner_observations(self, *, transport_pin, output_path):
        """Additive metadata for the independent factual acceptance checker."""
        self._lifetime()
        need(self._completed and self._closed and self._capture is not None and self._observations,
             'handoff_no_successful_postdrain_owner_evidence')
        need(type(transport_pin) is dict and set(transport_pin) == {'path', 'bytes', 'sha256'} and
             Path(transport_pin['path']).stat().st_size == transport_pin['bytes'] and
             self.factory.source_digest(transport_pin['path']) == transport_pin['sha256'],
             'handoff_exact_transport_receipt_pin_required')
        context = self.core.decode(self.server._context_bytes)
        record = {'schema': 'legal-ir-grok-planning-transport-owner-observations/v1',
                  'context_sha256': context['context_sha256'], 'request_sha256': context['request_sha256'],
                  'transport_receipt_sha256': transport_pin['sha256'],
                  'source_bindings_pin': dict(self.source_bindings_pin), 'capsule_pin': dict(self.capsule_pin),
                  'rpc_observations': self._observations, 'capture_observation': self._capture}
        return self.server_module.fresh_record(self.core, output_path, record, self.owner._disk.directory)
