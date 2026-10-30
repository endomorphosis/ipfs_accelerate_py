"""Concrete existing-capability composition for one NEW same-PID attempt.

Source only. Call inside the original genuine P root/phase/disk/source-lock
contexts, in the process that constructed the actual database daemon. This
module never acquires or recreates those capabilities and never rearms tasks.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import types

P_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets')
MODULE_NAME = 'ipfs_accelerate_py.cli_runtime.grok_same_pid_owner_profile'
MAIN_ATTEMPT_FIELDS = ('task_cid', 'attempt_id', 'claim_id', 'fencing_token', 'fence_epoch', 'lease_id')
EXPECTED_MODULE_NAMES = (
    'ipfs_accelerate_py.cli_runtime.grok_production_owner',
    'ipfs_accelerate_py.cli_runtime.grok_existing_phase_owner',
    'ipfs_accelerate_py.cli_runtime.grok_owned_unix_socket',
    'ipfs_accelerate_py.cli_runtime.grok_production_owner_server',
    'ipfs_accelerate_py.cli_runtime.grok_production_handoff', MODULE_NAME)


class SamePidProfileRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise SamePidProfileRefusal(reason)


def _require_main_bridge_join(binding, expected_attempt_fields, *, claim_owner_session_id,
                              attempt_owner_session_id, database_owner_session_id):
    """Pure metadata comparison; grants no claim/native/attempt authority.

    Main's _binding_payload contains these exact six attempt fields and has no
    owner_session_id column. Owner-session equality comes from the actual live
    claim, exact SDK attempt and actual DB owner, which the caller guards.
    """
    need(type(binding) is dict and type(expected_attempt_fields) is dict and
         set(expected_attempt_fields) == set(MAIN_ATTEMPT_FIELDS) and
         all(type(binding.get(key)) is type(value) and binding[key] == value
             for key, value in expected_attempt_fields.items()),
         'same_pid_profile_bridge_actual_claim_join_differs')
    need(type(claim_owner_session_id) is str and claim_owner_session_id and
         claim_owner_session_id == attempt_owner_session_id == database_owner_session_id,
         'same_pid_profile_live_owner_session_join_differs')


def _expected_module_paths(accelerator):
    """Pure source-path mapping; does not import a module or admit a tree."""
    return {name: accelerator/('ipfs_accelerate_py/cli_runtime/'+name.rsplit('.', 1)[-1]+'.py')
            for name in EXPECTED_MODULE_NAMES}


def _require_portal_projection_join(authority, expected):
    """Compare an already guarded lifecycle-only Portal projection."""
    fields = {'database_attempt_id', 'database_task_cid', 'database_claim_id',
              'database_attempt_number', 'database_binding_id', 'task_id'}
    need(type(authority) is dict and type(expected) is dict and set(expected) == fields and
         all(type(authority.get(key)) is type(value) and authority[key] == value
             for key, value in expected.items()),
         'same_pid_profile_existing_portal_projection_join_differs')


class ExistingSamePidOwnerProfile:
    """One exact DB attempt, one direct worker, existing finite allocation.

    Resource and DB lifetime belong to the calling owner context. The profile
    keeps the real handoff reference so an independent postdrain watcher can
    freeze owner observations and invoke both factual acceptance checks.
    """

    def __init__(self, *, core, factory_module, listener_module, server_module,
                 handoff_module, database_owner, allocation_store, root, phase,
                 disk, source_roots, source_pins, source_locks, source_profile_pin,
                 capsule_pin, source_generation, worker_path, socket_parent,
                 allocation_session_id, credential_slot, credential_fingerprint,
                 work_deadline_monotonic, native_deadline_monotonic,
                 expected_task_cid):
        need(all(type(value) is types.ModuleType and sys.modules.get(value.__name__) is value
                 for value in (core, factory_module, listener_module, server_module, handoff_module)),
             'same_pid_profile_genuine_normal_modules_required')
        need(type(source_roots) is tuple and len(source_roots) == 2 and source_roots[1] == P_ROOT and
             type(source_pins) is dict, 'same_pid_profile_exact_canonical_source_roots_required')
        accelerator = source_roots[0]
        daemon = factory_module.module('ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon',
            accelerator, 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py', source_pins)
        bridge = factory_module.module('ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge',
            accelerator, 'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py', source_pins)
        resources = factory_module.module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler',
            P_ROOT, 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py', source_pins)
        parented = factory_module.module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources',
            P_ROOT, 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py', source_pins)
        allocation = factory_module.module('ipfs_accelerate_py.llm_allocation.duckdb_store',
            accelerator, 'ipfs_accelerate_py/llm_allocation/duckdb_store.py', source_pins)
        need(type(database_owner) is daemon.DatabaseImplementationDaemon and
             type(allocation_store) is allocation.AllocationStore and allocation_store._ready is True and
             type(root) is resources.ResourceLease and type(phase) is resources.ResourceLease and
             type(disk) is parented.ParentedPhaseResourceReservation,
             'same_pid_profile_actual_database_and_P_capabilities_required')
        need(os.environ.get('IPFS_ACCELERATE_LLM_ALLOCATION_DB') == str(allocation_store.path),
             'same_pid_profile_existing_allocation_store_environment_required')
        need(root.owner_pid == phase.owner_pid == disk.native_lease.owner_pid == os.getpid() and
             root.parent_lease_id is None and phase.parent_lease_id == root.lease_id and
             disk._parent is phase and disk.native_lease.parent_lease_id == phase.lease_id and
             root._scheduler is phase._scheduler and
             disk.native_lease._scheduler.state_path == root._scheduler.state_path and disk._child is None,
             'same_pid_profile_existing_unused_same_PID_native_chain_required')
        need(type(expected_task_cid) is str and expected_task_cid and
             type(work_deadline_monotonic) in (int, float) and type(native_deadline_monotonic) in (int, float) and
             time.monotonic() < work_deadline_monotonic <= native_deadline_monotonic <= disk._deadline,
             'same_pid_profile_existing_absolute_deadlines_required')
        need(type(source_profile_pin) is dict and set(source_profile_pin) == {'path', 'bytes', 'sha256'} and
             source_pins.get(source_profile_pin['path']) == source_profile_pin['sha256'],
             'same_pid_profile_complete_source_profile_required')
        modules = {core.__name__: core, factory_module.__name__: factory_module,
                   listener_module.__name__: listener_module, server_module.__name__: server_module,
                   handoff_module.__name__: handoff_module, MODULE_NAME: sys.modules.get(MODULE_NAME)}
        expected_paths = _expected_module_paths(accelerator)
        need(set(modules) == set(expected_paths) and all(type(value) is types.ModuleType and
             value.__file__ == value.__spec__.origin and
             Path(value.__file__) == expected_paths[name] and
             source_pins.get(value.__file__) == factory_module.source_digest(value.__file__)
             for name, value in modules.items()), 'same_pid_profile_complete_normal_package_source_join_required')
        self.core, self.factory = core, factory_module
        self.listener_module, self.server_module, self.handoff_module = listener_module, server_module, handoff_module
        self.daemon_module, self.bridge_module = daemon, bridge
        self.database_owner, self.allocation_store = database_owner, allocation_store
        self.root, self.phase, self.disk = root, phase, disk
        self.source_roots, self.source_pins, self.source_locks = source_roots, dict(source_pins), source_locks
        self.source_profile_pin, self.capsule_pin = dict(source_profile_pin), dict(capsule_pin)
        self.worker_path, self.socket_parent = Path(worker_path), Path(socket_parent)
        self.expected_task_cid = expected_task_cid
        self._configuration = core.canonical({'source_generation': source_generation,
            'allocation_session_id': allocation_session_id, 'credential_slot': credential_slot,
            'credential_fingerprint': credential_fingerprint,
            'work_deadline_monotonic': work_deadline_monotonic,
            'native_deadline_monotonic': native_deadline_monotonic})
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._birth = core.process(self._pid)['birth']
        self._portal = self._attempt = self._listener = self._server = self._handoff = None
        self._prepared = False

    def __reduce__(self):
        raise TypeError('same-PID owner profile cannot be serialized')

    def _lifetime(self):
        need(os.getpid() == self._pid and threading.get_ident() == self._thread and
             self.core.process(self._pid)['birth'] == self._birth,
             'same_pid_profile_owner_process_or_thread_changed')

    def bind_portal_attempt(self, *, portal_daemon, attempt, paths, binding):
        """New optional bridge hook supplies its exact attempt object directly."""
        self._lifetime()
        need(self._attempt is None and self._portal is None and
             type(attempt) is self.daemon_module.DatabaseTaskAttempt and
             type(portal_daemon) is self.daemon_module.PortalImplementationDaemon and
             type(paths) is self.bridge_module.DatabasePortalAttemptPaths and
             type(binding) is dict and attempt.task_cid == self.expected_task_cid,
             'same_pid_profile_one_exact_authorized_attempt_required')
        self.database_owner._owned_running_attempt(attempt)
        claim = self.database_owner._attempt_claim(attempt)
        self.database_owner._protect_attempt_claim(attempt, claim)
        _require_main_bridge_join(binding, {key: getattr(attempt, key) for key in MAIN_ATTEMPT_FIELDS},
            claim_owner_session_id=claim.owner_session_id, attempt_owner_session_id=attempt.owner_session_id,
            database_owner_session_id=self.database_owner.owner_session_id)
        authority = portal_daemon._database_attempt_authority
        _require_portal_projection_join(authority, {
            'database_attempt_id': attempt.attempt_id, 'database_task_cid': attempt.task_cid,
            'database_claim_id': attempt.claim_id, 'database_attempt_number': attempt.attempt_number,
            'database_binding_id': binding['binding_id'], 'task_id': binding['task_alias']})
        need(portal_daemon.production_owner_handoff_factory is None,
             'same_pid_profile_existing_portal_projection_join_differs')
        self._attempt, self._portal = attempt, portal_daemon
        # The full original binding remains sealed/verified by the bridge.
        # Retain only the immutable, already-joined alias used below; it is not
        # an RPC frame and unrelated full-binding size does not change gates.
        self._bound_task_alias = binding['task_alias']
        # This is an ordinary live bound method, never a JSON grant or closure
        # populated with reconstructed leases. Its source participates in pins.
        portal_daemon.production_owner_handoff_factory = self.prepare_handoff

    def prepare_handoff(self, *, portal_daemon, task, portal_attempt, workspace,
                        command, checkpoint_dir, maximum_timeout_seconds):
        self._lifetime()
        need(not self._prepared and portal_daemon is self._portal and self._attempt is not None and
             type(task) is self.daemon_module.PortalTask and type(portal_attempt) is int,
             'same_pid_profile_one_real_provider_dispatch_required')
        need(os.environ.get('IPFS_ACCELERATE_LLM_ALLOCATION_DB') == str(self.allocation_store.path),
             'same_pid_profile_allocation_store_environment_changed')
        self._prepared = True
        attempt = self._attempt
        claim = self.database_owner._attempt_claim(attempt)
        self.database_owner._protect_attempt_claim(attempt, claim)
        need(task.task_id == self._bound_task_alias and str(workspace) == str(Path(workspace).resolve(strict=True)),
             'same_pid_profile_current_task_and_final_workspace_required')
        configuration = self.core.decode(self._configuration)
        launch = {'schema': self.core.CONTEXT_SCHEMA, **configuration,
            'database_task_cid': str(claim.task_cid), 'database_attempt_id': str(claim.attempt_id),
            'database_claim_id': str(claim.claim_id), 'database_attempt_number': int(claim.attempt_number),
            'coordination_lease_id': str(claim.lease_id), 'owner_session_id': str(claim.owner_session_id),
            'fencing_token': int(claim.fencing_token), 'fence_epoch': int(claim.fence_epoch),
            'portal_task_id': task.task_id, 'portal_task_revision_cid': portal_daemon._canonical_ref(task),
            'portal_attempt': portal_attempt, 'native_attempt_id': str(attempt.attempt_id),
            'allocation_store_path': str(self.allocation_store.path), 'workspace': str(workspace),
            'source_bindings_sha256': self.source_profile_pin['sha256'], 'capsule_sha256': self.capsule_pin['sha256'],
            'worker_source_sha256': self.source_pins[str(self.worker_path)],
            'owner_source_sha256': self.source_pins[self.core.__file__],
            'root_lease_id': self.root.lease_id, 'phase_lease_id': self.phase.lease_id,
            'disk_native_lease_id': self.disk.native_lease.lease_id, 'disk_reservation_id': self.disk.reservation_id}
        owner = self.factory.bind_existing_production_owner(core=self.core, database_owner=self.database_owner,
            attempt=attempt, root=self.root, phase=self.phase, disk=self.disk, source_roots=self.source_roots,
            source_pins=dict(self.source_pins), source_locks=self.source_locks, launch_binding=launch,
            native_deadline_monotonic=configuration['native_deadline_monotonic'],
            source_profile_pin={key: self.source_profile_pin[key] for key in ('path', 'sha256')})
        directory = self.disk.directory/'production-owner-handoff'
        directory.mkdir(mode=0o700, exist_ok=False)
        observations = directory/'rpc-observations'
        observations.mkdir(mode=0o700, exist_ok=False)
        try:
            self._listener = self.listener_module.open_owned_unix_listener(disk=self.disk, parent=self.socket_parent)
            self._server = self.server_module.ProductionOwnerServer(core=self.core, factory_module=self.factory,
                listener_module=self.listener_module, owner=owner, listener=self._listener,
                context_path=directory/'owner-context.json', observation_directory=observations)
            self._handoff = self.handoff_module.ProductionProviderHandoff(server_module=self.server_module,
                server=self._server, allocation_store=self.allocation_store, capsule_pin=dict(self.capsule_pin),
                source_bindings_pin=dict(self.source_profile_pin), worker_path=self.worker_path)
            return self._handoff
        except BaseException:
            # Factory has not returned: the Portal callsite holds no reference
            # yet. Cleanup is this profile's responsibility, and exact native
            # absence is still required. Failed regular evidence is retained.
            self.close_retained_listener_after_failure()
            raise

    @property
    def handoff(self):
        self._lifetime()
        need(self._handoff is not None, 'same_pid_profile_handoff_not_created')
        return self._handoff

    def close_retained_listener_after_failure(self):
        self._lifetime()
        if self._listener is not None and not self._listener.closed:
            if self._server is not None:
                self._server.close_after_group_drain()
            else:
                self._listener.close_after_group_drain()


def bind_existing_database_callbacks(*, profile, bridge):
    """Use the existing public callback API before one new dispatch.

    The caller, inside its already admitted same-PID contexts, constructed the
    actual initialized database owner and bridge. This does not call run_once,
    admit a root, allocate a task, release a claim, retry, rearm or loop.
    """
    need(type(profile) is ExistingSamePidOwnerProfile and
         type(bridge) is profile.bridge_module.DatabasePortalExecutionBridge and
         bridge.provider_handoff_binder == profile.bind_portal_attempt and
         bridge.task_source is profile.database_owner.task_source,
         'same_pid_profile_exact_existing_bridge_and_database_source_required')
    profile._lifetime()
    profile.database_owner.bind_execution_callbacks(provider_fn=bridge.run_provider,
        effect_fn=bridge.apply_effect, validation_fn=bridge.validate_effect)
