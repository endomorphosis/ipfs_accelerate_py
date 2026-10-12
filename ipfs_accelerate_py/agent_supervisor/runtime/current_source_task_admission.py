"""One live native task admission for an existing owner's source reader.

This host-only handle retains the actual coordinator and typed gateway. It
does not allocate a worktree, authenticate a coding child, compile a context,
or create a database, CAS, gateway or owner. Source binding remains bind-once;
a failure after publication never silently rolls it back or rebinds it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import threading

from ..merge.database_coordination import ProcessSerializedDatabaseCoordinator
from ..task_sources import typed_state_owner as typed
from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from .current_source_lookup import CurrentSourceLookup
from .numerical_obligation_client import process_identity
from .quack_state_server import QuackStateServer, ServerLifecycle
from .repository_source_inventory import _cid

SCHEMA = "supervisor-native-source-task-admission@1"
_SEAL = object()
_CLAIM_FIELDS = frozenset(("attempt_id", "claim_id", "attempt_number", "lease_id",
                         "owner_session_id", "fencing_token", "fence_epoch"))


class CurrentSourceTaskAdmissionError(ValueError):
    """The retained task/claim/source binding is no longer admitted."""


def _need(value, reason):
    if not value:
        raise CurrentSourceTaskAdmissionError(reason)


def _owned_file(path):
    _need(isinstance(path, Path), "native authority path required")
    _need(path.is_absolute() and path.resolve(strict=True) == path,
          "native authority path changed")
    parent = path.parent.lstat()
    _need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid()
          and not parent.st_mode & 0o022, "native authority parent must remain owner controlled")
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
          and info.st_nlink == 1 and not info.st_mode & 0o022,
          "native authority requires an existing owner regular file")
    return (path, info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode),
            parent.st_dev, parent.st_ino, parent.st_uid, stat.S_IMODE(parent.st_mode))


class NativeSourceTaskAdmission:
    """Exact host-issued handle; serialized snapshots cannot admit anything."""

    def __init__(self, seal, *, server, attempt_authority, task_cid,
                 expected_claim_identity, peer_identity, client_id):
        _need(seal is _SEAL and type(self) is NativeSourceTaskAdmission,
              "native source task admission must be captured by the host")
        _need(type(server) is QuackStateServer
              and type(attempt_authority) is ProcessSerializedDatabaseCoordinator,
              "original server and native process-serialized coordinator required")
        _cid(task_cid)
        _need(type(expected_claim_identity) is dict
              and set(expected_claim_identity) == _CLAIM_FIELDS,
              "exact seven-field native claim identity required")
        claim = typed._validated_database_claim_identity(expected_claim_identity)
        _need(type(peer_identity) is tuple and len(peer_identity) == 3
              and all(type(value) is int for value in peer_identity)
              and peer_identity[0] > 0 and peer_identity[1] == os.geteuid()
              and peer_identity[2] > 0, "actual owner-UID kernel peer tuple required")
        _need(type(client_id) is str and client_id.startswith("database-implementation-daemon:")
              and len(client_id.encode()) <= 1024 and not any(c in client_id for c in "\0\r\n"),
              "expected owner-attested native daemon client required")
        self._seal, self._server, self._coordinator = seal, server, attempt_authority
        self._claim = dict(task_cid=task_cid, **claim)
        self._claim_bytes = canonical_json_bytes(self._claim)
        self._peer, self._client_id, self._owner_pid = peer_identity, client_id, os.getpid()
        self._lock, self._checking = threading.RLock(), False
        self._source = self._source_refs = self._source_bytes = None
        self._binding_in_progress = False
        self._snapshot = self._grant = None
        self._original_snapshot = self._original_grant = None
        with server._lock:
            _need(server.lifecycle is ServerLifecycle.READY, "native source owner is not ready")
            server.ready()
            self._gateway, self._connection, self._owner_lock = (
                server._command_gateway, server._connection, server._lock)
            _need(type(self._gateway) is typed.TypedStateOwnerGateway
                  and self._gateway._connection is self._connection
                  and self._gateway._transaction_lock is self._owner_lock,
                  "exact original native gateway required")
            self._identity = server.identity
            self._server_config = server.config
            self._owner_bytes = canonical_json_bytes(dict(self._gateway.identity))
            self._control_identity = _owned_file(Path(server.config.database_path))
        self._coordinator_identity = _owned_file(attempt_authority.database_path)
        self._coordinator_config = self._config()
        self._original = (server, self._gateway, self._connection, self._owner_lock,
                          attempt_authority, self._identity)
        self.require_current()

    def _config(self):
        c = self._coordinator
        return (c.database_path, c.serialization_lock_path, c._state_lock, c._clock_ms,
                c._default_lease_ms, c._lock_timeout_seconds, c._require_existing_authority)

    def _retained(self):
        _need(self._seal is _SEAL and type(self) is NativeSourceTaskAdmission
              and os.getpid() == self._owner_pid
              and all(left is right for left, right in zip(self._original,
                  (self._server, self._gateway, self._connection, self._owner_lock,
                   self._coordinator, self._identity)))
              and self._server.lifecycle is ServerLifecycle.READY
              and self._server._command_gateway is self._gateway
              and self._server._connection is self._connection
              and self._server._lock is self._owner_lock
              and self._server.identity is self._identity
              and self._server.config is self._server_config
              and _owned_file(Path(self._server.config.database_path)) == self._control_identity
              and self._identity.process_birth.pid == self._owner_pid
              and self._gateway._connection is self._connection
              and self._gateway._transaction_lock is self._owner_lock
              and canonical_json_bytes(dict(self._gateway.identity)) == self._owner_bytes,
              "native source task original owner changed")
        # Do not acquire the coordinator lifecycle lock while SERVER is held.
        # execute_with_task_claim_fence independently requires the open adapter.
        _need(type(self._coordinator) is ProcessSerializedDatabaseCoordinator
              and not self._coordinator._closed and self._config() == self._coordinator_config
              and _owned_file(self._coordinator.database_path) == self._coordinator_identity,
              "native source task coordinator changed or closed")
        _need(canonical_json_bytes(self._claim) == self._claim_bytes,
              "native source task claim selection changed")
        _need(self._snapshot is self._original_snapshot and self._grant is self._original_grant,
              "native source task captured snapshot changed")

    def _native(self):
        self._retained()
        self._gateway._require_live_server_binding()
        generation = self._connection.execute(
            "SELECT generation, fence_epoch, database_uuid FROM store_generations "
            "ORDER BY generation DESC LIMIT 1").fetchone()
        owner = self._gateway.identity
        _need(generation is not None and tuple(generation[index] for index in range(3))
              == (owner["generation"], owner["fence_epoch"], owner["database_uuid"]),
              "native source task owner generation or fence changed")
        peer = process_identity(self._peer[0])
        _need((peer["pid"], peer["uid"], peer["start_time_ticks"]) == self._peer,
              "native source task daemon died or changed")
        rows = self._connection.execute(
            "SELECT task_alias, status, revision, body_json FROM tasks WHERE task_cid = ? LIMIT 2",
            [self._claim["task_cid"]]).fetchall()
        _need(len(rows) == 1, "native source task absent or ambiguous")
        alias, status, revision, raw = (rows[0][index] for index in range(4))
        _need(type(alias) is str and 0 < len(alias.encode()) <= 4096
              and status == "in_progress" and type(revision) is int and revision >= 1,
              "native source task is not the current in-progress task")
        body, canonical = typed._closed_canonical_json_object(raw,
            noun="native source task full body", maximum_bytes=256 * 1024)
        receipt = body.get("completion_receipt")
        _need(type(receipt) is dict
              and receipt.get("operation") == "database_attempt_admitted"
              and receipt.get("claim_phase_schema") == typed.TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
              and receipt.get("attempt_execution_phase") == "claimed"
              and type(receipt.get("attempt_execution_revision")) is int
              and receipt["attempt_execution_revision"] == 1,
              "native source task lacks its admitted provider claim")
        _need(canonical_json_bytes(typed._validated_database_claim_identity(receipt))
              == canonical_json_bytes({key: self._claim[key] for key in _CLAIM_FIELDS}),
              "native source task receipt claim differs")
        attestation = typed._validated_database_claim_process_attestation(receipt)
        with self._gateway._grants_lock:
            grants = [g for g in self._gateway._grants.values()
                      if g.grant_id == attestation["grant_id"]]
        _need(len(grants) == 1, "native source task active daemon grant unavailable")
        grant = self._gateway._require_active_grant(grants[0], peer_identity=self._peer)
        _need(grant.client_id == self._client_id and (self._grant is None
              or grant.grant_id == self._grant.grant_id), "native source task daemon grant changed")
        typed._require_database_claim_process_attestation(receipt, grant=grant)
        material = dict(schema=SCHEMA, task_cid=self._claim["task_cid"], task_alias=alias,
            task_revision=revision, expected_task_body_cid=content_identity(body),
            claim_identity={key: self._claim[key] for key in _CLAIM_FIELDS},
            task_content_read_only=True, proof_authority=False, execution_authority=False,
            completion_authority=False, allocation_ownership_verified=False)
        observed = (canonical_json_bytes(material), canonical.encode())
        _need(self._snapshot is None or observed == self._snapshot,
              "native source task revision, phase or full body changed")
        self._retained()
        return observed, grant

    def require_current(self):
        """Check COOR then SERVER; same owned live lease/grant renewal is allowed."""
        with self._lock:
            _need(not self._checking, "native source task admission reentry refused")
            self._checking = True
            try:
                self._retained()
                def checked(_lease):
                    with self._owner_lock:
                        before, grant = self._native()
                        after, current = self._native()
                        _need(before == after and grant.grant_id == current.grant_id,
                              "native source task changed during its fence")
                        return after, current
                observed, grant = self._coordinator.execute_with_task_claim_fence(self._claim, checked)
                with self._owner_lock:
                    final, current = self._native()
                    _need(final == observed and current.grant_id == grant.grant_id,
                          "native source task changed after its coordinator fence")
                self._retained()
                if self._snapshot is None:
                    self._snapshot, self._grant = observed, grant
                    self._original_snapshot, self._original_grant = observed, grant
                return json.loads(self._snapshot[0])
            finally:
                self._checking = False

    def bind_current_source_lookup(self, *, repository, cas, expected_inventory_cid,
                                   request, limits, context_cid, allowed_paths,
                                   max_response_bytes=16384, max_slice_bytes=4096):
        """Host supplies source/workspace policy; no allocation authority is inferred."""
        with self._lock:
            _need(not self._binding_in_progress and self._source is None,
                  "native source task bind is already consumed or in progress")
            self._binding_in_progress = True
            try:
                before = self.require_current()
                with self._owner_lock:
                    _need(self._gateway._current_source_lookup is None,
                          "native source lookup is already bound")
                    try:
                        self._server.bind_current_source_lookup(repository=repository, cas=cas,
                            expected_inventory_cid=expected_inventory_cid, request=request, limits=limits,
                            task_cid=before["task_cid"], task_revision=before["task_revision"],
                            expected_task_body_cid=before["expected_task_body_cid"], context_cid=context_cid,
                            allowed_paths=allowed_paths, max_response_bytes=max_response_bytes,
                            max_slice_bytes=max_slice_bytes)
                    finally:
                        # Preserve an actually published reader even if its last
                        # native validation raises; there is no rollback/rebind.
                        service = self._gateway._current_source_lookup
                        if type(service) is CurrentSourceLookup:
                            self._source, self._source_refs = service, (service, cas, service._root)
                            self._source_bytes = self._gateway._current_source_lookup_binding
                            service._native_source_task_admission = self
                    self.require_source_binding(server=self._server, service=self._source)
                self.require_current()
                return json.loads(self._source_bytes)
            finally:
                self._binding_in_progress = False

    def require_source_binding(self, *, server, service):
        """Source join only; caller holding SERVER must not acquire COOR here."""
        self._retained()
        _need(server is self._server and service is self._source
              and type(service) is CurrentSourceLookup and self._source_refs is not None
              and getattr(service, "_native_source_task_admission", None) is self
              and service is self._source_refs[0] and service._store.cas is self._source_refs[1]
              and service._root == self._source_refs[2], "native source task binding changed")
        self._gateway._assert_current_source_binding(service)
        actual = canonical_json_bytes(service.binding(self._gateway.identity))
        _need(actual == self._source_bytes, "native source task canonical binding changed")
        value, task = json.loads(actual), json.loads(self._snapshot[0])
        _need(all(value[key] == task[key] for key in
                  ("task_cid", "task_revision", "expected_task_body_cid")),
              "native source task binding belongs to another task snapshot")


def capture_native_source_task_admission(*, server, attempt_authority, task_cid,
                                        expected_claim_identity, peer_identity, client_id):
    return NativeSourceTaskAdmission(_SEAL, server=server, attempt_authority=attempt_authority,
        task_cid=task_cid, expected_claim_identity=expected_claim_identity,
        peer_identity=peer_identity, client_id=client_id)
