"""Native issuance of paired task and queue roles after launch admission.

This broker neither discovers nor migrates queues. The native launch parent
supplies an already admitted retained owner and the exact configured scopes.
No task transition, consumer lease, or completion is created by bootstrap.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
import os
import re
import threading
import time
from typing import Any, Mapping

from ..merge.database_worktree_registry import process_birth_id
from ..merge.owner_merge_queue import SERVICE_OPERATIONS as QUEUE_OPERATIONS
from ..merge.owner_recovery_runtime import SERVICE_OPERATIONS as RECOVERY_OPERATIONS
from ..merge.owner_recovery_runtime import recovery_scope_cid
from ..merge.worktree_lifecycle import read_process_birth
from ..task_sources.owner_merge_bootstrap import (
    REQUEST_FIELDS, REQUEST_SCHEMA, RESPONSE_SCHEMA, OwnerMergeBootstrapBundle,
)
from ..task_sources.control_plane_contracts import canonical_json_bytes
from ..task_sources.state_owner_bootstrap import (
    MAX_STATE_OWNER_BOOTSTRAP_BYTES,
    STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA, StateOwnerBootstrapError,
)


class NativeOwnerMergeBroker:
    """Issue and retain three separate exact-peer capabilities in owner memory."""

    def __init__(self, *, task_server, queue_server, repository_id, target_branch,
                 scope_bindings, issue_task, revoke_task, validate_peer,
                 ttl_seconds=86_400.0, renew_task=None):
        self.task_server = task_server
        self.queue_server = queue_server
        self.repository_id = repository_id
        self.target_branch = target_branch
        self.issue_task = issue_task
        self.revoke_task = revoke_task
        self.validate_peer = validate_peer
        self.renew_task = renew_task
        if renew_task is not None and not callable(renew_task):
            raise StateOwnerBootstrapError("native task renewal callback is invalid")
        self.ttl_seconds = float(ttl_seconds)
        if not math.isfinite(self.ttl_seconds) or not 1 <= self.ttl_seconds <= 86400:
            raise StateOwnerBootstrapError("native merge grant lifetime is outside its bound")
        self.scopes = {}
        for value in scope_bindings:
            scope = dict(value)
            if set(scope) != {"board_namespace", "config_cid", "plan_cid", "lane_id", "attempt_root"}:
                raise StateOwnerBootstrapError("native merge scope fields differ")
            if any(type(item) is not str or not item for item in scope.values()):
                raise StateOwnerBootstrapError("native merge scope identity is invalid")
            if scope["lane_id"] in self.scopes:
                raise StateOwnerBootstrapError("native merge lane scope repeats")
            self.scopes[scope["lane_id"]] = scope
        if not self.scopes or len(self.scopes) > 256:
            raise StateOwnerBootstrapError("native merge lane population is invalid")
        if not all(callable(item) for item in (issue_task, revoke_task, validate_peer)):
            raise StateOwnerBootstrapError("native merge requires owner callbacks")
        self._lock = threading.RLock()
        self._issued = {}
        self._identities = self._owner_identities()

    def _owner_identities(self):
        birth = read_process_birth(os.getpid())
        if birth is None:
            raise StateOwnerBootstrapError("native merge parent birth is unavailable")
        identities = []
        for server in (self.task_server, self.queue_server):
            identity = server.identity
            if identity is None:
                raise StateOwnerBootstrapError("native merge owner is unavailable")
            identity = identity.to_dict()
            if (identity.get("process_birth") != birth.to_dict()
                    or identity.get("process_birth_id") != process_birth_id(birth)):
                raise StateOwnerBootstrapError("native merge roles are not owned by this parent")
            gateway = server._command_gateway
            if gateway is None or dict(gateway.identity) != identity:
                raise StateOwnerBootstrapError("native merge gateway owner identity differs")
            identities.append(identity)
        if any(identities[0].get(key) == identities[1].get(key)
               for key in ("database_uuid", "store_id", "server_id", "listen_uri")):
            raise StateOwnerBootstrapError("native merge roles are not distinct")
        return tuple(identities)

    def _require_owners(self):
        if self._owner_identities() != self._identities:
            raise StateOwnerBootstrapError("native merge owner generation changed")

    def _live_grant(self, server, grant_id, birth, *, expected_grant=None):
        gateway = server._command_gateway
        if gateway is None:
            raise StateOwnerBootstrapError("native merge grant owner is unavailable")
        with gateway._grants_lock:
            grants = tuple(g for g in gateway._grants.values() if g.grant_id == grant_id)
        if len(grants) != 1:
            raise StateOwnerBootstrapError("native merge grant is no longer current")
        if expected_grant is not None and grants[0] != expected_grant:
            raise StateOwnerBootstrapError("native merge retained grant binding changed")
        peer = (birth.pid, os.stat(f"/proc/{birth.pid}").st_uid,
                birth.start_time_ticks)
        # The immutable process birth was re-read before this call; the gateway
        # also compares its original kernel peer PID/UID/start-time grant.
        return gateway._require_active_grant(grants[0], peer_identity=peer)

    def _retire(self, client_id, entry):
        for grant_id in entry.get("queue_grants", ()):
            self.queue_server.revoke_typed_client_grant(grant_id)
        task_grant = entry.get("task_grant")
        if task_grant:
            self.revoke_task(client_id, task_grant)
        if self._issued.get(client_id) is entry:
            self._issued.pop(client_id, None)

    def admit(self, request: Mapping[str, Any], *, peer_pid: int, peer_uid: int):
        with self._lock:
            self._require_owners()
            if (type(request) is not dict or set(request) != REQUEST_FIELDS
                    or request.get("schema") != REQUEST_SCHEMA
                    or type(request.get("request_id")) is not str
                    or re.fullmatch(r"[0-9a-f]{32}", request["request_id"]) is None
                    or type(request.get("pid")) is not int
                    or request["pid"] != peer_pid or peer_uid != os.geteuid()):
                raise StateOwnerBootstrapError("native merge request identity differs")
            birth = read_process_birth(peer_pid)
            if (birth is None or request["process_birth"] != birth.to_dict()
                    or request["process_birth_id"] != process_birth_id(birth)):
                raise StateOwnerBootstrapError("native merge peer birth differs")
            client_id = request["client_id"]
            if (type(client_id) is not str or not client_id or len(client_id) > 256
                    or any(ord(char) < 32 for char in client_id)):
                raise StateOwnerBootstrapError("native merge client identity is invalid")
            lane_id = str(self.validate_peer(peer_pid=peer_pid, client_id=client_id))
            scope = self.scopes.get(lane_id)
            if (scope is None or request["store_id"] != self._identities[0]["store_id"]
                    or request["config_cid"] != scope["config_cid"]
                    or request["plan_cid"] != scope["plan_cid"]):
                raise StateOwnerBootstrapError("native merge request scope differs")
            prior = self._issued.get(client_id)
            if prior is not None:
                prior_birth = read_process_birth(prior["request"]["pid"])
                if (prior_birth is None
                        or prior_birth.to_dict() != prior["request"]["process_birth"]):
                    self._retire(client_id, prior)
                elif prior["request"] == request:
                    if prior["response"]["scope_binding"] != scope:
                        raise StateOwnerBootstrapError("native merge retained lane scope changed")
                    for server, grant_id in ((self.task_server, prior["task_grant"]),
                        *((self.queue_server, key) for key in prior["queue_grants"])):
                        self._live_grant(server, grant_id, birth,
                            expected_grant=prior["grant_records"][grant_id])
                    return copy.deepcopy(prior["response"])
                else:
                    raise StateOwnerBootstrapError("live native merge peer cannot replace its request")
            task_request = {key: request[key] for key in (
                "pid", "process_birth", "process_birth_id", "client_id", "store_id"
            )}
            task_request["schema"] = STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA
            entry = {"request": copy.deepcopy(request), "queue_grants": [], "task_grant": "", "grant_records": {}}
            try:
                task, task_grant = self.issue_task(task_request, peer_pid=peer_pid, peer_uid=peer_uid)
                entry["task_grant"] = task_grant
                entry["grant_records"][task_grant] = self._live_grant(
                    self.task_server, task_grant, birth)
                scope_id = recovery_scope_cid(store_id=self._identities[1]["store_id"],
                    repository_id=self.repository_id, target_branch=self.target_branch,
                    scope_binding=scope)
                roles = {}
                for name, operations in (("queue", QUEUE_OPERATIONS), ("recovery", RECOVERY_OPERATIONS)):
                    entity_scopes = {"repository_id": self.repository_id,
                        "target_branch": self.target_branch, "consumer_id": client_id}
                    if name == "recovery":
                        entity_scopes["recovery_scope_cid"] = scope_id
                    token, grant = self.queue_server.issue_typed_client_grant_record(
                        client_id=client_id, process_birth_id=request["process_birth_id"],
                        peer_pid=peer_pid, ttl_seconds=self.ttl_seconds,
                        allowed_operations=tuple(operations), entity_scopes=entity_scopes)
                    entry["queue_grants"].append(grant.grant_id)
                    entry["grant_records"][grant.grant_id] = grant
                    roles[name] = {"socket_path": str(self.queue_server.typed_command_socket_path()),
                        "store_id": self._identities[1]["store_id"],
                        "server_id": self._identities[1]["server_id"],
                        "client_id": client_id, "process_birth_id": request["process_birth_id"],
                        "token": token}
                self._require_owners()
                response = {"schema": RESPONSE_SCHEMA, "ok": True,
                    "request_id": request["request_id"], "task": task,
                    "task_owner_identity": self._identities[0],
                    "queue_owner_identity": self._identities[1], **roles,
                    "repository_id": self.repository_id, "target_branch": self.target_branch,
                    "scope_binding": scope, "recovery_scope_cid": scope_id}
                OwnerMergeBootstrapBundle.from_response(response, request=request,
                    peer_pid=os.getpid(), peer_uid=os.geteuid())
                if len(canonical_json_bytes(response)) > MAX_STATE_OWNER_BOOTSTRAP_BYTES:
                    raise StateOwnerBootstrapError("native merge bundle exceeds its frame bound")
                entry["response"] = copy.deepcopy(response)
                entry["renew_at"] = time.monotonic() + min(300.0, self.ttl_seconds / 3)
                self._issued[client_id] = entry
                return copy.deepcopy(response)
            except BaseException:
                self._retire(client_id, entry)
                raise

    def maintain(self):
        """Renew live exact-peer grants; never revive expired or displaced authority."""
        with self._lock:
            self._require_owners()
            now = time.monotonic()
            for client_id, entry in tuple(self._issued.items()):
                if now < entry["renew_at"]:
                    continue
                request = entry["request"]
                birth = read_process_birth(request["pid"])
                if birth is None or birth.to_dict() != request["process_birth"]:
                    self._retire(client_id, entry)
                    continue
                lane_id = str(self.validate_peer(peer_pid=birth.pid, client_id=client_id))
                if self.scopes.get(lane_id) != entry["response"]["scope_binding"]:
                    raise StateOwnerBootstrapError("native merge retained lane scope changed")
                roles = ((self.task_server, entry["task_grant"]),
                    *((self.queue_server, key) for key in entry["queue_grants"]))
                # Validate the complete retained bundle before extending any
                # role. Matching an ID alone must not bless changed permissions.
                for server, grant_id in roles:
                    self._live_grant(server, grant_id, birth,
                        expected_grant=entry["grant_records"][grant_id])
                for server, grant_id in roles:
                    issued = entry["grant_records"][grant_id]
                    if server is self.task_server and self.renew_task is not None:
                        renewed = self.renew_task(client_id, issued)
                    else:
                        renewed = server.renew_typed_client_grant(
                            grant_id, ttl_seconds=self.ttl_seconds)
                    if (type(renewed) is not type(issued)
                            or replace(renewed, issued_at=issued.issued_at,
                                expires_at=issued.expires_at) != issued):
                        raise StateOwnerBootstrapError("native merge renewal changed grant authority")
                    self._live_grant(server, grant_id, birth, expected_grant=renewed)
                    entry["grant_records"][grant_id] = renewed
                entry["renew_at"] = now + min(300.0, self.ttl_seconds / 3)

    def owns_task_grant(self, client_id, grant_id):
        """Tell the parent which task grants are exclusively bundle-maintained."""
        with self._lock:
            entry = self._issued.get(client_id)
            return bool(entry and entry["task_grant"] == grant_id)

    def close(self):
        with self._lock:
            for client_id, entry in tuple(self._issued.items()):
                self._retire(client_id, entry)
