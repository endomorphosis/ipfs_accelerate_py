"""Versioned native task/queue/recovery handoff on an inherited private socket.

Only the native controller provisions these roles. Neither a bundle nor its
client parser supplies source, migration, or prior-consumer closure authority.
Tokens remain private to the exact daemon birth and are never environment state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket
import struct
import uuid
from typing import Any, Mapping

from ..merge.database_worktree_registry import process_birth_id
from ..merge.worktree_lifecycle import read_process_birth
from .state_owner_bootstrap import (
    StateOwnerBootstrapCredentials,
    StateOwnerBootstrapError,
    _connect_inherited_listener,
    _receive_exact,
    _send_frame,
    validate_state_owner_bootstrap_listener,
    MAX_STATE_OWNER_BOOTSTRAP_BYTES,
)

REQUEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/owner-merge-bootstrap-request@1"
RESPONSE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/owner-merge-bootstrap-response@1"
REQUEST_FIELDS = frozenset(
    {
        "schema",
        "request_id",
        "pid",
        "process_birth",
        "process_birth_id",
        "client_id",
        "store_id",
        "config_cid",
        "plan_cid",
    }
)


def receive_bundle_frame(channel):
    """Closed bounded JSON reader shared by this native broker and its client."""
    size = int.from_bytes(_receive_exact(channel, 4), "big")
    if not 2 <= size <= MAX_STATE_OWNER_BOOTSTRAP_BYTES:
        raise StateOwnerBootstrapError("native bundle frame size is invalid")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(
            _receive_exact(channel, size),
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
        pending = [(value, 0)]
        visited = 0
        while pending:
            current, depth = pending.pop()
            visited += 1
            if depth > 12 or visited > 1024:
                raise ValueError("nested frame exceeds bound")
            if type(current) is dict:
                pending.extend((item, depth + 1) for item in current.values())
            elif type(current) is list:
                pending.extend((item, depth + 1) for item in current)
            elif type(current) not in (str, int, bool, type(None)):
                raise ValueError("unsupported frame scalar")
        if type(value) is not dict:
            raise ValueError("frame is not object")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise StateOwnerBootstrapError("native bundle frame is malformed") from exc


def _closed(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise StateOwnerBootstrapError(
            "native merge bundle differs from closed contract"
        )
    return value


def _text(value):
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or any(ord(c) < 32 for c in value)
    ):
        raise StateOwnerBootstrapError("native merge bundle identity is malformed")
    return value


@dataclass(frozen=True)
class MergeRoleCredential:
    socket_path: str
    store_id: str
    server_id: str
    client_id: str
    process_birth_id: str
    token: str = field(repr=False)

    @classmethod
    def parse(cls, payload, *, owner_identity, client_id, birth_id):
        fields = {
            "socket_path",
            "store_id",
            "server_id",
            "client_id",
            "process_birth_id",
            "token",
        }
        _closed(payload, fields)
        values = {key: _text(payload[key]) for key in fields}
        path = Path(values["socket_path"])
        if (
            not path.is_absolute()
            or str(path) != values["socket_path"]
            or ".." in path.parts
            or len(values["token"]) < 16
            or values["client_id"] != client_id
            or values["process_birth_id"] != birth_id
            or values["store_id"] != owner_identity.get("store_id")
            or values["server_id"] != owner_identity.get("server_id")
        ):
            raise StateOwnerBootstrapError(
                "native merge credential differs from its owner or peer"
            )
        return cls(**values)


@dataclass(frozen=True)
class OwnerMergeBootstrapBundle:
    task: StateOwnerBootstrapCredentials = field(repr=False)
    queue: MergeRoleCredential = field(repr=False)
    recovery: MergeRoleCredential = field(repr=False)
    queue_owner_identity: Mapping[str, Any]
    scope_binding: Mapping[str, str]
    repository_id: str
    target_branch: str
    recovery_scope_cid: str
    request_id: str

    @classmethod
    def from_response(cls, payload, *, request, peer_pid, peer_uid):
        from ..merge.owner_recovery_runtime import recovery_scope_cid

        _closed(request, REQUEST_FIELDS)
        _closed(
            payload,
            {
                "schema",
                "ok",
                "request_id",
                "task",
                "task_owner_identity",
                "queue",
                "recovery",
                "queue_owner_identity",
                "repository_id",
                "target_branch",
                "scope_binding",
                "recovery_scope_cid",
            },
        )
        if (
            payload["schema"] != RESPONSE_SCHEMA
            or payload["ok"] is not True
            or payload["request_id"] != request["request_id"]
        ):
            raise StateOwnerBootstrapError(
                "native merge bootstrap was denied or replayed for another request"
            )
        birth = read_process_birth(peer_pid)
        if peer_uid != os.geteuid() or birth is None:
            raise StateOwnerBootstrapError(
                "native merge bootstrap owner peer is unavailable"
            )
        for key in ("task_owner_identity", "queue_owner_identity"):
            identity = payload[key]
            if (
                type(identity) is not dict
                or identity.get("process_birth") != birth.to_dict()
                or identity.get("process_birth_id") != process_birth_id(birth)
            ):
                raise StateOwnerBootstrapError(
                    "native bundle owner differs from its kernel peer"
                )
        task_identity, queue_identity = (
            payload["task_owner_identity"],
            payload["queue_owner_identity"],
        )
        if any(
            task_identity.get(key) == queue_identity.get(key)
            for key in ("server_id", "store_id", "database_uuid", "listen_uri")
        ):
            raise StateOwnerBootstrapError(
                "native queue role must be distinct from the task owner"
            )
        task = StateOwnerBootstrapCredentials.from_response(
            payload["task"],
            client_id=request["client_id"],
            store_id=request["store_id"],
            expected_process_birth_id=request["process_birth_id"],
        )
        if (
            task.server_id != task_identity.get("server_id")
            or task.store_id != task_identity.get("store_id")
            or task.endpoint != task_identity.get("listen_uri")
        ):
            raise StateOwnerBootstrapError("task credential owner differs from bundle")
        scope = _closed(
            payload["scope_binding"],
            {"board_namespace", "config_cid", "plan_cid", "lane_id", "attempt_root"},
        )
        scope = {key: _text(value) for key, value in scope.items()}
        if (
            scope["config_cid"] != request["config_cid"]
            or scope["plan_cid"] != request["plan_cid"]
        ):
            raise StateOwnerBootstrapError(
                "native bundle differs from the admitted launch source"
            )
        repository, target = (
            _text(payload["repository_id"]),
            _text(payload["target_branch"]),
        )
        expected_scope = recovery_scope_cid(
            store_id=queue_identity["store_id"],
            repository_id=repository,
            target_branch=target,
            scope_binding=scope,
        )
        if expected_scope != payload["recovery_scope_cid"]:
            raise StateOwnerBootstrapError("native bundle recovery namespace differs")
        queue = MergeRoleCredential.parse(
            payload["queue"],
            owner_identity=queue_identity,
            client_id=request["client_id"],
            birth_id=request["process_birth_id"],
        )
        recovery = MergeRoleCredential.parse(
            payload["recovery"],
            owner_identity=queue_identity,
            client_id=request["client_id"],
            birth_id=request["process_birth_id"],
        )
        if (
            queue.socket_path != recovery.socket_path
            or queue.socket_path == task.socket_path
            or len({task.token, queue.token, recovery.token}) != 3
        ):
            raise StateOwnerBootstrapError(
                "native paired credentials do not have distinct exact roles"
            )
        return cls(
            task,
            queue,
            recovery,
            dict(queue_identity),
            dict(scope),
            repository,
            target,
            expected_scope,
            request["request_id"],
        )

    def attach_merge_runtime(
        self,
        *,
        repository_root: Path,
        attempt_root: Path,
        board_namespace: str,
        lane_id: str,
        admitted_config_cid: str,
        admitted_plan_cid: str,
    ):
        from ..merge.owner_merge_queue import OwnerMergeQueueClient
        from ..merge.owner_merge_queue_adapter import OwnerMergeQueueAdapter
        from ..merge.owner_recovery_runtime import OwnerRecoveryRuntimeClient
        from ..merge.owner_recovery_adapter import OwnerMergeRecoveryRuntime
        from .typed_state_owner import TypedStateOwnerConnection

        expected = {
            "attempt_root": str(Path(attempt_root).absolute()),
            "board_namespace": board_namespace,
            "lane_id": lane_id,
            "config_cid": admitted_config_cid,
            "plan_cid": admitted_plan_cid,
        }
        if dict(self.scope_binding) != expected:
            raise StateOwnerBootstrapError(
                "current factory scope differs from the native bundle"
            )
        connections = []
        try:
            for credential in (self.queue, self.recovery):
                connection = TypedStateOwnerConnection(
                    socket_path=credential.socket_path,
                    token=credential.token,
                    client_id=credential.client_id,
                    process_birth_id=credential.process_birth_id,
                    store_id=credential.store_id,
                )
                connections.append(connection)
                if dict(connection.identity) != dict(self.queue_owner_identity):
                    raise StateOwnerBootstrapError(
                        "attached queue owner differs from the admitted bundle"
                    )
            queue = OwnerMergeQueueAdapter(
                OwnerMergeQueueClient(
                    connections[0],
                    repository_id=self.repository_id,
                    target_branch=self.target_branch,
                    consumer_id=self.queue.client_id,
                )
            )
            api = OwnerRecoveryRuntimeClient(
                connections[1],
                repository_id=self.repository_id,
                target_branch=self.target_branch,
                consumer_id=self.recovery.client_id,
                recovery_scope_cid=self.recovery_scope_cid,
            )
            runtime = OwnerMergeRecoveryRuntime(
                queue,
                api,
                repository_root=repository_root,
                attempt_root=attempt_root,
                board_namespace=board_namespace,
                config_cid=admitted_config_cid,
                plan_cid=admitted_plan_cid,
                lane_id=lane_id,
            )
            return AttachedOwnerMergeRuntime(runtime, tuple(connections))
        except BaseException:
            for connection in reversed(connections):
                connection.close()
            raise


@dataclass
class AttachedOwnerMergeRuntime:
    runtime: Any
    connections: tuple[Any, ...] = field(repr=False)

    def close(self):
        # Closing local channels never releases an unknown callback's lease.
        for connection in reversed(self.connections):
            connection.close()


def request_owner_merge_bootstrap(
    descriptor: int,
    *,
    client_id: str,
    store_id: str,
    config_cid: str,
    plan_cid: str,
    timeout_seconds: float = 30.0,
):
    birth = read_process_birth(os.getpid())
    if birth is None:
        raise StateOwnerBootstrapError(
            "daemon birth unavailable for paired owner bootstrap"
        )
    request = {
        "schema": REQUEST_SCHEMA,
        "request_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "process_birth": birth.to_dict(),
        "process_birth_id": process_birth_id(birth),
        "client_id": _text(client_id),
        "store_id": _text(store_id),
        "config_cid": _text(config_cid),
        "plan_cid": _text(plan_cid),
    }
    validate_state_owner_bootstrap_listener(descriptor)
    try:
        # One exact request may be retried after a lost response; the broker must
        # return only its still-current exact grant bundle, never create a lease.
        for attempt in range(2):
            channel = None
            try:
                channel = _connect_inherited_listener(
                    os.dup(descriptor), timeout_seconds=max(0.1, timeout_seconds / 2)
                )
                peer_pid, peer_uid, _ = struct.unpack(
                    "3i",
                    channel.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                    ),
                )
                _send_frame(channel, request)
                response = receive_bundle_frame(channel)
                break
            except (OSError, StateOwnerBootstrapError):
                if attempt:
                    raise StateOwnerBootstrapError(
                        "native paired bootstrap transport unavailable"
                    ) from None
            finally:
                if channel is not None:
                    channel.close()
        return OwnerMergeBootstrapBundle.from_response(
            response, request=request, peer_pid=peer_pid, peer_uid=peer_uid
        )
    finally:
        os.close(descriptor)
