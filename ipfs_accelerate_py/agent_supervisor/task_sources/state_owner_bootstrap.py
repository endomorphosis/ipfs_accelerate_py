"""Private inherited-channel bootstrap for PID-bound state-owner clients.

The typed state owner binds grants to the kernel peer PID and process birth.
Long-running supervisors cannot therefore place one reusable token in their
environment and pass it to replacement daemons.  This module supplies the
small, closed bootstrap used by a managed daemon to identify its *current*
birth over an inherited Unix socket and receive that birth's token.

Only the descriptor number is permitted in argv.  The token and owner socket
path travel in the framed response and are installed after the descriptor is
closed.  Provider subprocess environments are scrubbed independently by the
normal database-program boundary.
"""

from __future__ import annotations

import json
import os
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ..merge.database_worktree_registry import process_birth_id
from ..merge.worktree_lifecycle import read_process_birth
from .control_plane_contracts import canonical_json_bytes
from .task_execution_route_policy import TaskExecutionRoutePolicy
from .typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
)

STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-owner-bootstrap-request@1"
)
STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-owner-bootstrap-response@1"
)
MAX_STATE_OWNER_BOOTSTRAP_BYTES: Final = 65_536


class StateOwnerBootstrapError(RuntimeError):
    """The private state-owner credential bootstrap failed closed."""


def _send_frame(channel: socket.socket, payload: Mapping[str, Any]) -> None:
    body = canonical_json_bytes(dict(payload))
    if len(body) < 2 or len(body) > MAX_STATE_OWNER_BOOTSTRAP_BYTES:
        raise StateOwnerBootstrapError("state-owner bootstrap frame exceeds its bound")
    channel.sendall(len(body).to_bytes(4, "big") + body)


def _receive_exact(channel: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(length)
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise StateOwnerBootstrapError("state-owner bootstrap channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_frame(channel: socket.socket) -> dict[str, Any]:
    size = int.from_bytes(_receive_exact(channel, 4), "big")
    if size < 2 or size > MAX_STATE_OWNER_BOOTSTRAP_BYTES:
        raise StateOwnerBootstrapError("state-owner bootstrap frame size is invalid")
    try:
        payload = json.loads(_receive_exact(channel, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateOwnerBootstrapError("state-owner bootstrap frame is malformed") from exc
    if not isinstance(payload, dict):
        raise StateOwnerBootstrapError("state-owner bootstrap frame must be an object")
    return payload


def _connect_inherited_listener(
    descriptor: int, *, timeout_seconds: float
) -> socket.socket:
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 3:
        raise StateOwnerBootstrapError("state-owner bootstrap descriptor is invalid")
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise StateOwnerBootstrapError(
            "state-owner bootstrap descriptor is unavailable"
        ) from exc
    if not stat.S_ISSOCK(metadata.st_mode):
        raise StateOwnerBootstrapError(
            "state-owner bootstrap requires an inherited Unix socket"
        )
    listener: socket.socket | None = None
    try:
        listener = socket.socket(fileno=descriptor)
        if listener.family != socket.AF_UNIX or (
            listener.type & socket.SOCK_STREAM
        ) != socket.SOCK_STREAM:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap descriptor has the wrong socket type"
            )
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap descriptor is not a rendezvous listener"
            )
        address = listener.getsockname()
        if not isinstance(address, (str, bytes)) or not address:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap rendezvous identity is unavailable"
            )
        # Close this daemon's inherited copy before making a fresh connection.
        # The fresh connect is what gives the owner an authoritative
        # SO_PEERCRED identity for this exact process birth.
        listener.close()
        listener = None
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(max(0.1, float(timeout_seconds)))
        channel.connect(address)
    except StateOwnerBootstrapError:
        if listener is not None:
            listener.close()
        raise
    except (OSError, TypeError, ValueError) as exc:
        if listener is not None:
            listener.close()
        raise StateOwnerBootstrapError(
            "state-owner bootstrap socket could not be opened"
        ) from exc
    return channel


def validate_state_owner_bootstrap_listener(descriptor: int) -> str | bytes:
    """Validate an inherited rendezvous listener without consuming its FD.

    Launch parents use this before forwarding the capability through another
    process boundary.  Validation operates on a duplicate so the descriptor
    retained for the eventual daemon remains open and keeps its original
    number.
    """

    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 3:
        raise StateOwnerBootstrapError("state-owner bootstrap descriptor is invalid")
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise StateOwnerBootstrapError(
            "state-owner bootstrap descriptor is unavailable"
        ) from exc
    if not stat.S_ISSOCK(metadata.st_mode):
        raise StateOwnerBootstrapError(
            "state-owner bootstrap requires an inherited Unix socket"
        )
    duplicate = -1
    listener: socket.socket | None = None
    try:
        duplicate = os.dup(descriptor)
        listener = socket.socket(fileno=duplicate)
        duplicate = -1
        if listener.family != socket.AF_UNIX or (
            listener.type & socket.SOCK_STREAM
        ) != socket.SOCK_STREAM:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap descriptor has the wrong socket type"
            )
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap descriptor is not a rendezvous listener"
            )
        address = listener.getsockname()
        if not isinstance(address, (str, bytes)) or not address:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap rendezvous identity is unavailable"
            )
        return address
    except StateOwnerBootstrapError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StateOwnerBootstrapError(
            "state-owner bootstrap listener could not be validated"
        ) from exc
    finally:
        if listener is not None:
            listener.close()
        elif duplicate >= 0:
            os.close(duplicate)


@dataclass(frozen=True)
class StateOwnerBootstrapCredentials:
    """One daemon-birth-bound credential received from its state owner."""

    endpoint: str
    socket_path: str
    store_id: str
    server_id: str
    client_id: str
    process_birth_id: str
    token: str
    execution_route_policy: TaskExecutionRoutePolicy

    @classmethod
    def from_response(
        cls,
        payload: Mapping[str, Any],
        *,
        client_id: str,
        store_id: str,
        expected_process_birth_id: str,
    ) -> StateOwnerBootstrapCredentials:
        fields = {
            "schema",
            "ok",
            "endpoint",
            "socket_path",
            "store_id",
            "server_id",
            "client_id",
            "process_birth_id",
            "token",
            "execution_route_policy",
        }
        if set(payload) != fields:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap response fields differ from the closed schema"
            )
        if (
            payload.get("schema") != STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA
            or payload.get("ok") is not True
        ):
            raise StateOwnerBootstrapError("state-owner bootstrap was denied")
        values = {
            name: str(payload.get(name) or "").strip()
            for name in fields - {"schema", "ok", "execution_route_policy"}
        }
        if any(not value or len(value) > 4_096 for value in values.values()):
            raise StateOwnerBootstrapError("state-owner bootstrap identity is invalid")
        if len(values["token"]) < 16:
            raise StateOwnerBootstrapError("state-owner bootstrap token is unavailable")
        if (
            values["client_id"] != client_id
            or values["store_id"] != store_id
            or values["process_birth_id"] != expected_process_birth_id
        ):
            raise StateOwnerBootstrapError(
                "state-owner bootstrap response differs from the requesting birth"
            )
        socket_path = os.path.abspath(values["socket_path"])
        if socket_path != values["socket_path"]:
            raise StateOwnerBootstrapError("state-owner socket path is not absolute")
        raw_policy = payload.get("execution_route_policy")
        if not isinstance(raw_policy, Mapping):
            raise StateOwnerBootstrapError(
                "state-owner bootstrap execution route policy is unavailable"
            )
        try:
            execution_route_policy = TaskExecutionRoutePolicy.from_dict(raw_policy)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise StateOwnerBootstrapError(
                "state-owner bootstrap execution route policy is invalid"
            ) from exc
        return cls(
            endpoint=values["endpoint"],
            socket_path=socket_path,
            store_id=values["store_id"],
            server_id=values["server_id"],
            client_id=values["client_id"],
            process_birth_id=values["process_birth_id"],
            token=values["token"],
            execution_route_policy=execution_route_policy,
        )

    def install_environment(self) -> dict[str, str | bool]:
        """Install the two transport values and return only redacted metadata."""

        os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = self.token
        os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = self.socket_path
        return {
            "endpoint": self.endpoint,
            "store_id": self.store_id,
            "server_id": self.server_id,
            "client_id": self.client_id,
            "process_birth_id": self.process_birth_id,
            "execution_route_policy_id": self.execution_route_policy.policy_id,
            "execution_route_plan_root_cid": (
                self.execution_route_policy.plan_root_cid
            ),
            "execution_route_policy": (
                self.execution_route_policy.public_summary()
            ),
            "credential_transport": "private_inherited_socket",
            "credential_in_argv": False,
        }


def request_state_owner_bootstrap(
    descriptor: int,
    *,
    client_id: str,
    store_id: str,
    timeout_seconds: float = 30.0,
) -> StateOwnerBootstrapCredentials:
    """Request one credential for the calling process's exact kernel birth."""

    selected_client = str(client_id or "").strip()
    selected_store = str(store_id or "").strip()
    if not selected_client or not selected_store:
        raise StateOwnerBootstrapError("state-owner bootstrap scope is incomplete")
    birth = read_process_birth(os.getpid())
    if birth is None:
        raise StateOwnerBootstrapError("state-owner bootstrap process birth is unavailable")
    birth_id = process_birth_id(birth)
    channel = _connect_inherited_listener(
        descriptor,
        timeout_seconds=timeout_seconds,
    )
    try:
        _send_frame(
            channel,
            {
                "schema": STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
                "pid": os.getpid(),
                "process_birth": birth.to_dict(),
                "process_birth_id": birth_id,
                "client_id": selected_client,
                "store_id": selected_store,
            },
        )
        response = _receive_frame(channel)
    except (OSError, TimeoutError) as exc:
        raise StateOwnerBootstrapError(
            "state-owner bootstrap transport failed"
        ) from exc
    finally:
        channel.close()
    return StateOwnerBootstrapCredentials.from_response(
        response,
        client_id=selected_client,
        store_id=selected_store,
        expected_process_birth_id=birth_id,
    )


__all__ = [
    "MAX_STATE_OWNER_BOOTSTRAP_BYTES",
    "STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA",
    "STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA",
    "StateOwnerBootstrapCredentials",
    "StateOwnerBootstrapError",
    "_receive_frame",
    "_send_frame",
    "request_state_owner_bootstrap",
    "validate_state_owner_bootstrap_listener",
]
