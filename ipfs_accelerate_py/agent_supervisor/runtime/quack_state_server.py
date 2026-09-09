"""Loopback Quack state-owner service (DQP-006).

Interfaces: ``QuackStateServer@1``, ``StateServerIdentity@1``

One long-lived process exclusively owns ``control.duckdb``, admits the pinned
DuckDB/Quack capability profile, applies migrations before readiness, starts
Quack on an allocated loopback port, publishes database/schema/server/
process-birth identity, checkpoints cleanly, and stops through a fenced
control path.

Security invariants enforced here:

* Auth tokens never appear in argv, logs, status, exports, or provider
  environments — only opaque secret handles are published.
* A second concurrent owner fails closed.
* Ready requires a live identity query plus matching store, generation,
  schema, and server identities.
* Non-loopback binds require a separately reviewed remote policy that is
  unavailable by default.
"""

from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import socket
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, NoReturn, Protocol

from ..federation.event_wait import EventSource, StateOwnerEventWait
from ..federation.events import EventBatch, EventWaitRequest
from ..federation.outbox_worker import (
    EventDrivenOutboxWorker,
    StateOwnerOutboxWake,
)
from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
    owner_liveness,
)
from ..task_sources.control_plane_contracts import (
    CONTRACT_VERSION,
    REDACTION_MARKER,
    SECRET_HANDLE_PREFIXES,
    ControlPlaneStoreIdentity,
    SecretHandle,
    StateAuthorityClass,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
    is_secret_handle,
    redact_mapping,
)
from ..task_sources.control_plane_migrations import (
    META_DATABASE_UUID,
    META_SCHEMA_FINGERPRINT,
    META_SCHEMA_VERSION,
    MigrationRunReport,
    compute_schema_fingerprint,
    duckdb_available,
)
from ..task_sources.control_plane_schema import (
    CONTROL_PLANE_SCHEMA_REVISION,
    install_control_plane_schema,
)
from ..task_sources.database_task_source import (
    execute_quack_owner_command,
    quack_owner_command_error_code,
)
from ..task_sources.duckdb_state import (
    DEFAULT_MEMORY_LIMIT,
    DUCKDB_CONNECTION_POLICY_SETTINGS,
    QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
    QUACK_MUTATION_DOMAIN_EVENT_INSERT,
    QUACK_MUTATION_EVIDENCE_DELETE,
    QUACK_MUTATION_EVIDENCE_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
    QUACK_MUTATION_QUEUE_BACKOFF,
    QUACK_MUTATION_TASK_REVISION_INSERT,
    QUACK_MUTATION_TASK_STATUS_CAS,
    QUACK_MUTATION_TASK_STATUS_TRANSITION,
    QUACK_MUTATION_VALIDATION_RECORD,
    QUACK_MUTATION_VALIDATION_RESULT_INSERT,
    QUACK_MUTATION_VALIDATION_RUN_INSERT,
    QUACK_OWNER_COMMAND_MAX_BYTES,
    QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
    QUACK_OWNER_MUTATION_MAX_CLOCK_SKEW_MS,
    QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES,
    QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES,
    QUACK_OWNER_MUTATION_MAX_STEPS,
    QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
    QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
    QUACK_OWNER_MUTATION_REQUEST_TTL_MS,
    QUACK_OWNER_MUTATION_RESULT_SCHEMA,
    DuckDBConnection,
    open_duckdb_connection,
    open_quack_state_owner_connection,
    quack_owner_command_response,
    quack_owner_mutation_content_id,
    quack_owner_mutation_inbox_path,
    quack_owner_mutation_mac,
    validate_quack_owner_command_request,
)
from ..task_sources.intent_repository import (
    COMPLETION_EVIDENCE_SCHEMA,
    DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    QUEUE_ENTRY_SCHEMA,
    IntentRepository,
    missing_current_evidence_on,
)
from ..task_sources.quack_capabilities import (
    QuackCapabilityReport,
    probe_quack_capabilities,
)
from ..task_sources.quack_owner_mutation import (
    MAX_MUTATION_REQUEST_BYTES,
    MAX_MUTATION_RESULT_ROWS,
    QuackOwnerMutationEnvelopeError,
    build_mutation_result,
    mutation_envelope_exists_at,
    open_mutation_inbox_directory,
    parse_mutation_request,
    parse_mutation_result,
    read_envelope_at,
    unlink_mutation_envelope_at,
    write_envelope_atomic_at,
)
from ..task_sources.typed_state_owner import (
    DATABASE_TASK_COMMAND_GRANT_TTL_SECONDS,
    DATABASE_TASK_COMMANDS,
    HASH_OBSERVATION_SERVICE_OPERATION,
    MAX_GRANT_BROKER_FRAME_BYTES,
    TYPED_STATE_OWNER_CREDENTIAL_DATABASE_TASK_COMMAND,
    TYPED_STATE_OWNER_CREDENTIAL_HASH_OBSERVATION,
    TYPED_STATE_OWNER_CREDENTIAL_READ_TRANSPORT,
    TYPED_STATE_OWNER_GRANT_BROKER_CREDENTIAL_KINDS,
    TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
    TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_SOCKET_FILENAME,
    TYPED_STATE_OWNER_TOKEN_FILENAME,
    OwnerClientGrant,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerDatabaseTaskCommandError,
    TypedStateOwnerGateway,
    _kernel_peer_identity,
    kernel_process_birth_id,
)

_UTC: Final = timezone.utc  # noqa: UP017 - Python 3.8 compatibility.

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

QUACK_STATE_SERVER_INTERFACE: Final = "QuackStateServer@1"
STATE_SERVER_IDENTITY_INTERFACE: Final = "StateServerIdentity@1"
QUACK_STATE_SERVER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-state-server@1"
)
STATE_SERVER_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-server-identity@1"
)
REMOTE_BIND_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/remote-bind-policy@1"
)
OWNER_MARKER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-owner-marker@1"
)
QUACK_STATE_SERVER_VERSION: Final[int] = 1
QUACK_ISOLATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-isolation-receipt@4"
)

DEFAULT_LOOPBACK_HOST: Final = "127.0.0.1"
DEFAULT_STORE_ID: Final = "control.duckdb"
DEFAULT_SECRET_HANDLE_PREFIX: Final = "handle:quack-token"
TOKEN_FILENAME_SUFFIX: Final = ".quack-token"
OWNER_MARKER_SUFFIX: Final = ".state-owner.json"
OWNER_LOCK_SUFFIX: Final = ".state-owner.lock"
OWNER_MARKER_MAX_BYTES: Final[int] = 64 * 1024
STATUS_FILENAME: Final = "quack-state-server.status.json"
CONTROL_STOP_FILENAME: Final = "quack-state-server.stop"
CONTROL_STOP_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-stop-request@1"
)
CONTROL_STOP_MAX_BYTES: Final[int] = 64 * 1024
CONTROL_STOP_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "server_id", "fence_token", "requested_at"}
)
READ_REPLICA_NAME_INFIX: Final = ".read-replica"
READ_REPLICA_MAX_BYTES: Final[int] = 8 * 1024 * 1024 * 1024
READ_REPLICA_COPY_CHUNK_BYTES: Final[int] = 1024 * 1024
READ_REPLICA_COPY_TIMEOUT_SECONDS: Final[float] = 30.0
READ_REPLICA_STOP_TIMEOUT_SECONDS: Final[float] = 2.0
MUTATION_INBOX_DIRNAME: Final = "mutations"
MAX_MUTATIONS_PER_POLL: Final[int] = 64
MUTATION_REQUEST_NAME: Final[re.Pattern[str]] = re.compile(
    r"^(?P<request_id>b[a-z2-7]{40,127})\.request\.json$"
)
MUTATION_PROCESSING_NAME: Final[re.Pattern[str]] = re.compile(
    r"^(?P<request_id>b[a-z2-7]{40,127})\.processing\.json$"
)
MUTATION_MAX_DIRECTORY_ENTRIES: Final[int] = 4_096
MUTATION_MAX_PER_PASS: Final[int] = 32
TYPED_OWNER_COMMAND_REQUEST_NAME: Final[re.Pattern[str]] = re.compile(
    r"^(?P<request_id>[0-9a-f]{32})\.request\.json$"
)
_MUTATION_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "protocol_revision",
        "request_id",
        "issued_at_ms",
        "expires_at_ms",
        "operation",
        "binding",
        "steps",
        "request_cid",
        "auth_mac",
    }
)
_MUTATION_READY_FROM: Final[frozenset[str]] = frozenset(
    {
        "todo",
        "ready",
        "open",
        "pending",
        "queued",
        "proposed",
        "admitted",
        "retrying",
    }
)
_MUTATION_ALLOWED_TO: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        **{
            status: frozenset({"in_progress"})
            for status in _MUTATION_READY_FROM - {"retrying"}
        },
        "retrying": frozenset({"in_progress", "blocked"}),
        "claimed": frozenset({"in_progress", "ready", "blocked"}),
        "running": frozenset({"ready", "completed", "blocked", "retrying"}),
        "in_progress": frozenset({"ready", "completed", "blocked", "retrying"}),
        "blocked": frozenset({"retrying", "ready"}),
    }
)

_MUTATION_SQL_TEMPLATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        QUACK_MUTATION_TASK_STATUS_CAS: (
            "UPDATE tasks SET status = ?, revision = ?, updated_at = ?, "
            "body_json = ? WHERE task_cid = ? AND revision = ?"
        ),
        QUACK_MUTATION_TASK_REVISION_INSERT: (
            "INSERT INTO task_revisions "
            "(task_cid, revision, status, body_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_COMPLETION_RECEIPT_INSERT: (
            "INSERT INTO completion_receipts "
            "(receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, "
            "fencing_token, completed_at, validation_run_id, "
            "evidence_digest, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_DOMAIN_EVENT_INSERT: (
            "INSERT INTO domain_events "
            "(event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_VALIDATION_RUN_INSERT: (
            "INSERT INTO validation_runs "
            "(run_id, task_cid, attempt_id, started_at, finished_at, status, "
            "command_digest, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_VALIDATION_RESULT_INSERT: (
            "INSERT INTO validation_results "
            "(result_id, run_id, task_cid, ordinal, outcome, "
            "evidence_digest, body_json) VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_EVIDENCE_DELETE: (
            "DELETE FROM evidence_nodes WHERE evidence_id = ?"
        ),
        QUACK_MUTATION_EVIDENCE_INSERT: (
            "INSERT INTO evidence_nodes "
            "(evidence_id, parent_evidence_id, task_cid, evidence_kind, "
            "digest, created_at, body_json) VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT: (
            "INSERT INTO leases "
            "(task_cid, claim_cid, resolution_cid, claimant_did, "
            "logical_epoch, fencing_token, expires_at_ms, attempt, "
            "state, started_at_ms, release_reason, retry_not_before_ms, "
            "owner_session_id, fence_epoch, revision, extension_schema, "
            "extension_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE: (
            "UPDATE leases SET attempt = ?, retry_not_before_ms = ?, "
            "release_reason = ?, state = 'released', "
            "extension_schema = ?, extension_json = ?, "
            "revision = revision + 1 WHERE task_cid = ?"
        ),
    }
)

LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "127.0.0.1",
        "::1",
        "localhost",
        "ip6-localhost",
    }
)

# Keys that must never appear with secret values in published surfaces.
_TOKEN_BEARING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "quack_token",
        "secret",
        "token",
        "token_bytes",
        "token_value",
    }
)

_PROVIDER_ENV_DENY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "API_KEY",
    "APIKEY",
    "AUTHORIZATION",
    "BEARER",
    "QUACK_AUTH",
)
_PROVIDER_ENV_DENY_NAMES: Final[frozenset[str]] = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
    }
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QuackStateServerError(RuntimeError):
    """Base fail-closed error for the state-owner service."""


class QuackStateServerOwnershipError(QuackStateServerError):
    """Another live owner holds exclusive database ownership."""


class QuackStateServerBindError(QuackStateServerError):
    """Bind address is not admitted by policy."""


class QuackStateServerCapabilityError(QuackStateServerError):
    """DuckDB/Quack capability admission failed."""


class QuackStateServerMigrationError(QuackStateServerError):
    """Schema migration failed or is incomplete before readiness."""


class QuackStateServerReadyError(QuackStateServerError):
    """Server is not ready: live query or identity match failed."""


class QuackStateServerTokenError(QuackStateServerError):
    """Token material would leak or cannot be stored safely."""


class QuackStateServerControlError(QuackStateServerError):
    """Fenced control-path command failed."""


class QuackStateServerNotRunningError(QuackStateServerError):
    """Operation requires a started state-owner."""


class QuackStateServerMutationError(QuackStateServerError):
    """A bounded owner-side mutation request is invalid or inadmissible."""

    def __init__(
        self, code: str, *, observed: Mapping[str, Any] | None = None
    ) -> None:
        self.code = str(code or "mutation_rejected")
        self.observed = dict(observed or {})
        super().__init__(self.code)


class QuackStateServerIsolationError(QuackStateServerError):
    """The external-access owner mode lacks an admitted isolation receipt."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(moment: datetime | None = None) -> str:
    value = moment or datetime.now(_UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return (
        value.astimezone(_UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _schema_fingerprint_digest(value: str) -> str:
    """Bridge canonical CID fingerprints to SHA-256, rejecting other forms."""

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("sha256:"):
        digest = text.removeprefix("sha256:")
        if len(digest) == 64:
            try:
                bytes.fromhex(digest)
            except ValueError:
                pass
            else:
                return f"sha256:{digest.lower()}"
    if text.startswith("b"):
        try:
            encoded = text[1:].upper()
            encoded += "=" * ((8 - len(encoded) % 8) % 8)
            raw = base64.b32decode(encoded)
        except (ValueError, TypeError):
            raw = b""
        prefix = b"\x01\xa9\x02\x12\x20"
        if raw.startswith(prefix) and len(raw) == len(prefix) + 32:
            return f"sha256:{raw[len(prefix):].hex()}"
    return ""


def _is_loopback_host(host: str) -> bool:
    text = str(host or "").strip().lower()
    if not text:
        return False
    if text in LOOPBACK_HOSTS:
        return True
    # Accept IPv4 mapped loopback and trailing interface specs without colon
    # ambiguity for bare IPv4.
    if text.startswith("127."):
        return True
    try:
        info = socket.getaddrinfo(text, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, _type, _proto, _canon, sockaddr in info:
        address = sockaddr[0]
        try:
            packed = socket.inet_pton(
                family if family in (socket.AF_INET, socket.AF_INET6) else socket.AF_INET,
                address,
            )
        except (OSError, ValueError):
            continue
        if family == socket.AF_INET and packed[0] == 127:
            return True
        if family == socket.AF_INET6 and packed == b"\x00" * 15 + b"\x01":
            return True
    return False


def _allocate_loopback_port(host: str = DEFAULT_LOOPBACK_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host if _is_loopback_host(host) else DEFAULT_LOOPBACK_HOST, 0))
        return int(sock.getsockname()[1])


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None  # ownership transferred to the file object
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    text = json.dumps(dict(payload), sort_keys=True, indent=2, separators=(",", ": "))
    _atomic_write_text(path, text + "\n", mode=mode)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_stop_control(path: Path) -> dict[str, str] | None:
    """Read one exact private stop request without treating corruption as absence."""

    required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise QuackStateServerControlError(
            "secure stop control read flags are unavailable"
        )
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise QuackStateServerControlError(
            "stop control path is unavailable or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > CONTROL_STOP_MAX_BYTES
        ):
            raise QuackStateServerControlError(
                "stop control must be an owned private single-link regular file"
            )
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16 * 1024))
            if not chunk:
                raise QuackStateServerControlError(
                    "stop control was truncated while read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise QuackStateServerControlError(
                "stop control grew while read"
            )
        after = os.fstat(descriptor)
        try:
            named = os.lstat(path)
        except OSError as exc:
            raise QuackStateServerControlError(
                "stop control name changed while read"
            ) from exc
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(named):
        raise QuackStateServerControlError(
            "stop control inode or metadata changed while read"
        )
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_mutation_duplicate_guard,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise QuackStateServerControlError(
            "stop control content is invalid"
        ) from exc
    if (
        type(payload) is not dict
        or set(payload) != CONTROL_STOP_REQUEST_FIELDS
        or payload.get("schema") != CONTROL_STOP_REQUEST_SCHEMA
        or any(
            type(payload.get(field)) is not str or not payload[field]
            for field in ("server_id", "fence_token", "requested_at")
        )
        or re.fullmatch(r"[0-9a-f]{32}", payload["fence_token"]) is None
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            payload["requested_at"],
        )
        is None
    ):
        raise QuackStateServerControlError(
            "stop control content is invalid"
        )
    return {
        "schema": payload["schema"],
        "server_id": payload["server_id"],
        "fence_token": payload["fence_token"],
        "requested_at": payload["requested_at"],
    }


def _mutation_duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _read_bounded_canonical_json(path: Path) -> dict[str, Any]:
    """Read one regular, non-symlink, canonical bounded JSON object."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise QuackStateServerMutationError("request_not_regular")
        if info.st_size > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
            raise QuackStateServerMutationError("request_too_large")
        raw = os.read(descriptor, QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
        raise QuackStateServerMutationError("request_too_large")
    try:
        payload = json.loads(raw, object_pairs_hook=_mutation_duplicate_guard)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise QuackStateServerMutationError("request_not_canonical_json") from exc
    if not isinstance(payload, dict):
        raise QuackStateServerMutationError("request_not_object")
    try:
        canonical = canonical_json_bytes(payload)
    except Exception as exc:
        raise QuackStateServerMutationError("request_not_canonical_json") from exc
    if canonical != raw.strip():
        raise QuackStateServerMutationError("request_not_canonical_json")
    return payload


def _mutation_parameters(step: Mapping[str, Any], count: int) -> list[Any]:
    if set(step) != {"template_id", "parameters"}:
        raise QuackStateServerMutationError("step_schema_invalid")
    parameters = step.get("parameters")
    if not isinstance(parameters, list) or len(parameters) != count:
        raise QuackStateServerMutationError("step_parameters_invalid")
    for value in parameters:
        if value is None:
            continue
        if type(value) is int and -(2**63) <= value < 2**63:
            continue
        if type(value) is str and len(value.encode("utf-8")) <= QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES:
            continue
        raise QuackStateServerMutationError("step_parameters_invalid")
    return parameters


def _canonical_object(text: object, *, code: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES:
        raise QuackStateServerMutationError(code)
    try:
        value = json.loads(text, object_pairs_hook=_mutation_duplicate_guard)
    except (json.JSONDecodeError, ValueError) as exc:
        raise QuackStateServerMutationError(code) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != text.encode("utf-8"):
        raise QuackStateServerMutationError(code)
    return value


def _contains_token_material(value: Any, token: str | None) -> bool:
    if not token:
        return False
    if isinstance(value, str):
        return token in value
    if isinstance(value, Mapping):
        return any(_contains_token_material(item, token) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_token_material(item, token) for item in value)
    return False


def _is_token_bearing_key(key: str) -> bool:
    """Return whether a key name is credential material (not a handle reference).

    Keys such as ``secret_handle`` and ``credential_generation`` are public
    references / counters and must not be wiped. Raw token fields are redacted.
    """

    lowered = key.lower().replace("-", "_").strip()
    if lowered in {
        "secret_handle",
        "credential_generation",
        "credential_id",
        "credentials_path",
    }:
        return False
    if lowered in _TOKEN_BEARING_KEYS:
        return True
    # Exact suffix/prefix forms that carry secret bytes.
    if lowered.endswith("_token") or lowered.startswith("token_"):
        return True
    if lowered.endswith("_secret") or lowered.startswith("secret_"):
        return True
    if lowered.endswith("_password") or lowered == "password":
        return True
    if lowered in {"credentials", "credential", "authorization", "bearer"}:
        return True
    return False


def _strip_token_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with token-bearing keys redacted (values never preserved)."""

    out: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if _is_token_bearing_key(key_text):
            # Preserve opaque secret handles only.
            if isinstance(value, str) and is_secret_handle(value):
                out[key_text] = value
            else:
                out[key_text] = REDACTION_MARKER
            continue
        if isinstance(value, Mapping):
            out[key_text] = _strip_token_keys(value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            out[key_text] = [
                _strip_token_keys(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            out[key_text] = value
    return out


def sanitize_for_export(payload: Mapping[str, Any], *, token: str | None = None) -> dict[str, Any]:
    """Redact secrets and refuse to emit raw token material."""

    redacted = redact_mapping(_strip_token_keys(payload))
    if not isinstance(redacted, dict):
        redacted = {"value": redacted}
    if _contains_token_material(redacted, token):
        raise QuackStateServerTokenError(
            "export surface would contain raw auth token material"
        )
    return redacted


def provider_safe_environment(
    base: Mapping[str, str] | None = None,
    *,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Project an environment safe for implementation-provider subprocesses.

    Never includes Quack tokens, secret handles with token-bearing names, or
    any variable whose name suggests credential material.
    """

    merged: dict[str, str] = {}
    for source in (base or {}, extra or {}):
        for key, value in source.items():
            name = str(key)
            upper = name.upper()
            if upper in _PROVIDER_ENV_DENY_NAMES or any(
                token in upper for token in _PROVIDER_ENV_DENY_SUBSTRINGS
            ):
                continue
            text = str(value)
            # Fail closed if a secret-handle-shaped value is smuggled under a
            # non-denied name that still looks credential-adjacent.
            if is_secret_handle(text) and any(
                part in upper for part in ("QUACK", "AUTH", "HANDLE")
            ):
                continue
            merged[name] = text
    return merged


def listen_uri(host: str, port: int) -> str:
    host_text = str(host).strip()
    if ":" in host_text and not host_text.startswith("["):
        return f"quack:[{host_text}]:{int(port)}"
    return f"quack:{host_text}:{int(port)}"


# ---------------------------------------------------------------------------
# Remote bind policy (unavailable by default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteBindPolicy:
    """Separately reviewed policy admitting non-loopback binds.

    No default instance is provided. Callers must construct an explicit policy
    with a non-empty review receipt before binding off loopback.
    """

    SCHEMA: ClassVar[str] = REMOTE_BIND_POLICY_SCHEMA

    policy_id: str
    reviewed_by: str
    review_receipt: str
    allowed_hosts: tuple[str, ...]
    require_tls: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id or "").strip()
        reviewed_by = str(self.reviewed_by or "").strip()
        receipt = str(self.review_receipt or "").strip()
        hosts = tuple(str(item).strip() for item in self.allowed_hosts if str(item).strip())
        if not policy_id:
            raise QuackStateServerBindError("remote bind policy_id is required")
        if not reviewed_by:
            raise QuackStateServerBindError("remote bind reviewed_by is required")
        if not receipt:
            raise QuackStateServerBindError(
                "remote bind review_receipt is required; policy unavailable by default"
            )
        if not hosts:
            raise QuackStateServerBindError(
                "remote bind policy must list at least one allowed host"
            )
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "reviewed_by", reviewed_by)
        object.__setattr__(self, "review_receipt", receipt)
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "notes", str(self.notes or ""))

    def admits(self, host: str) -> bool:
        candidate = str(host or "").strip().lower()
        return candidate in {item.lower() for item in self.allowed_hosts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "policy_id": self.policy_id,
            "reviewed_by": self.reviewed_by,
            "review_receipt": self.review_receipt,
            "allowed_hosts": list(self.allowed_hosts),
            "require_tls": bool(self.require_tls),
            "notes": self.notes,
        }


def assert_bind_admitted(
    host: str,
    *,
    remote_policy: RemoteBindPolicy | None = None,
) -> None:
    """Fail closed unless host is loopback or covered by a reviewed policy."""

    if _is_loopback_host(host):
        return
    if remote_policy is None:
        raise QuackStateServerBindError(
            f"non-loopback bind {host!r} requires a separately reviewed "
            "remote policy; the policy is unavailable by default"
        )
    if not remote_policy.admits(host):
        raise QuackStateServerBindError(
            f"host {host!r} is not admitted by remote policy "
            f"{remote_policy.policy_id!r}"
        )
    if remote_policy.require_tls:
        raise QuackStateServerBindError(
            "non-loopback Quack TLS is not implemented; an explicit reviewed "
            "plaintext policy is required"
        )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateServerIdentity:
    """Published identity of one Quack state-owner generation.

    Interface: ``StateServerIdentity@1``.
    """

    INTERFACE: ClassVar[str] = STATE_SERVER_IDENTITY_INTERFACE
    SCHEMA: ClassVar[str] = STATE_SERVER_IDENTITY_SCHEMA

    server_id: str
    store_id: str
    database_uuid: str
    schema_revision: int
    schema_fingerprint: str
    generation: int
    fence_epoch: int
    revision: int
    process_birth: ProcessBirthIdentity
    listen_uri: str
    extension_fingerprint: str
    credential_generation: int
    secret_handle: str
    repository_id: str = ""
    startup_epoch: int = 0
    started_at: str = ""
    status: str = "starting"

    def __post_init__(self) -> None:
        object.__setattr__(self, "server_id", str(self.server_id).strip())
        object.__setattr__(self, "store_id", str(self.store_id).strip())
        object.__setattr__(self, "database_uuid", str(self.database_uuid).strip())
        object.__setattr__(self, "schema_fingerprint", str(self.schema_fingerprint).strip())
        object.__setattr__(self, "listen_uri", str(self.listen_uri).strip())
        object.__setattr__(
            self, "extension_fingerprint", str(self.extension_fingerprint or "").strip()
        )
        object.__setattr__(self, "secret_handle", str(self.secret_handle).strip())
        object.__setattr__(self, "repository_id", str(self.repository_id or "").strip())
        object.__setattr__(self, "started_at", str(self.started_at or "").strip())
        object.__setattr__(self, "status", str(self.status or "starting").strip())
        if not self.server_id:
            raise ValueError("server_id is required")
        if not self.store_id:
            raise ValueError("store_id is required")
        if not self.database_uuid:
            raise ValueError("database_uuid is required")
        if not self.schema_fingerprint:
            raise ValueError("schema_fingerprint is required")
        if not self.listen_uri:
            raise ValueError("listen_uri is required")
        if not self.secret_handle or not is_secret_handle(self.secret_handle):
            raise QuackStateServerTokenError(
                "secret_handle must be an opaque handle, not raw token material"
            )
        if int(self.generation) < 1:
            raise ValueError("generation must be >= 1")
        if int(self.schema_revision) < 0:
            raise ValueError("schema_revision must be >= 0")
        if int(self.credential_generation) < 1:
            raise ValueError("credential_generation must be >= 1")
        if not isinstance(self.process_birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")

    @property
    def process_birth_id(self) -> str:
        birth = self.process_birth
        material = (
            f"{birth.pid}:{birth.start_time_ticks}:{birth.boot_id}:{birth.parent_pid}"
        )
        return f"birth:{_sha256_text(material)[7:39]}"

    def store_identity(self) -> ControlPlaneStoreIdentity:
        return ControlPlaneStoreIdentity(
            repository_id=self.repository_id or f"repository:{self.store_id}",
            database_uuid=self.database_uuid,
            store_id=self.store_id,
            schema_revision=int(self.schema_revision),
            generation=int(self.generation),
            schema_fingerprint=self.schema_fingerprint,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            server_birth_id=self.process_birth_id,
            extension_fingerprint=self.extension_fingerprint,
            metadata={
                "server_id": self.server_id,
                "listen_uri": self.listen_uri,
            },
        )

    def store_generation(self) -> StoreGeneration:
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(self.generation),
            schema_revision=int(self.schema_revision),
            fence_epoch=int(self.fence_epoch),
            revision=int(self.revision),
            database_uuid=self.database_uuid,
            birth_id=self.process_birth_id,
        )

    def matches(
        self,
        *,
        store_id: str | None = None,
        generation: int | None = None,
        schema_revision: int | None = None,
        schema_fingerprint: str | None = None,
        server_id: str | None = None,
        database_uuid: str | None = None,
        process_birth_id: str | None = None,
    ) -> bool:
        if store_id is not None and store_id != self.store_id:
            return False
        if generation is not None and int(generation) != int(self.generation):
            return False
        if schema_revision is not None and int(schema_revision) != int(
            self.schema_revision
        ):
            return False
        if schema_fingerprint is not None and schema_fingerprint != self.schema_fingerprint:
            return False
        if server_id is not None and server_id != self.server_id:
            return False
        if database_uuid is not None and database_uuid != self.database_uuid:
            return False
        if process_birth_id is not None and process_birth_id != self.process_birth_id:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Public identity projection — never includes raw token material."""

        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "server_id": self.server_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "schema_revision": int(self.schema_revision),
            "schema_fingerprint": self.schema_fingerprint,
            "generation": int(self.generation),
            "fence_epoch": int(self.fence_epoch),
            "revision": int(self.revision),
            "process_birth": self.process_birth.to_dict(),
            "process_birth_id": self.process_birth_id,
            "listen_uri": self.listen_uri,
            "extension_fingerprint": self.extension_fingerprint,
            "credential_generation": int(self.credential_generation),
            "secret_handle": self.secret_handle,
            "repository_id": self.repository_id,
            "startup_epoch": int(self.startup_epoch),
            "started_at": self.started_at,
            "status": self.status,
        }
        return sanitize_for_export(payload)

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def with_status(self, status: str) -> StateServerIdentity:
        return StateServerIdentity(
            server_id=self.server_id,
            store_id=self.store_id,
            database_uuid=self.database_uuid,
            schema_revision=self.schema_revision,
            schema_fingerprint=self.schema_fingerprint,
            generation=self.generation,
            fence_epoch=self.fence_epoch,
            revision=self.revision,
            process_birth=self.process_birth,
            listen_uri=self.listen_uri,
            extension_fingerprint=self.extension_fingerprint,
            credential_generation=self.credential_generation,
            secret_handle=self.secret_handle,
            repository_id=self.repository_id,
            startup_epoch=self.startup_epoch,
            started_at=self.started_at,
            status=status,
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuackStateServerConfig:
    """Static configuration for one state-owner instance."""

    database_path: Path
    state_dir: Path
    host: str = DEFAULT_LOOPBACK_HOST
    port: int = 0
    container_bind_host: str = ""
    container_port: int = 0
    repository_id: str = ""
    store_id: str = DEFAULT_STORE_ID
    allow_experimental: bool = False
    remote_bind_policy: RemoteBindPolicy | None = None
    application_version: str | None = None
    tool_version: str | None = None
    secret_handle: str = ""
    isolation_receipt_path: Path | None = None
    typed_command_socket_path_override: Path | None = None
    repository_root: Path | None = None

    def __post_init__(self) -> None:
        raw_root = self.repository_root
        repository_root: Path | None = None
        if raw_root is not None and str(raw_root).strip():
            root = Path(raw_root).expanduser()
            if not root.is_absolute():
                raise ValueError("repository_root must be an absolute path")
            repository_root = root.resolve()
        object.__setattr__(self, "repository_root", repository_root)

        def _sealed_path(value: Path, *, name: str) -> Path:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                if repository_root is None:
                    raise ValueError(
                        f"relative {name} requires an explicit repository_root"
                    )
                candidate = repository_root / candidate
            sealed = candidate.resolve()
            if repository_root is not None:
                try:
                    sealed.relative_to(repository_root)
                except ValueError as exc:
                    raise ValueError(
                        f"{name} escapes the explicit repository_root"
                    ) from exc
            return sealed

        object.__setattr__(
            self,
            "database_path",
            _sealed_path(self.database_path, name="database_path"),
        )
        object.__setattr__(
            self,
            "state_dir",
            _sealed_path(self.state_dir, name="state_dir"),
        )
        object.__setattr__(
            self,
            "isolation_receipt_path",
            None
            if self.isolation_receipt_path is None
            else Path(self.isolation_receipt_path),
        )
        socket_override = self.typed_command_socket_path_override
        if socket_override is not None:
            socket_override = Path(socket_override)
            if not socket_override.is_absolute():
                raise ValueError("typed command socket override must be absolute")
            if repository_root is not None:
                try:
                    socket_override.resolve().relative_to(repository_root)
                except ValueError as exc:
                    raise ValueError(
                        "typed command socket override escapes the explicit repository_root"
                    ) from exc
            object.__setattr__(
                self,
                "typed_command_socket_path_override",
                socket_override,
            )
        object.__setattr__(self, "host", str(self.host or DEFAULT_LOOPBACK_HOST).strip())
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(
            self,
            "container_bind_host",
            str(self.container_bind_host or self.host).strip(),
        )
        object.__setattr__(
            self, "container_port", int(self.container_port or self.port)
        )
        object.__setattr__(self, "repository_id", str(self.repository_id or "").strip())
        object.__setattr__(
            self, "store_id", str(self.store_id or DEFAULT_STORE_ID).strip()
        )
        handle = str(self.secret_handle or "").strip()
        if handle and not is_secret_handle(handle):
            raise QuackStateServerTokenError(
                "config secret_handle must be opaque handle, not raw token"
            )
        object.__setattr__(self, "secret_handle", handle)
        if self.port < 0 or self.port > 65535:
            raise ValueError("port must be in 0..65535")
        if self.container_port < 0 or self.container_port > 65535:
            raise ValueError("container_port must be in 0..65535")
        if self.port != self.container_port:
            raise ValueError("advertised and container ports must match exactly")
        if (
            self.container_bind_host != self.host
            and self.isolation_receipt_path is None
        ):
            raise QuackStateServerBindError(
                "distinct container bind requires an isolation receipt"
            )
        assert_bind_admitted(self.host, remote_policy=self.remote_bind_policy)

    def resolved_secret_handle(self, server_id: str, generation: int) -> str:
        if self.secret_handle:
            return self.secret_handle
        return f"{DEFAULT_SECRET_HANDLE_PREFIX}:{server_id}:g{int(generation)}"


def _mountinfo_entries() -> tuple[dict[str, Any], ...]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    entries: list[dict[str, Any]] = []
    for line in lines[:8_192]:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) <= separator + 2:
            continue
        mountpoint = (
            fields[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\134", "\\")
        )
        entries.append(
            {
                "target": str(Path(mountpoint).resolve()),
                "options": frozenset(fields[5].split(",")),
                "fstype": fields[separator + 1],
                "source": fields[separator + 2],
            }
        )
    return tuple(entries)


def _observe_quack_owner_isolation(
    config: QuackStateServerConfig, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Observe isolation from inside the prospective owner process."""

    marker = Path("/.dockerenv")
    marker_regular = False
    try:
        marker_regular = stat.S_ISREG(marker.stat().st_mode) and not marker.is_symlink()
    except OSError:
        pass
    try:
        hostname = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    except OSError:
        hostname = ""
    container_id = str(receipt.get("container_id") or "").strip()
    expected_hostname = str(receipt.get("container_hostname") or "").strip()
    hostname_matches = bool(hostname and hostname == expected_hostname)
    entries = _mountinfo_entries()
    by_target = {entry["target"]: entry for entry in entries}
    root = by_target.get("/")
    repository = str(Path(str(receipt.get("repository_path") or "")).resolve())
    repository_mount = by_target.get(repository)
    root_read_only = bool(root and "ro" in root["options"] and "rw" not in root["options"])
    repository_read_only = bool(
        repository_mount
        and "ro" in repository_mount["options"]
        and "rw" not in repository_mount["options"]
    )
    pseudo = {
        "proc", "sysfs", "tmpfs", "devtmpfs", "devpts", "cgroup", "cgroup2",
        "mqueue", "overlay", "squashfs", "securityfs", "pstore", "bpf",
        "tracefs", "configfs", "fusectl", "autofs", "hugetlbfs",
    }
    rw_host_bind_targets = sorted(
        entry["target"]
        for entry in entries
        if entry["target"] != "/"
        and "rw" in entry["options"]
        and entry["fstype"] not in pseudo
    )
    docker_socket_absent = not Path("/var/run/docker.sock").exists()
    try:
        init_command = Path("/proc/1/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        init_command = ""
    host_proc_hidden = bool(
        marker_regular
        and hostname_matches
        and init_command
        and not any(name in init_command.lower() for name in ("systemd", "init --system"))
    )
    home = Path(os.environ.get("HOME", "") or "/nonexistent").resolve()
    try:
        home_info = home.stat()
        home_names = {item.name for item in home.iterdir()}
    except OSError:
        home_info = None
        home_names = set()
    forbidden_home = {
        ".aws", ".azure", ".codex", ".config", ".docker", ".gnupg", ".ssh",
        ".huggingface", ".netrc",
    }
    private_home = bool(
        home_info
        and stat.S_ISDIR(home_info.st_mode)
        and not (home_info.st_mode & 0o077)
        and not (home_names & forbidden_home)
    )
    provider_fragments = (
        "OPENAI", "ANTHROPIC", "GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE",
        "AWS_ACCESS", "AZURE_CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS",
    )
    provider_auth_absent = not any(
        any(fragment in name.upper() for fragment in provider_fragments)
        and bool(value)
        for name, value in os.environ.items()
    )
    return MappingProxyType(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/quack-owner-isolation-observation@1",
            "container_marker_regular": marker_regular,
            # The full ID is externally inspect-bound in the signed receipt;
            # host-network containers cannot re-observe it through hostname or
            # cgroup.  Exact, launcher-chosen hostname is the independent live
            # link to that receipt, alongside the namespace/mount controls.
            "container_id": container_id if hostname_matches else "",
            "container_hostname": hostname,
            "root_read_only": root_read_only,
            "repository_read_only": repository_read_only,
            "rw_host_bind_targets": rw_host_bind_targets,
            "docker_socket_absent": docker_socket_absent,
            "host_proc_hidden": host_proc_hidden,
            "private_home": private_home,
            "provider_auth_absent": provider_auth_absent,
        }
    )


# ---------------------------------------------------------------------------
# Owner marker + exclusive lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerMarker:
    """Non-authoritative OS bootstrap projection of exclusive ownership."""

    SCHEMA: ClassVar[str] = OWNER_MARKER_SCHEMA

    server_id: str
    process_birth: ProcessBirthIdentity
    database_path: str
    started_at: str
    fence_token: str
    generation: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "server_id": self.server_id,
            "process_birth": self.process_birth.to_dict(),
            "database_path": self.database_path,
            "started_at": self.started_at,
            "fence_token": self.fence_token,
            "generation": int(self.generation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OwnerMarker:
        birth = ProcessBirthIdentity.from_dict(payload.get("process_birth"))
        return cls(
            server_id=str(payload.get("server_id") or ""),
            process_birth=birth,
            database_path=str(payload.get("database_path") or ""),
            started_at=str(payload.get("started_at") or ""),
            fence_token=str(payload.get("fence_token") or ""),
            generation=int(payload.get("generation") or 1),
        )


def _owner_lock_stat_is_secure(info: os.stat_result) -> bool:
    """Return whether ``info`` is the one admitted owner-lock file shape."""

    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600
    )


def _assert_owner_lock_name_matches_descriptor(
    *,
    directory_descriptor: int,
    lock_name: str,
    lock_descriptor: int,
) -> None:
    """Fail closed unless one secure directory entry still names ``lock_descriptor``."""

    descriptor_info = os.fstat(lock_descriptor)
    try:
        named_info = os.stat(
            lock_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise QuackStateServerOwnershipError(
            "exclusive owner lock name changed during acquisition"
        ) from exc
    if (
        not _owner_lock_stat_is_secure(descriptor_info)
        or not _owner_lock_stat_is_secure(named_info)
        or (descriptor_info.st_dev, descriptor_info.st_ino)
        != (named_info.st_dev, named_info.st_ino)
    ):
        raise QuackStateServerOwnershipError(
            "exclusive owner lock inode or metadata changed during acquisition"
        )


def _acquire_canonical_owner_path_fence(lock_path: Path) -> socket.socket:
    """Bind one per-path kernel mutex outside the replaceable filesystem tree."""

    if not hasattr(socket, "AF_UNIX"):
        raise QuackStateServerOwnershipError(
            "canonical owner path fencing requires AF_UNIX"
        )
    material = os.fsencode(f"{os.geteuid()}\0{lock_path}")
    address = (
        b"\0ipfs-accelerate-quack-owner:"
        + hashlib.sha256(material).hexdigest().encode("ascii")
    )
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        channel.set_inheritable(False)
        channel.bind(address)
    except OSError as exc:
        channel.close()
        if exc.errno == errno.EADDRINUSE:
            raise BlockingIOError(
                errno.EWOULDBLOCK,
                "canonical owner path fence is already held",
            ) from exc
        raise QuackStateServerOwnershipError(
            "canonical owner path fence is unavailable"
        ) from exc
    return channel


class _ExclusiveOwnerLockHandle:
    """Own the lock inode, directory identity, and canonical kernel fence."""

    __slots__ = (
        "_canonical_path_fence",
        "_directory_descriptor",
        "_directory_path",
        "_lock_descriptor",
    )

    def __init__(
        self,
        *,
        lock_descriptor: int,
        directory_descriptor: int,
        directory_path: Path,
        canonical_path_fence: socket.socket,
    ) -> None:
        self._lock_descriptor: int | None = lock_descriptor
        self._directory_descriptor: int | None = directory_descriptor
        self._directory_path: Path | None = directory_path
        self._canonical_path_fence: socket.socket | None = canonical_path_fence

    def fileno(self) -> int:
        descriptor = self._lock_descriptor
        if descriptor is None:
            raise ValueError("I/O operation on closed owner lock")
        return descriptor

    @property
    def closed(self) -> bool:
        return self._lock_descriptor is None

    def directory_fileno(self) -> int:
        descriptor = self._directory_descriptor
        if descriptor is None:
            raise ValueError("I/O operation on closed owner lock directory")
        return descriptor

    @property
    def directory_path(self) -> Path:
        path = self._directory_path
        if path is None:
            raise ValueError("I/O operation on closed owner lock directory")
        return path

    def assert_canonical_parent(self) -> None:
        """Fail unless the canonical parent name still resolves to our inode."""

        directory_descriptor = self.directory_fileno()
        directory_path = self.directory_path
        opened = os.fstat(directory_descriptor)
        try:
            named = os.lstat(directory_path)
        except OSError as exc:
            raise QuackStateServerOwnershipError(
                "exclusive owner lock directory name changed"
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid)
            != (named.st_dev, named.st_ino, named.st_mode, named.st_uid)
        ):
            raise QuackStateServerOwnershipError(
                "exclusive owner lock directory identity changed"
            )

    def close(self) -> None:
        lock_descriptor = self._lock_descriptor
        directory_descriptor = self._directory_descriptor
        path_fence = self._canonical_path_fence
        self._lock_descriptor = None
        self._directory_descriptor = None
        self._directory_path = None
        self._canonical_path_fence = None
        try:
            if lock_descriptor is not None:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(lock_descriptor)
        finally:
            try:
                if directory_descriptor is not None:
                    os.close(directory_descriptor)
            finally:
                if path_fence is not None:
                    path_fence.close()

    def __enter__(self) -> _ExclusiveOwnerLockHandle:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def read_locked_owner_marker(
    handle: Any,
    marker_path: Path,
) -> OwnerMarker | None:
    """Read one owner marker relative to the exact locked parent descriptor."""

    marker = Path(marker_path)
    try:
        directory_descriptor = int(handle.directory_fileno())
        directory_path = Path(handle.directory_path)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QuackStateServerOwnershipError(
            "owner marker read lacks the locked directory authority"
        ) from exc
    marker_parent = Path(os.path.abspath(os.fspath(marker.parent)))
    if marker_parent != directory_path:
        raise QuackStateServerOwnershipError(
            "owner marker is outside the locked directory authority"
        )
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise QuackStateServerOwnershipError(
            "secure owner marker read flags are unavailable"
        )
    try:
        descriptor = os.open(
            marker.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise QuackStateServerOwnershipError(
            "owner marker path is unavailable or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > OWNER_MARKER_MAX_BYTES
        ):
            raise QuackStateServerOwnershipError(
                "owner marker must be an owned private single-link regular file"
            )
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16 * 1024))
            if not chunk:
                raise QuackStateServerOwnershipError(
                    "owner marker was truncated while read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise QuackStateServerOwnershipError(
                "owner marker grew while read"
            )
        after = os.fstat(descriptor)
        try:
            named = os.stat(
                marker.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise QuackStateServerOwnershipError(
                "owner marker name changed while read"
            ) from exc
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(named):
        raise QuackStateServerOwnershipError(
            "owner marker inode or metadata changed while read"
        )
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_mutation_duplicate_guard,
        )
        expected_fields = {
            "schema",
            "server_id",
            "process_birth",
            "database_path",
            "started_at",
            "fence_token",
            "generation",
        }
        if type(payload) is not dict or set(payload) != expected_fields:
            raise ValueError("owner marker fields are invalid")
        parsed = OwnerMarker.from_dict(payload)
        if parsed.to_dict() != payload:
            raise ValueError("owner marker content is noncanonical")
        return parsed
    except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise QuackStateServerOwnershipError(
            "owner marker content is invalid"
        ) from exc


def _write_locked_owner_marker(
    handle: Any,
    marker_path: Path,
    marker: OwnerMarker,
) -> None:
    """Atomically publish one marker inside the exact locked parent inode."""

    try:
        handle.assert_canonical_parent()
        directory_descriptor = int(handle.directory_fileno())
        directory_path = Path(handle.directory_path)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QuackStateServerOwnershipError(
            "owner marker write lacks the locked directory authority"
        ) from exc
    path = Path(marker_path)
    if Path(os.path.abspath(os.fspath(path.parent))) != directory_path:
        raise QuackStateServerOwnershipError(
            "owner marker is outside the locked directory authority"
        )
    encoded = (
        json.dumps(
            marker.to_dict(),
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")
        + b"\n"
    )
    if not encoded or len(encoded) > OWNER_MARKER_MAX_BYTES:
        raise QuackStateServerOwnershipError("owner marker bytes are invalid")
    temporary_name = "." + path.name + "." + secrets.token_hex(8)
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("owner marker write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        handle.assert_canonical_parent()
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
        handle.assert_canonical_parent()
        admitted = read_locked_owner_marker(handle, path)
        if admitted != marker:
            raise QuackStateServerOwnershipError(
                "owner marker publication was not durably admitted"
            )
    except QuackStateServerOwnershipError:
        raise
    except OSError as exc:
        raise QuackStateServerOwnershipError(
            "owner marker cannot be securely published"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass


def acquire_exclusive_owner_lock(lock_path: Path) -> Any:
    """Open, attest, and non-blockingly lock the canonical owner-lock inode.

    The returned binary handle owns the advisory lock until it is closed (or
    explicitly unlocked).  Opening relative to a no-follow directory
    descriptor and repeating the name-to-descriptor identity check *after*
    ``flock`` prevents admission through a symlink, hardlink, or
    acquisition-time name replacement.  A per-UID, per-canonical-path abstract
    Unix socket remains bound for the handle lifetime, so replacing either the
    lock name or its whole parent cannot mint a second canonical authority.
    The parent descriptor is retained for exact marker I/O and rename checks;
    unrelated database paths do not contend on a shared directory lock.

    ``BlockingIOError`` is preserved for ordinary lock contention.  An unsafe
    path or identity race raises :class:`QuackStateServerOwnershipError`.
    """

    path = Path(os.path.abspath(os.fspath(lock_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise QuackStateServerOwnershipError(
            "secure exclusive owner lock flags are unavailable"
        )

    try:
        if path.parent.resolve(strict=True) != path.parent:
            raise QuackStateServerOwnershipError(
                "exclusive owner lock directory is aliased"
            )
    except OSError as exc:
        raise QuackStateServerOwnershipError(
            "exclusive owner lock directory is unavailable or unsafe"
        ) from exc
    path_fence = _acquire_canonical_owner_path_fence(path)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        directory_descriptor = os.open(path.parent, directory_flags)
    except OSError as exc:
        path_fence.close()
        raise QuackStateServerOwnershipError(
            "exclusive owner lock directory is unavailable or unsafe"
        ) from exc

    lock_descriptor: int | None = None
    lock_locked = False
    try:
        directory_info = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_uid != os.geteuid()
        ):
            raise QuackStateServerOwnershipError(
                "exclusive owner lock directory must be owned by the current user"
            )
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
        )
        try:
            lock_descriptor = os.open(
                path.name,
                lock_flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise QuackStateServerOwnershipError(
                "exclusive owner lock path is unavailable or unsafe"
            ) from exc

        initial_info = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(initial_info.st_mode)
            or initial_info.st_uid != os.geteuid()
            or initial_info.st_nlink != 1
        ):
            raise QuackStateServerOwnershipError(
                "exclusive owner lock must be an owned single-link regular file"
            )
        if stat.S_IMODE(initial_info.st_mode) != 0o600:
            # Safely tighten legacy ``Path.open('a+b')`` locks, which were
            # commonly created as 0644 through the process umask.  Only the
            # already-attested owned, single-link regular descriptor is
            # changed; the post-flock name check still rejects replacement.
            os.fchmod(lock_descriptor, 0o600)
        _assert_owner_lock_name_matches_descriptor(
            directory_descriptor=directory_descriptor,
            lock_name=path.name,
            lock_descriptor=lock_descriptor,
        )

        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_locked = True
        _assert_owner_lock_name_matches_descriptor(
            directory_descriptor=directory_descriptor,
            lock_name=path.name,
            lock_descriptor=lock_descriptor,
        )
        handle = _ExclusiveOwnerLockHandle(
            lock_descriptor=lock_descriptor,
            directory_descriptor=directory_descriptor,
            directory_path=path.parent,
            canonical_path_fence=path_fence,
        )
        lock_descriptor = None  # ownership transferred to ``handle``
        directory_descriptor = None
        path_fence = None
        return handle
    except BlockingIOError:
        raise
    except QuackStateServerOwnershipError:
        raise
    except OSError as exc:
        raise QuackStateServerOwnershipError(
            "exclusive owner lock could not be securely acquired"
        ) from exc
    finally:
        if lock_descriptor is not None:
            if lock_locked:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lock_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if path_fence is not None:
            path_fence.close()


class ExclusiveOwnerLease:
    """Process-exclusive owner lock with process-birth marker recovery."""

    def __init__(
        self,
        *,
        lock_path: Path,
        marker_path: Path,
        liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.marker_path = Path(marker_path)
        self._liveness = liveness or (lambda birth: owner_liveness(birth))
        self._handle: Any | None = None
        self._marker: OwnerMarker | None = None
        self._fence_token: str = ""

    @property
    def fence_token(self) -> str:
        return self._fence_token

    @property
    def marker(self) -> OwnerMarker | None:
        return self._marker

    def assert_stop_authority(
        self,
        *,
        server_id: str,
        fence_token: str,
    ) -> OwnerMarker:
        """Validate one stop capability against the held lock and live marker.

        This method is deliberately observation-only.  Callers use it before
        stopping workers, changing lifecycle, or closing any owner resource.
        """

        handle = self._handle
        cached = self._marker
        if handle is None or cached is None or not self._fence_token:
            raise QuackStateServerControlError(
                "stop requires a current owner lease and marker"
            )
        if (
            type(server_id) is not str
            or not server_id
            or type(fence_token) is not str
            or re.fullmatch(r"[0-9a-f]{32}", fence_token) is None
        ):
            raise QuackStateServerControlError(
                "stop authority is malformed"
            )
        handle.assert_canonical_parent()
        current = self._read_marker(handle=handle)
        if current is None or current != cached:
            raise QuackStateServerControlError(
                "stop owner marker does not match the held lease"
            )
        if (
            current.server_id != server_id
            or cached.server_id != server_id
            or current.fence_token != fence_token
            or cached.fence_token != fence_token
            or self._fence_token != fence_token
        ):
            raise QuackStateServerControlError(
                "stop authority does not match the live owner fence"
            )
        return current

    def bind_generation(self, generation: int) -> OwnerMarker:
        """Replace the provisional marker generation under the held lease."""

        handle = self._handle
        cached = self._marker
        if handle is None or cached is None or not self._fence_token:
            raise QuackStateServerOwnershipError(
                "owner generation binding requires the current lease"
            )
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise QuackStateServerOwnershipError(
                "owner generation binding is invalid"
            )
        handle.assert_canonical_parent()
        current = self._read_marker(handle=handle)
        if current is None or current != cached:
            raise QuackStateServerOwnershipError(
                "owner marker changed before generation binding"
            )
        updated = OwnerMarker(
            server_id=cached.server_id,
            process_birth=cached.process_birth,
            database_path=cached.database_path,
            started_at=cached.started_at,
            fence_token=cached.fence_token,
            generation=generation,
        )
        _write_locked_owner_marker(handle, self.marker_path, updated)
        observed = self._read_marker(handle=handle)
        if observed != updated:
            raise QuackStateServerOwnershipError(
                "owner marker generation binding was not observed"
            )
        self._marker = updated
        return updated

    def _read_marker(self, *, handle: Any) -> OwnerMarker | None:
        """Read only through the exact directory authority held by the lease."""

        return read_locked_owner_marker(handle, self.marker_path)

    def acquire(
        self,
        *,
        server_id: str,
        process_birth: ProcessBirthIdentity,
        database_path: Path,
        generation: int = 1,
    ) -> OwnerMarker:
        if self._handle is not None:
            raise QuackStateServerOwnershipError("owner lease already held in-process")

        try:
            handle = acquire_exclusive_owner_lock(self.lock_path)
        except BlockingIOError as exc:
            raise QuackStateServerOwnershipError(
                "second state-owner refused; exclusive lock held by unknown"
            ) from exc

        try:
            # Lock held: evaluate marker for live vs stale owner.
            existing = self._read_marker(handle=handle)
            if existing is not None and existing.process_birth.pid > 0:
                liveness = self._liveness(existing.process_birth)
                if liveness is OwnerLiveness.ALIVE:
                    raise QuackStateServerOwnershipError(
                        f"second state-owner refused; live owner "
                        f"{existing.server_id} pid={existing.process_birth.pid}"
                    )
                if liveness is OwnerLiveness.UNKNOWN:
                    raise QuackStateServerOwnershipError(
                        "state-owner marker liveness is unknown; refuse reclaim"
                    )
                # DEAD: stale marker recovery continues under our exclusive lock.

            fence = secrets.token_hex(16)
            marker = OwnerMarker(
                server_id=server_id,
                process_birth=process_birth,
                database_path=str(database_path),
                started_at=_utc_iso(),
                fence_token=fence,
                generation=int(generation),
            )
            _write_locked_owner_marker(handle, self.marker_path, marker)
        except BaseException:
            # Until self._handle is assigned, emergency cleanup cannot discover
            # this local authority. Close both filesystem locks and the canonical
            # kernel path fence on every fallible admission path.
            handle.close()
            raise
        self._handle = handle
        self._marker = marker
        self._fence_token = fence
        return marker

    def release(self, *, fence_token: str | None = None) -> None:
        if self._handle is None:
            return
        expected = fence_token if fence_token is not None else self._fence_token
        self._handle.assert_canonical_parent()
        current = self._read_marker(handle=self._handle)
        if current is not None and expected and current.fence_token != expected:
            raise QuackStateServerControlError(
                "stop fence token does not match owner marker"
            )
        try:
            if current is not None and (
                not expected or current.fence_token == expected
            ):
                try:
                    os.unlink(
                        self.marker_path.name,
                        dir_fd=self._handle.directory_fileno(),
                    )
                    os.fsync(self._handle.directory_fileno())
                except FileNotFoundError:
                    pass
        finally:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
                self._marker = None
                self._fence_token = ""

    def emergency_close_after_owner_shutdown(self) -> None:
        """Drop local lock authority only after every DB/listener is closed.

        Ordinary :meth:`release` intentionally retains the lease when a stale
        caller supplies the wrong fence.  A canonical-parent or marker I/O
        failure is different once the owning server has already closed its
        transport and database: retaining an unreachable abstract socket and
        integer lock descriptor would wedge safe restart for the rest of this
        process.  This path removes the marker only when it can still prove the
        exact cached marker through the locked directory; otherwise it leaves
        filesystem evidence untouched and releases only the local OS handles.
        """

        handle = self._handle
        if handle is None:
            return
        try:
            try:
                handle.assert_canonical_parent()
                current = read_locked_owner_marker(handle, self.marker_path)
                if (
                    current is not None
                    and self._marker is not None
                    and current == self._marker
                    and current.fence_token == self._fence_token
                ):
                    try:
                        os.unlink(
                            self.marker_path.name,
                            dir_fd=handle.directory_fileno(),
                        )
                        os.fsync(handle.directory_fileno())
                    except FileNotFoundError:
                        pass
                    handle.assert_canonical_parent()
            except (OSError, QuackStateServerOwnershipError, ValueError):
                # The server is already inert.  Never mutate an unproved path;
                # retain any questionable marker for explicit later recovery.
                pass
        finally:
            handle.close()
            self._handle = None
            self._marker = None
            self._fence_token = ""

    def assert_canonical_parent(self) -> None:
        if self._handle is None:
            raise QuackStateServerOwnershipError("owner lease is not held")
        self._handle.assert_canonical_parent()


# ---------------------------------------------------------------------------
# Token vault (handle-only public surface)
# ---------------------------------------------------------------------------


class TokenVault:
    """Store Quack auth tokens behind opaque secret handles.

    The token is staged through a mode-0600 file only during state-owner
    startup.  Once Quack has bound, the owner removes that file and retains
    the transport credential in process memory; supervisor processes receive
    a distinct typed-command credential. Public APIs expose only the handle.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._token: str | None = None
        self._handle: str | None = None
        self._path: Path | None = None
        self._generation: int = 0

    @property
    def secret_handle(self) -> str | None:
        return self._handle

    @property
    def generation(self) -> int:
        return self._generation

    def mint(self, *, secret_handle: str, generation: int = 1) -> SecretHandle:
        if not is_secret_handle(secret_handle):
            raise QuackStateServerTokenError(
                "token vault requires an opaque secret handle"
            )
        token = secrets.token_urlsafe(32)
        # Ensure the token never collides with handle prefixes.
        if any(token.startswith(prefix) for prefix in SECRET_HANDLE_PREFIXES):
            token = f"x{token}"
        path = self.state_dir / f"{secret_handle.replace(':', '_').replace('/', '_')}{TOKEN_FILENAME_SUFFIX}"
        _atomic_write_text(path, token, mode=0o600)
        self._token = token
        self._handle = secret_handle
        self._path = path
        self._generation = int(generation)
        return SecretHandle(handle=secret_handle, generation=int(generation))

    def resolve(self, secret_handle: str | None = None) -> str:
        """Return raw token for in-process transport only — never log or export."""

        handle = secret_handle or self._handle
        if not handle or handle != self._handle or not self._token:
            raise QuackStateServerTokenError("token is not available for handle")
        return self._token

    def remove_persisted_copy(self) -> None:
        """Remove startup material while retaining the in-process token."""

        if self._path is None:
            return
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        self._path = None

    def destroy(self) -> None:
        self._token = None
        if self._path is not None:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            self._path = None
        self._handle = None
        self._generation = 0

    def assert_absent_from(self, surface: Any, *, surface_name: str) -> None:
        if self._token and _contains_token_material(surface, self._token):
            raise QuackStateServerTokenError(
                f"auth token leaked into {surface_name}"
            )


# ---------------------------------------------------------------------------
# Transport adapter
# ---------------------------------------------------------------------------


class QuackTransport(Protocol):
    """Minimal Quack serve/query surface used by the state-owner."""

    def start(
        self,
        connection: Any,
        *,
        host: str,
        port: int,
        token: str,
        identity: StateServerIdentity,
    ) -> Mapping[str, Any]:
        """Start serving; must not log or return the raw token."""

    def live_query(
        self,
        connection: Any,
        *,
        identity: StateServerIdentity,
        token: str,
    ) -> Mapping[str, Any]:
        """Return live identity observation used for readiness."""

    def stop(self, connection: Any | None = None) -> None:
        """Stop serving (best effort)."""


class InProcessQuackTransport:
    """Default transport: load Quack and call ``quack_serve`` when available.

    When the real extension is absent, tests inject a fake transport. This
    default refuses to claim readiness without a successful serve + live query.
    """

    def __init__(self) -> None:
        self._started = False
        self._serve_uri = ""
        self._listen_uri = ""
        self._server_identity: dict[str, Any] = {}

    def start(
        self,
        connection: Any,
        *,
        host: str,
        port: int,
        token: str,
        identity: StateServerIdentity,
    ) -> Mapping[str, Any]:
        serve_uri = listen_uri(host, port)
        advertised_uri = identity.listen_uri
        allow_other_hostname = not _is_loopback_host(host)
        # DuckDB/Quack 1.5.5 exposes quack_serve as a table function.  Named
        # arguments are required after the address; the token stays bound and
        # never enters SQL text or the returned public observation.
        try:
            connection.execute("LOAD quack")
        except Exception as exc:
            raise QuackStateServerCapabilityError(
                f"failed to LOAD quack for state-owner: {type(exc).__name__}"
            ) from exc

        disable_ssl = _is_loopback_host(host)
        # Quack's current table function uses named optional parameters.
        # Values remain parameter-bound so the token is absent from SQL text,
        # argv, status, and logs.  Older qualified beta signatures remain a
        # narrow compatibility path.  Isolated container binds pass
        # ``allow_other_hostname`` so the advertised loopback identity can
        # differ from the in-container listen address.
        serve_attempts = (
            (
                "SELECT * FROM quack_serve(?, token := ?, "
                "allow_other_hostname := ?, disable_ssl := ?)",
                [serve_uri, token, allow_other_hostname, disable_ssl],
            ),
            (
                "SELECT * FROM quack_serve(token := ?, "
                "allow_other_hostname := ?, disable_ssl := ?)",
                [token, allow_other_hostname, disable_ssl],
            ),
            (
                "SELECT * FROM quack_serve(?, token := ?, "
                "allow_other_hostname := false, disable_ssl := true)",
                [serve_uri, token],
            ),
            ("SELECT quack_serve(?, ?, ?)", [host, int(port), token]),
            ("SELECT quack_serve(?, ?)", [f"{host}:{int(port)}", token]),
            ("CALL quack_serve(?, ?, ?)", [host, int(port), token]),
        )
        last_error: Exception | None = None
        for sql, params in serve_attempts:
            try:
                connection.execute(sql, params)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - depends on extension
                last_error = exc
                continue
        if last_error is not None:
            raise QuackStateServerCapabilityError(
                f"quack_serve failed: {type(last_error).__name__}"
            ) from last_error

        self._started = True
        self._serve_uri = serve_uri
        self._listen_uri = advertised_uri
        self._server_identity = {
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "schema_revision": identity.schema_revision,
            "schema_fingerprint": identity.schema_fingerprint,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "listen_uri": advertised_uri,
        }
        # Return public observation only.
        return MappingProxyType(dict(self._server_identity))

    def live_query(
        self,
        connection: Any,
        *,
        identity: StateServerIdentity,
        token: str,
    ) -> Mapping[str, Any]:
        if not self._started:
            raise QuackStateServerReadyError("transport has not started")
        # Prove the listener, authentication, request worker, and response path
        # are all usable.  Named token/disable_ssl arguments match the admitted
        # Quack 1.5.5 surface; positional 4-arg calls are a last compatibility
        # attempt only.
        query_attempts = (
            (
                "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := ?)",
                [self._listen_uri, "SELECT 1 AS quack_live", token, True],
            ),
            (
                "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
                [self._listen_uri, "SELECT 1 AS quack_live", token],
            ),
        )
        rows = None
        last_error: Exception | None = None
        # Quack publishes its connection id asynchronously after quack_serve
        # returns.  Retry only that exact startup observation, using a fresh
        # client each time; no readiness claim is made until a real query is
        # observed.  All other failures remain immediate and fail closed.
        startup_deadline = time.monotonic() + 2.0
        while True:
            try:
                import duckdb

                client = duckdb.connect(":memory:")
                try:
                    client.execute("LOAD quack")
                    for sql, params in query_attempts:
                        try:
                            rows = client.execute(sql, params).fetchall()
                            last_error = None
                            break
                        except Exception as exc:  # pragma: no cover - extension-version path
                            last_error = exc
                finally:
                    client.close()
            except Exception as exc:
                raise QuackStateServerReadyError(
                    f"authenticated remote live query failed: {type(exc).__name__}"
                ) from exc
            if last_error is None:
                break
            transient_startup = (
                type(last_error).__name__ == "InvalidInputException"
                and "Invalid connection id" in str(last_error)
            )
            if not transient_startup or time.monotonic() >= startup_deadline:
                break
            time.sleep(0.05)
        if last_error is not None:
            raise QuackStateServerReadyError(
                f"authenticated remote live query failed: {type(last_error).__name__}"
            ) from last_error
        if rows is None or len(rows) != 1:
            raise QuackStateServerReadyError(
                "authenticated remote live query returned an unexpected result: "
                f"{rows!r}"
            )
        live_row = rows[0]
        if isinstance(live_row, Mapping):
            live_value = live_row.get("quack_live")
            if live_value is None and len(live_row) == 1:
                live_value = next(iter(live_row.values()))
        elif isinstance(live_row, (list, tuple)):
            live_value = live_row[0] if live_row else None
        else:
            live_value = live_row
        if live_value not in (1, "1", True):
            raise QuackStateServerReadyError(
                "authenticated remote live query returned an unexpected result: "
                f"{rows!r}"
            )
        observed = dict(self._server_identity)
        observed["live"] = True
        if not identity.matches(
            store_id=str(observed.get("store_id") or ""),
            generation=int(observed.get("generation") or 0),
            schema_revision=int(observed.get("schema_revision") or -1),
            schema_fingerprint=str(observed.get("schema_fingerprint") or ""),
            server_id=str(observed.get("server_id") or ""),
            database_uuid=str(observed.get("database_uuid") or ""),
            process_birth_id=str(observed.get("process_birth_id") or ""),
        ):
            raise QuackStateServerReadyError(
                "live query identity does not match published state-owner identity"
            )
        return MappingProxyType(observed)

    def stop(self, connection: Any | None = None) -> None:
        if connection is not None and self._serve_uri:
            try:
                connection.execute(
                    "SELECT * FROM quack_stop(?)",
                    [self._serve_uri],
                ).fetchall()
            except Exception:
                # Closing the exclusive owning connection is the final stop
                # boundary; callers still receive lifecycle bookkeeping.
                pass
        self._started = False
        self._serve_uri = ""
        self._listen_uri = ""
        self._server_identity = {}


class FakeQuackTransport:
    """Test double that never binds a real port or needs the Quack extension."""

    def __init__(self, *, fail_live_query: bool = False) -> None:
        self.started = False
        self.stopped = False
        self.fail_live_query = fail_live_query
        self.start_calls: list[dict[str, Any]] = []
        self._identity: StateServerIdentity | None = None
        self._token_seen = False

    def start(
        self,
        connection: Any,
        *,
        host: str,
        port: int,
        token: str,
        identity: StateServerIdentity,
    ) -> Mapping[str, Any]:
        del connection
        self._token_seen = bool(token)
        # Record call without retaining the raw token.
        self.start_calls.append(
            {
                "host": host,
                "port": int(port),
                "token_present": bool(token),
                "token_length": len(token),
                "server_id": identity.server_id,
            }
        )
        self.started = True
        self._identity = identity
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": identity.listen_uri,
            }
        )

    def live_query(
        self,
        connection: Any,
        *,
        identity: StateServerIdentity,
        token: str,
    ) -> Mapping[str, Any]:
        del connection, token
        if self.fail_live_query:
            raise QuackStateServerReadyError("injected live query failure")
        if not self.started or self._identity is None:
            raise QuackStateServerReadyError("transport has not started")
        return MappingProxyType(
            {
                "live": True,
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
            }
        )

    def stop(self, connection: Any | None = None) -> None:
        del connection
        self.started = False
        self.stopped = True


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


@dataclass
class _DatabaseNamespaceAnchor:
    """Retained parent and inode capabilities for the authoritative database."""

    directory_descriptor: int
    directory_path: Path
    directory_identity: tuple[int, int, int, int]
    database_name: str
    database_descriptor: int = -1
    database_identity: tuple[int, int, int, int, int] | None = None


class ServerLifecycle(str, Enum):  # noqa: UP042 - Python 3.8 compatibility.
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class TypedStateOwnerGrantBroker:
    """Deliver one of two closed credentials across a private owner facet.

    This is an in-process facet of the existing Quack state owner, not a new
    state authority or daemon. It accepts no SQL, path, role, operation, or
    caller-selected scope; only the closed read/task credential discriminator.
    The sealed bootstrap descriptor and ``SO_PEERCRED`` bind delivery to an
    admitted trusted process; credentials are never written to disk or
    published in process metadata.
    """

    def __init__(
        self,
        *,
        socket_path: Path,
        bootstrap_secret: str,
        store_id: str,
        resolve_credential: Callable[[str, str, str, int], str],
    ) -> None:
        self.socket_path = Path(socket_path)
        self._secret = str(bootstrap_secret)
        self._store_id = str(store_id)
        self._resolve_credential = resolve_credential
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._last_error_type = ""
        self._clients_lock = threading.Lock()
        self._clients: set[threading.Thread] = set()
        self._channels: set[socket.socket] = set()
        self._client_capacity = threading.BoundedSemaphore(16)
        self._owner_uid = os.geteuid()
        self._bound_socket_identity: tuple[int, int] | None = None

    def _unlink_matching_socket(self, expected: os.stat_result) -> None:
        """Unlink only the same observed same-UID socket inode."""

        try:
            current = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise QuackStateServerControlError(
                "cannot reinspect typed grant broker socket"
            ) from exc
        if (
            not stat.S_ISSOCK(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_uid != self._owner_uid
            or (current.st_dev, current.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise QuackStateServerControlError(
                "typed grant broker socket changed during recovery"
            )
        self.socket_path.unlink()

    def _recover_stale_socket(self, observed: os.stat_result) -> None:
        """Recover a dead same-UID listener only after a failed connect probe."""

        if (
            not stat.S_ISSOCK(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != self._owner_uid
        ):
            raise QuackStateServerControlError(
                "typed grant broker socket is not a same-UID socket"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        try:
            probe.connect(str(self.socket_path))
        except ConnectionRefusedError:
            # A socket node with no listening endpoint is the only positive
            # stale observation admitted for recovery.
            self._unlink_matching_socket(observed)
        except FileNotFoundError:
            # Another same-UID recovery already retired it.
            return
        except OSError as exc:
            raise QuackStateServerControlError(
                "typed grant broker socket liveness is indeterminate"
            ) from exc
        else:
            raise QuackStateServerControlError(
                "typed grant broker socket already serves a live listener"
            )
        finally:
            probe.close()

    def start(self) -> None:
        if self._thread is not None:
            raise QuackStateServerControlError("typed grant broker already started")
        self._stop.clear()
        self._failed.clear()
        self._last_error_type = ""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            observed = os.lstat(self.socket_path.parent)
        except OSError as exc:
            raise QuackStateServerControlError(
                "typed grant broker state directory is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != self._owner_uid
        ):
            raise QuackStateServerControlError(
                "typed grant broker state directory is unsafe"
            )
        try:
            socket_metadata = os.lstat(self.socket_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise QuackStateServerControlError(
                "cannot inspect typed grant broker socket"
            ) from exc
        else:
            self._recover_stale_socket(socket_metadata)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        bound_metadata: os.stat_result | None = None
        try:
            listener.bind(str(self.socket_path))
            bound = True
            bound_metadata = os.lstat(self.socket_path)
            if (
                not stat.S_ISSOCK(bound_metadata.st_mode)
                or bound_metadata.st_uid != self._owner_uid
            ):
                raise QuackStateServerControlError(
                    "typed grant broker created an unsafe socket"
                )
            self._bound_socket_identity = (
                bound_metadata.st_dev,
                bound_metadata.st_ino,
            )
            os.chmod(self.socket_path, 0o600)
            listener.listen(16)
            listener.settimeout(0.25)
        except BaseException:
            listener.close()
            if bound and bound_metadata is not None:
                try:
                    self._unlink_matching_socket(bound_metadata)
                finally:
                    self._bound_socket_identity = None
            raise
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve,
            name="typed-state-owner-grant-broker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        with self._clients_lock:
            clients = tuple(self._clients)
            channels = tuple(self._channels)
        for channel in channels:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                channel.close()
            except OSError:
                pass
        for client in clients:
            client.join(timeout=1.0)
        try:
            current = os.lstat(self.socket_path)
        except FileNotFoundError:
            current = None
        if (
            current is not None
            and self._bound_socket_identity
            == (current.st_dev, current.st_ino)
        ):
            self._unlink_matching_socket(current)
        self._bound_socket_identity = None
        self._secret = ""

    def _latch_failure(self, exc: BaseException) -> None:
        """Permanently make an owner-side broker fault unhealthy."""

        with self._clients_lock:
            if not self._last_error_type:
                self._last_error_type = type(exc).__name__
            self._failed.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                channel, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError as exc:
                if not self._stop.is_set() and not self._failed.is_set():
                    self._latch_failure(exc)
                return
            if not self._client_capacity.acquire(blocking=False):
                channel.close()
                continue
            client = threading.Thread(
                target=self._serve_channel,
                args=(channel,),
                name="typed-state-owner-grant-client",
                daemon=True,
            )
            with self._clients_lock:
                self._clients.add(client)
                self._channels.add(channel)
            try:
                client.start()
            except BaseException as exc:
                with self._clients_lock:
                    self._clients.discard(client)
                    self._channels.discard(channel)
                self._client_capacity.release()
                channel.close()
                self._latch_failure(exc)
                return

    def _serve_channel(self, channel: socket.socket) -> None:
        try:
            self._serve_one(channel)
        except BaseException as exc:
            # Protocol and authorization denials are handled inside
            # _serve_one. Anything escaping it is an owner-side fault.
            self._latch_failure(exc)
        finally:
            try:
                channel.close()
            except OSError:
                pass
            current = threading.current_thread()
            with self._clients_lock:
                self._clients.discard(current)
                self._channels.discard(channel)
            self._client_capacity.release()

    def alive(self) -> bool:
        thread = self._thread
        return bool(
            thread is not None
            and thread.is_alive()
            and not self._failed.is_set()
        )

    def capability(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "available": self.alive(),
                "last_error_type": self._last_error_type,
            }
        )

    @staticmethod
    def _read_frame(channel: socket.socket) -> dict[str, Any]:
        payload = bytearray()
        while b"\n" not in payload:
            chunk = channel.recv(4096)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_GRANT_BROKER_FRAME_BYTES:
                raise QuackStateServerControlError(
                    "typed grant broker request exceeds its closed bound"
                )
        try:
            decoded = json.loads(bytes(payload).split(b"\n", 1)[0])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise QuackStateServerControlError(
                "typed grant broker request is malformed"
            ) from exc
        if not isinstance(decoded, dict):
            raise QuackStateServerControlError(
                "typed grant broker request must be an object"
            )
        return decoded

    def _serve_one(self, channel: socket.socket) -> None:
        response = {
            "schema": TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
            "credential_kind": "",
            "ok": False,
            "token": "",
            "error_code": "grant_denied",
        }
        authorized_identity: tuple[str, str, str, int] | None = None
        try:
            channel.settimeout(0.5)
            if getattr(socket, "SO_PEERCRED", None) is None:
                self._latch_failure(
                    QuackStateServerControlError(
                        "kernel peer credentials are unavailable"
                    )
                )
                return
            peer_pid, peer_uid, peer_start_time = _kernel_peer_identity(channel)
            request = self._read_frame(channel)
            if set(request) != {
                "schema",
                "credential_kind",
                "bootstrap_secret",
                "client_id",
                "process_birth_id",
                "store_id",
            }:
                raise QuackStateServerControlError(
                    "typed grant broker request has unknown fields"
                )
            client_id = str(request.get("client_id") or "").strip()
            process_birth_id = str(request.get("process_birth_id") or "").strip()
            credential_kind = str(request.get("credential_kind") or "").strip()
            response["credential_kind"] = credential_kind
            supplied_secret = str(request.get("bootstrap_secret") or "")
            if (
                request.get("schema") != TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA
                or credential_kind
                not in TYPED_STATE_OWNER_GRANT_BROKER_CREDENTIAL_KINDS
                or request.get("store_id") != self._store_id
                or peer_uid != self._owner_uid
                or peer_pid < 1
                or not 1 <= len(client_id) <= 256
                or process_birth_id
                != kernel_process_birth_id(
                    peer_pid,
                    start_time_ticks=peer_start_time,
                )
                or not hmac.compare_digest(supplied_secret, self._secret)
            ):
                raise QuackStateServerControlError(
                    "typed grant broker request is unauthorized"
                )
            authorized_identity = (
                credential_kind,
                client_id,
                process_birth_id,
                peer_pid,
            )
        except (
            OSError,
            ValueError,
            QuackStateServerError,
            TypedStateOwnerAuthorizationError,
        ):
            # Malformed, disconnected, or unauthorized peers are expected
            # denials and cannot poison delivery to later admitted clients.
            authorized_identity = None
        if authorized_identity is not None and not self._failed.is_set():
            try:
                token = self._resolve_credential(*authorized_identity)
            except BaseException as exc:
                self._latch_failure(exc)
            else:
                if (
                    not isinstance(token, str)
                    or not 8 <= len(token) <= 256
                    or any(
                        not character.isalnum() and character not in "_-"
                        for character in token
                    )
                ):
                    self._latch_failure(
                        QuackStateServerControlError(
                            "credential resolver returned invalid material"
                        )
                    )
                else:
                    response.update(
                        {"ok": True, "token": token, "error_code": ""}
                    )
        encoded = canonical_json_bytes(response) + b"\n"
        if len(encoded) <= MAX_GRANT_BROKER_FRAME_BYTES:
            try:
                channel.sendall(encoded)
            except OSError:
                pass


@dataclass
class QuackStateServer:
    """Long-lived exclusive owner of one control-plane DuckDB database.

    Interface: ``QuackStateServer@1``.
    """

    INTERFACE: ClassVar[str] = QUACK_STATE_SERVER_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_STATE_SERVER_SCHEMA

    config: QuackStateServerConfig
    transport: QuackTransport | None = None
    capability_probe: Callable[..., QuackCapabilityReport] | None = None
    migrate: Callable[..., MigrationRunReport] | None = None
    connection_factory: Callable[[Path], Any] | None = None
    process_birth_factory: Callable[[], ProcessBirthIdentity] | None = None
    owner_liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None
    isolation_observer: Callable[
        [QuackStateServerConfig, Mapping[str, Any]], Mapping[str, Any]
    ] | None = None
    clock: Callable[[], float] = field(default=time.time)
    _lifecycle: ServerLifecycle = field(default=ServerLifecycle.CREATED, init=False)
    _identity: StateServerIdentity | None = field(default=None, init=False)
    _connection: Any | None = field(default=None, init=False)
    _transport_connection: Any | None = field(default=None, init=False)
    _transport_connection_is_writer: bool = field(default=False, init=False)
    _owner: ExclusiveOwnerLease | None = field(default=None, init=False)
    _vault: TokenVault | None = field(default=None, init=False)
    _capability: QuackCapabilityReport | None = field(default=None, init=False)
    _migration_report: MigrationRunReport | None = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _bound_port: int = field(default=0, init=False)
    _logs: list[str] = field(default_factory=list, init=False, repr=False)
    _mutation_recovery_complete: bool = field(default=False, init=False, repr=False)
    _isolation_admission: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _read_replica_observation: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _read_replica_refresh_sequence: int = field(default=0, init=False, repr=False)
    _event_source: EventSource | None = field(default=None, init=False, repr=False)
    _event_wait: StateOwnerEventWait | None = field(default=None, init=False, repr=False)
    _command_gateway: TypedStateOwnerGateway | None = field(
        default=None, init=False, repr=False
    )
    _grant_broker: TypedStateOwnerGrantBroker | None = field(
        default=None, init=False, repr=False
    )
    _grant_broker_secret_fd: int = field(default=-1, init=False, repr=False)
    _database_namespace_anchor: _DatabaseNamespaceAnchor | None = field(
        default=None, init=False, repr=False
    )
    _database_namespace_drifted: bool = field(
        default=False, init=False, repr=False
    )
    _federation_repository: Any | None = field(default=None, init=False, repr=False)
    _outbox_wake: StateOwnerOutboxWake | None = field(
        default=None, init=False, repr=False
    )
    _outbox_worker: EventDrivenOutboxWorker | None = field(
        default=None, init=False, repr=False
    )
    _outbox_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _outbox_stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _outbox_drain_count: int = field(default=0, init=False)
    _outbox_last_error_type: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, QuackStateServerConfig):
            raise TypeError("config must be QuackStateServerConfig")
        if self.transport is None:
            self.transport = InProcessQuackTransport()
        if self.capability_probe is None:
            self.capability_probe = probe_quack_capabilities
        if self.process_birth_factory is None:
            self.process_birth_factory = current_process_birth
        if self.isolation_observer is None:
            self.isolation_observer = _observe_quack_owner_isolation
        self._vault = TokenVault(self.config.state_dir)

    # -- public properties -------------------------------------------------

    @property
    def lifecycle(self) -> ServerLifecycle:
        return self._lifecycle

    @property
    def identity(self) -> StateServerIdentity | None:
        return self._identity

    @property
    def secret_handle(self) -> str | None:
        return None if self._vault is None else self._vault.secret_handle

    # -- owner-local event wait -------------------------------------------

    def bind_event_source(self, source: EventSource) -> dict[str, object]:
        """Bind one typed source to the server-owned event condition.

        The binding is intentionally monotonic for a server lifecycle.  Every
        consumer waits on the same :class:`StateOwnerEventWait`; callers cannot
        replace it with a competing source after consumers have registered.
        """

        if not callable(getattr(source, "events_for_subscription", None)) or not callable(
            getattr(source, "store_generation", None)
        ):
            raise QuackStateServerControlError(
                "event source must expose the closed subscription and generation interfaces"
            )
        with self._lock:
            if self._lifecycle in {
                ServerLifecycle.STOPPING,
                ServerLifecycle.STOPPED,
                ServerLifecycle.FAILED,
            }:
                raise QuackStateServerNotRunningError(
                    "cannot bind event source to a terminal state owner"
                )
            if self._event_wait is not None:
                if self._event_source is source:
                    return self.event_wait_capability()
                raise QuackStateServerControlError(
                    "state owner already has a different event source"
                )
            self._event_source = source
            self._event_wait = StateOwnerEventWait(source)
            gateway = self._command_gateway
            if gateway is not None:
                gateway.bind_event_wait_handlers(
                    wait=self._gateway_wait_for_events,
                    cancel=self._gateway_cancel_event_wait,
                    clear_cancellation=self._gateway_clear_event_wait_cancellation,
                )
            return self.event_wait_capability()

    def _require_gateway_event_scope(
        self,
        grant: OwnerClientGrant,
        *,
        consumer_id: str,
        subscription_id: str,
    ) -> None:
        """Resolve tenant/federation from durable state, never wire input."""

        with self._lock:
            source = self._event_source
        resolver = getattr(source, "resolve_event_wait_scope", None)
        if not callable(resolver):
            raise QuackStateServerControlError(
                "event source cannot resolve authoritative wait scope"
            )
        tenant_id, federation_id = resolver(
            consumer_id=consumer_id,
            subscription_id=subscription_id,
        )
        if tenant_id != grant.tenant_id or federation_id != grant.federation_id:
            raise QuackStateServerControlError(
                "event wait scope differs from the client grant"
            )

    def _gateway_wait_for_events(
        self,
        request: EventWaitRequest,
        grant: OwnerClientGrant,
    ) -> EventBatch:
        self._require_gateway_event_scope(
            grant,
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
        )
        return self.wait_for_events(request)

    def _gateway_cancel_event_wait(
        self,
        consumer_id: str,
        grant: OwnerClientGrant,
    ) -> None:
        subscription_id = dict(grant.entity_scopes).get("subscription_id", "")
        if not subscription_id:
            raise QuackStateServerControlError(
                "event wait cancellation requires subscription scope"
            )
        self._require_gateway_event_scope(
            grant,
            consumer_id=consumer_id,
            subscription_id=subscription_id,
        )
        self.cancel_event_wait(consumer_id)

    def _gateway_clear_event_wait_cancellation(
        self,
        consumer_id: str,
        grant: OwnerClientGrant,
    ) -> None:
        subscription_id = dict(grant.entity_scopes).get("subscription_id", "")
        if not subscription_id:
            raise QuackStateServerControlError(
                "event wait cancellation requires subscription scope"
            )
        self._require_gateway_event_scope(
            grant,
            consumer_id=consumer_id,
            subscription_id=subscription_id,
        )
        self.clear_event_wait_cancellation(consumer_id)

    def _gateway_execute_database_task_command(
        self,
        command: str,
        payload: Mapping[str, Any],
        request_id: str,
        grant: OwnerClientGrant,
    ) -> Mapping[str, Any]:
        """Run one typed task command on the exclusive owner connection."""

        if command not in grant.allowed_database_task_commands:
            raise TypedStateOwnerAuthorizationError(
                "database-task command is outside the owner grant"
            )
        # The gateway already holds its exclusive connection lock. Bound the
        # opposite lifecycle-lock ordering so shutdown wins without waiting on
        # a client thread indefinitely; no mutation starts after STOPPING.
        if not self._lock.acquire(timeout=0.25):
            raise QuackStateServerNotRunningError(
                "state owner is entering a lifecycle transition"
            )
        try:
            if (
                self._lifecycle is not ServerLifecycle.READY
                or self._connection is None
                or self._identity is None
            ):
                raise QuackStateServerNotRunningError(
                    "database-task command requires a ready state owner"
                )
            self._assert_database_namespace()
            identity = self._identity
            repository = IntentRepository(
                self.config.database_path,
                bound_connection=self._connection,
                owner_id="quack-state-owner",
                session_id=f"quack-owner-{identity.generation}",
                install_schema=False,
            )
            try:
                try:
                    result = execute_quack_owner_command(
                        repository,
                        command,
                        payload,
                        request_id=request_id,
                        store_id=identity.store_id,
                        store_generation=str(identity.generation),
                    )
                    self._assert_database_namespace()
                except QuackStateServerOwnershipError:
                    raise
                except Exception as exc:
                    raise TypedStateOwnerDatabaseTaskCommandError(
                        quack_owner_command_error_code(exc)
                    ) from exc
            finally:
                repository.close()
            try:
                self._refresh_read_replica()
                self._write_status()
            except BaseException as refresh_exc:
                # The effect may already be durable. Withdraw readiness so a
                # caller cannot mistake a stale replica for current truth.
                try:
                    self._stop_transport_connection(observe_closed=True)
                except Exception:
                    pass
                if self._read_replica_observation:
                    self._read_replica_observation["live"] = False
                self._lifecycle = ServerLifecycle.FAILED
                self._log(
                    "typed owner command replica publication failed: "
                    + type(refresh_exc).__name__
                )
                try:
                    self._write_status()
                except Exception:
                    pass
                raise TypedStateOwnerDatabaseTaskCommandError(
                    "read_replica_refresh_unknown_outcome"
                ) from refresh_exc
            self._assert_database_namespace()
            return MappingProxyType(dict(result))
        finally:
            self._lock.release()

    def bind_federation_repository(
        self,
        client: Any,
        *,
        require_quack_authority: bool = True,
    ) -> Any:
        """Construct the canonical repository with this owner's notify hook.

        The import is local to keep module import cold.  This is the preferred
        wiring point: the repository publishes its event sequence only after
        ``submit_command`` durably commits, and the same repository is the
        bounded source used by all owner-local waiters.  No SQL or database
        path is accepted here.
        """

        from ..federation.registry import FederationStateRepository
        from ..task_sources.quack_state_client import QuackStateClient

        if not isinstance(client, QuackStateClient):
            raise QuackStateServerControlError(
                "federation event binding requires QuackStateClient"
            )
        with self._lock:
            identity = self._identity
            lifecycle = self._lifecycle
        session = client.session
        observed = None if session is None else session.store_identity
        if (
            lifecycle is not ServerLifecycle.READY
            or identity is None
            or session is None
            or observed is None
            or session.server_id != identity.server_id
            or observed.store_id != identity.store_id
            or observed.database_uuid != identity.database_uuid
            or observed.schema_revision != identity.schema_revision
            or observed.generation != identity.generation
            or observed.schema_fingerprint != identity.schema_fingerprint
        ):
            raise QuackStateServerControlError(
                "event repository client identity differs from the ready state owner"
            )
        repository = FederationStateRepository(
            client,
            event_notifier=self.notify_committed_event,
            outbox_notifier=self.notify_committed_outbox,
            require_quack_authority=require_quack_authority,
        )
        self.bind_event_source(repository)
        client.bind_event_wait_source(repository, owner_boundary=self)
        with self._lock:
            if (
                self._federation_repository is not None
                and self._federation_repository is not repository
            ):
                raise QuackStateServerControlError(
                    "state owner already has a canonical federation repository"
                )
            self._federation_repository = repository
            if self._outbox_wake is None:
                self._outbox_wake = StateOwnerOutboxWake()
            gateway = self._command_gateway
            if gateway is not None:
                gateway.bind_commit_observer(self._observe_typed_commit)
        return repository

    def notify_committed_outbox(self, global_sequence: int) -> bool:
        """Signal the owner outbox pump after an event/outbox commit."""

        with self._lock:
            wake = self._outbox_wake
            lifecycle = self._lifecycle
        if lifecycle in {ServerLifecycle.STOPPING, ServerLifecycle.STOPPED}:
            return False
        if wake is None:
            raise QuackStateServerControlError(
                "state-owner outbox wake source is not bound"
            )
        return wake.notify_committed(global_sequence)

    def _observe_typed_commit(
        self,
        command: Any,
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> None:
        """Translate a durable transaction manifest into condition signals."""

        operation = str(command.parameters.get("operation") or "")
        if operation in {
            "federation.create",
            "budget.reserve",
            "budget.release",
            "supervisor.register",
            "supervisor.runtime.attest",
            "supervisor.transition",
            "subagent.register",
            "subagent.slot.reserve",
            "subagent.slot.release",
            "subagent.outcome",
            "subscription.register",
        }:
            sequences = [
                int(bound.get("global_sequence") or 0)
                for name, bound in manifest
                if name == "casf_insert_domain_event"
            ]
            if sequences:
                self.notify_committed_outbox(max(sequences))
        if operation == "event.outbox.disposition":
            sequences = [
                int(bound.get("global_sequence") or 0)
                for name, bound in manifest
                if name == "casf_mark_outbox_routed"
            ]
            if sequences:
                self.notify_committed_event(max(sequences))
        if operation == "event.acknowledge":
            # Acknowledging an older event releases a bounded delivery slot.
            # Signal the outbox pump by notification generation without
            # rewinding or inventing an event watermark.
            sequences = [
                int(bound.get("global_sequence") or 0)
                for name, bound in manifest
                if name == "casf_insert_event_acknowledgement"
            ]
            if sequences:
                self.notify_committed_outbox(max(sequences))

    def start_federation_outbox_worker(
        self,
        *,
        health_deadline_seconds: float = 30.0,
    ) -> Mapping[str, Any]:
        """Drain the restart backlog and enter a condition-blocked owner loop."""

        from ..federation.durable_event_router import DurableEventRouter

        deadline = float(health_deadline_seconds)
        if not 1.0 <= deadline <= 300.0:
            raise QuackStateServerControlError(
                "outbox health deadline must be between 1 and 300 seconds"
            )
        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                raise QuackStateServerNotRunningError(
                    "outbox worker requires a ready state owner"
                )
            if self._outbox_thread is not None:
                if self._outbox_thread.is_alive():
                    return self.outbox_worker_capability()
                raise QuackStateServerControlError(
                    "prior outbox worker exited and requires explicit recovery"
                )
            repository = self._federation_repository
            wake = self._outbox_wake
            if repository is None or wake is None:
                raise QuackStateServerControlError(
                    "outbox worker requires the canonical federation repository"
                )
        worker = EventDrivenOutboxWorker(
            repository,
            lambda _scope: DurableEventRouter(repository),
            wake,
        )
        # Backlog recovery is bounded and happens before live readiness is
        # reported for the pump.  Never hold the lifecycle lock across the
        # typed owner transaction: its post-commit observer acquires that lock.
        initial = worker.drain_once()
        with self._lock:
            if self._outbox_thread is not None:
                raise QuackStateServerControlError(
                    "outbox worker was concurrently started"
                )
            self._outbox_drain_count = 1
            self._outbox_worker = worker
            self._outbox_stop.clear()
            self._outbox_last_error_type = ""

            def run() -> None:
                while not self._outbox_stop.is_set():
                    try:
                        receipt = worker.wait_and_drain(
                            deadline_monotonic=time.monotonic() + deadline
                        )
                        if receipt is not None:
                            with self._lock:
                                self._outbox_drain_count += 1
                            # The operator's fail-closed health classifier
                            # compares this owner-published watermark with the
                            # authoritative event cursor.  Publish every
                            # advancing/notification-backed drain so that a
                            # healthy event-driven worker is not represented
                            # by its stale startup snapshot.  Timeouts remain
                            # write-free because they return ``None``.
                            self._write_status()
                    except BaseException as exc:
                        self._outbox_last_error_type = type(exc).__name__
                        # Publish the terminal worker observation immediately.
                        # The owner process and typed gateway can remain alive
                        # after this thread exits, so process liveness alone is
                        # not an event-routing health witness.
                        self._write_status()
                        return

            thread = threading.Thread(
                target=run,
                name="casf-state-owner-outbox",
                daemon=True,
            )
            self._outbox_thread = thread
            thread.start()
            result = {
                "available": bool(thread.is_alive() and not self._outbox_last_error_type),
                "initial_event_count": initial.event_count,
                "initial_delivery_count": initial.delivery_count,
                "watermark": worker.watermark,
                "thread_alive": thread.is_alive(),
                "server_owned": True,
                "polling": False,
                "last_error_type": self._outbox_last_error_type,
            }
        # Startup readiness is not complete until the public owner status
        # carries the live router-worker observation used by the operator.
        self._write_status()
        return MappingProxyType(result)

    def outbox_worker_capability(self) -> Mapping[str, Any]:
        with self._lock:
            thread = self._outbox_thread
            worker = self._outbox_worker
            wake = self._outbox_wake
            thread_alive = bool(thread is not None and thread.is_alive())
            available = bool(thread_alive and not self._outbox_last_error_type)
            return MappingProxyType(
                {
                    "available": available,
                    "thread_alive": thread_alive,
                    "server_owned": True,
                    "polling": False,
                    "watermark": 0 if worker is None else worker.watermark,
                    "committed_sequence": (
                        0 if wake is None else wake.committed_sequence
                    ),
                    "wakeup_count": 0 if wake is None else wake.wakeup_count,
                    "notification_generation": (
                        0 if wake is None else wake.notification_generation
                    ),
                    "drain_count": self._outbox_drain_count,
                    "last_error_type": self._outbox_last_error_type,
                }
            )

    def notify_committed_event(self, global_sequence: int) -> bool:
        """Wake owner-local consumers after an authoritative commit.

        This hook never writes state or invents an event.  A notification that
        races shutdown is safely ignored because the durable outbox remains
        replayable; while READY, absence of the sealed wait source fails
        closed as a server configuration error.
        """

        with self._lock:
            event_wait = self._event_wait
            lifecycle = self._lifecycle
        if lifecycle in {ServerLifecycle.STOPPING, ServerLifecycle.STOPPED}:
            return False
        if event_wait is None:
            raise QuackStateServerControlError(
                "state-owner event wait source is not bound"
            )
        event_wait.notify_committed(global_sequence)
        return True

    def wait_for_events(self, request: EventWaitRequest) -> EventBatch:
        """Execute one typed bounded wait through the live owner boundary."""

        if not isinstance(request, EventWaitRequest):
            raise QuackStateServerControlError(
                "event wait requires EventWaitRequest"
            )
        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                raise QuackStateServerNotRunningError(
                    "event wait requires a ready state owner"
                )
            event_wait = self._event_wait
            identity = self._identity
        if event_wait is None:
            raise QuackStateServerControlError(
                "state-owner event wait source is not bound"
            )
        if identity is None:
            raise QuackStateServerControlError(
                "ready state owner has no sealed identity"
            )
        # Never hold the server lifecycle lock across a bounded blocking wait.
        batch = event_wait.wait_for_events(request)
        if batch.store_generation != identity.generation:
            raise QuackStateServerControlError(
                "event wait batch store generation differs from the state owner"
            )
        return batch

    def cancel_event_wait(self, consumer_id: str) -> None:
        """Cancel one consumer without disturbing other blocked consumers."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackStateServerControlError("consumer_id is required")
        with self._lock:
            event_wait = self._event_wait
        if event_wait is None:
            raise QuackStateServerControlError(
                "state-owner event wait source is not bound"
            )
        event_wait.cancel(consumer)

    def clear_event_wait_cancellation(self, consumer_id: str) -> None:
        """Clear an explicit cancellation before a later consumer wait."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackStateServerControlError("consumer_id is required")
        with self._lock:
            event_wait = self._event_wait
        if event_wait is None:
            raise QuackStateServerControlError(
                "state-owner event wait source is not bound"
            )
        event_wait.clear_cancel(consumer)

    def event_wait_capability(self) -> dict[str, object]:
        """Return an observation, never an event-driven promotion claim."""

        with self._lock:
            event_wait = self._event_wait
        if event_wait is None:
            return {
                "interface": "StateOwnerEventWait@1",
                "available": False,
                "server_owned": True,
                "event_driven_qualified": False,
                "reason": "typed event source is not bound",
            }
        capability = dict(event_wait.capability())
        capability.update(
            {
                "available": True,
                "query_count": event_wait.query_count,
                "wakeup_count": event_wait.wakeup_count,
                "notification_generation": event_wait.notification_generation,
                # Owner-local blocking is real, but remote Quack push remains
                # unavailable and the program promotion gate remains closed.
                "event_driven_qualified": False,
            }
        )
        return capability

    # -- logging (token-safe) ---------------------------------------------

    def _log(self, message: str) -> None:
        token = None if self._vault is None else getattr(self._vault, "_token", None)
        text = str(message)
        if token and token in text:
            text = text.replace(token, REDACTION_MARKER)
        self._logs.append(text)
        _logger.info("%s", text)

    def logs(self) -> tuple[str, ...]:
        return tuple(self._logs)

    # -- paths -------------------------------------------------------------

    def owner_lock_path(self) -> Path:
        db = self.config.database_path
        return db.with_name(f".{db.name}{OWNER_LOCK_SUFFIX}")

    def owner_marker_path(self) -> Path:
        db = self.config.database_path
        return db.with_name(f".{db.name}{OWNER_MARKER_SUFFIX}")

    def status_path(self) -> Path:
        return self.config.state_dir / STATUS_FILENAME

    def stop_control_path(self) -> Path:
        return self.config.state_dir / CONTROL_STOP_FILENAME

    @staticmethod
    def _database_directory_identity(
        value: os.stat_result,
    ) -> tuple[int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_uid),
        )

    @staticmethod
    def _database_file_identity(
        value: os.stat_result,
    ) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_uid),
            int(value.st_nlink),
        )

    def _database_namespace_failure(self, reason: str) -> NoReturn:
        self._database_namespace_drifted = True
        raise QuackStateServerOwnershipError(
            "authoritative database namespace drifted: " + reason
        )

    def _open_database_parent_anchor(self) -> _DatabaseNamespaceAnchor:
        """Retain the exact canonical parent before migration can mutate it."""

        if self._database_namespace_anchor is not None:
            self._database_namespace_failure("parent anchor was opened twice")
        required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        if any(not hasattr(os, name) for name in required):
            raise QuackStateServerOwnershipError(
                "authoritative database namespace flags are unavailable"
            )
        parent = self.config.database_path.parent
        descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            opened = os.fstat(descriptor)
            named = os.lstat(parent)
            identity = self._database_directory_identity(opened)
            if (
                identity != self._database_directory_identity(named)
                or not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or parent.resolve(strict=True) != parent
                or self.config.database_path.name in {"", ".", ".."}
            ):
                raise QuackStateServerOwnershipError(
                    "authoritative database parent is not exact and private"
                )
            anchor = _DatabaseNamespaceAnchor(
                directory_descriptor=descriptor,
                directory_path=parent,
                directory_identity=identity,
                database_name=self.config.database_path.name,
            )
            self._database_namespace_anchor = anchor
            self._database_namespace_drifted = False
            return anchor
        except BaseException:
            os.close(descriptor)
            raise

    def _assert_database_namespace(
        self,
        *,
        require_database: bool = True,
    ) -> _DatabaseNamespaceAnchor:
        """Require retained descriptors and public names to remain identical."""

        anchor = self._database_namespace_anchor
        if anchor is None:
            self._database_namespace_failure("parent anchor is absent")
        try:
            opened_parent = os.fstat(anchor.directory_descriptor)
            named_parent = os.lstat(anchor.directory_path)
        except OSError:
            self._database_namespace_failure("parent is unavailable")
        if (
            self._database_directory_identity(opened_parent)
            != anchor.directory_identity
            or self._database_directory_identity(named_parent)
            != anchor.directory_identity
            or not stat.S_ISDIR(opened_parent.st_mode)
            or anchor.directory_path != self.config.database_path.parent
            or anchor.database_name != self.config.database_path.name
            or anchor.directory_path.resolve(strict=True) != anchor.directory_path
        ):
            self._database_namespace_failure("parent name or inode changed")
        if not require_database:
            if anchor.database_descriptor < 0:
                self._database_namespace_drifted = False
            return anchor
        if anchor.database_descriptor < 0 or anchor.database_identity is None:
            self._database_namespace_failure("database inode anchor is absent")
        try:
            opened_database = os.fstat(anchor.database_descriptor)
            named_database = os.stat(
                anchor.database_name,
                dir_fd=anchor.directory_descriptor,
                follow_symlinks=False,
            )
            public_database = os.lstat(self.config.database_path)
        except OSError:
            self._database_namespace_failure("database name is unavailable")
        expected = anchor.database_identity
        if (
            self._database_file_identity(opened_database) != expected
            or self._database_file_identity(named_database) != expected
            or self._database_file_identity(public_database) != expected
            or not stat.S_ISREG(opened_database.st_mode)
            or opened_database.st_uid != os.geteuid()
            or opened_database.st_nlink != 1
        ):
            self._database_namespace_failure("database name or inode changed")
        self._database_namespace_drifted = False
        return anchor

    def _bind_database_inode_after_migration(self) -> _DatabaseNamespaceAnchor:
        """Bind the exact post-migration database inode for this owner lifetime."""

        anchor = self._assert_database_namespace(require_database=False)
        if anchor.database_descriptor >= 0 or anchor.database_identity is not None:
            self._database_namespace_failure("database inode was bound twice")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            descriptor = os.open(
                anchor.database_name,
                flags,
                dir_fd=anchor.directory_descriptor,
            )
        except FileNotFoundError:
            if self.connection_factory is None:
                self._database_namespace_failure(
                    "migration did not create the authoritative database"
                )
            # Injected connection factories are a hermetic test seam.  Give
            # that seam a real private inode so namespace-fence tests exercise
            # the same retained capability without claiming fake DB bytes live.
            descriptor = os.open(
                anchor.database_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=anchor.directory_descriptor,
            )
            os.fsync(descriptor)
            os.fsync(anchor.directory_descriptor)
        except OSError:
            self._database_namespace_failure("database inode could not be opened")
        try:
            opened = os.fstat(descriptor)
            named = os.stat(
                anchor.database_name,
                dir_fd=anchor.directory_descriptor,
                follow_symlinks=False,
            )
            public = os.lstat(self.config.database_path)
            identity = self._database_file_identity(opened)
            if (
                identity != self._database_file_identity(named)
                or identity != self._database_file_identity(public)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
            ):
                self._database_namespace_failure(
                    "post-migration database inode is unsafe"
                )
            anchor.database_descriptor = descriptor
            anchor.database_identity = identity
            descriptor = -1
            return self._assert_database_namespace()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _close_database_namespace_anchor(self) -> None:
        """Close namespace capabilities only after every DB/endpoint is inert."""

        anchor = self._database_namespace_anchor
        if anchor is None:
            return
        if self._connection is not None or self._transport_connection is not None:
            raise QuackStateServerControlError(
                "database namespace anchor cannot close while owner resources live"
            )
        if anchor.database_descriptor >= 0:
            os.close(anchor.database_descriptor)
            anchor.database_descriptor = -1
        os.close(anchor.directory_descriptor)
        anchor.directory_descriptor = -1
        self._database_namespace_anchor = None
        self._database_namespace_drifted = False

    def read_replica_path(self) -> Path:
        """Return the non-authoritative Quack transport snapshot path."""

        database = self.config.database_path
        return database.with_name(
            f"{database.stem}{READ_REPLICA_NAME_INFIX}{database.suffix}"
        )

    def typed_command_socket_path(self) -> Path:
        return (
            self.config.typed_command_socket_path_override
            or self.config.state_dir / TYPED_STATE_OWNER_SOCKET_FILENAME
        )

    def typed_command_token_path(self) -> Path:
        return self.config.state_dir / TYPED_STATE_OWNER_TOKEN_FILENAME

    def issue_typed_client_grant(
        self,
        *,
        client_id: str,
        process_birth_id: str = "",
        allowed_operations: Sequence[str] = (),
        allowed_command_operations: Sequence[str] = (),
        allowed_database_task_commands: Sequence[str] = (),
        tenant_id: str = "",
        federation_id: str = "",
        entity_scopes: Mapping[str, str] | None = None,
        peer_pid: int | None = None,
        ttl_seconds: float = 3_600.0,
    ) -> str:
        """Mint a bounded grant from owner code, never from a client request."""

        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                raise QuackStateServerNotRunningError(
                    "client grant issuance requires a ready state owner"
                )
            gateway = self._command_gateway
        if gateway is None:
            raise QuackStateServerControlError(
                "typed command gateway is unavailable"
            )
        token, _grant = gateway.issue_grant(
            client_id=client_id,
            process_birth_id=process_birth_id,
            allowed_operations=allowed_operations,
            allowed_command_operations=allowed_command_operations,
            allowed_database_task_commands=allowed_database_task_commands,
            tenant_id=tenant_id,
            federation_id=federation_id,
            entity_scopes=entity_scopes,
            peer_pid=peer_pid,
            ttl_seconds=ttl_seconds,
        )
        return token

    def start_supervisor_grant_broker(self) -> Mapping[str, str]:
        """Start the closed credential handoff for configured supervisors.

        The broker is a facet of this exact owner process. It delivers the
        strictly read-only Quack transport credential, a short-lived same-peer
        grant for the six-command task vocabulary, or hash observations.
        Callers cannot select operations, authority, paths, or scopes.
        """

        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                raise QuackStateServerNotRunningError(
                    "supervisor grant broker requires a ready state owner"
                )
            if self._grant_broker is not None:
                raise QuackStateServerControlError(
                    "supervisor grant broker is already running"
                )
            identity = self._identity
            vault = self._vault
            gateway = self._command_gateway
        if identity is None or vault is None or gateway is None:
            raise QuackStateServerControlError(
                "Quack credential authority is unavailable"
            )
        bootstrap_secret = secrets.token_hex(32)
        memfd_flags = int(getattr(os, "MFD_CLOEXEC", 0x0001)) | int(
            getattr(os, "MFD_ALLOW_SEALING", 0x0002)
        )
        try:
            secret_fd = os.memfd_create(
                "ipfs-accelerate-state-grant", flags=memfd_flags
            )
            os.write(secret_fd, bootstrap_secret.encode("ascii"))
            os.fchmod(secret_fd, 0o400)
            seal_flags = (
                int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
                | int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
                | int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
                | int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
            )
            fcntl.fcntl(
                secret_fd,
                int(getattr(fcntl, "F_ADD_SEALS", 1033)),
                seal_flags,
            )
        except (AttributeError, OSError):
            try:
                os.close(secret_fd)
            except (NameError, OSError):
                pass
            raise QuackStateServerControlError(
                "sealed typed-grant descriptor is unavailable"
            ) from None

        def resolve_credential(
            credential_kind: str,
            client_id: str,
            process_birth_id: str,
            peer_pid: int,
        ) -> str:
            if credential_kind == TYPED_STATE_OWNER_CREDENTIAL_READ_TRANSPORT:
                return vault.resolve(identity.secret_handle)
            if credential_kind == TYPED_STATE_OWNER_CREDENTIAL_HASH_OBSERVATION:
                token, _grant = gateway.issue_grant(
                    client_id=client_id,
                    process_birth_id=process_birth_id,
                    allowed_operations=(HASH_OBSERVATION_SERVICE_OPERATION,),
                    peer_pid=peer_pid,
                    ttl_seconds=DATABASE_TASK_COMMAND_GRANT_TTL_SECONDS,
                )
                return token
            if (
                credential_kind
                == TYPED_STATE_OWNER_CREDENTIAL_DATABASE_TASK_COMMAND
            ):
                token, _grant = gateway.issue_grant(
                    client_id=client_id,
                    process_birth_id=process_birth_id,
                    allowed_operations=(),
                    allowed_command_operations=(),
                    allowed_database_task_commands=tuple(
                        sorted(DATABASE_TASK_COMMANDS)
                    ),
                    peer_pid=peer_pid,
                    ttl_seconds=DATABASE_TASK_COMMAND_GRANT_TTL_SECONDS,
                )
                return token
            raise QuackStateServerControlError(
                "typed grant broker credential kind is not admitted"
            )

        broker = TypedStateOwnerGrantBroker(
            socket_path=(
                self.typed_command_socket_path().parent
                / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME
            ),
            bootstrap_secret=bootstrap_secret,
            store_id=identity.store_id,
            resolve_credential=resolve_credential,
        )
        try:
            broker.start()
        except BaseException:
            os.close(secret_fd)
            raise
        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                broker.stop()
                os.close(secret_fd)
                raise QuackStateServerNotRunningError(
                    "state owner stopped during grant-broker startup"
                )
            self._grant_broker = broker
            self._grant_broker_secret_fd = secret_fd
        self._write_status()
        return MappingProxyType(
            {
                TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV: str(
                    broker.socket_path
                ),
                TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV: str(secret_fd),
                TYPED_STATE_OWNER_SOCKET_ENV: str(
                    self.typed_command_socket_path()
                ),
            }
        )

    def bind_typed_status_scope(self) -> None:
        """Bind the persisted status bootstrap to the admitted live slice."""

        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY:
                raise QuackStateServerNotRunningError(
                    "status scope binding requires a ready state owner"
                )
            gateway = self._command_gateway
        if gateway is None:
            raise QuackStateServerControlError(
                "typed command gateway is unavailable"
            )
        gateway.bind_status_bootstrap_scope()

    def revoke_typed_client_grant(self, grant_id: str) -> None:
        """Revoke one previously issued grant at the exclusive owner."""

        with self._lock:
            gateway = self._command_gateway
        if gateway is None:
            raise QuackStateServerControlError(
                "typed command gateway is unavailable"
            )
        gateway.revoke_grant(grant_id)

    def mutation_inbox_path(self) -> Path:
        """Return the owner-only inbox used for unsupported remote DML."""

        return quack_owner_mutation_inbox_path(self.runtime_registry_path)

    @property
    def runtime_registry_path(self) -> Path:
        """Return the sealed owner/worker runtime-registry identity binding."""

        return self.config.state_dir

    def _prepare_mutation_inbox(self) -> int:
        """Create, admit, and return a pinned owner-only inbox descriptor."""

        try:
            return open_mutation_inbox_directory(self.mutation_inbox_path())
        except QuackOwnerMutationEnvelopeError as exc:
            raise QuackStateServerReadyError(
                "mutation inbox is not a safe owner directory"
            ) from exc

    @staticmethod
    def _mutation_result_rows(result: Any) -> tuple[tuple[str, ...], list[list[Any]], int]:
        """Project one already-executed DML cursor into a bounded response."""

        columns = tuple(str(item) for item in getattr(result, "_columns", ()) or ())
        rowcount = int(getattr(result, "rowcount", -1) or -1)
        raw_rows = result.fetchall() if callable(getattr(result, "fetchall", None)) else []
        if raw_rows and not columns:
            description = getattr(result, "description", None) or ()
            columns = tuple(str(item[0]) for item in description)
            if not columns and isinstance(raw_rows[0], Mapping):
                columns = tuple(str(item) for item in raw_rows[0])
        rows: list[list[Any]] = []
        for raw in raw_rows:
            if isinstance(raw, Mapping):
                rows.append([raw[name] for name in columns])
            else:
                rows.append(list(raw))
            if len(rows) > MAX_MUTATION_RESULT_ROWS:
                raise QuackOwnerMutationEnvelopeError(
                    "mutation result row count exceeds its bound",
                    code="result_not_serializable",
                )
        return columns, rows, rowcount

    def process_mutation_inbox(
        self,
        *,
        max_requests: int = MAX_MUTATIONS_PER_POLL,
        now_ms: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Serialize mutation processing with lifecycle and owner teardown."""

        catalog_bound = MUTATION_MAX_PER_PASS
        if not isinstance(catalog_bound, int) or catalog_bound < 1:
            catalog_bound = 32
        # Drain closed-catalog bundles first so protocol-@2 `*.request.json`
        # files are not misread as envelope mutations.
        self.service_mutation_inbox(
            max_requests=min(max(int(max_requests), 1), catalog_bound)
        )
        with self._lock:
            return self._process_mutation_inbox_locked(
                max_requests=max_requests,
                now_ms=now_ms,
            )

    def _process_mutation_inbox_locked(
        self,
        *,
        max_requests: int = MAX_MUTATIONS_PER_POLL,
        now_ms: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Apply authenticated bounded DML on the exclusive owner connection.

        The method is intentionally polling-oriented: Quack remains the read
        transport while UPDATE/DELETE/CAS operations stay on the process that
        owns the DuckDB file.  Every request is HMAC-bound to the current
        store generation and every response is signed with the same secret.
        """

        if (
            self._lifecycle is not ServerLifecycle.READY
            or self._connection is None
            or self._identity is None
            or self._vault is None
        ):
            raise QuackStateServerReadyError(
                "mutation inbox requires the live exclusive state owner"
            )
        self._assert_database_namespace()
        if (
            isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests < 1
            or max_requests > MAX_MUTATIONS_PER_POLL
        ):
            raise ValueError(
                f"max_requests must be in [1, {MAX_MUTATIONS_PER_POLL}]"
            )
        token = self._vault.resolve(self._identity.secret_handle)
        inbox_fd = self._prepare_mutation_inbox()
        suffix = ".request.json"
        summaries: list[Mapping[str, Any]] = []
        try:
            request_names = sorted(
                name for name in os.listdir(inbox_fd) if name.endswith(suffix)
            )[:max_requests]
            for request_name in request_names:
                request_id = request_name[: -len(suffix)]
                summaries.append(
                    self._process_mutation_request_at(
                        inbox_fd=inbox_fd,
                        request_name=request_name,
                        done_name=f"{request_id}.done.json",
                        request_id=request_id,
                        token=token,
                        now_ms=now_ms,
                    )
                )
        finally:
            os.close(inbox_fd)
        self._assert_database_namespace()
        return tuple(summaries)

    def _process_mutation_request_at(
        self,
        *,
        inbox_fd: int,
        request_name: str,
        done_name: str,
        request_id: str,
        token: str,
        now_ms: int | None,
    ) -> Mapping[str, Any]:
        """Process one request entirely relative to a pinned inbox handle."""

        assert self._connection is not None
        assert self._identity is not None
        self._assert_database_namespace()
        error_code = ""
        error = ""
        columns: tuple[str, ...] = ()
        rows: list[list[Any]] = []
        rowcount = -1
        ok = False

        # A signed result is an idempotency tombstone. This handles the
        # crash/retry edge where publication completed but request cleanup did
        # not: never execute that DML twice.
        if mutation_envelope_exists_at(inbox_fd, done_name):
            try:
                prior = parse_mutation_result(
                    read_envelope_at(inbox_fd, done_name),
                    token=token,
                    expected_request_id=request_id,
                    expected_store_id=self._identity.store_id,
                    expected_generation=self._identity.generation,
                )
            except QuackOwnerMutationEnvelopeError:
                # Never overwrite an unauthenticated or unsafe collision and
                # never let it cause the request to run.
                try:
                    unlink_mutation_envelope_at(
                        inbox_fd,
                        request_name,
                        missing_ok=True,
                    )
                except OSError:
                    pass
                return MappingProxyType(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error_code": "result_collision",
                        "rowcount": -1,
                    }
                )
            try:
                unlink_mutation_envelope_at(
                    inbox_fd,
                    request_name,
                    missing_ok=True,
                )
            except OSError:
                pass
            return MappingProxyType(
                {
                    "request_id": request_id,
                    "ok": bool(prior["ok"]),
                    "error_code": str(prior["error_code"]),
                    "rowcount": int(prior["rowcount"]),
                    "replayed": True,
                }
            )
        try:
            request = parse_mutation_request(
                read_envelope_at(
                    inbox_fd,
                    request_name,
                    max_bytes=MAX_MUTATION_REQUEST_BYTES,
                ),
                token=token,
                expected_request_id=request_id,
                expected_store_id=self._identity.store_id,
                expected_generation=self._identity.generation,
                now_ms=now_ms,
            )
            parameters = request["parameters"]
            self._assert_database_namespace()
            if parameters is None:
                result = self._connection.execute(request["sql"])
            else:
                result = self._connection.execute(request["sql"], parameters)
            self._assert_database_namespace()
            columns, rows, rowcount = self._mutation_result_rows(result)
            ok = True
        except QuackStateServerOwnershipError:
            raise
        except QuackOwnerMutationEnvelopeError as exc:
            error_code = exc.code
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - typed owner boundary
            error_code = "execution_failed"
            error = f"owner mutation execution failed: {type(exc).__name__}"
        try:
            if ok:
                self._assert_database_namespace()
            response = build_mutation_result(
                request_id=request_id,
                store_id=self._identity.store_id,
                generation=self._identity.generation,
                ok=ok,
                token=token,
                rowcount=rowcount,
                columns=columns,
                rows=rows,
                error_code=error_code,
                error=error,
                completed_at_ms=now_ms,
            )
        except QuackOwnerMutationEnvelopeError:
            # An invalid filename cannot be reflected into a valid signed
            # receipt. Remove it so a forged path cannot spin forever.
            try:
                unlink_mutation_envelope_at(
                    inbox_fd,
                    request_name,
                    missing_ok=True,
                )
            except OSError:
                pass
            return MappingProxyType(
                {
                    "request_id": "",
                    "ok": False,
                    "error_code": "malformed_filename",
                }
            )
        try:
            # A result is an idempotency tombstone, so even a concurrent
            # collision must never be overwritten after execution.
            write_envelope_atomic_at(
                inbox_fd,
                done_name,
                response,
                replace=False,
            )
        except (OSError, QuackOwnerMutationEnvelopeError) as exc:
            try:
                unlink_mutation_envelope_at(
                    inbox_fd,
                    request_name,
                    missing_ok=True,
                )
            except OSError:
                pass
            return MappingProxyType(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error_code": (
                        "result_collision"
                        if isinstance(exc, FileExistsError)
                        else "result_publication_failed"
                    ),
                    "rowcount": rowcount,
                    "outcome_unknown": ok,
                }
            )
        try:
            unlink_mutation_envelope_at(
                inbox_fd,
                request_name,
                missing_ok=True,
            )
        except OSError:
            pass
        return MappingProxyType(
            {
                "request_id": request_id,
                "ok": ok,
                "error_code": error_code,
                "rowcount": rowcount,
            }
        )

    # -- capability + migration -------------------------------------------

    def _admit_capability(self) -> QuackCapabilityReport:
        assert self.capability_probe is not None
        report = self.capability_probe()
        if report.passes_health_check:
            return report
        if self.config.allow_experimental and report.experimental_usable:
            return report
        raise QuackStateServerCapabilityError(
            f"Quack capability admission failed: status={report.status.value} "
            f"reason={report.reason_code}"
        )

    def _ensure_migrated(self) -> MigrationRunReport:
        if self.migrate is not None:
            report = self.migrate(self.config.database_path)
            return report
        if not duckdb_available():
            raise QuackStateServerMigrationError(
                "DuckDB is required to migrate before serving"
            )
        try:
            return install_control_plane_schema(
                self.config.database_path,
                application_version=self.config.application_version,
                tool_version=self.config.tool_version,
                owner_id=f"quack-state-server:{os.getpid()}",
            )
        except Exception as exc:
            raise QuackStateServerMigrationError(
                f"control-plane migration failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _admit_isolated_owner(self) -> dict[str, Any] | None:
        """Admit the explicit container-only external-access owner mode."""

        receipt_path = self.config.isolation_receipt_path
        if receipt_path is None:
            return None
        state_dir = self.config.state_dir.expanduser().resolve()
        path = receipt_path.expanduser()
        if not path.is_absolute():
            path = state_dir / path
        path = path.resolve()
        try:
            path.relative_to(state_dir)
        except ValueError as exc:
            raise QuackStateServerIsolationError(
                "isolation receipt must be stored under the owner state directory"
            ) from exc
        try:
            info = path.stat()
        except OSError as exc:
            raise QuackStateServerIsolationError(
                "isolation receipt is unavailable"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 32_768
        ):
            raise QuackStateServerIsolationError(
                "isolation receipt must be an owner-only bounded regular file"
            )
        try:
            raw = path.read_bytes()
            payload = json.loads(raw, object_pairs_hook=_mutation_duplicate_guard)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise QuackStateServerIsolationError(
                "isolation receipt is not canonical JSON"
            ) from exc
        fields = {
            "schema",
            "runtime",
            "container_id",
            "container_hostname",
            "network_mode",
            "container_bind_host",
            "container_port",
            "published_host",
            "published_port",
            "published_protocol",
            "owner_write_root",
            "database_path",
            "state_dir",
            "repository_path",
            "allowed_rw_mount_targets",
            "issuer",
            "issued_at",
            "receipt_cid",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise QuackStateServerIsolationError(
                "isolation receipt has unknown or missing fields"
            )
        unsigned = dict(payload)
        receipt_cid = unsigned.pop("receipt_cid", None)
        database_path = str(self.config.database_path.expanduser().resolve())
        owner_write_root = str(self.config.database_path.parent.resolve())
        raw_repository_path = str(payload.get("repository_path") or "").strip()
        repository_path = str(Path(raw_repository_path).resolve()) if raw_repository_path else ""
        allowed_rw = payload.get("allowed_rw_mount_targets")
        if (
            payload.get("schema") != QUACK_ISOLATION_RECEIPT_SCHEMA
            or payload.get("runtime") not in {"docker", "podman"}
            or not isinstance(payload.get("container_id"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(payload.get("container_id") or "")
            )
            is None
            or not isinstance(payload.get("container_hostname"), str)
            or re.fullmatch(
                r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?",
                str(payload.get("container_hostname") or ""),
            )
            is None
            or payload.get("network_mode") != "bridge"
            or payload.get("container_bind_host") != "0.0.0.0"
            or payload.get("container_bind_host")
            != self.config.container_bind_host
            or type(payload.get("container_port")) is not int
            or not 1 <= int(payload.get("container_port") or 0) <= 65535
            or payload.get("container_port") != self.config.container_port
            or payload.get("published_host") != DEFAULT_LOOPBACK_HOST
            or payload.get("published_host") != self.config.host
            or type(payload.get("published_port")) is not int
            or not 1 <= int(payload.get("published_port") or 0) <= 65535
            or payload.get("published_port") != self.config.port
            or payload.get("published_port") != payload.get("container_port")
            or payload.get("published_protocol") != "tcp"
            or payload.get("owner_write_root") != owner_write_root
            or payload.get("database_path") != database_path
            or payload.get("state_dir") != str(state_dir)
            or self.config.database_path.parent.resolve()
            != Path(owner_write_root)
            or state_dir != Path(owner_write_root) / "quack-owner"
            or self.config.database_path.parent != Path(owner_write_root)
            or self.config.database_path.name in {"", ".", ".."}
            or not repository_path
            or not Path(raw_repository_path).is_absolute()
            or raw_repository_path != repository_path
            or not isinstance(allowed_rw, list)
            or any(not isinstance(item, str) for item in allowed_rw)
            or allowed_rw != [owner_write_root]
            or not str(payload.get("issuer") or "").strip()
            or not str(payload.get("issued_at") or "").strip()
            or receipt_cid != content_identity(unsigned)
            or canonical_json_bytes(payload) != raw.strip()
        ):
            raise QuackStateServerIsolationError(
                "isolation receipt identity or required controls are invalid"
            )
        assert self.isolation_observer is not None
        try:
            observed = dict(self.isolation_observer(self.config, payload))
        except Exception as exc:
            raise QuackStateServerIsolationError(
                "live isolation observation failed"
            ) from exc
        observation_fields = {
            "schema",
            "container_marker_regular",
            "container_id",
            "container_hostname",
            "root_read_only",
            "repository_read_only",
            "rw_host_bind_targets",
            "docker_socket_absent",
            "host_proc_hidden",
            "private_home",
            "provider_auth_absent",
        }
        if (
            set(observed) != observation_fields
            or observed.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/quack-owner-isolation-observation@1"
            or observed.get("container_marker_regular") is not True
            or observed.get("container_id") != payload.get("container_id")
            or observed.get("container_hostname")
            != payload.get("container_hostname")
            or observed.get("root_read_only") is not True
            or observed.get("repository_read_only") is not True
            or observed.get("rw_host_bind_targets") != sorted(allowed_rw)
            or observed.get("docker_socket_absent") is not True
            or observed.get("host_proc_hidden") is not True
            or observed.get("private_home") is not True
            or observed.get("provider_auth_absent") is not True
        ):
            raise QuackStateServerIsolationError(
                "live container isolation does not satisfy the admitted receipt"
            )
        return payload

    def _open_connection(
        self, *, isolation_admission: Mapping[str, Any] | None
    ) -> Any:
        """Open the authoritative writer with the ordinary sealed policy.

        The writer never loads Quack and never enables external access.  Quack
        is loaded only on the distinct read-only transport replica.
        """

        del isolation_admission
        if self.connection_factory is not None:
            return self.connection_factory(self.config.database_path)
        if not duckdb_available():
            raise QuackStateServerError("DuckDB is required for the state-owner")
        if isinstance(self.transport, InProcessQuackTransport):
            return open_duckdb_connection(
                self.config.database_path,
                threads=1,
                memory_limit=DEFAULT_MEMORY_LIMIT,
                quack_owner=True,
            )
        return open_quack_state_owner_connection(self.config.database_path)

    def _read_replica_enabled(self) -> bool:
        """Return whether this is the real, non-injected transport path."""

        return self.connection_factory is None

    def _verify_loaded_transport_extensions(self, connection: Any) -> None:
        capability = self._capability
        if capability is None or not capability.extension.install_path:
            raise QuackStateServerCapabilityError(
                "state-owner lacks a prequalified local Quack extension path"
            )
        extension = connection.execute(
            "SELECT loaded, install_path, extension_version "
            "FROM duckdb_extensions() WHERE extension_name = 'quack'"
        ).fetchone()
        if (
            extension is None
            or extension[0] is not True
            or Path(str(extension[1])).resolve()
            != Path(capability.extension.install_path).resolve()
            or (
                capability.extension.extension_version
                and str(extension[2]) != capability.extension.extension_version
            )
        ):
            raise QuackStateServerCapabilityError(
                "loaded Quack bytes differ from the admitted capability"
            )

        # Quack's HTTP transport uses the installed core httpfs build.  Admit
        # only the build colocated with the already-qualified Quack bytes.
        httpfs = connection.execute(
            "SELECT installed, install_path, extension_version, "
            "install_mode, installed_from FROM duckdb_extensions() "
            "WHERE extension_name = 'httpfs'"
        ).fetchone()
        if (
            httpfs is None
            or httpfs[0] is not True
            or not str(httpfs[1] or "")
            or Path(str(httpfs[1])).resolve().parent
            != Path(capability.extension.install_path).resolve().parent
            or not str(httpfs[2] or "")
            or str(httpfs[3]) != "REPOSITORY"
            or str(httpfs[4]) != "core"
        ):
            raise QuackStateServerCapabilityError(
                "Quack transport lacks the colocated core httpfs build"
            )

    def _open_read_replica_connection(
        self, *, isolation_admission: Mapping[str, Any] | None
    ) -> Any:
        """Open and seal the non-authoritative snapshot in read-only mode."""

        if not duckdb_available():
            raise QuackStateServerError("DuckDB is required for the state-owner")
        anchor = self._assert_database_namespace()
        replica = self.read_replica_path()
        if replica.parent != anchor.directory_path:
            self._database_namespace_failure(
                "read-replica path is outside the retained parent"
            )
        replica_descriptor = -1
        isolation = isolation_admission
        try:
            import duckdb

            replica_descriptor = os.open(
                replica.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=anchor.directory_descriptor,
            )
            opened_replica = os.fstat(replica_descriptor)
            named_replica = os.stat(
                replica.name,
                dir_fd=anchor.directory_descriptor,
                follow_symlinks=False,
            )
            replica_identity = self._database_file_identity(opened_replica)
            if (
                replica_identity != self._database_file_identity(named_replica)
                or not stat.S_ISREG(opened_replica.st_mode)
                or opened_replica.st_uid != os.geteuid()
                or opened_replica.st_nlink != 1
            ):
                raise QuackStateServerReadyError(
                    "read-replica inode is not exact and private"
                )
            descriptor_path = Path("/proc/self/fd") / str(replica_descriptor)
            if not descriptor_path.exists():
                raise QuackStateServerReadyError(
                    "descriptor-bound read-replica open is unavailable"
                )
            connection = duckdb.connect(
                str(descriptor_path),
                read_only=True,
                config={
                    "autoinstall_known_extensions": "false",
                    "autoload_known_extensions": "false",
                    "enable_external_access": "true",
                    "allow_unsigned_extensions": "false",
                    "threads": "1",
                    "memory_limit": DEFAULT_MEMORY_LIMIT,
                },
            )
            connection.execute("LOAD quack")
            self._verify_loaded_transport_extensions(connection)
            connection.execute("LOAD httpfs")
            functions = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT function_name FROM duckdb_functions() "
                    "WHERE function_name IN ('quack_serve', 'quack_query')"
                ).fetchall()
            }
            if functions != {"quack_serve", "quack_query"}:
                raise QuackStateServerCapabilityError(
                    "loaded Quack extension lacks the admitted serve/query surface"
                )
            access_mode = connection.execute(
                "SELECT current_setting('access_mode')"
            ).fetchone()
            if access_mode is None or str(access_mode[0]).lower() != "read_only":
                raise QuackStateServerCapabilityError(
                    "Quack transport replica is not read-only"
                )
            # Loading is the only external-access window.  Serving fixed
            # loopback HTTP continues to work after this is disabled, while
            # raw authenticated SQL cannot COPY, read files, or reach URLs.
            connection.execute("SET enable_external_access = false")
            connection.execute("SET allow_persistent_secrets = false")
            connection.execute("SET lock_configuration = true")
            names = tuple(name for name, _configured, _expected in DUCKDB_CONNECTION_POLICY_SETTINGS)
            settings = connection.execute(
                "SELECT " + ", ".join(f"current_setting('{name}')" for name in names)
            ).fetchone()
            expected = tuple(
                value
                for _name, _configured, value in DUCKDB_CONNECTION_POLICY_SETTINGS
            )
            if settings is None or tuple(settings) != expected:
                raise QuackStateServerCapabilityError(
                    "read-only Quack transport did not seal the DuckDB policy"
                )
            after_replica = os.fstat(replica_descriptor)
            current_replica = os.stat(
                replica.name,
                dir_fd=anchor.directory_descriptor,
                follow_symlinks=False,
            )
            if (
                self._database_file_identity(after_replica) != replica_identity
                or self._database_file_identity(current_replica)
                != replica_identity
            ):
                raise QuackStateServerReadyError(
                    "read-replica namespace changed while opening"
                )
            self._assert_database_namespace()
            if isolation is not None:
                self._log(
                    "isolated Quack owner mode admitted "
                    f"receipt_cid={isolation['receipt_cid']}"
                )
            return DuckDBConnection.wrap(connection)
        except BaseException:
            if "connection" in locals():
                try:
                    connection.close()
                except Exception:
                    pass
            raise
        finally:
            if replica_descriptor >= 0:
                os.close(replica_descriptor)

    def _copy_authoritative_read_replica(self) -> tuple[str, int]:
        """Checkpoint and atomically refresh the bounded replica file."""

        if self._connection is None:
            raise QuackStateServerReadyError("authoritative writer is unavailable")
        anchor = self._assert_database_namespace()
        source = self.config.database_path
        replica = self.read_replica_path()
        if (
            source.parent != anchor.directory_path
            or replica.parent != anchor.directory_path
            or source == replica
        ):
            raise QuackStateServerReadyError("read-replica path is outside owner root")
        temporary_name = f".{replica.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        source_descriptor: int | None = None
        target_descriptor: int | None = None
        started = time.monotonic()
        try:
            self._assert_database_namespace()
            self._connection.execute("CHECKPOINT")
            anchor = self._assert_database_namespace()
            source_descriptor = os.open(
                anchor.database_name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=anchor.directory_descriptor,
            )
            before = os.fstat(source_descriptor)
            if (
                anchor.database_identity is None
                or self._database_file_identity(before)
                != anchor.database_identity
                or not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > READ_REPLICA_MAX_BYTES
            ):
                raise QuackStateServerReadyError(
                    "authoritative database is not an owner-only bounded regular file"
                )
            target_descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=anchor.directory_descriptor,
            )
            digest = hashlib.sha256()
            copied = 0
            while copied < before.st_size:
                if time.monotonic() - started > READ_REPLICA_COPY_TIMEOUT_SECONDS:
                    raise QuackStateServerReadyError("read-replica copy timed out")
                chunk = os.read(
                    source_descriptor,
                    min(READ_REPLICA_COPY_CHUNK_BYTES, before.st_size - copied),
                )
                if not chunk:
                    raise QuackStateServerReadyError(
                        "authoritative database changed during replica copy"
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_descriptor, view)
                    if written <= 0:
                        raise QuackStateServerReadyError(
                            "read-replica copy made no progress"
                        )
                    view = view[written:]
                copied += len(chunk)
            after = os.fstat(source_descriptor)
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
                raise QuackStateServerReadyError(
                    "authoritative database changed during replica copy"
                )
            os.fchmod(target_descriptor, 0o600)
            os.fsync(target_descriptor)
            os.close(target_descriptor)
            target_descriptor = None
            os.replace(
                temporary_name,
                replica.name,
                src_dir_fd=anchor.directory_descriptor,
                dst_dir_fd=anchor.directory_descriptor,
            )
            os.fsync(anchor.directory_descriptor)
            self._assert_database_namespace()
            return f"sha256:{digest.hexdigest()}", copied
        except QuackStateServerError:
            raise
        except Exception as exc:
            raise QuackStateServerReadyError(
                f"read-replica refresh failed: {type(exc).__name__}"
            ) from exc
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            if target_descriptor is not None:
                os.close(target_descriptor)
            try:
                os.unlink(temporary_name, dir_fd=anchor.directory_descriptor)
            except FileNotFoundError:
                pass

    def _wait_for_transport_endpoint_closed(self) -> None:
        """Independently observe endpoint closure before replacing a replica."""

        if self._bound_port <= 0:
            return
        deadline = time.monotonic() + READ_REPLICA_STOP_TIMEOUT_SECONDS
        # Quack closes asynchronously. A tight connect loop can itself keep
        # feeding the listener and prevent the shutdown it is trying to
        # observe, so allow one scheduling turn and probe with bounded
        # exponential spacing.
        delay = 0.02
        time.sleep(delay)
        while True:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(0.05)
            try:
                observation = probe.connect_ex(
                    (self.config.host, self._bound_port)
                )
            finally:
                probe.close()
            # Only a loopback refusal positively proves that no listener owns
            # this address.  Timeouts, backlog exhaustion, routing errors, and
            # other nonzero results are unknown rather than evidence of close.
            if observation == errno.ECONNREFUSED:
                return
            if time.monotonic() >= deadline:
                raise QuackStateServerReadyError(
                    "stale Quack transport remained reachable during refresh"
                )
            time.sleep(delay)
            delay = min(delay * 2.0, 0.25)

    def _stop_transport_connection(self, *, observe_closed: bool) -> None:
        connection = self._transport_connection
        connection_is_writer = self._transport_connection_is_writer
        stop_errors: list[Exception] = []
        if connection is not None:
            try:
                if self.transport is not None:
                    self.transport.stop(connection)
            except Exception as exc:
                stop_errors.append(exc)
            if not connection_is_writer and hasattr(connection, "close"):
                try:
                    connection.close()
                except Exception as exc:
                    stop_errors.append(exc)
        if observe_closed:
            # Observation is the authority.  A transport API error does not
            # prove the endpoint remained live, and a successful API return
            # does not prove it closed.  Keep the exact transport handle until
            # refusal is observed so a caller can retry shutdown after an API
            # error without relying on process exit to recover the listener.
            self._wait_for_transport_endpoint_closed()
        if self._transport_connection is connection:
            self._transport_connection = None
            self._transport_connection_is_writer = False
            if self._read_replica_observation:
                self._read_replica_observation["live"] = False
        if stop_errors:
            self._log(
                "transport shutdown API warning after endpoint closure: "
                + type(stop_errors[0]).__name__
            )

    def _close_authoritative_connection_observed(self) -> None:
        """Close the writer and retain its handle unless close returned."""

        connection = self._connection
        if connection is None:
            return
        close = getattr(connection, "close", None)
        if not callable(close):
            raise QuackStateServerControlError(
                "authoritative database connection cannot be closed"
            )
        close()
        self._connection = None

    def _assert_live_identity_observation(
        self, observed: Mapping[str, Any]
    ) -> None:
        if self._identity is None:
            raise QuackStateServerReadyError("state-owner identity is unavailable")
        required_fields = (
            "store_id",
            "generation",
            "schema_revision",
            "schema_fingerprint",
            "server_id",
            "database_uuid",
            "process_birth_id",
        )
        missing = [
            name for name in required_fields if observed.get(name) in (None, "")
        ]
        if missing:
            raise QuackStateServerReadyError(
                "live query missing identity fields: " + ", ".join(missing)
            )
        try:
            observed_generation = int(observed["generation"])
            observed_schema_revision = int(observed["schema_revision"])
        except (TypeError, ValueError) as exc:
            raise QuackStateServerReadyError(
                "live query identity fields are not integers"
            ) from exc
        if not self._identity.matches(
            store_id=str(observed["store_id"]),
            generation=observed_generation,
            schema_revision=observed_schema_revision,
            schema_fingerprint=str(observed["schema_fingerprint"]),
            server_id=str(observed["server_id"]),
            database_uuid=str(observed["database_uuid"]),
            process_birth_id=str(observed["process_birth_id"]),
        ):
            raise QuackStateServerReadyError(
                "live query identities do not match published state-owner identity"
            )

    def _refresh_read_replica(
        self, *, verify_request: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Synchronously replace and prove the transport snapshot.

        The endpoint is stopped before checkpoint/copy and remains unavailable
        until the new read-only connection, live identity, and optional exact
        mutation effects have all been observed.
        """

        if (
            self._connection is None
            or self._identity is None
            or self._vault is None
            or self.transport is None
        ):
            raise QuackStateServerReadyError(
                "read-replica refresh lacks writer, identity, vault, or transport"
            )
        self._assert_database_namespace()
        identity = self._identity
        token = self._vault.resolve(identity.secret_handle)
        if not self._read_replica_enabled():
            if self._transport_connection is None:
                public = self.transport.start(
                    self._connection,
                    host=self.config.container_bind_host,
                    port=int(self.config.container_port or self._bound_port),
                    token=token,
                    identity=identity,
                )
                self._vault.assert_absent_from(
                    public, surface_name="transport.start"
                )
                self._transport_connection = self._connection
                self._transport_connection_is_writer = True
            observed = self.transport.live_query(
                self._transport_connection,
                identity=identity,
                token=token,
            )
            self._assert_live_identity_observation(observed)
            self._read_replica_refresh_sequence += 1
            self._read_replica_observation = {
                "schema": "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1",
                "authority": "injected_test_transport",
                "path": str(self.read_replica_path()),
                "source_database_path": str(self.config.database_path),
                "server_id": identity.server_id,
                "database_uuid": identity.database_uuid,
                "generation": identity.generation,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "storage_schema_fingerprint": str(
                    self._mutation_binding().get("schema_fingerprint") or ""
                ),
                "sha256": "",
                "size_bytes": 0,
                "refresh_sequence": self._read_replica_refresh_sequence,
                "refreshed_at_ms": int(self.clock() * 1000),
                "live": True,
            }
            self._assert_database_namespace()
            return dict(self._read_replica_observation)

        self._stop_transport_connection(observe_closed=True)
        try:
            digest, size_bytes = self._copy_authoritative_read_replica()
            replica_connection = self._open_read_replica_connection(
                isolation_admission=self._isolation_admission
            )
            self._transport_connection = replica_connection
            self._transport_connection_is_writer = False
            writer_meta = self._read_meta(self._connection)
            replica_meta = self._read_meta(replica_connection)
            if replica_meta != writer_meta:
                raise QuackStateServerReadyError(
                    "read-replica metadata differs from authoritative writer"
                )
            public = self.transport.start(
                replica_connection,
                host=self.config.container_bind_host,
                port=int(self.config.container_port or self._bound_port),
                token=token,
                identity=identity,
            )
            self._vault.assert_absent_from(public, surface_name="transport.start")
            observed = self.transport.live_query(
                replica_connection,
                identity=identity,
                token=token,
            )
            self._assert_live_identity_observation(observed)
            if verify_request is not None:
                steps = verify_request["steps"]
                effects_present = self._mutation_effects_present(
                    verify_request["operation"],
                    steps,
                    connection=replica_connection,
                )
                if not effects_present:
                    raise QuackStateServerReadyError(
                        "read-replica lacks the committed mutation effects"
                    )
            self._read_replica_refresh_sequence += 1
            self._read_replica_observation = {
                "schema": "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1",
                "authority": "non_authoritative_read_replica",
                "path": str(self.read_replica_path()),
                "source_database_path": str(self.config.database_path),
                "server_id": identity.server_id,
                "database_uuid": identity.database_uuid,
                "generation": identity.generation,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "storage_schema_fingerprint": str(
                    self._mutation_binding().get("schema_fingerprint") or ""
                ),
                "sha256": digest,
                "size_bytes": size_bytes,
                "refresh_sequence": self._read_replica_refresh_sequence,
                "refreshed_at_ms": int(self.clock() * 1000),
                "live": True,
            }
            self._assert_database_namespace()
            return dict(self._read_replica_observation)
        except BaseException:
            try:
                self._stop_transport_connection(observe_closed=True)
            except Exception:
                pass
            if self._read_replica_observation:
                self._read_replica_observation["live"] = False
            raise

    def _unstall_stale_board_gates(self, connection: Any) -> None:
        """Reconcile and unstall through the one bound intent authority.

        Every operation occurs before transport/read-replica publication while
        this process holds the exclusive owner fence.  Integrity failures are
        startup failures; they are never converted to a warning.
        """

        identity = self._identity
        if identity is None:
            raise QuackStateServerReadyError(
                "board recovery requires the exact starting owner identity"
            )
        repository = IntentRepository(
            self.config.database_path,
            bound_connection=connection,
            owner_id="quack-state-owner",
            session_id=f"quack-owner-{identity.generation}",
            install_schema=False,
        )
        try:
            recovery = repository.reconcile_legacy_stale_unstall_projection_drift()
            if recovery.changed:
                candidates = recovery.details.get("candidates") or ()
                self._log(
                    "board projection recovery "
                    f"candidates={len(candidates)} event_id={recovery.event_id}"
                )
            result = repository.unstall_stale_in_progress_tasks(
                orphan_previous_generation=True
            )
            repository.assert_projection_matches_events()
        finally:
            repository.close()
        unstalled = result.get("unstalled") or []
        sanitized = result.get("sanitized_malformed_validation_retry_seeds") or []
        if not unstalled and not sanitized:
            return
        aliases = ",".join(
            str(item.get("task_alias") or item.get("task_cid") or "")
            for item in unstalled[:8]
            if isinstance(item, Mapping)
        )
        sanitized_aliases = ",".join(
            str(item.get("task_alias") or item.get("task_cid") or "")
            for item in sanitized[:8]
            if isinstance(item, Mapping)
        )
        self._log(
            f"board unstall gates={len(unstalled)} aliases={aliases} "
            f"malformed_seed_unstalls={len(sanitized)} "
            f"seed_aliases={sanitized_aliases}"
        )

    def _read_meta(self, connection: Any) -> dict[str, str]:
        def get(key: str) -> str:
            try:
                row = connection.execute(
                    "SELECT value FROM control_plane_metadata WHERE key = ?",
                    [key],
                ).fetchone()
            except Exception:
                return ""
            if row is None:
                return ""
            if isinstance(row, Mapping):
                return str(row.get("value") or "")
            return str(row[0] if row else "")

        fingerprint = get(META_SCHEMA_FINGERPRINT)
        if not fingerprint:
            try:
                fingerprint = compute_schema_fingerprint(connection)
            except Exception:
                fingerprint = ""
        fingerprint = _schema_fingerprint_digest(fingerprint)
        return {
            "database_uuid": get(META_DATABASE_UUID),
            "schema_version": get(META_SCHEMA_VERSION),
            "schema_fingerprint": fingerprint,
        }

    def _next_generation(self, connection: Any) -> int:
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM store_generations"
            ).fetchone()
        except Exception:
            return 1
        if row is None:
            return 1
        current = int(row[0] if not isinstance(row, Mapping) else row.get(list(row.keys())[0], 0))
        return max(1, current + 1)

    def _publish_identity_rows(
        self,
        connection: Any,
        identity: StateServerIdentity,
        capability: QuackCapabilityReport,
    ) -> None:
        started = identity.started_at or _utc_iso()
        # Best-effort inserts; tables exist after migration.
        try:
            connection.execute(
                """
                INSERT INTO store_generations (
                    generation, schema_revision, fence_epoch, revision,
                    database_uuid, birth_id, created_at, extension_schema, extension_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '', '{}')
                """,
                [
                    identity.generation,
                    identity.schema_revision,
                    identity.fence_epoch,
                    identity.revision,
                    identity.database_uuid,
                    identity.process_birth_id,
                    started,
                ],
            )
        except Exception:
            pass
        try:
            connection.execute(
                """
                INSERT INTO state_servers (
                    server_id, store_id, database_uuid, process_birth_id,
                    listen_uri, extension_fingerprint, schema_revision, generation,
                    started_at, stopped_at, status, revision, extension_schema, extension_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, '', '{}')
                """,
                [
                    identity.server_id,
                    identity.store_id,
                    identity.database_uuid,
                    identity.process_birth_id,
                    identity.listen_uri,
                    identity.extension_fingerprint,
                    identity.schema_revision,
                    identity.generation,
                    started,
                    identity.status,
                    identity.revision,
                ],
            )
        except Exception:
            pass
        try:
            connection.execute(
                """
                INSERT INTO server_epochs (
                    server_id, epoch, fence_epoch, started_at, ended_at
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                [
                    identity.server_id,
                    identity.startup_epoch or identity.generation,
                    identity.fence_epoch,
                    started,
                ],
            )
        except Exception:
            pass
        try:
            snapshot_id = f"cap:{identity.server_id}:{identity.generation}"
            body = json.dumps(
                {
                    "status": capability.status.value,
                    "profile_id": capability.profile.profile_id,
                    "extension_fingerprint": capability.extension_fingerprint,
                },
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO capability_snapshots (
                    snapshot_id, server_id, profile_id, duckdb_version,
                    extension_name, extension_fingerprint, status, observed_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    snapshot_id,
                    identity.server_id,
                    capability.profile.profile_id,
                    capability.duckdb_version or "",
                    capability.profile.extension_name,
                    capability.extension_fingerprint or "",
                    capability.status.value,
                    started,
                    body,
                ],
            )
        except Exception:
            pass
        try:
            # Store only the opaque handle + generation — never the token.
            connection.execute(
                """
                INSERT INTO credentials (
                    credential_id, secret_handle, generation, purpose,
                    created_at, rotated_at, revoked_at, revision
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                [
                    f"cred:{identity.server_id}:{identity.credential_generation}",
                    identity.secret_handle,
                    identity.credential_generation,
                    "quack-auth",
                    started,
                    identity.revision,
                ],
            )
        except Exception:
            pass

    def _mark_server_stopped(self, connection: Any, identity: StateServerIdentity) -> None:
        try:
            connection.execute(
                """
                UPDATE state_servers
                SET status = ?, stopped_at = ?, revision = revision + 1
                WHERE server_id = ?
                """,
                ["stopped", _utc_iso(), identity.server_id],
            )
        except Exception:
            pass
        try:
            connection.execute(
                """
                UPDATE server_epochs
                SET ended_at = ?
                WHERE server_id = ? AND ended_at IS NULL
                """,
                [_utc_iso(), identity.server_id],
            )
        except Exception:
            pass

    # -- closed owner-mutation inbox --------------------------------------

    def _mutation_binding(self) -> dict[str, Any]:
        if self._identity is None or self._connection is None:
            raise QuackStateServerMutationError("server_not_ready")
        schema_row = self._connection.execute(
            "SELECT value FROM control_plane_metadata "
            "WHERE key = 'schema_fingerprint'"
        ).fetchone()
        schema_fingerprint = str(schema_row[0] if schema_row else "")
        identity = self._identity
        return {
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "schema_revision": identity.schema_revision,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "listen_uri": identity.listen_uri,
            "extension_fingerprint": identity.extension_fingerprint,
            "schema_fingerprint": schema_fingerprint,
        }

    def _validate_mutation_request(
        self,
        payload: Mapping[str, Any],
        *,
        request_id: str,
        allow_expired: bool = False,
    ) -> dict[str, Any]:
        if set(payload) != _MUTATION_REQUEST_FIELDS:
            raise QuackStateServerMutationError("request_schema_invalid")
        if (
            payload.get("schema") != QUACK_OWNER_MUTATION_REQUEST_SCHEMA
            or payload.get("protocol_revision")
            != QUACK_OWNER_MUTATION_PROTOCOL_REVISION
            or payload.get("request_id") != request_id
        ):
            raise QuackStateServerMutationError("request_schema_invalid")
        operation = payload.get("operation")
        binding = payload.get("binding")
        steps = payload.get("steps")
        issued_at_ms = payload.get("issued_at_ms")
        expires_at_ms = payload.get("expires_at_ms")
        request_cid = payload.get("request_cid")
        auth_mac = payload.get("auth_mac")
        if (
            operation not in {
                QUACK_MUTATION_TASK_STATUS_TRANSITION,
                QUACK_MUTATION_VALIDATION_RECORD,
                QUACK_MUTATION_QUEUE_BACKOFF,
            }
            or not isinstance(binding, dict)
            or binding != self._mutation_binding()
            or not isinstance(steps, list)
            or not 1 <= len(steps) <= QUACK_OWNER_MUTATION_MAX_STEPS
            or type(issued_at_ms) is not int
            or type(expires_at_ms) is not int
            or expires_at_ms - issued_at_ms != QUACK_OWNER_MUTATION_REQUEST_TTL_MS
            or not isinstance(request_cid, str)
            or not isinstance(auth_mac, str)
        ):
            raise QuackStateServerMutationError("request_binding_invalid")
        now_ms = int(self.clock() * 1000)
        if issued_at_ms > now_ms + QUACK_OWNER_MUTATION_MAX_CLOCK_SKEW_MS:
            raise QuackStateServerMutationError("request_from_future")
        if not allow_expired and expires_at_ms < now_ms:
            raise QuackStateServerMutationError("request_expired")
        semantic = {
            "schema": "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-semantic@1",
            "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
            "operation": operation,
            "binding": binding,
            "steps": steps,
        }
        if request_id != quack_owner_mutation_content_id(semantic):
            raise QuackStateServerMutationError("request_identity_invalid")
        unsigned = dict(payload)
        unsigned.pop("auth_mac", None)
        unsigned.pop("request_cid", None)
        if request_cid != quack_owner_mutation_content_id(unsigned):
            raise QuackStateServerMutationError("request_cid_invalid")
        assert self._vault is not None and self._identity is not None
        token = self._vault.resolve(self._identity.secret_handle)
        authenticated = {**unsigned, "request_cid": request_cid}
        if not hmac.compare_digest(
            auth_mac, quack_owner_mutation_mac(authenticated, token)
        ):
            raise QuackStateServerMutationError("request_mac_invalid")
        return dict(payload)

    def _mutation_result(
        self,
        request: Mapping[str, Any],
        *,
        ok: bool,
        error_code: str = "",
        rowcounts: Sequence[int] = (),
        observed: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        unsigned = {
            "schema": QUACK_OWNER_MUTATION_RESULT_SCHEMA,
            "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
            "request_id": request["request_id"],
            "request_cid": request["request_cid"],
            "issued_at_ms": request["issued_at_ms"],
            "expires_at_ms": request["expires_at_ms"],
            "operation": request["operation"],
            "binding": dict(request["binding"]),
            "ok": bool(ok),
            "error_code": str(error_code or ""),
            "rowcounts": [int(value) for value in rowcounts][
                :QUACK_OWNER_MUTATION_MAX_STEPS
            ],
            "observed": dict(observed or {}),
        }
        result_cid = quack_owner_mutation_content_id(unsigned)
        authenticated = {**unsigned, "result_cid": result_cid}
        assert self._vault is not None and self._identity is not None
        token = self._vault.resolve(self._identity.secret_handle)
        return {
            **authenticated,
            "result_mac": quack_owner_mutation_mac(authenticated, token),
        }

    def _write_mutation_result(
        self, path: Path, request: Mapping[str, Any], **kwargs: Any
    ) -> None:
        receipt = self._mutation_result(request, **kwargs)
        encoded = canonical_json_bytes(receipt)
        if len(encoded) > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
            raise QuackStateServerMutationError("result_too_large")
        _atomic_write_text(path, encoded.decode("utf-8") + "\n", mode=0o600)

    def _existing_result_is_valid(
        self, path: Path, request: Mapping[str, Any]
    ) -> bool:
        try:
            payload = _read_bounded_canonical_json(path)
        except (OSError, QuackStateServerMutationError):
            return False
        fields = {
            "schema", "protocol_revision", "request_id", "request_cid",
            "issued_at_ms", "expires_at_ms",
            "operation", "binding", "ok", "error_code", "rowcounts",
            "observed", "result_cid", "result_mac",
        }
        if set(payload) != fields:
            return False
        result_cid = payload.get("result_cid")
        result_mac = payload.get("result_mac")
        unsigned = dict(payload)
        unsigned.pop("result_cid", None)
        unsigned.pop("result_mac", None)
        authenticated = {**unsigned, "result_cid": result_cid}
        assert self._vault is not None and self._identity is not None
        token = self._vault.resolve(self._identity.secret_handle)
        return bool(
            payload.get("schema") == QUACK_OWNER_MUTATION_RESULT_SCHEMA
            and payload.get("protocol_revision")
            == QUACK_OWNER_MUTATION_PROTOCOL_REVISION
            and payload.get("request_id") == request.get("request_id")
            and payload.get("request_cid") == request.get("request_cid")
            and payload.get("issued_at_ms") == request.get("issued_at_ms")
            and payload.get("expires_at_ms") == request.get("expires_at_ms")
            and payload.get("operation") == request.get("operation")
            and payload.get("binding") == request.get("binding")
            and isinstance(result_cid, str)
            and result_cid == quack_owner_mutation_content_id(unsigned)
            and isinstance(result_mac, str)
            and hmac.compare_digest(
                result_mac, quack_owner_mutation_mac(authenticated, token)
            )
        )

    @staticmethod
    def _validate_domain_event(parameters: list[Any]) -> dict[str, Any]:
        event_body = _canonical_object(parameters[9], code="event_body_invalid")
        if set(event_body) != {
            "schema", "event_type", "subject_id", "body", "recorded_at", "owner_id"
        }:
            raise QuackStateServerMutationError("event_body_invalid")
        if (
            event_body.get("event_type") != parameters[4]
            or event_body.get("recorded_at") != parameters[8]
            or not isinstance(event_body.get("body"), dict)
        ):
            raise QuackStateServerMutationError("event_body_invalid")
        expected_id = content_identity(
            {
                "stream_id": parameters[1],
                "sequence": parameters[2],
                "global_sequence": parameters[3],
                "event_type": parameters[4],
                "body": event_body,
            }
        )
        if parameters[0] != expected_id:
            raise QuackStateServerMutationError("event_identity_invalid")
        return event_body

    def _validate_event_head(self, parameters: list[Any]) -> None:
        assert self._connection is not None
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0), "
            "(SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events) "
            "FROM domain_events WHERE stream_id = ?",
            [parameters[1]],
        ).fetchone()
        if (
            row is None
            or type(parameters[2]) is not int
            or type(parameters[3]) is not int
            or parameters[2] != int(row[0]) + 1
            or parameters[3] != int(row[1]) + 1
        ):
            raise QuackStateServerMutationError("event_head_conflict")

    def _task_effects_present(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        connection: Any | None = None,
    ) -> bool:
        active = connection if connection is not None else self._connection
        assert active is not None
        update = _mutation_parameters(steps[0], 6)
        revision = _mutation_parameters(steps[1], 5)
        event = _mutation_parameters(steps[-1], 10)
        task = active.execute(
            "SELECT status, revision, updated_at, body_json "
            "FROM tasks WHERE task_cid = ?",
            [update[4]],
        ).fetchone()
        task_matches = task is not None and (
            task[0], int(task[1]), task[2], task[3]
        ) == (
            update[0], update[1], update[2], update[3]
        )
        revision_row = active.execute(
            "SELECT status, body_json, recorded_at FROM task_revisions "
            "WHERE task_cid = ? AND revision = ?",
            [revision[0], revision[1]],
        ).fetchone()
        event_row = active.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json "
            "FROM domain_events WHERE event_id = ?",
            [event[0]],
        ).fetchone()
        completion_row = None
        completion: list[Any] | None = None
        if len(steps) == 4:
            completion = _mutation_parameters(steps[2], 10)
            completion_row = active.execute(
                "SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, "
                "fencing_token, completed_at, validation_run_id, evidence_digest, "
                "body_json FROM completion_receipts WHERE receipt_cid = ?",
                [completion[0]],
            ).fetchone()
        revision_matches = revision_row is not None and tuple(
            revision_row[index] for index in range(3)
        ) == tuple(revision[2:5])
        event_matches = event_row is not None and tuple(
            event_row[index] for index in range(10)
        ) == tuple(event)
        completion_matches = completion is None or (
            completion_row is not None
            and tuple(completion_row[index] for index in range(10))
            == tuple(completion)
        )
        # The task-revision primary key is shared by legitimate competing CAS
        # requests, so a winner may occupy it before this request is checked.
        # Only the request-specific event identity establishes that this exact
        # semantic bundle was partially or wholly persisted.  Revision and
        # completion keys can be shared by legitimate competing CAS requests.
        if event_row is None:
            return False
        if event_row is not None and not event_matches:
            raise QuackStateServerMutationError("replay_integrity_failure")
        if not (task_matches and revision_matches and event_matches and completion_matches):
            raise QuackStateServerMutationError("replay_integrity_failure")
        return True

    def _validation_effects_present(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        connection: Any | None = None,
    ) -> bool:
        active = connection if connection is not None else self._connection
        assert active is not None
        run = _mutation_parameters(steps[0], 8)
        result = _mutation_parameters(steps[1], 7)
        event = _mutation_parameters(steps[-1], 10)
        run_row = active.execute(
            "SELECT run_id, task_cid, attempt_id, started_at, finished_at, status, "
            "command_digest, body_json FROM validation_runs WHERE run_id = ?",
            [run[0]],
        ).fetchone()
        result_row = active.execute(
            "SELECT result_id, run_id, task_cid, ordinal, outcome, evidence_digest, "
            "body_json FROM validation_results WHERE result_id = ?",
            [result[0]],
        ).fetchone()
        event_row = active.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json "
            "FROM domain_events WHERE event_id = ?",
            [event[0]],
        ).fetchone()
        evidence: list[Any] | None = None
        evidence_row = None
        if len(steps) == 5:
            evidence = _mutation_parameters(steps[3], 7)
            evidence_row = active.execute(
                "SELECT evidence_id, parent_evidence_id, task_cid, evidence_kind, "
                "digest, created_at, body_json FROM evidence_nodes "
                "WHERE evidence_id = ?",
                [evidence[0]],
            ).fetchone()
        persisted = [run_row, result_row, event_row]
        if evidence is not None:
            persisted.append(evidence_row)
        if not any(row is not None for row in persisted):
            return False
        exact = (
            run_row is not None
            and tuple(run_row[index] for index in range(8)) == tuple(run)
            and result_row is not None
            and tuple(result_row[index] for index in range(7)) == tuple(result)
            and event_row is not None
            and tuple(event_row[index] for index in range(10)) == tuple(event)
            and (
                evidence is None
                or (
                    evidence_row is not None
                    and tuple(evidence_row[index] for index in range(7))
                    == tuple(evidence)
                )
            )
        )
        if not exact:
            raise QuackStateServerMutationError("replay_integrity_failure")
        return True

    def _mutation_effects_present(
        self,
        operation: object,
        steps: Sequence[Mapping[str, Any]],
        *,
        connection: Any | None = None,
    ) -> bool:
        if operation == QUACK_MUTATION_TASK_STATUS_TRANSITION:
            return self._task_effects_present(steps, connection=connection)
        if operation == QUACK_MUTATION_VALIDATION_RECORD:
            return self._validation_effects_present(steps, connection=connection)
        if operation == QUACK_MUTATION_QUEUE_BACKOFF:
            return self._queue_backoff_effects_present(
                steps, connection=connection
            )
        raise QuackStateServerMutationError("operation_not_allowlisted")

    def _queue_backoff_effects_present(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        connection: Any | None = None,
    ) -> bool:
        active = connection if connection is not None else self._connection
        assert active is not None
        templates = [step.get("template_id") for step in steps]
        inserting = templates == [
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        updating = templates == [
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        if not inserting and not updating:
            raise QuackStateServerMutationError("operation_shape_invalid")
        lease = _mutation_parameters(steps[0], 17 if inserting else 6)
        event = _mutation_parameters(steps[-1], 10)
        event_row = active.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json "
            "FROM domain_events WHERE event_id = ?",
            [event[0]],
        ).fetchone()
        if inserting:
            lease_row = active.execute(
                "SELECT task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, "
                "state, started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision, extension_schema, "
                "extension_json FROM leases WHERE task_cid = ?",
                [lease[0]],
            ).fetchone()
            lease_matches = lease_row is not None and tuple(
                lease_row[index] for index in range(17)
            ) == tuple(lease)
        else:
            lease_row = active.execute(
                "SELECT attempt, retry_not_before_ms, release_reason, state, "
                "extension_schema, extension_json FROM leases WHERE task_cid = ?",
                [lease[5]],
            ).fetchone()
            lease_matches = lease_row is not None and (
                int(lease_row[0]),
                int(lease_row[1]),
                lease_row[2],
                lease_row[3],
                lease_row[4],
                lease_row[5],
            ) == (
                lease[0],
                lease[1],
                lease[2],
                "released",
                lease[3],
                lease[4],
            )
        event_matches = event_row is not None and tuple(
            event_row[index] for index in range(10)
        ) == tuple(event)
        if event_row is None:
            return False
        if not event_matches or not lease_matches:
            raise QuackStateServerMutationError("replay_integrity_failure")
        return True

    def _validate_task_transition(
        self, steps: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        templates = [step.get("template_id") for step in steps]
        completing = len(steps) == 4
        expected = [
            QUACK_MUTATION_TASK_STATUS_CAS,
            QUACK_MUTATION_TASK_REVISION_INSERT,
            *([QUACK_MUTATION_COMPLETION_RECEIPT_INSERT] if completing else []),
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        if templates != expected:
            raise QuackStateServerMutationError("operation_shape_invalid")
        update = _mutation_parameters(steps[0], 6)
        revision = _mutation_parameters(steps[1], 5)
        event = _mutation_parameters(steps[-1], 10)
        event_body = self._validate_domain_event(event)
        inner = event_body["body"]
        if not isinstance(inner, dict):
            raise QuackStateServerMutationError("event_body_invalid")
        assert self._connection is not None
        task = self._connection.execute(
            "SELECT task_alias, goal_cid, status, revision, body_json "
            "FROM tasks WHERE task_cid = ?", [update[4]]
        ).fetchone()
        if task is None:
            raise QuackStateServerMutationError("task_missing")
        old_status, old_revision = str(task[2]), int(task[3])
        if old_revision != update[5]:
            raise QuackStateServerMutationError("cas_conflict")
        allowed = update[0] in _MUTATION_ALLOWED_TO.get(old_status, frozenset())
        if not allowed or update[1] != update[5] + 1:
            raise QuackStateServerMutationError("transition_invalid")
        _canonical_object(update[3], code="task_body_invalid")
        if revision != [update[4], update[1], update[0], update[3], update[2]]:
            raise QuackStateServerMutationError("revision_binding_invalid")
        expected_event_type = (
            "intent.completion_recorded" if update[0] == "completed" else "intent.task_status_changed"
        )
        if completing != (update[0] == "completed"):
            raise QuackStateServerMutationError("completion_shape_invalid")
        if (
            event[1] != "stream:intent"
            or event[4] != expected_event_type
            or event[5] != update[4]
            or event_body["subject_id"] != update[4]
            or inner.get("task_cid") != update[4]
            or inner.get("task_alias") != str(task[0])
            or inner.get("goal_cid") != str(task[1])
            or inner.get("previous_status") != old_status
            or inner.get("status") != update[0]
            or inner.get("revision") != update[1]
            or inner.get("recorded_at") != update[2]
        ):
            raise QuackStateServerMutationError("event_binding_invalid")
        self._validate_event_head(event)
        if completing:
            completion = _mutation_parameters(steps[2], 10)
            completion_body = _canonical_object(
                completion[9], code="completion_receipt_invalid"
            )
            evidence_digests = completion_body.get("evidence_digests")
            completion_receipt = completion_body.get("receipt")
            if (
                set(completion_body)
                != {"schema", "receipt", "evidence_digests", "revision"}
                or completion_body.get("schema") != COMPLETION_EVIDENCE_SCHEMA
                or not isinstance(completion_receipt, dict)
                or completion[1] != update[4]
                or completion[2] != str(task[1])
                or completion[6] != update[2]
                or completion_body.get("revision") != update[1]
                or not isinstance(evidence_digests, list)
                or any(not isinstance(item, str) for item in evidence_digests)
                or inner.get("completion_receipt_cid") != completion[0]
                or inner.get("evidence_digest") != completion[8]
            ):
                raise QuackStateServerMutationError("completion_receipt_invalid")
            expected_evidence_digest = content_identity(
                {
                    "task_cid": update[4],
                    "revision": update[1],
                    "receipt": completion_receipt,
                    "evidence_digests": evidence_digests,
                }
            )
            if completion[8] != expected_evidence_digest:
                raise QuackStateServerMutationError("completion_receipt_invalid")
            expected_receipt = content_identity(
                {
                    "namespace": "completion-receipt",
                    "task_cid": update[4],
                    "revision": update[1],
                    "evidence_digest": completion[8],
                }
            )
            if completion[0] != expected_receipt:
                raise QuackStateServerMutationError("completion_receipt_invalid")
            missing_evidence = missing_current_evidence_on(
                self._connection,
                update[4],
                evidence_digests=evidence_digests,
                now_ms=int(self.clock() * 1000),
                evidence_freshness_seconds=DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
            )
            if missing_evidence:
                raise QuackStateServerMutationError("completion_evidence_stale")
        return {
            "task_cid": update[4],
            "old_status": old_status,
            "old_revision": old_revision,
            "new_status": update[0],
            "new_revision": update[1],
            "event_id": event[0],
        }, completing

    def _validate_validation_record(
        self, steps: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        passed = len(steps) == 5
        expected = [
            QUACK_MUTATION_VALIDATION_RUN_INSERT,
            QUACK_MUTATION_VALIDATION_RESULT_INSERT,
            *([QUACK_MUTATION_EVIDENCE_DELETE, QUACK_MUTATION_EVIDENCE_INSERT] if passed else []),
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        if [step.get("template_id") for step in steps] != expected:
            raise QuackStateServerMutationError("operation_shape_invalid")
        run = _mutation_parameters(steps[0], 8)
        result = _mutation_parameters(steps[1], 7)
        event = _mutation_parameters(steps[-1], 10)
        run_body = _canonical_object(run[7], code="validation_body_invalid")
        _canonical_object(result[6], code="validation_body_invalid")
        event_body = self._validate_domain_event(event)
        inner = event_body["body"]
        if not isinstance(inner, dict):
            raise QuackStateServerMutationError("event_body_invalid")
        argv = run_body.get("argv")
        if (
            not isinstance(argv, list)
            or any(not isinstance(item, str) for item in argv)
            or run[4] != run[3]
            or run[5] not in {"passed", "failed", "error", "skipped"}
            or result[1] != run[0]
            or result[2] != run[1]
            or result[3] != 0
            or result[4] != run[5]
            or passed != (run[5] == "passed")
            or run[6] != content_identity({"argv": argv})
            or run[0] != content_identity(
                {
                    "task_cid": run[1],
                    "attempt_id": run[2],
                    "argv": argv,
                    "recorded_at": run[3],
                }
            )
            or result[0] != content_identity(
                {"run_id": run[0], "outcome": result[4], "evidence_digest": result[5]}
            )
        ):
            raise QuackStateServerMutationError("validation_binding_invalid")
        assert self._connection is not None
        if self._connection.execute(
            "SELECT 1 FROM tasks WHERE task_cid = ?", [run[1]]
        ).fetchone() is None:
            raise QuackStateServerMutationError("task_missing")
        if (
            event[1] != "stream:intent"
            or event[4] != "intent.validation_recorded"
            or event[5] != run[1]
            or event[6] != run[2]
            or event_body["subject_id"] != result[0]
            or inner.get("result_id") != result[0]
            or inner.get("run_id") != run[0]
            or inner.get("task_cid") != run[1]
            or inner.get("outcome") != run[5]
            or inner.get("evidence_digest") != result[5]
        ):
            raise QuackStateServerMutationError("event_binding_invalid")
        self._validate_event_head(event)
        if passed:
            delete = _mutation_parameters(steps[2], 1)
            evidence = _mutation_parameters(steps[3], 7)
            evidence_body = _canonical_object(evidence[6], code="evidence_binding_invalid")
            expected_eid = content_identity(
                {
                    "task_cid": run[1],
                    "evidence_kind": "validation",
                    "digest": result[5],
                    "run_id": run[0],
                }
            )
            if (
                delete[0] != expected_eid
                or evidence[0] != expected_eid
                or evidence[1] != ""
                or evidence[2] != run[1]
                or evidence[3] != "validation"
                or evidence[4] != result[5]
                or evidence_body.get("run_id") != run[0]
                or evidence_body.get("result_id") != result[0]
                or evidence_body.get("outcome") != "passed"
            ):
                raise QuackStateServerMutationError("evidence_binding_invalid")
        return {
            "task_cid": run[1],
            "run_id": run[0],
            "result_id": result[0],
            "event_id": event[0],
            "outcome": run[5],
        }

    def _validate_queue_backoff(
        self, steps: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        templates = [step.get("template_id") for step in steps]
        inserting = templates == [
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        updating = templates == [
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ]
        if not inserting and not updating:
            raise QuackStateServerMutationError("operation_shape_invalid")
        lease = _mutation_parameters(steps[0], 17 if inserting else 6)
        event = _mutation_parameters(steps[-1], 10)
        event_body = self._validate_domain_event(event)
        inner = event_body["body"]
        if not isinstance(inner, dict):
            raise QuackStateServerMutationError("event_body_invalid")
        if inserting:
            task_cid = lease[0]
            attempt = lease[7]
            started_at_ms = lease[9]
            reason = lease[10]
            retry_not_before_ms = lease[11]
            owner_session_id = lease[12]
            extension_schema = lease[15]
            extension_json = lease[16]
            if (
                not isinstance(task_cid, str)
                or not task_cid
                or lease[1] != f"claim:queue:{task_cid}"
                or lease[2] != f"resolution:queue:{task_cid}"
                or not isinstance(lease[3], str)
                or lease[4] != 1
                or lease[5] != 1
                or lease[6] != 0
                or type(attempt) is not int
                or attempt < 1
                or lease[8] != "released"
                or type(started_at_ms) is not int
                or not isinstance(reason, str)
                or type(retry_not_before_ms) is not int
                or not isinstance(owner_session_id, str)
                or lease[13] != 1
                or lease[14] != 1
            ):
                raise QuackStateServerMutationError("lease_binding_invalid")
        else:
            task_cid = lease[5]
            attempt = lease[0]
            retry_not_before_ms = lease[1]
            reason = lease[2]
            extension_schema = lease[3]
            extension_json = lease[4]
            owner_session_id = event[7]
            started_at_ms = None
            if (
                type(attempt) is not int
                or attempt < 1
                or type(retry_not_before_ms) is not int
                or not isinstance(reason, str)
                or not isinstance(task_cid, str)
                or not task_cid
                or not isinstance(owner_session_id, str)
            ):
                raise QuackStateServerMutationError("lease_binding_invalid")
        extension = _canonical_object(extension_json, code="lease_binding_invalid")
        delay_ms = inner.get("delay_ms")
        selection_penalty = inner.get("selection_penalty")
        if (
            extension_schema != QUEUE_ENTRY_SCHEMA
            or set(extension) != {"selection_penalty", "consecutive_failures", "reason"}
            or extension.get("reason") != reason
            or extension.get("consecutive_failures") != attempt
            or extension.get("selection_penalty") != selection_penalty
            or type(delay_ms) is not int
            or delay_ms < 0
            or type(selection_penalty) is not int
            or selection_penalty < 0
            or (
                inserting
                and started_at_ms is not None
                and retry_not_before_ms != started_at_ms + delay_ms
            )
        ):
            raise QuackStateServerMutationError("lease_binding_invalid")
        if (
            event[1] != "stream:intent"
            or event[4] != "intent.queue_backoff"
            or event[5] != task_cid
            or event[7] != owner_session_id
            or event_body["subject_id"] != task_cid
            or inner.get("task_cid") != task_cid
            or inner.get("attempt") != attempt
            or inner.get("retry_not_before_ms") != retry_not_before_ms
            or inner.get("reason") != reason
            or inner.get("revision") != attempt
            or set(inner)
            != {
                "task_cid",
                "attempt",
                "retry_not_before_ms",
                "delay_ms",
                "selection_penalty",
                "reason",
                "revision",
            }
        ):
            raise QuackStateServerMutationError("event_binding_invalid")
        self._validate_event_head(event)
        assert self._connection is not None
        if self._connection.execute(
            "SELECT 1 FROM tasks WHERE task_cid = ?", [task_cid]
        ).fetchone() is None:
            raise QuackStateServerMutationError("task_missing")
        existing = self._connection.execute(
            "SELECT 1 FROM leases WHERE task_cid = ?", [task_cid]
        ).fetchone()
        if inserting and existing is not None:
            raise QuackStateServerMutationError("lease_conflict")
        if updating and existing is None:
            raise QuackStateServerMutationError("lease_missing")
        return {
            "task_cid": task_cid,
            "attempt": attempt,
            "retry_not_before_ms": retry_not_before_ms,
            "event_id": event[0],
            "inserted": inserting,
        }

    def _execute_mutation_request(
        self, request: Mapping[str, Any]
    ) -> tuple[list[int], dict[str, Any]]:
        assert self._connection is not None
        self._assert_database_namespace()
        steps = request["steps"]
        self._connection.execute("BEGIN TRANSACTION")
        try:
            if request["operation"] == QUACK_MUTATION_TASK_STATUS_TRANSITION:
                if self._task_effects_present(steps):
                    self._connection.execute("ROLLBACK")
                    update = _mutation_parameters(steps[0], 6)
                    observed = {
                        "task_cid": update[4], "new_status": update[0],
                        "new_revision": update[1], "idempotent_replay": True,
                    }
                    return [1] * len(steps), self._settle_mutation_replica(
                        request, observed=observed
                    )
                observed, _completing = self._validate_task_transition(steps)
            elif request["operation"] == QUACK_MUTATION_VALIDATION_RECORD:
                if self._validation_effects_present(steps):
                    self._connection.execute("ROLLBACK")
                    run = _mutation_parameters(steps[0], 8)
                    observed = {
                        "task_cid": run[1], "run_id": run[0],
                        "idempotent_replay": True,
                    }
                    return [1] * len(steps), self._settle_mutation_replica(
                        request, observed=observed
                    )
                observed = self._validate_validation_record(steps)
            elif request["operation"] == QUACK_MUTATION_QUEUE_BACKOFF:
                if self._queue_backoff_effects_present(steps):
                    self._connection.execute("ROLLBACK")
                    event = _mutation_parameters(steps[-1], 10)
                    inserting = (
                        steps[0].get("template_id")
                        == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT
                    )
                    lease = _mutation_parameters(steps[0], 17 if inserting else 6)
                    observed = {
                        "task_cid": lease[0] if inserting else lease[5],
                        "event_id": event[0],
                        "idempotent_replay": True,
                    }
                    return [1] * len(steps), self._settle_mutation_replica(
                        request, observed=observed
                    )
                observed = self._validate_queue_backoff(steps)
            else:
                raise QuackStateServerMutationError("operation_not_allowlisted")
            rowcounts: list[int] = []
            for index, step in enumerate(steps):
                template_id = str(step["template_id"])
                sql = _MUTATION_SQL_TEMPLATES.get(template_id)
                if sql is None:
                    raise QuackStateServerMutationError("template_not_allowlisted")
                cursor = self._connection.execute(sql, step["parameters"])
                rowcount = int(getattr(cursor, "rowcount", -1))
                if index == 0 and request["operation"] == QUACK_MUTATION_TASK_STATUS_TRANSITION and rowcount != 1:
                    raise QuackStateServerMutationError("cas_conflict")
                rowcounts.append(rowcount)
            self._assert_database_namespace()
            self._connection.execute("COMMIT")
            self._assert_database_namespace()
            return rowcounts, self._settle_mutation_replica(
                request, observed=observed
            )
        except BaseException:
            try:
                self._connection.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def _settle_mutation_replica(
        self,
        request: Mapping[str, Any],
        *,
        observed: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Require a fresh, live, exact replica projection before success."""

        self._assert_database_namespace()
        result = dict(observed)
        try:
            result["read_replica"] = self._refresh_read_replica(
                verify_request=request
            )
            # Status is the handle resolver's independently read freshness
            # binding and must settle before the signed mutation result.
            self._write_status()
        except QuackStateServerOwnershipError:
            raise
        except BaseException as exc:
            try:
                self._stop_transport_connection(observe_closed=True)
            except Exception:
                pass
            failure_observation = dict(self._read_replica_observation)
            failure_observation["live"] = False
            raise QuackStateServerMutationError(
                "read_replica_refresh_unknown_outcome",
                observed={
                    **result,
                    "canonical_effects_present": True,
                    "read_replica": failure_observation,
                    "refresh_failure_class": type(exc).__name__,
                },
            ) from exc
        self._assert_database_namespace()
        return result

    def _recover_stale_mutation_claims(self, inbox: Path) -> None:
        now = time.time()
        for path in tuple(inbox.iterdir())[:MUTATION_MAX_DIRECTORY_ENTRIES]:
            match = MUTATION_PROCESSING_NAME.fullmatch(path.name)
            if match is None:
                continue
            done = inbox / f"{match.group('request_id')}.done.json"
            try:
                age_ms = (now - path.stat().st_mtime) * 1000
            except OSError:
                continue
            if age_ms < QUACK_OWNER_MUTATION_REQUEST_TTL_MS:
                continue
            request: dict[str, Any] | None = None
            try:
                request = self._validate_mutation_request(
                    _read_bounded_canonical_json(path),
                    request_id=match.group("request_id"),
                    allow_expired=True,
                )
                if done.is_file() and self._existing_result_is_valid(done, request):
                    path.unlink(missing_ok=True)
                    continue
                done.unlink(missing_ok=True)
                effects = self._mutation_effects_present(
                    request["operation"], request["steps"]
                )
                observed = {"reconciled_after_interruption": True}
                if effects:
                    observed = self._settle_mutation_replica(
                        request,
                        observed=observed,
                    )
                self._write_mutation_result(
                    done,
                    request,
                    ok=effects,
                    error_code="" if effects else "unknown_external_outcome",
                    rowcounts=[1] * len(request["steps"]) if effects else [],
                    observed=observed,
                )
            except QuackStateServerOwnershipError:
                raise
            except QuackStateServerMutationError as exc:
                if request is not None:
                    self._write_mutation_result(
                        done,
                        request,
                        ok=False,
                        error_code=exc.code,
                        rowcounts=[],
                        observed={
                            "reconciled_after_interruption": True,
                            **exc.observed,
                        },
                    )
                    if exc.code == "read_replica_refresh_unknown_outcome":
                        self._lifecycle = ServerLifecycle.FAILED
                        self._write_status()
            except Exception:
                # Unauthenticated or unreadable claims never receive a signed
                # oracle response.
                pass
            finally:
                path.unlink(missing_ok=True)

    def service_mutation_inbox(self, *, max_requests: int = MUTATION_MAX_PER_PASS) -> int:
        """Claim and execute bounded, authenticated mutation bundles."""

        if type(max_requests) is not int or not 1 <= max_requests <= MUTATION_MAX_PER_PASS:
            raise ValueError("max_requests is outside the closed service bound")
        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY or self._connection is None:
                raise QuackStateServerNotRunningError("state-owner is not ready")
            self._assert_database_namespace()
            inbox = self.mutation_inbox_path()
            inbox.mkdir(parents=True, exist_ok=True)
            if inbox.is_symlink():
                raise QuackStateServerReadyError(
                    "mutation inbox is not a safe owner directory"
                )
            os.chmod(inbox, 0o700)
            entries = tuple(inbox.iterdir())
            if len(entries) > MUTATION_MAX_DIRECTORY_ENTRIES:
                raise QuackStateServerMutationError("inbox_population_exceeded")
            self._recover_stale_mutation_claims(inbox)
            if self._lifecycle is not ServerLifecycle.READY:
                return 0
            serviced = 0
            for request_path in sorted(entries, key=lambda item: item.name):
                if serviced >= max_requests:
                    break
                match = MUTATION_REQUEST_NAME.fullmatch(request_path.name)
                if match is None:
                    continue
                request_id = match.group("request_id")
                processing = inbox / f"{request_id}.processing.json"
                done = inbox / f"{request_id}.done.json"
                claimed = False
                try:
                    if processing.exists():
                        continue
                    try:
                        # Claim the pathname before reading any bytes.  This
                        # competes atomically with client cancellation, so an
                        # owner can execute only the inode it actually won.
                        os.replace(request_path, processing)
                        claimed = True
                    except FileNotFoundError:
                        continue
                    payload = _read_bounded_canonical_json(processing)
                    request = self._validate_mutation_request(
                        payload, request_id=request_id
                    )
                    if done.is_file() and self._existing_result_is_valid(done, request):
                        # Exact deterministic replay consumes no second effect;
                        # the client authenticates the retained receipt.
                        processing.unlink(missing_ok=True)
                        serviced += 1
                        continue
                    done.unlink(missing_ok=True)
                    try:
                        rowcounts, observed = self._execute_mutation_request(request)
                    except QuackStateServerOwnershipError:
                        raise
                    except QuackStateServerMutationError as exc:
                        self._write_mutation_result(
                            done, request, ok=False, error_code=exc.code,
                            rowcounts=[], observed=exc.observed,
                        )
                        if exc.code == "read_replica_refresh_unknown_outcome":
                            self._lifecycle = ServerLifecycle.FAILED
                            self._write_status()
                    except Exception:
                        self._write_mutation_result(
                            done, request, ok=False, error_code="owner_transaction_failed",
                            rowcounts=[], observed={},
                        )
                    else:
                        self._assert_database_namespace()
                        self._write_mutation_result(
                            done, request, ok=True, rowcounts=rowcounts,
                            observed=observed,
                        )
                    finally:
                        processing.unlink(missing_ok=True)
                    serviced += 1
                except QuackStateServerOwnershipError:
                    raise
                except (OSError, QuackStateServerMutationError, ValueError):
                    # Unauthenticated/malformed files receive no signed oracle.
                    if claimed:
                        processing.unlink(missing_ok=True)
                    else:
                        request_path.unlink(missing_ok=True)
                    serviced += 1
            self._assert_database_namespace()
            return serviced

    def _process_typed_owner_commands(self, *, max_requests: int) -> int:
        """Apply closed task commands through the canonical intent repository."""

        assert self._connection is not None
        assert self._identity is not None
        assert self._vault is not None
        self._assert_database_namespace()
        token = self._vault.resolve(self._identity.secret_handle)
        inbox_fd = self._prepare_mutation_inbox()
        serviced = 0
        try:
            request_names = sorted(
                name
                for name in os.listdir(inbox_fd)
                if TYPED_OWNER_COMMAND_REQUEST_NAME.fullmatch(name)
            )[:max_requests]
            for request_name in request_names:
                match = TYPED_OWNER_COMMAND_REQUEST_NAME.fullmatch(request_name)
                assert match is not None
                request_id = match.group("request_id")
                done_name = f"{request_id}.done.json"
                if mutation_envelope_exists_at(inbox_fd, done_name):
                    # A completion is the durable idempotency tombstone.  The
                    # authenticated client retires both files after admission.
                    continue
                try:
                    request = read_envelope_at(
                        inbox_fd,
                        request_name,
                        max_bytes=QUACK_OWNER_COMMAND_MAX_BYTES,
                    )
                    if request.get("schema") != QUACK_OWNER_COMMAND_REQUEST_SCHEMA:
                        raise QuackOwnerMutationEnvelopeError(
                            "typed owner command schema is not admitted"
                        )
                    command, payload = validate_quack_owner_command_request(
                        request,
                        token=token,
                        expected_request_id=request_id,
                        expected_store_id=self._identity.store_id,
                        expected_store_generation=str(self._identity.generation),
                    )
                except (
                    OSError,
                    ValueError,
                    QuackOwnerMutationEnvelopeError,
                ):
                    # Unauthenticated input receives no signed oracle and is
                    # removed so it cannot starve the bounded owner inbox.
                    unlink_mutation_envelope_at(
                        inbox_fd,
                        request_name,
                        missing_ok=True,
                    )
                    serviced += 1
                    continue

                repository = IntentRepository(
                    self.config.database_path,
                    bound_connection=self._connection,
                    owner_id="quack-state-owner",
                    session_id=f"quack-owner-{self._identity.generation}",
                    install_schema=False,
                )
                try:
                    try:
                        self._assert_database_namespace()
                        result = execute_quack_owner_command(
                            repository,
                            command,
                            payload,
                            request_id=request_id,
                            store_id=self._identity.store_id,
                            store_generation=str(self._identity.generation),
                        )
                        self._assert_database_namespace()
                    except QuackStateServerOwnershipError:
                        raise
                    except Exception as exc:
                        response = quack_owner_command_response(
                            request,
                            token=token,
                            error_code=quack_owner_command_error_code(exc),
                            error_message="typed owner command rejected",
                        )
                    else:
                        # The typed task command commits on the exclusive
                        # writer connection.  Do not acknowledge that effect
                        # while Quack still serves the pre-command snapshot:
                        # strict claim admission immediately re-reads the task
                        # and must observe the exact claim receipt/revision.
                        # This is the same synchronous publication barrier used
                        # by the closed mutation-bundle path.
                        try:
                            self._refresh_read_replica()
                            self._write_status()
                        except BaseException as refresh_exc:
                            try:
                                self._stop_transport_connection(
                                    observe_closed=True
                                )
                            except Exception:
                                pass
                            if self._read_replica_observation:
                                self._read_replica_observation["live"] = False
                            self._lifecycle = ServerLifecycle.FAILED
                            self._log(
                                "typed owner command replica publication "
                                "failed: " + type(refresh_exc).__name__
                            )
                            try:
                                self._write_status()
                            except Exception:
                                pass
                            # Publish no response: the exact signed request
                            # remains durable and its owner-side idempotency
                            # record permits recovery to replay publication
                            # without replaying the effect.
                            raise QuackStateServerMutationError(
                                "read_replica_refresh_unknown_outcome",
                                observed={
                                    "canonical_effects_present": True,
                                    "read_replica": dict(
                                        self._read_replica_observation
                                    ),
                                    "refresh_failure_class": type(
                                        refresh_exc
                                    ).__name__,
                                },
                            ) from refresh_exc
                        else:
                            response = quack_owner_command_response(
                                request,
                                token=token,
                                result=result,
                            )
                finally:
                    repository.close()
                self._assert_database_namespace()
                write_envelope_atomic_at(
                    inbox_fd,
                    done_name,
                    response,
                    replace=False,
                )
                serviced += 1
        finally:
            os.close(inbox_fd)
        self._assert_database_namespace()
        return serviced

    def service_database_task_command_inbox(
        self,
        *,
        expected_store_generation: str,
        max_requests: int = MUTATION_MAX_PER_PASS,
    ) -> Mapping[str, int]:
        """Apply closed DatabaseTaskSource commands on the owner connection.

        The raw transport credential and writable DuckDB handle remain inside
        this state-owner process. Callers provide only the sealed logical
        generation; the inbox service accepts the closed task-command schema
        and routes every effect through IntentRepository transition gates.
        """

        if type(max_requests) is not int or not 1 <= max_requests <= MUTATION_MAX_PER_PASS:
            raise ValueError("task command service bound is invalid")
        expected_generation = str(expected_store_generation or "").strip()
        if not expected_generation:
            raise QuackStateServerControlError(
                "task command service lacks the logical store generation"
            )
        with self._lock:
            if (
                self._lifecycle is not ServerLifecycle.READY
                or self._connection is None
                or self._identity is None
                or self._vault is None
            ):
                raise QuackStateServerNotRunningError("state-owner is not ready")
            if str(self._identity.generation) != expected_generation:
                raise QuackStateServerControlError(
                    "task command logical generation is stale"
                )
            self._assert_database_namespace()
            serviced = self._process_typed_owner_commands(
                max_requests=max_requests,
            )
            self._assert_database_namespace()
            return MappingProxyType({"serviced": serviced})

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> StateServerIdentity:
        """Acquire exclusive ownership, migrate, serve, and publish identity."""

        with self._lock:
            if self._lifecycle in {ServerLifecycle.READY, ServerLifecycle.STARTING}:
                if self._identity is not None:
                    return self._identity
                raise QuackStateServerError("server is starting without identity")
            if (
                self._owner is not None
                or self._database_namespace_anchor is not None
            ):
                raise QuackStateServerOwnershipError(
                    "prior failed owner retains database namespace authority"
                )

            self._lifecycle = ServerLifecycle.STARTING
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            self.config.database_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                assert_bind_admitted(
                    self.config.host,
                    remote_policy=self.config.remote_bind_policy,
                )
                # Container isolation is an independently observed authority
                # gate.  It must precede every database mutation, including
                # schema installation, and is carried into connection birth so
                # it cannot be replaced by a second receipt-shaped assertion.
                isolation_admission = self._admit_isolated_owner()
                self._isolation_admission = (
                    None
                    if isolation_admission is None
                    else dict(isolation_admission)
                )
                birth = (
                    self.process_birth_factory()
                    if self.process_birth_factory is not None
                    else current_process_birth()
                )
                server_id = f"server:{uuid.uuid4()}"
                owner = ExclusiveOwnerLease(
                    lock_path=self.owner_lock_path(),
                    marker_path=self.owner_marker_path(),
                    liveness=self.owner_liveness_probe,
                )
                # Generation is finalized after opening the DB; provisional 1.
                owner.acquire(
                    server_id=server_id,
                    process_birth=birth,
                    database_path=self.config.database_path,
                    generation=1,
                )
                self._owner = owner
                # A previous generation's ready projection must never survive
                # as a launch signal while this generation is qualifying.
                self.status_path().unlink(missing_ok=True)

                # The OS lease/fence is the exact single-writer gate.  No
                # migration or database connection may be reached by a losing
                # concurrent starter.
                owner.assert_canonical_parent()
                self._open_database_parent_anchor()
                self._assert_database_namespace(require_database=False)
                capability = self._admit_capability()
                self._capability = capability
                self._log(
                    f"capability admitted status={capability.status.value} "
                    f"fingerprint={capability.extension_fingerprint or 'none'}"
                )

                self._assert_database_namespace(require_database=False)
                migration = self._ensure_migrated()
                self._assert_database_namespace(require_database=False)
                self._bind_database_inode_after_migration()
                self._migration_report = migration
                self._log("control-plane schema migration complete before ready")

                self._assert_database_namespace()
                connection = self._open_connection(
                    isolation_admission=isolation_admission
                )
                self._connection = connection
                self._assert_database_namespace()
                owner.assert_canonical_parent()
                meta = self._read_meta(connection)
                database_uuid = meta.get("database_uuid") or str(uuid.uuid4())
                schema_fingerprint = meta.get("schema_fingerprint") or ""
                if not schema_fingerprint:
                    raise QuackStateServerMigrationError(
                        "schema fingerprint missing after migration"
                    )
                try:
                    schema_revision = int(meta.get("schema_version") or 0)
                except ValueError:
                    schema_revision = CONTROL_PLANE_SCHEMA_REVISION
                if schema_revision < 1:
                    raise QuackStateServerMigrationError(
                        "schema must be migrated before ready "
                        f"(schema_version={schema_revision})"
                    )

                generation = self._next_generation(connection)
                owner.bind_generation(generation)
                port = int(self.config.port) or _allocate_loopback_port(
                    self.config.host
                    if _is_loopback_host(self.config.host)
                    else DEFAULT_LOOPBACK_HOST
                )
                self._bound_port = port
                uri = listen_uri(self.config.host, port)
                secret_handle = self.config.resolved_secret_handle(server_id, generation)
                assert self._vault is not None
                self._vault.mint(secret_handle=secret_handle, generation=1)
                token = self._vault.resolve(secret_handle)

                identity = StateServerIdentity(
                    server_id=server_id,
                    store_id=self.config.store_id,
                    database_uuid=database_uuid,
                    schema_revision=schema_revision,
                    schema_fingerprint=schema_fingerprint,
                    generation=generation,
                    fence_epoch=generation,
                    revision=0,
                    process_birth=birth,
                    listen_uri=uri,
                    extension_fingerprint=capability.extension_fingerprint or "",
                    credential_generation=1,
                    secret_handle=secret_handle,
                    repository_id=self.config.repository_id
                    or f"repository:{self.config.store_id}",
                    startup_epoch=int(self.clock()),
                    started_at=_utc_iso(),
                    status="starting",
                )
                self._identity = identity

                assert self.transport is not None
                # Publish identity before quack_serve occupies this connection.
                # Auth callbacks open a fresh DuckDB session; DML on the serve
                # connection after listen starts is reported as Authentication
                # failed rather than lock contention.
                self._publish_identity_rows(connection, identity, capability)
                # Last exclusive-writer window: quack_serve occupies this
                # connection and later DML contends with auth callbacks.
                self._unstall_stale_board_gates(connection)
                public_obs = self.transport.start(
                    connection,
                    host=self.config.host,
                    port=port,
                    token=token,
                    identity=identity,
                )
                # Track the provisional writer-backed endpoint so the replica
                # refresh stops that exact listener before starting the
                # canonical read-only endpoint.  Without this binding two
                # Quack listeners can survive under one advertised URI, making
                # readiness flaky and shutdown observationally false.
                self._transport_connection = connection
                self._transport_connection_is_writer = True
                # Ensure transport observation never echoed the token.
                self._vault.assert_absent_from(public_obs, surface_name="transport.start")
                # A supervisor must never be able to reuse the HTTP Quack
                # credential to obtain a generic SQL surface.  Only the owner
                # retains it after identity mint; the replica transport later
                # resolves the in-process token.
                self._vault.remove_persisted_copy()

                identity = identity.with_status("ready")
                ready_update = connection.execute(
                    "UPDATE state_servers SET status = 'ready', revision = revision + 1 "
                    "WHERE server_id = ? AND generation = ? AND status = 'starting'",
                    [identity.server_id, identity.generation],
                )
                if getattr(ready_update, "rowcount", -1) not in {-1, 1}:
                    raise QuackStateServerMigrationError(
                        "could not publish the exact ready server generation"
                    )
                self._identity = identity
                # The endpoint serves only a checkpointed, atomically replaced,
                # non-authoritative read-only replica containing this exact
                # ready identity.  Startup fails closed if any proof fails.
                self._refresh_read_replica()
                gateway = TypedStateOwnerGateway(
                    connection=connection,
                    socket_path=self.typed_command_socket_path(),
                    store_id=identity.store_id,
                    identity=identity.to_dict(),
                )
                status_bootstrap_token = gateway.configure_status_bootstrap()
                gateway.bind_database_task_command_handler(
                    self._gateway_execute_database_task_command
                )
                gateway.start()
                self._command_gateway = gateway
                if self._event_wait is not None:
                    gateway.bind_event_wait_handlers(
                        wait=self._gateway_wait_for_events,
                        cancel=self._gateway_cancel_event_wait,
                        clear_cancellation=(
                            self._gateway_clear_event_wait_cancellation
                        ),
                    )
                _atomic_write_text(
                    self.typed_command_token_path(),
                    status_bootstrap_token,
                    mode=0o600,
                )
                self._assert_database_namespace()
                self._lifecycle = ServerLifecycle.READY
                # READY is a live remote claim: prove the authenticated Quack
                # data path before publishing status or returning to a launcher.
                self.ready()
                self._write_status()
                self._log(
                    f"state-owner ready server_id={identity.server_id} "
                    f"listen_uri={identity.listen_uri}"
                )
                # Final token absence checks on published surfaces.
                self._vault.assert_absent_from(self.status(), surface_name="status")
                self._vault.assert_absent_from(self.logs(), surface_name="logs")
                return identity
            except Exception as exc:
                self._lifecycle = ServerLifecycle.FAILED
                self._log(f"state-owner start failed: {type(exc).__name__}")
                self._emergency_cleanup()
                try:
                    self._write_status()
                except Exception:
                    pass
                raise

    def _emergency_cleanup(self) -> None:
        self._outbox_stop.set()
        if self._outbox_wake is not None:
            self._outbox_wake.shutdown()
        if self._outbox_thread is not None:
            self._outbox_thread.join(timeout=3.0)
        if self._event_wait is not None:
            self._event_wait.shutdown()
        try:
            if self._grant_broker is not None:
                self._grant_broker.stop()
        finally:
            self._grant_broker = None
            if self._grant_broker_secret_fd >= 0:
                try:
                    os.close(self._grant_broker_secret_fd)
                except OSError:
                    pass
                self._grant_broker_secret_fd = -1
        try:
            if self._command_gateway is not None:
                self._command_gateway.stop()
        except Exception:
            pass
        self._command_gateway = None
        try:
            self.typed_command_token_path().unlink()
        except FileNotFoundError:
            pass
        endpoint_closed = False
        try:
            self._stop_transport_connection(observe_closed=True)
            endpoint_closed = True
        except Exception as exc:
            self._log(
                "emergency transport closure was not observed: "
                + type(exc).__name__
            )
        database_closed = self._connection is None
        try:
            self._close_authoritative_connection_observed()
            database_closed = True
        except Exception as exc:
            self._log(
                "emergency database closure was not observed: "
                + type(exc).__name__
            )
        if not endpoint_closed:
            try:
                # Closing the owning connection can finish an asynchronous or
                # failed transport stop.  Admit that outcome only from a fresh
                # socket observation made after the database close attempt.
                self._wait_for_transport_endpoint_closed()
                endpoint_closed = True
            except Exception:
                pass
        try:
            if self._vault is not None:
                self._vault.destroy()
        except Exception:
            pass
        if self._owner is not None:
            if (
                endpoint_closed
                and database_closed
                and not self._database_namespace_drifted
            ):
                try:
                    self._owner.release()
                except QuackStateServerOwnershipError as exc:
                    self._database_namespace_drifted = True
                    self._log(
                        "emergency database namespace authority retained: "
                        + type(exc).__name__
                    )
                except Exception:
                    # Both externally relevant resources are now observed inert.
                    # Do not leak the kernel path fence if marker cleanup fails.
                    self._owner.emergency_close_after_owner_shutdown()
                    self._owner = None
                    self._close_database_namespace_anchor()
                else:
                    self._owner = None
                    self._close_database_namespace_anchor()
            else:
                # Process exit remains a safe kernel cleanup boundary.  An
                # in-process restart is not admitted while either external
                # resource may still be live.
                self._log(
                    "emergency owner authority retained until endpoint and "
                    "database closure and exact database namespace are observed"
                )
        elif endpoint_closed and database_closed:
            self._close_database_namespace_anchor()

    def ready(self) -> dict[str, Any]:
        """Return readiness observation or raise if not ready.

        Ready requires:
        * lifecycle is READY
        * live transport query succeeds
        * store / generation / schema / server identities match the published set
        """

        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY or self._identity is None:
                raise QuackStateServerReadyError(
                    f"state-owner is not ready (lifecycle={self._lifecycle.value})"
                )
            if (
                self._connection is None
                or self._transport_connection is None
                or self.transport is None
                or self._vault is None
            ):
                raise QuackStateServerReadyError("state-owner missing connection/transport")
            self._assert_database_namespace()
            gateway_health = (
                {} if self._command_gateway is None
                else self._command_gateway.capability()
            )
            if (
                self._command_gateway is None
                or gateway_health.get("available") is not True
                or gateway_health.get("database_task_command_bound") is not True
                or gateway_health.get("last_observer_error_type")
            ):
                raise QuackStateServerReadyError(
                    "typed owner command gateway is unavailable"
                )
            if self._grant_broker is not None and not self._grant_broker.alive():
                raise QuackStateServerReadyError(
                    "configured supervisor credential broker is unavailable"
                )
            federation_event_path_bound = bool(
                self._federation_repository is not None
                or self._outbox_worker is not None
            )
            if (
                federation_event_path_bound
                and gateway_health.get("commit_observer_bound") is not True
            ):
                raise QuackStateServerReadyError(
                    "federation commit observer is unavailable"
                )
            if self._outbox_worker is not None:
                outbox_health = self.outbox_worker_capability()
                if (
                    outbox_health.get("available") is not True
                    or outbox_health.get("thread_alive") is not True
                    or outbox_health.get("last_error_type")
                ):
                    raise QuackStateServerReadyError(
                        "state-owner outbox worker is unavailable"
                    )

            identity = self._identity
            token = self._vault.resolve(identity.secret_handle)
            observed = self.transport.live_query(
                self._transport_connection,
                identity=identity,
                token=token,
            )
            meta = self._read_meta(self._connection)
            if meta.get("database_uuid") and meta["database_uuid"] != identity.database_uuid:
                raise QuackStateServerReadyError(
                    "database_uuid drift between live store and published identity"
                )
            if meta.get("schema_fingerprint") and meta[
                "schema_fingerprint"
            ] != identity.schema_fingerprint:
                raise QuackStateServerReadyError(
                    "schema_fingerprint drift between live store and published identity"
                )
            try:
                live_schema_revision = int(meta.get("schema_version") or identity.schema_revision)
            except ValueError:
                live_schema_revision = identity.schema_revision
            if live_schema_revision != identity.schema_revision:
                raise QuackStateServerReadyError(
                    "schema_revision drift between live store and published identity"
                )
            replica_meta = self._read_meta(self._transport_connection)
            if replica_meta != meta:
                raise QuackStateServerReadyError(
                    "read-replica metadata differs from authoritative writer"
                )
            self._assert_live_identity_observation(observed)
            replica = self._read_replica_observation
            if (
                replica.get("live") is not True
                or replica.get("path") != str(self.read_replica_path())
                or replica.get("source_database_path")
                != str(self.config.database_path)
                or replica.get("server_id") != identity.server_id
                or replica.get("database_uuid") != identity.database_uuid
                or replica.get("generation") != identity.generation
                or replica.get("schema_revision") != identity.schema_revision
                or replica.get("schema_fingerprint")
                != identity.schema_fingerprint
                or replica.get("storage_schema_fingerprint")
                != self._mutation_binding().get("schema_fingerprint")
            ):
                raise QuackStateServerReadyError(
                    "read-replica freshness binding is absent or stale"
                )

            result = {
                "ready": True,
                "server_id": identity.server_id,
                "store_id": identity.store_id,
                "generation": identity.generation,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "database_uuid": identity.database_uuid,
                "process_birth_id": identity.process_birth_id,
                "listen_uri": identity.listen_uri,
                "secret_handle": identity.secret_handle,
                "live": True,
                "read_replica": dict(replica),
            }
            sanitized = sanitize_for_export(result, token=token)
            self._vault.assert_absent_from(sanitized, surface_name="ready")
            self._assert_database_namespace()
            return sanitized

    def is_ready(self) -> bool:
        try:
            self.ready()
            return True
        except QuackStateServerError:
            return False

    def checkpoint(self) -> dict[str, Any]:
        """Force a clean DuckDB checkpoint while owning the database."""

        with self._lock:
            if self._lifecycle is not ServerLifecycle.READY or self._connection is None:
                raise QuackStateServerNotRunningError(
                    "checkpoint requires a ready state-owner"
                )
            try:
                self._assert_database_namespace()
                self._connection.execute("CHECKPOINT")
                self._assert_database_namespace()
            except Exception as exc:
                if isinstance(exc, QuackStateServerOwnershipError):
                    raise
                raise QuackStateServerError(
                    f"checkpoint failed: {type(exc).__name__}"
                ) from exc
            receipt = {
                "checkpointed": True,
                "server_id": self._identity.server_id if self._identity else "",
                "database_path": str(self.config.database_path),
                "at": _utc_iso(),
            }
            return sanitize_for_export(receipt)

    def _assert_live_stop_authority(
        self,
        *,
        identity: StateServerIdentity,
        owner: ExclusiveOwnerLease,
        server_id: str,
        fence_token: str,
    ) -> OwnerMarker:
        """Bind a stop capability to the lease, marker, and live identity."""

        marker = owner.assert_stop_authority(
            server_id=server_id,
            fence_token=fence_token,
        )
        if (
            self._identity is not identity
            or self._owner is not owner
            or identity.server_id != server_id
            or marker.server_id != identity.server_id
            or marker.process_birth != identity.process_birth
            or int(marker.generation) != int(identity.generation)
            or marker.database_path != str(self.config.database_path)
        ):
            raise QuackStateServerControlError(
                "stop authority does not match the live state-owner identity"
            )
        return marker

    def _admit_stop_before_effects(
        self,
        *,
        fence_token: str | None,
    ) -> tuple[StateServerIdentity, ExclusiveOwnerLease, str]:
        """Select and validate one stop authority without mutating live state."""

        identity = self._identity
        owner = self._owner
        if identity is None or owner is None:
            raise QuackStateServerNotRunningError(
                "stop requires a live owner identity and lease"
            )
        selected_fence = owner.fence_token if fence_token is None else fence_token
        selected_server = identity.server_id
        control = _read_stop_control(self.stop_control_path())
        if control is not None:
            control_fence = control["fence_token"]
            if fence_token is not None and fence_token != control_fence:
                raise QuackStateServerControlError(
                    "explicit stop fence does not match stop control"
                )
            selected_fence = control_fence
            selected_server = control["server_id"]
        self._assert_live_stop_authority(
            identity=identity,
            owner=owner,
            server_id=selected_server,
            fence_token=selected_fence,
        )
        self._assert_database_namespace()
        return identity, owner, selected_fence

    def stop(self, *, fence_token: str | None = None) -> dict[str, Any]:
        """Stop through the fenced control path and release exclusive ownership."""

        # Authority admission is the first READY-owner action.  In particular,
        # a stale or malformed control cannot stop the outbox, change lifecycle,
        # close a broker/gateway/listener/database/vault, or shed the owner lock.
        with self._lock:
            if self._lifecycle is ServerLifecycle.STOPPED:
                return {"stopped": True, "already": True}
            if self._lifecycle is ServerLifecycle.CREATED:
                if self._event_wait is not None:
                    self._event_wait.shutdown()
                self._lifecycle = ServerLifecycle.STOPPED
                return {"stopped": True, "already": True}
            admitted_identity, admitted_owner, expected_fence = (
                self._admit_stop_before_effects(fence_token=fence_token)
            )

        # Stop the pump before taking the lifecycle lock.  It may be finishing
        # a typed transaction whose post-commit observer briefly needs that
        # lock; joining while holding it would deadlock shutdown.
        self._outbox_stop.set()
        if self._outbox_wake is not None:
            self._outbox_wake.shutdown()
        if self._outbox_thread is not None:
            self._outbox_thread.join(timeout=5.0)
            if self._outbox_thread.is_alive():
                raise QuackStateServerControlError(
                    "outbox worker did not stop within its bounded deadline"
                )
        with self._lock:
            if self._lifecycle is ServerLifecycle.STOPPED:
                return {"stopped": True, "already": True}

            # Re-observe the exact admitted objects after the outbox join and
            # before the first lifecycle/resource mutation.  The first check
            # above makes rejection side-effect free; this check prevents a
            # concurrent lifecycle replacement from consuming that admission.
            self._assert_live_stop_authority(
                identity=admitted_identity,
                owner=admitted_owner,
                server_id=admitted_identity.server_id,
                fence_token=expected_fence,
            )
            self._assert_database_namespace()

            self._lifecycle = ServerLifecycle.STOPPING
            if self._event_wait is not None:
                self._event_wait.shutdown()
            identity = self._identity
            owner = self._owner

            try:
                if self._grant_broker is not None:
                    self._grant_broker.stop()
                    self._grant_broker = None
                if self._grant_broker_secret_fd >= 0:
                    os.close(self._grant_broker_secret_fd)
                    self._grant_broker_secret_fd = -1
                if self._command_gateway is not None:
                    self._command_gateway.stop()
                    self._command_gateway = None
                try:
                    self.typed_command_token_path().unlink()
                except FileNotFoundError:
                    pass
            except Exception as exc:
                self._log(f"typed command gateway stop warning: {type(exc).__name__}")

            endpoint_closed = False
            try:
                self._stop_transport_connection(observe_closed=True)
                endpoint_closed = True
            except Exception as exc:
                self._log(
                    "transport closure was not observed: "
                    + type(exc).__name__
                )

            try:
                if self._connection is not None and identity is not None:
                    self._assert_database_namespace()
                    self._mark_server_stopped(self._connection, identity)
                    self._assert_database_namespace()
                    self._connection.execute("CHECKPOINT")
                    self._assert_database_namespace()
            except QuackStateServerOwnershipError:
                raise
            except Exception as exc:
                self._log(f"stop bookkeeping warning: {type(exc).__name__}")

            database_closed = self._connection is None
            try:
                self._close_authoritative_connection_observed()
                database_closed = True
            except Exception as exc:
                self._log(
                    "database closure was not observed: "
                    + type(exc).__name__
                )

            if not endpoint_closed:
                try:
                    self._wait_for_transport_endpoint_closed()
                    endpoint_closed = True
                except Exception:
                    pass

            if self._vault is not None:
                self._vault.destroy()

            if not endpoint_closed or not database_closed:
                # Never exchange an uncertain external shutdown for a second
                # owner.  Keeping both the inode lock and abstract path fence
                # forces a retry (or process exit) after the resources are inert.
                raise QuackStateServerControlError(
                    "state-owner shutdown closure was not positively observed"
                )

            if owner is not None:
                try:
                    owner.release(fence_token=expected_fence)
                except QuackStateServerOwnershipError:
                    # Namespace drift is never permission to shed the retained
                    # kernel/file capabilities, even after resources closed.
                    raise
                except QuackStateServerControlError:
                    # A stale or incorrect fence is never permission to shed
                    # the ownership capability, even though shutdown began.
                    raise
                except BaseException:
                    # Listener and database handles were closed above.  Release
                    # only the now-inert local authority so a path/marker error
                    # cannot wedge a later safe restart in this same process.
                    owner.emergency_close_after_owner_shutdown()
                    self._owner = None
                    raise
            self._owner = None
            self._close_database_namespace_anchor()

            if identity is not None:
                self._identity = identity.with_status("stopped")
            self._lifecycle = ServerLifecycle.STOPPED
            self._write_status()
            try:
                self.stop_control_path().unlink()
            except FileNotFoundError:
                pass
            receipt = {
                "stopped": True,
                "server_id": identity.server_id if identity else "",
                "at": _utc_iso(),
            }
            return sanitize_for_export(receipt)

    def request_stop(self, *, fence_token: str | None = None) -> dict[str, Any]:
        """Write a fenced stop request for the control path (does not stop inline)."""

        with self._lock:
            if self._identity is None or self._owner is None:
                raise QuackStateServerNotRunningError(
                    "cannot request stop without a live owner"
                )
            token = fence_token or self._owner.fence_token
            self._assert_live_stop_authority(
                identity=self._identity,
                owner=self._owner,
                server_id=self._identity.server_id,
                fence_token=token,
            )
            payload = {
                "schema": CONTROL_STOP_REQUEST_SCHEMA,
                "server_id": self._identity.server_id,
                "fence_token": token,
                "requested_at": _utc_iso(),
            }
            # fence_token is an ownership fence, not the Quack auth token.
            _atomic_write_json(self.stop_control_path(), payload, mode=0o600)
            return {
                "requested": True,
                "server_id": self._identity.server_id,
                "control_path": str(self.stop_control_path()),
            }

    def status(self) -> dict[str, Any]:
        """Public status projection — never includes raw auth token material."""

        with self._lock:
            identity = self._identity
            storage_schema_fingerprint = ""
            if identity is not None and self._connection is not None:
                try:
                    storage_schema_fingerprint = str(
                        self._mutation_binding().get("schema_fingerprint") or ""
                    )
                except QuackStateServerMutationError:
                    storage_schema_fingerprint = ""
            payload: dict[str, Any] = {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "lifecycle": self._lifecycle.value,
                "database_path": str(self.config.database_path),
                "state_dir": str(self.config.state_dir),
                "host": self.config.host,
                "port": int(self._bound_port or self.config.port),
                "container_bind_host": self.config.container_bind_host,
                "container_port": int(
                    self.config.container_port or self._bound_port
                ),
                "store_id": self.config.store_id,
                "secret_handle": identity.secret_handle if identity else self.secret_handle,
                "identity": identity.to_dict() if identity else None,
                "capability_status": (
                    self._capability.status.value if self._capability else None
                ),
                "extension_fingerprint": (
                    self._capability.extension_fingerprint if self._capability else ""
                ),
                "storage_schema_fingerprint": storage_schema_fingerprint,
                "read_replica": dict(self._read_replica_observation),
                "owner_marker_path": str(self.owner_marker_path()),
                "status_path": str(self.status_path()),
                "event_wait": self.event_wait_capability(),
                "outbox_worker": dict(self.outbox_worker_capability()),
                "typed_command_gateway": (
                    self._command_gateway.capability()
                    if self._command_gateway is not None
                    else {
                        "interface": "TypedStateOwnerCommandGateway@1",
                        "available": False,
                        "server_owned": True,
                        "raw_sql_permitted": False,
                    }
                ),
                "configured_supervisor_credential_broker": {
                    "available": bool(
                        self._grant_broker is not None
                        and self._grant_broker.alive()
                    ),
                    "server_owned": True,
                    "socket_path": (
                        str(self._grant_broker.socket_path)
                        if self._grant_broker is not None
                        else ""
                    ),
                    "credential_published": False,
                    "task_mutation_path": (
                        "typed_state_owner_database_task_command"
                    ),
                    "last_error_type": (
                        str(
                            self._grant_broker.capability().get(
                                "last_error_type"
                            )
                            or ""
                        )
                        if self._grant_broker is not None
                        else ""
                    ),
                },
            }
            token = None
            if self._vault is not None:
                token = getattr(self._vault, "_token", None)
            sanitized = sanitize_for_export(payload, token=token)
            if self._vault is not None:
                self._vault.assert_absent_from(sanitized, surface_name="status")
            return sanitized

    def export_identity(self) -> dict[str, Any]:
        """Export identity receipt suitable for clients (handle only)."""

        with self._lock:
            if self._identity is None:
                raise QuackStateServerNotRunningError("no identity to export")
            payload = {
                "export": True,
                "authority_class": StateAuthorityClass.EXPORT.value,
                "identity": self._identity.to_dict(),
                "store_identity": self._identity.store_identity().to_dict(),
                "store_generation": self._identity.store_generation().to_dict(),
            }
            token = None if self._vault is None else getattr(self._vault, "_token", None)
            return sanitize_for_export(payload, token=token)

    def provider_environment(
        self,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Environment projection for implementation providers (no credentials)."""

        # Explicitly do not pass secret handles or tokens to providers.
        return provider_safe_environment(base)

    def _write_status(self) -> None:
        try:
            _atomic_write_text(
                self.status_path(),
                canonical_json_bytes(self.status()).decode("utf-8") + "\n",
                mode=0o600,
            )
        except Exception as exc:
            self._log(f"status write warning: {type(exc).__name__}")

    def argv_safe_launch_spec(self) -> list[str]:
        """Return an argv vector that never embeds the auth token."""

        identity = self._identity
        argv = [
            "quack_state_server",
            "start",
            "--database",
            str(self.config.database_path),
            "--state-dir",
            str(self.config.state_dir),
            "--host",
            self.config.host,
            "--container-bind-host",
            self.config.container_bind_host,
            "--store-id",
            self.config.store_id,
        ]
        if self.config.repository_root is not None:
            argv.extend(
                ["--repository-root", str(self.config.repository_root)]
            )
        if self._bound_port or self.config.port:
            argv.extend(["--port", str(int(self._bound_port or self.config.port))])
            argv.extend(
                [
                    "--container-port",
                    str(int(self.config.container_port or self._bound_port)),
                ]
            )
        if identity is not None:
            argv.extend(["--secret-handle", identity.secret_handle])
        elif self.config.secret_handle:
            argv.extend(["--secret-handle", self.config.secret_handle])
        token = None if self._vault is None else getattr(self._vault, "_token", None)
        if _contains_token_material(argv, token):
            raise QuackStateServerTokenError("argv would contain auth token")
        return argv


def reclaim_stale_owner_marker(
    *,
    marker_path: Path,
    lock_path: Path,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
) -> dict[str, Any]:
    """Remove a stale owner marker when process birth is proved dead.

    Does not start a server. Fails closed if the owner is live or unknown.
    """

    probe = liveness or (lambda birth: owner_liveness(birth))
    marker_path = Path(marker_path)
    lock = Path(lock_path)
    handle: Any | None = None
    try:
        try:
            handle = acquire_exclusive_owner_lock(lock)
        except BlockingIOError:
            return {"reclaimed": False, "reason": "lock_held"}

        # Never inspect a marker before acquiring the canonical lock.  In
        # particular, an unlocked path read could follow a symlink or admit a
        # name swap while merely trying to describe the presumed holder.
        try:
            marker = read_locked_owner_marker(handle, marker_path)
        except QuackStateServerOwnershipError:
            return {"reclaimed": False, "reason": "marker_invalid"}
        if marker is None:
            return {"reclaimed": False, "reason": "no_marker"}

        state = probe(marker.process_birth)
        if state is OwnerLiveness.ALIVE:
            return {
                "reclaimed": False,
                "reason": "owner_alive",
                "server_id": marker.server_id,
            }
        if state is OwnerLiveness.UNKNOWN:
            return {
                "reclaimed": False,
                "reason": "owner_liveness_unknown",
                "server_id": marker.server_id,
            }

        # Revalidate both the canonical parent and the exact marker content
        # after the liveness observation.  A changed marker requires a fresh
        # liveness decision and therefore cannot be reclaimed by this call.
        handle.assert_canonical_parent()
        try:
            current = read_locked_owner_marker(handle, marker_path)
        except QuackStateServerOwnershipError:
            return {"reclaimed": False, "reason": "marker_changed"}
        if current is None:
            return {"reclaimed": False, "reason": "no_marker"}
        if current != marker:
            return {
                "reclaimed": False,
                "reason": "marker_changed",
                "server_id": current.server_id,
            }
        try:
            os.unlink(marker_path.name, dir_fd=handle.directory_fileno())
            os.fsync(handle.directory_fileno())
        except FileNotFoundError:
            return {"reclaimed": False, "reason": "no_marker"}
        handle.assert_canonical_parent()
        return {
            "reclaimed": True,
            "reason": "stale_owner_dead",
            "server_id": marker.server_id,
        }
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def build_server(
    *,
    database_path: Path | str,
    state_dir: Path | str,
    repository_root: Path | str | None = None,
    host: str = DEFAULT_LOOPBACK_HOST,
    port: int = 0,
    container_bind_host: str = "",
    container_port: int = 0,
    repository_id: str = "",
    store_id: str = DEFAULT_STORE_ID,
    allow_experimental: bool = False,
    remote_bind_policy: RemoteBindPolicy | None = None,
    secret_handle: str = "",
    isolation_receipt_path: Path | str | None = None,
    transport: QuackTransport | None = None,
    capability_probe: Callable[..., QuackCapabilityReport] | None = None,
    migrate: Callable[..., MigrationRunReport] | None = None,
    connection_factory: Callable[[Path], Any] | None = None,
    process_birth_factory: Callable[[], ProcessBirthIdentity] | None = None,
    owner_liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    isolation_observer: Callable[
        [QuackStateServerConfig, Mapping[str, Any]], Mapping[str, Any]
    ] | None = None,
    event_source: EventSource | None = None,
    typed_command_socket_path: Path | str | None = None,
) -> QuackStateServer:
    """Construct a configured :class:`QuackStateServer`."""

    config = QuackStateServerConfig(
        database_path=Path(database_path),
        state_dir=Path(state_dir),
        repository_root=(
            Path(repository_root) if repository_root is not None else None
        ),
        host=host,
        port=port,
        container_bind_host=container_bind_host,
        container_port=container_port,
        repository_id=repository_id,
        store_id=store_id,
        allow_experimental=allow_experimental,
        remote_bind_policy=remote_bind_policy,
        secret_handle=secret_handle,
        isolation_receipt_path=(
            None if isolation_receipt_path is None else Path(isolation_receipt_path)
        ),
        typed_command_socket_path_override=(
            None
            if typed_command_socket_path is None
            else Path(typed_command_socket_path)
        ),
    )
    server = QuackStateServer(
        config=config,
        transport=transport,
        capability_probe=capability_probe,
        migrate=migrate,
        connection_factory=connection_factory,
        process_birth_factory=process_birth_factory,
        owner_liveness_probe=owner_liveness_probe,
        isolation_observer=isolation_observer,
    )
    if event_source is not None:
        server.bind_event_source(event_source)
    return server


__all__ = (
    "DEFAULT_LOOPBACK_HOST",
    "DEFAULT_STORE_ID",
    "ExclusiveOwnerLease",
    "FakeQuackTransport",
    "InProcessQuackTransport",
    "OwnerMarker",
    "QUACK_STATE_SERVER_INTERFACE",
    "QuackStateServer",
    "QuackStateServerBindError",
    "QuackStateServerCapabilityError",
    "QuackStateServerConfig",
    "QuackStateServerControlError",
    "QuackStateServerError",
    "QuackStateServerMigrationError",
    "QuackStateServerNotRunningError",
    "QuackStateServerOwnershipError",
    "QuackStateServerReadyError",
    "QuackStateServerTokenError",
    "RemoteBindPolicy",
    "STATE_SERVER_IDENTITY_INTERFACE",
    "ServerLifecycle",
    "StateServerIdentity",
    "TokenVault",
    "acquire_exclusive_owner_lock",
    "assert_bind_admitted",
    "build_server",
    "listen_uri",
    "provider_safe_environment",
    "read_locked_owner_marker",
    "reclaim_stale_owner_marker",
    "sanitize_for_export",
)
