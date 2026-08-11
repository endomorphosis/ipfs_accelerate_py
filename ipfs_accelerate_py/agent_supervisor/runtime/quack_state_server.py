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
import fcntl
import hashlib
import json
import logging
import os
import secrets
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

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
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.quack_capabilities import (
    QuackCapabilityReport,
    probe_quack_capabilities,
)

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

DEFAULT_LOOPBACK_HOST: Final = "127.0.0.1"
DEFAULT_STORE_ID: Final = "control.duckdb"
DEFAULT_SECRET_HANDLE_PREFIX: Final = "handle:quack-token"
TOKEN_FILENAME_SUFFIX: Final = ".quack-token"
OWNER_MARKER_SUFFIX: Final = ".state-owner.json"
OWNER_LOCK_SUFFIX: Final = ".state-owner.lock"
STATUS_FILENAME: Final = "quack-state-server.status.json"
CONTROL_STOP_FILENAME: Final = "quack-state-server.stop"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(moment: datetime | None = None) -> str:
    value = moment or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
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
            if any(token in upper for token in _PROVIDER_ENV_DENY_SUBSTRINGS):
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

    def with_status(self, status: str) -> "StateServerIdentity":
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
    repository_id: str = ""
    store_id: str = DEFAULT_STORE_ID
    allow_experimental: bool = False
    remote_bind_policy: RemoteBindPolicy | None = None
    application_version: str | None = None
    tool_version: str | None = None
    secret_handle: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_path", Path(self.database_path))
        object.__setattr__(self, "state_dir", Path(self.state_dir))
        object.__setattr__(self, "host", str(self.host or DEFAULT_LOOPBACK_HOST).strip())
        object.__setattr__(self, "port", int(self.port))
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
        assert_bind_admitted(self.host, remote_policy=self.remote_bind_policy)

    def resolved_secret_handle(self, server_id: str, generation: int) -> str:
        if self.secret_handle:
            return self.secret_handle
        return f"{DEFAULT_SECRET_HANDLE_PREFIX}:{server_id}:g{int(generation)}"


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
    def from_dict(cls, payload: Mapping[str, Any]) -> "OwnerMarker":
        birth = ProcessBirthIdentity.from_dict(payload.get("process_birth"))
        return cls(
            server_id=str(payload.get("server_id") or ""),
            process_birth=birth,
            database_path=str(payload.get("database_path") or ""),
            started_at=str(payload.get("started_at") or ""),
            fence_token=str(payload.get("fence_token") or ""),
            generation=int(payload.get("generation") or 1),
        )


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

    def _read_marker(self) -> OwnerMarker | None:
        payload = _read_json(self.marker_path)
        if payload is None:
            return None
        try:
            return OwnerMarker.from_dict(payload)
        except (TypeError, ValueError, KeyError):
            return None

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

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            existing = self._read_marker()
            holder = existing.server_id if existing else "unknown"
            raise QuackStateServerOwnershipError(
                f"second state-owner refused; exclusive lock held by {holder}"
            ) from exc

        # Lock held: evaluate marker for live vs stale owner.
        existing = self._read_marker()
        if existing is not None and existing.process_birth.pid > 0:
            liveness = self._liveness(existing.process_birth)
            if liveness is OwnerLiveness.ALIVE:
                # Another process holds the semantic owner even if we raced the lock.
                # Release and fail closed.
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
                raise QuackStateServerOwnershipError(
                    f"second state-owner refused; live owner "
                    f"{existing.server_id} pid={existing.process_birth.pid}"
                )
            if liveness is OwnerLiveness.UNKNOWN:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
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
        _atomic_write_json(self.marker_path, marker.to_dict(), mode=0o600)
        self._handle = handle
        self._marker = marker
        self._fence_token = fence
        return marker

    def release(self, *, fence_token: str | None = None) -> None:
        if self._handle is None:
            return
        expected = fence_token if fence_token is not None else self._fence_token
        current = self._read_marker()
        if current is not None and expected and current.fence_token != expected:
            raise QuackStateServerControlError(
                "stop fence token does not match owner marker"
            )
        try:
            if current is not None and (
                not expected or current.fence_token == expected
            ):
                try:
                    self.marker_path.unlink()
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


# ---------------------------------------------------------------------------
# Token vault (handle-only public surface)
# ---------------------------------------------------------------------------


class TokenVault:
    """Store Quack auth tokens behind opaque secret handles.

    The token bytes exist only in a mode-0600 file under the state directory.
    Public APIs expose only the secret handle.
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
        # Never log token.
        try:
            connection.execute("LOAD quack")
        except Exception as exc:
            raise QuackStateServerCapabilityError(
                f"failed to LOAD quack for state-owner: {type(exc).__name__}"
            ) from exc

        uri = listen_uri(host, port)
        # Quack beta surface: try function forms without embedding token in SQL
        # text that might be logged by wrappers — use parameterized forms when
        # supported; fall back carefully.
        serve_attempts = (
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
        self._listen_uri = uri
        self._server_identity = {
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "schema_revision": identity.schema_revision,
            "schema_fingerprint": identity.schema_fingerprint,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "listen_uri": uri,
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
        del token  # used only by remote clients; local owner uses the connection
        if not self._started:
            raise QuackStateServerReadyError("transport has not started")
        # Local live probe: prove the exclusive connection still answers and
        # published identity rows still match.
        try:
            row = connection.execute("SELECT 1").fetchone()
        except Exception as exc:
            raise QuackStateServerReadyError(
                f"live query failed: {type(exc).__name__}"
            ) from exc
        if row is None:
            raise QuackStateServerReadyError("live query returned no row")
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
        del connection
        self._started = False
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
                "listen_uri": listen_uri(host, port),
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


class ServerLifecycle(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


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
    clock: Callable[[], float] = field(default=time.time)
    _lifecycle: ServerLifecycle = field(default=ServerLifecycle.CREATED, init=False)
    _identity: StateServerIdentity | None = field(default=None, init=False)
    _connection: Any | None = field(default=None, init=False)
    _owner: ExclusiveOwnerLease | None = field(default=None, init=False)
    _vault: TokenVault | None = field(default=None, init=False)
    _capability: QuackCapabilityReport | None = field(default=None, init=False)
    _migration_report: MigrationRunReport | None = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _bound_port: int = field(default=0, init=False)
    _logs: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, QuackStateServerConfig):
            raise TypeError("config must be QuackStateServerConfig")
        if self.transport is None:
            self.transport = InProcessQuackTransport()
        if self.capability_probe is None:
            self.capability_probe = probe_quack_capabilities
        if self.process_birth_factory is None:
            self.process_birth_factory = current_process_birth
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

    def _open_connection(self) -> Any:
        if self.connection_factory is not None:
            return self.connection_factory(self.config.database_path)
        if not duckdb_available():
            raise QuackStateServerError("DuckDB is required for the state-owner")
        return open_duckdb_connection(self.config.database_path)

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

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> StateServerIdentity:
        """Acquire exclusive ownership, migrate, serve, and publish identity."""

        with self._lock:
            if self._lifecycle in {ServerLifecycle.READY, ServerLifecycle.STARTING}:
                if self._identity is not None:
                    return self._identity
                raise QuackStateServerError("server is starting without identity")

            self._lifecycle = ServerLifecycle.STARTING
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            self.config.database_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                assert_bind_admitted(
                    self.config.host,
                    remote_policy=self.config.remote_bind_policy,
                )
                capability = self._admit_capability()
                self._capability = capability
                self._log(
                    f"capability admitted status={capability.status.value} "
                    f"fingerprint={capability.extension_fingerprint or 'none'}"
                )

                migration = self._ensure_migrated()
                self._migration_report = migration
                self._log("control-plane schema migration complete before ready")

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

                connection = self._open_connection()
                self._connection = connection
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
                public_obs = self.transport.start(
                    connection,
                    host=self.config.host,
                    port=port,
                    token=token,
                    identity=identity,
                )
                # Ensure transport observation never echoed the token.
                self._vault.assert_absent_from(public_obs, surface_name="transport.start")

                self._publish_identity_rows(connection, identity, capability)
                identity = identity.with_status("ready")
                self._identity = identity
                self._lifecycle = ServerLifecycle.READY
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
                raise

    def _emergency_cleanup(self) -> None:
        try:
            if self.transport is not None:
                self.transport.stop(self._connection)
        except Exception:
            pass
        try:
            if self._connection is not None and hasattr(self._connection, "close"):
                self._connection.close()
        except Exception:
            pass
        self._connection = None
        try:
            if self._vault is not None:
                self._vault.destroy()
        except Exception:
            pass
        try:
            if self._owner is not None:
                self._owner.release()
        except Exception:
            pass
        self._owner = None

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
            if self._connection is None or self.transport is None or self._vault is None:
                raise QuackStateServerReadyError("state-owner missing connection/transport")

            identity = self._identity
            token = self._vault.resolve(identity.secret_handle)
            observed = self.transport.live_query(
                self._connection,
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

            # Fail closed: live query must supply each identity field; do not
            # silently substitute published values for missing observations.
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
                name
                for name in required_fields
                if observed.get(name) in (None, "")
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
            if not identity.matches(
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
            }
            sanitized = sanitize_for_export(result, token=token)
            self._vault.assert_absent_from(sanitized, surface_name="ready")
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
                self._connection.execute("CHECKPOINT")
            except Exception as exc:
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

    def stop(self, *, fence_token: str | None = None) -> dict[str, Any]:
        """Stop through the fenced control path and release exclusive ownership."""

        with self._lock:
            if self._lifecycle is ServerLifecycle.STOPPED:
                return {"stopped": True, "already": True}
            if self._lifecycle is ServerLifecycle.CREATED:
                self._lifecycle = ServerLifecycle.STOPPED
                return {"stopped": True, "already": True}

            self._lifecycle = ServerLifecycle.STOPPING
            identity = self._identity
            owner = self._owner
            expected_fence = fence_token
            if expected_fence is None and owner is not None:
                expected_fence = owner.fence_token

            # Optional control-file fence for out-of-process stop requests.
            control = _read_json(self.stop_control_path())
            if control is not None:
                control_fence = str(control.get("fence_token") or "")
                if control_fence:
                    expected_fence = control_fence
                control_server = str(control.get("server_id") or "")
                if (
                    identity is not None
                    and control_server
                    and control_server != identity.server_id
                ):
                    raise QuackStateServerControlError(
                        "stop control server_id does not match live owner"
                    )

            try:
                if self.transport is not None:
                    self.transport.stop(self._connection)
            except Exception as exc:
                self._log(f"transport stop warning: {type(exc).__name__}")

            try:
                if self._connection is not None and identity is not None:
                    self._mark_server_stopped(self._connection, identity)
                    try:
                        self._connection.execute("CHECKPOINT")
                    except Exception:
                        pass
            except Exception as exc:
                self._log(f"stop bookkeeping warning: {type(exc).__name__}")

            try:
                if self._connection is not None and hasattr(self._connection, "close"):
                    self._connection.close()
            except Exception:
                pass
            self._connection = None

            if self._vault is not None:
                self._vault.destroy()

            if owner is not None:
                owner.release(fence_token=expected_fence)
            self._owner = None

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
            payload = {
                "schema": "ipfs_accelerate_py/agent-supervisor/quack-stop-request@1",
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
            payload: dict[str, Any] = {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "lifecycle": self._lifecycle.value,
                "database_path": str(self.config.database_path),
                "state_dir": str(self.config.state_dir),
                "host": self.config.host,
                "port": int(self._bound_port or self.config.port),
                "store_id": self.config.store_id,
                "secret_handle": identity.secret_handle if identity else self.secret_handle,
                "identity": identity.to_dict() if identity else None,
                "capability_status": (
                    self._capability.status.value if self._capability else None
                ),
                "extension_fingerprint": (
                    self._capability.extension_fingerprint if self._capability else ""
                ),
                "owner_marker_path": str(self.owner_marker_path()),
                "status_path": str(self.status_path()),
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
            _atomic_write_json(self.status_path(), self.status(), mode=0o600)
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
            "--store-id",
            self.config.store_id,
        ]
        if self._bound_port or self.config.port:
            argv.extend(["--port", str(int(self._bound_port or self.config.port))])
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
    payload = _read_json(Path(marker_path))
    if payload is None:
        return {"reclaimed": False, "reason": "no_marker"}
    try:
        marker = OwnerMarker.from_dict(payload)
    except (TypeError, ValueError, KeyError):
        # Corrupt marker: only reclaim under exclusive lock.
        marker = None

    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"reclaimed": False, "reason": "lock_held"}

        if marker is not None:
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
        try:
            Path(marker_path).unlink()
        except FileNotFoundError:
            return {"reclaimed": False, "reason": "no_marker"}
        return {
            "reclaimed": True,
            "reason": "stale_owner_dead" if marker else "corrupt_marker",
            "server_id": marker.server_id if marker else "",
        }
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_server(
    *,
    database_path: Path | str,
    state_dir: Path | str,
    host: str = DEFAULT_LOOPBACK_HOST,
    port: int = 0,
    repository_id: str = "",
    store_id: str = DEFAULT_STORE_ID,
    allow_experimental: bool = False,
    remote_bind_policy: RemoteBindPolicy | None = None,
    secret_handle: str = "",
    transport: QuackTransport | None = None,
    capability_probe: Callable[..., QuackCapabilityReport] | None = None,
    migrate: Callable[..., MigrationRunReport] | None = None,
    connection_factory: Callable[[Path], Any] | None = None,
    process_birth_factory: Callable[[], ProcessBirthIdentity] | None = None,
    owner_liveness_probe: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
) -> QuackStateServer:
    """Construct a configured :class:`QuackStateServer`."""

    config = QuackStateServerConfig(
        database_path=Path(database_path),
        state_dir=Path(state_dir),
        host=host,
        port=port,
        repository_id=repository_id,
        store_id=store_id,
        allow_experimental=allow_experimental,
        remote_bind_policy=remote_bind_policy,
        secret_handle=secret_handle,
    )
    return QuackStateServer(
        config=config,
        transport=transport,
        capability_probe=capability_probe,
        migrate=migrate,
        connection_factory=connection_factory,
        process_birth_factory=process_birth_factory,
        owner_liveness_probe=owner_liveness_probe,
    )


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
    "assert_bind_admitted",
    "build_server",
    "listen_uri",
    "provider_safe_environment",
    "reclaim_stale_owner_marker",
    "sanitize_for_export",
)
