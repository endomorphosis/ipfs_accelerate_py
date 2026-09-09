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
import stat as stat_module
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
STALE_OWNER_RECOVERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-stale-owner-recovery@1"
)
QUACK_STATE_SERVER_VERSION: Final[int] = 1

DEFAULT_LOOPBACK_HOST: Final = "127.0.0.1"
DEFAULT_STORE_ID: Final = "control.duckdb"
DEFAULT_SECRET_HANDLE_PREFIX: Final = "handle:quack-token"
TOKEN_FILENAME_SUFFIX: Final = ".quack-token"
TOKEN_RETIREMENT_LOCK_SUFFIX: Final = ".retirement.lock"
TOKEN_ROLLBACK_TEMP_PREFIX: Final = ".quack-token-rollback."
TOKEN_COMPROMISE_MARKER_SUFFIX: Final = ".compromised.json"
TOKEN_COMPROMISE_MARKER_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-compromise@1"
)
TOKEN_HANDOFF_RETIREMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-retirement@1"
)
TOKEN_HANDOFF_CLOSED_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-retirement-closed@1"
)
TOKEN_HANDOFF_AUTHORITY_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-authority-binding@1"
)
TOKEN_HANDOFF_REARM_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-rearm@1"
)
TOKEN_HANDOFF_REARM_PROBE_SCHEMA: Final = (
    "ipfs_accelerate_py/quack-token-handoff-rearm-probe@1"
)
TOKEN_HANDOFF_REARM_PROBE_REASONS: Final = frozenset(
    {
        "retirement_lock_held",
        "handoff_unsafe",
        "handoff_already_present",
        "coordinator_pid_absent",
        "coordinator_pid_empty",
        "coordinator_pid_dead",
        "coordinator_pid_alive",
        "coordinator_pid_unknown",
        "coordinator_pid_malformed",
        "coordinator_pid_unsafe",
        "coordinator_pid_changed",
    }
)
OWNER_MARKER_SUFFIX: Final = ".state-owner.json"
OWNER_LOCK_SUFFIX: Final = ".state-owner.lock"
STATUS_FILENAME: Final = "quack-state-server.status.json"
CONTROL_STOP_FILENAME: Final = "quack-state-server.stop"
STALE_OWNER_RECOVERY_RECEIPT_FILENAME: Final = (
    "quack-stale-owner-recovery-receipt.json"
)
PROVISIONAL_OWNER_MARKER_GENERATION: Final[int] = 1

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


class _TokenHandoffLockHeld(QuackStateServerTokenError):
    """The per-handoff lock is held by an active retirement or rearm."""


class QuackStateServerTokenCompromisedError(QuackStateServerTokenError):
    """A token inode survived retirement under an unexpected hardlink.

    ``receipt`` is deliberately secret-free.  The terminal transaction is
    attached so callers can inspect its closed/wiped state, but it cannot be
    committed or rolled back after compromise was detected.
    """

    def __init__(
        self,
        message: str,
        *,
        transaction: TokenHandoffRetirement | None,
        receipt: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.transaction = transaction
        self.receipt = MappingProxyType(dict(receipt))
        self.state = "compromised"


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


def _validate_recovery_receipt_filename(value: str) -> str:
    """Return one confined, portable recovery-receipt basename.

    Recovery settles canonical database rows before publishing its receipt, so
    a caller-controlled receipt path must be rejected before any recovery work
    begins.  In particular, accepting a platform-specific separator here could
    move the final publication outside ``state_dir`` on another host.
    """

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or value in {STATUS_FILENAME, CONTROL_STOP_FILENAME}
        or "/" in value
        or "\\" in value
        or any(not character.isprintable() for character in value)
    ):
        raise QuackStateServerControlError(
            "stale-owner recovery receipt filename is not a confined basename"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise QuackStateServerControlError(
            "stale-owner recovery receipt filename is not a confined basename"
        ) from exc
    # The local control profile bounds names to 255 encoded bytes.  This keeps
    # an otherwise valid-looking name from failing only after the canonical
    # stop transaction has committed.
    if len(encoded) > 255:
        raise QuackStateServerControlError(
            "stale-owner recovery receipt filename is not a confined basename"
        )
    return value


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


def _read_stable_regular_json(
    path: Path,
    *,
    noun: str,
    maximum_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Read one privileged control JSON file through a stable no-follow fd."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QuackStateServerControlError(
            f"{noun} is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise QuackStateServerControlError(
                f"{noun} is not a bounded regular file"
            )
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            block = os.read(
                descriptor,
                min(65_536, maximum_bytes + 1 - len(raw)),
            )
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QuackStateServerControlError(f"{noun} is unreadable") from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or len(raw) != before.st_size
        or len(raw) > maximum_bytes
    ):
        raise QuackStateServerControlError(f"{noun} changed while read")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            bytes(raw).decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise QuackStateServerControlError(f"{noun} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise QuackStateServerControlError(f"{noun} is not a JSON object")
    return payload


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
    expected_generation: int | None = None
    expected_database_uuid: str | None = None
    expected_store_id: str | None = None
    expected_listen_uri: str | None = None
    reuse_expected_generation: bool = False

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
        expected_generation = self.expected_generation
        if expected_generation is not None:
            if type(expected_generation) is not int or expected_generation < 1:
                raise ValueError("expected_generation must be a positive integer or None")
            object.__setattr__(self, "expected_generation", expected_generation)
        for field_name in (
            "expected_database_uuid",
            "expected_store_id",
            "expected_listen_uri",
        ):
            expected_value = getattr(self, field_name)
            if expected_value is None:
                continue
            if (
                not isinstance(expected_value, str)
                or not expected_value
                or expected_value.strip() != expected_value
            ):
                raise ValueError(f"{field_name} must be a non-empty canonical string or None")
            object.__setattr__(self, field_name, expected_value)
        object.__setattr__(
            self, "reuse_expected_generation", bool(self.reuse_expected_generation)
        )
        if self.reuse_expected_generation and self.expected_generation is None:
            raise ValueError(
                "reuse_expected_generation requires expected_generation"
            )
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

    A mode-0600 file provides a one-time trusted-coordinator handoff.  It must
    be retired before any untrusted provider child is launched.  Public APIs
    expose only the secret handle.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.state_dir.chmod(0o700)
        except OSError as exc:
            raise QuackStateServerTokenError(
                "token vault state directory could not be confined"
            ) from exc
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
        path = self.state_dir / _token_handoff_filename(secret_handle)
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


def _token_handoff_filename(secret_handle: str) -> str:
    """Return the confined handoff filename for one opaque handle."""

    handle = str(secret_handle or "").strip()
    if not is_secret_handle(handle) or any(
        ord(character) < 0x20 for character in handle
    ):
        raise QuackStateServerTokenError(
            "token handoff requires a valid opaque secret handle"
        )
    filename = (
        handle.replace(":", "_").replace("/", "_") + TOKEN_FILENAME_SUFFIX
    )
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise QuackStateServerTokenError("token handoff filename is not confined")
    return filename


def _wipe_token_bytes(value: bytearray) -> None:
    """Best-effort in-place wipe for the mutable credential copy we own."""

    for index in range(len(value)):
        value[index] = 0
    value.clear()


def _open_token_handoff_directory(
    state_dir: Path | str,
) -> tuple[Path, int, os.stat_result]:
    """Open every lexical directory component without following symlinks."""

    try:
        expanded = Path(state_dir).expanduser()
        directory = Path(os.path.abspath(os.fspath(expanded)))
    except (OSError, TypeError, ValueError) as exc:
        raise QuackStateServerTokenError(
            "token handoff state directory is unavailable"
        ) from exc
    if not directory.is_absolute() or directory.anchor != os.sep:
        raise QuackStateServerTokenError(
            "token handoff state directory is not an absolute POSIX path"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(os.sep, directory_flags)
        for component in directory.parts[1:]:
            if component in {"", ".", ".."}:
                raise QuackStateServerTokenError(
                    "token handoff state directory has an unsafe component"
                )
            before = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat_module.S_ISLNK(before.st_mode):
                raise QuackStateServerTokenError(
                    "token handoff state directory contains a symbolic link"
                )
            if not stat_module.S_ISDIR(before.st_mode):
                raise QuackStateServerTokenError(
                    "token handoff state directory component is not a directory"
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                current = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                expected_identity = (int(before.st_dev), int(before.st_ino))
                if (
                    (int(opened.st_dev), int(opened.st_ino))
                    != expected_identity
                    or (int(current.st_dev), int(current.st_ino))
                    != expected_identity
                    or not stat_module.S_ISDIR(opened.st_mode)
                    or not stat_module.S_ISDIR(current.st_mode)
                ):
                    raise QuackStateServerTokenError(
                        "token handoff state directory changed while opening"
                    )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        final_stat = os.fstat(descriptor)
        if (
            not stat_module.S_ISDIR(final_stat.st_mode)
            or final_stat.st_uid != os.geteuid()
            or stat_module.S_IMODE(final_stat.st_mode) & 0o022
        ):
            raise QuackStateServerTokenError(
                "token handoff state directory is not owner-confined"
            )
        return directory, descriptor, final_stat
    except QuackStateServerTokenError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise QuackStateServerTokenError(
            "token handoff state directory could not be opened safely"
        ) from exc


def _acquire_token_handoff_lock(
    *,
    directory_fd: int,
    filename: str,
) -> tuple[int, str, tuple[int, int, int, int, int]]:
    """Acquire one stable owner-only per-handoff lock without following links."""

    lock_filename = filename + TOKEN_RETIREMENT_LOCK_SUFFIX
    common_flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    created = False
    try:
        try:
            lock_fd = os.open(
                lock_filename,
                common_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            created = True
        except FileExistsError:
            lock_fd = os.open(
                lock_filename,
                common_flags,
                dir_fd=directory_fd,
            )
    except OSError as exc:
        raise QuackStateServerTokenError(
            "token handoff retirement lock could not be opened safely"
        ) from exc
    try:
        if created:
            os.fchmod(lock_fd, 0o600)
            os.fsync(lock_fd)
            os.fsync(directory_fd)
        opened = os.fstat(lock_fd)
        observed = os.stat(
            lock_filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or opened.st_uid != os.geteuid()
            or observed.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or observed.st_nlink != 1
            or stat_module.S_IMODE(opened.st_mode) != 0o600
            or stat_module.S_IMODE(observed.st_mode) != 0o600
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(observed.st_dev), int(observed.st_ino))
        ):
            raise QuackStateServerTokenError(
                "token handoff retirement lock identity or mode is invalid"
            )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise _TokenHandoffLockHeld(
                "token handoff retirement is already locked"
            ) from exc
        locked = os.fstat(lock_fd)
        current = os.stat(
            lock_filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        identity = (
            int(locked.st_dev),
            int(locked.st_ino),
            int(locked.st_uid),
            stat_module.S_IMODE(locked.st_mode),
            int(locked.st_ctime_ns),
        )
        if (
            (int(current.st_dev), int(current.st_ino)) != identity[:2]
            or current.st_uid != identity[2]
            or stat_module.S_IMODE(current.st_mode) != identity[3]
            or int(current.st_ctime_ns) != identity[4]
            or locked.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise QuackStateServerTokenError(
                "token handoff retirement lock changed during acquisition"
            )
        return lock_fd, lock_filename, identity
    except BaseException:
        os.close(lock_fd)
        raise


def _token_compromise_marker_filename(filename: str) -> str:
    return filename + TOKEN_COMPROMISE_MARKER_SUFFIX


def _token_compromise_receipt(
    *,
    secret_handle: str,
    credential_sha256: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": TOKEN_COMPROMISE_MARKER_SCHEMA,
        "compromised": True,
        "terminal": True,
        "reason": str(reason),
        "secret_handle": secret_handle,
        "credential_sha256": credential_sha256,
    }


def _persist_token_compromise_marker(
    *,
    directory_fd: int,
    filename: str,
    secret_handle: str,
    credential_sha256: str,
    reason: str,
) -> None:
    """Durably mark credential-link compromise before releasing its flock."""

    marker_name = _token_compromise_marker_filename(filename)
    payload = (
        json.dumps(
            _token_compromise_receipt(
                secret_handle=secret_handle,
                credential_sha256=credential_sha256,
                reason=reason,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                marker_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            # Any extant marker path remains a fail-closed barrier.  Never
            # replace it, even when malformed or attacker-created.
            os.fsync(directory_fd)
            return
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short token compromise marker write")
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        observed = os.stat(
            marker_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or opened.st_uid != os.geteuid()
            or observed.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or observed.st_nlink != 1
            or stat_module.S_IMODE(opened.st_mode) != 0o600
            or stat_module.S_IMODE(observed.st_mode) != 0o600
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(observed.st_dev), int(observed.st_ino))
            or opened.st_size != len(payload)
            or observed.st_size != len(payload)
        ):
            raise QuackStateServerTokenError(
                "token compromise marker could not be verified"
            )
        os.fsync(directory_fd)
    except OSError as exc:
        raise QuackStateServerTokenError(
            "token compromise marker could not be persisted"
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _require_no_token_compromise_marker(
    *,
    directory_fd: int,
    filename: str,
    secret_handle: str,
    credential_sha256: str,
) -> None:
    try:
        os.stat(
            _token_compromise_marker_filename(filename),
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise QuackStateServerTokenError(
            "token compromise marker cannot be inspected"
        ) from exc
    receipt = _token_compromise_receipt(
        secret_handle=secret_handle,
        credential_sha256=credential_sha256,
        reason="durable_compromise_marker_exists",
    )
    raise QuackStateServerTokenCompromisedError(
        "token handoff is blocked by a durable compromise marker",
        transaction=None,
        receipt=receipt,
    )


def _rollback_temp_name_is_exact(name: str) -> bool:
    suffix = name.removeprefix(TOKEN_ROLLBACK_TEMP_PREFIX)
    return (
        name.startswith(TOKEN_ROLLBACK_TEMP_PREFIX)
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _verify_exact_token_path(
    *,
    directory_fd: int,
    filename: str,
    expected_token: bytes | bytearray,
    expected_link_count: int,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    descriptor = os.open(
        filename,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_fd,
    )
    observed_bytes = bytearray()
    try:
        opened = os.fstat(descriptor)
        observed = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        fingerprint = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_uid),
            stat_module.S_IMODE(opened.st_mode),
            int(opened.st_nlink),
            int(opened.st_size),
            int(opened.st_ctime_ns),
            int(opened.st_mtime_ns),
        )
        observed_fingerprint = (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_uid),
            stat_module.S_IMODE(observed.st_mode),
            int(observed.st_nlink),
            int(observed.st_size),
            int(observed.st_ctime_ns),
            int(observed.st_mtime_ns),
        )
        identity = fingerprint[:2]
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or fingerprint != observed_fingerprint
            or fingerprint[2] != os.geteuid()
            or fingerprint[3] != 0o600
            or fingerprint[4] != expected_link_count
            or fingerprint[5] != len(expected_token)
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise QuackStateServerTokenError(
                "token handoff crash artifact identity is invalid"
            )
        while len(observed_bytes) <= len(expected_token):
            chunk = os.read(
                descriptor,
                min(len(expected_token) + 1 - len(observed_bytes), 256),
            )
            if not chunk:
                break
            observed_bytes.extend(chunk)
        post_opened = os.fstat(descriptor)
        post_observed = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        post_fingerprint = (
            int(post_opened.st_dev),
            int(post_opened.st_ino),
            int(post_opened.st_uid),
            stat_module.S_IMODE(post_opened.st_mode),
            int(post_opened.st_nlink),
            int(post_opened.st_size),
            int(post_opened.st_ctime_ns),
            int(post_opened.st_mtime_ns),
        )
        post_observed_fingerprint = (
            int(post_observed.st_dev),
            int(post_observed.st_ino),
            int(post_observed.st_uid),
            stat_module.S_IMODE(post_observed.st_mode),
            int(post_observed.st_nlink),
            int(post_observed.st_size),
            int(post_observed.st_ctime_ns),
            int(post_observed.st_mtime_ns),
        )
        if (
            post_fingerprint != fingerprint
            or post_observed_fingerprint != observed_fingerprint
            or not secrets.compare_digest(observed_bytes, expected_token)
        ):
            raise QuackStateServerTokenError(
                "token handoff crash artifact bytes are invalid"
            )
        return identity
    finally:
        _wipe_token_bytes(observed_bytes)
        os.close(descriptor)


def _recover_linked_rollback_temp(
    *,
    directory_fd: int,
    filename: str,
    secret_handle: str,
    credential_sha256: str,
    expected_token: bytes | bytearray,
) -> bool:
    """Heal only exact unique pre-link, linked, or disjoint temp artifacts."""

    rollback_names = [
        name
        for name in os.listdir(directory_fd)
        if name.startswith(TOKEN_ROLLBACK_TEMP_PREFIX)
    ]
    try:
        canonical = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        canonical = None
    if canonical is None and not rollback_names:
        return False
    if canonical is not None and not rollback_names and canonical.st_nlink == 1:
        return False
    compromise_reason = "unexpected_rollback_temp_artifact"
    try:
        if len(rollback_names) != 1 or not _rollback_temp_name_is_exact(
            rollback_names[0]
        ):
            raise QuackStateServerTokenError(
                "rollback temp artifact set is not unique and exact"
            )
        temporary_name = rollback_names[0]
        if canonical is None:
            # Crash before publication: promote the unique complete temp with
            # an atomic no-replace hardlink, then remove only that exact alias.
            identity = _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=temporary_name,
                expected_token=expected_token,
                expected_link_count=1,
            )
            os.link(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=2,
                expected_identity=identity,
            )
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=temporary_name,
                expected_token=expected_token,
                expected_link_count=2,
                expected_identity=identity,
            )
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=1,
                expected_identity=identity,
            )
            return True
        canonical_identity = (int(canonical.st_dev), int(canonical.st_ino))
        if canonical.st_nlink == 2:
            # Crash after publication: canonical and temp must be the only two
            # names of one exact complete inode.
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=2,
                expected_identity=canonical_identity,
            )
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=temporary_name,
                expected_token=expected_token,
                expected_link_count=2,
                expected_identity=canonical_identity,
            )
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=1,
                expected_identity=canonical_identity,
            )
            return True
        if canonical.st_nlink == 1:
            # Crash before publication left a disjoint exact temp while a
            # normal exact canonical handoff was independently restored.
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=1,
                expected_identity=canonical_identity,
            )
            temporary_identity = _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=temporary_name,
                expected_token=expected_token,
                expected_link_count=1,
            )
            if temporary_identity == canonical_identity:
                raise QuackStateServerTokenError(
                    "disjoint rollback temp unexpectedly aliases canonical"
                )
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            zeroes = bytearray(len(expected_token))
            try:
                opened = os.fstat(temporary_fd)
                if (
                    (int(opened.st_dev), int(opened.st_ino))
                    != temporary_identity
                    or opened.st_nlink != 1
                ):
                    raise QuackStateServerTokenError(
                        "rollback temp changed before wipe"
                    )
                written = 0
                while written < len(zeroes):
                    count = os.write(temporary_fd, zeroes[written:])
                    if count <= 0:
                        raise OSError("short rollback temp wipe")
                    written += count
                os.fsync(temporary_fd)
                current = os.stat(
                    temporary_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    (int(current.st_dev), int(current.st_ino))
                    != temporary_identity
                    or current.st_nlink != 1
                ):
                    raise QuackStateServerTokenError(
                        "rollback temp changed during wipe"
                    )
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            finally:
                _wipe_token_bytes(zeroes)
                os.close(temporary_fd)
            _verify_exact_token_path(
                directory_fd=directory_fd,
                filename=filename,
                expected_token=expected_token,
                expected_link_count=1,
                expected_identity=canonical_identity,
            )
            return True
        compromise_reason = "unexpected_canonical_credential_hardlink"
        raise QuackStateServerTokenError(
            "canonical credential has unexpected links"
        )
    except (OSError, QuackStateServerTokenError):
        if canonical is not None and canonical.st_nlink != 1:
            compromise_reason = "unexpected_canonical_credential_hardlink"
    _persist_token_compromise_marker(
        directory_fd=directory_fd,
        filename=filename,
        secret_handle=secret_handle,
        credential_sha256=credential_sha256,
        reason=compromise_reason,
    )
    raise QuackStateServerTokenCompromisedError(
        "token handoff has an unsafe credential hardlink artifact",
        transaction=None,
        receipt=_token_compromise_receipt(
            secret_handle=secret_handle,
            credential_sha256=credential_sha256,
            reason=compromise_reason,
        ),
    )


_TOKEN_HANDOFF_CONSTRUCTION_AUTHORITY: Final = object()


class TokenHandoffRetirement:
    """Rollback-capable retirement of one exact token handoff.

    Only :func:`begin_token_handoff_retirement` constructs this object.  It
    retains an authenticated directory descriptor and the removed bytes until
    commit or rollback.  Credential bytes are never exposed by ``repr`` or a
    receipt.

    This is deliberately a process-local transaction.  ``close()``, context
    management, and finalization make ordinary exception/abandonment paths
    rollback-safe, but an uncatchable process exit cannot run Python cleanup.
    """

    __slots__ = (
        "_already_absent",
        "_credential_sha256",
        "_directory",
        "_directory_fd",
        "_directory_identity",
        "_expected_commit_receipt",
        "_filename",
        "_lock_fd",
        "_lock_filename",
        "_lock_identity",
        "_owner_pid",
        "_receipt",
        "_secret_handle",
        "_state",
        "_token_bytes",
    )

    def __init__(
        self,
        *,
        _construction_authority: object,
        directory: Path,
        directory_fd: int,
        directory_identity: tuple[int, int, int, int],
        filename: str,
        lock_fd: int,
        lock_filename: str,
        lock_identity: tuple[int, int, int, int, int],
        secret_handle: str,
        credential_sha256: str,
        token_bytes: bytearray,
        already_absent: bool,
    ) -> None:
        if _construction_authority is not _TOKEN_HANDOFF_CONSTRUCTION_AUTHORITY:
            raise TypeError(
                "TokenHandoffRetirement must be created by "
                "begin_token_handoff_retirement"
            )
        self._directory = directory
        self._directory_fd = int(directory_fd)
        self._directory_identity = directory_identity
        self._filename = filename
        self._lock_fd = int(lock_fd)
        self._lock_filename = lock_filename
        self._lock_identity = lock_identity
        self._owner_pid = os.getpid()
        self._secret_handle = secret_handle
        self._credential_sha256 = credential_sha256
        self._token_bytes = token_bytes
        self._already_absent = bool(already_absent)
        self._expected_commit_receipt = MappingProxyType(
            {
                "schema": TOKEN_HANDOFF_RETIREMENT_SCHEMA,
                "retired": True,
                "already_absent": self._already_absent,
                "secret_handle": self._secret_handle,
            }
        )
        self._state = "begun"
        self._receipt: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(secret_handle={self._secret_handle!r}, "
            f"state={self._state!r}, already_absent={self._already_absent!r})"
        )

    @property
    def state(self) -> str:
        return self._state

    @property
    def secret_handle(self) -> str:
        return self._secret_handle

    @property
    def credential_sha256(self) -> str:
        """Return the non-secret digest binding the retired exact token."""

        return self._credential_sha256

    @property
    def expected_commit_receipt(self) -> dict[str, Any]:
        """Return a defensive copy of the exact secret-free commit receipt."""

        if self._state not in {"begun", "committed"}:
            raise QuackStateServerTokenError(
                "token handoff retirement can no longer commit"
            )
        return dict(self._expected_commit_receipt)

    def validate_active(
        self,
        *,
        state_dir: Path | str,
        secret_handle: str,
        credential_sha256: str,
    ) -> dict[str, Any]:
        """Bind this active transaction to exact caller-sealed authority."""

        self._require_owner_process()
        if self._state != "begun":
            raise QuackStateServerTokenError(
                "token handoff retirement is not active"
            )
        if (
            not isinstance(secret_handle, str)
            or not secrets.compare_digest(secret_handle, self._secret_handle)
            or not isinstance(credential_sha256, str)
            or not secrets.compare_digest(
                credential_sha256,
                self._credential_sha256,
            )
        ):
            raise QuackStateServerTokenError(
                "token handoff active authority binding differs"
            )
        requested_path = Path(state_dir)
        if (
            not requested_path.is_absolute()
            or Path(os.path.abspath(os.fspath(requested_path))) != requested_path
        ):
            raise QuackStateServerTokenError(
                "token handoff active state directory is not canonical absolute"
            )
        requested_directory, requested_fd, requested_stat = (
            _open_token_handoff_directory(state_dir)
        )
        try:
            requested_identity = (
                int(requested_stat.st_dev),
                int(requested_stat.st_ino),
                int(requested_stat.st_uid),
                stat_module.S_IMODE(requested_stat.st_mode),
            )
            if (
                requested_directory != self._directory
                or requested_identity != self._directory_identity
            ):
                raise QuackStateServerTokenError(
                    "token handoff active state directory differs"
                )
        finally:
            os.close(requested_fd)
        self._require_no_compromise_marker()
        self._verify_directory()
        self._verify_lock()
        self._require_path_absent()
        self._verify_lock()
        self._verify_directory()
        return {
            "schema": TOKEN_HANDOFF_AUTHORITY_BINDING_SCHEMA,
            "state_dir": str(self._directory),
            "secret_handle": self._secret_handle,
            "credential_sha256": self._credential_sha256,
        }

    def __enter__(self) -> TokenHandoffRetirement:
        self._require_owner_process()
        if self._state != "begun":
            raise QuackStateServerTokenError(
                "token handoff retirement context is no longer active"
            )
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        try:
            if self._state == "begun":
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
        except BaseException:
            # A scope exit is terminal from the caller's perspective.  If its
            # recovery attempt failed before reaching a typed terminal state,
            # release the retained lock/descriptors and wipe the token rather
            # than leaving cleanup dependent on a later garbage collection.
            if self._state == "begun":
                self._abandon()
            raise
        return False

    def __del__(self) -> None:
        """Best-effort rollback for a transaction abandoned by its caller."""

        try:
            if getattr(self, "_state", None) == "begun":
                if getattr(self, "_owner_pid", -1) == os.getpid():
                    try:
                        self.rollback()
                    except BaseException:
                        if self._state == "begun":
                            self._abandon()
                else:
                    # A fork child must not unlock or mutate its parent's
                    # transaction.  It only closes its inherited descriptors.
                    self._release_resources(unlock=False)
                    self._state = "fork_discarded"
            elif (
                getattr(self, "_lock_fd", -1) >= 0
                or getattr(self, "_directory_fd", -1) >= 0
            ):
                self._release_resources(
                    unlock=getattr(self, "_owner_pid", -1) == os.getpid()
                )
        except BaseException:
            # Destructors must never surface errors.  Descriptor fields are
            # detached before close attempts and token wiping is still tried.
            pass

    def _require_owner_process(self) -> None:
        if self._owner_pid != os.getpid():
            raise QuackStateServerTokenError(
                "token handoff retirement belongs to another process"
            )

    def _verify_directory(self) -> None:
        if self._directory_fd < 0:
            raise QuackStateServerTokenError(
                "token handoff retirement directory is closed"
            )
        verification_fd = -1
        try:
            opened = os.fstat(self._directory_fd)
            _path, verification_fd, observed = _open_token_handoff_directory(
                self._directory
            )
        except (OSError, QuackStateServerTokenError) as exc:
            raise QuackStateServerTokenError(
                "token handoff state directory changed during retirement"
            ) from exc
        finally:
            if verification_fd >= 0:
                os.close(verification_fd)
        opened_identity = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_uid),
            stat_module.S_IMODE(opened.st_mode),
        )
        if (
            opened_identity != self._directory_identity
            or (int(observed.st_dev), int(observed.st_ino))
            != self._directory_identity[:2]
            or not stat_module.S_ISDIR(opened.st_mode)
            or not stat_module.S_ISDIR(observed.st_mode)
            or int(observed.st_uid) != self._directory_identity[2]
            or stat_module.S_IMODE(observed.st_mode)
            != self._directory_identity[3]
            or self._directory_identity[2] != os.geteuid()
            or bool(self._directory_identity[3] & 0o022)
        ):
            raise QuackStateServerTokenError(
                "token handoff state directory identity or confinement changed"
            )

    def _verify_lock(self) -> None:
        if self._lock_fd < 0:
            raise QuackStateServerTokenError(
                "token handoff retirement lock is closed"
            )
        try:
            opened = os.fstat(self._lock_fd)
            observed = os.stat(
                self._lock_filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise QuackStateServerTokenError(
                "token handoff retirement lock changed"
            ) from exc
        identity = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_uid),
            stat_module.S_IMODE(opened.st_mode),
            int(opened.st_ctime_ns),
        )
        if (
            identity != self._lock_identity
            or (int(observed.st_dev), int(observed.st_ino))
            != self._lock_identity[:2]
            or not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or opened.st_nlink != 1
            or observed.st_nlink != 1
            or observed.st_uid != self._lock_identity[2]
            or stat_module.S_IMODE(observed.st_mode)
            != self._lock_identity[3]
            or int(observed.st_ctime_ns) != self._lock_identity[4]
        ):
            raise QuackStateServerTokenError(
                "token handoff retirement lock identity or mode changed"
            )

    def _require_path_absent(self) -> None:
        try:
            os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise QuackStateServerTokenError(
                "token handoff pathname cannot be verified"
            ) from exc
        raise QuackStateServerTokenError(
            "token handoff pathname was recreated during retirement"
        )

    def _require_no_compromise_marker(self) -> None:
        _require_no_token_compromise_marker(
            directory_fd=self._directory_fd,
            filename=self._filename,
            secret_handle=self._secret_handle,
            credential_sha256=self._credential_sha256,
        )

    def _persist_compromise_marker(self, *, reason: str) -> None:
        _persist_token_compromise_marker(
            directory_fd=self._directory_fd,
            filename=self._filename,
            secret_handle=self._secret_handle,
            credential_sha256=self._credential_sha256,
            reason=reason,
        )

    def _release_resources(self, *, unlock: bool = True) -> None:
        """Detach all owned resources, attempting every cleanup independently."""

        lock_fd = getattr(self, "_lock_fd", -1)
        directory_fd = getattr(self, "_directory_fd", -1)
        # Detach first: a close error must not leave a stale integer that could
        # later refer to an unrelated, reused descriptor.
        self._lock_fd = -1
        self._directory_fd = -1
        if lock_fd >= 0:
            if unlock:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    # Closing the descriptor is the final kernel-level release.
                    pass
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        _wipe_token_bytes(self._token_bytes)

    def _finish(self, *, state: str, receipt: dict[str, Any]) -> dict[str, Any]:
        # Freeze the exact result before exposing its terminal state.  Thus any
        # exception after ``state`` changes remains recoverable via an
        # idempotent operation/property rather than becoming ambiguous.
        frozen_receipt = MappingProxyType(dict(receipt))
        self._receipt = frozen_receipt
        self._state = state
        self._release_resources()
        return dict(frozen_receipt)

    def _abandon(self) -> None:
        """Close retained resources after a failure without claiming recovery."""

        self._state = "failed"
        self._receipt = None
        self._release_resources()

    def _finish_compromised(
        self,
        *,
        reason: str,
        observed_link_count: int,
    ) -> QuackStateServerTokenCompromisedError:
        marker_error: BaseException | None = None
        try:
            _persist_token_compromise_marker(
                directory_fd=self._directory_fd,
                filename=self._filename,
                secret_handle=self._secret_handle,
                credential_sha256=self._credential_sha256,
                reason=reason,
            )
        except BaseException as exc:
            marker_error = exc
        receipt = {
            "schema": (
                "ipfs_accelerate_py/"
                "quack-token-handoff-retirement-compromised@1"
            ),
            "compromised": True,
            "terminal": True,
            "reason": str(reason),
            "observed_link_count": int(observed_link_count),
            "marker_persisted": marker_error is None,
            "secret_handle": self._secret_handle,
            "credential_sha256": self._credential_sha256,
        }
        self._finish(state="compromised", receipt=receipt)
        error = QuackStateServerTokenCompromisedError(
            "token handoff retirement detected surviving credential links",
            transaction=self,
            receipt=receipt,
        )
        if marker_error is not None:
            error.__cause__ = marker_error
        return error

    def close(self) -> dict[str, Any] | None:
        """Rollback an active transaction and release all retained resources."""

        if self._state == "begun":
            try:
                return self.rollback()
            except BaseException:
                if self._state == "begun":
                    self._abandon()
                raise
        if self._receipt is None:
            return None
        return dict(self._receipt)

    def close_without_rollback(
        self,
        *,
        reason: str = "child_liveness_unproven",
    ) -> dict[str, Any]:
        """Terminally close and wipe without republishing credential bytes."""

        self._require_owner_process()
        if self._state == "closed":
            assert self._receipt is not None
            if reason != "child_liveness_unproven":
                raise QuackStateServerTokenError(
                    "token handoff close reason is not allowed"
                )
            return dict(self._receipt)
        if self._state != "begun":
            raise QuackStateServerTokenError(
                "token handoff retirement cannot close without rollback"
            )
        if reason != "child_liveness_unproven":
            raise QuackStateServerTokenError(
                "token handoff close reason is not allowed"
            )
        return self._finish(
            state="closed",
            receipt={
                "schema": TOKEN_HANDOFF_CLOSED_SCHEMA,
                "closed": True,
                "terminal": True,
                "reason": reason,
                "completion_authority": False,
                "task_authority": False,
                "secret_handle": self._secret_handle,
                "credential_sha256": self._credential_sha256,
            },
        )

    def commit(self) -> dict[str, Any]:
        """Make the retirement final while the exact pathname stays absent."""

        self._require_owner_process()
        if self._state == "committed":
            assert self._receipt is not None
            return dict(self._receipt)
        if self._state != "begun":
            raise QuackStateServerTokenError(
                "token handoff retirement cannot commit after rollback"
            )
        self._require_no_compromise_marker()
        self._verify_directory()
        self._verify_lock()
        self._require_path_absent()
        self._verify_lock()
        self._verify_directory()
        return self._finish(
            state="committed",
            receipt=self.expected_commit_receipt,
        )

    def _unlink_temporary_token(
        self,
        *,
        temporary_fd: int,
        temporary_name: str,
        temporary_identity: tuple[int, int],
    ) -> int:
        """Best-effort unlink of our exact temporary inode; return link count."""

        try:
            current = os.stat(
                temporary_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current = None
        except OSError:
            current = None
        if current is not None and (
            int(current.st_dev),
            int(current.st_ino),
        ) == temporary_identity:
            try:
                os.unlink(temporary_name, dir_fd=self._directory_fd)
            except OSError:
                pass
        try:
            os.fsync(self._directory_fd)
        except OSError:
            pass
        try:
            return int(os.fstat(temporary_fd).st_nlink)
        except OSError:
            return -1

    def _canonical_token_identity_is(
        self,
        identity: tuple[int, int],
    ) -> bool:
        try:
            observed = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except OSError:
            return False
        return (
            stat_module.S_ISREG(observed.st_mode)
            and (int(observed.st_dev), int(observed.st_ino)) == identity
        )

    def _filename_is_expected_token(self) -> bool:
        """Return whether the canonical path is our complete owner-only token."""

        descriptor = -1
        observed_bytes = bytearray()
        try:
            descriptor = os.open(
                self._filename,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._directory_fd,
            )
            opened = os.fstat(descriptor)
            observed = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat_module.S_ISREG(opened.st_mode)
                or not stat_module.S_ISREG(observed.st_mode)
                or opened.st_uid != os.geteuid()
                or observed.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or observed.st_nlink != 1
                or stat_module.S_IMODE(opened.st_mode) != 0o600
                or stat_module.S_IMODE(observed.st_mode) != 0o600
                or (int(opened.st_dev), int(opened.st_ino))
                != (int(observed.st_dev), int(observed.st_ino))
                or opened.st_size != len(self._token_bytes)
                or observed.st_size != len(self._token_bytes)
            ):
                return False
            while len(observed_bytes) <= len(self._token_bytes):
                chunk = os.read(
                    descriptor,
                    min(len(self._token_bytes) + 1 - len(observed_bytes), 256),
                )
                if not chunk:
                    break
                observed_bytes.extend(chunk)
            return secrets.compare_digest(observed_bytes, self._token_bytes)
        except OSError:
            return False
        finally:
            _wipe_token_bytes(observed_bytes)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _verify_restored_token(
        self,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        """Reopen and verify the atomically published canonical handoff."""

        verification_fd = os.open(
            self._filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=self._directory_fd,
        )
        verified_bytes = bytearray()
        try:
            opened = os.fstat(verification_fd)
            observed = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            initial_fingerprint = (
                int(opened.st_dev),
                int(opened.st_ino),
                int(opened.st_uid),
                stat_module.S_IMODE(opened.st_mode),
                int(opened.st_nlink),
                int(opened.st_size),
                int(opened.st_ctime_ns),
                int(opened.st_mtime_ns),
            )
            observed_fingerprint = (
                int(observed.st_dev),
                int(observed.st_ino),
                int(observed.st_uid),
                stat_module.S_IMODE(observed.st_mode),
                int(observed.st_nlink),
                int(observed.st_size),
                int(observed.st_ctime_ns),
                int(observed.st_mtime_ns),
            )
            if opened.st_nlink != 1 or observed.st_nlink != 1:
                self._persist_compromise_marker(
                    reason="restored_credential_has_unexpected_hardlink"
                )
            if (
                not stat_module.S_ISREG(opened.st_mode)
                or not stat_module.S_ISREG(observed.st_mode)
                or initial_fingerprint[:2] != expected_identity
                or observed_fingerprint[:2] != expected_identity
                or initial_fingerprint[2] != os.geteuid()
                or observed_fingerprint[2] != os.geteuid()
                or initial_fingerprint[3] != 0o600
                or observed_fingerprint[3] != 0o600
                or initial_fingerprint[4] != 1
                or observed_fingerprint[4] != 1
                or initial_fingerprint[5] != len(self._token_bytes)
                or observed_fingerprint[5] != len(self._token_bytes)
            ):
                raise QuackStateServerTokenError(
                    "restored token handoff final identity is invalid"
                )
            while len(verified_bytes) <= len(self._token_bytes):
                chunk = os.read(
                    verification_fd,
                    min(len(self._token_bytes) + 1 - len(verified_bytes), 256),
                )
                if not chunk:
                    break
                verified_bytes.extend(chunk)
            if not secrets.compare_digest(verified_bytes, self._token_bytes):
                raise QuackStateServerTokenError(
                    "restored token handoff bytes did not verify"
                )
            post_opened = os.fstat(verification_fd)
            post_observed = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            post_opened_fingerprint = (
                int(post_opened.st_dev),
                int(post_opened.st_ino),
                int(post_opened.st_uid),
                stat_module.S_IMODE(post_opened.st_mode),
                int(post_opened.st_nlink),
                int(post_opened.st_size),
                int(post_opened.st_ctime_ns),
                int(post_opened.st_mtime_ns),
            )
            post_observed_fingerprint = (
                int(post_observed.st_dev),
                int(post_observed.st_ino),
                int(post_observed.st_uid),
                stat_module.S_IMODE(post_observed.st_mode),
                int(post_observed.st_nlink),
                int(post_observed.st_size),
                int(post_observed.st_ctime_ns),
                int(post_observed.st_mtime_ns),
            )
            if post_opened.st_nlink != 1 or post_observed.st_nlink != 1:
                self._persist_compromise_marker(
                    reason="restored_credential_has_unexpected_hardlink"
                )
            if (
                post_opened_fingerprint != initial_fingerprint
                or post_observed_fingerprint != observed_fingerprint
            ):
                raise QuackStateServerTokenError(
                    "restored token handoff changed during final read"
                )
        finally:
            _wipe_token_bytes(verified_bytes)
            try:
                os.close(verification_fd)
            except OSError:
                pass

    def _restore_token_atomically(self) -> None:
        """Publish fully written rollback bytes without exposing a partial path."""

        temporary_name = TOKEN_ROLLBACK_TEMP_PREFIX + secrets.token_hex(16)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        temporary_fd = -1
        temporary_identity: tuple[int, int] | None = None
        published = False
        try:
            temporary_fd = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=self._directory_fd,
            )
            opened = os.fstat(temporary_fd)
            temporary_identity = (int(opened.st_dev), int(opened.st_ino))
            os.fchmod(temporary_fd, 0o600)
            written = 0
            view = memoryview(self._token_bytes)
            try:
                while written < len(view):
                    count = os.write(temporary_fd, view[written:])
                    if count <= 0:
                        raise OSError("short token handoff rollback write")
                    written += count
            finally:
                view.release()
            os.fsync(temporary_fd)
            opened = os.fstat(temporary_fd)
            observed = os.stat(
                temporary_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if opened.st_nlink != 1 or observed.st_nlink != 1:
                self._persist_compromise_marker(
                    reason="rollback_temporary_inode_has_surviving_hardlink"
                )
            if (
                not stat_module.S_ISREG(opened.st_mode)
                or not stat_module.S_ISREG(observed.st_mode)
                or opened.st_uid != os.geteuid()
                or observed.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or observed.st_nlink != 1
                or stat_module.S_IMODE(opened.st_mode) != 0o600
                or stat_module.S_IMODE(observed.st_mode) != 0o600
                or (int(opened.st_dev), int(opened.st_ino))
                != temporary_identity
                or (int(observed.st_dev), int(observed.st_ino))
                != temporary_identity
                or opened.st_size != len(self._token_bytes)
                or observed.st_size != len(self._token_bytes)
            ):
                raise QuackStateServerTokenError(
                    "rollback temporary token identity or mode is invalid"
                )
            os.lseek(temporary_fd, 0, os.SEEK_SET)
            verified = bytearray()
            try:
                while len(verified) <= len(self._token_bytes):
                    chunk = os.read(
                        temporary_fd,
                        min(len(self._token_bytes) + 1 - len(verified), 256),
                    )
                    if not chunk:
                        break
                    verified.extend(chunk)
                if not secrets.compare_digest(verified, self._token_bytes):
                    raise QuackStateServerTokenError(
                        "rollback temporary token bytes did not verify"
                    )
            finally:
                _wipe_token_bytes(verified)
            post_read = os.fstat(temporary_fd)
            post_path = os.stat(
                temporary_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if post_read.st_nlink != 1 or post_path.st_nlink != 1:
                self._persist_compromise_marker(
                    reason="rollback_temporary_inode_has_surviving_hardlink"
                )
            if (
                (int(post_read.st_dev), int(post_read.st_ino))
                != temporary_identity
                or (int(post_path.st_dev), int(post_path.st_ino))
                != temporary_identity
                or post_read.st_nlink != 1
                or post_path.st_nlink != 1
                or post_read.st_size != len(self._token_bytes)
                or post_path.st_size != len(self._token_bytes)
                or stat_module.S_IMODE(post_read.st_mode) != 0o600
                or stat_module.S_IMODE(post_path.st_mode) != 0o600
            ):
                raise QuackStateServerTokenError(
                    "rollback temporary token changed before publication"
                )
            # Persist the complete temporary inode before it can become the
            # canonical handoff.  A crash before publication can therefore
            # leave only a complete owner-only recovery artifact, never a
            # partial canonical credential.
            os.fsync(self._directory_fd)
            self._verify_directory()
            self._verify_lock()
            self._require_path_absent()
            os.link(
                temporary_name,
                self._filename,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            published = True
            linked = os.fstat(temporary_fd)
            canonical = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            temporary_path = os.stat(
                temporary_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if (
                linked.st_nlink != 2
                or canonical.st_nlink != 2
                or temporary_path.st_nlink != 2
            ):
                self._persist_compromise_marker(
                    reason="rollback_published_inode_has_surviving_hardlink"
                )
            if (
                (int(linked.st_dev), int(linked.st_ino)) != temporary_identity
                or (int(canonical.st_dev), int(canonical.st_ino))
                != temporary_identity
                or (int(temporary_path.st_dev), int(temporary_path.st_ino))
                != temporary_identity
                or linked.st_nlink != 2
                or canonical.st_nlink != 2
                or temporary_path.st_nlink != 2
            ):
                raise QuackStateServerTokenError(
                    "rollback token inode gained an unexpected hardlink"
                )
            os.unlink(temporary_name, dir_fd=self._directory_fd)
            remaining = os.fstat(temporary_fd)
            canonical = os.stat(
                self._filename,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if remaining.st_nlink != 1 or canonical.st_nlink != 1:
                self._persist_compromise_marker(
                    reason="rollback_published_inode_has_surviving_hardlink"
                )
            if (
                (int(remaining.st_dev), int(remaining.st_ino))
                != temporary_identity
                or (int(canonical.st_dev), int(canonical.st_ino))
                != temporary_identity
                or remaining.st_nlink != 1
                or canonical.st_nlink != 1
            ):
                raise QuackStateServerTokenError(
                    "rollback token inode retained an unexpected hardlink"
                )
            os.fsync(self._directory_fd)
            self._verify_directory()
            self._verify_lock()
            self._verify_restored_token(expected_identity=temporary_identity)
            self._verify_lock()
            self._verify_directory()
        except FileExistsError as exc:
            raise QuackStateServerTokenError(
                "token handoff pathname was recreated during retirement"
            ) from exc
        except (OSError, QuackStateServerTokenError) as exc:
            if isinstance(exc, QuackStateServerTokenError):
                raise
            raise QuackStateServerTokenError(
                "token handoff rollback could not restore exact bytes"
            ) from exc
        finally:
            if temporary_fd >= 0 and temporary_identity is not None:
                remaining_links = self._unlink_temporary_token(
                    temporary_fd=temporary_fd,
                    temporary_name=temporary_name,
                    temporary_identity=temporary_identity,
                )
                canonical_is_exact = self._canonical_token_identity_is(
                    temporary_identity
                )
                allowed_links = 1 if canonical_is_exact else 0
                if remaining_links < 0 or remaining_links > allowed_links:
                    compromise = self._finish_compromised(
                        reason=(
                            "rollback_published_inode_has_surviving_hardlink"
                            if published
                            else "rollback_temporary_inode_has_surviving_hardlink"
                        ),
                        observed_link_count=remaining_links,
                    )
                    try:
                        os.close(temporary_fd)
                    except OSError:
                        pass
                    raise compromise
            if temporary_fd >= 0:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass

    def rollback(self) -> dict[str, Any]:
        """Restore exact removed bytes, or preserve a pre-existing absence."""

        self._require_owner_process()
        if self._state == "rolled_back":
            assert self._receipt is not None
            return dict(self._receipt)
        if self._state != "begun":
            raise QuackStateServerTokenError(
                "committed token handoff retirement cannot be rolled back"
            )
        self._require_no_compromise_marker()
        self._verify_directory()
        self._verify_lock()
        self._require_path_absent()
        restored = False
        if not self._already_absent:
            try:
                self._restore_token_atomically()
            except QuackStateServerTokenCompromisedError:
                raise
            except BaseException:
                # A failure after atomic publication may have restored the
                # canonical path without establishing all durability checks.
                # Do not report a rollback receipt, and do not let finalization
                # mistake that indeterminate outcome for a retryable begin.
                if self._state == "begun" and self._filename_is_expected_token():
                    self._abandon()
                raise
            restored = True
        else:
            self._require_path_absent()
            self._verify_lock()
            self._verify_directory()
        return self._finish(
            state="rolled_back",
            receipt={
                "schema": (
                    "ipfs_accelerate_py/"
                    "quack-token-handoff-retirement-rollback@1"
                ),
                "rolled_back": True,
                "restored": restored,
                "already_absent": self._already_absent,
                "secret_handle": self._secret_handle,
            },
        )

def begin_token_handoff_retirement(
    *,
    state_dir: Path | str,
    secret_handle: str,
    expected_token: str,
) -> TokenHandoffRetirement:
    """Begin an exact retirement that stays rollback-capable until commit.

    The caller must first authenticate the live owner with ``expected_token``.
    A missing handoff is a valid transaction that rollback will not recreate.
    A present file is unlinked only after the pre-existing strict validation.
    """

    token = str(expected_token or "")
    try:
        token_bytes = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise QuackStateServerTokenError(
            "expected token is not an ASCII transport credential"
        ) from exc
    if not 8 <= len(token_bytes) <= 1024 or token.strip() != token:
        raise QuackStateServerTokenError("expected token has an invalid shape")
    credential_sha256 = "sha256:" + hashlib.sha256(token_bytes).hexdigest()

    handle = str(secret_handle or "")
    filename = _token_handoff_filename(secret_handle)
    directory, directory_fd, directory_stat = _open_token_handoff_directory(
        state_dir
    )
    directory_owned = True
    lock_fd = -1
    lock_owned = False
    observed = bytearray()
    observed_transferred = False
    try:
        lock_fd, lock_filename, lock_identity = _acquire_token_handoff_lock(
            directory_fd=directory_fd,
            filename=filename,
        )
        lock_owned = True
        _require_no_token_compromise_marker(
            directory_fd=directory_fd,
            filename=filename,
            secret_handle=handle,
            credential_sha256=credential_sha256,
        )
        _recover_linked_rollback_temp(
            directory_fd=directory_fd,
            filename=filename,
            secret_handle=handle,
            credential_sha256=credential_sha256,
            expected_token=token_bytes,
        )
        directory_identity = (
            int(directory_stat.st_dev),
            int(directory_stat.st_ino),
            int(directory_stat.st_uid),
            stat_module.S_IMODE(directory_stat.st_mode),
        )
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            token_fd = os.open(filename, file_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            transaction = TokenHandoffRetirement(
                _construction_authority=_TOKEN_HANDOFF_CONSTRUCTION_AUTHORITY,
                directory=directory,
                directory_fd=directory_fd,
                directory_identity=directory_identity,
                filename=filename,
                lock_fd=lock_fd,
                lock_filename=lock_filename,
                lock_identity=lock_identity,
                secret_handle=handle,
                credential_sha256=credential_sha256,
                token_bytes=observed,
                already_absent=True,
            )
            directory_owned = False
            lock_owned = False
            observed_transferred = True
            return transaction
        except OSError as exc:
            raise QuackStateServerTokenError(
                "token handoff could not be opened safely"
            ) from exc
        try:
            opened = os.fstat(token_fd)
            if opened.st_nlink != 1:
                reason = "credential_hardlink_detected_before_retirement"
                _persist_token_compromise_marker(
                    directory_fd=directory_fd,
                    filename=filename,
                    secret_handle=handle,
                    credential_sha256=credential_sha256,
                    reason=reason,
                )
                raise QuackStateServerTokenCompromisedError(
                    "token handoff credential has unexpected hardlinks",
                    transaction=None,
                    receipt=_token_compromise_receipt(
                        secret_handle=handle,
                        credential_sha256=credential_sha256,
                        reason=reason,
                    ),
                )
            if (
                not stat_module.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or stat_module.S_IMODE(opened.st_mode) != 0o600
            ):
                raise QuackStateServerTokenError(
                    "token handoff file ownership or mode is invalid"
                )
            while len(observed) <= 1024:
                chunk = os.read(token_fd, min(1025 - len(observed), 256))
                if not chunk:
                    break
                observed.extend(chunk)
            if len(observed) > 1024 or not secrets.compare_digest(
                observed,
                token_bytes,
            ):
                raise QuackStateServerTokenError(
                    "token handoff does not match the authenticated owner"
                )
            transaction = TokenHandoffRetirement(
                _construction_authority=_TOKEN_HANDOFF_CONSTRUCTION_AUTHORITY,
                directory=directory,
                directory_fd=directory_fd,
                directory_identity=directory_identity,
                filename=filename,
                lock_fd=lock_fd,
                lock_filename=lock_filename,
                lock_identity=lock_identity,
                secret_handle=handle,
                credential_sha256=credential_sha256,
                token_bytes=observed,
                already_absent=False,
            )
            # Revalidate every mutable file fact immediately before unlink.
            transaction._verify_directory()
            transaction._verify_lock()
            opened_now = os.fstat(token_fd)
            current = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if opened_now.st_nlink != 1 or (
                current.st_dev == opened.st_dev
                and current.st_ino == opened.st_ino
                and current.st_nlink != 1
            ):
                compromise = transaction._finish_compromised(
                    reason="credential_hardlink_detected_during_retirement",
                    observed_link_count=int(opened_now.st_nlink),
                )
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                raise compromise
            if (
                current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino
                or current.st_mode != opened.st_mode
                or current.st_uid != opened.st_uid
                or current.st_nlink != 1
                or current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
                or current.st_ctime_ns != opened.st_ctime_ns
                or opened_now.st_dev != opened.st_dev
                or opened_now.st_ino != opened.st_ino
                or opened_now.st_mode != opened.st_mode
                or opened_now.st_uid != opened.st_uid
                or opened_now.st_nlink != 1
                or opened_now.st_size != opened.st_size
                or opened_now.st_mtime_ns != opened.st_mtime_ns
                or opened_now.st_ctime_ns != opened.st_ctime_ns
            ):
                transaction._abandon()
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                raise QuackStateServerTokenError(
                    "token handoff changed during retirement"
                )
            try:
                os.unlink(filename, dir_fd=directory_fd)
            except OSError as exc:
                transaction._abandon()
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                raise QuackStateServerTokenError(
                    "token handoff could not be unlinked safely"
                ) from exc
            try:
                removed = os.fstat(token_fd)
            except OSError as exc:
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                try:
                    transaction.rollback()
                except QuackStateServerTokenCompromisedError:
                    raise
                except BaseException as rollback_exc:
                    if transaction.state == "begun":
                        transaction._abandon()
                    raise QuackStateServerTokenError(
                        "token handoff post-unlink verification and rollback failed"
                    ) from rollback_exc
                raise QuackStateServerTokenError(
                    "token handoff post-unlink verification failed and was rolled back"
                ) from exc
            if (
                removed.st_dev != opened.st_dev
                or removed.st_ino != opened.st_ino
            ):
                transaction._abandon()
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                raise QuackStateServerTokenError(
                    "token handoff descriptor identity changed after unlink"
                )
            if removed.st_nlink != 0:
                compromise = transaction._finish_compromised(
                    reason="retired_inode_has_surviving_hardlink",
                    observed_link_count=int(removed.st_nlink),
                )
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                raise compromise
            try:
                os.fsync(directory_fd)
                transaction._require_path_absent()
                transaction._verify_lock()
                transaction._verify_directory()
            except (OSError, QuackStateServerTokenError) as exc:
                directory_owned = False
                lock_owned = False
                observed_transferred = True
                try:
                    transaction.rollback()
                except QuackStateServerTokenCompromisedError:
                    raise
                except QuackStateServerTokenError as rollback_exc:
                    if transaction.state == "begun":
                        transaction._abandon()
                    raise QuackStateServerTokenError(
                        "token handoff retirement and rollback both failed"
                    ) from rollback_exc
                if isinstance(exc, QuackStateServerTokenError):
                    raise
                raise QuackStateServerTokenError(
                    "token handoff could not be retired durably"
                ) from exc
        finally:
            os.close(token_fd)
        directory_owned = False
        lock_owned = False
        observed_transferred = True
        return transaction
    finally:
        if not observed_transferred:
            _wipe_token_bytes(observed)
        if lock_owned:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if directory_owned:
            os.close(directory_fd)


def retire_token_handoff(
    *,
    state_dir: Path | str,
    secret_handle: str,
    expected_token: str,
) -> dict[str, Any]:
    """Retire a token handoff compatibly via begin followed by commit."""

    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=secret_handle,
        expected_token=expected_token,
    )
    try:
        return transaction.commit()
    except BaseException:
        if transaction.state == "begun":
            try:
                transaction.rollback()
            except QuackStateServerTokenCompromisedError:
                raise
            except QuackStateServerTokenError:
                if transaction.state == "begun":
                    transaction._abandon()
        raise


def _begin_token_handoff_rearm(
    *,
    state_dir: Path | str,
    secret_handle: str,
    expected_token: str,
) -> TokenHandoffRetirement:
    """Acquire the retirement lock with caller-supplied rearm bytes retained."""

    handle = str(secret_handle or "")
    filename = _token_handoff_filename(secret_handle)
    token = str(expected_token or "")
    try:
        token_bytes = bytearray(token, "ascii")
    except UnicodeEncodeError as exc:
        raise QuackStateServerTokenError(
            "expected token is not an ASCII transport credential"
        ) from exc
    if not 8 <= len(token_bytes) <= 1024 or token.strip() != token:
        _wipe_token_bytes(token_bytes)
        raise QuackStateServerTokenError("expected token has an invalid shape")
    credential_sha256 = "sha256:" + hashlib.sha256(token_bytes).hexdigest()
    try:
        directory, directory_fd, directory_stat = _open_token_handoff_directory(
            state_dir
        )
    except BaseException:
        _wipe_token_bytes(token_bytes)
        raise
    lock_fd = -1
    try:
        lock_fd, lock_filename, lock_identity = _acquire_token_handoff_lock(
            directory_fd=directory_fd,
            filename=filename,
        )
        _require_no_token_compromise_marker(
            directory_fd=directory_fd,
            filename=filename,
            secret_handle=handle,
            credential_sha256=credential_sha256,
        )
        _recover_linked_rollback_temp(
            directory_fd=directory_fd,
            filename=filename,
            secret_handle=handle,
            credential_sha256=credential_sha256,
            expected_token=token_bytes,
        )
        return TokenHandoffRetirement(
            _construction_authority=_TOKEN_HANDOFF_CONSTRUCTION_AUTHORITY,
            directory=directory,
            directory_fd=directory_fd,
            directory_identity=(
                int(directory_stat.st_dev),
                int(directory_stat.st_ino),
                int(directory_stat.st_uid),
                stat_module.S_IMODE(directory_stat.st_mode),
            ),
            filename=filename,
            lock_fd=lock_fd,
            lock_filename=lock_filename,
            lock_identity=lock_identity,
            secret_handle=handle,
            credential_sha256=credential_sha256,
            token_bytes=token_bytes,
            # Rearm is allowed to restore caller-authenticated bytes.  This
            # internal guard is never returned as a retirement transaction.
            already_absent=False,
        )
    except BaseException:
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        try:
            os.close(directory_fd)
        except OSError:
            pass
        _wipe_token_bytes(token_bytes)
        raise


def _token_handoff_rearm_receipt(
    transaction: TokenHandoffRetirement,
) -> dict[str, Any]:
    return {
        "schema": TOKEN_HANDOFF_REARM_SCHEMA,
        "rearmed": True,
        "secret_handle": transaction.secret_handle,
        "credential_sha256": transaction.credential_sha256,
    }


def _verify_rearm_target(
    transaction: TokenHandoffRetirement,
) -> bool:
    """Return true for an exact existing token, false for a stable absence."""

    transaction._verify_directory()
    transaction._verify_lock()
    try:
        observed = os.stat(
            transaction._filename,
            dir_fd=transaction._directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        transaction._verify_lock()
        transaction._verify_directory()
        return False
    except OSError as exc:
        raise QuackStateServerTokenError(
            "token handoff rearm target cannot be inspected"
        ) from exc
    expected_identity = (int(observed.st_dev), int(observed.st_ino))
    if observed.st_nlink != 1:
        raise transaction._finish_compromised(
            reason="credential_hardlink_detected_during_rearm",
            observed_link_count=int(observed.st_nlink),
        )
    try:
        transaction._verify_restored_token(
            expected_identity=expected_identity,
        )
    except (OSError, QuackStateServerTokenError) as exc:
        raise QuackStateServerTokenError(
            "existing token handoff differs from exact rearm bytes"
        ) from exc
    transaction._verify_lock()
    transaction._verify_directory()
    return True


def rearm_token_handoff(
    *,
    state_dir: Path | str,
    secret_handle: str,
    expected_token: str,
) -> dict[str, Any]:
    """Explicitly and atomically republish one exact coordinator credential.

    An exact owner-only existing handoff is the only idempotent success.  Any
    other existing object is left untouched and fails closed.  This operation
    is deliberately explicit; :class:`TokenVault` never rearms automatically.
    """

    transaction = _begin_token_handoff_rearm(
        state_dir=state_dir,
        secret_handle=secret_handle,
        expected_token=expected_token,
    )
    receipt = _token_handoff_rearm_receipt(transaction)
    try:
        if _verify_rearm_target(transaction):
            return transaction._finish(state="rearmed", receipt=receipt)
        transaction.rollback()
        return dict(receipt)
    except BaseException:
        if transaction.state == "begun":
            transaction._abandon()
        raise


def _coordinator_pid_projection_liveness(pid: int) -> OwnerLiveness:
    """Classify a bare scheduler PID conservatively."""

    return owner_liveness(
        ProcessBirthIdentity(
            pid=int(pid),
            start_time_ticks=0,
            boot_id="",
            parent_pid=0,
        )
    )


def _classify_coordinator_pid_projection(
    coordinator_pid_path: Path | str,
) -> str:
    """Return absent/empty/dead/alive/unknown/malformed/unsafe."""

    try:
        path = Path(coordinator_pid_path)
        if (
            not path.is_absolute()
            or Path(os.path.abspath(os.fspath(path))) != path
            or path.name in {"", ".", ".."}
        ):
            return "unsafe"
        _directory, directory_fd, _directory_stat = (
            _open_token_handoff_directory(path.parent)
        )
    except (OSError, TypeError, ValueError, QuackStateServerTokenError):
        return "unsafe"
    descriptor = -1
    observed_bytes = bytearray()
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unsafe"
        opened = os.fstat(descriptor)
        observed = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        fingerprint = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_uid),
            stat_module.S_IMODE(opened.st_mode),
            int(opened.st_nlink),
            int(opened.st_size),
            int(opened.st_ctime_ns),
            int(opened.st_mtime_ns),
        )
        observed_fingerprint = (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_uid),
            stat_module.S_IMODE(observed.st_mode),
            int(observed.st_nlink),
            int(observed.st_size),
            int(observed.st_ctime_ns),
            int(observed.st_mtime_ns),
        )
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or fingerprint != observed_fingerprint
            or fingerprint[2] != os.geteuid()
            or fingerprint[3] != 0o600
            or fingerprint[4] != 1
            or fingerprint[5] > 32
        ):
            return "unsafe"
        while len(observed_bytes) <= 32:
            chunk = os.read(descriptor, 33 - len(observed_bytes))
            if not chunk:
                break
            observed_bytes.extend(chunk)
        post_opened = os.fstat(descriptor)
        post_observed = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        post_fingerprint = (
            int(post_opened.st_dev),
            int(post_opened.st_ino),
            int(post_opened.st_uid),
            stat_module.S_IMODE(post_opened.st_mode),
            int(post_opened.st_nlink),
            int(post_opened.st_size),
            int(post_opened.st_ctime_ns),
            int(post_opened.st_mtime_ns),
        )
        post_observed_fingerprint = (
            int(post_observed.st_dev),
            int(post_observed.st_ino),
            int(post_observed.st_uid),
            stat_module.S_IMODE(post_observed.st_mode),
            int(post_observed.st_nlink),
            int(post_observed.st_size),
            int(post_observed.st_ctime_ns),
            int(post_observed.st_mtime_ns),
        )
        if (
            post_fingerprint != fingerprint
            or post_observed_fingerprint != observed_fingerprint
        ):
            return "unsafe"
        if not observed_bytes:
            return "empty"
        try:
            pid = int(observed_bytes.decode("ascii").removesuffix("\n"))
        except (UnicodeDecodeError, ValueError):
            return "malformed"
        if pid <= 0 or observed_bytes != f"{pid}\n".encode("ascii"):
            return "malformed"
        liveness = _coordinator_pid_projection_liveness(pid)
        if liveness is OwnerLiveness.ALIVE:
            return "alive"
        if liveness is OwnerLiveness.DEAD:
            return "dead"
        return "unknown"
    except OSError:
        return "unsafe"
    finally:
        _wipe_token_bytes(observed_bytes)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _quarantine_coordinator_pid_projection(
    coordinator_pid_path: Path | str,
    *,
    expected_state: str,
) -> bool:
    """Atomically move exact empty/dead PID evidence to a private sibling dir."""

    if expected_state not in {"empty", "dead"}:
        return False
    if _classify_coordinator_pid_projection(coordinator_pid_path) != expected_state:
        return False
    path = Path(coordinator_pid_path)
    try:
        _directory, directory_fd, _directory_stat = (
            _open_token_handoff_directory(path.parent)
        )
    except (OSError, QuackStateServerTokenError):
        return False
    descriptor = -1
    quarantine_fd = -1
    reservation_fd = -1
    quarantine_name = ".quack-coordinator-pid-quarantine"
    evidence_name = (
        f"{path.name}.{expected_state}.{time.time_ns()}."
        f"{secrets.token_hex(8)}"
    )
    payload = bytearray()
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        observed = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        identity = (int(opened.st_dev), int(opened.st_ino))
        fingerprint = (
            *identity,
            int(opened.st_uid),
            stat_module.S_IMODE(opened.st_mode),
            int(opened.st_nlink),
            int(opened.st_size),
            int(opened.st_ctime_ns),
            int(opened.st_mtime_ns),
        )
        observed_fingerprint = (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_uid),
            stat_module.S_IMODE(observed.st_mode),
            int(observed.st_nlink),
            int(observed.st_size),
            int(observed.st_ctime_ns),
            int(observed.st_mtime_ns),
        )
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or not stat_module.S_ISREG(observed.st_mode)
            or fingerprint != observed_fingerprint
            or fingerprint[2] != os.geteuid()
            or fingerprint[3] != 0o600
            or fingerprint[4] != 1
            or fingerprint[5] > 32
        ):
            return False
        while len(payload) <= 32:
            chunk = os.read(descriptor, 33 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if expected_state == "empty":
            if payload:
                return False
        else:
            try:
                pid = int(payload.decode("ascii").removesuffix("\n"))
            except (UnicodeDecodeError, ValueError):
                return False
            if (
                pid <= 0
                or payload != f"{pid}\n".encode("ascii")
                or _coordinator_pid_projection_liveness(pid)
                is not OwnerLiveness.DEAD
            ):
                return False
        post_opened = os.fstat(descriptor)
        post_observed = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            (
                int(post_opened.st_dev),
                int(post_opened.st_ino),
                int(post_opened.st_uid),
                stat_module.S_IMODE(post_opened.st_mode),
                int(post_opened.st_nlink),
                int(post_opened.st_size),
                int(post_opened.st_ctime_ns),
                int(post_opened.st_mtime_ns),
            )
            != fingerprint
            or (
                int(post_observed.st_dev),
                int(post_observed.st_ino),
                int(post_observed.st_uid),
                stat_module.S_IMODE(post_observed.st_mode),
                int(post_observed.st_nlink),
                int(post_observed.st_size),
                int(post_observed.st_ctime_ns),
                int(post_observed.st_mtime_ns),
            )
            != observed_fingerprint
        ):
            return False
        try:
            os.mkdir(quarantine_name, 0o700, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileExistsError:
            pass
        quarantine_fd = os.open(
            quarantine_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        quarantine_opened = os.fstat(quarantine_fd)
        quarantine_observed = os.stat(
            quarantine_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat_module.S_ISDIR(quarantine_opened.st_mode)
            or not stat_module.S_ISDIR(quarantine_observed.st_mode)
            or quarantine_opened.st_uid != os.geteuid()
            or quarantine_observed.st_uid != os.geteuid()
            or stat_module.S_IMODE(quarantine_opened.st_mode) != 0o700
            or stat_module.S_IMODE(quarantine_observed.st_mode) != 0o700
            or (int(quarantine_opened.st_dev), int(quarantine_opened.st_ino))
            != (int(quarantine_observed.st_dev), int(quarantine_observed.st_ino))
        ):
            return False
        reservation_fd = os.open(
            evidence_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=quarantine_fd,
        )
        reserved = os.fstat(reservation_fd)
        reserved_path = os.stat(
            evidence_name,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        reserved_identity = (int(reserved.st_dev), int(reserved.st_ino))
        if (
            not stat_module.S_ISREG(reserved.st_mode)
            or reserved.st_uid != os.geteuid()
            or stat_module.S_IMODE(reserved.st_mode) != 0o600
            or reserved.st_nlink != 1
            or reserved.st_size != 0
            or (int(reserved_path.st_dev), int(reserved_path.st_ino))
            != reserved_identity
        ):
            return False
        # Python has no portable renameat2(RENAME_NOREPLACE).  Reserve a
        # cryptographically unique destination with O_EXCL, verify that exact
        # slot, then atomically replace only our own empty inode.
        os.rename(
            path.name,
            evidence_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=quarantine_fd,
        )
        os.fsync(directory_fd)
        os.fsync(quarantine_fd)
        final = os.stat(
            evidence_name,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        try:
            os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            return False
        return (
            (int(final.st_dev), int(final.st_ino)) == identity
            and stat_module.S_ISREG(final.st_mode)
            and final.st_uid == os.geteuid()
            and stat_module.S_IMODE(final.st_mode) == 0o600
            and final.st_nlink == 1
        )
    except OSError:
        return False
    finally:
        _wipe_token_bytes(payload)
        for owned_fd in (
            descriptor,
            reservation_fd,
            quarantine_fd,
            directory_fd,
        ):
            if owned_fd >= 0:
                try:
                    os.close(owned_fd)
                except OSError:
                    pass


def _token_handoff_rearm_probe_receipt(
    *,
    secret_handle: str,
    credential_sha256: str,
    rearmed: bool,
    pid_quarantined: bool,
    reason: str,
) -> dict[str, Any]:
    if reason not in TOKEN_HANDOFF_REARM_PROBE_REASONS:
        raise QuackStateServerTokenError(
            "token handoff rearm probe reason is not allowed"
        )
    return {
        "schema": TOKEN_HANDOFF_REARM_PROBE_SCHEMA,
        "closed": True,
        "rearmed": bool(rearmed),
        "pid_quarantined": bool(pid_quarantined),
        "recovery_admitted": bool(rearmed or pid_quarantined),
        "completion_authority": False,
        "task_authority": False,
        "reason": str(reason),
        "secret_handle": secret_handle,
        "credential_sha256": credential_sha256,
    }


def rearm_token_handoff_if_coordinator_absent(
    *,
    state_dir: Path | str,
    secret_handle: str,
    expected_token: str,
    coordinator_pid_path: Path | str,
) -> dict[str, Any]:
    """Conservatively repair a handoff after proven coordinator absence.

    Exact empty/dead PID evidence is atomically moved into a private adjacent
    quarantine and never discarded.  A live, unknown, malformed, unsafe, or
    concurrently locked condition is a typed non-authoritative no-op.  The
    probe closes the ordinary SIGKILL recovery gap only while the state owner
    remains alive with the exact credential; simultaneous process loss still
    requires external credential recovery.
    """

    try:
        transaction = _begin_token_handoff_rearm(
            state_dir=state_dir,
            secret_handle=secret_handle,
            expected_token=expected_token,
        )
    except _TokenHandoffLockHeld:
        token = str(expected_token or "")
        try:
            encoded = token.encode("ascii")
        except UnicodeEncodeError:
            encoded = b""
        return _token_handoff_rearm_probe_receipt(
            secret_handle=str(secret_handle or ""),
            credential_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
            rearmed=False,
            pid_quarantined=False,
            reason="retirement_lock_held",
        )
    handle = transaction.secret_handle
    credential_sha256 = transaction.credential_sha256
    try:
        try:
            already_present = _verify_rearm_target(transaction)
        except QuackStateServerTokenError:
            transaction._abandon()
            return _token_handoff_rearm_probe_receipt(
                secret_handle=handle,
                credential_sha256=credential_sha256,
                rearmed=False,
                pid_quarantined=False,
                reason="handoff_unsafe",
            )
        pid_state = _classify_coordinator_pid_projection(coordinator_pid_path)
        if pid_state not in {"absent", "empty", "dead"}:
            transaction._abandon()
            return _token_handoff_rearm_probe_receipt(
                secret_handle=handle,
                credential_sha256=credential_sha256,
                rearmed=False,
                pid_quarantined=False,
                reason=f"coordinator_pid_{pid_state}",
            )
        pid_quarantined = False
        if pid_state in {"empty", "dead"}:
            pid_quarantined = _quarantine_coordinator_pid_projection(
                coordinator_pid_path,
                expected_state=pid_state,
            )
            pid_confirmed = pid_quarantined
        else:
            pid_confirmed = (
                _classify_coordinator_pid_projection(coordinator_pid_path)
                == "absent"
            )
        if not pid_confirmed:
            transaction._abandon()
            return _token_handoff_rearm_probe_receipt(
                secret_handle=handle,
                credential_sha256=credential_sha256,
                rearmed=False,
                pid_quarantined=False,
                reason="coordinator_pid_changed",
            )
        if already_present:
            transaction._abandon()
            return _token_handoff_rearm_probe_receipt(
                secret_handle=handle,
                credential_sha256=credential_sha256,
                rearmed=False,
                pid_quarantined=pid_quarantined,
                reason=(
                    f"coordinator_pid_{pid_state}"
                    if pid_quarantined
                    else "handoff_already_present"
                ),
            )
        transaction.rollback()
        return _token_handoff_rearm_probe_receipt(
            secret_handle=handle,
            credential_sha256=credential_sha256,
            rearmed=True,
            pid_quarantined=pid_quarantined,
            reason=f"coordinator_pid_{pid_state}",
        )
    except BaseException:
        if transaction.state == "begun":
            transaction._abandon()
        raise


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
        return open_duckdb_connection(
            self.config.database_path, prefer_quack=False
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
        expected = self.config.expected_generation
        if (
            self.config.reuse_expected_generation
            and expected is not None
            and current == expected
        ):
            return expected
        return max(1, current + 1)

    def _assert_expected_startup_binding(
        self,
        *,
        generation: int,
        database_uuid: str,
        store_id: str,
        uri: str,
    ) -> None:
        """Reject a start that differs from an operator-sealed successor identity."""

        mismatches: list[str] = []
        if (
            self.config.expected_generation is not None
            and generation != self.config.expected_generation
        ):
            mismatches.append("generation")
        if (
            self.config.expected_database_uuid is not None
            and database_uuid != self.config.expected_database_uuid
        ):
            mismatches.append("database_uuid")
        if (
            self.config.expected_store_id is not None
            and store_id != self.config.expected_store_id
        ):
            mismatches.append("store_id")
        if (
            self.config.expected_listen_uri is not None
            and uri != self.config.expected_listen_uri
        ):
            mismatches.append("listen_uri")
        if mismatches:
            raise QuackStateServerControlError(
                "state-owner startup binding differs: " + ", ".join(mismatches)
            )

    @staticmethod
    def _next_credential_generation(connection: Any, secret_handle: str) -> int:
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM credentials "
                "WHERE secret_handle = ?",
                [secret_handle],
            ).fetchone()
        except Exception as exc:
            raise QuackStateServerMigrationError(
                "credential generation authority is unavailable after migration"
            ) from exc
        current = int(
            row[0]
            if row is not None and not isinstance(row, Mapping)
            else (
                row.get(list(row.keys())[0], 0)
                if isinstance(row, Mapping) and row
                else 0
            )
        )
        return current + 1

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
                # Generation is finalized after opening the DB.  The marker is
                # an OS-bootstrap projection and therefore retains this
                # deliberately provisional generation rather than claiming the
                # later durable store generation.
                owner.acquire(
                    server_id=server_id,
                    process_birth=birth,
                    database_path=self.config.database_path,
                    generation=PROVISIONAL_OWNER_MARKER_GENERATION,
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
                uri = listen_uri(self.config.host, port)
                self._assert_expected_startup_binding(
                    generation=generation,
                    database_uuid=database_uuid,
                    store_id=self.config.store_id,
                    uri=uri,
                )
                self._bound_port = port
                secret_handle = self.config.resolved_secret_handle(server_id, generation)
                credential_generation = self._next_credential_generation(
                    connection, secret_handle
                )
                assert self._vault is not None
                self._vault.mint(
                    secret_handle=secret_handle,
                    generation=credential_generation,
                )
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
                    credential_generation=credential_generation,
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
                connection.execute(
                    "UPDATE state_servers SET status = ?, revision = revision + 1 "
                    "WHERE server_id = ? AND generation = ?",
                    ["ready", identity.server_id, identity.generation],
                )
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


def recover_stale_state_server(
    *,
    database_path: Path,
    state_dir: Path,
    expected_store_id: str,
    expected_generation: int,
    expected_database_uuid: str,
    liveness: Callable[[ProcessBirthIdentity], OwnerLiveness] | None = None,
    stopped_at: str | None = None,
    receipt_filename: str = STALE_OWNER_RECOVERY_RECEIPT_FILENAME,
) -> dict[str, Any]:
    """Settle a process-dead owner's canonical stop bookkeeping exactly once.

    This is an operator recovery path, not ordinary shutdown.  It requires a
    valid owner marker, a matching published status identity, a proved-dead
    process birth, the exclusive owner lock, and exact canonical database
    rows.  It updates only the rows written by normal ``stop()``, publishes a
    stopped status projection, removes the stale marker, and writes its
    content-addressed recovery receipt last.
    """

    receipt_basename = _validate_recovery_receipt_filename(receipt_filename)
    database = Path(database_path).resolve(strict=True)
    runtime = Path(state_dir).resolve(strict=True)
    marker_path = database.with_name(f".{database.name}.state-owner.json")
    lock_path = database.with_name(f".{database.name}.state-owner.lock")
    status_path = runtime / STATUS_FILENAME
    stop_path = runtime / CONTROL_STOP_FILENAME
    receipt_path = runtime / receipt_basename
    if receipt_path in {
        database,
        marker_path,
        lock_path,
        status_path,
        stop_path,
    }:
        raise QuackStateServerControlError(
            "stale-owner recovery receipt filename is not a confined basename"
        )
    try:
        database_stat = database.lstat()
    except OSError as exc:
        raise QuackStateServerControlError(
            "stale-owner recovery database is unavailable"
        ) from exc
    if (
        not stat_module.S_ISREG(database_stat.st_mode)
        or database_stat.st_nlink != 1
    ):
        raise QuackStateServerControlError(
            "stale-owner recovery database is not a single-link regular file"
        )

    try:
        marker_path.lstat()
    except FileNotFoundError:
        marker_payload: dict[str, Any] | None = None
        marker: OwnerMarker | None = None
    except OSError as exc:
        raise QuackStateServerControlError(
            "stale-owner recovery marker is unavailable"
        ) from exc
    else:
        marker_payload = _read_stable_regular_json(
            marker_path,
            noun="stale-owner recovery marker",
        )
        try:
            marker = OwnerMarker.from_dict(marker_payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise QuackStateServerControlError(
                "stale-owner recovery marker is invalid"
            ) from exc

    status_payload = _read_stable_regular_json(
        status_path,
        noun="stale-owner recovery status",
    )
    identity_payload = status_payload.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise QuackStateServerControlError(
            "stale-owner recovery status has no identity"
        )
    try:
        identity = StateServerIdentity(
            server_id=str(identity_payload.get("server_id") or ""),
            store_id=str(identity_payload.get("store_id") or ""),
            database_uuid=str(identity_payload.get("database_uuid") or ""),
            schema_revision=int(identity_payload.get("schema_revision") or 0),
            schema_fingerprint=str(
                identity_payload.get("schema_fingerprint") or ""
            ),
            generation=int(identity_payload.get("generation") or 0),
            fence_epoch=int(identity_payload.get("fence_epoch") or 0),
            revision=int(identity_payload.get("revision") or 0),
            process_birth=ProcessBirthIdentity.from_dict(
                identity_payload.get("process_birth")
            ),
            listen_uri=str(identity_payload.get("listen_uri") or ""),
            extension_fingerprint=str(
                identity_payload.get("extension_fingerprint") or ""
            ),
            credential_generation=int(
                identity_payload.get("credential_generation") or 0
            ),
            secret_handle=str(identity_payload.get("secret_handle") or ""),
            repository_id=str(identity_payload.get("repository_id") or ""),
            startup_epoch=int(identity_payload.get("startup_epoch") or 0),
            started_at=str(identity_payload.get("started_at") or ""),
            status=str(identity_payload.get("status") or ""),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise QuackStateServerControlError(
            "stale-owner recovery status identity is invalid"
        ) from exc
    lifecycle = str(status_payload.get("lifecycle") or "")
    recovery_projection = status_payload.get("recovered_stale_owner") is True
    if (
        lifecycle not in {ServerLifecycle.READY.value, ServerLifecycle.STOPPED.value}
        or identity.status != lifecycle
        or str(status_payload.get("database_path") or "") != str(database)
        or str(status_payload.get("state_dir") or "") != str(runtime)
        or identity.store_id != str(expected_store_id)
        or identity.generation != int(expected_generation)
        or identity.database_uuid != str(expected_database_uuid)
        or (lifecycle == ServerLifecycle.STOPPED.value and not recovery_projection)
        or (marker is None and not recovery_projection)
    ):
        raise QuackStateServerControlError(
            "stale-owner recovery identity binding differs"
        )
    if marker is not None and (
        marker.server_id != identity.server_id
        or marker.process_birth != identity.process_birth
        # The owner marker is acquired before DuckDB can assign the durable
        # store generation.  Its @1 contract consequently carries the fixed
        # provisional lease generation, while the status and canonical rows
        # below bind and verify the independently assigned store generation.
        or marker.generation != PROVISIONAL_OWNER_MARKER_GENERATION
        or Path(marker.database_path).resolve() != database
    ):
        raise QuackStateServerControlError(
            "stale-owner recovery marker binding differs"
        )
    liveness_probe = liveness or owner_liveness
    if liveness_probe(identity.process_birth) is not OwnerLiveness.DEAD:
        raise QuackStateServerControlError(
            "stale-owner recovery requires proved-dead process birth"
        )

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except OSError as exc:
        raise QuackStateServerControlError(
            "stale-owner recovery lock is unavailable"
        ) from exc
    lock_handle = os.fdopen(lock_descriptor, "a+b", closefd=True)
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QuackStateServerControlError(
                "stale-owner recovery owner lock is held"
            ) from exc
        if liveness_probe(identity.process_birth) is not OwnerLiveness.DEAD:
            raise QuackStateServerControlError(
                "stale-owner process liveness changed under lock"
            )
        current_database_stat = database.lstat()
        database_identity_fields = ("st_dev", "st_ino", "st_mode", "st_nlink")
        if any(
            getattr(database_stat, field) != getattr(current_database_stat, field)
            for field in database_identity_fields
        ):
            raise QuackStateServerControlError(
                "stale-owner recovery database changed under lock"
            )
        if (
            _read_stable_regular_json(
                status_path,
                noun="stale-owner recovery status",
            )
            != status_payload
        ):
            raise QuackStateServerControlError(
                "stale-owner status changed under lock"
            )
        if marker_payload is not None:
            if (
                _read_stable_regular_json(
                    marker_path,
                    noun="stale-owner recovery marker",
                )
                != marker_payload
            ):
                raise QuackStateServerControlError(
                    "stale-owner marker changed under lock"
                )
        else:
            try:
                marker_path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise QuackStateServerControlError(
                    "stale-owner marker appeared under lock"
                )

        connection = open_duckdb_connection(
            database,
            prefer_quack=False,
            timeout_seconds=5.0,
            memory_limit="256MB",
            threads=1,
        )
        stop_time = ""
        try:
            connection.execute("BEGIN TRANSACTION")
            rows = connection.execute(
                "SELECT store_id, database_uuid, process_birth_id, generation, "
                "status, stopped_at, revision FROM state_servers "
                "WHERE server_id = ?",
                [identity.server_id],
            ).fetchall()
            generation_rows = connection.execute(
                "SELECT database_uuid, birth_id FROM store_generations "
                "WHERE generation = ?",
                [identity.generation],
            ).fetchall()
            epoch_rows = connection.execute(
                "SELECT ended_at FROM server_epochs WHERE server_id = ? "
                "AND epoch = ?",
                [identity.server_id, identity.startup_epoch or identity.generation],
            ).fetchall()
            if (
                len(rows) != 1
                or tuple(rows[0][index] for index in range(4))
                != (
                    identity.store_id,
                    identity.database_uuid,
                    identity.process_birth_id,
                    identity.generation,
                )
                or len(generation_rows) != 1
                or tuple(generation_rows[0][index] for index in range(2))
                != (identity.database_uuid, identity.process_birth_id)
                or len(epoch_rows) != 1
            ):
                raise QuackStateServerControlError(
                    "stale-owner recovery canonical rows differ"
                )
            row_status = str(rows[0][4] or "")
            row_stopped_at = str(rows[0][5] or "")
            prior_revision = int(rows[0][6])
            if row_status == ServerLifecycle.READY.value:
                if rows[0][5] is not None or epoch_rows[0][0] is not None:
                    raise QuackStateServerControlError(
                        "stale-owner ready bookkeeping is inconsistent"
                    )
                stop_time = str(stopped_at or _utc_iso())
                connection.execute(
                    "UPDATE state_servers SET status = 'stopped', stopped_at = ?, "
                    "revision = revision + 1 WHERE server_id = ? AND generation = ? "
                    "AND status = 'ready' AND stopped_at IS NULL AND revision = ?",
                    [
                        stop_time,
                        identity.server_id,
                        identity.generation,
                        prior_revision,
                    ],
                )
                connection.execute(
                    "UPDATE server_epochs SET ended_at = ? WHERE server_id = ? "
                    "AND epoch = ? AND ended_at IS NULL",
                    [
                        stop_time,
                        identity.server_id,
                        identity.startup_epoch or identity.generation,
                    ],
                )
                expected_revision = prior_revision + 1
            elif row_status == ServerLifecycle.STOPPED.value:
                stop_time = row_stopped_at
                if (
                    not stop_time
                    or str(epoch_rows[0][0] or "") != stop_time
                    or (stopped_at is not None and str(stopped_at) != stop_time)
                ):
                    raise QuackStateServerControlError(
                        "stale-owner settled bookkeeping is inconsistent"
                    )
                expected_revision = prior_revision
            else:
                raise QuackStateServerControlError(
                    "stale-owner recovery canonical status differs"
                )
            settled = connection.execute(
                "SELECT status, stopped_at, revision FROM state_servers "
                "WHERE server_id = ? AND generation = ?",
                [identity.server_id, identity.generation],
            ).fetchall()
            settled_epochs = connection.execute(
                "SELECT ended_at FROM server_epochs WHERE server_id = ? "
                "AND epoch = ?",
                [identity.server_id, identity.startup_epoch or identity.generation],
            ).fetchall()
            if (
                len(settled) != 1
                or tuple(settled[0][index] for index in range(3))
                != ("stopped", stop_time, expected_revision)
                or len(settled_epochs) != 1
                or settled_epochs[0][0] != stop_time
            ):
                raise QuackStateServerControlError(
                    "stale-owner recovery stop CAS failed"
                )
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            connection.close()

        stopped_identity = identity.with_status("stopped")
        stopped_status = dict(status_payload)
        stopped_status["lifecycle"] = ServerLifecycle.STOPPED.value
        stopped_status["identity"] = stopped_identity.to_dict()
        stopped_status["recovered_stale_owner"] = True
        stopped_status["recovery_stopped_at"] = stop_time
        _atomic_write_json(status_path, stopped_status, mode=0o600)
        if marker_payload is not None:
            try:
                marker_path.unlink()
            except FileNotFoundError as exc:
                raise QuackStateServerControlError(
                    "stale-owner marker disappeared before recovery publication"
                ) from exc
        try:
            stop_path.unlink()
        except FileNotFoundError:
            pass
        receipt: dict[str, Any] = {
            "schema": STALE_OWNER_RECOVERY_SCHEMA,
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "owner_liveness": OwnerLiveness.DEAD.value,
            "prior_status": ServerLifecycle.READY.value,
            "resulting_status": ServerLifecycle.STOPPED.value,
            "stopped_at": stop_time,
            "database_bookkeeping_settled": True,
            "owner_marker_removed": True,
            "replay_safe": True,
            "task_completion_authority": False,
        }
        receipt["receipt_cid"] = content_identity(receipt)
        if receipt_path.exists():
            published = _read_stable_regular_json(
                receipt_path,
                noun="stale-owner recovery receipt",
            )
            if published != receipt:
                raise QuackStateServerControlError(
                    "stale-owner recovery receipt conflicts"
                )
            return receipt
        _atomic_write_json(receipt_path, receipt, mode=0o600)
        return receipt
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


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
    expected_generation: int | None = None,
    expected_database_uuid: str | None = None,
    expected_store_id: str | None = None,
    expected_listen_uri: str | None = None,
    reuse_expected_generation: bool = False,
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
        expected_generation=expected_generation,
        expected_database_uuid=expected_database_uuid,
        expected_store_id=expected_store_id,
        expected_listen_uri=expected_listen_uri,
        reuse_expected_generation=reuse_expected_generation,
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
    "QuackStateServerTokenCompromisedError",
    "QuackStateServerTokenError",
    "RemoteBindPolicy",
    "STATE_SERVER_IDENTITY_INTERFACE",
    "ServerLifecycle",
    "StateServerIdentity",
    "TokenHandoffRetirement",
    "TokenVault",
    "assert_bind_admitted",
    "begin_token_handoff_retirement",
    "build_server",
    "listen_uri",
    "provider_safe_environment",
    "reclaim_stale_owner_marker",
    "rearm_token_handoff",
    "rearm_token_handoff_if_coordinator_absent",
    "retire_token_handoff",
    "sanitize_for_export",
)
