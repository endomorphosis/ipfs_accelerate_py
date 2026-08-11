"""Durable content-addressed run registry with handle reconstruction and CAS.

Owns high-level durable run records for the prompt-first entrypoints layer
(ASE-013).  Event/artifact stores, process adoption, and task-source mutation
remain outside this module.

Design:

* **Immutable run roots** bind identity fields once (run ID, invocation,
  prompt, target receipt, namespace binding, creation time).
* **Content-addressed handle snapshots** store complete :class:`RunHandle`
  records under their canonical ``content_id``.
* **CAS heads** advance ``run_revision`` under a process-safe exclusive lock;
  two concurrent writers with the same expected revision cannot both commit.
* **Current-run selection** is exact: a unique compatible candidate is chosen
  deterministically; multiple or incompatible populations are reported without
  guessing.
* **Corruption** is quarantined.  Lookup and reconstruction never return a
  canonical-looking handle when integrity checks fail.

On-disk layout under ``registry_root``::

    .run-registry.lock
    quarantine/
    namespaces/<fs-safe-namespace>/
      current.json                 # optional CAS pointer for selected run
      runs/<run_id>/
        root.json                  # immutable root
        head.json                  # CAS head (revision + handle CID)
        handles/<handle_cid>.json  # full RunHandle snapshots

Restart reconstructs a complete handle by verifying root + head + snapshot.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    MultiformatsIdentityError,
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .contracts import (
    ContinuationAction,
    EntrypointContractError,
    RunHandle,
    RunHealth,
    RunState,
)
from .state_resolver import (
    ADOPTABLE_HEALTH,
    ADOPTABLE_RUN_STATES,
    TERMINAL_RUN_STATES,
    ClassifiedRunCandidate,
    RunAdoptionAction,
    RunCandidateClass,
    RunCandidateEvidence,
    RunCandidateResolution,
    RunCandidateResolutionRequest,
    RunCandidateResolver,
    WorktreeIsolationMode,
    classify_run_candidate,
)
# The original file registry remains a compatibility/projection reader.  New
# lifecycle code imports this owner through the familiar registry module so it
# cannot accidentally treat a JSON head as mutable authority.
from .run_registry_backend import DuckDBRunRegistryBackend

RUN_REGISTRY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/run-registry@1"
)
RUN_ROOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/run-root@1"
)
RUN_HEAD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/run-head@1"
)
RUN_TX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/run-registry-tx@1"
)
NAMESPACE_CURRENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/namespace-current@1"
)
RUN_REGISTRY_REQUIREMENT_ID: Final = (
    "run_registry.RUN_REGISTRY_AND_PROMPT_BROKER_REQUIREMENT_ID"
)

LOCK_NAME: Final = ".run-registry.lock"
QUARANTINE_DIR: Final = "quarantine"
NAMESPACES_DIR: Final = "namespaces"
RUNS_DIR: Final = "runs"
HANDLES_DIR: Final = "handles"
ROOT_NAME: Final = "root.json"
HEAD_NAME: Final = "head.json"
CURRENT_NAME: Final = "current.json"

DEFAULT_MAX_LIST: Final = 256
HARD_MAX_LIST: Final = 4_096
MAX_NAMESPACE_BYTES: Final = 512
MAX_REPOSITORY_ID_BYTES: Final = 1_024
MAX_REASON_CODES: Final = 64
MAX_RECORD_BYTES: Final = 2 * 1024 * 1024

# Identity fields that must not diverge from the immutable root after create.
_IMMUTABLE_HANDLE_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "target_resolution_receipt_cid",
    "invocation_cid",
    "prompt_cid",
    "created_at_ms",
)

_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_CID_RE = re.compile(r"^[a-z0-9]+$")


class RunRegistryError(EntrypointContractError):
    """Base error for durable run-registry operations."""


class RunRegistryBoundsError(RunRegistryError):
    """Raised when a bound or size limit is exceeded."""


class RunNotFoundError(RunRegistryError):
    """Raised when a run identity is absent from the registry."""


class RunExistsError(RunRegistryError):
    """Raised when create would overwrite an existing run root."""


class RunCasConflictError(RunRegistryError):
    """Raised when a compare-and-swap revision does not match the head."""

    def __init__(
        self,
        message: str,
        *,
        receipt: "RegistryTransactionReceipt | None" = None,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt


class RunRegistryCorruptionError(RunRegistryError):
    """Raised after corrupt state is quarantined (or cannot be trusted)."""

    def __init__(
        self,
        message: str,
        *,
        quarantine_path: str = "",
        run_id: str = "",
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.quarantine_path = quarantine_path
        self.run_id = run_id
        self.reason_codes = tuple(reason_codes)


class RunIncompatibleError(RunRegistryError):
    """Raised when a handle diverges from the immutable run root."""


class RegistryTxOutcome(str, Enum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"
    NOOP = "noop"
    REPAIRED = "repaired"


class RegistryOperation(str, Enum):
    CREATE = "create"
    CAS_UPDATE = "cas_update"
    SET_CURRENT = "set_current"
    QUARANTINE = "quarantine"
    REPAIR = "repair"
    LOOKUP = "lookup"
    RECONSTRUCT = "reconstruct"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_nonempty(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunRegistryError(f"{name} must be a nonempty string")
    text = value.strip()
    if len(text.encode("utf-8")) > maximum:
        raise RunRegistryBoundsError(f"{name} exceeds {maximum} bytes")
    return text


def _require_cid(value: Any, name: str) -> str:
    text = _require_nonempty(value, name, maximum=MAX_REPOSITORY_ID_BYTES)
    try:
        return validate_cid(text)
    except (MultiformatsIdentityError, ValueError, TypeError) as exc:
        raise RunRegistryError(f"{name} must be a canonical CIDv1") from exc


def _require_token(value: Any, name: str) -> str:
    text = _require_nonempty(value, name, maximum=MAX_NAMESPACE_BYTES)
    if not _TOKEN_RE.match(text):
        raise RunRegistryError(
            f"{name} must match {_TOKEN_RE.pattern} (got {text!r})"
        )
    return text


def _fs_safe_token(token: str) -> str:
    """Map a token with ``:`` into a single path component."""

    if not token or token in {".", ".."} or "/" in token or "\\" in token:
        raise RunRegistryError("unsafe filesystem token")
    # Preserve readability while remaining a single path segment.
    return token.replace(":", "~")


def _fs_safe_run_id(run_id: str) -> str:
    text = _require_cid(run_id, "run_id")
    if not _CID_RE.match(text) or text in {".", ".."}:
        raise RunRegistryError("run_id is not a safe path component")
    return text


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_dag_json_bytes(dict(payload))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = _temp_path(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        # Best-effort directory fsync for durability on crash.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = -1
        if dir_fd >= 0:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _temp_path(path: Path) -> tuple[int, str]:
    directory = str(path.parent)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    for _ in range(32):
        name = f".{path.name}.{uuid.uuid4().hex}.tmp"
        full = os.path.join(directory, name)
        try:
            fd = os.open(full, flags, 0o600)
        except FileExistsError:
            continue
        return fd, full
    raise RunRegistryError("unable to allocate temporary write path")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _canonical_json_bytes(payload)
    if len(data) > MAX_RECORD_BYTES:
        raise RunRegistryBoundsError("registry record exceeds byte bound")
    _atomic_write_bytes(path, data)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RunRegistryCorruptionError(
            f"registry path must not be a symlink: {path.name}",
            reason_codes=("symlink_rejected",),
        )
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RunRegistryCorruptionError(
            f"registry path unreadable: {path.name}",
            reason_codes=("unreadable",),
        ) from exc
    if len(raw) > MAX_RECORD_BYTES:
        raise RunRegistryCorruptionError(
            f"registry record exceeds byte bound: {path.name}",
            reason_codes=("record_too_large",),
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunRegistryCorruptionError(
            f"registry JSON is corrupt: {path.name}",
            reason_codes=("invalid_json",),
        ) from exc
    if not isinstance(value, dict):
        raise RunRegistryCorruptionError(
            f"registry JSON must be an object: {path.name}",
            reason_codes=("non_object_json",),
        )
    return value


def _reason_codes(values: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= MAX_REASON_CODES:
            break
    return tuple(cleaned)


@dataclass(frozen=True)
class RunRootRecord:
    """Immutable identity binding for one durable run."""

    SCHEMA: ClassVar[str] = RUN_ROOT_SCHEMA

    run_id: str
    run_namespace: str
    repository_id: str
    checkout_id: str
    target_resolution_receipt_cid: str
    invocation_cid: str
    prompt_cid: str
    objective_cid: str
    lifecycle_profile_cid: str
    created_at_ms: int
    initial_handle_cid: str
    initial_revision: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_cid(self.run_id, "run_id"))
        object.__setattr__(
            self, "run_namespace", _require_token(self.run_namespace, "run_namespace")
        )
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(
                self.repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
            ),
        )
        object.__setattr__(self, "checkout_id", str(self.checkout_id or "").strip())
        for name in (
            "target_resolution_receipt_cid",
            "invocation_cid",
            "prompt_cid",
            "initial_handle_cid",
        ):
            object.__setattr__(self, name, _require_cid(getattr(self, name), name))
        for name in ("objective_cid", "lifecycle_profile_cid"):
            raw = str(getattr(self, name) or "").strip()
            if raw:
                object.__setattr__(self, name, _require_cid(raw, name))
            else:
                object.__setattr__(self, name, "")
        created = int(self.created_at_ms)
        if created < 0:
            raise RunRegistryError("created_at_ms must be non-negative")
        object.__setattr__(self, "created_at_ms", created)
        revision = int(self.initial_revision)
        if revision < 1:
            raise RunRegistryError("initial_revision must be >= 1")
        object.__setattr__(self, "initial_revision", revision)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "run_namespace": self.run_namespace,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "target_resolution_receipt_cid": self.target_resolution_receipt_cid,
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "objective_cid": self.objective_cid,
            "lifecycle_profile_cid": self.lifecycle_profile_cid,
            "created_at_ms": self.created_at_ms,
            "initial_handle_cid": self.initial_handle_cid,
            "initial_revision": self.initial_revision,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunRootRecord":
        if not isinstance(value, Mapping):
            raise RunRegistryCorruptionError(
                "run root must be an object",
                reason_codes=("root_not_object",),
            )
        schema = value.get("schema")
        if schema != cls.SCHEMA:
            raise RunRegistryCorruptionError(
                "run root schema mismatch",
                reason_codes=("root_schema_mismatch",),
            )
        required = (
            "run_id",
            "run_namespace",
            "repository_id",
            "checkout_id",
            "target_resolution_receipt_cid",
            "invocation_cid",
            "prompt_cid",
            "objective_cid",
            "lifecycle_profile_cid",
            "created_at_ms",
            "initial_handle_cid",
            "initial_revision",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise RunRegistryCorruptionError(
                f"run root missing fields: {missing}",
                reason_codes=("root_missing_fields",),
            )
        result = cls(
            run_id=value["run_id"],
            run_namespace=value["run_namespace"],
            repository_id=value["repository_id"],
            checkout_id=value["checkout_id"],
            target_resolution_receipt_cid=value["target_resolution_receipt_cid"],
            invocation_cid=value["invocation_cid"],
            prompt_cid=value["prompt_cid"],
            objective_cid=value["objective_cid"],
            lifecycle_profile_cid=value["lifecycle_profile_cid"],
            created_at_ms=int(value["created_at_ms"]),
            initial_handle_cid=value["initial_handle_cid"],
            initial_revision=int(value["initial_revision"]),
        )
        claimed = value.get("content_id")
        if claimed is not None and claimed != result.content_id:
            raise RunRegistryCorruptionError(
                "run root content_id does not match payload",
                reason_codes=("root_identity_mismatch",),
                run_id=result.run_id,
            )
        return result

    @classmethod
    def from_handle(
        cls,
        handle: RunHandle,
        *,
        run_namespace: str,
        repository_id: str,
        checkout_id: str = "",
    ) -> "RunRootRecord":
        return cls(
            run_id=handle.run_id,
            run_namespace=run_namespace,
            repository_id=repository_id,
            checkout_id=checkout_id,
            target_resolution_receipt_cid=handle.target_resolution_receipt_cid,
            invocation_cid=handle.invocation_cid,
            prompt_cid=handle.prompt_cid,
            objective_cid=handle.objective_cid,
            lifecycle_profile_cid=handle.lifecycle_profile_cid,
            created_at_ms=handle.created_at_ms,
            initial_handle_cid=handle.content_id,
            initial_revision=handle.run_revision,
        )


@dataclass(frozen=True)
class RunHeadRecord:
    """Mutable CAS head for one run (status/cursors + revision pointer)."""

    SCHEMA: ClassVar[str] = RUN_HEAD_SCHEMA

    run_id: str
    run_revision: int
    handle_cid: str
    semantic_id: str
    state: RunState
    health: RunHealth
    event_cursor: str
    updated_at_ms: int
    previous_handle_cid: str = ""
    previous_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_cid(self.run_id, "run_id"))
        revision = int(self.run_revision)
        if revision < 1:
            raise RunRegistryError("run_revision must be >= 1")
        object.__setattr__(self, "run_revision", revision)
        object.__setattr__(
            self, "handle_cid", _require_cid(self.handle_cid, "handle_cid")
        )
        object.__setattr__(
            self, "semantic_id", _require_cid(self.semantic_id, "semantic_id")
        )
        state = self.state
        if not isinstance(state, RunState):
            state = RunState(str(state))
        health = self.health
        if not isinstance(health, RunHealth):
            health = RunHealth(str(health))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "health", health)
        object.__setattr__(self, "event_cursor", str(self.event_cursor or ""))
        updated = int(self.updated_at_ms)
        if updated < 0:
            raise RunRegistryError("updated_at_ms must be non-negative")
        object.__setattr__(self, "updated_at_ms", updated)
        prev = str(self.previous_handle_cid or "").strip()
        if prev:
            object.__setattr__(
                self, "previous_handle_cid", _require_cid(prev, "previous_handle_cid")
            )
        else:
            object.__setattr__(self, "previous_handle_cid", "")
        object.__setattr__(self, "previous_revision", int(self.previous_revision or 0))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "run_revision": self.run_revision,
            "handle_cid": self.handle_cid,
            "semantic_id": self.semantic_id,
            "state": self.state.value,
            "health": self.health.value,
            "event_cursor": self.event_cursor,
            "updated_at_ms": self.updated_at_ms,
            "previous_handle_cid": self.previous_handle_cid,
            "previous_revision": self.previous_revision,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    @property
    def integrity_cid(self) -> str:
        """Authoritative registry integrity evidence for adoption."""

        return self.content_id

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunHeadRecord":
        if not isinstance(value, Mapping):
            raise RunRegistryCorruptionError(
                "run head must be an object",
                reason_codes=("head_not_object",),
            )
        if value.get("schema") != cls.SCHEMA:
            raise RunRegistryCorruptionError(
                "run head schema mismatch",
                reason_codes=("head_schema_mismatch",),
            )
        try:
            result = cls(
                run_id=value["run_id"],
                run_revision=int(value["run_revision"]),
                handle_cid=value["handle_cid"],
                semantic_id=value["semantic_id"],
                state=value["state"],
                health=value["health"],
                event_cursor=value.get("event_cursor", ""),
                updated_at_ms=int(value["updated_at_ms"]),
                previous_handle_cid=value.get("previous_handle_cid", ""),
                previous_revision=int(value.get("previous_revision", 0) or 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRegistryCorruptionError(
                "run head fields are incomplete or invalid",
                reason_codes=("head_invalid_fields",),
            ) from exc
        claimed = value.get("content_id")
        if claimed is not None and claimed != result.content_id:
            raise RunRegistryCorruptionError(
                "run head content_id does not match payload",
                reason_codes=("head_identity_mismatch",),
                run_id=result.run_id,
            )
        return result

    @classmethod
    def from_handle(
        cls,
        handle: RunHandle,
        *,
        previous_handle_cid: str = "",
        previous_revision: int = 0,
    ) -> "RunHeadRecord":
        return cls(
            run_id=handle.run_id,
            run_revision=handle.run_revision,
            handle_cid=handle.content_id,
            semantic_id=handle.semantic_id,
            state=handle.state,
            health=handle.health,
            event_cursor=handle.event_cursor,
            updated_at_ms=handle.updated_at_ms,
            previous_handle_cid=previous_handle_cid,
            previous_revision=previous_revision,
        )


@dataclass(frozen=True)
class RegistryTransactionReceipt:
    """Exact transaction evidence for create/CAS/set_current/quarantine."""

    SCHEMA: ClassVar[str] = RUN_TX_SCHEMA

    operation: RegistryOperation
    outcome: RegistryTxOutcome
    run_id: str
    run_revision: int
    handle_cid: str
    integrity_cid: str
    previous_revision: int
    previous_handle_cid: str
    reason_codes: tuple[str, ...]
    committed_at_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RegistryOperation):
            object.__setattr__(
                self, "operation", RegistryOperation(str(self.operation))
            )
        if not isinstance(self.outcome, RegistryTxOutcome):
            object.__setattr__(self, "outcome", RegistryTxOutcome(str(self.outcome)))
        run_id = str(self.run_id or "").strip()
        if run_id:
            object.__setattr__(self, "run_id", _require_cid(run_id, "run_id"))
        else:
            object.__setattr__(self, "run_id", "")
        object.__setattr__(self, "run_revision", int(self.run_revision or 0))
        handle_cid = str(self.handle_cid or "").strip()
        if handle_cid:
            object.__setattr__(
                self, "handle_cid", _require_cid(handle_cid, "handle_cid")
            )
        else:
            object.__setattr__(self, "handle_cid", "")
        integrity = str(self.integrity_cid or "").strip()
        if integrity:
            object.__setattr__(
                self, "integrity_cid", _require_cid(integrity, "integrity_cid")
            )
        else:
            object.__setattr__(self, "integrity_cid", "")
        object.__setattr__(
            self, "previous_revision", int(self.previous_revision or 0)
        )
        prev = str(self.previous_handle_cid or "").strip()
        if prev:
            object.__setattr__(
                self,
                "previous_handle_cid",
                _require_cid(prev, "previous_handle_cid"),
            )
        else:
            object.__setattr__(self, "previous_handle_cid", "")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "committed_at_ms", int(self.committed_at_ms))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "operation": self.operation.value,
            "outcome": self.outcome.value,
            "run_id": self.run_id,
            "run_revision": self.run_revision,
            "handle_cid": self.handle_cid,
            "integrity_cid": self.integrity_cid,
            "previous_revision": self.previous_revision,
            "previous_handle_cid": self.previous_handle_cid,
            "reason_codes": list(self.reason_codes),
            "committed_at_ms": self.committed_at_ms,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class NamespaceCurrentRecord:
    """CAS pointer selecting the current run for one namespace binding."""

    SCHEMA: ClassVar[str] = NAMESPACE_CURRENT_SCHEMA

    run_namespace: str
    repository_id: str
    checkout_id: str
    selected_run_id: str
    integrity_cid: str
    pointer_revision: int
    updated_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_namespace", _require_token(self.run_namespace, "run_namespace")
        )
        object.__setattr__(
            self,
            "repository_id",
            _require_nonempty(
                self.repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
            ),
        )
        object.__setattr__(self, "checkout_id", str(self.checkout_id or "").strip())
        selected = str(self.selected_run_id or "").strip()
        if selected:
            object.__setattr__(
                self, "selected_run_id", _require_cid(selected, "selected_run_id")
            )
        else:
            object.__setattr__(self, "selected_run_id", "")
        integrity = str(self.integrity_cid or "").strip()
        if integrity:
            object.__setattr__(
                self, "integrity_cid", _require_cid(integrity, "integrity_cid")
            )
        else:
            object.__setattr__(self, "integrity_cid", "")
        revision = int(self.pointer_revision)
        if revision < 1:
            raise RunRegistryError("pointer_revision must be >= 1")
        object.__setattr__(self, "pointer_revision", revision)
        object.__setattr__(self, "updated_at_ms", int(self.updated_at_ms))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_namespace": self.run_namespace,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "selected_run_id": self.selected_run_id,
            "integrity_cid": self.integrity_cid,
            "pointer_revision": self.pointer_revision,
            "updated_at_ms": self.updated_at_ms,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NamespaceCurrentRecord":
        if not isinstance(value, Mapping):
            raise RunRegistryCorruptionError(
                "namespace current must be an object",
                reason_codes=("current_not_object",),
            )
        if value.get("schema") != cls.SCHEMA:
            raise RunRegistryCorruptionError(
                "namespace current schema mismatch",
                reason_codes=("current_schema_mismatch",),
            )
        try:
            result = cls(
                run_namespace=value["run_namespace"],
                repository_id=value["repository_id"],
                checkout_id=value.get("checkout_id", ""),
                selected_run_id=value.get("selected_run_id", ""),
                integrity_cid=value.get("integrity_cid", ""),
                pointer_revision=int(value["pointer_revision"]),
                updated_at_ms=int(value["updated_at_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRegistryCorruptionError(
                "namespace current fields are incomplete or invalid",
                reason_codes=("current_invalid_fields",),
            ) from exc
        claimed = value.get("content_id")
        if claimed is not None and claimed != result.content_id:
            raise RunRegistryCorruptionError(
                "namespace current content_id does not match payload",
                reason_codes=("current_identity_mismatch",),
            )
        return result


@dataclass(frozen=True)
class RunSelectionResult:
    """Deterministic current-run selection over integrity-checked records."""

    action: RunAdoptionAction
    selected_run_id: str
    selected_handle: RunHandle | None
    integrity_cid: str
    candidates: tuple[RunCandidateEvidence, ...]
    classified: tuple[ClassifiedRunCandidate, ...]
    resolution: RunCandidateResolution
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{RUN_REGISTRY_SCHEMA}/selection@1",
            "action": self.action.value,
            "selected_run_id": self.selected_run_id,
            "selected_handle_cid": (
                self.selected_handle.content_id if self.selected_handle else ""
            ),
            "integrity_cid": self.integrity_cid,
            "candidates": [item.to_dict() for item in self.candidates],
            "classified": [item.to_dict() for item in self.classified],
            "resolution": self.resolution.to_dict(),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RepairReport:
    """Result of a bounded partial-state repair pass."""

    repaired_run_ids: tuple[str, ...]
    quarantined_run_ids: tuple[str, ...]
    receipts: tuple[RegistryTransactionReceipt, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired_run_ids": list(self.repaired_run_ids),
            "quarantined_run_ids": list(self.quarantined_run_ids),
            "receipts": [item.to_dict() for item in self.receipts],
            "reason_codes": list(self.reason_codes),
        }


class RunRegistry:
    """Durable run registry with reconstruction, CAS, and exact selection."""

    def __init__(
        self,
        registry_root: str | Path,
        *,
        clock_ms: Callable[[], int] | None = None,
        max_list: int = DEFAULT_MAX_LIST,
    ) -> None:
        root = Path(registry_root).expanduser().resolve()
        self.registry_root = root
        self.clock_ms = clock_ms or _now_ms
        if not 1 <= int(max_list) <= HARD_MAX_LIST:
            raise RunRegistryBoundsError(
                f"max_list must be in 1..{HARD_MAX_LIST}"
            )
        self.max_list = int(max_list)
        self._thread_lock = threading.RLock()
        self._closed = False
        self.registry_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.registry_root / QUARANTINE_DIR).mkdir(
            parents=True, exist_ok=True, mode=0o700
        )
        (self.registry_root / NAMESPACES_DIR).mkdir(
            parents=True, exist_ok=True, mode=0o700
        )
        self.lock_path = self.registry_root / LOCK_NAME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "RunRegistry":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def requirement_id(self) -> str:
        return RUN_REGISTRY_REQUIREMENT_ID

    def restart_behavior(self) -> dict[str, Any]:
        """Explicit restart reconstruction semantics (evidence surface)."""

        return {
            "requirement_id": RUN_REGISTRY_REQUIREMENT_ID,
            "handles_survive_restart": True,
            "reconstruction": "root_plus_head_plus_handle_snapshot",
            "cas_survives_restart": True,
            "corruption_policy": "quarantine_fail_closed",
            "selection_policy": "unique_compatible_or_explicit_ambiguity",
            "non_authoritative": (
                "directory_name",
                "pid_file",
                "timestamp_heuristic",
                "prompt_text",
            ),
        }

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _namespace_dir(self, run_namespace: str) -> Path:
        token = _require_token(run_namespace, "run_namespace")
        return self.registry_root / NAMESPACES_DIR / _fs_safe_token(token)

    def _run_dir(self, run_namespace: str, run_id: str) -> Path:
        return self._namespace_dir(run_namespace) / RUNS_DIR / _fs_safe_run_id(run_id)

    def _root_path(self, run_namespace: str, run_id: str) -> Path:
        return self._run_dir(run_namespace, run_id) / ROOT_NAME

    def _head_path(self, run_namespace: str, run_id: str) -> Path:
        return self._run_dir(run_namespace, run_id) / HEAD_NAME

    def _handles_dir(self, run_namespace: str, run_id: str) -> Path:
        return self._run_dir(run_namespace, run_id) / HANDLES_DIR

    def _handle_path(
        self, run_namespace: str, run_id: str, handle_cid: str
    ) -> Path:
        safe = _fs_safe_run_id(handle_cid)
        return self._handles_dir(run_namespace, run_id) / f"{safe}.json"

    def _current_path(self, run_namespace: str) -> Path:
        return self._namespace_dir(run_namespace) / CURRENT_NAME

    def _index_path(self) -> Path:
        return self.registry_root / "run_index.json"

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if self._closed:
            raise RunRegistryError("run registry is closed")
        self._thread_lock.acquire()
        handle = None
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = self.lock_path.open("a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if handle is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
            finally:
                self._thread_lock.release()

    # ------------------------------------------------------------------
    # Index (run_id -> namespace)
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, str]:
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            payload = _read_json(path)
        except RunRegistryCorruptionError:
            # Index is advisory for lookup; corruption is repaired on write.
            return {}
        mapping = payload.get("runs")
        if not isinstance(mapping, Mapping):
            return {}
        result: dict[str, str] = {}
        for run_id, namespace in mapping.items():
            if isinstance(run_id, str) and isinstance(namespace, str):
                result[run_id] = namespace
        return result

    def _save_index(self, mapping: Mapping[str, str]) -> None:
        ordered = {key: mapping[key] for key in sorted(mapping)}
        payload = {
            "schema": f"{RUN_REGISTRY_SCHEMA}/index@1",
            "runs": ordered,
        }
        payload["content_id"] = cid_for_dag_json(
            {"schema": payload["schema"], "runs": ordered}
        )
        _atomic_write_json(self._index_path(), payload)

    def _index_put(self, run_id: str, run_namespace: str) -> None:
        mapping = self._load_index()
        mapping[run_id] = run_namespace
        self._save_index(mapping)

    def _index_remove(self, run_id: str) -> None:
        mapping = self._load_index()
        if run_id in mapping:
            del mapping[run_id]
            self._save_index(mapping)

    def _resolve_namespace(self, run_id: str) -> str:
        mapping = self._load_index()
        namespace = mapping.get(run_id)
        if namespace:
            return namespace
        # Fall back to a bounded directory scan for restart without index.
        for found_id, found_ns in self._scan_all_run_ids():
            if found_id == run_id:
                return found_ns
        raise RunNotFoundError(f"run not found: {run_id}")

    def _scan_all_run_ids(self) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        base = self.registry_root / NAMESPACES_DIR
        if not base.is_dir():
            return results
        for ns_dir in sorted(base.iterdir()):
            if not ns_dir.is_dir() or ns_dir.name.startswith("."):
                continue
            runs = ns_dir / RUNS_DIR
            if not runs.is_dir():
                continue
            # Recover original namespace from root when possible.
            for run_dir in sorted(runs.iterdir()):
                if not run_dir.is_dir():
                    continue
                root_path = run_dir / ROOT_NAME
                namespace = ""
                if root_path.is_file() and not root_path.is_symlink():
                    try:
                        root = RunRootRecord.from_dict(_read_json(root_path))
                        namespace = root.run_namespace
                        run_id = root.run_id
                    except (RunRegistryError, OSError, ValueError, TypeError):
                        run_id = run_dir.name
                        namespace = ns_dir.name.replace("~", ":")
                else:
                    run_id = run_dir.name
                    namespace = ns_dir.name.replace("~", ":")
                results.append((run_id, namespace))
        return results

    # ------------------------------------------------------------------
    # Handle snapshot IO
    # ------------------------------------------------------------------

    def _write_handle_snapshot(
        self, run_namespace: str, handle: RunHandle
    ) -> Path:
        path = self._handle_path(run_namespace, handle.run_id, handle.content_id)
        if path.exists():
            # Content-addressed: existing bytes must match exactly.
            existing = _read_json(path)
            loaded = RunHandle.from_dict(existing)
            if loaded.content_id != handle.content_id:
                raise RunRegistryCorruptionError(
                    "handle snapshot CID collision with different content",
                    run_id=handle.run_id,
                    reason_codes=("handle_cid_collision",),
                )
            return path
        payload = handle.to_dict()
        _atomic_write_json(path, payload)
        return path

    def _load_handle_snapshot(
        self, run_namespace: str, run_id: str, handle_cid: str
    ) -> RunHandle:
        path = self._handle_path(run_namespace, run_id, handle_cid)
        if not path.exists():
            raise RunRegistryCorruptionError(
                "handle snapshot missing for head pointer",
                run_id=run_id,
                reason_codes=("handle_snapshot_missing",),
            )
        payload = _read_json(path)
        try:
            handle = RunHandle.from_dict(payload)
        except EntrypointContractError as exc:
            raise RunRegistryCorruptionError(
                "handle snapshot is not a valid RunHandle",
                run_id=run_id,
                reason_codes=("handle_invalid",),
            ) from exc
        if handle.content_id != handle_cid:
            raise RunRegistryCorruptionError(
                "handle snapshot content_id does not match path/head",
                run_id=run_id,
                reason_codes=("handle_cid_mismatch",),
            )
        if handle.run_id != run_id:
            raise RunRegistryCorruptionError(
                "handle snapshot run_id does not match directory",
                run_id=run_id,
                reason_codes=("handle_run_id_mismatch",),
            )
        return handle

    def _load_root(self, run_namespace: str, run_id: str) -> RunRootRecord:
        path = self._root_path(run_namespace, run_id)
        if not path.exists():
            raise RunNotFoundError(f"run root not found: {run_id}")
        return RunRootRecord.from_dict(_read_json(path))

    def _load_head(self, run_namespace: str, run_id: str) -> RunHeadRecord:
        path = self._head_path(run_namespace, run_id)
        if not path.exists():
            raise RunRegistryCorruptionError(
                "run head missing",
                run_id=run_id,
                reason_codes=("head_missing",),
            )
        return RunHeadRecord.from_dict(_read_json(path))

    def _verify_handle_against_root(
        self, root: RunRootRecord, handle: RunHandle
    ) -> None:
        root_values = {
            "run_id": root.run_id,
            "target_resolution_receipt_cid": root.target_resolution_receipt_cid,
            "invocation_cid": root.invocation_cid,
            "prompt_cid": root.prompt_cid,
            "created_at_ms": root.created_at_ms,
        }
        for name in _IMMUTABLE_HANDLE_FIELDS:
            if getattr(handle, name) != root_values[name]:
                raise RunIncompatibleError(
                    f"handle field {name} diverges from immutable run root"
                )

    def _reconstruct_unlocked(
        self, run_namespace: str, run_id: str
    ) -> tuple[RunRootRecord, RunHeadRecord, RunHandle]:
        root = self._load_root(run_namespace, run_id)
        head = self._load_head(run_namespace, run_id)
        if head.run_id != root.run_id:
            raise RunRegistryCorruptionError(
                "head run_id does not match root",
                run_id=run_id,
                reason_codes=("head_root_run_id_mismatch",),
            )
        handle = self._load_handle_snapshot(
            run_namespace, run_id, head.handle_cid
        )
        if handle.run_revision != head.run_revision:
            raise RunRegistryCorruptionError(
                "handle revision does not match head",
                run_id=run_id,
                reason_codes=("head_handle_revision_mismatch",),
            )
        if handle.semantic_id != head.semantic_id:
            raise RunRegistryCorruptionError(
                "handle semantic_id does not match head",
                run_id=run_id,
                reason_codes=("head_handle_semantic_mismatch",),
            )
        if handle.state is not head.state or handle.health is not head.health:
            raise RunRegistryCorruptionError(
                "handle state/health does not match head",
                run_id=run_id,
                reason_codes=("head_handle_status_mismatch",),
            )
        try:
            self._verify_handle_against_root(root, handle)
        except RunIncompatibleError as exc:
            raise RunRegistryCorruptionError(
                str(exc),
                run_id=run_id,
                reason_codes=("immutable_field_divergence",),
            ) from exc
        return root, head, handle

    # ------------------------------------------------------------------
    # Quarantine
    # ------------------------------------------------------------------

    def _quarantine(
        self,
        *,
        run_id: str,
        run_namespace: str,
        reason_codes: Sequence[str],
        detail: Mapping[str, Any] | None = None,
    ) -> tuple[Path, RegistryTransactionReceipt]:
        stamp = self.clock_ms()
        token = uuid.uuid4().hex
        safe_run = run_id if _CID_RE.match(run_id or "") else "unknown"
        destination = (
            self.registry_root
            / QUARANTINE_DIR
            / f"{stamp}-{safe_run}-{token}.json"
        )
        run_dir = None
        try:
            if run_namespace and run_id:
                run_dir = self._run_dir(run_namespace, run_id)
        except RunRegistryError:
            run_dir = None

        snapshot: dict[str, Any] = {
            "schema": f"{RUN_REGISTRY_SCHEMA}/quarantine@1",
            "run_id": run_id,
            "run_namespace": run_namespace,
            "reason_codes": list(_reason_codes(reason_codes)),
            "quarantined_at_ms": stamp,
            "detail": dict(detail or {}),
        }
        if run_dir is not None and run_dir.exists():
            for name in (ROOT_NAME, HEAD_NAME):
                path = run_dir / name
                if path.is_file() and not path.is_symlink():
                    try:
                        snapshot[name] = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                        snapshot[name] = {"raw_unreadable": True}
            handles = run_dir / HANDLES_DIR
            if handles.is_dir():
                listed: list[str] = []
                for child in sorted(handles.iterdir()):
                    if child.is_file():
                        listed.append(child.name)
                snapshot["handle_files"] = listed[:HARD_MAX_LIST]
            # Relocate the run directory into quarantine for fail-closed lookup.
            relocated = (
                self.registry_root
                / QUARANTINE_DIR
                / f"run-{stamp}-{safe_run}-{token}"
            )
            try:
                os.replace(str(run_dir), str(relocated))
                snapshot["relocated_run_dir"] = str(relocated)
            except OSError:
                snapshot["relocated_run_dir"] = ""

        snapshot["content_id"] = cid_for_dag_json(
            {
                "schema": snapshot["schema"],
                "run_id": snapshot["run_id"],
                "run_namespace": snapshot["run_namespace"],
                "reason_codes": snapshot["reason_codes"],
                "quarantined_at_ms": snapshot["quarantined_at_ms"],
            }
        )
        _atomic_write_json(destination, snapshot)
        if run_id:
            self._index_remove(run_id)
        receipt = RegistryTransactionReceipt(
            operation=RegistryOperation.QUARANTINE,
            outcome=RegistryTxOutcome.QUARANTINED,
            run_id=run_id if run_id and _CID_RE.match(run_id) else "",
            run_revision=0,
            handle_cid="",
            integrity_cid=snapshot["content_id"],
            previous_revision=0,
            previous_handle_cid="",
            reason_codes=_reason_codes(reason_codes),
            committed_at_ms=stamp,
        )
        return destination, receipt

    def _fail_corrupt(
        self,
        *,
        run_id: str,
        run_namespace: str,
        error: RunRegistryCorruptionError,
    ) -> RunRegistryCorruptionError:
        path, _receipt = self._quarantine(
            run_id=run_id,
            run_namespace=run_namespace,
            reason_codes=error.reason_codes or ("corruption",),
            detail={"message": str(error)},
        )
        return RunRegistryCorruptionError(
            str(error),
            quarantine_path=str(path),
            run_id=run_id,
            reason_codes=error.reason_codes or ("corruption",),
        )

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def create(
        self,
        handle: RunHandle,
        *,
        run_namespace: str,
        repository_id: str,
        checkout_id: str = "",
    ) -> RegistryTransactionReceipt:
        """Persist a new immutable root and initial CAS head."""

        if not isinstance(handle, RunHandle):
            raise RunRegistryError("handle must be a RunHandle")
        namespace = _require_token(run_namespace, "run_namespace")
        repo = _require_nonempty(
            repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
        )
        checkout = str(checkout_id or "").strip()

        with self._exclusive():
            run_dir = self._run_dir(namespace, handle.run_id)
            root_path = self._root_path(namespace, handle.run_id)
            if root_path.exists() or run_dir.exists():
                raise RunExistsError(
                    f"run already registered: {handle.run_id}"
                )
            # Ensure index uniqueness across namespaces.
            try:
                existing_ns = self._resolve_namespace(handle.run_id)
            except RunNotFoundError:
                existing_ns = ""
            if existing_ns:
                raise RunExistsError(
                    f"run already registered under namespace {existing_ns}"
                )

            root = RunRootRecord.from_handle(
                handle,
                run_namespace=namespace,
                repository_id=repo,
                checkout_id=checkout,
            )
            head = RunHeadRecord.from_handle(handle)
            self._write_handle_snapshot(namespace, handle)
            _atomic_write_json(root_path, root.to_dict())
            _atomic_write_json(self._head_path(namespace, handle.run_id), head.to_dict())
            self._index_put(handle.run_id, namespace)
            return RegistryTransactionReceipt(
                operation=RegistryOperation.CREATE,
                outcome=RegistryTxOutcome.COMMITTED,
                run_id=handle.run_id,
                run_revision=handle.run_revision,
                handle_cid=handle.content_id,
                integrity_cid=head.integrity_cid,
                previous_revision=0,
                previous_handle_cid="",
                reason_codes=("created",),
                committed_at_ms=self.clock_ms(),
            )

    def cas_update(
        self,
        handle: RunHandle,
        *,
        expected_revision: int | None = None,
        expected_handle_cid: str | None = "",
        expected_semantic_id: str | None = "",
    ) -> RegistryTransactionReceipt:
        """Compare-and-swap the run head to a new handle revision.

        Exactly one of the expected fields may be omitted, but every provided
        expected value must match the current head.  The new handle must keep
        immutable root fields and advance ``run_revision`` by exactly one.
        """

        if not isinstance(handle, RunHandle):
            raise RunRegistryError("handle must be a RunHandle")

        with self._exclusive():
            try:
                namespace = self._resolve_namespace(handle.run_id)
            except RunNotFoundError:
                raise
            try:
                root, head, _current = self._reconstruct_unlocked(
                    namespace, handle.run_id
                )
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=handle.run_id,
                    run_namespace=namespace,
                    error=exc,
                ) from exc

            expected_rev = (
                int(expected_revision)
                if expected_revision is not None
                else head.run_revision
            )
            expected_hc = str(expected_handle_cid or "").strip() or head.handle_cid
            expected_sid = (
                str(expected_semantic_id or "").strip() or head.semantic_id
            )

            conflict_reasons: list[str] = []
            if head.run_revision != expected_rev:
                conflict_reasons.append("revision_mismatch")
            if head.handle_cid != expected_hc:
                conflict_reasons.append("handle_cid_mismatch")
            if head.semantic_id != expected_sid:
                conflict_reasons.append("semantic_id_mismatch")

            if conflict_reasons:
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.CAS_UPDATE,
                    outcome=RegistryTxOutcome.CONFLICT,
                    run_id=handle.run_id,
                    run_revision=head.run_revision,
                    handle_cid=head.handle_cid,
                    integrity_cid=head.integrity_cid,
                    previous_revision=head.run_revision,
                    previous_handle_cid=head.handle_cid,
                    reason_codes=tuple(conflict_reasons),
                    committed_at_ms=self.clock_ms(),
                )
                raise RunCasConflictError(
                    "CAS conflict: expected revision does not match head",
                    receipt=receipt,
                )

            if handle.run_revision != head.run_revision + 1:
                raise RunRegistryError(
                    "CAS handle must advance run_revision by exactly one"
                )
            try:
                self._verify_handle_against_root(root, handle)
            except RunIncompatibleError:
                raise
            if handle.updated_at_ms < head.updated_at_ms:
                raise RunRegistryError(
                    "CAS handle updated_at_ms cannot move backwards"
                )

            # Idempotent success: same content already at next revision.
            if (
                handle.content_id == head.handle_cid
                and handle.run_revision == head.run_revision
            ):
                return RegistryTransactionReceipt(
                    operation=RegistryOperation.CAS_UPDATE,
                    outcome=RegistryTxOutcome.NOOP,
                    run_id=handle.run_id,
                    run_revision=head.run_revision,
                    handle_cid=head.handle_cid,
                    integrity_cid=head.integrity_cid,
                    previous_revision=head.run_revision,
                    previous_handle_cid=head.handle_cid,
                    reason_codes=("idempotent_same_head",),
                    committed_at_ms=self.clock_ms(),
                )

            new_head = RunHeadRecord.from_handle(
                handle,
                previous_handle_cid=head.handle_cid,
                previous_revision=head.run_revision,
            )
            self._write_handle_snapshot(namespace, handle)
            _atomic_write_json(
                self._head_path(namespace, handle.run_id), new_head.to_dict()
            )
            return RegistryTransactionReceipt(
                operation=RegistryOperation.CAS_UPDATE,
                outcome=RegistryTxOutcome.COMMITTED,
                run_id=handle.run_id,
                run_revision=handle.run_revision,
                handle_cid=handle.content_id,
                integrity_cid=new_head.integrity_cid,
                previous_revision=head.run_revision,
                previous_handle_cid=head.handle_cid,
                reason_codes=("cas_committed",),
                committed_at_ms=self.clock_ms(),
            )

    def set_current(
        self,
        *,
        run_namespace: str,
        repository_id: str,
        run_id: str,
        checkout_id: str = "",
        expected_pointer_revision: int | None = None,
    ) -> RegistryTransactionReceipt:
        """CAS-update the namespace current-run pointer."""

        namespace = _require_token(run_namespace, "run_namespace")
        repo = _require_nonempty(
            repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
        )
        checkout = str(checkout_id or "").strip()
        target_run = _require_cid(run_id, "run_id")

        with self._exclusive():
            try:
                root, head, handle = self._reconstruct_unlocked(
                    namespace, target_run
                )
            except RunNotFoundError:
                raise
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=target_run,
                    run_namespace=namespace,
                    error=exc,
                ) from exc

            if root.run_namespace != namespace:
                raise RunIncompatibleError("run namespace does not match pointer")
            if root.repository_id != repo:
                raise RunIncompatibleError("run repository_id does not match pointer")
            if checkout and root.checkout_id and root.checkout_id != checkout:
                raise RunIncompatibleError("run checkout_id does not match pointer")

            current_path = self._current_path(namespace)
            previous_revision = 0
            previous_handle = ""
            if current_path.exists():
                try:
                    current = NamespaceCurrentRecord.from_dict(
                        _read_json(current_path)
                    )
                except RunRegistryCorruptionError as exc:
                    # Corrupt current pointer is quarantined and rewritten.
                    self._quarantine(
                        run_id=target_run,
                        run_namespace=namespace,
                        reason_codes=exc.reason_codes or ("current_corrupt",),
                        detail={"path": str(current_path)},
                    )
                    current = None
                else:
                    previous_revision = current.pointer_revision
                    previous_handle = current.selected_run_id
                    if (
                        expected_pointer_revision is not None
                        and current.pointer_revision != int(expected_pointer_revision)
                    ):
                        receipt = RegistryTransactionReceipt(
                            operation=RegistryOperation.SET_CURRENT,
                            outcome=RegistryTxOutcome.CONFLICT,
                            run_id=target_run,
                            run_revision=head.run_revision,
                            handle_cid=head.handle_cid,
                            integrity_cid=head.integrity_cid,
                            previous_revision=current.pointer_revision,
                            previous_handle_cid="",
                            reason_codes=("pointer_revision_mismatch",),
                            committed_at_ms=self.clock_ms(),
                        )
                        raise RunCasConflictError(
                            "CAS conflict on namespace current pointer",
                            receipt=receipt,
                        )
            elif expected_pointer_revision not in (None, 0):
                receipt = RegistryTransactionReceipt(
                    operation=RegistryOperation.SET_CURRENT,
                    outcome=RegistryTxOutcome.CONFLICT,
                    run_id=target_run,
                    run_revision=head.run_revision,
                    handle_cid=head.handle_cid,
                    integrity_cid=head.integrity_cid,
                    previous_revision=0,
                    previous_handle_cid="",
                    reason_codes=("pointer_absent_expected_revision",),
                    committed_at_ms=self.clock_ms(),
                )
                raise RunCasConflictError(
                    "CAS conflict: current pointer absent",
                    receipt=receipt,
                )

            record = NamespaceCurrentRecord(
                run_namespace=namespace,
                repository_id=repo,
                checkout_id=checkout or root.checkout_id,
                selected_run_id=handle.run_id,
                integrity_cid=head.integrity_cid,
                pointer_revision=previous_revision + 1,
                updated_at_ms=self.clock_ms(),
            )
            _atomic_write_json(current_path, record.to_dict())
            return RegistryTransactionReceipt(
                operation=RegistryOperation.SET_CURRENT,
                outcome=RegistryTxOutcome.COMMITTED,
                run_id=handle.run_id,
                run_revision=head.run_revision,
                handle_cid=head.handle_cid,
                integrity_cid=head.integrity_cid,
                previous_revision=previous_revision,
                previous_handle_cid=previous_handle if previous_handle else "",
                reason_codes=("current_set",),
                committed_at_ms=record.updated_at_ms,
            )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunHandle:
        """Exact lookup of the current handle for ``run_id``."""

        return self.reconstruct(run_id)

    def reconstruct(self, run_id: str) -> RunHandle:
        """Restart-safe reconstruction of a complete :class:`RunHandle`."""

        target = _require_cid(run_id, "run_id")
        with self._exclusive():
            try:
                namespace = self._resolve_namespace(target)
            except RunNotFoundError:
                raise
            try:
                _root, _head, handle = self._reconstruct_unlocked(namespace, target)
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=target,
                    run_namespace=namespace,
                    error=exc,
                ) from exc
            return handle

    def get_head(self, run_id: str) -> RunHeadRecord:
        target = _require_cid(run_id, "run_id")
        with self._exclusive():
            namespace = self._resolve_namespace(target)
            try:
                _root, head, _handle = self._reconstruct_unlocked(namespace, target)
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=target,
                    run_namespace=namespace,
                    error=exc,
                ) from exc
            return head

    def get_root(self, run_id: str) -> RunRootRecord:
        target = _require_cid(run_id, "run_id")
        with self._exclusive():
            namespace = self._resolve_namespace(target)
            try:
                return self._load_root(namespace, target)
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=target,
                    run_namespace=namespace,
                    error=exc,
                ) from exc

    def integrity_cid(self, run_id: str) -> str:
        return self.get_head(run_id).integrity_cid

    def exists(self, run_id: str) -> bool:
        try:
            target = _require_cid(run_id, "run_id")
        except RunRegistryError:
            return False
        with self._exclusive():
            try:
                self._resolve_namespace(target)
            except RunNotFoundError:
                return False
            return True

    def list_runs(
        self,
        *,
        run_namespace: str | None = None,
        repository_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[RunHandle, ...]:
        """Bounded listing of reconstructible handles (deterministic order)."""

        bound = self.max_list if limit is None else int(limit)
        if not 1 <= bound <= HARD_MAX_LIST:
            raise RunRegistryBoundsError(
                f"limit must be in 1..{HARD_MAX_LIST}"
            )
        ns_filter = (
            _require_token(run_namespace, "run_namespace")
            if run_namespace is not None
            else None
        )
        repo_filter = (
            _require_nonempty(
                repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
            )
            if repository_id is not None
            else None
        )

        handles: list[RunHandle] = []
        with self._exclusive():
            for run_id, namespace in sorted(self._scan_all_run_ids()):
                if ns_filter is not None and namespace != ns_filter:
                    continue
                try:
                    root, _head, handle = self._reconstruct_unlocked(
                        namespace, run_id
                    )
                except (RunRegistryError, OSError, ValueError, TypeError):
                    # Skip unreadable entries; callers may run repair().
                    continue
                if repo_filter is not None and root.repository_id != repo_filter:
                    continue
                handles.append(handle)
                if len(handles) >= bound:
                    break
        handles.sort(key=lambda item: item.run_id)
        return tuple(handles)

    def list_candidates(
        self,
        *,
        run_namespace: str,
        repository_id: str,
        checkout_id: str = "",
        limit: int | None = None,
    ) -> tuple[RunCandidateEvidence, ...]:
        """Produce integrity-checked candidates for adoption resolution."""

        bound = self.max_list if limit is None else int(limit)
        if not 1 <= bound <= HARD_MAX_LIST:
            raise RunRegistryBoundsError(
                f"limit must be in 1..{HARD_MAX_LIST}"
            )
        namespace = _require_token(run_namespace, "run_namespace")
        # Validate repository_id early; selection classifies compatibility.
        _require_nonempty(
            repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
        )
        _ = checkout_id  # reserved for future checkout-scoped listing bounds
        candidates: list[RunCandidateEvidence] = []
        with self._exclusive():
            for run_id, found_ns in sorted(self._scan_all_run_ids()):
                if found_ns != namespace:
                    continue
                try:
                    root, head, handle = self._reconstruct_unlocked(
                        found_ns, run_id
                    )
                except RunRegistryCorruptionError as exc:
                    # Corrupt entries are quarantined and never offered as
                    # adoption candidates (fail closed without aborting list).
                    self._quarantine(
                        run_id=run_id,
                        run_namespace=found_ns,
                        reason_codes=exc.reason_codes or ("corruption",),
                        detail={"message": str(exc)},
                    )
                    continue
                except (RunRegistryError, OSError, ValueError, TypeError):
                    continue
                candidates.append(
                    RunCandidateEvidence(
                        run_id=handle.run_id,
                        run_namespace=root.run_namespace,
                        repository_id=root.repository_id,
                        checkout_id=root.checkout_id,
                        state=handle.state,
                        health=handle.health,
                        registry_integrity_cid=head.integrity_cid,
                        objective_cid=handle.objective_cid,
                        profile_cid=handle.lifecycle_profile_cid,
                        state_revision_cid=handle.state_revision_cid,
                        observed_from_directory_name=False,
                        observed_from_pid_file=False,
                        stale_marker=False,
                    )
                )
                if len(candidates) >= bound:
                    break
        candidates.sort(key=lambda item: item.run_id)
        return tuple(candidates)

    def select_current(
        self,
        *,
        run_namespace: str,
        repository_id: str,
        checkout_id: str = "",
        isolation: WorktreeIsolationMode = WorktreeIsolationMode.SHARED_REPOSITORY,
        expected_objective_cid: str = "",
        expected_profile_cid: str = "",
        explicit_run_id: str = "",
    ) -> RunSelectionResult:
        """Deterministic unique-compatible selection (or explicit ambiguity)."""

        namespace = _require_token(run_namespace, "run_namespace")
        repo = _require_nonempty(
            repository_id, "repository_id", maximum=MAX_REPOSITORY_ID_BYTES
        )
        checkout = str(checkout_id or "").strip()
        candidates = self.list_candidates(
            run_namespace=namespace,
            repository_id=repo,
            checkout_id=checkout,
        )
        request = RunCandidateResolutionRequest(
            repository_id=repo,
            run_namespace=namespace,
            checkout_id=checkout,
            isolation=isolation,
            candidates=candidates,
            explicit_run_id=str(explicit_run_id or "").strip(),
            expected_objective_cid=str(expected_objective_cid or "").strip(),
            expected_profile_cid=str(expected_profile_cid or "").strip(),
        )
        resolution = RunCandidateResolver().resolve(request)
        selected_handle: RunHandle | None = None
        integrity = ""
        if resolution.selected_run_id:
            selected_handle = self.reconstruct(resolution.selected_run_id)
            integrity = self.integrity_cid(resolution.selected_run_id)
        return RunSelectionResult(
            action=resolution.action,
            selected_run_id=resolution.selected_run_id,
            selected_handle=selected_handle,
            integrity_cid=integrity,
            candidates=candidates,
            classified=resolution.classified,
            resolution=resolution,
            reason_codes=resolution.reason_codes,
        )

    def get_current(
        self,
        *,
        run_namespace: str,
        repository_id: str | None = None,
    ) -> RunHandle | None:
        """Return the namespace current pointer handle when set and valid."""

        namespace = _require_token(run_namespace, "run_namespace")
        with self._exclusive():
            path = self._current_path(namespace)
            if not path.exists():
                return None
            try:
                current = NamespaceCurrentRecord.from_dict(_read_json(path))
            except RunRegistryCorruptionError as exc:
                self._quarantine(
                    run_id="",
                    run_namespace=namespace,
                    reason_codes=exc.reason_codes or ("current_corrupt",),
                    detail={"path": str(path)},
                )
                try:
                    path.unlink()
                except OSError:
                    pass
                raise RunRegistryCorruptionError(
                    "namespace current pointer is corrupt and was quarantined",
                    quarantine_path="",
                    reason_codes=exc.reason_codes or ("current_corrupt",),
                ) from exc
            if repository_id is not None and current.repository_id != repository_id:
                raise RunIncompatibleError(
                    "current pointer repository_id does not match request"
                )
            if not current.selected_run_id:
                return None
            try:
                _root, head, handle = self._reconstruct_unlocked(
                    namespace, current.selected_run_id
                )
            except RunRegistryCorruptionError as exc:
                raise self._fail_corrupt(
                    run_id=current.selected_run_id,
                    run_namespace=namespace,
                    error=exc,
                ) from exc
            if head.integrity_cid != current.integrity_cid:
                # Pointer is stale relative to head; fail closed rather than
                # silently adopt a drifted integrity binding.
                raise RunRegistryCorruptionError(
                    "current pointer integrity does not match run head",
                    run_id=current.selected_run_id,
                    reason_codes=("current_integrity_drift",),
                )
            return handle

    def repair(self) -> RepairReport:
        """Bounded repair of partial registry state; quarantine the rest."""

        repaired: list[str] = []
        quarantined: list[str] = []
        receipts: list[RegistryTransactionReceipt] = []
        reasons: list[str] = []

        with self._exclusive():
            for run_id, namespace in sorted(self._scan_all_run_ids()):
                root_path = self._root_path(namespace, run_id)
                head_path = self._head_path(namespace, run_id)
                handles_dir = self._handles_dir(namespace, run_id)
                try:
                    if root_path.exists() and head_path.exists():
                        self._reconstruct_unlocked(namespace, run_id)
                        self._index_put(run_id, namespace)
                        continue
                    if root_path.exists() and not head_path.exists() and handles_dir.is_dir():
                        root = RunRootRecord.from_dict(_read_json(root_path))
                        snapshots: list[RunHandle] = []
                        for child in sorted(handles_dir.iterdir()):
                            if not child.is_file() or child.is_symlink():
                                continue
                            try:
                                snapshots.append(
                                    RunHandle.from_dict(_read_json(child))
                                )
                            except (EntrypointContractError, RunRegistryError):
                                continue
                        if len(snapshots) == 1:
                            handle = snapshots[0]
                            if handle.run_id != root.run_id:
                                raise RunRegistryCorruptionError(
                                    "orphan handle run_id mismatch",
                                    run_id=run_id,
                                    reason_codes=("repair_handle_mismatch",),
                                )
                            self._verify_handle_against_root(root, handle)
                            head = RunHeadRecord.from_handle(handle)
                            _atomic_write_json(head_path, head.to_dict())
                            self._index_put(run_id, namespace)
                            receipt = RegistryTransactionReceipt(
                                operation=RegistryOperation.REPAIR,
                                outcome=RegistryTxOutcome.REPAIRED,
                                run_id=run_id,
                                run_revision=handle.run_revision,
                                handle_cid=handle.content_id,
                                integrity_cid=head.integrity_cid,
                                previous_revision=0,
                                previous_handle_cid="",
                                reason_codes=("repaired_missing_head",),
                                committed_at_ms=self.clock_ms(),
                            )
                            receipts.append(receipt)
                            repaired.append(run_id)
                            reasons.append("repaired_missing_head")
                            continue
                        raise RunRegistryCorruptionError(
                            "cannot uniquely repair missing head",
                            run_id=run_id,
                            reason_codes=("repair_ambiguous_handles",),
                        )
                    raise RunRegistryCorruptionError(
                        "partial run directory cannot be repaired",
                        run_id=run_id,
                        reason_codes=("repair_unrecoverable",),
                    )
                except RunRegistryCorruptionError as exc:
                    _path, receipt = self._quarantine(
                        run_id=run_id,
                        run_namespace=namespace,
                        reason_codes=exc.reason_codes or ("repair_quarantine",),
                        detail={"message": str(exc)},
                    )
                    receipts.append(receipt)
                    quarantined.append(run_id)
                    reasons.extend(exc.reason_codes or ("repair_quarantine",))
                except (RunRegistryError, OSError, ValueError, TypeError) as exc:
                    _path, receipt = self._quarantine(
                        run_id=run_id,
                        run_namespace=namespace,
                        reason_codes=("repair_exception",),
                        detail={"message": str(exc)},
                    )
                    receipts.append(receipt)
                    quarantined.append(run_id)
                    reasons.append("repair_exception")

            # Rebuild index from surviving roots.
            mapping: dict[str, str] = {}
            for run_id, namespace in self._scan_all_run_ids():
                try:
                    self._reconstruct_unlocked(namespace, run_id)
                except (RunRegistryError, OSError, ValueError, TypeError):
                    continue
                mapping[run_id] = namespace
            self._save_index(mapping)

        return RepairReport(
            repaired_run_ids=tuple(sorted(set(repaired))),
            quarantined_run_ids=tuple(sorted(set(quarantined))),
            receipts=tuple(receipts),
            reason_codes=_reason_codes(reasons),
        )


__all__ = (
    "DEFAULT_MAX_LIST",
    "HARD_MAX_LIST",
    "NAMESPACE_CURRENT_SCHEMA",
    "RUN_HEAD_SCHEMA",
    "RUN_REGISTRY_REQUIREMENT_ID",
    "RUN_REGISTRY_SCHEMA",
    "RUN_ROOT_SCHEMA",
    "RUN_TX_SCHEMA",
    "NamespaceCurrentRecord",
    "RegistryOperation",
    "RegistryTransactionReceipt",
    "RegistryTxOutcome",
    "RepairReport",
    "RunCasConflictError",
    "RunExistsError",
    "RunHeadRecord",
    "RunIncompatibleError",
    "RunNotFoundError",
    "RunRegistry",
    "RunRegistryBoundsError",
    "RunRegistryCorruptionError",
    "RunRegistryError",
    "RunRootRecord",
    "RunSelectionResult",
    "classify_run_candidate",
    "RunAdoptionAction",
    "RunCandidateClass",
    "RunCandidateEvidence",
    "WorktreeIsolationMode",
    "ADOPTABLE_HEALTH",
    "ADOPTABLE_RUN_STATES",
    "TERMINAL_RUN_STATES",
    "ClassifiedRunCandidate",
    "ContinuationAction",
    "RunHandle",
    "RunHealth",
    "RunState",
    "DuckDBRunRegistryBackend",
)
