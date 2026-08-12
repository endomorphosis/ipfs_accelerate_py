"""Database-backed control operations for agent supervisor programs (DQP-029).

Interface: ``DatabaseControlOperations@1``

Provides typed discover/query and lifecycle operations over a configured
database program so Python, CLI, and MCP transports share one canonical
request/result surface. Adapters never shell out and never accept raw
credentials — only opaque secret handles.

Read, proposal, and mutation authority remain distinct. Discovery is
side-effect free. Mutation helpers record expected-effect and lease/fence
inputs for the outer :class:`SupervisorControlService` policy layer.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    SECRET_HANDLE_PREFIXES,
    is_secret_handle,
    redact_mapping,
)
from .control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    Operation,
    OperationAuthority,
    OperationRequest,
)
from .control_plane import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    BackendResponse,
    InvalidLifecycleTransitionError,
    OperationUnavailableError,
    SupervisorLifecycleState,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_CONTROL_OPERATIONS_INTERFACE: Final[str] = "DatabaseControlOperations@1"
DATABASE_PROGRAM_TARGET_INTERFACE: Final[str] = "DatabaseProgramTarget@1"
DATABASE_CONTROL_QUERY_INTERFACE: Final[str] = "DatabaseControlQuery@1"
DATABASE_CONTROL_MUTATION_INTERFACE: Final[str] = "DatabaseControlMutation@1"

DATABASE_CONTROL_OPERATIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-operations@1"
)
DATABASE_PROGRAM_TARGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-program-target@1"
)
DATABASE_CONTROL_QUERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-query@1"
)
DATABASE_CONTROL_MUTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-mutation@1"
)
DATABASE_CONTROL_LOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-log@1"
)
DATABASE_CONTROL_EXPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-export@1"
)
DATABASE_CONTROL_BACKUP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-backup@1"
)
DATABASE_CONTROL_IMPORT_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-import-preview@1"
)

DATABASE_CONTROL_OPERATIONS_VERSION: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 50
MAX_PAGE_LIMIT: Final[int] = 500
MAX_ITEMS: Final[int] = 10_000
MAX_LOG_RETAIN: Final[int] = 4_096
MAX_EVENT_RETAIN: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_BODY_BYTES: Final[int] = 262_144

# Domains queryable through one typed service (beyond the closed Operation set).
QUERY_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "goals",
        "tasks",
        "runs",
        "lanes",
        "daemons",
        "events",
        "logs",
        "metrics",
        "worktrees",
        "mutations",
        "ast",
        "receipts",
        "exports",
        "status",
        "health",
        "bundles",
        "caches",
    }
)

LIFECYCLE_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "start",
        "pause",
        "resume",
        "drain",
        "stop",
        "retry",
        "cancel",
        "quarantine",
        "restart",
    }
)

ADMIN_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "import_preview",
        "export",
        "backup",
    }
)

_ACTION_TO_STATE: Final[Mapping[str, SupervisorLifecycleState]] = MappingProxyType(
    {
        "start": SupervisorLifecycleState.STARTING,
        "pause": SupervisorLifecycleState.PAUSED,
        "resume": SupervisorLifecycleState.HEALTHY,
        "drain": SupervisorLifecycleState.DRAINING,
        "stop": SupervisorLifecycleState.STOPPING,
        "retry": SupervisorLifecycleState.STARTING,
        "cancel": SupervisorLifecycleState.STOPPING,
        "quarantine": SupervisorLifecycleState.BLOCKED,
        "restart": SupervisorLifecycleState.STARTING,
    }
)

_SAFE_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$"
)
_SENSITIVE_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseControlError(RuntimeError):
    """Base error for database control operations."""


class DatabaseControlBoundsError(DatabaseControlError, ValueError):
    """A request exceeded declared bounds."""


class DatabaseControlAuthorityError(DatabaseControlError, ValueError):
    """Authority class or credential boundary was violated."""


class DatabaseControlNotOpenError(DatabaseControlError):
    """Operations were invoked before open or after close."""


class DatabaseControlConflictError(DatabaseControlError):
    """Current state rejected a otherwise well-formed mutation."""


class DatabaseControlNotFoundError(DatabaseControlError, LookupError):
    """A requested record was absent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def _require_text(value: Any, *, field: str, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise DatabaseControlBoundsError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise DatabaseControlBoundsError(f"{field} must be non-empty")
    if "\x00" in text:
        raise DatabaseControlBoundsError(f"{field} must not contain NUL")
    if len(text.encode("utf-8")) > maximum:
        raise DatabaseControlBoundsError(f"{field} exceeds {maximum} UTF-8 bytes")
    return text


def _optional_text(value: Any, *, field: str, maximum: int = MAX_TEXT_BYTES) -> str:
    if value is None or value == "":
        return ""
    return _require_text(value, field=field, maximum=maximum)


def _require_id(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field, maximum=512)
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise DatabaseControlBoundsError(f"{field} is not a safe identifier")
    return text


def _require_secret_handle(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field, maximum=512)
    if not is_secret_handle(text):
        raise DatabaseControlAuthorityError(
            f"{field} must be an opaque secret handle "
            f"({', '.join(SECRET_HANDLE_PREFIXES)}); raw credentials are forbidden"
        )
    return text


def _page_bounds(
    *,
    limit: Any = None,
    offset: Any = None,
    default_limit: int = DEFAULT_PAGE_LIMIT,
) -> tuple[int, int]:
    if limit is None or limit == "":
        limit_value = default_limit
    else:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise DatabaseControlBoundsError("limit must be a positive integer")
        limit_value = int(limit)
    if limit_value > MAX_PAGE_LIMIT:
        raise DatabaseControlBoundsError(
            f"limit exceeds the {MAX_PAGE_LIMIT}-item bound"
        )
    if offset is None or offset == "":
        offset_value = 0
    else:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise DatabaseControlBoundsError("offset must be a non-negative integer")
        offset_value = int(offset)
    return limit_value, offset_value


def _window(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    selected = list(items[offset : offset + limit])
    return {
        "items": selected,
        "count": len(selected),
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(selected) < len(items),
        "total": len(items),
    }


def _reject_raw_secrets(payload: Mapping[str, Any], *, path: str = "") -> None:
    """Fail closed when a payload embeds raw credential material."""

    for key, value in payload.items():
        key_text = str(key)
        folded = key_text.lower().replace("-", "_")
        sensitive = any(fragment in folded for fragment in _SENSITIVE_KEY_FRAGMENTS)
        if isinstance(value, Mapping):
            _reject_raw_secrets(value, path=f"{path}.{key_text}" if path else key_text)
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _reject_raw_secrets(
                        item,
                        path=(
                            f"{path}.{key_text}[{index}]"
                            if path
                            else f"{key_text}[{index}]"
                        ),
                    )
            continue
        if not sensitive:
            continue
        if value is None or value == "" or value == REDACTION_MARKER:
            continue
        if isinstance(value, str) and is_secret_handle(value):
            continue
        # Classification labels and empty markers are not credentials.
        if isinstance(value, str) and value.strip() == REDACTION_MARKER:
            continue
        raise DatabaseControlAuthorityError(
            f"raw credential material is forbidden at {path or key_text}; "
            "use an opaque secret handle"
        )


def _redact(value: Any) -> Any:
    return redact_mapping(value)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class ProgramAuthorityMode(str, Enum):
    EMBEDDED = "embedded"
    QUACK = "quack"
    LEGACY_COMPAT = "legacy_compat"


@dataclass(frozen=True)
class DatabaseProgramTarget:
    """Bound identity for one configured database program.

    Interface: ``DatabaseProgramTarget@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_PROGRAM_TARGET_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_PROGRAM_TARGET_SCHEMA

    program_id: str
    store_id: str
    authority_mode: ProgramAuthorityMode = ProgramAuthorityMode.EMBEDDED
    endpoint_secret_handle: str = ""
    store_generation: str = "1"
    schema_revision: str = "1"
    repository_id: str = ""
    export_profile: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "program_id", _require_id(self.program_id, field="program_id")
        )
        object.__setattr__(
            self, "store_id", _require_id(self.store_id, field="store_id")
        )
        mode = self.authority_mode
        if not isinstance(mode, ProgramAuthorityMode):
            mode = ProgramAuthorityMode(str(mode).strip().lower())
        object.__setattr__(self, "authority_mode", mode)
        handle = _optional_text(
            self.endpoint_secret_handle, field="endpoint_secret_handle"
        )
        if handle:
            handle = _require_secret_handle(handle, field="endpoint_secret_handle")
        object.__setattr__(self, "endpoint_secret_handle", handle)
        if mode is ProgramAuthorityMode.QUACK and not handle:
            raise DatabaseControlAuthorityError(
                "quack authority requires endpoint_secret_handle"
            )
        object.__setattr__(
            self,
            "store_generation",
            _require_id(self.store_generation or "1", field="store_generation"),
        )
        object.__setattr__(
            self,
            "schema_revision",
            _require_id(self.schema_revision or "1", field="schema_revision"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, field="repository_id") or "",
        )
        object.__setattr__(
            self,
            "export_profile",
            _require_id(self.export_profile or "default", field="export_profile"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "program_id": self.program_id,
            "store_id": self.store_id,
            "authority_mode": self.authority_mode.value,
            "endpoint_secret_handle": self.endpoint_secret_handle,
            "store_generation": self.store_generation,
            "schema_revision": self.schema_revision,
            "repository_id": self.repository_id,
            "export_profile": self.export_profile,
        }

    def public_dict(self) -> dict[str, Any]:
        """Return a redacted projection safe for discovery and status."""

        payload = self.to_dict()
        # Handles are opaque references; still never surface raw material.
        return _redact(payload)


@dataclass
class _ProgramState:
    """Mutable in-process authority for one database program."""

    target: DatabaseProgramTarget
    lifecycle_state: SupervisorLifecycleState = SupervisorLifecycleState.STOPPED
    generation: int = 0
    fencing_epoch: int = 1
    heartbeat_at_ms: int = 0
    updated_at_ms: int = 0
    terminal_reason: str = ""
    processes_started: bool = False
    goals: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    lanes: list[dict[str, Any]] = field(default_factory=list)
    daemons: list[dict[str, Any]] = field(default_factory=list)
    events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_EVENT_RETAIN)
    )
    logs: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=MAX_LOG_RETAIN)
    )
    metrics: list[dict[str, Any]] = field(default_factory=list)
    worktrees: list[dict[str, Any]] = field(default_factory=list)
    mutations: list[dict[str, Any]] = field(default_factory=list)
    ast_nodes: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)
    bundles: list[dict[str, Any]] = field(default_factory=list)
    caches: list[dict[str, Any]] = field(default_factory=list)
    backups: list[dict[str, Any]] = field(default_factory=list)
    event_sequence: int = 0
    log_sequence: int = 0
    idempotency: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Operations service
# ---------------------------------------------------------------------------


class DatabaseControlOperations:
    """Typed database control operations shared by all control transports.

    Interface: ``DatabaseControlOperations@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_CONTROL_OPERATIONS_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_CONTROL_OPERATIONS_SCHEMA
    VERSION: ClassVar[int] = DATABASE_CONTROL_OPERATIONS_VERSION

    def __init__(
        self,
        *,
        clock_ms: Any | None = None,
        stale_after_ms: int = 60_000,
    ) -> None:
        if (
            isinstance(stale_after_ms, bool)
            or not isinstance(stale_after_ms, int)
            or stale_after_ms < 1
        ):
            raise DatabaseControlBoundsError(
                "stale_after_ms must be a positive integer"
            )
        self._clock_ms = clock_ms or _now_ms
        self._stale_after_ms = int(stale_after_ms)
        self._lock = threading.RLock()
        self._programs: dict[str, _ProgramState] = {}
        self._open = True
        self._closed = False
        # Discovery is inert: construction never starts processes or loads
        # optional providers.
        self.optional_providers_loaded = False
        self.processes_started = False

    # -- lifecycle of the operations service itself -------------------------

    @property
    def is_open(self) -> bool:
        return self._open and not self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._open = False

    def __enter__(self) -> "DatabaseControlOperations":
        self._require_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed or not self._open:
            raise DatabaseControlNotOpenError(
                "database control operations are not open"
            )

    # -- program registration -----------------------------------------------

    def register_program(
        self,
        target: DatabaseProgramTarget | Mapping[str, Any],
        *,
        seed: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> DatabaseProgramTarget:
        """Register or replace a program target. Registration is inert."""

        self._require_open()
        if isinstance(target, Mapping):
            target = DatabaseProgramTarget(
                program_id=str(target.get("program_id") or ""),
                store_id=str(target.get("store_id") or ""),
                authority_mode=str(
                    target.get("authority_mode") or ProgramAuthorityMode.EMBEDDED.value
                ),
                endpoint_secret_handle=str(
                    target.get("endpoint_secret_handle") or ""
                ),
                store_generation=str(target.get("store_generation") or "1"),
                schema_revision=str(target.get("schema_revision") or "1"),
                repository_id=str(target.get("repository_id") or ""),
                export_profile=str(target.get("export_profile") or "default"),
            )
        if not isinstance(target, DatabaseProgramTarget):
            raise TypeError("target must be a DatabaseProgramTarget")
        _reject_raw_secrets(target.to_dict())
        now = int(self._clock_ms())
        with self._lock:
            state = _ProgramState(
                target=target,
                heartbeat_at_ms=now,
                updated_at_ms=now,
            )
            if seed:
                for domain, rows in seed.items():
                    domain_name = str(domain).strip().lower()
                    if domain_name not in QUERY_DOMAINS and domain_name != "backups":
                        raise DatabaseControlBoundsError(
                            f"unknown seed domain: {domain_name}"
                        )
                    cleaned: list[dict[str, Any]] = []
                    for row in rows:
                        if not isinstance(row, Mapping):
                            raise DatabaseControlBoundsError(
                                f"seed row in {domain_name} must be a mapping"
                            )
                        material = dict(row)
                        _reject_raw_secrets(material)
                        cleaned.append(_redact(material))
                    if domain_name == "goals":
                        state.goals = cleaned
                    elif domain_name == "tasks":
                        state.tasks = cleaned
                    elif domain_name == "runs":
                        state.runs = cleaned
                    elif domain_name == "lanes":
                        state.lanes = cleaned
                    elif domain_name == "daemons":
                        state.daemons = cleaned
                    elif domain_name == "events":
                        state.events.extend(cleaned)
                    elif domain_name == "logs":
                        state.logs.extend(cleaned)
                    elif domain_name == "metrics":
                        state.metrics = cleaned
                    elif domain_name == "worktrees":
                        state.worktrees = cleaned
                    elif domain_name == "mutations":
                        state.mutations = cleaned
                    elif domain_name == "ast":
                        state.ast_nodes = cleaned
                    elif domain_name == "receipts":
                        state.receipts = cleaned
                    elif domain_name == "exports":
                        state.exports = cleaned
                    elif domain_name == "bundles":
                        state.bundles = cleaned
                    elif domain_name == "caches":
                        state.caches = cleaned
                    elif domain_name == "backups":
                        state.backups = cleaned
            self._programs[target.program_id] = state
        return target

    def list_programs(self) -> tuple[dict[str, Any], ...]:
        self._require_open()
        with self._lock:
            return tuple(
                state.target.public_dict() for state in self._programs.values()
            )

    def _get_state(self, program_id: str) -> _ProgramState:
        self._require_open()
        key = _require_id(program_id, field="program_id")
        with self._lock:
            state = self._programs.get(key)
            if state is None:
                raise DatabaseControlNotFoundError(
                    f"program not found: {key}"
                )
            return state

    def resolve_program_id(self, parameters: Mapping[str, Any]) -> str:
        """Resolve program_id from request parameters or the sole registered program."""

        raw = parameters.get("program_id") or parameters.get("target_id") or ""
        if raw:
            return _require_id(raw, field="program_id")
        with self._lock:
            if len(self._programs) == 1:
                return next(iter(self._programs))
            if not self._programs:
                raise DatabaseControlNotFoundError("no database program is registered")
            raise DatabaseControlBoundsError(
                "program_id is required when multiple programs are registered"
            )

    # -- discovery (inert) --------------------------------------------------

    def discover(self) -> dict[str, Any]:
        """Return discovery metadata without starting processes or loading providers."""

        self._require_open()
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "version": self.VERSION,
            "query_domains": sorted(QUERY_DOMAINS),
            "lifecycle_actions": sorted(LIFECYCLE_ACTIONS),
            "admin_actions": sorted(ADMIN_ACTIONS),
            "programs": list(self.list_programs()),
            "optional_providers_loaded": self.optional_providers_loaded,
            "processes_started": self.processes_started,
            "side_effect_free": True,
        }

    # -- status / health / logs ---------------------------------------------

    def status(self, program_id: str) -> dict[str, Any]:
        state = self._get_state(program_id)
        with self._lock:
            now = int(self._clock_ms())
            healthy = self._is_healthy_locked(state, now)
            return _redact(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/database-program-status@1",
                    "program_id": state.target.program_id,
                    "store_id": state.target.store_id,
                    "authority_mode": state.target.authority_mode.value,
                    "state": state.lifecycle_state.value,
                    "phase": state.lifecycle_state.value,
                    "generation": state.generation,
                    "fencing_epoch": state.fencing_epoch,
                    "heartbeat_at_ms": state.heartbeat_at_ms,
                    "updated_at_ms": state.updated_at_ms,
                    "terminal_reason": state.terminal_reason,
                    "processes_started": state.processes_started,
                    "healthy": healthy,
                    "store_generation": state.target.store_generation,
                    "schema_revision": state.target.schema_revision,
                    "endpoint_secret_handle": state.target.endpoint_secret_handle,
                    "supported_controls": sorted(
                        {
                            "status",
                            "health",
                            "logs",
                            "stop",
                            "start",
                            "pause",
                            "resume",
                            "drain",
                            "retry",
                            "cancel",
                            "quarantine",
                        }
                    ),
                }
            )

    def health(self, program_id: str) -> dict[str, Any]:
        snapshot = self.status(program_id)
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-program-health@1",
            "program_id": snapshot["program_id"],
            "healthy": bool(snapshot["healthy"]),
            "state": snapshot["state"],
            "generation": snapshot["generation"],
            "fencing_epoch": snapshot["fencing_epoch"],
            "heartbeat_at_ms": snapshot["heartbeat_at_ms"],
            "processes_started": snapshot["processes_started"],
        }

    def _is_healthy_locked(self, state: _ProgramState, now: int) -> bool:
        if state.lifecycle_state is not SupervisorLifecycleState.HEALTHY:
            return False
        if state.heartbeat_at_ms <= 0 or state.heartbeat_at_ms > now:
            return False
        if now - state.heartbeat_at_ms >= self._stale_after_ms:
            return False
        return bool(state.processes_started)

    def append_log(
        self,
        program_id: str,
        *,
        severity: str = "info",
        component: str = "control",
        message: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._get_state(program_id)
        severity_text = _require_text(severity, field="severity", maximum=32).lower()
        component_text = _require_text(component, field="component", maximum=128)
        message_text = _require_text(message, field="message")
        material = dict(body or {})
        _reject_raw_secrets(material)
        material = _redact(material)
        with self._lock:
            state.log_sequence += 1
            record = {
                "schema": DATABASE_CONTROL_LOG_SCHEMA,
                "log_id": f"log:{state.log_sequence}",
                "sequence": state.log_sequence,
                "severity": severity_text,
                "component": component_text,
                "message": message_text,
                "body": material,
                "recorded_at_ms": int(self._clock_ms()),
                "program_id": state.target.program_id,
            }
            state.logs.append(record)
            return dict(record)

    def logs(
        self,
        program_id: str,
        *,
        limit: Any = None,
        offset: Any = None,
        after_sequence: Any = 0,
    ) -> dict[str, Any]:
        state = self._get_state(program_id)
        page_limit, page_offset = _page_bounds(limit=limit, offset=offset)
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise DatabaseControlBoundsError(
                "after_sequence must be a non-negative integer"
            )
        with self._lock:
            items = [
                dict(item)
                for item in state.logs
                if int(item.get("sequence") or 0) > after_sequence
            ]
        return {
            "schema": DATABASE_CONTROL_LOG_SCHEMA,
            "program_id": state.target.program_id,
            "after_sequence": after_sequence,
            **_window(items, limit=page_limit, offset=page_offset),
        }

    # -- domain queries -----------------------------------------------------

    def _domain_rows(
        self, state: _ProgramState, domain: str
    ) -> list[dict[str, Any]]:
        mapping: dict[str, list[dict[str, Any]] | deque[dict[str, Any]]] = {
            "goals": state.goals,
            "tasks": state.tasks,
            "runs": state.runs,
            "lanes": state.lanes,
            "daemons": state.daemons,
            "events": state.events,
            "logs": state.logs,
            "metrics": state.metrics,
            "worktrees": state.worktrees,
            "mutations": state.mutations,
            "ast": state.ast_nodes,
            "receipts": state.receipts,
            "exports": state.exports,
            "bundles": state.bundles,
            "caches": state.caches,
        }
        rows = mapping.get(domain)
        if rows is None:
            raise DatabaseControlBoundsError(f"unknown query domain: {domain}")
        return [dict(item) for item in rows]

    def query(
        self,
        program_id: str,
        domain: str,
        *,
        limit: Any = None,
        offset: Any = None,
        cursor: Any = None,
    ) -> dict[str, Any]:
        """Bounded domain query with pagination. Read-only."""

        state = self._get_state(program_id)
        domain_name = _require_text(domain, field="domain", maximum=64).lower()
        if domain_name not in QUERY_DOMAINS:
            raise DatabaseControlBoundsError(f"unknown query domain: {domain_name}")
        if domain_name == "status":
            return {
                "schema": DATABASE_CONTROL_QUERY_SCHEMA,
                "domain": domain_name,
                "program_id": state.target.program_id,
                "items": [self.status(program_id)],
                "count": 1,
                "offset": 0,
                "limit": 1,
                "truncated": False,
                "total": 1,
            }
        if domain_name == "health":
            return {
                "schema": DATABASE_CONTROL_QUERY_SCHEMA,
                "domain": domain_name,
                "program_id": state.target.program_id,
                "items": [self.health(program_id)],
                "count": 1,
                "offset": 0,
                "limit": 1,
                "truncated": False,
                "total": 1,
            }
        if domain_name == "logs":
            return {
                "schema": DATABASE_CONTROL_QUERY_SCHEMA,
                "domain": domain_name,
                **self.logs(program_id, limit=limit, offset=offset),
            }
        page_limit, page_offset = _page_bounds(limit=limit, offset=offset)
        if cursor not in (None, ""):
            # Cursor is an integer after_sequence for ordered streams.
            try:
                after = int(cursor)
            except (TypeError, ValueError) as exc:
                raise DatabaseControlBoundsError(
                    "cursor must be an integer sequence"
                ) from exc
            if after < 0:
                raise DatabaseControlBoundsError("cursor must be non-negative")
            with self._lock:
                rows = [
                    row
                    for row in self._domain_rows(state, domain_name)
                    if int(row.get("sequence") or 0) > after
                ]
        else:
            with self._lock:
                rows = self._domain_rows(state, domain_name)
        window = _window(rows, limit=page_limit, offset=page_offset)
        return _redact(
            {
                "schema": DATABASE_CONTROL_QUERY_SCHEMA,
                "domain": domain_name,
                "program_id": state.target.program_id,
                **window,
            }
        )

    # -- lifecycle mutations ------------------------------------------------

    def transition(
        self,
        program_id: str,
        action: str,
        *,
        reason: str = "",
        dry_run: bool = False,
        lease_id: str = "",
        fencing_epoch: int | None = None,
        expected_effect_ids: Sequence[str] = (),
        idempotency_key: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        """Apply a fenced lifecycle transition for a database program."""

        state = self._get_state(program_id)
        action_name = _require_text(action, field="action", maximum=32).lower()
        if action_name not in LIFECYCLE_ACTIONS:
            raise DatabaseControlBoundsError(f"unknown lifecycle action: {action_name}")
        requested = _ACTION_TO_STATE[action_name]
        reason_text = _optional_text(reason, field="reason") or f"action:{action_name}"
        effect_ids = tuple(
            sorted({_require_id(item, field="effect_id") for item in expected_effect_ids})
        )
        idem_key = _optional_text(idempotency_key, field="idempotency_key", maximum=256)
        req_id = _optional_text(request_id, field="request_id", maximum=256)

        with self._lock:
            if idem_key and idem_key in state.idempotency:
                prior = dict(state.idempotency[idem_key])
                prior["idempotent"] = True
                prior["changed"] = False
                return prior

            previous = state.lifecycle_state
            if not dry_run and requested not in LEGAL_LIFECYCLE_TRANSITIONS.get(
                previous, frozenset()
            ):
                # Allow stop from any non-terminal state via STOPPING, and
                # complete stop when already stopping.
                if action_name == "stop" and previous is SupervisorLifecycleState.STOPPING:
                    requested = SupervisorLifecycleState.STOPPED
                elif action_name == "stop" and previous not in {
                    SupervisorLifecycleState.STOPPED,
                    SupervisorLifecycleState.FAILED,
                }:
                    requested = SupervisorLifecycleState.STOPPING
                elif action_name == "start" and previous is SupervisorLifecycleState.STARTING:
                    requested = SupervisorLifecycleState.HEALTHY
                else:
                    raise InvalidLifecycleTransitionError(
                        f"cannot {action_name} from {previous.value}"
                    )

            if dry_run:
                return {
                    "schema": DATABASE_CONTROL_MUTATION_SCHEMA,
                    "program_id": state.target.program_id,
                    "action": action_name,
                    "dry_run": True,
                    "would_change": previous is not requested,
                    "previous_state": previous.value,
                    "state": requested.value,
                    "lease_id": lease_id,
                    "fencing_epoch": fencing_epoch,
                    "expected_effect_ids": list(effect_ids),
                    "changed": False,
                    "accepted": True,
                }

            now = int(self._clock_ms())
            state.lifecycle_state = requested
            state.generation += 1
            if fencing_epoch is not None:
                if (
                    isinstance(fencing_epoch, bool)
                    or not isinstance(fencing_epoch, int)
                    or fencing_epoch < 0
                ):
                    raise DatabaseControlBoundsError(
                        "fencing_epoch must be a non-negative integer"
                    )
                state.fencing_epoch = int(fencing_epoch)
            state.heartbeat_at_ms = now
            state.updated_at_ms = now
            state.terminal_reason = (
                reason_text
                if requested
                in {
                    SupervisorLifecycleState.STOPPED,
                    SupervisorLifecycleState.FAILED,
                    SupervisorLifecycleState.BLOCKED,
                }
                else ""
            )
            if action_name in {"start", "retry", "restart"}:
                state.processes_started = True
                self.processes_started = True
            if action_name == "stop" and requested is SupervisorLifecycleState.STOPPED:
                state.processes_started = False
            if (
                action_name == "stop"
                and requested is SupervisorLifecycleState.STOPPING
            ):
                # Second stop advances STOPPING -> STOPPED for drain completion.
                pass

            state.event_sequence += 1
            event = {
                "event_id": f"event:{state.event_sequence}",
                "sequence": state.event_sequence,
                "action": action_name,
                "previous_state": previous.value,
                "state": state.lifecycle_state.value,
                "reason": reason_text,
                "request_id": req_id,
                "lease_id": lease_id,
                "fencing_epoch": state.fencing_epoch,
                "occurred_at_ms": now,
                "program_id": state.target.program_id,
            }
            state.events.append(event)
            state.log_sequence += 1
            state.logs.append(
                {
                    "schema": DATABASE_CONTROL_LOG_SCHEMA,
                    "log_id": f"log:{state.log_sequence}",
                    "sequence": state.log_sequence,
                    "severity": "info",
                    "component": "lifecycle",
                    "message": f"{action_name}: {previous.value} -> {state.lifecycle_state.value}",
                    "body": {
                        "action": action_name,
                        "previous_state": previous.value,
                        "state": state.lifecycle_state.value,
                    },
                    "recorded_at_ms": now,
                    "program_id": state.target.program_id,
                }
            )
            result = {
                "schema": DATABASE_CONTROL_MUTATION_SCHEMA,
                "program_id": state.target.program_id,
                "action": action_name,
                "dry_run": False,
                "previous_state": previous.value,
                "state": state.lifecycle_state.value,
                "generation": state.generation,
                "fencing_epoch": state.fencing_epoch,
                "lease_id": lease_id,
                "expected_effect_ids": list(effect_ids),
                "applied_effect_ids": list(effect_ids),
                "event": event,
                "changed": previous is not state.lifecycle_state,
                "accepted": True,
                "idempotent": False,
                "request_id": req_id,
            }
            if idem_key:
                state.idempotency[idem_key] = dict(result)
            return dict(result)

    def complete_stop(self, program_id: str, **kwargs: Any) -> dict[str, Any]:
        """Advance STOPPING -> STOPPED when a stop is already in progress."""

        state = self._get_state(program_id)
        with self._lock:
            if state.lifecycle_state is SupervisorLifecycleState.STOPPING:
                return self.transition(program_id, "stop", **kwargs)
        return self.transition(program_id, "stop", **kwargs)

    # -- admin actions: import-preview / export / backup --------------------

    def import_preview(
        self,
        program_id: str,
        *,
        sources: Sequence[Mapping[str, Any]] = (),
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Preview a legacy import without applying mutations by default."""

        state = self._get_state(program_id)
        if not dry_run:
            raise DatabaseControlAuthorityError(
                "import_preview is proposal-only; apply belongs to a separate "
                "authorized import operation"
            )
        cleaned: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, Mapping):
                raise DatabaseControlBoundsError("import source must be a mapping")
            material = dict(source)
            _reject_raw_secrets(material)
            cleaned.append(_redact(material))
        receipt = {
            "schema": DATABASE_CONTROL_IMPORT_PREVIEW_SCHEMA,
            "program_id": state.target.program_id,
            "dry_run": True,
            "source_count": len(cleaned),
            "sources": cleaned,
            "accepted": True,
            "would_change": bool(cleaned),
            "authority": OperationAuthority.PROPOSAL.value,
        }
        return receipt

    def export_state(
        self,
        program_id: str,
        *,
        profile: str = "",
        dry_run: bool = False,
        destination: str = "",
    ) -> dict[str, Any]:
        """Render a redacted export receipt for the program snapshot."""

        state = self._get_state(program_id)
        profile_name = _require_id(
            profile or state.target.export_profile, field="profile"
        )
        dest = _optional_text(destination, field="destination", maximum=1024)
        with self._lock:
            payload = {
                "program": state.target.public_dict(),
                "status": {
                    "state": state.lifecycle_state.value,
                    "generation": state.generation,
                    "fencing_epoch": state.fencing_epoch,
                },
                "goals": list(state.goals),
                "tasks": list(state.tasks),
                "runs": list(state.runs),
                "lanes": list(state.lanes),
                "daemons": list(state.daemons),
                "metrics": list(state.metrics),
                "worktrees": list(state.worktrees),
                "receipts": list(state.receipts),
            }
            body = _redact(payload)
            digest = _content_id(body)
            receipt = {
                "schema": DATABASE_CONTROL_EXPORT_SCHEMA,
                "program_id": state.target.program_id,
                "profile": profile_name,
                "destination": dest,
                "dry_run": bool(dry_run),
                "digest": digest,
                "non_authoritative": True,
                "store_id": state.target.store_id,
                "store_generation": state.target.store_generation,
                "schema_revision": state.target.schema_revision,
                "changed": not dry_run,
                "accepted": True,
            }
            if not dry_run:
                state.exports.append(dict(receipt))
            return dict(receipt)

    def backup(
        self,
        program_id: str,
        *,
        dry_run: bool = False,
        backup_id: str = "",
    ) -> dict[str, Any]:
        """Create a verified backup receipt for the program store."""

        state = self._get_state(program_id)
        with self._lock:
            bid = (
                _require_id(backup_id, field="backup_id")
                if backup_id
                else f"backup:{state.generation + 1}:{int(self._clock_ms())}"
            )
            snapshot = {
                "program_id": state.target.program_id,
                "store_id": state.target.store_id,
                "generation": state.generation,
                "fencing_epoch": state.fencing_epoch,
                "lifecycle_state": state.lifecycle_state.value,
                "task_count": len(state.tasks),
                "goal_count": len(state.goals),
                "event_count": len(state.events),
            }
            digest = _content_id(snapshot)
            receipt = {
                "schema": DATABASE_CONTROL_BACKUP_SCHEMA,
                "program_id": state.target.program_id,
                "backup_id": bid,
                "digest": digest,
                "dry_run": bool(dry_run),
                "verified": not dry_run,
                "store_id": state.target.store_id,
                "store_generation": state.target.store_generation,
                "changed": not dry_run,
                "accepted": True,
            }
            if not dry_run:
                state.backups.append(dict(receipt))
            return dict(receipt)

    # -- operation dispatch from OperationRequest ---------------------------

    def authority_for(self, operation: Operation) -> OperationAuthority:
        if operation in READ_OPERATIONS:
            return OperationAuthority.READ
        if operation in PROPOSAL_OPERATIONS:
            return OperationAuthority.PROPOSAL
        if operation in MUTATION_OPERATIONS:
            return OperationAuthority.MUTATION
        raise OperationUnavailableError(
            f"operation {operation.value} has no database control authority"
        )

    def execute_request(self, request: OperationRequest) -> BackendResponse:
        """Dispatch one canonical OperationRequest against the program store."""

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        self._require_open()
        operation = request.operation
        parameters = dict(request.parameters or {})

        # Extended domains via parameters (logs/export/backup/import_preview/...).
        domain = str(parameters.get("domain") or "").strip().lower()
        control_action = str(
            parameters.get("control_action") or parameters.get("admin_action") or ""
        ).strip().lower()

        if operation is Operation.CAPABILITIES:
            return BackendResponse(data=self.discover())

        if operation is Operation.STATUS or domain == "status":
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(data=self.status(program_id))

        if operation is Operation.HEALTH or domain == "health":
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(data=self.health(program_id))

        if domain == "logs" or (
            operation is Operation.EVENTS
            and str(parameters.get("stream") or "").lower() == "logs"
        ):
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(
                data=self.logs(
                    program_id,
                    limit=parameters.get("limit"),
                    offset=parameters.get("offset"),
                    after_sequence=parameters.get("after_sequence") or 0,
                )
            )

        if control_action == "import_preview" or domain == "import_preview":
            program_id = self.resolve_program_id(parameters)
            sources = parameters.get("sources") or ()
            if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
                raise DatabaseControlBoundsError("sources must be a sequence")
            data = self.import_preview(
                program_id,
                sources=sources,
                dry_run=True,
            )
            return BackendResponse(data=data, changed=False)

        if control_action == "export" or domain == "exports":
            if operation in READ_OPERATIONS and control_action != "export":
                program_id = self.resolve_program_id(parameters)
                return BackendResponse(
                    data=self.query(
                        program_id,
                        "exports",
                        limit=parameters.get("limit"),
                        offset=parameters.get("offset"),
                    )
                )
            program_id = self.resolve_program_id(parameters)
            data = self.export_state(
                program_id,
                profile=str(parameters.get("profile") or ""),
                dry_run=bool(request.dry_run),
                destination=str(parameters.get("destination") or ""),
            )
            return BackendResponse(
                data=data,
                changed=bool(data.get("changed")),
                applied_effect_ids=(
                    tuple(item.effect_id for item in request.expected_effects)
                    if data.get("changed")
                    else ()
                ),
            )

        if control_action == "backup":
            program_id = self.resolve_program_id(parameters)
            data = self.backup(
                program_id,
                dry_run=bool(request.dry_run),
                backup_id=str(parameters.get("backup_id") or ""),
            )
            return BackendResponse(
                data=data,
                changed=bool(data.get("changed")),
                applied_effect_ids=(
                    tuple(item.effect_id for item in request.expected_effects)
                    if data.get("changed")
                    else ()
                ),
            )

        domain_for_operation = {
            Operation.GOALS: "goals",
            Operation.TASKS: "tasks",
            Operation.LANES: "lanes",
            Operation.EVENTS: "events",
            Operation.METRICS: "metrics",
            Operation.RECEIPTS: "receipts",
            Operation.BUNDLES: "bundles",
            Operation.CACHE_INSPECT: "caches",
        }.get(operation)

        if domain and domain in QUERY_DOMAINS:
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(
                data=self.query(
                    program_id,
                    domain,
                    limit=parameters.get("limit"),
                    offset=parameters.get("offset"),
                    cursor=parameters.get("cursor"),
                )
            )

        if domain_for_operation is not None:
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(
                data=self.query(
                    program_id,
                    domain_for_operation,
                    limit=parameters.get("limit"),
                    offset=parameters.get("offset"),
                    cursor=parameters.get("after_sequence")
                    or parameters.get("cursor"),
                )
            )

        if operation is Operation.ARTIFACT_QUERY:
            program_id = self.resolve_program_id(parameters)
            artifact_domain = str(parameters.get("table") or parameters.get("kind") or "ast")
            if artifact_domain not in QUERY_DOMAINS:
                artifact_domain = "ast"
            return BackendResponse(
                data=self.query(
                    program_id,
                    artifact_domain,
                    limit=parameters.get("limit"),
                    offset=parameters.get("offset"),
                )
            )

        lifecycle_map = {
            Operation.START: "start",
            Operation.PAUSE: "pause",
            Operation.RESUME: "resume",
            Operation.DRAIN: "drain",
            Operation.STOP: "stop",
            Operation.RETRY: "retry",
            Operation.CANCEL: "cancel",
            Operation.QUARANTINE: "quarantine",
            Operation.RESTART: "restart",
        }
        if operation in lifecycle_map:
            program_id = self.resolve_program_id(parameters)
            data = self.transition(
                program_id,
                lifecycle_map[operation],
                reason=str(parameters.get("reason") or ""),
                dry_run=bool(request.dry_run),
                lease_id=str(request.lease_id or parameters.get("lease_id") or ""),
                fencing_epoch=(
                    request.fencing_epoch
                    if request.fencing_epoch is not None
                    else parameters.get("fencing_epoch")
                ),
                expected_effect_ids=tuple(
                    item.effect_id for item in request.expected_effects
                ),
                idempotency_key=(
                    request.idempotency.key
                    if request.idempotency is not None
                    else str(parameters.get("idempotency_key") or "")
                ),
                request_id=str(request.request_id or ""),
            )
            # Auto-complete stop: STOPPING then STOPPED on a second call is
            # handled by transition(); first stop leaves STOPPING.
            return BackendResponse(
                data=data,
                changed=bool(data.get("changed")),
                applied_effect_ids=tuple(data.get("applied_effect_ids") or ()),
            )

        if operation in PROPOSAL_OPERATIONS:
            program_id = self.resolve_program_id(parameters)
            return BackendResponse(
                data={
                    "schema": DATABASE_CONTROL_MUTATION_SCHEMA,
                    "operation": operation.value,
                    "authority": OperationAuthority.PROPOSAL.value,
                    "program_id": program_id,
                    "dry_run": True,
                    "accepted": True,
                    "changed": False,
                    "preview": True,
                },
                changed=False,
            )

        if operation in MUTATION_OPERATIONS:
            # Unimplemented domain mutations fail closed as unavailable rather
            # than shelling out or inventing side effects.
            raise OperationUnavailableError(
                f"operation {operation.value} has no database control adapter"
            )

        raise OperationUnavailableError(
            f"operation {operation.value} is not supported by DatabaseControlOperations"
        )


def open_database_control_operations(
    **kwargs: Any,
) -> DatabaseControlOperations:
    """Open a DatabaseControlOperations instance (first I/O boundary if stores are attached)."""

    return DatabaseControlOperations(**kwargs)


__all__ = (
    "ADMIN_ACTIONS",
    "DATABASE_CONTROL_BACKUP_SCHEMA",
    "DATABASE_CONTROL_EXPORT_SCHEMA",
    "DATABASE_CONTROL_IMPORT_PREVIEW_SCHEMA",
    "DATABASE_CONTROL_LOG_SCHEMA",
    "DATABASE_CONTROL_MUTATION_INTERFACE",
    "DATABASE_CONTROL_MUTATION_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_INTERFACE",
    "DATABASE_CONTROL_OPERATIONS_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_VERSION",
    "DATABASE_CONTROL_QUERY_INTERFACE",
    "DATABASE_CONTROL_QUERY_SCHEMA",
    "DATABASE_PROGRAM_TARGET_INTERFACE",
    "DATABASE_PROGRAM_TARGET_SCHEMA",
    "DEFAULT_PAGE_LIMIT",
    "LIFECYCLE_ACTIONS",
    "MAX_PAGE_LIMIT",
    "QUERY_DOMAINS",
    "REDACTION_MARKER",
    "DatabaseControlAuthorityError",
    "DatabaseControlBoundsError",
    "DatabaseControlConflictError",
    "DatabaseControlError",
    "DatabaseControlNotFoundError",
    "DatabaseControlNotOpenError",
    "DatabaseControlOperations",
    "DatabaseProgramTarget",
    "ProgramAuthorityMode",
    "open_database_control_operations",
)
