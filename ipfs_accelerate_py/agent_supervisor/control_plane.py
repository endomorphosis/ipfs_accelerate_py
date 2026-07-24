"""Shared, transport-neutral control service for the agent supervisor.

The CLI and MCP surfaces intentionally do not implement supervisor policy.
They construct :class:`~.control_contracts.OperationRequest` records and pass
them to :class:`SupervisorControlService`.  This module is the single place
that applies target allowlists, bounds, authorization freshness, lease
fencing, idempotency, dry-run semantics, stable errors, and audit receipts.

Backends are ordinary Python callables.  No operation is converted to a shell
command.  The included :class:`RepositorySupervisorBackend` supplies bounded
read adapters for the package's existing JSON, objective, task-board, event,
cache, and artifact APIs.  Runtime deployments register the mutating package
APIs they operate; an unregistered mutation fails closed as ``unavailable``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, Union

from .control_contracts import (
    CONTROL_CONTRACT_VERSION,
    MUTATION_OPERATIONS,
    READ_OPERATIONS,
    AuthorizationBindingError,
    CapabilityReport,
    ControlBounds,
    ControlBoundsError,
    ControlContractError,
    ControlDiscoveryManifest,
    ControlDiscoveryRuntimeState,
    ControlSurface,
    DryRunPreview,
    EffectClaim,
    ErrorCode,
    ExpectedEffect,
    LifecycleAction,
    LifecycleCommand,
    Operation,
    OperationAuthority,
    OperationCapability,
    OperationError,
    OperationRequest,
    OperationResult,
    OperationStatus,
    PathEscapeError,
    canonical_control_json_bytes,
)


CONTROL_SERVICE_VERSION: Final[str] = "1.0.0"
CONTROL_AUDIT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-audit-receipt@1"
)
CONTROL_BACKEND_RESPONSE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-backend-response@1"
)
DEFAULT_QUERY_LIMIT: Final[int] = 50
DEFAULT_MAX_QUERY_ITEMS: Final[int] = 256
DEFAULT_MAX_OFFSET: Final[int] = 1_000_000
CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py",
    "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
    "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
    "ipfs_accelerate_py.agent_supervisor.formal_verification_provider",
)


class SupervisorControlError(RuntimeError):
    """Base exception raised by service configuration and client misuse."""


class TargetNotAllowedError(SupervisorControlError):
    """Raised internally when a request targets a root outside an allowlist."""


class OperationUnavailableError(SupervisorControlError):
    """A configured backend does not implement the requested operation."""


class LeaseValidationError(SupervisorControlError):
    """A mutation does not carry a current authoritative lease and fence."""


class StaleLeaseError(LeaseValidationError):
    """The supplied lease or fencing epoch is no longer current."""


class StaleTreeError(SupervisorControlError):
    """The request's repository/tree identity is no longer current."""


class IdempotencyConflictError(SupervisorControlError):
    """An idempotency key was already used for a different request."""


class BackendNotFoundError(SupervisorControlError):
    """The requested backend object does not exist."""


class BackendConflictError(SupervisorControlError):
    """The backend rejected an otherwise valid request due to current state."""


class BackendCancelledError(SupervisorControlError):
    """Backend execution was cancelled."""


class BackendTimeoutError(SupervisorControlError):
    """Backend execution exceeded its bound."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _utc_timestamp(now_ms: int) -> str:
    return (
        datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _current_child_process_ids() -> tuple[int, ...]:
    """Read the current OS child inventory without importing a process API."""

    process_id = os.getpid()
    children_path = Path(
        f"/proc/{process_id}/task/{process_id}/children"
    )
    try:
        raw = children_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ()
    result: set[int] = set()
    for item in raw.split():
        try:
            child_id = int(item)
        except ValueError:
            continue
        if child_id > 0:
            result.add(child_id)
    return tuple(sorted(result))


def capture_control_discovery_runtime_state(
    *,
    service_resolution_count: int = 0,
    optional_provider_load_count: int = 0,
    process_start_count: int = 0,
    optional_provider_prefixes: Sequence[
        str
    ] = CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES,
) -> ControlDiscoveryRuntimeState:
    """Capture a read-only discovery state for an independently instrumented run.

    Callers which intercept service resolution, optional-provider loading, or
    process starts pass their cumulative counters here.  Module and child
    inventories supplement those counters without importing a provider or a
    process-management package merely to inspect discovery.
    """

    prefixes = tuple(
        sorted(
            {
                str(item).strip()
                for item in optional_provider_prefixes
                if str(item).strip()
            }
        )
    )
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix)
                for prefix in prefixes
            )
        )
    )
    return ControlDiscoveryRuntimeState(
        optional_provider_modules=loaded,
        child_process_ids=_current_child_process_ids(),
        service_resolution_count=service_resolution_count,
        optional_provider_load_count=optional_provider_load_count,
        process_start_count=process_start_count,
    )


def _canonical_json_value(value: Any) -> Any:
    """Return a strict, bounded-contract-compatible JSON projection."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise ValueError("backend data contains a non-finite float")
        # Control contracts deliberately reject floats.  Stable decimal text
        # retains the observation without weakening their canonical format.
        return format(value, ".17g")
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical_json_value(item) for item in value]
    if is_dataclass(value):
        return _canonical_json_value(asdict(value))
    for method_name in ("to_record", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _canonical_json_value(method())
    raise ValueError(
        f"backend data contains unsupported value type {type(value).__name__}"
    )


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_control_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _normalized_absolute(value: Union[str, Path], *, label: str) -> Path:
    raw = os.fspath(value)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(os.path.normpath(raw))
    if path == Path(path.anchor):
        raise ValueError(f"{label} must not be the filesystem root")
    return path.resolve(strict=False)


def _normalize_allowlist(
    values: Iterable[Union[str, Path]], *, label: str
) -> tuple[Path, ...]:
    paths = {_normalized_absolute(item, label=label) for item in values}
    if not paths:
        raise ValueError(f"{label} must not be empty")
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_parameter(
    request: OperationRequest,
    *names: str,
    required: bool = True,
) -> str:
    for name in names:
        value = request.parameters.get(name)
        if value not in (None, ""):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a repository-relative string")
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise PathEscapeError(f"{name} must be repository-relative")
            return candidate.as_posix().removeprefix("./")
    if required:
        raise ValueError(f"one of {', '.join(names)} is required")
    return ""


def _bounded_window(request: OperationRequest) -> tuple[int, int]:
    raw_limit = request.parameters.get(
        "limit", min(DEFAULT_QUERY_LIMIT, request.bounds.max_items)
    )
    raw_offset = request.parameters.get("offset", 0)
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or raw_limit < 1
    ):
        raise ControlBoundsError("limit must be a positive integer")
    if raw_limit > request.bounds.max_items:
        raise ControlBoundsError("limit exceeds the request item bound")
    if (
        isinstance(raw_offset, bool)
        or not isinstance(raw_offset, int)
        or raw_offset < 0
    ):
        raise ControlBoundsError("offset must be a non-negative integer")
    if raw_offset > DEFAULT_MAX_OFFSET:
        raise ControlBoundsError("offset exceeds the absolute query bound")
    return raw_limit, raw_offset


@dataclass(frozen=True)
class BackendResponse:
    """Normalized return from a direct Python operation adapter.

    ``changed`` is required for mutating operations.  A mapping returned
    directly by a handler is accepted for compatibility and means that all
    declared effects were applied when the operation is a real mutation.
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    changed: bool = False
    applied_effect_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise TypeError("backend response data must be a mapping")
        if not isinstance(self.changed, bool):
            raise TypeError("backend response changed must be boolean")
        effect_ids = tuple(
            sorted({str(item).strip() for item in self.applied_effect_ids})
        )
        if any(not item for item in effect_ids):
            raise ValueError("applied effect IDs must not be empty")
        object.__setattr__(self, "applied_effect_ids", effect_ids)
        object.__setattr__(
            self,
            "warnings",
            tuple(sorted({str(item).strip() for item in self.warnings if str(item).strip()})),
        )
        object.__setattr__(
            self,
            "checks",
            tuple(sorted({str(item).strip() for item in self.checks if str(item).strip()})),
        )


OperationHandler = Callable[[OperationRequest], Union[BackendResponse, Mapping[str, Any], Any]]


class LeaseFenceValidator(Protocol):
    """Checks authoritative current lease state before mutation dispatch."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


class AuthorizationValidator(Protocol):
    """Optional live policy check in addition to the bound contract decision."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


class TargetIdentityValidator(Protocol):
    """Checks repository and tree identities against authoritative state."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


@dataclass(frozen=True)
class ControlAuditReceipt:
    """Content-addressed audit record for one service decision."""

    request_id: str
    operation: str
    authority: str
    status: str
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    caller: str
    authorization_decision_id: str
    grant_ids: tuple[str, ...]
    dry_run: bool
    idempotency_key: str
    lease_id: str
    fencing_epoch: Union[int, None]
    effect_ids: tuple[str, ...]
    applied_effect_ids: tuple[str, ...]
    error_code: str
    occurred_at_ms: int
    occurred_at: str
    receipt_id: str = ""

    def __post_init__(self) -> None:
        payload = self._payload()
        expected = _content_id(payload)
        if self.receipt_id and self.receipt_id != expected:
            raise ValueError("audit receipt identity does not match its payload")
        object.__setattr__(self, "receipt_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_AUDIT_RECEIPT_SCHEMA,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "authority": self.authority,
            "status": self.status,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "caller": self.caller,
            "authorization_decision_id": self.authorization_decision_id,
            "grant_ids": self.grant_ids,
            "dry_run": self.dry_run,
            "idempotency_key": self.idempotency_key,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "effect_ids": self.effect_ids,
            "applied_effect_ids": self.applied_effect_ids,
            "error_code": self.error_code,
            "occurred_at_ms": self.occurred_at_ms,
            "occurred_at": self.occurred_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}


class ControlStateStore(Protocol):
    """Persistence boundary for idempotency results and audit records."""

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        ...

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        ...

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        ...

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        ...


class InMemoryControlStateStore:
    """Thread-safe state store suitable for embedding and deterministic tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._idempotency: dict[str, tuple[str, OperationResult]] = {}
        self._receipts: list[dict[str, Any]] = []

    @staticmethod
    def _key(request: OperationRequest) -> str:
        return "\x1f".join(
            (
                request.state_root,
                request.repository_id,
                request.objective_id,
                request.caller,
                request.operation.value,
                request.idempotency_key,
            )
        )

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        with self._lock:
            return self._idempotency.get(self._key(request))

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        key = self._key(request)
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is not None and existing[0] != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to another request"
                )
            self._idempotency[key] = (request.request_id, result)

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        del request
        with self._lock:
            self._receipts.append(receipt.to_dict())

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        del request
        with self._lock:
            newest = list(reversed(self._receipts))
            return newest[offset : offset + limit]


class JsonlControlStateStore(InMemoryControlStateStore):
    """Durable JSONL audit and exact-result idempotency store.

    Records retain canonical :class:`OperationResult` payloads, so a restarted
    service can return the exact prior result without invoking a mutating
    backend again.  Multi-writer deployments may replace this with a database
    implementation of :class:`ControlStateStore` for transactional reservation
    across processes.
    """

    def __init__(
        self,
        filename: str = "control-audit.jsonl",
        idempotency_filename: str = "control-idempotency.jsonl",
    ) -> None:
        super().__init__()
        for value, label in (
            (filename, "control audit filename"),
            (idempotency_filename, "control idempotency filename"),
        ):
            if (
                not str(value).strip()
                or Path(value) == Path(".")
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(f"{label} must be relative")
        self._filename = filename
        self._idempotency_filename = idempotency_filename

    @staticmethod
    def _idempotency_record_key(request: OperationRequest) -> dict[str, str]:
        return {
            "repository_id": request.repository_id,
            "objective_id": request.objective_id,
            "caller": request.caller,
            "operation": request.operation.value,
            "idempotency_key": request.idempotency_key,
        }

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        cached = super().get_idempotent(request)
        if cached is not None or not request.idempotency_key:
            return cached
        path = Path(request.state_root) / self._idempotency_filename
        if not path.exists():
            return None
        expected = self._idempotency_record_key(request)
        matching: Union[Mapping[str, Any], None] = None
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(record, Mapping) and not any(
                            record.get(name) != value
                            for name, value in expected.items()
                        ):
                            matching = record
            except OSError as exc:
                raise IdempotencyConflictError(
                    "idempotency state is unreadable"
                ) from exc
        if matching is not None:
            raw_result = matching.get("result")
            try:
                if not isinstance(raw_result, Mapping):
                    raise ValueError("stored result is absent")
                result = OperationResult.from_dict(raw_result)
                request_id = str(matching.get("request_id") or "")
            except Exception as exc:
                raise IdempotencyConflictError(
                    "idempotency state contains an invalid matching result"
                ) from exc
            # Populate the fast path only after canonical decoding succeeds.
            with self._lock:
                self._idempotency[self._key(request)] = (request_id, result)
            return request_id, result
        return None

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        existing = self.get_idempotent(request)
        if existing is not None:
            if existing[0] != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to another request"
                )
            return
        path = Path(request.state_root) / self._idempotency_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "ipfs_accelerate_py/agent-supervisor/control-idempotency@1",
            **self._idempotency_record_key(request),
            "request_id": request.request_id,
            "result": result.to_record(),
        }
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        super().put_idempotent(request, result)

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        super().append_receipt(request, receipt)
        path = Path(request.state_root) / self._filename
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            receipt.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        path = Path(request.state_root) / self._filename
        if not path.exists():
            return super().query_receipts(
                request, limit=limit, offset=offset
            )
        from collections import deque

        records: Any = deque(maxlen=offset + limit)
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, Mapping):
                            records.append(item)
            except OSError:
                return ()
        newest = list(reversed(records))
        return newest[offset : offset + limit]


class RepositorySupervisorBackend:
    """Direct package-API backend for bounded supervisor inspection.

    Paths are always explicit repository- or state-relative parameters.  This
    avoids unsafe discovery defaults and lets the service validate every path
    against the selected allowlisted root before any package API is called.
    Mutations and domain-specific proposal operations are supplied as Python
    handlers by the embedding runtime.
    """

    def __init__(
        self,
        handlers: Union[Mapping[Union[Operation, str], OperationHandler], None] = None,
    ) -> None:
        normalized: dict[Operation, OperationHandler] = {}
        for name, handler in dict(handlers or {}).items():
            operation = name if isinstance(name, Operation) else Operation(str(name))
            if not callable(handler):
                raise TypeError(f"handler for {operation.value} must be callable")
            normalized[operation] = handler
        self._handlers = MappingProxyType(normalized)

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        builtins = set(READ_OPERATIONS)
        return tuple(sorted(builtins | set(self._handlers), key=lambda item: item.value))

    def _resolve(
        self,
        request: OperationRequest,
        relative: str,
        *,
        state: bool,
    ) -> Path:
        root = Path(request.state_root if state else request.repository_root).resolve()
        resolved = (root / relative).resolve(strict=False)
        if not _under(resolved, root):
            raise PathEscapeError("requested path escapes its selected root")
        return resolved

    @staticmethod
    def _read_text(path: Path, *, maximum_bytes: int) -> str:
        if not path.exists():
            raise BackendNotFoundError(f"control data not found: {path.name}")
        if not path.is_file():
            raise ValueError(f"control data path is not a file: {path.name}")
        if path.stat().st_size > maximum_bytes:
            raise ControlBoundsError(
                f"control data exceeds the {maximum_bytes}-byte request bound"
            )
        return path.read_text(encoding="utf-8")

    @classmethod
    def _read_json(cls, path: Path, *, maximum_bytes: int) -> Any:
        return json.loads(
            cls._read_text(path, maximum_bytes=maximum_bytes)
        )

    @staticmethod
    def _window(items: Sequence[Any], request: OperationRequest) -> dict[str, Any]:
        limit, offset = _bounded_window(request)
        selected = list(items[offset : offset + limit])
        return {
            "items": selected,
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(selected) < len(items),
        }

    def _json_document(
        self,
        request: OperationRequest,
        names: Sequence[str],
        *,
        state: bool = True,
    ) -> Mapping[str, Any]:
        relative = _relative_parameter(request, *names)
        value = self._read_json(
            self._resolve(request, relative, state=state),
            maximum_bytes=request.bounds.max_serialized_bytes,
        )
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._window(value, request)
        return {"value": value}

    def _goals(self, request: OperationRequest) -> Mapping[str, Any]:
        from .objective_graph import parse_goal_heap

        relative = _relative_parameter(request, "objective_path", "path")
        path = self._resolve(request, relative, state=False)
        if not path.exists():
            raise BackendNotFoundError(f"objective heap not found: {relative}")
        text = self._read_text(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        goals = [
            {
                "goal_id": item.goal_id,
                "title": item.title,
                "status": item.status,
                "fields": dict(item.fields),
            }
            for item in parse_goal_heap(text)
        ]
        return self._window(goals, request)

    def _tasks(self, request: OperationRequest) -> Mapping[str, Any]:
        from .todo_vector_index import parse_todo_blocks

        relative = _relative_parameter(request, "todo_path", "path")
        path = self._resolve(request, relative, state=False)
        if not path.exists():
            raise BackendNotFoundError(f"task board not found: {relative}")
        prefix = request.parameters.get("task_header_prefix", "")
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("task_header_prefix must be a non-empty string")
        text = self._read_text(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        tasks = [
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "fields": fields,
                "status": str(fields.get("status") or "todo").strip().lower(),
            }
            for task_id, title, source_line, fields in parse_todo_blocks(
                text,
                task_header_prefix=prefix,
            )
        ]
        return self._window(tasks, request)

    def _events(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(request, "events_path", "path")
        path = self._resolve(request, relative, state=True)
        return self._jsonl_window(
            path,
            request,
            newest_first=bool(request.parameters.get("newest_first", False)),
        )

    @staticmethod
    def _jsonl_window(
        path: Path,
        request: OperationRequest,
        *,
        newest_first: bool = False,
    ) -> Mapping[str, Any]:
        limit, offset = _bounded_window(request)
        if not path.exists():
            return {
                "items": [],
                "count": 0,
                "offset": offset,
                "limit": limit,
                "truncated": False,
            }
        if not path.is_file():
            raise ValueError(f"JSONL path is not a file: {path.name}")
        needed = offset + limit + 1
        if newest_first:
            from collections import deque

            retained: Any = deque(maxlen=needed)
        else:
            retained = []
        valid_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                if len(raw_line.encode("utf-8")) > request.bounds.max_text_bytes:
                    raise ControlBoundsError(
                        "JSONL record exceeds the request text bound"
                    )
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, Mapping):
                    continue
                if newest_first:
                    retained.append(dict(value))
                elif valid_count >= offset:
                    retained.append(dict(value))
                    if len(retained) >= limit + 1:
                        break
                valid_count += 1
        if newest_first:
            values = list(reversed(retained))
            selected = values[offset : offset + limit]
            truncated = len(values) > offset + limit
        else:
            selected = retained[:limit]
            truncated = len(retained) > limit
        return {
            "items": selected,
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
        }

    def _receipts(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(
            request, "receipts_path", "path", required=False
        )
        if not relative:
            raise OperationUnavailableError(
                "receipt reads are served by the control state store"
            )
        path = self._resolve(request, relative, state=True)
        if path.is_dir():
            limit, offset = _bounded_window(request)
            candidates = heapq.nsmallest(
                offset + limit + 1,
                (
                    item
                    for item in path.rglob("*.json")
                    if item.is_file() and _under(item.resolve(), path.resolve())
                ),
                key=lambda item: item.as_posix(),
            )
            selected = candidates[offset : offset + limit]
            records = []
            for item in selected:
                try:
                    value = self._read_json(
                        item,
                        maximum_bytes=request.bounds.max_serialized_bytes,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                records.append(
                    {
                        "path": item.relative_to(path).as_posix(),
                        "receipt": value,
                    }
                )
            return {
                "items": records,
                "count": len(records),
                "offset": offset,
                "limit": limit,
                "truncated": offset + len(selected) < len(candidates),
            }
        if path.suffix == ".jsonl":
            return self._jsonl_window(path, request)
        value = self._read_json(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        values = value if isinstance(value, list) else [value]
        return self._window(values, request)

    def _cache(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(request, "cache_path", "path")
        path = self._resolve(request, relative, state=True)
        if not path.exists():
            raise BackendNotFoundError(f"cache path not found: {relative}")
        limit, offset = _bounded_window(request)
        if path.is_file():
            value = self._read_json(
                path, maximum_bytes=request.bounds.max_serialized_bytes
            )
            return {"path": relative, "kind": "file", "value": value}
        entries = heapq.nsmallest(
            offset + limit + 1,
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
        selected = entries[offset : offset + limit]
        return {
            "path": relative,
            "kind": "directory",
            "entries": [
                {
                    "path": item.relative_to(path).as_posix(),
                    "size_bytes": item.stat().st_size,
                }
                for item in selected
            ],
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(selected) < len(entries),
        }

    def _artifact(self, request: OperationRequest) -> Mapping[str, Any]:
        from .artifact_store import query_artifact

        relative = _relative_parameter(request, "artifact_path", "path")
        root_name = str(request.parameters.get("root") or "state")
        if root_name not in {"repository", "state"}:
            raise ValueError("artifact root must be 'repository' or 'state'")
        state = root_name == "state"
        path = self._resolve(request, relative, state=state)
        limit, _offset = _bounded_window(request)
        columns = request.parameters.get("columns", ("*",))
        if isinstance(columns, str):
            columns = (columns,)
        if not isinstance(columns, Sequence):
            raise ValueError("columns must be a sequence")
        if len(columns) > request.bounds.max_items:
            raise ControlBoundsError("columns exceed the request item bound")
        sql = str(request.parameters.get("sql") or "").strip()
        if sql:
            raise ValueError(
                "raw SQL is disabled at the supervisor control boundary"
            )
        where = str(request.parameters.get("where") or "").strip()
        folded_where = where.lower()
        if (
            ";" in where
            or "--" in where
            or "/*" in where
            or re.search(
                r"\b(select|from|join|attach|copy|pragma|install|load|read_\w+)\b",
                folded_where,
            )
        ):
            raise ValueError(
                "where must be a simple expression without subqueries or I/O"
            )
        return query_artifact(
            path,
            table=str(request.parameters.get("table") or "") or None,
            columns=tuple(str(item) for item in columns),
            where=where,
            sql="",
            limit=limit,
            kind=str(request.parameters.get("kind") or "") or None,
        )

    def execute(self, request: OperationRequest) -> Union[BackendResponse, Mapping[str, Any], Any]:
        handler = self._handlers.get(request.operation)
        if handler is not None:
            return handler(request)
        if request.operation in {
            Operation.STATUS,
            Operation.HEALTH,
            Operation.METRICS,
            Operation.BUNDLES,
            Operation.LANES,
        }:
            names = {
                Operation.STATUS: ("status_path", "path"),
                Operation.HEALTH: ("health_path", "path"),
                Operation.METRICS: ("metrics_path", "path"),
                Operation.BUNDLES: ("bundle_index_path", "path"),
                Operation.LANES: ("lane_manifest_path", "path"),
            }[request.operation]
            return self._json_document(request, names)
        if request.operation is Operation.GOALS:
            return self._goals(request)
        if request.operation is Operation.TASKS:
            return self._tasks(request)
        if request.operation is Operation.EVENTS:
            return self._events(request)
        if request.operation is Operation.RECEIPTS:
            return self._receipts(request)
        if request.operation is Operation.CACHE_INSPECT:
            return self._cache(request)
        if request.operation is Operation.ARTIFACT_QUERY:
            return self._artifact(request)
        raise OperationUnavailableError(
            f"operation {request.operation.value} has no direct Python adapter"
        )


@dataclass(frozen=True)
class SupervisorTarget:
    """Binding used by :class:`SupervisorClient` to construct read requests."""

    repository_root: str
    state_root: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str

    def request(
        self,
        operation: Union[Operation, str],
        *,
        parameters: Union[Mapping[str, Any], None] = None,
        bounds: Union[ControlBounds, None] = None,
        dry_run: bool = False,
        expected_effects: Sequence[ExpectedEffect] = (),
    ) -> OperationRequest:
        selected = operation if isinstance(operation, Operation) else Operation(operation)
        if selected in MUTATION_OPERATIONS and not dry_run:
            raise AuthorizationBindingError(
                "SupervisorTarget constructs only read, proposal, or dry-run requests"
            )
        return OperationRequest(
            operation=selected,
            repository_root=self.repository_root,
            state_root=self.state_root,
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            objective_id=self.objective_id,
            objective_revision=self.objective_revision,
            policy_id=self.policy_id,
            policy_revision=self.policy_revision,
            caller=self.caller,
            parameters=dict(parameters or {}),
            bounds=bounds or ControlBounds(),
            dry_run=dry_run,
            expected_effects=tuple(expected_effects),
        )


class SupervisorControlService:
    """Typed policy and dispatch boundary shared by Python, CLI, and MCP."""

    def __init__(
        self,
        *,
        repository_allowlist: Union[Iterable[Union[str, Path]], None] = None,
        state_allowlist: Union[Iterable[Union[str, Path]], None] = None,
        allowed_repository_roots: Union[Iterable[Union[str, Path]], None] = None,
        allowed_state_roots: Union[Iterable[Union[str, Path]], None] = None,
        backend: Union[RepositorySupervisorBackend, Any, None] = None,
        handlers: Union[Mapping[Union[Operation, str], OperationHandler], None] = None,
        lease_validator: Union[LeaseFenceValidator, Callable[[OperationRequest], Any], None] = None,
        authorization_validator: Union[
            AuthorizationValidator, Callable[[OperationRequest], Any], None
        ] = None,
        identity_validator: Union[
            TargetIdentityValidator, Callable[[OperationRequest], Any], None
        ] = None,
        state_store: Union[ControlStateStore, None] = None,
        service_id: str = "ipfs-accelerate-agent-supervisor",
        service_version: str = CONTROL_SERVICE_VERSION,
        max_query_items: int = DEFAULT_MAX_QUERY_ITEMS,
        require_lease_validator: bool = True,
        clock_ms: Callable[[], int] = _now_ms,
    ) -> None:
        repositories = (
            repository_allowlist
            if repository_allowlist is not None
            else allowed_repository_roots
        )
        states = (
            state_allowlist if state_allowlist is not None else allowed_state_roots
        )
        if repositories is None or states is None:
            raise ValueError(
                "explicit repository and state root allowlists are required"
            )
        self._repository_roots = _normalize_allowlist(
            repositories, label="repository allowlist"
        )
        self._state_roots = _normalize_allowlist(states, label="state allowlist")
        if backend is not None and handlers:
            raise ValueError("supply backend or handlers, not both")
        self._backend = backend or RepositorySupervisorBackend(handlers)
        self._lease_validator = lease_validator
        self._authorization_validator = authorization_validator
        self._identity_validator = identity_validator
        self._state_store = state_store or JsonlControlStateStore()
        self._service_id = str(service_id).strip()
        self._service_version = str(service_version).strip()
        if not self._service_id or not self._service_version:
            raise ValueError("service_id and service_version must not be empty")
        if (
            isinstance(max_query_items, bool)
            or not isinstance(max_query_items, int)
            or max_query_items < 1
        ):
            raise ValueError("max_query_items must be a positive integer")
        self._max_query_items = min(max_query_items, DEFAULT_MAX_QUERY_ITEMS)
        self._require_lease_validator = bool(require_lease_validator)
        self._clock_ms = clock_ms
        self._lock = threading.RLock()
        self._capability_report = self._build_capability_report()

    @property
    def repository_allowlist(self) -> tuple[str, ...]:
        return tuple(item.as_posix() for item in self._repository_roots)

    @property
    def state_allowlist(self) -> tuple[str, ...]:
        return tuple(item.as_posix() for item in self._state_roots)

    def _build_capability_report(self) -> CapabilityReport:
        bounds = ControlBounds(
            max_items=self._max_query_items,
            max_paths=min(128, self._max_query_items),
            max_effects=min(64, self._max_query_items),
        )
        registered = getattr(self._backend, "registered_operations", None)
        if registered is None:
            supported = set(Operation)
        else:
            supported = {
                item if isinstance(item, Operation) else Operation(str(item))
                for item in registered
            }
        # These two reads are implemented by the service itself.
        supported.update({Operation.CAPABILITIES, Operation.RECEIPTS})
        capabilities = tuple(
            OperationCapability(
                operation=operation,
                authority=operation.authority,
                bounds=bounds,
                supports_dry_run=operation in MUTATION_OPERATIONS,
                requires_idempotency=operation in MUTATION_OPERATIONS,
                requires_authorization=operation in MUTATION_OPERATIONS,
            )
            for operation in sorted(supported, key=lambda item: item.value)
        )
        return CapabilityReport(
            service_id=self._service_id,
            service_version=self._service_version,
            capabilities=capabilities,
            optional_providers_loaded=bool(
                getattr(self._backend, "optional_providers_loaded", False)
            ),
            processes_started=bool(
                getattr(self._backend, "processes_started", False)
            ),
        )

    def capability_report(self) -> CapabilityReport:
        """Return the side-effect-free capability handshake."""

        return self._capability_report

    def discovery_manifest(self) -> ControlDiscoveryManifest:
        """Return deterministic Python discovery metadata without dispatch."""

        expected = tuple(sorted(Operation, key=lambda item: item.value))
        if self._capability_report.supported_operations != expected:
            raise OperationUnavailableError(
                "Python discovery requires the complete control vocabulary"
            )
        return ControlDiscoveryManifest(surface=ControlSurface.PYTHON)

    def client(
        self,
        target: Union[SupervisorTarget, Mapping[str, Any], None] = None,
        **binding: Any,
    ) -> "SupervisorClient":
        return SupervisorClient(self, target=target, **binding)

    def _check_target(self, request: OperationRequest) -> None:
        repository = _normalized_absolute(
            request.repository_root, label="repository_root"
        )
        state = _normalized_absolute(request.state_root, label="state_root")
        if repository not in self._repository_roots:
            raise TargetNotAllowedError("repository_root is not allowlisted")
        if state not in self._state_roots:
            raise TargetNotAllowedError("state_root is not allowlisted")
        if self._identity_validator is not None:
            self._invoke_validator(
                self._identity_validator,
                request,
                denial=StaleTreeError,
            )

    def _check_bounds(self, request: OperationRequest) -> None:
        limit, _offset = _bounded_window(request)
        if (
            "limit" in request.parameters
            and limit > self._max_query_items
        ):
            raise ControlBoundsError("limit exceeds the service query bound")

    @staticmethod
    def _invoke_validator(
        validator: Any,
        request: OperationRequest,
        *,
        denial: type[Exception],
    ) -> None:
        method = getattr(validator, "validate", None)
        if not callable(method):
            method = getattr(validator, "authorize", None)
        result = method(request) if callable(method) else validator(request)
        if result is False:
            raise denial("validator denied the bound request")

    def _check_authorization(self, request: OperationRequest) -> None:
        decision = request.authorization
        if decision is not None:
            now = self._clock_ms()
            if decision.evaluated_at_ms > now:
                raise AuthorizationBindingError(
                    "authorization decision is not yet valid"
                )
            if decision.expires_at_ms is not None and now >= decision.expires_at_ms:
                raise AuthorizationBindingError(
                    "authorization decision has expired"
                )
        if self._authorization_validator is not None:
            self._invoke_validator(
                self._authorization_validator,
                request,
                denial=AuthorizationBindingError,
            )

    def _check_lease(self, request: OperationRequest) -> None:
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            return
        if self._lease_validator is None:
            if self._require_lease_validator:
                raise LeaseValidationError(
                    "a live lease/fencing validator is required for mutation"
                )
            return
        try:
            self._invoke_validator(
                self._lease_validator,
                request,
                denial=StaleLeaseError,
            )
        except StaleLeaseError:
            raise
        except LeaseValidationError:
            raise
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "stale" in name or "expired" in name or "fenc" in name:
                raise StaleLeaseError(str(exc) or "lease is stale") from exc
            raise LeaseValidationError(str(exc) or "lease validation failed") from exc

    def _check_idempotency(
        self, request: OperationRequest
    ) -> Union[OperationResult, None]:
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            return None
        existing = self._state_store.get_idempotent(request)
        if existing is None:
            return None
        request_id, result = existing
        if request_id != request.request_id:
            raise IdempotencyConflictError(
                "idempotency key is already bound to another request"
            )
        result.validate_against(request)
        return result

    @staticmethod
    def _normalize_backend_response(value: Any, request: OperationRequest) -> BackendResponse:
        if isinstance(value, BackendResponse):
            response = value
        elif isinstance(value, Mapping):
            response = BackendResponse(
                data=value,
                changed=request.operation in MUTATION_OPERATIONS
                and not request.dry_run,
                applied_effect_ids=tuple(
                    item.effect_id for item in request.expected_effects
                )
                if request.operation in MUTATION_OPERATIONS
                and not request.dry_run
                else (),
            )
        else:
            response = BackendResponse(data={"result": _canonical_json_value(value)})
        declared = {item.effect_id for item in request.expected_effects}
        if not set(response.applied_effect_ids).issubset(declared):
            raise ControlContractError(
                "backend claimed an effect not declared by the request"
            )
        if response.applied_effect_ids and not response.changed:
            raise ControlContractError(
                "backend cannot apply effects while reporting no change"
            )
        return BackendResponse(
            data=_canonical_json_value(response.data),
            changed=response.changed,
            applied_effect_ids=response.applied_effect_ids,
            warnings=response.warnings,
            checks=response.checks,
        )

    def _dispatch(self, request: OperationRequest) -> BackendResponse:
        if request.operation is Operation.CAPABILITIES:
            return BackendResponse(data={"report": self.capability_report().to_record()})
        if request.operation is Operation.RECEIPTS and not any(
            request.parameters.get(name)
            for name in ("receipts_path", "path")
        ):
            limit, offset = _bounded_window(request)
            items = self._state_store.query_receipts(
                request, limit=limit, offset=offset
            )
            return BackendResponse(
                data={
                    "items": list(items),
                    "count": len(items),
                    "limit": limit,
                    "offset": offset,
                    "truncated": len(items) == limit,
                }
            )
        execute = getattr(self._backend, "execute", None)
        if not callable(execute):
            raise OperationUnavailableError(
                "control backend does not provide execute(request)"
            )
        return self._normalize_backend_response(execute(request), request)

    @staticmethod
    def _status_for_error(code: ErrorCode) -> OperationStatus:
        if code in {ErrorCode.UNAUTHORIZED, ErrorCode.FORBIDDEN}:
            return OperationStatus.DENIED
        if code is ErrorCode.NOT_FOUND:
            return OperationStatus.NOT_FOUND
        if code in {
            ErrorCode.CONFLICT,
            ErrorCode.STALE_TREE,
            ErrorCode.STALE_LEASE,
            ErrorCode.IDEMPOTENCY_CONFLICT,
            ErrorCode.INVALID_LIFECYCLE_TRANSITION,
        }:
            return OperationStatus.CONFLICT
        if code is ErrorCode.UNAVAILABLE:
            return OperationStatus.UNAVAILABLE
        if code is ErrorCode.TIMED_OUT:
            return OperationStatus.TIMED_OUT
        if code is ErrorCode.CANCELLED:
            return OperationStatus.CANCELLED
        return OperationStatus.FAILED

    @staticmethod
    def _stable_error(exc: BaseException) -> OperationError:
        message = str(exc).strip() or type(exc).__name__
        field = ""
        retryable = False
        if isinstance(exc, TargetNotAllowedError):
            code = ErrorCode.FORBIDDEN
            field = "repository_root" if "repository" in message else "state_root"
        elif isinstance(exc, AuthorizationBindingError):
            code = ErrorCode.UNAUTHORIZED
        elif isinstance(exc, StaleLeaseError):
            code = ErrorCode.STALE_LEASE
            retryable = True
        elif isinstance(exc, StaleTreeError):
            code = ErrorCode.STALE_TREE
            retryable = True
        elif isinstance(exc, LeaseValidationError):
            code = ErrorCode.STALE_LEASE
            retryable = True
        elif isinstance(exc, IdempotencyConflictError):
            code = ErrorCode.IDEMPOTENCY_CONFLICT
        elif isinstance(exc, BackendNotFoundError) or isinstance(
            exc, FileNotFoundError
        ):
            code = ErrorCode.NOT_FOUND
        elif isinstance(exc, BackendConflictError):
            code = ErrorCode.CONFLICT
        elif isinstance(exc, BackendCancelledError):
            code = ErrorCode.CANCELLED
        elif isinstance(exc, (BackendTimeoutError, TimeoutError)):
            code = ErrorCode.TIMED_OUT
            retryable = True
        elif isinstance(exc, OperationUnavailableError):
            code = ErrorCode.UNAVAILABLE
        elif isinstance(exc, PermissionError):
            code = ErrorCode.FORBIDDEN
        elif isinstance(exc, PathEscapeError):
            code = ErrorCode.PATH_ESCAPE
            field = "path"
        elif isinstance(exc, ControlBoundsError):
            code = ErrorCode.BOUNDS_EXCEEDED
        elif isinstance(exc, (ControlContractError, ValueError, TypeError, json.JSONDecodeError)):
            code = ErrorCode.INVALID_REQUEST
        else:
            code = ErrorCode.INTERNAL_ERROR
            message = "control operation failed"
        return OperationError(
            code=code,
            message=message[:2048],
            retryable=retryable,
            field=field,
            details={"exception_type": type(exc).__name__},
        )

    def _receipt(
        self,
        request: OperationRequest,
        *,
        status: OperationStatus,
        applied_effect_ids: Iterable[str] = (),
        error: Union[OperationError, None] = None,
    ) -> ControlAuditReceipt:
        now = self._clock_ms()
        authorization = request.authorization
        return ControlAuditReceipt(
            request_id=request.request_id,
            operation=request.operation.value,
            authority=request.effective_authority.value,
            status=status.value,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            authorization_decision_id=(
                authorization.decision_id if authorization is not None else ""
            ),
            grant_ids=(
                authorization.grant_ids if authorization is not None else ()
            ),
            dry_run=request.dry_run,
            idempotency_key=request.idempotency_key,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch,
            effect_ids=tuple(item.effect_id for item in request.expected_effects),
            applied_effect_ids=tuple(sorted(set(applied_effect_ids))),
            error_code=error.code.value if error else "",
            occurred_at_ms=now,
            occurred_at=_utc_timestamp(now),
        )

    @staticmethod
    def _claims(
        request: OperationRequest,
        applied_effect_ids: Iterable[str],
        receipt_id: str,
    ) -> tuple[EffectClaim, ...]:
        applied = set(applied_effect_ids)
        return tuple(
            EffectClaim(
                effect_id=effect.effect_id,
                kind=effect.kind,
                resource=effect.resource,
                paths=effect.paths,
                applied=effect.effect_id in applied,
                receipt_id=receipt_id if effect.effect_id in applied else "",
            )
            for effect in request.expected_effects
        )

    @staticmethod
    def _preview(
        request: OperationRequest,
        response: Union[BackendResponse, None] = None,
    ) -> DryRunPreview:
        return DryRunPreview(
            request_id=request.request_id,
            operation=request.operation,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            expected_effects=request.expected_effects,
            checks=(response.checks if response else ("authorization", "bounds", "allowlists")),
            warnings=response.warnings if response else (),
            would_change=bool(request.expected_effects)
            if response is None
            else response.changed,
        )

    def _success_result(
        self, request: OperationRequest, response: BackendResponse
    ) -> OperationResult:
        applied = (
            response.applied_effect_ids
            if request.operation in MUTATION_OPERATIONS and not request.dry_run
            else ()
        )
        receipt = self._receipt(
            request,
            status=OperationStatus.SUCCEEDED,
            applied_effect_ids=applied,
        )
        preview = None
        authority = request.effective_authority
        if request.dry_run and request.operation in MUTATION_OPERATIONS:
            preview = self._preview(request, response)
            authority = OperationAuthority.PROPOSAL
        elif request.operation is Operation.OBJECTIVE_PREVIEW:
            preview = self._preview(request, response)
        result = OperationResult(
            request_id=request.request_id,
            operation=request.operation,
            authority=authority,
            status=OperationStatus.SUCCEEDED,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            bounds=request.bounds,
            data=response.data,
            # Mutation-shaped expected effects belong in the proposal preview
            # during dry-run.  Even an unapplied mutation EffectClaim would
            # exceed the result's proposal-only authority.
            effects=(
                ()
                if request.dry_run and request.operation in MUTATION_OPERATIONS
                else self._claims(request, applied, receipt.receipt_id)
            ),
            preview=preview,
            idempotency_key=request.idempotency_key,
            audit_receipt_id=receipt.receipt_id,
        )
        result.validate_against(request)
        self._state_store.append_receipt(request, receipt)
        if request.operation in MUTATION_OPERATIONS and not request.dry_run:
            self._state_store.put_idempotent(request, result)
        return result

    def _error_result(
        self, request: OperationRequest, exc: BaseException
    ) -> OperationResult:
        error = self._stable_error(exc)
        status = self._status_for_error(error.code)
        receipt = self._receipt(request, status=status, error=error)
        result = OperationResult(
            request_id=request.request_id,
            operation=request.operation,
            authority=request.effective_authority,
            status=status,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            bounds=request.bounds,
            data={},
            error=error,
            idempotency_key=request.idempotency_key,
            audit_receipt_id=receipt.receipt_id,
        )
        result.validate_against(request)
        try:
            self._state_store.append_receipt(request, receipt)
        except Exception:
            # Stable operation errors must never be replaced by an audit sink
            # failure.  Successful operations deliberately fail if durable
            # auditing fails, because reporting an unaudited mutation as
            # success would violate the control boundary.
            pass
        return result

    def execute(
        self, request: Union[OperationRequest, Mapping[str, Any]]
    ) -> OperationResult:
        """Validate, dispatch, audit, and return one typed operation result."""

        if not isinstance(request, OperationRequest):
            request = OperationRequest.from_dict(request)
        with self._lock:
            try:
                self._check_target(request)
                self._check_bounds(request)
                self._check_authorization(request)
                replay = self._check_idempotency(request)
                if replay is not None:
                    return replay
                self._check_lease(request)
                if request.dry_run and request.operation in MUTATION_OPERATIONS:
                    # A dry run never invokes a mutating adapter.
                    response = BackendResponse(
                        data={"dry_run": True, "would_change": bool(request.expected_effects)},
                        changed=bool(request.expected_effects),
                        checks=("authorization", "bounds", "allowlists", "expected_effects"),
                    )
                else:
                    response = self._dispatch(request)
                return self._success_result(request, response)
            except Exception as exc:
                return self._error_result(request, exc)

    handle = execute
    dispatch = execute

    def _operation(
        self, operation: Operation, request: OperationRequest
    ) -> OperationResult:
        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        if request.operation is not operation:
            raise ValueError(
                f"request operation must be {operation.value}, got {request.operation.value}"
            )
        return self.execute(request)

    def capabilities(
        self, request: Union[OperationRequest, None] = None
    ) -> Union[CapabilityReport, OperationResult]:
        return (
            self.capability_report()
            if request is None
            else self._operation(Operation.CAPABILITIES, request)
        )

    def status(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.STATUS, request)

    def health(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.HEALTH, request)

    def metrics(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.METRICS, request)

    def goals(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.GOALS, request)

    def tasks(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.TASKS, request)

    def bundles(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.BUNDLES, request)

    def lanes(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.LANES, request)

    def events(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.EVENTS, request)

    def receipts(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RECEIPTS, request)

    def cache_inspect(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.CACHE_INSPECT, request)

    cache = cache_inspect

    def artifact_query(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.ARTIFACT_QUERY, request)

    def objective_preview(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_PREVIEW, request)

    preview = objective_preview

    def objective_refine(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_REFINE, request)

    refine = objective_refine

    def objective_reconcile(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_RECONCILE, request)

    reconcile = objective_reconcile

    def backlog_refill(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.BACKLOG_REFILL, request)

    refill = backlog_refill

    def plan(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.PLAN, request)

    def start(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.START, request)

    def pause(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.PAUSE, request)

    def resume(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RESUME, request)

    def drain(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.DRAIN, request)

    def stop(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.STOP, request)

    def retry(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RETRY, request)

    def cancel(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.CANCEL, request)

    def quarantine(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.QUARANTINE, request)

    def validation_replay(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.VALIDATION_REPLAY, request)

    replay_validation = validation_replay

    def lifecycle(
        self,
        request: OperationRequest,
        command: Union[LifecycleCommand, None] = None,
    ) -> OperationResult:
        if command is not None:
            if request.operation is not command.operation:
                raise ValueError("lifecycle command does not match request operation")
            if request.dry_run != command.dry_run:
                raise ValueError("lifecycle command dry_run does not match request")
            target = str(request.parameters.get("target_id") or "")
            if target != command.target_id:
                raise ValueError("lifecycle command target does not match request")
            reason = str(request.parameters.get("reason") or "")
            if reason != command.reason:
                raise ValueError("lifecycle command reason does not match request")
            requested_state = str(
                request.parameters.get("requested_state") or ""
            )
            if requested_state != command.requested_state:
                raise ValueError(
                    "lifecycle command requested_state does not match request"
                )
        if request.operation not in {
            action.operation for action in LifecycleAction
        }:
            raise ValueError("request is not a lifecycle operation")
        return self.execute(request)


class SupervisorClient:
    """Read-oriented facade over :class:`SupervisorControlService`.

    A target binding lets the facade construct read and proposal requests.
    Mutations are exposed only for callers which already hold a fully formed
    :class:`OperationRequest`; the client never manufactures authorization,
    idempotency, leases, or fencing epochs.
    """

    def __init__(
        self,
        service: SupervisorControlService,
        target: Union[SupervisorTarget, Mapping[str, Any], None] = None,
        *,
        bounds: Union[ControlBounds, None] = None,
        **binding: Any,
    ) -> None:
        if not isinstance(service, SupervisorControlService):
            raise TypeError("service must be a SupervisorControlService")
        if target is not None and binding:
            raise ValueError("supply target or binding fields, not both")
        if isinstance(target, Mapping):
            target = SupervisorTarget(**dict(target))
        elif target is None and binding:
            target = SupervisorTarget(**binding)
        elif target is not None and not isinstance(target, SupervisorTarget):
            raise TypeError("target must be a SupervisorTarget or mapping")
        self._service = service
        self._target = target
        self._bounds = bounds or ControlBounds()

    @property
    def service(self) -> SupervisorControlService:
        return self._service

    @property
    def target(self) -> Union[SupervisorTarget, None]:
        return self._target

    def capabilities(self) -> CapabilityReport:
        return self._service.capability_report()

    def execute(self, request: OperationRequest) -> OperationResult:
        return self._service.execute(request)

    def _read(
        self,
        operation: Operation,
        parameters: Union[Mapping[str, Any], None] = None,
        **values: Any,
    ) -> OperationResult:
        if self._target is None:
            raise ValueError("a SupervisorTarget is required to construct requests")
        merged = dict(parameters or {})
        overlap = set(merged).intersection(values)
        if overlap:
            raise ValueError(
                "duplicate request parameters: " + ", ".join(sorted(overlap))
            )
        merged.update(values)
        request = self._target.request(
            operation, parameters=merged, bounds=self._bounds
        )
        return self._service.execute(request)

    def _authorized(
        self, operation: Operation, request: OperationRequest
    ) -> OperationResult:
        return self._service._operation(operation, request)

    def status(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.STATUS, parameters, **values)

    def health(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.HEALTH, parameters, **values)

    def metrics(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.METRICS, parameters, **values)

    def goals(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.GOALS, parameters, **values)

    def tasks(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.TASKS, parameters, **values)

    def bundles(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.BUNDLES, parameters, **values)

    def lanes(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.LANES, parameters, **values)

    def events(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.EVENTS, parameters, **values)

    def receipts(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.RECEIPTS, parameters, **values)

    def cache_inspect(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.CACHE_INSPECT, parameters, **values)

    cache = cache_inspect

    def artifact_query(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.ARTIFACT_QUERY, parameters, **values)

    def objective_preview(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.OBJECTIVE_PREVIEW, parameters, **values)

    preview = objective_preview

    def plan(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.PLAN, parameters, **values)

    def objective_refine(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.OBJECTIVE_REFINE, request)

    refine = objective_refine

    def objective_reconcile(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.OBJECTIVE_RECONCILE, request)

    reconcile = objective_reconcile

    def backlog_refill(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.BACKLOG_REFILL, request)

    refill = backlog_refill

    def start(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.START, request)

    def pause(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.PAUSE, request)

    def resume(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RESUME, request)

    def drain(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.DRAIN, request)

    def stop(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.STOP, request)

    def retry(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RETRY, request)

    def cancel(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.CANCEL, request)

    def quarantine(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.QUARANTINE, request)

    def validation_replay(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.VALIDATION_REPLAY, request)

    replay_validation = validation_replay

    def lifecycle(
        self,
        request: OperationRequest,
        command: Union[LifecycleCommand, None] = None,
    ) -> OperationResult:
        return self._service.lifecycle(request, command)


ReadOnlySupervisorClient = SupervisorClient
SupervisorReadClient = SupervisorClient
PythonSupervisorBackend = RepositorySupervisorBackend
ControlService = SupervisorControlService


__all__ = [
    "CONTROL_AUDIT_RECEIPT_SCHEMA",
    "CONTROL_BACKEND_RESPONSE_SCHEMA",
    "CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES",
    "CONTROL_SERVICE_VERSION",
    "BackendCancelledError",
    "BackendConflictError",
    "BackendNotFoundError",
    "BackendResponse",
    "BackendTimeoutError",
    "ControlAuditReceipt",
    "ControlService",
    "ControlStateStore",
    "IdempotencyConflictError",
    "InMemoryControlStateStore",
    "JsonlControlStateStore",
    "LeaseFenceValidator",
    "LeaseValidationError",
    "OperationHandler",
    "OperationUnavailableError",
    "PythonSupervisorBackend",
    "ReadOnlySupervisorClient",
    "RepositorySupervisorBackend",
    "StaleLeaseError",
    "StaleTreeError",
    "SupervisorClient",
    "SupervisorControlError",
    "SupervisorControlService",
    "SupervisorReadClient",
    "SupervisorTarget",
    "TargetNotAllowedError",
    "TargetIdentityValidator",
    "capture_control_discovery_runtime_state",
]
