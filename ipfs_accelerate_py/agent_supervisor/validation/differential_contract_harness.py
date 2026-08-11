"""Hermetic differential contract harness for profiled public surfaces.

The harness deliberately knows very little about any concrete product domain.
A surface adapter receives a canonical contract case and an isolated fixture.
Adapters for different transport families stay small while comparison,
provenance, and cleanup semantics remain identical.

Domain fixtures, invariant vocabularies, error codes, drift taxonomies, schemas,
and goal identifiers are injected by callers (profiles, tests, job assembly).
Only explicitly enumerated transport representations are normalized.  Paths,
content identifiers, sizes, errors, state changes, and selection signals remain
observable contract data.

Witnesses are comparison evidence only: they are never completion evidence,
correctness evidence, or repair authority.  Declared network denial is an
in-process guard, not OS isolation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import errno as errno_module
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import socket
import stat as stat_module
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, TypeAlias, runtime_checkable
from unittest import mock

# ---------------------------------------------------------------------------
# Schema identity (neutral defaults; profiles may override)
# ---------------------------------------------------------------------------

DIFFERENTIAL_WITNESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/differential-contract-witness@1"
)
DIFFERENTIAL_TRACE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/canonical-operation-trace@1"
)
MAX_TRACE_STEPS: Final[int] = 256
MAX_FIXTURE_BYTES: Final[int] = 16 * 1024 * 1024
DEFAULT_STEP_TIMEOUT_SECONDS: Final[float] = 30.0

# Authority bounds: a differential witness is comparison evidence only.
WITNESS_IS_COMPLETION_EVIDENCE: Final[bool] = False
WITNESS_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
WITNESS_AUTHORIZES_REPAIR: Final[bool] = False


class DifferentialHarnessError(ValueError):
    """Raised when a requested differential run is not safe or well formed."""

    def __init__(self, message: str, *, reason_codes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(code) for code in reason_codes if str(code))


class SurfaceAvailability(str, Enum):
    """Whether a surface can contribute authoritative evidence."""

    REAL = "real"
    MOCK = "mock"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class HermeticNetworkError(PermissionError):
    """Raised when a differential case attempts external network access."""


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class DriftKind(str, Enum):
    """Closed generic drift taxonomy.  Domain profiles map invariants onto these."""

    PATH = "path"
    BYTES_TEXT = "bytes_text"
    STAT_LIST = "stat_list"
    RENAME_ATOMICITY = "rename_atomicity"
    JOURNAL = "journal"
    CACHE = "cache"
    AUTHORIZATION = "authorization"
    FALLBACK = "fallback"
    SILENT_SUCCESS = "silent_success"
    RESULT = "result"
    ERROR = "error"
    FIXTURE = "fixture"
    CLEANUP = "cleanup"
    IDENTITY = "identity"
    TIMEOUT = "timeout"
    BUDGET = "budget"
    UNKNOWN = "unknown"


class NormalizationRule(str, Enum):
    """Closed set of contract-approved representation normalizations."""

    TRANSPORT_ENVELOPE = "transport_envelope"
    ERROR_ENVELOPE = "error_envelope"
    BYTES_LIKE = "bytes_like"
    EXPLICIT_UTF8_TEXT = "explicit_utf8_text"
    STAT_FIELD_ALIASES = "stat_field_aliases"


DEFAULT_NORMALIZATION_RULES: Final[tuple[NormalizationRule, ...]] = (
    NormalizationRule.TRANSPORT_ENVELOPE,
    NormalizationRule.ERROR_ENVELOPE,
    NormalizationRule.BYTES_LIKE,
    NormalizationRule.EXPLICIT_UTF8_TEXT,
    NormalizationRule.STAT_FIELD_ALIASES,
)

# Generic exception → error-code defaults.  Domain profiles may replace the
# classifier entirely; these defaults intentionally use transport-neutral codes.
DEFAULT_ERROR_CODE_IO_FAILURE: Final[str] = "io_failure"
DEFAULT_ERROR_CODES: Final[Mapping[str, str]] = {
    "PermissionError": "permission_denied",
    "FileNotFoundError": "not_found",
    "FileExistsError": "already_exists",
    "IsADirectoryError": "not_a_file",
    "NotADirectoryError": "not_a_directory",
    "TimeoutError": "deadline_exceeded",
    "NotImplementedError": "unsupported",
    "CancelledError": "cancelled",
    "TypeError": "invalid_argument",
    "ValueError": "invalid_argument",
}


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_value(value: Any) -> JsonValue:
    """Produce lossless, deterministic JSON for provenance records."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DifferentialHarnessError(
                "non-finite floats are not canonical",
                reason_codes=("non_finite_float",),
            )
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, bytes):
        return {"$type": "bytes", "hex": value.hex()}
    if isinstance(value, (bytearray, memoryview)):
        raw = bytes(value)
        return {"$type": type(value).__name__, "hex": raw.hex()}
    if isinstance(value, os.PathLike):
        return {"$type": "path", "value": os.fspath(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DifferentialHarnessError(
                    f"canonical mappings require string keys, got {type(key).__name__}",
                    reason_codes=("non_string_key",),
                )
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=_canonical_json)
    return {
        "$type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_id(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _copy_record(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    converted = _json_value(value)
    if not isinstance(converted, dict):  # pragma: no cover - typing guard
        raise DifferentialHarnessError(
            "record must be an object",
            reason_codes=("record_type",),
        )
    return converted


def _operation_value(operation: Any) -> str:
    if isinstance(operation, Enum):
        value = operation.value
    else:
        value = operation
    if not isinstance(value, str) or not value.strip():
        raise DifferentialHarnessError(
            "operation must be a non-empty string",
            reason_codes=("operation_empty",),
        )
    return value


# ---------------------------------------------------------------------------
# Trace model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceStep:
    vector_id: str
    operation: str
    description: str
    request: Mapping[str, Any]
    expected: Mapping[str, Any]
    invariant_ids: tuple[str, ...]
    source_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.vector_id, str) or not self.vector_id.strip():
            raise DifferentialHarnessError(
                "trace vector_id must be non-empty",
                reason_codes=("vector_id_empty",),
            )
        object.__setattr__(self, "operation", _operation_value(self.operation))
        if not isinstance(self.description, str) or not self.description.strip():
            raise DifferentialHarnessError(
                "trace description must be non-empty",
                reason_codes=("description_empty",),
            )
        if not isinstance(self.request, Mapping) or not isinstance(
            self.expected, Mapping
        ):
            raise DifferentialHarnessError(
                "trace request and expected values must be mappings",
                reason_codes=("request_expected_type",),
            )
        object.__setattr__(self, "request", _copy_record(self.request))
        object.__setattr__(self, "expected", _copy_record(self.expected))
        object.__setattr__(
            self, "invariant_ids", tuple(sorted(set(self.invariant_ids)))
        )
        object.__setattr__(
            self, "source_contract_ids", tuple(sorted(set(self.source_contract_ids)))
        )

    @classmethod
    def from_vector(cls, vector: Any) -> "TraceStep":
        """Build a step from a CanonicalVector-like object or mapping."""

        if isinstance(vector, Mapping):
            return cls(
                vector_id=str(vector["vector_id"]),
                operation=_operation_value(vector["operation"]),
                description=str(vector["description"]),
                request=vector.get("request", {}),
                expected=vector.get("expected", {}),
                invariant_ids=tuple(vector.get("invariant_ids", ())),
                source_contract_ids=tuple(vector.get("source_contract_ids", ())),
            )
        return cls(
            vector_id=str(vector.vector_id),
            operation=_operation_value(vector.operation),
            description=str(vector.description),
            request=vector.request,
            expected=vector.expected,
            invariant_ids=tuple(vector.invariant_ids),
            source_contract_ids=tuple(getattr(vector, "source_contract_ids", ())),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "operation": self.operation,
            "description": self.description,
            "request": dict(self.request),
            "expected": dict(self.expected),
            "invariant_ids": list(self.invariant_ids),
            "source_contract_ids": list(self.source_contract_ids),
        }


@dataclass(frozen=True)
class CanonicalOperationTrace:
    steps: tuple[TraceStep, ...]
    contract_pack_cid: str
    schema: str = DIFFERENTIAL_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if not self.steps:
            raise DifferentialHarnessError(
                "a differential trace cannot be empty",
                reason_codes=("trace_empty",),
            )
        if len(self.steps) > MAX_TRACE_STEPS:
            raise DifferentialHarnessError(
                f"trace has {len(self.steps)} steps; maximum is {MAX_TRACE_STEPS}",
                reason_codes=("trace_over_budget",),
            )
        ids = [step.vector_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise DifferentialHarnessError(
                "trace vector_ids must be unique",
                reason_codes=("vector_id_duplicate",),
            )
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise DifferentialHarnessError(
                "trace schema must be non-empty",
                reason_codes=("schema_empty",),
            )
        if not isinstance(self.contract_pack_cid, str) or not self.contract_pack_cid:
            raise DifferentialHarnessError(
                "trace contract_pack_cid must be non-empty",
                reason_codes=("contract_pack_cid_empty",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_pack_cid": self.contract_pack_cid,
            "steps": [step.to_record() for step in self.steps],
        }

    @property
    def content_id(self) -> str:
        return _content_id(self.to_record())


@runtime_checkable
class ContractTraceProvider(Protocol):
    """Supplies finite, deterministic traces from a contract profile or corpus."""

    def build_trace(
        self, *, vector_ids: Iterable[str] | None = None
    ) -> CanonicalOperationTrace: ...


@dataclass(frozen=True)
class VectorTraceProvider:
    """Trace provider over an explicit finite vector sequence."""

    vectors: tuple[Any, ...]
    contract_pack_cid: str
    schema: str = DIFFERENTIAL_TRACE_SCHEMA

    def build_trace(
        self, *, vector_ids: Iterable[str] | None = None
    ) -> CanonicalOperationTrace:
        selected = self.vectors
        if vector_ids is not None:
            requested = tuple(vector_ids)
            if not requested:
                raise DifferentialHarnessError(
                    "vector_ids cannot be empty",
                    reason_codes=("vector_ids_empty",),
                )
            if len(requested) != len(set(requested)):
                raise DifferentialHarnessError(
                    "vector_ids cannot contain duplicates",
                    reason_codes=("vector_ids_duplicate",),
                )
            available: dict[str, Any] = {}
            for vector in self.vectors:
                if isinstance(vector, Mapping):
                    available[str(vector["vector_id"])] = vector
                else:
                    available[str(vector.vector_id)] = vector
            unknown = sorted(set(requested) - set(available))
            if unknown:
                raise DifferentialHarnessError(
                    f"unknown contract vector_ids: {', '.join(unknown)}",
                    reason_codes=("unknown_vector_id",),
                )
            selected = tuple(available[vector_id] for vector_id in requested)
        return CanonicalOperationTrace(
            steps=tuple(TraceStep.from_vector(vector) for vector in selected),
            contract_pack_cid=self.contract_pack_cid,
            schema=self.schema,
        )


@dataclass(frozen=True)
class ProfileTraceProvider:
    """Trace provider that reads vectors from a ProgramContractProfile-like object."""

    profile: Any
    schema: str = DIFFERENTIAL_TRACE_SCHEMA

    def build_trace(
        self, *, vector_ids: Iterable[str] | None = None
    ) -> CanonicalOperationTrace:
        vectors = tuple(getattr(self.profile, "vectors", ()))
        if not vectors:
            raise DifferentialHarnessError(
                "profile has no canonical vectors",
                reason_codes=("profile_vectors_empty",),
            )
        content_id = str(
            getattr(self.profile, "content_id", None)
            or _content_id(getattr(self.profile, "to_record", lambda: {})())
        )
        return VectorTraceProvider(
            vectors=vectors,
            contract_pack_cid=content_id,
            schema=self.schema,
        ).build_trace(vector_ids=vector_ids)


def build_canonical_operation_trace(
    *,
    provider: ContractTraceProvider | None = None,
    vectors: Sequence[Any] | None = None,
    contract_pack_cid: str | None = None,
    vector_ids: Iterable[str] | None = None,
    schema: str = DIFFERENTIAL_TRACE_SCHEMA,
) -> CanonicalOperationTrace:
    """Build a finite, deterministic trace from a provider or explicit vectors."""

    if provider is not None:
        return provider.build_trace(vector_ids=vector_ids)
    if vectors is None:
        raise DifferentialHarnessError(
            "a ContractTraceProvider or explicit vectors are required",
            reason_codes=("trace_source_required",),
        )
    if not contract_pack_cid:
        raise DifferentialHarnessError(
            "contract_pack_cid is required when building from vectors",
            reason_codes=("contract_pack_cid_required",),
        )
    return VectorTraceProvider(
        vectors=tuple(vectors),
        contract_pack_cid=contract_pack_cid,
        schema=schema,
    ).build_trace(vector_ids=vector_ids)


# ---------------------------------------------------------------------------
# Fixture model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureEntry:
    path: str
    kind: str
    content_hex: str | None = None
    mode: int = 0o600

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.path)
        if (
            not self.path
            or normalized.is_absolute()
            or ".." in normalized.parts
            or str(normalized) in ("", ".")
            or str(normalized) != self.path
        ):
            raise DifferentialHarnessError(
                f"fixture path must be relative and contained: {self.path!r}",
                reason_codes=("fixture_path_escape",),
            )
        if self.path != unicodedata.normalize("NFC", self.path):
            raise DifferentialHarnessError(
                f"fixture paths must be NFC canonical: {self.path!r}",
                reason_codes=("fixture_path_nfc",),
            )
        if self.kind not in {"file", "directory"}:
            raise DifferentialHarnessError(
                f"unsupported fixture entry kind: {self.kind!r}",
                reason_codes=("fixture_kind",),
            )
        if self.kind == "file":
            if self.content_hex is None:
                raise DifferentialHarnessError(
                    "fixture files require content_hex",
                    reason_codes=("fixture_content_required",),
                )
            try:
                decoded = bytes.fromhex(self.content_hex)
            except ValueError as exc:
                raise DifferentialHarnessError(
                    f"invalid fixture content hex for {self.path!r}",
                    reason_codes=("fixture_content_hex",),
                ) from exc
            if decoded.hex() != self.content_hex:
                raise DifferentialHarnessError(
                    f"fixture content hex must be canonical lowercase: {self.path!r}",
                    reason_codes=("fixture_content_hex",),
                )
        elif self.content_hex is not None:
            raise DifferentialHarnessError(
                "fixture directories cannot declare content",
                reason_codes=("fixture_directory_content",),
            )
        if (
            not isinstance(self.mode, int)
            or isinstance(self.mode, bool)
            or self.mode < 0
            or self.mode > 0o777
        ):
            raise DifferentialHarnessError(
                f"fixture mode must be an integer from 0o000 to 0o777: {self.mode!r}",
                reason_codes=("fixture_mode",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "content_hex": self.content_hex,
            "mode": self.mode,
        }

    @property
    def size_bytes(self) -> int:
        if self.kind != "file" or self.content_hex is None:
            return 0
        return len(bytes.fromhex(self.content_hex))


@runtime_checkable
class FixtureAdapter(Protocol):
    """Materializes an isolated fixture tree for one observation."""

    fixture_id: str

    def to_record(self) -> Mapping[str, Any]: ...

    @property
    def content_id(self) -> str: ...

    def materialize(self, root: Path) -> str: ...

    def total_bytes(self) -> int: ...


@dataclass(frozen=True)
class FixtureSpec:
    """Tree fixture recipe.  Domain profiles supply their own entries."""

    fixture_id: str
    entries: tuple[FixtureEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise DifferentialHarnessError(
                "fixture_id must be non-empty",
                reason_codes=("fixture_id_empty",),
            )
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise DifferentialHarnessError(
                "fixture paths must be unique",
                reason_codes=("fixture_path_duplicate",),
            )
        file_paths = {
            PurePosixPath(entry.path)
            for entry in self.entries
            if entry.kind == "file"
        }
        for entry in self.entries:
            parent = PurePosixPath(entry.path).parent
            while str(parent) != ".":
                if parent in file_paths:
                    raise DifferentialHarnessError(
                        f"fixture file {parent} cannot contain {entry.path}",
                        reason_codes=("fixture_file_parent",),
                    )
                parent = parent.parent

    def to_record(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "entries": [
                entry.to_record()
                for entry in sorted(self.entries, key=lambda item: item.path)
            ],
        }

    @property
    def content_id(self) -> str:
        return _content_id(self.to_record())

    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    def materialize(self, root: Path) -> str:
        root.mkdir(parents=True, exist_ok=False)
        for entry in sorted(
            self.entries,
            key=lambda item: (len(PurePosixPath(item.path).parts), item.path),
        ):
            target = root.joinpath(*PurePosixPath(entry.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "directory":
                target.mkdir(exist_ok=True)
            else:
                target.write_bytes(bytes.fromhex(entry.content_hex or ""))
            os.chmod(target, entry.mode)
        return snapshot_tree(root).content_id


def build_fixture_spec(
    fixture_id: str, entries: Sequence[FixtureEntry | Mapping[str, Any]]
) -> FixtureSpec:
    """Build a FixtureSpec from entries or compact mapping recipes."""

    resolved: list[FixtureEntry] = []
    for entry in entries:
        if isinstance(entry, FixtureEntry):
            resolved.append(entry)
        else:
            resolved.append(
                FixtureEntry(
                    path=str(entry["path"]),
                    kind=str(entry["kind"]),
                    content_hex=entry.get("content_hex"),
                    mode=int(entry.get("mode", 0o600)),
                )
            )
    return FixtureSpec(fixture_id=fixture_id, entries=tuple(resolved))


@dataclass(frozen=True)
class TreeSnapshot:
    entries: tuple[Mapping[str, Any], ...]
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {"entries": [dict(entry) for entry in self.entries], "cid": self.content_id}


def snapshot_tree(root: Path) -> TreeSnapshot:
    """Snapshot without following symlinks or reading outside ``root``."""

    if not root.exists():
        return TreeSnapshot(entries=(), content_id=_content_id({"missing": True}))
    records: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat_module.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                    "mode": mode,
                }
            )
        elif path.is_dir():
            records.append({"path": relative, "kind": "directory", "mode": mode})
        elif path.is_file():
            content = path.read_bytes()
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode,
                    "size": len(content),
                    "content_cid": _content_id(
                        {
                            "media_type": "application/octet-stream",
                            "hex": content.hex(),
                        }
                    ),
                }
            )
        else:
            records.append({"path": relative, "kind": "special", "mode": mode})
    return TreeSnapshot(entries=tuple(records), content_id=_content_id(records))


# ---------------------------------------------------------------------------
# Surface adapters
# ---------------------------------------------------------------------------


@dataclass
class SurfaceRunContext:
    """Capabilities exposed to an adapter for one isolated trace case."""

    root: Path
    fixture: FixtureAdapter
    step: TraceStep
    state: dict[str, Any] = field(default_factory=dict)
    network_allowed: bool = False

    def resolve_path(self, path: str, *, allow_root: bool = True) -> Path:
        if not isinstance(path, str):
            raise DifferentialHarnessError(
                "surface paths must be strings",
                reason_codes=("path_type",),
            )
        if "\x00" in path:
            raise DifferentialHarnessError(
                "surface paths cannot contain NUL",
                reason_codes=("path_nul",),
            )
        relative = PurePosixPath(path.lstrip("/"))
        if ".." in relative.parts:
            raise PermissionError("path traversal outside the fixture is denied")
        target = self.root.joinpath(*relative.parts).resolve(strict=False)
        fixture_root = self.root.resolve(strict=True)
        try:
            target.relative_to(fixture_root)
        except ValueError as exc:
            raise PermissionError("path escaped the fixture root") from exc
        if target == fixture_root and not allow_root:
            raise PermissionError("the fixture root is not a valid target")
        return target


SurfaceExecutor: TypeAlias = Callable[
    [TraceStep, SurfaceRunContext], Any | Awaitable[Any]
]


@runtime_checkable
class SurfaceAdapter(Protocol):
    surface_id: str
    family: str
    availability: SurfaceAvailability
    implementation: str
    public_surface: str
    package_names: tuple[str, ...]
    unavailable_reason: str | None

    def execute(
        self, step: TraceStep, context: SurfaceRunContext
    ) -> Any | Awaitable[Any]: ...


@dataclass(frozen=True)
class CallableSurfaceAdapter:
    """Adapt a real callable, a declared mock, an unavailable, or unknown surface."""

    surface_id: str
    family: str
    executor: SurfaceExecutor | None
    implementation: str
    public_surface: str = "python"
    availability: SurfaceAvailability = SurfaceAvailability.REAL
    package_names: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise DifferentialHarnessError(
                "surface_id must be non-empty",
                reason_codes=("surface_id_empty",),
            )
        if not isinstance(self.family, str) or not self.family.strip():
            raise DifferentialHarnessError(
                "surface family must be a non-empty string",
                reason_codes=("family_empty",),
            )
        if not isinstance(self.availability, SurfaceAvailability):
            raise DifferentialHarnessError(
                "surface availability must be a SurfaceAvailability",
                reason_codes=("availability_type",),
            )
        if (
            not isinstance(self.implementation, str)
            or not self.implementation.strip()
        ):
            raise DifferentialHarnessError(
                "surface implementation must be non-empty",
                reason_codes=("implementation_empty",),
            )
        if not isinstance(self.public_surface, str) or not self.public_surface.strip():
            raise DifferentialHarnessError(
                "surface public_surface must be non-empty",
                reason_codes=("public_surface_empty",),
            )
        if self.availability is SurfaceAvailability.UNAVAILABLE:
            if self.executor is not None:
                raise DifferentialHarnessError(
                    "unavailable surfaces cannot have an executor",
                    reason_codes=("unavailable_has_executor",),
                )
            if not self.unavailable_reason:
                raise DifferentialHarnessError(
                    "unavailable surfaces require an unavailable_reason",
                    reason_codes=("unavailable_reason_required",),
                )
        elif self.availability is SurfaceAvailability.UNKNOWN:
            # Unknown surfaces may optionally execute; if they do not, reason is required.
            if self.executor is None and not self.unavailable_reason:
                raise DifferentialHarnessError(
                    "unknown surfaces without an executor require an unavailable_reason",
                    reason_codes=("unknown_reason_required",),
                )
        elif self.executor is None:
            raise DifferentialHarnessError(
                "real and mock surfaces require an executor",
                reason_codes=("executor_required",),
            )
        object.__setattr__(
            self,
            "package_names",
            tuple(
                sorted(
                    {
                        name
                        for name in self.package_names
                        if isinstance(name, str) and name.strip()
                    }
                )
            ),
        )

    @classmethod
    def unavailable(
        cls,
        surface_id: str,
        family: str,
        *,
        implementation: str,
        reason: str,
        public_surface: str = "python",
        package_names: Sequence[str] = (),
    ) -> "CallableSurfaceAdapter":
        return cls(
            surface_id=surface_id,
            family=family,
            executor=None,
            implementation=implementation,
            public_surface=public_surface,
            availability=SurfaceAvailability.UNAVAILABLE,
            package_names=tuple(package_names),
            unavailable_reason=reason,
        )

    @classmethod
    def unknown(
        cls,
        surface_id: str,
        family: str,
        *,
        implementation: str,
        reason: str,
        public_surface: str = "python",
        package_names: Sequence[str] = (),
    ) -> "CallableSurfaceAdapter":
        return cls(
            surface_id=surface_id,
            family=family,
            executor=None,
            implementation=implementation,
            public_surface=public_surface,
            availability=SurfaceAvailability.UNKNOWN,
            package_names=tuple(package_names),
            unavailable_reason=reason,
        )

    def execute(
        self, step: TraceStep, context: SurfaceRunContext
    ) -> Any | Awaitable[Any]:
        if self.executor is None:  # pragma: no cover - validated and skipped
            raise RuntimeError("unavailable surface cannot execute")
        return self.executor(step, context)


# ---------------------------------------------------------------------------
# Identity, errors, normalization, drift classifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeIdentity:
    executable: str
    python_version: str
    implementation: str
    platform: str
    byteorder: str
    filesystem_encoding: str
    packages: Mapping[str, str]
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform": self.platform,
            "byteorder": self.byteorder,
            "filesystem_encoding": self.filesystem_encoding,
            "packages": dict(self.packages),
            "cid": self.content_id,
        }


def capture_runtime_identity(package_names: Iterable[str] = ()) -> RuntimeIdentity:
    packages: dict[str, str] = {}
    for name in sorted(set(package_names)):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "<unavailable>"
    record = {
        "executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "filesystem_encoding": sys.getfilesystemencoding(),
        "packages": packages,
    }
    return RuntimeIdentity(**record, content_id=_content_id(record))


@dataclass(frozen=True)
class ImplementationIdentity:
    module: str
    qualname: str
    source_path: str | None
    source_cid: str | None
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "qualname": self.qualname,
            "source_path": self.source_path,
            "source_cid": self.source_cid,
            "cid": self.content_id,
        }


def _implementation_identity(adapter: SurfaceAdapter) -> ImplementationIdentity:
    target = getattr(adapter, "executor", None) or adapter.execute
    module = getattr(target, "__module__", type(target).__module__)
    qualname = getattr(target, "__qualname__", type(target).__qualname__)
    source_path: str | None = None
    source_cid: str | None = None
    try:
        candidate = inspect.getsourcefile(target)
    except (OSError, TypeError):
        candidate = None
    if candidate:
        source = Path(candidate)
        try:
            source_path = str(source.resolve())
            source_cid = _content_id(
                {"path": source_path, "content_hex": source.read_bytes().hex()}
            )
        except OSError:
            source_path = str(source)
    record = {
        "module": module,
        "qualname": qualname,
        "source_path": source_path,
        "source_cid": source_cid,
    }
    return ImplementationIdentity(**record, content_id=_content_id(record))


@dataclass(frozen=True)
class ErrorIdentity:
    code: str
    exception_module: str | None
    exception_type: str | None
    message: str | None
    errno: int | None
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "exception_module": self.exception_module,
            "exception_type": self.exception_type,
            "message": self.message,
            "errno": self.errno,
            "cid": self.content_id,
        }


@runtime_checkable
class ErrorClassifier(Protocol):
    """Maps exceptions and reported error envelopes onto contract error codes."""

    def classify_exception(self, error: BaseException) -> str: ...

    def default_code(self) -> str: ...


@dataclass(frozen=True)
class MappingErrorClassifier:
    """Error classifier driven by exception-type and errno maps (profile data)."""

    exception_codes: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ERROR_CODES)
    )
    errno_codes: Mapping[int, str] = field(default_factory=dict)
    unsupported_errnos: frozenset[int] = field(default_factory=frozenset)
    default: str = DEFAULT_ERROR_CODE_IO_FAILURE

    def default_code(self) -> str:
        return self.default

    def classify_exception(self, error: BaseException) -> str:
        # Walk the concrete MRO so subclasses (e.g. PermissionError children)
        # inherit the mapped contract code without listing every leaf type.
        for base in type(error).mro():
            if base is object or base is BaseException or base is Exception:
                continue
            mapped = self.exception_codes.get(base.__name__)
            if mapped is not None:
                return mapped
        if isinstance(error, asyncio.CancelledError):
            return self.exception_codes.get("CancelledError", "cancelled")
        if isinstance(error, OSError):
            errno_value = getattr(error, "errno", None)
            if isinstance(errno_value, int):
                if errno_value in self.errno_codes:
                    return self.errno_codes[errno_value]
                if errno_value in self.unsupported_errnos:
                    return self.exception_codes.get(
                        "NotImplementedError", "unsupported"
                    )
                # Built-in errno fallbacks when profile left errno_codes empty.
                if not self.errno_codes:
                    by_errno = {
                        value: code
                        for symbol, code in (
                            ("EACCES", "permission_denied"),
                            ("EPERM", "permission_denied"),
                            ("ENOENT", "not_found"),
                            ("EEXIST", "already_exists"),
                            ("EISDIR", "not_a_file"),
                            ("ENOTDIR", "not_a_directory"),
                            ("ENOTEMPTY", "directory_not_empty"),
                            ("EINVAL", "invalid_argument"),
                            ("ENOSPC", "resource_exhausted"),
                            ("EDQUOT", "resource_exhausted"),
                            ("ETIMEDOUT", "deadline_exceeded"),
                        )
                        if (value := getattr(errno_module, symbol, None)) is not None
                    }
                    if errno_value in by_errno:
                        return by_errno[errno_value]
                    unsupported = {
                        getattr(errno_module, name)
                        for name in ("ENOSYS", "ENOTSUP", "EOPNOTSUPP")
                        if hasattr(errno_module, name)
                    }
                    if errno_value in unsupported:
                        return "unsupported"
        return self.default


def default_error_classifier() -> MappingErrorClassifier:
    return MappingErrorClassifier()


def _exception_identity(
    error: BaseException, classifier: ErrorClassifier
) -> ErrorIdentity:
    code = classifier.classify_exception(error)
    errno = getattr(error, "errno", None)
    record = {
        "code": code,
        "exception_module": type(error).__module__,
        "exception_type": type(error).__qualname__,
        "message": str(error),
        "errno": errno if isinstance(errno, int) else None,
    }
    return ErrorIdentity(**record, content_id=_content_id(record))


def _reported_error_identity(
    value: Mapping[str, Any], classifier: ErrorClassifier
) -> ErrorIdentity | None:
    candidate: Mapping[str, Any] = value
    for key in ("result", "data", "body"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
    error: Any = candidate.get("error")
    if (
        error is None
        and candidate.get("ok") is not False
        and candidate.get("success") is not False
    ):
        return None
    if isinstance(error, Mapping):
        code = str(error.get("code", classifier.default_code()))
        message = error.get("message")
    elif isinstance(error, str):
        code = error
        message = candidate.get("message")
    else:
        code = str(candidate.get("code", classifier.default_code()))
        message = candidate.get("message")
    record = {
        "code": code,
        "exception_module": None,
        "exception_type": None,
        "message": None if message is None else str(message),
        "errno": None,
    }
    return ErrorIdentity(**record, content_id=_content_id(record))


def _has_invariant(step: TraceStep, invariant: str) -> bool:
    """Match bare kinds and ``invariant:<kind>`` identifiers."""

    return invariant in step.invariant_ids or f"invariant:{invariant}" in step.invariant_ids


@runtime_checkable
class ResultNormalizer(Protocol):
    """Applies only contract-approved representation normalizations."""

    def rules_for(self, step: TraceStep) -> tuple[NormalizationRule, ...]: ...

    def normalize(
        self,
        step: TraceStep,
        value: Any,
        *,
        rules: Iterable[NormalizationRule] | None = None,
    ) -> JsonValue: ...


@dataclass(frozen=True)
class ContractResultNormalizer:
    """Default normalizer.  Invariant kinds that enable rules are profile data."""

    always_rules: tuple[NormalizationRule, ...] = (
        NormalizationRule.TRANSPORT_ENVELOPE,
        NormalizationRule.ERROR_ENVELOPE,
        NormalizationRule.BYTES_LIKE,
    )
    utf8_text_invariants: frozenset[str] = frozenset({"bytes_text"})
    stat_alias_invariants: frozenset[str] = frozenset({"stat_list"})
    default_error_code: str = DEFAULT_ERROR_CODE_IO_FAILURE

    def rules_for(self, step: TraceStep) -> tuple[NormalizationRule, ...]:
        rules = list(self.always_rules)
        if any(_has_invariant(step, kind) for kind in self.stat_alias_invariants):
            rules.append(NormalizationRule.STAT_FIELD_ALIASES)
        if any(_has_invariant(step, kind) for kind in self.utf8_text_invariants):
            rules.append(NormalizationRule.EXPLICIT_UTF8_TEXT)
        return tuple(rules)

    def normalize(
        self,
        step: TraceStep,
        value: Any,
        *,
        rules: Iterable[NormalizationRule] | None = None,
    ) -> JsonValue:
        selected = set(self.rules_for(step) if rules is None else rules)
        return _normalize_value(value, selected, default_error_code=self.default_error_code)


def default_result_normalizer() -> ContractResultNormalizer:
    return ContractResultNormalizer()


def _unwrap_transport(value: Any, rules: set[NormalizationRule]) -> Any:
    if not isinstance(value, Mapping):
        return value
    if NormalizationRule.TRANSPORT_ENVELOPE not in rules:
        return value
    keys = set(value)
    successful = value.get("ok") is True or value.get("success") is True
    if successful and "result" in value and keys <= {
        "ok",
        "success",
        "status",
        "result",
        "request_id",
    }:
        return value["result"]
    if successful and "data" in value and keys <= {
        "ok",
        "success",
        "status",
        "data",
        "request_id",
    }:
        return value["data"]
    if (
        isinstance(value.get("status"), int)
        and 200 <= value["status"] < 300
        and "body" in value
        and keys <= {"status", "body", "headers", "request_id"}
    ):
        return value["body"]
    return value


def _normalize_value(
    value: Any,
    rules: set[NormalizationRule],
    *,
    default_error_code: str = DEFAULT_ERROR_CODE_IO_FAILURE,
) -> JsonValue:
    value = _unwrap_transport(value, rules)
    if isinstance(value, bytes):
        if NormalizationRule.BYTES_LIKE in rules:
            return {"bytes_hex": value.hex()}
        return _json_value(value)
    if isinstance(value, (bytearray, memoryview)):
        if NormalizationRule.BYTES_LIKE in rules:
            return {"bytes_hex": bytes(value).hex()}
        return _json_value(value)
    if isinstance(value, Mapping):
        keys = set(value)
        if (
            NormalizationRule.EXPLICIT_UTF8_TEXT in rules
            and "text" in value
            and keys <= {"text", "encoding", "errors"}
            and value.get("encoding", "utf-8").lower().replace("_", "-") == "utf-8"
            and isinstance(value["text"], str)
        ):
            errors = str(value.get("errors", "strict"))
            return {"bytes_hex": value["text"].encode("utf-8", errors).hex()}
        result = {
            str(key): _normalize_value(
                item, rules, default_error_code=default_error_code
            )
            for key, item in value.items()
        }
        if NormalizationRule.ERROR_ENVELOPE in rules and "error" in result:
            error = result["error"]
            if isinstance(error, str):
                result["error"] = {"code": error}
            elif isinstance(error, Mapping) and "code" not in error:
                result["error"] = {
                    "code": str(error.get("type", default_error_code)),
                    **dict(error),
                }
            result.pop("ok", None)
            result.pop("success", None)
            result.pop("status", None)
        if NormalizationRule.STAT_FIELD_ALIASES in rules:
            aliases = {"length": "size", "kind": "type", "content_id": "cid"}
            for alias, canonical in aliases.items():
                if alias in result and canonical not in result:
                    result[canonical] = result.pop(alias)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_value(item, rules, default_error_code=default_error_code)
            for item in value
        ]
    return _json_value(value)


def normalize_contract_result(
    step: TraceStep,
    value: Any,
    *,
    rules: Iterable[NormalizationRule] | None = None,
    normalizer: ResultNormalizer | None = None,
) -> JsonValue:
    """Normalize only the closed set of approved transport representations."""

    selected_normalizer = normalizer or default_result_normalizer()
    return selected_normalizer.normalize(step, value, rules=rules)


_MISSING = object()


def _project_like(expected: Any, actual: Any) -> Any:
    """Project permitted extra metadata away, retaining every expected field."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return actual
        return {
            key: (
                {"$missing": True}
                if actual.get(key, _MISSING) is _MISSING
                else _project_like(value, actual[key])
            )
            for key, value in expected.items()
        }
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return actual
        return [
            _project_like(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)
        ]
    return actual


def _error_container(value: JsonValue) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if isinstance(value.get("error"), Mapping):
            return value["error"]
        nested = value.get("result")
        if isinstance(nested, Mapping) and isinstance(nested.get("error"), Mapping):
            return nested["error"]
    return None


def _expects_no_effects(expected: JsonValue) -> bool:
    if isinstance(expected, Mapping):
        if expected.get("effects") == "none":
            return True
        error = expected.get("error")
        return isinstance(error, Mapping) and error.get("effects") == "none"
    return False


def _add_derived_effects(
    expected: JsonValue, actual: JsonValue, *, unchanged: bool
) -> JsonValue:
    if not _expects_no_effects(expected) or not isinstance(actual, Mapping):
        return actual
    copied: dict[str, JsonValue] = dict(actual)
    value = "none" if unchanged else "changed"
    expected_error = (
        expected.get("error") if isinstance(expected, Mapping) else None
    )
    if isinstance(expected_error, Mapping):
        actual_error = copied.get("error")
        if isinstance(actual_error, Mapping):
            copied["error"] = {**actual_error, "effects": value}
        elif "effects" not in copied:
            # Preserve a successful response as success: inventing an error here
            # would hide the silent-success drift that this harness must expose.
            copied["effects"] = value
    else:
        copied["effects"] = value
    return copied


def _mismatch_paths(expected: Any, actual: Any, prefix: str = "$") -> tuple[str, ...]:
    mismatches: list[str] = []
    if type(expected) is not type(actual):
        return (prefix,)
    if isinstance(expected, Mapping):
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}"
            if key not in expected or key not in actual:
                mismatches.append(child)
            else:
                mismatches.extend(_mismatch_paths(expected[key], actual[key], child))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append(prefix)
        else:
            for index, (left, right) in enumerate(zip(expected, actual)):
                mismatches.extend(_mismatch_paths(left, right, f"{prefix}[{index}]"))
    elif expected != actual:
        mismatches.append(prefix)
    return tuple(mismatches)


@runtime_checkable
class DriftClassifier(Protocol):
    """Maps step invariants and observation outcomes onto drift kinds."""

    def classify(
        self, step: TraceStep, observation: "SurfaceObservation"
    ) -> tuple[DriftKind, ...]: ...


@dataclass(frozen=True)
class InvariantDriftClassifier:
    """Drift classifier parameterized by invariant-kind → DriftKind maps."""

    invariant_to_kinds: Mapping[str, tuple[DriftKind, ...]] = field(
        default_factory=dict
    )

    def classify(
        self, step: TraceStep, observation: "SurfaceObservation"
    ) -> tuple[DriftKind, ...]:
        kinds: set[DriftKind] = set()
        for invariant, mapped in self.invariant_to_kinds.items():
            if _has_invariant(step, invariant):
                kinds.update(mapped)
        if observation.silent_success:
            kinds.add(DriftKind.SILENT_SUCCESS)
        if not observation.cleanup.succeeded:
            kinds.add(DriftKind.CLEANUP)
        if observation.error is not None:
            kinds.add(DriftKind.ERROR)
        else:
            kinds.add(DriftKind.RESULT)
        if observation.status is ObservationStatus.UNKNOWN:
            kinds.add(DriftKind.UNKNOWN)
        return tuple(sorted(kinds, key=lambda kind: kind.value))


def default_drift_classifier() -> InvariantDriftClassifier:
    """Neutral default: only observation-derived kinds (no domain invariant map)."""

    return InvariantDriftClassifier(invariant_to_kinds={})


# ---------------------------------------------------------------------------
# Execution permit (timeout / budget)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPermit:
    """Bounds for a hermetic differential run.  Over-budget runs fail closed."""

    timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS
    max_steps: int = MAX_TRACE_STEPS
    max_fixture_bytes: int = MAX_FIXTURE_BYTES
    network_allowed: bool = False
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise DifferentialHarnessError(
                "timeout_seconds must be positive",
                reason_codes=("timeout_invalid",),
            )
        if self.max_steps <= 0 or self.max_steps > MAX_TRACE_STEPS:
            raise DifferentialHarnessError(
                f"max_steps must be in 1..{MAX_TRACE_STEPS}",
                reason_codes=("budget_steps",),
            )
        if self.max_fixture_bytes <= 0:
            raise DifferentialHarnessError(
                "max_fixture_bytes must be positive",
                reason_codes=("budget_fixture",),
            )
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise DifferentialHarnessError(
                "max_wall_seconds must be positive when set",
                reason_codes=("budget_wall",),
            )

    def enforce_trace(self, trace: CanonicalOperationTrace) -> None:
        if len(trace.steps) > self.max_steps:
            raise DifferentialHarnessError(
                f"trace has {len(trace.steps)} steps; permit max_steps is {self.max_steps}",
                reason_codes=("over_budget_steps",),
            )

    def enforce_fixture(self, fixture: FixtureAdapter) -> None:
        total = fixture.total_bytes()
        if total > self.max_fixture_bytes:
            raise DifferentialHarnessError(
                f"fixture is {total} bytes; permit max_fixture_bytes is {self.max_fixture_bytes}",
                reason_codes=("over_budget_fixture",),
            )


def default_execution_permit() -> ExecutionPermit:
    return ExecutionPermit()


def _await_result(
    value: Any | Awaitable[Any], *, timeout_seconds: float | None
) -> Any:
    if not inspect.isawaitable(value):
        return value

    def run_coro(coro: Awaitable[Any]) -> Any:
        return asyncio.run(coro)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if timeout_seconds is None:
            return run_coro(value)
        result_box: list[Any] = []
        failure_box: list[BaseException] = []

        def run() -> None:
            try:
                result_box.append(run_coro(value))
            except BaseException as exc:  # preserved for exact error identity
                failure_box.append(exc)

        thread = threading.Thread(
            target=run, name="differential-await", daemon=True
        )
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(
                f"async surface exceeded timeout of {timeout_seconds}s"
            )
        if failure_box:
            raise failure_box[0]
        return result_box[0]

    result: list[Any] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as exc:  # preserved for exact error identity
            failure.append(exc)

    thread = threading.Thread(target=run, name="differential-await", daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(
            f"async surface exceeded timeout of {timeout_seconds}s"
        )
    if failure:
        raise failure[0]
    return result[0]


def _execute_with_timeout(
    adapter: SurfaceAdapter,
    step: TraceStep,
    context: SurfaceRunContext,
    *,
    timeout_seconds: float | None,
) -> Any:
    """Run adapter.execute with optional wall-clock timeout for sync callables."""

    if timeout_seconds is None:
        return _await_result(adapter.execute(step, context), timeout_seconds=None)

    result_box: list[Any] = []
    failure_box: list[BaseException] = []

    def run() -> None:
        try:
            raw = adapter.execute(step, context)
            result_box.append(_await_result(raw, timeout_seconds=timeout_seconds))
        except BaseException as exc:
            failure_box.append(exc)

    thread = threading.Thread(
        target=run, name="differential-step-timeout", daemon=True
    )
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(
            f"surface {adapter.surface_id!r} exceeded timeout of {timeout_seconds}s "
            f"on vector {step.vector_id!r}"
        )
    if failure_box:
        raise failure_box[0]
    return result_box[0]


_NETWORK_GUARD_LOCK = threading.RLock()


@contextmanager
def _deny_network() -> Iterable[None]:
    """Deny common in-process socket paths while a surface case executes."""

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise HermeticNetworkError(
            "network access is disabled in differential contract fixtures"
        )

    with _NETWORK_GUARD_LOCK:
        with (
            mock.patch.object(socket.socket, "connect", denied),
            mock.patch.object(socket.socket, "connect_ex", denied),
            mock.patch.object(socket.socket, "sendto", denied),
            mock.patch.object(socket, "create_connection", denied),
        ):
            yield


# ---------------------------------------------------------------------------
# Observations and witnesses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupReceipt:
    root: str
    attempted: bool
    succeeded: bool
    before_cleanup_cid: str
    residue_cid: str | None
    error: str | None
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "before_cleanup_cid": self.before_cleanup_cid,
            "residue_cid": self.residue_cid,
            "error": self.error,
            "cid": self.content_id,
        }


@dataclass(frozen=True)
class SurfaceObservation:
    surface_id: str
    vector_id: str
    operation: str
    status: ObservationStatus
    authoritative: bool
    request_cid: str
    fixture_spec_cid: str
    fixture_before_cid: str
    fixture_after_cid: str
    raw_result: JsonValue | None
    raw_result_cid: str | None
    normalized_result: JsonValue | None
    normalized_result_cid: str | None
    canonical_projection: JsonValue | None
    error: ErrorIdentity | None
    normalization_rules: tuple[str, ...]
    contract_match: bool
    mismatch_paths: tuple[str, ...]
    silent_success: bool
    cleanup: CleanupReceipt
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "vector_id": self.vector_id,
            "operation": self.operation,
            "status": self.status.value,
            "authoritative": self.authoritative,
            "request_cid": self.request_cid,
            "fixture_spec_cid": self.fixture_spec_cid,
            "fixture_before_cid": self.fixture_before_cid,
            "fixture_after_cid": self.fixture_after_cid,
            "raw_result": self.raw_result,
            "raw_result_cid": self.raw_result_cid,
            "normalized_result": self.normalized_result,
            "normalized_result_cid": self.normalized_result_cid,
            "canonical_projection": self.canonical_projection,
            "error": None if self.error is None else self.error.to_record(),
            "normalization_rules": list(self.normalization_rules),
            "contract_match": self.contract_match,
            "mismatch_paths": list(self.mismatch_paths),
            "silent_success": self.silent_success,
            "cleanup": self.cleanup.to_record(),
            "cid": self.content_id,
        }


@dataclass(frozen=True)
class SurfaceRun:
    surface_id: str
    family: str
    availability: SurfaceAvailability
    authoritative: bool
    implementation: str
    public_surface: str
    unavailable_reason: str | None
    runtime: RuntimeIdentity
    implementation_identity: ImplementationIdentity
    observations: tuple[SurfaceObservation, ...]
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "family": self.family,
            "availability": self.availability.value,
            "authoritative": self.authoritative,
            "implementation": self.implementation,
            "public_surface": self.public_surface,
            "unavailable_reason": self.unavailable_reason,
            "runtime": self.runtime.to_record(),
            "implementation_identity": self.implementation_identity.to_record(),
            "observations": [
                observation.to_record() for observation in self.observations
            ],
            "cid": self.content_id,
        }


@dataclass(frozen=True)
class DriftFinding:
    vector_id: str
    operation: str
    kinds: tuple[DriftKind, ...]
    surface_ids: tuple[str, ...]
    authoritative: bool
    mismatch_paths: tuple[str, ...]
    expected_cid: str
    observed_cids: tuple[str, ...]
    description: str
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "operation": self.operation,
            "kinds": [kind.value for kind in self.kinds],
            "surface_ids": list(self.surface_ids),
            "authoritative": self.authoritative,
            "mismatch_paths": list(self.mismatch_paths),
            "expected_cid": self.expected_cid,
            "observed_cids": list(self.observed_cids),
            "description": self.description,
            "cid": self.content_id,
        }


@dataclass(frozen=True)
class DifferentialWitness:
    schema: str
    goal_id: str
    trace: CanonicalOperationTrace
    fixture: FixtureAdapter
    surface_runs: tuple[SurfaceRun, ...]
    findings: tuple[DriftFinding, ...]
    authoritative_surface_ids: tuple[str, ...]
    non_authoritative_surface_ids: tuple[str, ...]
    unavailable_surface_ids: tuple[str, ...]
    unknown_surface_ids: tuple[str, ...]
    authoritative_agreement: bool
    all_cleanup_succeeded: bool
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "goal_id": self.goal_id,
            "authority": {
                "completion": WITNESS_IS_COMPLETION_EVIDENCE,
                "correctness": WITNESS_IS_CORRECTNESS_EVIDENCE,
                "repair": WITNESS_AUTHORIZES_REPAIR,
            },
            "trace": {**self.trace.to_record(), "cid": self.trace.content_id},
            "fixture": {
                **dict(self.fixture.to_record()),
                "cid": self.fixture.content_id,
            },
            "surface_runs": [run.to_record() for run in self.surface_runs],
            "findings": [finding.to_record() for finding in self.findings],
            "authoritative_surface_ids": list(self.authoritative_surface_ids),
            "non_authoritative_surface_ids": list(
                self.non_authoritative_surface_ids
            ),
            "unavailable_surface_ids": list(self.unavailable_surface_ids),
            "unknown_surface_ids": list(self.unknown_surface_ids),
            "authoritative_agreement": self.authoritative_agreement,
            "all_cleanup_succeeded": self.all_cleanup_succeeded,
            "cid": self.content_id,
        }


def _cleanup(root: Path, before_cleanup_cid: str) -> CleanupReceipt:
    error: str | None = None
    try:
        shutil.rmtree(root)
    except OSError as exc:
        error = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    succeeded = not root.exists()
    residue_cid = snapshot_tree(root).content_id if root.exists() else None
    record = {
        "root": str(root),
        "attempted": True,
        "succeeded": succeeded,
        "before_cleanup_cid": before_cleanup_cid,
        "residue_cid": residue_cid,
        "error": error,
    }
    return CleanupReceipt(**record, content_id=_content_id(record))


def _observation_for_step(
    adapter: SurfaceAdapter,
    step: TraceStep,
    fixture: FixtureAdapter,
    *,
    temp_parent: Path | None,
    normalizer: ResultNormalizer,
    error_classifier: ErrorClassifier,
    permit: ExecutionPermit,
    wall_deadline: float | None,
) -> SurfaceObservation:
    if wall_deadline is not None and time.monotonic() > wall_deadline:
        raise DifferentialHarnessError(
            "differential run exceeded max_wall_seconds budget",
            reason_codes=("over_budget_wall",),
        )

    safe_surface_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in adapter.surface_id[:24]
    )
    root = Path(
        tempfile.mkdtemp(
            prefix=f"diff-contract-{safe_surface_id}-",
            dir=None if temp_parent is None else str(temp_parent),
        )
    )
    # mkdtemp creates the directory; FixtureSpec owns creation so remove only
    # this empty, resolved directory before materialization.
    root.rmdir()
    fixture_before_cid = fixture.materialize(root)
    expected_fixture_cid = snapshot_tree(root).content_id
    if fixture_before_cid != expected_fixture_cid:  # pragma: no cover - guard
        raise DifferentialHarnessError(
            "fixture snapshot was not reproducible",
            reason_codes=("fixture_not_reproducible",),
        )

    raw_value: Any = None
    raw_record: JsonValue | None = None
    error_identity: ErrorIdentity | None = None
    status = ObservationStatus.SUCCESS
    explicit_success_with_error = False
    context = SurfaceRunContext(
        root=root,
        fixture=fixture,
        step=step,
        network_allowed=permit.network_allowed,
    )
    try:
        network_cm = (
            contextlib_null() if permit.network_allowed else _deny_network()
        )
        with network_cm:
            raw_value = _execute_with_timeout(
                adapter,
                step,
                context,
                timeout_seconds=permit.timeout_seconds,
            )
        raw_record = _json_value(raw_value)
        if isinstance(raw_value, Mapping):
            explicit_success_with_error = (
                (raw_value.get("ok") is True or raw_value.get("success") is True)
                and raw_value.get("error") is not None
            )
            error_identity = _reported_error_identity(raw_value, error_classifier)
            if error_identity is not None:
                status = ObservationStatus.ERROR
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            before_cleanup = snapshot_tree(root).content_id
            _cleanup(root, before_cleanup)
            raise
        if isinstance(exc, TimeoutError):
            error_identity = _exception_identity(exc, error_classifier)
            status = ObservationStatus.ERROR
            raw_record = {
                "error": {
                    "code": error_identity.code,
                    "exception_module": error_identity.exception_module,
                    "exception_type": error_identity.exception_type,
                    "message": error_identity.message,
                    "errno": error_identity.errno,
                }
            }
        else:
            error_identity = _exception_identity(exc, error_classifier)
            status = ObservationStatus.ERROR
            raw_record = {
                "error": {
                    "code": error_identity.code,
                    "exception_module": error_identity.exception_module,
                    "exception_type": error_identity.exception_type,
                    "message": error_identity.message,
                    "errno": error_identity.errno,
                }
            }

    fixture_after = snapshot_tree(root)
    rules = normalizer.rules_for(step)
    normalized_expected = normalizer.normalize(step, step.expected, rules=rules)
    normalized_result = normalizer.normalize(step, raw_record, rules=rules)
    normalized_result = _add_derived_effects(
        normalized_expected,
        normalized_result,
        unchanged=fixture_after.content_id == fixture_before_cid,
    )
    projection = _project_like(normalized_expected, normalized_result)
    mismatch_paths = _mismatch_paths(normalized_expected, projection)
    expected_error = _error_container(normalized_expected)
    actual_error = _error_container(normalized_result)
    silent_success = explicit_success_with_error or (
        expected_error is not None and actual_error is None
    )
    before_cleanup_cid = fixture_after.content_id
    cleanup = _cleanup(root, before_cleanup_cid)

    record_without_cid = {
        "surface_id": adapter.surface_id,
        "vector_id": step.vector_id,
        "operation": step.operation,
        "status": status.value,
        "authoritative": adapter.availability is SurfaceAvailability.REAL,
        "request_cid": _content_id(step.request),
        "fixture_spec_cid": fixture.content_id,
        "fixture_before_cid": fixture_before_cid,
        "fixture_after_cid": fixture_after.content_id,
        "raw_result": raw_record,
        "raw_result_cid": None if raw_record is None else _content_id(raw_record),
        "normalized_result": normalized_result,
        "normalized_result_cid": _content_id(normalized_result),
        "canonical_projection": projection,
        "error": None if error_identity is None else error_identity.to_record(),
        "normalization_rules": [rule.value for rule in rules],
        "contract_match": not mismatch_paths and cleanup.succeeded and not silent_success,
        "mismatch_paths": list(mismatch_paths),
        "silent_success": silent_success,
        "cleanup": cleanup.to_record(),
    }
    # Silent success is never a contract match.
    if silent_success and "$" not in mismatch_paths:
        # Ensure silent success is visible even when projection matched fields.
        extra_paths = list(mismatch_paths)
        if not extra_paths:
            extra_paths = ["$.silent_success"]
        record_without_cid["mismatch_paths"] = extra_paths
        record_without_cid["contract_match"] = False
        mismatch_paths = tuple(extra_paths)

    return SurfaceObservation(
        surface_id=adapter.surface_id,
        vector_id=step.vector_id,
        operation=step.operation,
        status=status,
        authoritative=adapter.availability is SurfaceAvailability.REAL,
        request_cid=record_without_cid["request_cid"],
        fixture_spec_cid=fixture.content_id,
        fixture_before_cid=fixture_before_cid,
        fixture_after_cid=fixture_after.content_id,
        raw_result=raw_record,
        raw_result_cid=record_without_cid["raw_result_cid"],
        normalized_result=normalized_result,
        normalized_result_cid=record_without_cid["normalized_result_cid"],
        canonical_projection=projection,
        error=error_identity,
        normalization_rules=tuple(rule.value for rule in rules),
        contract_match=bool(record_without_cid["contract_match"]),
        mismatch_paths=mismatch_paths,
        silent_success=silent_success,
        cleanup=cleanup,
        content_id=_content_id(record_without_cid),
    )


@contextmanager
def contextlib_null() -> Iterable[None]:
    yield


def _surface_run(
    adapter: SurfaceAdapter,
    trace: CanonicalOperationTrace,
    fixture: FixtureAdapter,
    *,
    temp_parent: Path | None,
    normalizer: ResultNormalizer,
    error_classifier: ErrorClassifier,
    permit: ExecutionPermit,
    wall_deadline: float | None,
) -> SurfaceRun:
    runtime = capture_runtime_identity(adapter.package_names)
    implementation = _implementation_identity(adapter)
    observations: tuple[SurfaceObservation, ...]
    if adapter.availability in {
        SurfaceAvailability.UNAVAILABLE,
        SurfaceAvailability.UNKNOWN,
    } and getattr(adapter, "executor", None) is None:
        observations = ()
    else:
        observations = tuple(
            _observation_for_step(
                adapter,
                step,
                fixture,
                temp_parent=temp_parent,
                normalizer=normalizer,
                error_classifier=error_classifier,
                permit=permit,
                wall_deadline=wall_deadline,
            )
            for step in trace.steps
        )
        # Adapter identity drift: re-capture after execution; identity must be stable.
        after_identity = _implementation_identity(adapter)
        if after_identity.content_id != implementation.content_id:
            raise DifferentialHarnessError(
                f"adapter identity drifted for surface {adapter.surface_id!r}",
                reason_codes=("adapter_identity_drift",),
            )

    record = {
        "surface_id": adapter.surface_id,
        "family": adapter.family,
        "availability": adapter.availability.value,
        "authoritative": adapter.availability is SurfaceAvailability.REAL,
        "implementation": adapter.implementation,
        "public_surface": adapter.public_surface,
        "unavailable_reason": adapter.unavailable_reason,
        "runtime": runtime.to_record(),
        "implementation_identity": implementation.to_record(),
        "observations": [observation.to_record() for observation in observations],
    }
    return SurfaceRun(
        surface_id=adapter.surface_id,
        family=adapter.family,
        availability=adapter.availability,
        authoritative=adapter.availability is SurfaceAvailability.REAL,
        implementation=adapter.implementation,
        public_surface=adapter.public_surface,
        unavailable_reason=adapter.unavailable_reason,
        runtime=runtime,
        implementation_identity=implementation,
        observations=observations,
        content_id=_content_id(record),
    )


def _build_findings(
    trace: CanonicalOperationTrace,
    runs: Sequence[SurfaceRun],
    *,
    normalizer: ResultNormalizer,
    drift_classifier: DriftClassifier,
) -> tuple[DriftFinding, ...]:
    by_step = {step.vector_id: step for step in trace.steps}
    findings: list[DriftFinding] = []
    for run in runs:
        for observation in run.observations:
            if observation.contract_match:
                continue
            step = by_step[observation.vector_id]
            expected = normalizer.normalize(step, step.expected)
            observed_cid = observation.normalized_result_cid or _content_id(None)
            kinds = drift_classifier.classify(step, observation)
            record = {
                "vector_id": step.vector_id,
                "operation": step.operation,
                "kinds": [kind.value for kind in kinds],
                "surface_ids": [run.surface_id],
                "authoritative": run.authoritative,
                "mismatch_paths": list(observation.mismatch_paths),
                "expected_cid": _content_id(expected),
                "observed_cids": [observed_cid],
                "description": "surface result differs from the canonical contract",
            }
            findings.append(
                DriftFinding(
                    vector_id=step.vector_id,
                    operation=step.operation,
                    kinds=kinds,
                    surface_ids=(run.surface_id,),
                    authoritative=run.authoritative,
                    mismatch_paths=observation.mismatch_paths,
                    expected_cid=record["expected_cid"],
                    observed_cids=(observed_cid,),
                    description=record["description"],
                    content_id=_content_id(record),
                )
            )

    # Pairwise comparisons use canonical projections.  Compatible surfaces
    # that only add transport metadata therefore do not become false drift.
    authoritative_runs = [run for run in runs if run.authoritative]
    for left_index, left in enumerate(authoritative_runs):
        left_by_id = {item.vector_id: item for item in left.observations}
        for right in authoritative_runs[left_index + 1 :]:
            right_by_id = {item.vector_id: item for item in right.observations}
            for step in trace.steps:
                left_observation = left_by_id[step.vector_id]
                right_observation = right_by_id[step.vector_id]
                if (
                    left_observation.canonical_projection
                    == right_observation.canonical_projection
                ):
                    continue
                # Contract findings already identify a single bad surface.
                # A pairwise record is valuable only when both projections
                # differ, avoiding duplicate noise for the ordinary case.
                if left_observation.contract_match or right_observation.contract_match:
                    continue
                paths = _mismatch_paths(
                    left_observation.canonical_projection,
                    right_observation.canonical_projection,
                )
                expected = normalizer.normalize(step, step.expected)
                observed_cids = (
                    left_observation.normalized_result_cid or _content_id(None),
                    right_observation.normalized_result_cid or _content_id(None),
                )
                kinds = tuple(
                    sorted(
                        set(drift_classifier.classify(step, left_observation))
                        | set(drift_classifier.classify(step, right_observation)),
                        key=lambda kind: kind.value,
                    )
                )
                record = {
                    "vector_id": step.vector_id,
                    "operation": step.operation,
                    "kinds": [kind.value for kind in kinds],
                    "surface_ids": [left.surface_id, right.surface_id],
                    "authoritative": True,
                    "mismatch_paths": list(paths),
                    "expected_cid": _content_id(expected),
                    "observed_cids": list(observed_cids),
                    "description": "authoritative surfaces disagree with each other",
                }
                findings.append(
                    DriftFinding(
                        vector_id=step.vector_id,
                        operation=step.operation,
                        kinds=kinds,
                        surface_ids=(left.surface_id, right.surface_id),
                        authoritative=True,
                        mismatch_paths=paths,
                        expected_cid=record["expected_cid"],
                        observed_cids=observed_cids,
                        description=record["description"],
                        content_id=_content_id(record),
                    )
                )
    return tuple(
        sorted(
            findings,
            key=lambda item: (
                item.vector_id,
                item.surface_ids,
                item.description,
                item.content_id,
            ),
        )
    )


def run_differential_contract_harness(
    surfaces: Iterable[SurfaceAdapter],
    *,
    trace: CanonicalOperationTrace | None = None,
    trace_provider: ContractTraceProvider | None = None,
    fixture: FixtureAdapter | None = None,
    normalizer: ResultNormalizer | None = None,
    error_classifier: ErrorClassifier | None = None,
    drift_classifier: DriftClassifier | None = None,
    permit: ExecutionPermit | None = None,
    schema: str = DIFFERENTIAL_WITNESS_SCHEMA,
    goal_id: str = "",
    temp_parent: str | os.PathLike[str] | None = None,
) -> DifferentialWitness:
    """Execute selected surfaces and return a self-identifying witness.

    Real surfaces are authoritative for drift.  Declared mocks execute and are
    retained as non-authoritative observations.  Unavailable and unknown
    surfaces without executors never execute and remain explicit in the witness.

    Domain fixtures, normalizers, error maps, and drift taxonomies are injected.
    The harness never embeds product-domain tree fixtures or invariant maps.
    """

    adapters = tuple(surfaces)
    if not adapters:
        raise DifferentialHarnessError(
            "at least one surface must be selected",
            reason_codes=("surfaces_required",),
        )
    for adapter in adapters:
        if not isinstance(adapter, SurfaceAdapter):
            raise DifferentialHarnessError(
                f"surface {adapter!r} does not implement SurfaceAdapter",
                reason_codes=("surface_protocol",),
            )
    ids = [adapter.surface_id for adapter in adapters]
    if len(ids) != len(set(ids)):
        raise DifferentialHarnessError(
            "surface_ids must be unique",
            reason_codes=("surface_id_duplicate",),
        )

    if trace is None:
        if trace_provider is None:
            raise DifferentialHarnessError(
                "trace or trace_provider is required",
                reason_codes=("trace_required",),
            )
        selected_trace = trace_provider.build_trace()
    else:
        selected_trace = trace

    if fixture is None:
        raise DifferentialHarnessError(
            "fixture is required; domain profiles must supply a FixtureAdapter",
            reason_codes=("fixture_required",),
        )
    selected_fixture = fixture
    selected_normalizer = normalizer or default_result_normalizer()
    selected_errors = error_classifier or default_error_classifier()
    selected_drift = drift_classifier or default_drift_classifier()
    selected_permit = permit or default_execution_permit()

    selected_permit.enforce_trace(selected_trace)
    selected_permit.enforce_fixture(selected_fixture)

    parent = None if temp_parent is None else Path(temp_parent).resolve()
    if parent is not None:
        if not parent.is_dir():
            raise DifferentialHarnessError(
                f"temp_parent is not an existing directory: {parent}",
                reason_codes=("temp_parent_missing",),
            )

    wall_deadline = None
    if selected_permit.max_wall_seconds is not None:
        wall_deadline = time.monotonic() + selected_permit.max_wall_seconds

    runs = tuple(
        _surface_run(
            adapter,
            selected_trace,
            selected_fixture,
            temp_parent=parent,
            normalizer=selected_normalizer,
            error_classifier=selected_errors,
            permit=selected_permit,
            wall_deadline=wall_deadline,
        )
        for adapter in adapters
    )
    findings = _build_findings(
        selected_trace,
        runs,
        normalizer=selected_normalizer,
        drift_classifier=selected_drift,
    )
    authoritative_ids = tuple(
        run.surface_id for run in runs if run.availability is SurfaceAvailability.REAL
    )
    non_authoritative_ids = tuple(
        run.surface_id for run in runs if run.availability is SurfaceAvailability.MOCK
    )
    unavailable_ids = tuple(
        run.surface_id
        for run in runs
        if run.availability is SurfaceAvailability.UNAVAILABLE
    )
    unknown_ids = tuple(
        run.surface_id
        for run in runs
        if run.availability is SurfaceAvailability.UNKNOWN
    )
    all_cleanup = all(
        observation.cleanup.succeeded
        for run in runs
        for observation in run.observations
    )
    # Incomplete cleanup is never silent success of the run.
    if not all_cleanup and any(
        run.observations for run in runs
    ):
        # findings already include CLEANUP kinds when classifier is default;
        # authoritative agreement requires full cleanup.
        pass
    agreement = all(not finding.authoritative for finding in findings) and all_cleanup
    record = {
        "schema": schema,
        "goal_id": goal_id,
        "authority": {
            "completion": WITNESS_IS_COMPLETION_EVIDENCE,
            "correctness": WITNESS_IS_CORRECTNESS_EVIDENCE,
            "repair": WITNESS_AUTHORIZES_REPAIR,
        },
        "trace": {**selected_trace.to_record(), "cid": selected_trace.content_id},
        "fixture": {
            **dict(selected_fixture.to_record()),
            "cid": selected_fixture.content_id,
        },
        "surface_runs": [run.to_record() for run in runs],
        "findings": [finding.to_record() for finding in findings],
        "authoritative_surface_ids": list(authoritative_ids),
        "non_authoritative_surface_ids": list(non_authoritative_ids),
        "unavailable_surface_ids": list(unavailable_ids),
        "unknown_surface_ids": list(unknown_ids),
        "authoritative_agreement": agreement,
        "all_cleanup_succeeded": all_cleanup,
    }
    return DifferentialWitness(
        schema=schema,
        goal_id=goal_id,
        trace=selected_trace,
        fixture=selected_fixture,
        surface_runs=runs,
        findings=findings,
        authoritative_surface_ids=authoritative_ids,
        non_authoritative_surface_ids=non_authoritative_ids,
        unavailable_surface_ids=unavailable_ids,
        unknown_surface_ids=unknown_ids,
        authoritative_agreement=agreement,
        all_cleanup_succeeded=all_cleanup,
        content_id=_content_id(record),
    )


def write_differential_witness(
    witness: DifferentialWitness, destination: str | os.PathLike[str]
) -> Path:
    """Atomically persist an exact witness JSON record."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                witness.to_record(),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


__all__ = [
    "DEFAULT_ERROR_CODE_IO_FAILURE",
    "DEFAULT_ERROR_CODES",
    "DEFAULT_NORMALIZATION_RULES",
    "DEFAULT_STEP_TIMEOUT_SECONDS",
    "DIFFERENTIAL_TRACE_SCHEMA",
    "DIFFERENTIAL_WITNESS_SCHEMA",
    "MAX_FIXTURE_BYTES",
    "MAX_TRACE_STEPS",
    "WITNESS_AUTHORIZES_REPAIR",
    "WITNESS_IS_COMPLETION_EVIDENCE",
    "WITNESS_IS_CORRECTNESS_EVIDENCE",
    "CallableSurfaceAdapter",
    "CanonicalOperationTrace",
    "CleanupReceipt",
    "ContractResultNormalizer",
    "ContractTraceProvider",
    "DifferentialHarnessError",
    "DifferentialWitness",
    "DriftClassifier",
    "DriftFinding",
    "DriftKind",
    "ErrorClassifier",
    "ErrorIdentity",
    "ExecutionPermit",
    "FixtureAdapter",
    "FixtureEntry",
    "FixtureSpec",
    "HermeticNetworkError",
    "ImplementationIdentity",
    "InvariantDriftClassifier",
    "MappingErrorClassifier",
    "NormalizationRule",
    "ObservationStatus",
    "ProfileTraceProvider",
    "ResultNormalizer",
    "RuntimeIdentity",
    "SurfaceAdapter",
    "SurfaceAvailability",
    "SurfaceObservation",
    "SurfaceRun",
    "SurfaceRunContext",
    "TraceStep",
    "TreeSnapshot",
    "VectorTraceProvider",
    "build_canonical_operation_trace",
    "build_fixture_spec",
    "capture_runtime_identity",
    "default_drift_classifier",
    "default_error_classifier",
    "default_execution_permit",
    "default_result_normalizer",
    "normalize_contract_result",
    "run_differential_contract_harness",
    "snapshot_tree",
    "write_differential_witness",
]
